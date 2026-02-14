"""
Fast TextWorld environment — drop-in replacement for TextWorldEnv.

Uses FastTextWorldSimulator instead of the textworld library's Inform7
subprocess engine. No subprocess, no IPC, no global locks.

Same interface as TextWorldEnv: compatible with the SkyRL RL training pipeline.
Same reward shaping (step_penalty, efficiency_bonus).
Same modality payload structure for memory documents.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import re
import string
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, ConversationType
from skyrl_gym.envs.textworld.fast_sim import FastTextWorldSimulator


# ---------------------------------------------------------------------------
# Incremental memory — identical to the one in env.py to preserve behavior
# ---------------------------------------------------------------------------

@dataclass
class _MemoryTurn:
    turn: int
    observation: str
    thinking: str
    action: str
    reward: float
    score: int


@dataclass
class _IncrementalMemory:
    memory_window: int = 5
    include_thinking: bool = False
    max_documents: int = 4
    current_segment: List[_MemoryTurn] = field(default_factory=list)
    memory_documents: List[str] = field(default_factory=list)

    def add_turn(
        self,
        turn: int,
        observation: str,
        thinking: str,
        action: str,
        reward: float,
        score: int,
    ) -> None:
        self.current_segment.append(
            _MemoryTurn(
                turn=turn,
                observation=observation[:200],
                thinking=thinking,
                action=action,
                reward=reward,
                score=score,
            )
        )
        if len(self.current_segment) >= self.memory_window:
            self._finalize_segment()

    def finalize(self) -> None:
        if self.current_segment:
            self._finalize_segment()

    def _finalize_segment(self) -> None:
        document = self._create_document_text()
        if document:
            self.memory_documents.append(document)
            if len(self.memory_documents) > self.max_documents:
                self.memory_documents = self.memory_documents[-self.max_documents:]
        self.current_segment = []

    def _create_document_text(self) -> str:
        if not self.current_segment:
            return ""

        start_turn = self.current_segment[0].turn
        end_turn = self.current_segment[-1].turn
        total_reward = sum(t.reward for t in self.current_segment)
        score_delta = self.current_segment[-1].score - self.current_segment[0].score

        lines = [
            f"Turns {start_turn}-{end_turn} | Reward: {total_reward} | Score Change: {score_delta}",
            "",
        ]
        for t in self.current_segment:
            lines.append(f"Turn {t.turn}:")
            if t.observation:
                lines.append(f"Obs: {t.observation}")
            if self.include_thinking and t.thinking:
                lines.append(f"Think: {t.thinking}")
            action_line = f"Act: {t.action}"
            if t.reward != 0:
                action_line += f" -> Reward: {t.reward}"
            lines.append(action_line)
            lines.append("")
        return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Action parsing regex — compiled once
# ---------------------------------------------------------------------------
_ACTION_RE = re.compile(r'\[ACTION:\s*([^\]]+)\]', re.IGNORECASE)
_ALLOWED_CHARS = set(string.ascii_letters + string.digits + string.whitespace + "-_.,!?'")


class FastTextWorldEnv(BaseTextEnv):
    """
    Drop-in replacement for TextWorldEnv that uses FastTextWorldSimulator.

    Uses the .json game specification directly instead of spawning an Inform7
    subprocess. Orders-of-magnitude faster startup and step execution.

    Accepts all the same config knobs as TextWorldEnv:
    - game_file: path to .z8 or .json game file
    - max_turns, memory_window, max_memory_docs, max_doc_tokens
    - step_penalty, efficiency_bonus
    - modality payload injection (placeholder_token, modality_id, etc.)
    """

    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] = {}):
        super().__init__()

        self.env_config = env_config
        self.extras = extras

        game_file = extras.get("game_file") or extras.get("extra_info", {}).get("game_file")
        if not game_file:
            raise ValueError("FastTextWorldEnv requires `game_file` in extras.")
        self.game_file = str(game_file)

        # If a .z8 file is given, look for the corresponding .json
        if self.game_file.endswith(".z8"):
            self.game_json = self.game_file[:-3] + ".json"
        elif self.game_file.endswith(".json"):
            self.game_json = self.game_file
        else:
            # Try appending .json
            self.game_json = self.game_file + ".json"

        self.max_turns = int(extras.get("max_turns") or env_config.get("max_turns", 50))
        self.memory_window = int(extras.get("memory_window") or env_config.get("memory_window", 5))
        self.max_memory_docs = int(extras.get("max_memory_docs") or env_config.get("max_memory_docs", 4))
        self.max_doc_tokens = int(extras.get("max_doc_tokens") or env_config.get("max_doc_tokens", 256))
        self.tokenizer_path = str(extras.get("tokenizer_path") or env_config.get("tokenizer_path", ""))
        self.step_penalty = float(extras.get("step_penalty") or env_config.get("step_penalty", 0.0))
        self.efficiency_bonus = float(extras.get("efficiency_bonus") or env_config.get("efficiency_bonus", 0.0))

        self.modality_id = str(extras.get("memory_modality_id") or env_config.get("memory_modality_id", "memo_memory"))
        self.placeholder_token = str(extras.get("placeholder_token") or env_config.get("placeholder_token", "<|image_pad|>"))
        self.max_placeholder_tokens = int(extras.get("max_placeholder_tokens") or env_config.get("max_placeholder_tokens", 8))

        self._sim: Optional[FastTextWorldSimulator] = None
        self._last_observation: str = ""
        self._won: bool = False
        self._score: int = 0

        self._memory = _IncrementalMemory(
            memory_window=self.memory_window,
            include_thinking=False,
            max_documents=self.max_memory_docs,
        )

    def init(self, prompt: ConversationType):
        self._sim = FastTextWorldSimulator(self.game_json)
        obs, info = self._sim.reset()
        self.turns = 0
        self._won = False
        self._score = 0
        self._memory = _IncrementalMemory(
            memory_window=self.memory_window,
            include_thinking=False,
            max_documents=self.max_memory_docs,
        )
        self._last_observation = obs.strip()

        # Keep a reference to the system message dict so we can dynamically
        # inject placeholder tokens when memory documents are created.
        self._system_message = prompt[0] if prompt else None
        self._base_system_content = prompt[0]["content"] if prompt else ""

        self._update_modalities_payload()

        first_obs = {"role": "user", "content": self._last_observation}
        return prompt + [first_obs], {}

    def _parse_action(self, action: str) -> str:
        """
        Parse action from model output. Model generates reasoning + [ACTION: command].
        Identical to TextWorldEnv._parse_action for compatibility.
        """
        if not action:
            return ""

        # Clean UTF-8 encoding issues first
        try:
            action = action.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
        except Exception:
            action = ""

        if not action:
            return ""

        # First try to extract [ACTION: ...] format
        action_match = _ACTION_RE.search(action)
        if action_match:
            parsed = action_match.group(1).strip()
        else:
            # Fallback: try old parsing logic
            cleaned = action.replace("<think>", "").replace("</think>", "").strip()
            if "</think>" in action:
                _, after = action.split("</think>", 1)
                cleaned = after.strip() or cleaned
            for prefix in ("Action:", "I will", "I'll", "Let me", "I would", "I should"):
                if cleaned.lower().startswith(prefix.lower()):
                    cleaned = cleaned[len(prefix):].strip()
            if "\n" in cleaned:
                cleaned = cleaned.split("\n", 1)[0].strip()
            parsed = cleaned.strip('"\'.')

        # Final sanitization: remove any non-ASCII or control characters
        parsed = ''.join(c for c in parsed if c in _ALLOWED_CHARS)
        parsed = parsed.strip()[:200]

        return parsed

    def _update_modalities_payload(self) -> None:
        """Update the modality payloads with current memory documents."""
        modalities_entry = self.extras.get("modalities")
        if modalities_entry is None or not hasattr(modalities_entry, "payloads"):
            return

        documents = list(self._memory.memory_documents)

        from skyrl_train.dataset.modalities import ModalityPlaceholderPlan

        if documents:
            docs_with_content = [doc for doc in documents if doc]

            if self._system_message is not None and docs_with_content:
                placeholder_block = " ".join(
                    [self.placeholder_token] * self.max_placeholder_tokens
                )
                self._system_message["content"] = (
                    f"{self._base_system_content}\n{placeholder_block}"
                )

            modalities_entry.payloads[self.modality_id] = [docs_with_content] if docs_with_content else []

            plan = ModalityPlaceholderPlan(
                modality_id=self.modality_id,
                placeholder_token=self.placeholder_token,
                occurrences=1 if docs_with_content else 0,
                reserved_tokens=[self.max_placeholder_tokens] if docs_with_content else [],
                payload=[docs_with_content] if docs_with_content else [],
            )
            modalities_entry.plans[self.modality_id] = plan
        else:
            if self._system_message is not None:
                self._system_message["content"] = self._base_system_content
            modalities_entry.payloads.pop(self.modality_id, None)
            modalities_entry.plans.pop(self.modality_id, None)

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1
        parsed_action = self._parse_action(action) or "look"

        obs, reward, done, info = self._sim.step(parsed_action)
        score = info.get("score", 0)
        won = info.get("won", False)

        # Step-efficiency reward shaping (identical to TextWorldEnv)
        shaped_reward = float(reward)
        if self.step_penalty > 0:
            shaped_reward -= self.step_penalty
        if done and won and self.efficiency_bonus > 0:
            efficiency = max(0.0, 1.0 - (self.turns / self.max_turns))
            shaped_reward += self.efficiency_bonus * efficiency
        reward = shaped_reward

        self._memory.add_turn(
            turn=self.turns,
            observation=self._last_observation,
            thinking="",
            action=parsed_action,
            reward=float(reward),
            score=int(score),
        )
        if done or self.turns >= self.max_turns:
            self._memory.finalize()

        self._update_modalities_payload()

        self._last_observation = obs.strip()
        self._won = won
        self._score = score
        done = done or self.turns >= self.max_turns

        observations = [] if done else [{"role": "user", "content": self._last_observation}]

        metadata = {
            "action": parsed_action,
            "score": int(score),
            "won": bool(won),
        }

        # Include modalities if available
        modalities_entry = self.extras.get("modalities")
        if modalities_entry is not None:
            metadata["modalities"] = modalities_entry

        return BaseTextEnvStepOutput(
            observations=observations,
            reward=float(reward),
            done=done,
            metadata=metadata,
        )

    def close(self):
        self._sim = None

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "steps": self.turns,
            "won": self._won,
            "score": self._score,
        }


@lru_cache(maxsize=4)
def _get_tokenizer(model_path: str):
    if not model_path:
        raise ValueError("FastTextWorldEnv requires `tokenizer_path` to encode memory documents.")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

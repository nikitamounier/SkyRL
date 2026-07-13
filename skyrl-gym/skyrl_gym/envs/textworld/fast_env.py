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
    raw_response: str = ""  # full raw model output for this turn — used to replay chat history of the current segment


@dataclass
class _IncrementalMemory:
    memory_window: int = 5
    include_thinking: bool = False
    max_documents: int = 4
    max_obs_chars: int = 0            # per-turn observation truncation; <=0 means no truncation
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
        raw_response: str = "",
    ) -> None:
        obs_for_memory = observation if self.max_obs_chars <= 0 else observation[: self.max_obs_chars]
        self.current_segment.append(
            _MemoryTurn(
                turn=turn,
                observation=obs_for_memory,
                thinking=thinking,
                action=action,
                reward=reward,
                score=score,
                raw_response=raw_response,
            )
        )
        if len(self.current_segment) >= self.memory_window:
            self._finalize_segment()

    def current_segment_chat_messages(self) -> List[Dict[str, str]]:
        """Replay the unfinalized turns as user/assistant chat messages.

        Each turn produces a (user=obs, assistant=raw_response) pair. The CURRENT turn's
        observation (the one the LLM is about to act on) is NOT included here — the caller
        is responsible for appending that as the final user message.

        Used by Phase-2 rollout / eval to fill the chat history with exactly the turns
        that the memory tokens have NOT yet absorbed (no overlap, no missing turns).
        """
        msgs: List[Dict[str, str]] = []
        for t in self.current_segment:
            if t.observation:
                msgs.append({"role": "user", "content": t.observation})
            msgs.append({"role": "assistant", "content": t.raw_response or t.action})
        return msgs

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

# Inform7 (TextWorld) command vocabulary appendix. Without this, the frozen LLM
# emits natural-English variants the parser rejects ("look around", "pick up X",
# "inspect", "examine <room>") and gets stuck in error loops with no signal.
_TEXTWORLD_VOCAB_HINT = """

Valid TextWorld commands (use exactly these forms — variations are rejected by the parser):
- `look` (describe room — NOT `look around` or `look at <room>`)
- `inventory` (list held items)
- `examine <object>` (NOT `examine <room>` or `inspect`)
- `take <object>` (NOT `pick up`, `grab`, `get`)
- `drop <object>`
- `open <object>` / `close <object>` (for doors and containers)
- `unlock <object> with <key>`
- `put <object> in <container>` / `put <object> on <surface>`
- `go <direction>` or just `<direction>` (north/south/east/west/up/down) — NOT `move`, `walk`, `head`
If a command is rejected (`You can't see any such thing.` or similar), DO NOT repeat it — try a different valid form."""


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

        # Memory-doc richness knobs (Phase 1 of cold-start RL fix).
        # `or` is unsafe for bools (False or X → X), so resolve explicitly.
        _inc = extras.get("include_thinking_in_memory")
        if _inc is None:
            _inc = env_config.get("include_thinking_in_memory", False)
        self.include_thinking_in_memory = bool(_inc)
        # `0` (or negative) = no truncation; full observation goes into memory docs.
        # Likewise `or` resolves 0 to fallback, so use explicit `is None`.
        _max_obs = extras.get("max_obs_chars_in_memory")
        if _max_obs is None:
            _max_obs = env_config.get("max_obs_chars_in_memory", 0)
        self.max_obs_chars_in_memory = int(_max_obs)

        # Phase-2 cold-start: inject placeholder block + a synthetic empty
        # payload from turn 0, so the LLM has memory-position tokens to attend
        # to while _IncrementalMemory is still filling its first window.
        # Without this, turns 1..memory_window-1 under MEMORY_ONLY_CONTEXT see
        # no chat history AND no memory tokens.
        _ipm = extras.get("inject_placeholder_pre_memory")
        if _ipm is None:
            _ipm = env_config.get("inject_placeholder_pre_memory", False)
        self.inject_placeholder_pre_memory = bool(_ipm)

        # How many of the most recent finalized memory docs to inline as
        # plaintext "[Memory recap]" in the system prompt. The encoder still
        # sees the full doc list via the payload — recap is for the LLM to
        # read directly. Default 5 gives ~5 turns of (obs, action, score)
        # so the LLM can avoid action loops under MEMORY_ONLY_CONTEXT.
        _rdc = extras.get("recap_doc_count")
        if _rdc is None:
            _rdc = env_config.get("recap_doc_count", 5)
        self.recap_doc_count = max(1, int(_rdc))
        # Per-doc character cap for inline recap (each doc truncated to this).
        # Total recap size ≈ recap_doc_count × this value.
        _rdcc = extras.get("recap_doc_char_cap")
        if _rdcc is None:
            _rdcc = env_config.get("recap_doc_char_cap", 600)
        self.recap_doc_char_cap = max(50, int(_rdcc))

        self._sim: Optional[FastTextWorldSimulator] = None
        self._last_observation: str = ""
        self._won: bool = False
        self._score: int = 0

        self._memory = _IncrementalMemory(
            memory_window=self.memory_window,
            include_thinking=self.include_thinking_in_memory,
            max_documents=self.max_memory_docs,
            max_obs_chars=self.max_obs_chars_in_memory,
        )

    def init(self, prompt: ConversationType):
        self._sim = FastTextWorldSimulator(self.game_json)
        obs, info = self._sim.reset()
        self.turns = 0
        self._won = False
        self._score = 0
        self._memory = _IncrementalMemory(
            memory_window=self.memory_window,
            include_thinking=self.include_thinking_in_memory,
            max_documents=self.max_memory_docs,
            max_obs_chars=self.max_obs_chars_in_memory,
        )
        self._last_observation = obs.strip()

        # Bootstrap memory doc so the encoder always has meaningful text to
        # embed — never an empty/garbage placeholder. Used by
        # _update_modalities_payload when no real per-turn docs exist yet
        # (relevant to MEMORY_ONLY_CONTEXT cold-start, where the LLM otherwise
        # has neither chat history nor memory).
        self._bootstrap_doc = (
            f"Game start. Initial observation:\n{self._last_observation}"
        )

        # Keep a reference to the system message dict so we can dynamically
        # inject placeholder tokens when memory documents are created.
        self._system_message = prompt[0] if prompt else None
        base_content = prompt[0]["content"] if prompt else ""
        # Append TextWorld Inform7 vocabulary hint. Debug rollouts showed the
        # frozen LLM emits natural-English variants ("look around", "pick up",
        # "inspect", "examine <room>") that the parser rejects, causing
        # permanent stuck loops with no scoring signal.
        base_content = base_content.rstrip() + _TEXTWORLD_VOCAB_HINT
        self._base_system_content = base_content
        if self._system_message is not None:
            self._system_message["content"] = base_content

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
        """Update the modality payloads with current memory documents.

        The system message has TWO sections (no [Memory recap] plaintext anymore):
          1. The base system content from the parquet prompt
          2. A `[Memory tokens]` placeholder block (`<|image_pad|>` × `max_placeholder_tokens`)
             whose positions get replaced at LLM forward time with the trainable
             encoder's compressed vectors.

        The recap text used to live here too as a redundant plaintext copy of the
        last K finalized docs. We removed it: with the no-redundancy IO design,
        finalized turns are ONLY in the memory tokens, and unfinalized
        (current-segment) turns are exposed via `current_segment_chat_messages()`
        as raw user/assistant pairs in the LLM input.
        """
        modalities_entry = self.extras.get("modalities")
        if modalities_entry is None or not hasattr(modalities_entry, "payloads"):
            return

        documents = list(self._memory.memory_documents)
        docs_with_content = [doc for doc in documents if doc] if documents else []

        from skyrl_train.dataset.modalities import ModalityPlaceholderPlan

        payload_docs: List[str] = []
        if docs_with_content:
            payload_docs = list(docs_with_content)
        elif self.inject_placeholder_pre_memory:
            bootstrap = getattr(self, "_bootstrap_doc", "") or "Game start."
            payload_docs = [bootstrap]

        if payload_docs and self._system_message is not None:
            placeholder_block = " ".join(
                [self.placeholder_token] * self.max_placeholder_tokens
            )
            self._system_message["content"] = (
                f"{self._base_system_content}"
                f"\n[Memory tokens]\n{placeholder_block}"
            )

            modalities_entry.payloads[self.modality_id] = [payload_docs]
            modalities_entry.plans[self.modality_id] = ModalityPlaceholderPlan(
                modality_id=self.modality_id,
                placeholder_token=self.placeholder_token,
                occurrences=1,
                reserved_tokens=[self.max_placeholder_tokens],
                payload=[payload_docs],
            )
        else:
            if self._system_message is not None:
                self._system_message["content"] = self._base_system_content
            modalities_entry.payloads.pop(self.modality_id, None)
            modalities_entry.plans.pop(self.modality_id, None)

    def current_segment_chat_messages(self) -> List[Dict[str, str]]:
        """Replay unfinalized turns as user/assistant chat messages.

        These are turns that the memory tokens have NOT yet absorbed (because they
        haven't reached `memory_window` yet). The Phase-2 rollout path includes
        these in the LLM input so the model has access to recent state without
        duplicating what the memory tokens already cover.

        Does NOT include the current turn's observation — the caller appends that
        as the final user message.
        """
        return self._memory.current_segment_chat_messages()

    def current_system_message(self) -> Dict[str, Any]:
        """Return a fresh dict reflecting the env's current (un-expanded) system message.

        The Phase-2 memory-only rollout path uses this to rebuild the LLM input
        each turn without trusting a stale `chat_history[0]` (which the multimodal
        processor copies and post-expands, breaking the alias to `_system_message`).
        """
        if self._system_message is None:
            return {"role": "system", "content": ""}
        return {
            "role": self._system_message.get("role", "system"),
            "content": self._system_message.get("content", ""),
        }

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1

        # Capture any <think>...</think> reasoning from the raw model response
        # BEFORE parsing the action — _parse_action strips think tags. We only
        # store thinking if the env was configured to include it.
        thinking = ""
        if self.include_thinking_in_memory and action:
            tm = re.search(r"<think>(.*?)</think>", action, re.DOTALL | re.IGNORECASE)
            if tm:
                thinking = tm.group(1).strip()

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
            thinking=thinking,
            action=parsed_action,
            reward=float(reward),
            score=int(score),
            raw_response=action or "",
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

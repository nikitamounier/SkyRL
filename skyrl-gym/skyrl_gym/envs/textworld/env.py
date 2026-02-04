from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import threading
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, ConversationType


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
                self.memory_documents = self.memory_documents[-self.max_documents :]
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


_TEXTWORLD_START_LOCK = threading.Lock()


class TextWorldEnv(BaseTextEnv):
    """
    TextWorld environment with incremental memory document creation.

    Expects `game_file` in extras. Optionally uses modalities payloads to supply
    memory documents to a memory encoder modality.
    """

    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] = {}):
        super().__init__()

        self.env_config = env_config
        self.extras = extras

        game_file = extras.get("game_file") or extras.get("extra_info", {}).get("game_file")
        if not game_file:
            raise ValueError("TextWorldEnv requires `game_file` in extras.")
        self.game_file = str(game_file)

        self.max_turns = int(extras.get("max_turns") or env_config.get("max_turns", 50))
        self.memory_window = int(extras.get("memory_window") or env_config.get("memory_window", 5))
        self.max_memory_docs = int(extras.get("max_memory_docs") or env_config.get("max_memory_docs", 4))
        self.max_doc_tokens = int(extras.get("max_doc_tokens") or env_config.get("max_doc_tokens", 256))
        self.tokenizer_path = str(extras.get("tokenizer_path") or env_config.get("tokenizer_path", ""))

        self.modality_id = str(extras.get("memory_modality_id") or env_config.get("memory_modality_id", "memo_memory"))

        self._env = None
        self._game_state = None
        self._last_observation: str = ""

        self._memory = _IncrementalMemory(
            memory_window=self.memory_window,
            include_thinking=False,
            max_documents=self.max_memory_docs,
        )

    def init(self, prompt: ConversationType):
        env = self._get_env()
        self._game_state = env.reset()
        self.turns = 0
        self._memory = _IncrementalMemory(
            memory_window=self.memory_window,
            include_thinking=False,
            max_documents=self.max_memory_docs,
        )
        self._last_observation = self._game_state.feedback.strip()

        self._update_modalities_payload()

        first_obs = {"role": "user", "content": self._last_observation}
        return prompt + [first_obs], {}

    def _get_env(self):
        if self._env is None:
            try:
                import textworld  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise ImportError(
                    "TextWorld is not installed. Install it to use TextWorldEnv."
                ) from exc
            # Use textworld.start so Inform7 wrappers are applied when available.
            # textworld.logic uses a global parser that is not thread-safe; guard startup.
            with _TEXTWORLD_START_LOCK:
                self._env = textworld.start(self.game_file)
        return self._env

    def _parse_action(self, action: str) -> str:
        if not action:
            return ""
        cleaned = action.replace("<think>", "").replace("</think>", "").strip()
        if "</think>" in action:
            _, after = action.split("</think>", 1)
            cleaned = after.strip() or cleaned
        for prefix in ("Action:", "I will", "I'll", "Let me", "I would", "I should"):
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix) :].strip()
        if "\n" in cleaned:
            cleaned = cleaned.split("\n", 1)[0].strip()
        return cleaned.strip('"\'.')

    def _update_modalities_payload(self) -> None:
        modalities_entry = self.extras.get("modalities")
        if modalities_entry is None or not hasattr(modalities_entry, "payloads"):
            return

        documents = list(self._memory.memory_documents)
        payloads: List[List[int]] = []
        for doc in documents:
            payloads.append(self._encode_document(doc))

        while len(payloads) < self.max_memory_docs:
            payloads.append([])

        modalities_entry.payloads[self.modality_id] = payloads

    def _encode_document(self, text: str) -> List[int]:
        if not text:
            return []
        tokenizer = _get_tokenizer(self.tokenizer_path)
        encoded = tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_doc_tokens,
        )
        return list(encoded)

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1
        parsed_action = self._parse_action(action) or "look"

        env = self._get_env()
        self._game_state, reward, done = env.step(parsed_action)
        score = getattr(self._game_state, "score", 0)

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

        self._last_observation = self._game_state.feedback.strip()
        done = done or self.turns >= self.max_turns

        observations = [] if done else [{"role": "user", "content": self._last_observation}]

        return BaseTextEnvStepOutput(
            observations=observations,
            reward=float(reward),
            done=done,
            metadata={
                "action": parsed_action,
                "score": int(score),
                "won": bool(getattr(self._game_state, "won", False)),
            },
        )

    def close(self):
        if self._env is not None:
            self._env.close()
        self._env = None
        self._game_state = None

    def get_metrics(self) -> Dict[str, Any]:
        score = 0
        if self._game_state is not None:
            raw_score = getattr(self._game_state, "score", 0)
            if raw_score is None:
                raw_score = 0
            score = int(raw_score)
        return {
            "steps": self.turns,
            "won": bool(getattr(self._game_state, "won", False)) if self._game_state is not None else False,
            "score": score,
        }


@lru_cache(maxsize=4)
def _get_tokenizer(model_path: str):
    if not model_path:
        raise ValueError("TextWorldEnv requires `tokenizer_path` to encode memory documents.")
    from transformers import AutoTokenizer  # local import to avoid hard dependency

    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

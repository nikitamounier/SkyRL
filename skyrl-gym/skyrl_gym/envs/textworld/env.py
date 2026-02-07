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
        import sys
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
        sys.stderr.write(f"[MEMORY] Turn {turn}: current_segment has {len(self.current_segment)} turns (window={self.memory_window})\n")
        sys.stderr.flush()
        if len(self.current_segment) >= self.memory_window:
            sys.stderr.write(f"[MEMORY] Finalizing segment with {len(self.current_segment)} turns\n")
            sys.stderr.flush()
            self._finalize_segment()

    def finalize(self) -> None:
        if self.current_segment:
            self._finalize_segment()

    def _finalize_segment(self) -> None:
        import sys
        document = self._create_document_text()
        sys.stderr.write(f"[MEMORY] Created document: {len(document)} chars\n")
        sys.stderr.flush()
        if document:
            self.memory_documents.append(document)
            sys.stderr.write(f"[MEMORY] Appended document. Total documents: {len(self.memory_documents)}\n")
            sys.stderr.flush()
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
        self.placeholder_token = str(extras.get("placeholder_token") or env_config.get("placeholder_token", "<|image_pad|>"))
        self.max_placeholder_tokens = int(extras.get("max_placeholder_tokens") or env_config.get("max_placeholder_tokens", 8))

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
        """
        Parse action from model output. Model generates reasoning + [ACTION: command].
        We train on the full output but only send the command to the game.
        """
        if not action:
            return ""

        # Clean UTF-8 encoding issues first
        try:
            # Ensure valid UTF-8 encoding
            action = action.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
        except Exception:
            action = ""

        if not action:
            return ""

        # First try to extract [ACTION: ...] format
        import re
        action_match = re.search(r'\[ACTION:\s*([^\]]+)\]', action, re.IGNORECASE)
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
                    cleaned = cleaned[len(prefix) :].strip()
            if "\n" in cleaned:
                cleaned = cleaned.split("\n", 1)[0].strip()
            parsed = cleaned.strip('"\'.')

        # Final sanitization: remove any non-ASCII or control characters that might cause issues
        # Keep only alphanumeric, spaces, and common punctuation
        import string
        allowed_chars = string.ascii_letters + string.digits + string.whitespace + "-_.,!?'"
        parsed = ''.join(c for c in parsed if c in allowed_chars)

        # Ensure it's valid UTF-8 and limited length
        parsed = parsed.strip()[:200]  # Limit action length

        return parsed

    def _update_modalities_payload(self) -> None:
        import sys
        modalities_entry = self.extras.get("modalities")
        if modalities_entry is None or not hasattr(modalities_entry, "payloads"):
            sys.stderr.write(f"[ENV PAYLOAD] modalities_entry is None or has no payloads\n")
            sys.stderr.flush()
            return

        documents = list(self._memory.memory_documents)
        sys.stderr.write(f"[ENV PAYLOAD] Found {len(documents)} memory documents\n")
        sys.stderr.flush()

        from skyrl_train.dataset.modalities import ModalityPlaceholderPlan

        if len(documents) > 0:
            payloads: List[List[int]] = []
            for idx, doc in enumerate(documents):
                encoded = self._encode_document(doc)
                sys.stderr.write(f"[ENV PAYLOAD] Doc {idx}: {len(doc) if doc else 0} chars -> {len(encoded)} tokens\n")
                sys.stderr.flush()
                payloads.append(encoded)

            while len(payloads) < self.max_memory_docs:
                payloads.append([])

            modalities_entry.payloads[self.modality_id] = payloads

            num_docs_with_content = sum(1 for p in payloads if p)
            plan = ModalityPlaceholderPlan(
                modality_id=self.modality_id,
                placeholder_token=self.placeholder_token,
                occurrences=num_docs_with_content,
                reserved_tokens=[self.max_placeholder_tokens] * num_docs_with_content,
                payload=payloads[:num_docs_with_content],
            )
            modalities_entry.plans[self.modality_id] = plan
            sys.stderr.write(f"[ENV PLAN] Created plan: {num_docs_with_content} docs, {self.max_placeholder_tokens} tokens each\n")
            sys.stderr.flush()
        else:
            # No documents - clear payloads and plans
            modalities_entry.payloads.pop(self.modality_id, None)
            modalities_entry.plans.pop(self.modality_id, None)
            sys.stderr.write(f"[ENV PLAN] No documents, cleared payloads and plan\n")
            sys.stderr.flush()

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

        # Handle Unicode errors in game feedback
        try:
            feedback = self._game_state.feedback.strip()
        except (UnicodeDecodeError, AttributeError) as e:
            # If feedback has encoding issues, try to clean it
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Unicode error in game feedback: {e}. Attempting to clean...")
            try:
                # Try to encode and decode with error handling
                if isinstance(self._game_state.feedback, bytes):
                    feedback = self._game_state.feedback.decode('utf-8', errors='replace').strip()
                else:
                    # If it's already a string, try to re-encode and decode
                    feedback = str(self._game_state.feedback).encode('utf-8', errors='replace').decode('utf-8').strip()
            except Exception as e2:
                logger.error(f"Failed to clean feedback: {e2}. Using placeholder.")
                feedback = "[Game output could not be decoded]"
        self._last_observation = feedback
        done = done or self.turns >= self.max_turns

        observations = [] if done else [{"role": "user", "content": self._last_observation}]

        metadata = {
            "action": parsed_action,
            "score": int(score),
            "won": bool(getattr(self._game_state, "won", False)),
        }

        # Include modalities if available (updated by _update_modalities_payload)
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

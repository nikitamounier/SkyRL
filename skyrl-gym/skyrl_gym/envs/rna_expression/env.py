import re
from typing import Any, Dict

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput

# Ordinal expression bins used by the BioReason-RNA task.
BINS = ["very low", "low", "medium", "high", "very high"]
_BIN_INDEX = {b: i for i, b in enumerate(BINS)}
_BOXED = re.compile(r"\\boxed\{([^}]*)\}")


def _normalize(text: str) -> str:
    return " ".join(text.strip().lower().replace("_", " ").split())


def parse_bin(action: str) -> str | None:
    """Extract the predicted expression bin from a completion.

    Prefers the last ``\\boxed{...}``; falls back to the last bin phrase mentioned.
    """
    candidates = [m.group(1) for m in _BOXED.finditer(action)]
    for raw in reversed(candidates):
        norm = _normalize(raw)
        if norm in _BIN_INDEX:
            return norm
    # fallback: last bin keyword appearing in the text (longest match first)
    norm_text = _normalize(action)
    best = None
    for b in BINS:
        idx = norm_text.rfind(b)
        if idx != -1 and (best is None or idx > best[0]):
            best = (idx, b)
    return best[1] if best else None


class RNAExpressionEnv(BaseTextEnv):
    """Real reward for RNA expression-bin prediction.

    Reward is ordinal-graded against the ground-truth bin: exact match = 1.0,
    off-by-one = 0.75, ... (1 - |delta|/4). A small format bonus is given for
    emitting a parseable ``\\boxed{<bin>}`` even when wrong; missing/invalid
    answers score 0.
    """

    def __init__(self, env_config: Any = None, extras: Dict[str, Any] = {}):
        super().__init__()
        spec = (extras or {}).get("reward_spec", {}) or {}
        assert "ground_truth" in spec, "reward_spec.ground_truth (expression bin) is required"
        self.gt = _normalize(str(spec["ground_truth"]))
        assert self.gt in _BIN_INDEX, f"ground_truth must be one of {BINS}, got {self.gt!r}"
        self.format_bonus = float(spec.get("format_bonus", 0.05))

    def _get_reward(self, action: str) -> float:
        pred = parse_bin(action)
        if pred is None:
            return 0.0
        delta = abs(_BIN_INDEX[pred] - _BIN_INDEX[self.gt])
        graded = 1.0 - delta / (len(BINS) - 1)        # 1.0, 0.75, 0.5, 0.25, 0.0
        if pred == self.gt:
            return 1.0
        return self.format_bonus + (1.0 - self.format_bonus) * graded

    def step(self, action: str) -> BaseTextEnvStepOutput:
        reward = self._get_reward(action)
        return BaseTextEnvStepOutput(observations=[], reward=reward, done=True, metadata={"gt": self.gt})

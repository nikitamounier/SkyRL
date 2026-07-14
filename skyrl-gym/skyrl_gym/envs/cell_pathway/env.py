import re
from typing import Any, Dict

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput

# 3-class pathway-response label used by the BioReasonCell gene->pathway task.
# Mirrors bioreason_cell evals/gpt_baseline/common.py CLASSES.
CLASSES = ("upregulated", "downregulated", "unchanged")
_BOXED = re.compile(r"\\boxed\{([^}]*)\}")


def parse_boxed(text: str | None) -> str | None:
    """Extract the predicted pathway-response class from a completion.

    Takes the last ``\\boxed{...}``, lowercases it, and returns the matching
    class. Falls back to a keyword scan when the raw boxed value is not an exact
    class string. Returns ``None`` when nothing parseable is found.
    """
    if not text:
        return None
    matches = _BOXED.findall(text)
    if not matches:
        return None
    value = str(matches[-1]).strip().lower()
    if value in CLASSES:
        return value
    for cls in CLASSES:
        if cls in value:
            return cls
    return None


class CellPathwayEnv(BaseTextEnv):
    """Real reward for BioReasonCell gene->pathway direction prediction.

    The model predicts a queried pathway's directional response to a genetic
    perturbation as ``\\boxed{upregulated|downregulated|unchanged}``. Reward is
    exact 3-class match = 1.0; a parseable-but-wrong box earns a small format
    bonus; a missing/invalid answer scores 0. This matches the 3-class direction
    accuracy used for the SFT evals.
    """

    def __init__(self, env_config: Any = None, extras: Dict[str, Any] = {}):
        super().__init__()
        spec = (extras or {}).get("reward_spec", {}) or {}
        assert "ground_truth" in spec, "reward_spec.ground_truth (pathway_change) is required"
        self.gt = str(spec["ground_truth"]).strip().lower()
        assert self.gt in CLASSES, f"ground_truth must be one of {CLASSES}, got {self.gt!r}"
        self.format_bonus = float(spec.get("format_bonus", 0.05))

    def _get_reward(self, action: str) -> float:
        pred = parse_boxed(action)
        if pred is None:
            return 0.0
        if pred == self.gt:
            return 1.0
        return self.format_bonus

    def step(self, action: str) -> BaseTextEnvStepOutput:
        reward = self._get_reward(action)
        return BaseTextEnvStepOutput(observations=[], reward=reward, done=True, metadata={"gt": self.gt})

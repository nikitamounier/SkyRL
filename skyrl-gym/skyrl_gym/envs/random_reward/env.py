import random
from typing import Any, Dict

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput


class RandomRewardEnv(BaseTextEnv):
    """Single-turn env that returns a random reward regardless of the action.

    Used as a plumbing/smoke-test env: the random reward yields nonzero
    advantage variance across the GRPO sample group, so the policy backward
    pass produces a nonzero gradient (unlike a task where every sample scores 0).
    """

    def __init__(self, env_config: Any = None, extras: Dict[str, Any] = {}):
        super().__init__()
        # Optional reward_spec.low/high bounds; default to [0, 1).
        spec = (extras or {}).get("reward_spec", {}) or {}
        self.low = float(spec.get("low", 0.0))
        self.high = float(spec.get("high", 1.0))

    def step(self, action: str) -> BaseTextEnvStepOutput:
        reward = random.uniform(self.low, self.high)
        return BaseTextEnvStepOutput(observations=[], reward=reward, done=True, metadata={})

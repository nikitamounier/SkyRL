"""Registers the internal gym envs."""

from skyrl_gym.envs.registration import register

register(
    id="aime",
    entry_point="skyrl_gym.envs.aime.env:AIMEEnv",
)

register(
    id="gsm8k",
    entry_point="skyrl_gym.envs.gsm8k.env:GSM8kEnv",
)

register(
    id="gsm8k_multi_turn",
    entry_point="skyrl_gym.envs.gsm8k.multi_turn_env:GSM8kMultiTurnEnv",
)

register(
    id="text2sql",
    entry_point="skyrl_gym.envs.sql.env:SQLEnv",
)

register(
    id="search",
    entry_point="skyrl_gym.envs.search.env:SearchEnv",
)

register(
    id="lcb",
    entry_point="skyrl_gym.envs.lcb.env:LCBEnv",
)

register(
    id="searchcode",
    entry_point="skyrl_gym.envs.searchcode.env:SearchCodeEnv",
)

register(
    id="textworld",
    entry_point="skyrl_gym.envs.textworld.env:TextWorldEnv",
)

register(
    id="fast_textworld",
    entry_point="skyrl_gym.envs.textworld.fast_env:FastTextWorldEnv",
)

register(
    id="cell_pathway",
    entry_point="skyrl_gym.envs.cell_pathway.env:CellPathwayEnv",
)

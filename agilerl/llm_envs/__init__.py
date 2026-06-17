"""Gymnasium-style environments for LLM training."""

from agilerl.llm_envs import search as _search
from agilerl.llm_envs.base import (
    HuggingFaceGym,
    IterablePromptBatchGym,
    apply_chat_template,
)
from agilerl.llm_envs.preference import PreferenceGym, PreferenceGymV2
from agilerl.llm_envs.reasoning import ReasoningGym, ReasoningGymV2
from agilerl.llm_envs.search import TIMEOUT, FormatRewardWrapper, SearchTool
from agilerl.llm_envs.sft import SFTGym
from agilerl.llm_envs.sync_vec_env import (
    SyncMultiTurnVecEnv,
    Trajectory,
    TrajectoryBuffer,
)
from agilerl.llm_envs.token_observation import TokenObservationWrapper

requests = _search.requests

__all__ = [
    "TIMEOUT",
    "FormatRewardWrapper",
    "HuggingFaceGym",
    "IterablePromptBatchGym",
    "PreferenceGym",
    "PreferenceGymV2",
    "ReasoningGym",
    "ReasoningGymV2",
    "SFTGym",
    "SearchTool",
    "SyncMultiTurnVecEnv",
    "TokenObservationWrapper",
    "Trajectory",
    "TrajectoryBuffer",
    "apply_chat_template",
    "requests",
]

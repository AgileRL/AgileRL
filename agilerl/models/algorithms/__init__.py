"""Algorithm specification implementations."""

from __future__ import annotations

from agilerl import HAS_ARENA_DEPENDENCIES, HAS_LLM_DEPENDENCIES

from .cqn import CQNSpec
from .ddpg import DDPGSpec
from .dqn import DQNSpec
from .ippo import IPPOSpec
from .maddpg import MADDPGSpec
from .matd3 import MATD3Spec
from .neural_ts import NeuralTSSpec
from .neural_ucb import NeuralUCBSpec
from .ppo import PPOSpec
from .rainbow_dqn import RainbowDQNSpec
from .td3 import TD3Spec

if HAS_ARENA_DEPENDENCIES:
    import agilerl.arena.models.algorithms as _arena_algorithms  # noqa: F401

if HAS_LLM_DEPENDENCIES:
    from .cispo import CISPOSpec
    from .dpo import DPOSpec
    from .grpo import GRPOSpec
    from .gspo import GSPOSpec
    from .llmppo import LLMPPOSpec
    from .llmreinforce import LLMREINFORCESpec
    from .sft import SFTSpec

__all__ = [
    "CQNSpec",
    "DDPGSpec",
    "DQNSpec",
    "IPPOSpec",
    "MADDPGSpec",
    "MATD3Spec",
    "NeuralTSSpec",
    "NeuralUCBSpec",
    "PPOSpec",
    "RainbowDQNSpec",
    "TD3Spec",
]

if HAS_LLM_DEPENDENCIES:
    __all__ += [
        "CISPOSpec",
        "DPOSpec",
        "GRPOSpec",
        "GSPOSpec",
        "LLMPPOSpec",
        "LLMREINFORCESpec",
        "SFTSpec",
    ]

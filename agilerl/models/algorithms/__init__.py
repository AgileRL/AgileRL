"""Algorithm specification implementations."""

from __future__ import annotations

import importlib

from agilerl import HAS_LLM_DEPENDENCIES

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

_submod_names: list[str] = [
    "cqn",
    "ddpg",
    "dqn",
    "ippo",
    "maddpg",
    "matd3",
    "neural_ts",
    "neural_ucb",
    "ppo",
    "rainbow_dqn",
    "td3",
]

if HAS_LLM_DEPENDENCIES:
    from .dpo import DPOSpec
    from .grpo import GRPOSpec

    _submod_names += ["dpo", "grpo"]

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
    "populate_registry",
]
if HAS_LLM_DEPENDENCIES:
    __all__ += ["DPOSpec", "GRPOSpec"]


def populate_registry() -> None:
    """Eagerly import every spec module to populate :data:`ALGO_REGISTRY`.

    Each spec module uses the ``@register`` decorator which inserts an
    entry into the global :data:`~agilerl.models.algo.ALGO_REGISTRY` at
    import time.  Call this function before performing registry lookups
    that require all algorithms to be available (e.g. manifest validation).
    """
    for submod in _submod_names:
        importlib.import_module(f"{__name__}.{submod}")

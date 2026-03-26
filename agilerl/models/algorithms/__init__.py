"""Algorithm specification implementations."""

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
    from .dpo import DPOSpec
    from .grpo import GRPOSpec

    __all__ += ["DPOSpec", "GRPOSpec"]

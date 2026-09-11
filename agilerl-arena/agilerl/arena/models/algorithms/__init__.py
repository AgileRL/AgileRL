# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Algorithm specification implementations."""

from __future__ import annotations

from .cispo import CISPOSpec
from .cqn import CQNSpec
from .ddpg import DDPGSpec
from .dpo import DPOSpec
from .dqn import DQNSpec
from .grpo import GRPOSpec
from .gspo import GSPOSpec
from .ippo import IPPOSpec
from .llmppo import LLMPPOSpec
from .llmreinforce import LLMREINFORCESpec
from .maddpg import MADDPGSpec
from .matd3 import MATD3Spec
from .neural_ts import NeuralTSSpec
from .neural_ucb import NeuralUCBSpec
from .ppo import PPOSpec
from .rainbow_dqn import RainbowDQNSpec
from .sft import SFTSpec
from .td3 import TD3Spec

__all__ = [
    "CISPOSpec",
    "CQNSpec",
    "DDPGSpec",
    "DPOSpec",
    "DQNSpec",
    "GRPOSpec",
    "GSPOSpec",
    "IPPOSpec",
    "LLMPPOSpec",
    "LLMREINFORCESpec",
    "MADDPGSpec",
    "MATD3Spec",
    "NeuralTSSpec",
    "NeuralUCBSpec",
    "PPOSpec",
    "RainbowDQNSpec",
    "SFTSpec",
    "TD3Spec",
]

# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Algorithm specification implementations."""

from __future__ import annotations

from .cispo import CISPOSpec
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
from .ppo import PPOSpec
from .rainbow_dqn import RainbowDQNSpec
from .sft import SFTSpec
from .td3 import TD3Spec

__all__ = [
    "CISPOSpec",
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
    "PPOSpec",
    "RainbowDQNSpec",
    "SFTSpec",
    "TD3Spec",
]

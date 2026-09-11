# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Algorithm spec classes. Importing this package registers each spec."""

from __future__ import annotations

from agilerl.arena.models.algorithms.bandit import BanditSpec
from agilerl.arena.models.algorithms.base import (
    AlgorithmSpec,
    AlgoSpec,
    LLMAlgorithmSpec,
    MultiAgentAlgorithmSpec,
    SingleAgentAlgorithmSpec,
)
from agilerl.arena.models.algorithms.cispo import CISPOSpec
from agilerl.arena.models.algorithms.cqn import CQNSpec
from agilerl.arena.models.algorithms.ddpg import DDPGSpec
from agilerl.arena.models.algorithms.dpo import DPOSpec
from agilerl.arena.models.algorithms.dqn import DQNSpec
from agilerl.arena.models.algorithms.grpo import GRPOSpec
from agilerl.arena.models.algorithms.gspo import GSPOSpec
from agilerl.arena.models.algorithms.ippo import IPPOSpec
from agilerl.arena.models.algorithms.llmppo import LLMPPOSpec
from agilerl.arena.models.algorithms.llmreinforce import LLMREINFORCESpec
from agilerl.arena.models.algorithms.maddpg import MADDPGSpec
from agilerl.arena.models.algorithms.matd3 import MATD3Spec
from agilerl.arena.models.algorithms.neural_ts import NeuralTSSpec
from agilerl.arena.models.algorithms.neural_ucb import NeuralUCBSpec
from agilerl.arena.models.algorithms.ppo import PPOSpec
from agilerl.arena.models.algorithms.rainbow_dqn import RainbowDQNSpec
from agilerl.arena.models.algorithms.rollout_llm import RolloutLLMSpec
from agilerl.arena.models.algorithms.sft import SFTSpec
from agilerl.arena.models.algorithms.td3 import TD3Spec

__all__ = [
    "AlgoSpec",
    "AlgorithmSpec",
    "BanditSpec",
    "CISPOSpec",
    "CQNSpec",
    "DDPGSpec",
    "DPOSpec",
    "DQNSpec",
    "GRPOSpec",
    "GSPOSpec",
    "IPPOSpec",
    "LLMAlgorithmSpec",
    "LLMPPOSpec",
    "LLMREINFORCESpec",
    "MADDPGSpec",
    "MATD3Spec",
    "MultiAgentAlgorithmSpec",
    "NeuralTSSpec",
    "NeuralUCBSpec",
    "PPOSpec",
    "RainbowDQNSpec",
    "RolloutLLMSpec",
    "SFTSpec",
    "SingleAgentAlgorithmSpec",
    "TD3Spec",
]

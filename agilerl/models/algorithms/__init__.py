# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Algorithm spec classes. Importing this package registers each spec."""

from __future__ import annotations

from agilerl.models.algorithms.base import (
    AlgorithmSpec,
    AlgoSpec,
    LLMAlgorithmSpec,
)
from agilerl.models.algorithms.cispo import CISPOSpec
from agilerl.models.algorithms.dpo import DPOSpec
from agilerl.models.algorithms.grpo import GRPOSpec
from agilerl.models.algorithms.gspo import GSPOSpec
from agilerl.models.algorithms.llmppo import LLMPPOSpec
from agilerl.models.algorithms.llmreinforce import LLMREINFORCESpec
from agilerl.models.algorithms.opd import OPDSpec
from agilerl.models.algorithms.rollout_llm import RolloutLLMSpec
from agilerl.models.algorithms.sft import SFTSpec

__all__ = [
    "AlgoSpec",
    "AlgorithmSpec",
    "CISPOSpec",
    "DPOSpec",
    "GRPOSpec",
    "GSPOSpec",
    "LLMAlgorithmSpec",
    "LLMPPOSpec",
    "LLMREINFORCESpec",
    "OPDSpec",
    "RolloutLLMSpec",
    "SFTSpec",
]

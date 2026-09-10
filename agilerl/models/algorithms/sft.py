# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""SFT algorithm specification."""

from __future__ import annotations

from agilerl.arena.models.algorithms.sft import SFTSpec as ArenaSFTSpec
from agilerl.models.algo import LLMAlgorithmSpec, register


@register()
class SFTSpec(LLMAlgorithmSpec, ArenaSFTSpec):
    """Specification for SFT algorithm."""

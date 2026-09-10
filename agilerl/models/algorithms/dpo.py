# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""DPO algorithm specification."""

from __future__ import annotations

from agilerl.arena.models.algorithms.dpo import DPOSpec as ArenaDPOSpec
from agilerl.models.algo import LLMAlgorithmSpec, register


@register()
class DPOSpec(LLMAlgorithmSpec, ArenaDPOSpec):
    """Specification for DPO algorithm."""

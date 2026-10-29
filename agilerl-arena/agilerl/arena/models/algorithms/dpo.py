# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""DPO algorithm specification."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from agilerl.arena.models.algorithms.base import LLMAlgorithmSpec
from agilerl.arena.models.descriptions import BETA, LR
from agilerl.arena.models.env import LLMEnvType
from agilerl.arena.models.registry import register


@register()
class DPOSpec(LLMAlgorithmSpec):
    """Direct Preference Optimization."""

    lr: float = Field(default=0.000005, ge=0.0, description=LR)
    beta: float = Field(default=0.1, ge=0.0, le=1.0, description=BETA)
    nll_alpha: float = Field(
        default=1.0,
        ge=0.0,
        description=(
            "Weight on the auxiliary likelihood term over chosen responses, "
            "which keeps the model fluent while preferences are optimized."
        ),
    )

    env_type: ClassVar[LLMEnvType] = LLMEnvType.DATASET
    objective: ClassVar[str] = "preference"

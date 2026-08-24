# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""DPO algorithm specification."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from agilerl.arena.models.algo import LLMAlgorithmSpec, register
from agilerl.arena.models.env import LLMEnvType


@register()
class DPOSpec(LLMAlgorithmSpec):
    """Specification for DPO algorithm."""

    lr: float = Field(default=0.000005)
    env_type: ClassVar[LLMEnvType] = LLMEnvType.DATASET

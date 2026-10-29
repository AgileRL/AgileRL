# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""SFT algorithm specification."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from agilerl.arena.models.algorithms.base import LLMAlgorithmSpec
from agilerl.arena.models.descriptions import LR
from agilerl.arena.models.env import LLMEnvType
from agilerl.arena.models.registry import register


@register()
class SFTSpec(LLMAlgorithmSpec):
    """Supervised fine-tuning. No reference model, so no KL penalty."""

    lr: float = Field(default=0.00005, ge=0.0, description=LR)
    use_separate_reference_adapter: bool = Field(
        default=False,
        description=("Always off: SFT has no reference policy to hold in an adapter."),
    )

    env_type: ClassVar[LLMEnvType] = LLMEnvType.DATASET
    objective: ClassVar[str] = "sft"

# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""SFT algorithm specification."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from agilerl.models.algorithms.base import LLMAlgorithmSpec
from agilerl.models.descriptions import LR
from agilerl.models.env import LLMEnvType
from agilerl.models.registry import register


@register()
class SFTSpec(LLMAlgorithmSpec):
    """Supervised fine-tuning. No reference model, so no KL penalty."""

    lr: float = Field(default=0.00005, gt=0.0, le=1.0, description=LR)
    use_separate_reference_adapter: bool = Field(
        default=False,
        description=("Always off: SFT has no reference policy to hold in an adapter."),
    )
    beta: float = Field(
        default=0.001,
        ge=0.0,
        le=1.0,
        exclude=True,
        description="Unused: SFT has no reference policy to penalise against.",
    )
    answer_continuation: bool = Field(
        default=False,
        exclude=True,
        description="Unused: SFT trains on dataset rows, with no rollout to continue.",
    )

    env_type: ClassVar[LLMEnvType] = LLMEnvType.DATASET
    objective: ClassVar[str] = "sft"
    supports_sequence_packing: ClassVar[bool] = True

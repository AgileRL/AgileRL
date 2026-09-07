# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""SFT algorithm specification."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field, model_validator
from typing_extensions import Self

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

    @model_validator(mode="after")
    def _validate_use_sequence_packing(self) -> Self:
        """SFT's forward is padded; packing would never be built for it."""
        if self.use_sequence_packing:
            msg = (
                "SFT does not accept use_sequence_packing, so the padding-free "
                "forward would never be built for this algorithm"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _validate_activation_offload(self) -> Self:
        """SFT's build never forwards an offload request, so reject it here."""
        if self.activation_offload:
            msg = (
                "SFT does not forward activation_offload to its constructor, so "
                "the request would never reach the training forward. Leave "
                "activation_offload unset for SFT."
            )
            raise ValueError(msg)
        return self

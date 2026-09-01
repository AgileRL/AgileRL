# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""SFT algorithm specification."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

from pydantic import Field

from agilerl.models.algo import LLMAlgorithmSpec, register
from agilerl.models.env_types import LLMEnvType


@register()
class SFTSpec(LLMAlgorithmSpec):
    """Specification for SFT algorithm."""

    lr: float = Field(default=0.00005)

    env_type: ClassVar[LLMEnvType] = LLMEnvType.DATASET
    objective: ClassVar[str] = "sft"

    @staticmethod
    def get_training_fn() -> Callable[..., Any]:
        """Get the training function for SFT.

        :return: Training function
        :rtype: Callable[..., Any]
        """
        from agilerl.training.llm import (  # circular import with agilerl.training
            train_llm_dataset,
        )

        return train_llm_dataset

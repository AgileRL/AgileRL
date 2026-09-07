# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""SFT algorithm specification."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agilerl.arena.models.algorithms.sft import SFTSpec as ArenaSFTSpec
from agilerl.models.algo import LLMAlgorithmSpec, register


@register()
class SFTSpec(LLMAlgorithmSpec, ArenaSFTSpec):
    """Specification for SFT algorithm."""

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

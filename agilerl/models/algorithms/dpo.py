# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""DPO algorithm specification."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agilerl.arena.models.algorithms.dpo import DPOSpec as ArenaDPOSpec
from agilerl.models.algo import LLMAlgorithmSpec, register


@register()
class DPOSpec(LLMAlgorithmSpec, ArenaDPOSpec):
    """Specification for DPO algorithm."""

    @staticmethod
    def get_training_fn() -> Callable[..., Any]:
        """Get the training function for DPO.

        :return: Training function
        :rtype: Callable[..., Any]
        """
        from agilerl.training.llm import (  # circular import with agilerl.training
            train_llm_dataset,
        )

        return train_llm_dataset

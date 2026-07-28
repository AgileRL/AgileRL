# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""GSPO algorithm specification."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agilerl.models.algo import register
from agilerl.models.algorithms.grpo import GRPOSpec


@register()
class GSPOSpec(GRPOSpec):
    """Specification for GSPO algorithm (GRPO with GSPO loss)."""

    @staticmethod
    def get_training_fn(*, multiturn: bool = False) -> Callable[..., Any]:
        """Get the training function for GSPO.

        :param multiturn: If ``True``, return the multi-turn training function.
        :type multiturn: bool
        :return: Training function
        :rtype: Callable[..., Any]
        """
        from agilerl.training.llm import (
            finetune_llm_multiturn,
            finetune_llm_reasoning,
        )

        return finetune_llm_multiturn if multiturn else finetune_llm_reasoning

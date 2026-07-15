"""CISPO algorithm specification."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import Field

from agilerl.models.algo import register
from agilerl.models.algorithms.grpo import GRPOSpec


@register()
class CISPOSpec(GRPOSpec):
    """Specification for CISPO algorithm (GRPO with CISPO loss)."""

    # CISPO uses asymmetric clip bounds [epsilon_low, epsilon_high].
    clip_coef: float | list[float] = Field(default=0.2)

    @staticmethod
    def get_training_fn(*, multiturn: bool = False) -> Callable[..., Any]:
        """Get the training function for CISPO.

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

# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

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
    def get_training_fn() -> Callable[..., Any]:
        """Get the training function for CISPO.

        :return: Training function
        :rtype: Callable[..., Any]
        """
        from agilerl.training.llm import train_llm_rollout

        return train_llm_rollout

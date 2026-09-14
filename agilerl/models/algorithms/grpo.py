# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""GRPO algorithm specification."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import Field

from agilerl.arena.models.algorithms.grpo import GRPOSpec as ArenaGRPOSpec
from agilerl.models.algo import LLMAlgorithmSpec, register


@register()
class GRPOSpec(LLMAlgorithmSpec, ArenaGRPOSpec):
    """Specification for GRPO algorithm."""

    max_output_tokens: int | None = Field(default=1024)
    # Construction uses algo_utils dataclasses, not arena pydantic models.
    vllm_config: Any = Field(default=None)
    cosine_lr_schedule_config: Any = Field(default=None)

    @staticmethod
    def get_training_fn() -> Callable[..., Any]:
        """Get the training function for GRPO.

        :return: Training function
        :rtype: Callable[..., Any]
        """
        from agilerl.training.llm import (  # circular import with agilerl.training
            train_llm_rollout,
        )

        return train_llm_rollout

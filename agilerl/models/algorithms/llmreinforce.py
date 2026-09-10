# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""LLMREINFORCE algorithm specification."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from agilerl.arena.models.algorithms.llmreinforce import (
    LLMREINFORCESpec as ArenaLLMREINFORCESpec,
)
from agilerl.models.algo import LLMAlgorithmSpec, register


@register()
class LLMREINFORCESpec(LLMAlgorithmSpec, ArenaLLMREINFORCESpec):
    """Specification for LLMREINFORCE algorithm."""

    max_output_tokens: int | None = Field(default=1024)
    # Construction uses algo_utils dataclasses, not arena pydantic models.
    vllm_config: Any = Field(default=None)
    cosine_lr_schedule_config: Any = Field(default=None)

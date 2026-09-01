# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""LLM REINFORCE algorithm specification."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from agilerl.arena.models.algorithms.rollout_llm import RolloutLLMSpec
from agilerl.arena.models.descriptions import (
    BETA,
    CLIP_COEF,
    IS_LEVEL,
    LR,
    TEMPERATURE,
    TURN_RATIO_POOLING,
)
from agilerl.arena.models.registry import register


@register()
class LLMREINFORCESpec(RolloutLLMSpec):
    """REINFORCE with return batch normalization."""

    temperature: float = Field(default=1.0, description=TEMPERATURE)
    beta: float = Field(default=0.01, ge=0.0, le=1.0, description=BETA)
    max_grad_norm: float = Field(
        default=1.0,
        ge=0.0,
        description="Gradients are clipped to this global norm before each step.",
    )
    lr: float = Field(default=5e-7, ge=0.0, description=LR)
    clip_coef: float = Field(default=0.2, ge=0.0, le=1.0, description=CLIP_COEF)
    gamma: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Discount factor applied across turns.",
    )
    importance_sampling_level: Literal["token", "turn", "trajectory"] = Field(
        default="token", description=IS_LEVEL
    )
    turn_ratio_pooling: Literal["sum", "mean"] = Field(
        default="sum", description=TURN_RATIO_POOLING
    )

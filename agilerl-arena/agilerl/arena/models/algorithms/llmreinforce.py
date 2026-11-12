# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""LLM REINFORCE algorithm specification."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from agilerl.arena.models.algorithms.rollout_llm import RolloutLLMSpec
from agilerl.arena.models.descriptions import (
    ADVANTAGE_GRANULARITY_TOKEN,
    BETA,
    CLIP_COEF,
    IS_LEVEL,
    LR,
    OFF_POLICY_TOKEN_MASK,
    TEMPERATURE,
    TOP_P,
)
from agilerl.arena.models.registry import register


@register()
class LLMREINFORCESpec(RolloutLLMSpec):
    """REINFORCE with return batch normalization."""

    temperature: float = Field(default=1.0, gt=0.0, le=2.0, description=TEMPERATURE)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0, description=TOP_P)
    beta: float = Field(default=0.001, ge=0.0, le=1.0, description=BETA)
    max_grad_norm: float = Field(
        default=1.0,
        ge=0.0,
        description="Gradients are clipped to this global norm before each step.",
    )
    lr: float = Field(default=1e-5, gt=0.0, le=1.0, description=LR)
    clip_coef: float = Field(default=0.2, ge=0.0, le=1.0, description=CLIP_COEF)
    gamma: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Discount factor applied across turns.",
    )
    advantage_granularity: Literal["turn", "token", "auto"] = Field(
        default="auto", description=ADVANTAGE_GRANULARITY_TOKEN
    )
    importance_sampling_level: Literal["token", "turn", "trajectory"] = Field(
        default="token", description=IS_LEVEL
    )
    off_policy_token_mask_bounds: tuple[float, float] | None = Field(
        default=(0.5, 5.0), description=OFF_POLICY_TOKEN_MASK
    )

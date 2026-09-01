# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""LLM PPO algorithm specification."""

from __future__ import annotations

from typing import Literal

from pydantic import AliasChoices, Field

from agilerl.arena.models.algorithms.rollout_llm import RolloutLLMSpec
from agilerl.arena.models.descriptions import (
    BETA,
    CLIP_COEF,
    GAE_LAMBDA,
    IS_LEVEL,
    LR_ACTOR,
    LR_CRITIC,
    TEMPERATURE,
    TURN_RATIO_POOLING,
    VF_COEF,
    WHITEN_ADVANTAGES,
)
from agilerl.arena.models.registry import register


@register()
class LLMPPOSpec(RolloutLLMSpec):
    """PPO for LLM fine-tuning, with a value head."""

    temperature: float = Field(default=1.0, description=TEMPERATURE)
    beta: float = Field(default=0.01, ge=0.0, le=1.0, description=BETA)
    max_grad_norm: float = Field(
        default=1.0,
        ge=0.0,
        description="Gradients are clipped to this global norm before each step.",
    )
    lr_actor: float = Field(
        default=5e-7,
        ge=0.0,
        validation_alias=AliasChoices("lr_actor", "lr"),
        description=LR_ACTOR,
    )
    lr_critic: float | None = Field(default=5e-5, ge=0.0, description=LR_CRITIC)
    vf_coef: float = Field(default=0.5, ge=0.0, description=VF_COEF)
    clip_coef: float = Field(default=0.2, ge=0.0, le=1.0, description=CLIP_COEF)
    gamma: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Discount factor applied across turns.",
    )
    gae_lambda: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=GAE_LAMBDA,
    )
    turn_level_clip: bool = Field(
        default=True,
        description="Apply the surrogate clip per turn rather than per token.",
    )
    whiten_advantages: bool = Field(default=True, description=WHITEN_ADVANTAGES)
    importance_sampling_level: Literal["auto", "token", "turn", "trajectory"] = Field(
        default="auto", description=IS_LEVEL
    )
    turn_ratio_pooling: Literal["sum", "mean"] = Field(
        default="sum", description=TURN_RATIO_POOLING
    )
    turn_value_reduction: Literal["mean", "final_value"] = Field(
        default="final_value",
        description=(
            "How a turn's value estimate is reduced from its tokens: the mean, "
            "or the value at the turn's final token."
        ),
    )
    use_memory_efficient_params: bool = Field(
        default=True,
        description=(
            "For colocated vLLM, offload the trainer's parameters while the "
            "engine generates, trading a copy for headroom."
        ),
    )

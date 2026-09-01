# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""GRPO algorithm specification."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator
from typing_extensions import Self

from agilerl.arena.models.algorithms.rollout_llm import RolloutLLMSpec
from agilerl.arena.models.descriptions import (
    ADVANTAGE_GRANULARITY,
    GROUP_SIZE,
    IS_LEVEL,
    LR,
    WHITEN_ADVANTAGES,
)
from agilerl.arena.models.registry import register


@register()
class GRPOSpec(RolloutLLMSpec):
    """Group Relative Policy Optimization."""

    group_size: int = Field(..., ge=1, description=GROUP_SIZE)
    lr: float = Field(default=5e-7, ge=0.0, description=LR)
    clip_coef: float | list[float] = Field(
        default=0.2,
        description=(
            "Surrogate clipping range. A pair gives asymmetric bounds "
            "[epsilon_low, epsilon_high]."
        ),
    )

    @field_validator("clip_coef", mode="before")
    @classmethod
    def _coerce_clip_coef(cls, value: object) -> object:
        """A symmetric scalar, or an explicit ``[low, high]`` pair on the ratio."""
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if value < 0:
                msg = "clip_coef must be greater than or equal to zero."
                raise ValueError(msg)
            return float(value)
        if isinstance(value, (list, tuple)):
            if len(value) != 2:
                msg = "clip_coef list/tuple must contain exactly two values."
                raise ValueError(msg)
            return [float(value[0]), float(value[1])]
        msg = "clip_coef must be a float or a list/tuple of two floats."
        raise TypeError(msg)

    @model_validator(mode="after")
    def _validate_scalar_clip_coef(self) -> Self:
        if isinstance(self.clip_coef, float) and self.clip_coef > 1.0:
            msg = "GRPO clip_coef scalar must be <= 1.0."
            raise ValueError(msg)
        return self

    adv_norm: Literal["mean_only", "mean_std"] = Field(
        default="mean_std",
        description=(
            "How group advantages are normalized. 'mean_only' subtracts the "
            "group mean; 'mean_std' also divides by its standard deviation."
        ),
    )
    use_kl_advantage_shaping: bool = Field(
        default=False,
        description=(
            "Fold the KL penalty into the advantage instead of adding it as a "
            "separate loss term."
        ),
    )
    importance_sampling_level: Literal["token", "turn", "trajectory"] | None = Field(
        default=None, description=IS_LEVEL
    )
    advantage_granularity: Literal["auto", "trajectory", "turn"] = Field(
        default="auto", description=ADVANTAGE_GRANULARITY
    )
    whiten_advantages: bool = Field(default=False, description=WHITEN_ADVANTAGES)
    adv_clip_range: float | None = Field(
        default=None,
        description="Clip advantages to +/- this value. Unset leaves them unclipped.",
    )
    filter_zero_adv: bool = Field(
        default=False,
        description=(
            "Drop groups whose completions all scored the same. They contribute "
            "no gradient, so training on them only costs compute."
        ),
    )
    adv_filter_eps: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "With filter_zero_adv, also drop groups whose largest absolute "
            "advantage is below this threshold."
        ),
    )
    use_memory_efficient_params: bool = Field(
        default=True,
        description=(
            "For colocated vLLM, offload the trainer's parameters while the "
            "engine generates, trading a copy for headroom."
        ),
    )
    turn_advantage_trajectory_fallback: bool = Field(
        default=True,
        description=(
            "Fall back to a trajectory-level advantage when per-turn credit "
            "cannot be assigned."
        ),
    )
    loss_norm: Literal["micro_batch", "accumulation_window"] = Field(
        default="micro_batch",
        description=(
            "Token count the loss is averaged over: each micro-batch on its "
            "own, or the whole accumulation window. The window is the unbiased "
            "choice when micro-batches have uneven lengths."
        ),
    )

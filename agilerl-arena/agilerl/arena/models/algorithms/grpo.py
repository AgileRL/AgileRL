# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""GRPO algorithm specification."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field, field_validator, model_validator

from agilerl.arena.models.algo import LLMAlgorithmSpec, register
from agilerl.arena.models.env import LLMEnvType
from agilerl.arena.models.networks import CosineLRScheduleConfig, VLLMConfig


@register()
class GRPOSpec(LLMAlgorithmSpec):
    """Specification for GRPO algorithm."""

    group_size: int = Field(..., ge=1)
    lr: float = Field(default=0.0001, ge=0.0)
    clip_coef: float | tuple[float, float] = Field(default=0.2)
    temperature: float = Field(default=0.9)
    max_output_tokens: int | None = Field(default=None, ge=1)
    min_output_tokens: int | None = Field(default=None)
    cosine_lr_schedule_config: CosineLRScheduleConfig | None = Field(default=None)
    vllm_config: VLLMConfig | None = Field(default=None)
    use_vllm: bool = Field(default=False)

    env_type: ClassVar[LLMEnvType] = LLMEnvType.REASONING

    @field_validator("clip_coef", mode="before")
    @classmethod
    def _coerce_clip_coef(cls, value: object) -> float | tuple[float, float]:
        """Accept a symmetric scalar or an explicit (min, max) ratio pair."""
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        if isinstance(value, (list, tuple)):
            if len(value) != 2:
                msg = "clip_coef list/tuple must contain exactly two values."
                raise ValueError(msg)
            return (float(value[0]), float(value[1]))
        msg = "clip_coef must be a float or a list/tuple of two floats."
        raise ValueError(msg)

    @model_validator(mode="after")
    def _validate_clip_coef(self) -> GRPOSpec:
        if isinstance(self.clip_coef, float) and not 0.0 <= self.clip_coef <= 1.0:
            msg = "clip_coef scalar must be between 0.0 and 1.0."
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _validate_vllm_config(self) -> GRPOSpec:
        if self.use_vllm and not self.vllm_config:
            msg = "VLLM config is not set, please provide a VLLM config in the algorithm section of the manifest."
            raise ValueError(msg)
        return self

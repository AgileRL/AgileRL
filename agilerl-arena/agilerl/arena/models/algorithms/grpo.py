# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""GRPO algorithm specification."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field, model_validator

from agilerl.arena.models.algo import LLMAlgorithmSpec, register
from agilerl.arena.models.env import LLMEnvType
from agilerl.arena.models.networks import CosineLRScheduleConfig, VLLMConfig


@register()
class GRPOSpec(LLMAlgorithmSpec):
    """Specification for GRPO algorithm."""

    group_size: int = Field(..., ge=1)
    lr: float = Field(default=0.0001, ge=0.0)
    clip_coef: float = Field(default=0.2, ge=0.0, le=1.0)
    temperature: float = Field(default=0.9)
    max_output_tokens: int | None = Field(default=1024, exclude=True)
    min_output_tokens: int | None = Field(default=None)
    cosine_lr_schedule_config: CosineLRScheduleConfig | None = Field(default=None)
    vllm_config: VLLMConfig | None = Field(default=None)
    use_vllm: bool = Field(default=False)

    env_type: ClassVar[LLMEnvType] = LLMEnvType.ROLLOUT

    @model_validator(mode="after")
    def _validate_vllm_config(self) -> GRPOSpec:
        if self.use_vllm and not self.vllm_config:
            msg = "VLLM config is not set, please provide a VLLM config in the algorithm section of the manifest."
            raise ValueError(msg)
        return self

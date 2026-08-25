# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""LLMPPO algorithm specification."""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import Field, model_validator

from agilerl.arena.models.algo import LLMAlgorithmSpec, register
from agilerl.arena.models.env import LLMEnvType
from agilerl.arena.models.networks import CosineLRScheduleConfig, VLLMConfig


@register()
class LLMPPOSpec(LLMAlgorithmSpec):
    """Specification for LLMPPO algorithm."""

    lr_actor: float = Field(default=5e-6, ge=0.0)
    lr_critic: float | None = Field(default=5e-6, ge=0.0)
    vf_coef: float = Field(default=0.5, ge=0.0)
    clip_coef: float = Field(default=0.2, ge=0.0, le=1.0)
    gamma: float = Field(default=0.99, ge=0.0, le=1.0)
    gae_lambda: float = Field(default=0.95, ge=0.0, le=1.0)
    temperature: float = Field(default=0.9)
    max_output_tokens: int | None = Field(default=1024, exclude=True)
    min_output_tokens: int | None = Field(default=None)
    action_granularity: Literal["turn", "token", "auto"] = Field(default="auto")
    cosine_lr_schedule_config: CosineLRScheduleConfig | None = Field(default=None)
    vllm_config: VLLMConfig | None = Field(default=None)
    use_vllm: bool = Field(default=False)

    env_type: ClassVar[LLMEnvType] = LLMEnvType.ROLLOUT

    @model_validator(mode="after")
    def _validate_vllm_config(self) -> LLMPPOSpec:
        if self.use_vllm and not self.vllm_config:
            msg = "VLLM config is not set, please provide a VLLM config in the algorithm section of the manifest."
            raise ValueError(msg)
        return self

# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""GRPO algorithm specification."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import Field, model_validator

from agilerl.models.algo import LLMAlgorithmSpec, register
from agilerl.models.env_types import LLMEnvType

if TYPE_CHECKING:
    from agilerl.utils.algo_utils import CosineLRScheduleConfig, VLLMConfig
else:
    CosineLRScheduleConfig = Any
    VLLMConfig = Any


@register()
class GRPOSpec(LLMAlgorithmSpec):
    """Specification for GRPO algorithm."""

    group_size: int = Field(..., ge=1)
    lr: float = Field(default=0.0001, ge=0.0)
    clip_coef: float = Field(default=0.2, ge=0.0, le=1.0)
    temperature: float = Field(default=0.9)
    max_output_tokens: int | None = Field(default=1024)
    min_output_tokens: int | None = Field(default=None)
    cosine_lr_schedule_config: CosineLRScheduleConfig | None = Field(default=None)
    vllm_config: VLLMConfig | None = Field(default=None)
    use_vllm: bool = Field(default=False)
    adv_norm: str = Field(default="mean_std")
    importance_sampling_level: Literal["token", "turn", "trajectory"] | None = Field(
        default=None
    )
    advantage_granularity: Literal["auto", "trajectory", "turn"] = Field(default="auto")
    whiten_advantages: bool = Field(default=False)
    adv_clip_range: float | None = Field(default=None)
    filter_zero_adv: bool = Field(default=False)
    turn_advantage_trajectory_fallback: bool = Field(default=True)
    loss_norm: Literal["micro_batch", "accumulation_window"] = Field(
        default="micro_batch"
    )

    env_type: ClassVar[LLMEnvType] = LLMEnvType.REASONING

    @model_validator(mode="after")
    def _validate_vllm_config(self) -> GRPOSpec:
        if self.use_vllm and not self.vllm_config:
            msg = "VLLM config is not set, please provide a VLLM config in the algorithm section of the manifest."
            raise ValueError(msg)
        return self

    @staticmethod
    def get_training_fn(*, multiturn: bool = False) -> Callable[..., Any]:
        """Get the training function for GRPO.

        :param multiturn: If ``True``, return the multi-turn training
            function instead of the single-turn reasoning function.
        :type multiturn: bool
        :return: Training function
        :rtype: Callable[..., Any]
        """
        from agilerl.training.llm import (
            finetune_llm_multiturn,
            finetune_llm_reasoning,
        )

        return finetune_llm_multiturn if multiturn else finetune_llm_reasoning

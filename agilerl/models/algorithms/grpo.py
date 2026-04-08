"""GRPO algorithm specification."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

from pydantic import Field, model_validator

from agilerl.algorithms import GRPO
from agilerl.models.algo import LLMAlgorithmSpec, register
from agilerl.models.env import LLMEnvType
from agilerl.training.train_llm import finetune_llm_reasoning
from agilerl.utils.algo_utils import CosineLRScheduleConfig, VLLMConfig


@register(arena=True)
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

    algo_class: ClassVar[type[GRPO]] = GRPO

    env_type: ClassVar[LLMEnvType] = "reasoning"

    @model_validator(mode="after")
    def _validate_vllm_config(self):
        if self.use_vllm and not self.vllm_config:
            msg = "VLLM config is not set, please provide a VLLM config in the algorithm section of the manifest."
            raise ValueError(msg)
        return self

    @staticmethod
    def get_training_fn() -> Callable[..., Any]:
        """Get the training function for GRPO.

        :return: Training function
        :rtype: Callable[..., Any]
        """
        return finetune_llm_reasoning

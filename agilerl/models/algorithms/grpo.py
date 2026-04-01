"""GRPO algorithm specification."""

from collections.abc import Callable
from typing import Any, ClassVar

from pydantic import Field

from agilerl.algorithms import GRPO
from agilerl.models.algo import LLMAlgorithmSpec, register
from agilerl.models.env import LLMEnvType
from agilerl.training.train_llm import finetune_llm_reasoning


@register(arena=True)
class GRPOSpec(LLMAlgorithmSpec):
    """Specification for GRPO algorithm."""

    group_size: int = Field(..., ge=1)
    lr: float = Field(default=0.0001, ge=0.0)
    clip_coef: float = Field(default=0.2, ge=0.0, le=1.0)
    temperature: float = Field(default=0.9)

    algo_class: ClassVar[type[GRPO]] = GRPO
    env_type: ClassVar[LLMEnvType] = "reasoning"

    @staticmethod
    def get_training_fn() -> Callable[..., Any]:
        """Get the training function for GRPO.

        :return: Training function
        :rtype: Callable[..., Any]
        """
        return finetune_llm_reasoning

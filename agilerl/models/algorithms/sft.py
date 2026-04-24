"""SFT algorithm specification."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import Field

from agilerl.models.algo import LLMAlgorithmSpec, register

if TYPE_CHECKING:
    from agilerl.models.env import LLMEnvType


@register(arena=False)
class SFTSpec(LLMAlgorithmSpec):
    """Specification for SFT algorithm."""

    lr: float = Field(default=0.00005)

    env_type: ClassVar[LLMEnvType] = "sft"

    @staticmethod
    def get_training_fn() -> Callable[..., Any]:
        """Get the training function for SFT.

        :return: Training function
        :rtype: Callable[..., Any]
        """
        from agilerl.training.train_llm import finetune_llm_sft

        return finetune_llm_sft

"""SFT algorithm specification."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import Field

from agilerl.models.algo import LLMAlgorithmSpec, register

if TYPE_CHECKING:
    from agilerl.models.env import LLMEnvType


@register()
class SFTSpec(LLMAlgorithmSpec):
    """Specification for SFT algorithm."""

    lr: float = Field(default=0.00005)

    env_type: ClassVar[LLMEnvType] = "dataset"
    objective: ClassVar[str] = "sft"

    @staticmethod
    def get_training_fn() -> Callable[..., Any]:
        """Get the training function for SFT.

        :return: Training function
        :rtype: Callable[..., Any]
        """
        from agilerl.training.llm import train_llm_dataset

        return train_llm_dataset

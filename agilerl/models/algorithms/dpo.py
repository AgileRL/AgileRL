"""DPO algorithm specification."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import Field

from agilerl.models.algo import LLMAlgorithmSpec, register

if TYPE_CHECKING:
    from agilerl.models.env import LLMEnvType


@register()
class DPOSpec(LLMAlgorithmSpec):
    """Specification for DPO algorithm."""

    lr: float = Field(default=0.000005)

    env_type: ClassVar[LLMEnvType] = "dataset"
    objective: ClassVar[str] = "preference"

    @staticmethod
    def get_training_fn() -> Callable[..., Any]:
        """Get the training function for DPO.

        :return: Training function
        :rtype: Callable[..., Any]
        """
        from agilerl.training.train_llm import train_llm_dataset

        return train_llm_dataset

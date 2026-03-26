"""DPO algorithm specification."""

from collections.abc import Callable
from typing import Any, ClassVar

from pydantic import Field

from agilerl.algorithms import DPO
from agilerl.models.algo import LLMAlgorithmSpec, register
from agilerl.training.train_llm import finetune_llm_preference


@register(arena=True)
class DPOSpec(LLMAlgorithmSpec):
    """Specification for DPO algorithm."""

    lr: float = Field(default=0.000005)

    algo_class: ClassVar[type[DPO]] = DPO

    @staticmethod
    def get_training_fn() -> Callable[..., Any]:
        """Get the training function for DPO.

        :return: Training function
        :rtype: Callable[..., Any]
        """
        return finetune_llm_preference

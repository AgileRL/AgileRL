"""DPO algorithm specification."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

from pydantic import Field

from agilerl.models.algo import LLMAlgorithmSpec, register
from agilerl.models.env_types import LLMEnvType


@register()
class DPOSpec(LLMAlgorithmSpec):
    """Specification for DPO algorithm."""

    lr: float = Field(default=0.000005)

    env_type: ClassVar[LLMEnvType] = LLMEnvType.PREFERENCE

    @staticmethod
    def get_training_fn() -> Callable[..., Any]:
        """Get the training function for DPO.

        :return: Training function
        :rtype: Callable[..., Any]
        """
        from agilerl.training.llm import finetune_llm_preference

        return finetune_llm_preference

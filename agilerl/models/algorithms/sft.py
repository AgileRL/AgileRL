"""SFT algorithm specification."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

from pydantic import Field

from agilerl.models.algo import LLMAlgorithmSpec, register
from agilerl.models.env_types import LLMEnvType


@register()
class SFTSpec(LLMAlgorithmSpec):
    """Specification for SFT algorithm."""

    lr: float = Field(default=0.00005)

    env_type: ClassVar[LLMEnvType] = LLMEnvType.SFT

    @staticmethod
    def get_training_fn() -> Callable[..., Any]:
        """Get the training function for SFT.

        :return: Training function
        :rtype: Callable[..., Any]
        """
        from agilerl.training.llm import finetune_llm_sft

        return finetune_llm_sft

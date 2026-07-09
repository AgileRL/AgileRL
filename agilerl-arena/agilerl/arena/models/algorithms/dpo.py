"""DPO algorithm specification."""

from __future__ import annotations

from typing import ClassVar

from agilerl.arena.models.algo import LLMAlgorithmSpec, register
from agilerl.arena.models.env import LLMEnvType
from pydantic import Field


@register()
class DPOSpec(LLMAlgorithmSpec):
    """Specification for DPO algorithm."""

    lr: float = Field(default=0.000005)
    env_type: ClassVar[LLMEnvType] = "preference"

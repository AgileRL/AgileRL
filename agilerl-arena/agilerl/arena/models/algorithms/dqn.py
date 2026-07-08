"""DQN algorithm specification."""

from __future__ import annotations

from pydantic import Field

from agilerl.arena.models.algo import RLAlgorithmSpec, register
from agilerl.arena.models.networks import QNetworkSpec


@register()
class DQNSpec(RLAlgorithmSpec):
    """Specification for DQN algorithm."""

    tau: float = Field(default=0.001)
    double: bool = Field(default=False)
    lr: float = Field(default=0.0001, ge=0.0)
    net_config: QNetworkSpec | None = Field(default=None)

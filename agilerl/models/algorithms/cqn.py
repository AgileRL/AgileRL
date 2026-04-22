"""CQN algorithm specification."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from pydantic import Field

from agilerl.models.algo import RLAlgorithmSpec, offline, register
from agilerl.models.networks import QNetworkSpec

if TYPE_CHECKING:
    from agilerl.modules.base import EvolvableModule
else:
    EvolvableModule = Any


@register(arena=False)
@offline()
class CQNSpec(RLAlgorithmSpec):
    """Specification for CQN algorithm."""

    tau: float = Field(default=0.001)
    double: bool = Field(default=False)
    lr: float = Field(default=0.0001, ge=0.0)
    net_config: QNetworkSpec | None = Field(default=None)
    actor_network: EvolvableModule | None = Field(default=None)

    @staticmethod
    def get_training_fn() -> Callable[..., Any]:
        """Get the training function for CQN.

        :return: Training function
        :rtype: Callable[..., Any]
        """
        from agilerl.training.train_offline import train_offline

        return train_offline

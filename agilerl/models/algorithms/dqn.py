"""DQN algorithm specification."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from pydantic import Field

from agilerl.models.algo import RLAlgorithmSpec, off_policy, register
from agilerl.models.networks import QNetworkSpec

if TYPE_CHECKING:
    from agilerl.modules import EvolvableModule
else:
    EvolvableModule = Any


@register(arena=True)
@off_policy()
class DQNSpec(RLAlgorithmSpec):
    """Specification for DQN algorithm."""

    tau: float = Field(default=0.001)
    double: bool = Field(default=False)
    lr: float = Field(default=0.0001, ge=0.0)
    cudagraphs: bool = Field(default=False)
    actor_network: EvolvableModule | None = Field(default=None)
    net_config: QNetworkSpec | None = Field(default=None)

    @staticmethod
    def get_training_fn() -> Callable[..., Any]:
        """Get the training function for DQN.

        :return: Training function
        :rtype: Callable[..., Any]
        """
        from agilerl.training.train_off_policy import train_off_policy

        return train_off_policy

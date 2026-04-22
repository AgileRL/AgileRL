"""NeuralUCB algorithm specification."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from pydantic import Field

from agilerl.models.algo import RLAlgorithmSpec, bandit, register
from agilerl.models.networks import QNetworkSpec

if TYPE_CHECKING:
    from agilerl.modules import EvolvableModule
else:
    EvolvableModule = Any


@register(arena=False)
@bandit()
class NeuralUCBSpec(RLAlgorithmSpec):
    """Specification for NeuralUCB (Neural Upper Confidence Bound) algorithm."""

    gamma: float = Field(default=1.0, ge=0.0)
    lamb: float = Field(default=1.0)
    reg: float = Field(default=0.000625)
    lr: float = Field(default=0.001, ge=0.0)
    learn_step: int = Field(default=2, ge=1)
    net_config: QNetworkSpec | None = Field(default=None)
    actor_network: EvolvableModule | None = Field(default=None)

    @staticmethod
    def get_training_fn() -> Callable[..., Any]:
        """Get the training function for NeuralUCB.

        :return: Training function
        :rtype: Callable[..., Any]
        """
        from agilerl.training.train_bandits import train_bandits

        return train_bandits

"""NeuralTS algorithm specification."""

from collections.abc import Callable
from typing import Any, ClassVar

from pydantic import Field

from agilerl.algorithms import NeuralTS
from agilerl.models.algo import RLAlgorithmSpec, bandit, register
from agilerl.models.networks import QNetworkSpec
from agilerl.modules import EvolvableModule
from agilerl.training.train_bandits import train_bandits


@register(arena=False)
@bandit()
class NeuralTSSpec(RLAlgorithmSpec):
    """Specification for NeuralTS (Neural Thompson Sampling) algorithm."""

    gamma: float = Field(default=1.0, ge=0.0)
    lamb: float = Field(default=1.0)
    reg: float = Field(default=0.000625)
    lr: float = Field(default=0.003, ge=0.0)
    learn_step: int = Field(default=2, ge=1)
    net_config: QNetworkSpec | None = Field(default=None)
    actor_network: EvolvableModule | None = Field(default=None, exclude=True)

    algo_class: ClassVar[type[NeuralTS]] = NeuralTS

    @staticmethod
    def get_training_fn() -> Callable[..., Any]:
        """Get the training function for NeuralTS.

        :return: Training function
        :rtype: Callable[..., Any]
        """
        return train_bandits

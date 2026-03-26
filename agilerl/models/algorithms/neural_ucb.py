"""NeuralUCB algorithm specification."""

from collections.abc import Callable
from typing import Any, ClassVar

from pydantic import Field

from agilerl.algorithms import NeuralUCB
from agilerl.models.algo import RLAlgorithmSpec, register
from agilerl.training.train_bandits import train_bandits


@register(arena=False)
class NeuralUCBSpec(RLAlgorithmSpec):
    """Specification for NeuralUCB (Neural Upper Confidence Bound) algorithm."""

    gamma: float = Field(default=1.0, ge=0.0)
    lamb: float = Field(default=1.0)
    reg: float = Field(default=0.000625)
    lr: float = Field(default=0.001, ge=0.0)
    learn_step: int = Field(default=2, ge=1)

    algo_class: ClassVar[type[NeuralUCB]] = NeuralUCB

    @staticmethod
    def get_training_fn() -> Callable[..., Any]:
        """Get the training function for NeuralUCB.

        :return: Training function
        :rtype: Callable[..., Any]
        """
        return train_bandits

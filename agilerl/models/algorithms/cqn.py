"""CQN algorithm specification."""

from collections.abc import Callable
from typing import Any, ClassVar

from pydantic import Field

from agilerl.algorithms import CQN
from agilerl.models.algo import RLAlgorithmSpec, offline, register
from agilerl.training.train_offline import train_offline


@register(arena=False)
@offline()
class CQNSpec(RLAlgorithmSpec):
    """Specification for CQN algorithm."""

    tau: float = Field(default=0.001)
    double: bool = Field(default=False)
    lr: float = Field(default=0.0001, ge=0.0)

    algo_class: ClassVar[type[CQN]] = CQN

    @staticmethod
    def get_training_fn() -> Callable[..., Any]:
        """Get the training function for CQN.

        :return: Training function
        :rtype: Callable[..., Any]
        """
        return train_offline

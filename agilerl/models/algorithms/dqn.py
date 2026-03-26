"""DQN algorithm specification."""

from collections.abc import Callable
from typing import Any, ClassVar

from pydantic import Field

from agilerl.algorithms import DQN
from agilerl.models.algo import RLAlgorithmSpec, off_policy, register
from agilerl.training.train_off_policy import train_off_policy


@register(arena=True)
@off_policy()
class DQNSpec(RLAlgorithmSpec):
    """Specification for DQN algorithm."""

    tau: float = Field(default=0.001)
    double: bool = Field(default=False)
    lr: float = Field(default=0.0001, ge=0.0)
    cudagraphs: bool = Field(default=False)

    algo_class: ClassVar[type[DQN]] = DQN

    @staticmethod
    def get_training_loop() -> Callable[..., Any]:
        """Get the training loop for DQN.

        :return: Training loop
        :rtype: Callable[..., Any]
        """
        return train_off_policy

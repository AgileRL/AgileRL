"""DQN algorithm specification."""

from collections.abc import Callable
from typing import Any, ClassVar

from pydantic import Field

from agilerl.algorithms import DQN
from agilerl.models.algo import RLAlgorithmSpec, off_policy, register
from agilerl.models.networks import QNetworkSpec
from agilerl.training.train_off_policy import train_off_policy


@register(arena=True)
@off_policy()
class DQNSpec(RLAlgorithmSpec):
    """Specification for DQN algorithm."""

    tau: float = Field(default=0.001)
    double: bool = Field(default=False)
    lr: float = Field(default=0.0001, ge=0.0)
    cudagraphs: bool = Field(default=False)
    net_config: QNetworkSpec | None = Field(default=None)

    algo_class: ClassVar[type[DQN]] = DQN

    @staticmethod
    def get_training_fn() -> Callable[..., Any]:
        """Get the training function for DQN.

        :return: Training function
        :rtype: Callable[..., Any]
        """
        return train_off_policy

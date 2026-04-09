"""TD3 algorithm specification."""

from collections.abc import Callable
from typing import Any, ClassVar

from pydantic import Field

from agilerl.algorithms import TD3
from agilerl.models.algo import RLAlgorithmSpec, off_policy, register
from agilerl.models.networks import DeterministicActorSpec
from agilerl.modules import EvolvableModule
from agilerl.training.train_off_policy import train_off_policy


@register(arena=True)
@off_policy()
class TD3Spec(RLAlgorithmSpec):
    """Specification for TD3 algorithm."""

    vect_noise_dim: int = Field(default=1, ge=1)
    lr_actor: float = Field(default=0.0001, ge=0.0)
    lr_critic: float = Field(default=0.001, ge=0.0)
    tau: float = Field(default=0.005)
    policy_freq: int = Field(default=2, ge=1)
    O_U_noise: bool = Field(default=True)
    expl_noise: float = Field(default=0.1)
    mean_noise: float = Field(default=0.0)
    theta: float = Field(default=0.15)
    dt: float = Field(default=0.01)
    share_encoders: bool = Field(default=False)
    net_config: DeterministicActorSpec | None = Field(default=None)
    actor_network: EvolvableModule | None = Field(default=None)
    critic_networks: list[EvolvableModule] | None = Field(default=None)

    algo_class: ClassVar[type[TD3]] = TD3

    @staticmethod
    def get_training_fn() -> Callable[..., Any]:
        """Get the training function for TD3.

        :return: Training function
        :rtype: Callable[..., Any]
        """
        return train_off_policy

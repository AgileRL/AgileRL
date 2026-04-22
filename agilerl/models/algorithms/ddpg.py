"""DDPG algorithm specification."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from pydantic import Field

from agilerl.models.algo import RLAlgorithmSpec, off_policy, register
from agilerl.models.networks import DeterministicActorSpec

if TYPE_CHECKING:
    from agilerl.modules import EvolvableModule
else:
    EvolvableModule = Any


@register(arena=True)
@off_policy()
class DDPGSpec(RLAlgorithmSpec):
    """Specification for DDPG algorithm."""

    vect_noise_dim: int = Field(default=1, ge=1)
    lr_actor: float = Field(default=0.0001, ge=0.0)
    lr_critic: float = Field(default=0.001, ge=0.0)
    tau: float = Field(default=0.001, ge=0.0, le=1.0)
    policy_freq: int = Field(default=2, ge=1)
    O_U_noise: bool = Field(default=True)
    expl_noise: float = Field(default=0.1)
    mean_noise: float = Field(default=0.0)
    theta: float = Field(default=0.15)
    dt: float = Field(default=0.01)
    actor_network: EvolvableModule | None = Field(default=None)
    critic_network: EvolvableModule | None = Field(default=None)
    share_encoders: bool = Field(default=False)
    net_config: DeterministicActorSpec | None = Field(default=None)

    @staticmethod
    def get_training_fn() -> Callable[..., Any]:
        """Get the training function for DDPG.

        :return: Training function
        :rtype: Callable[..., Any]
        """
        from agilerl.training.train_off_policy import train_off_policy

        return train_off_policy

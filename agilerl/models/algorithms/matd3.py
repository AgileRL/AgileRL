"""MATD3 algorithm specification."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from pydantic import Field

from agilerl.models.algo import MultiAgentRLAlgorithmSpec, off_policy, register
from agilerl.models.networks import DeterministicActorSpec

if TYPE_CHECKING:
    from agilerl.modules import ModuleDict
else:
    ModuleDict = Any


@register(arena=True)
@off_policy()
class MATD3Spec(MultiAgentRLAlgorithmSpec):
    """Specification for MATD3 algorithm."""

    vect_noise_dim: int = Field(default=1, ge=1)
    lr_actor: float = Field(default=0.001, ge=0.0)
    lr_critic: float = Field(default=0.01, ge=0.0)
    tau: float = Field(default=0.015, ge=0.0, le=1.0)
    O_U_noise: bool = Field(default=True)
    expl_noise: float = Field(default=0.1)
    policy_freq: int = Field(default=2, ge=1)
    mean_noise: float = Field(default=0.0)
    theta: float = Field(default=0.15)
    dt: float = Field(default=0.01)
    torch_compiler: str | None = Field(default=None)
    net_config: DeterministicActorSpec | None = Field(default=None)
    actor_networks: ModuleDict | None = Field(default=None)
    critic_networks: list[ModuleDict] | None = Field(default=None)

    @staticmethod
    def get_training_fn() -> Callable[..., Any]:
        """Get the training function for MATD3.

        :return: Training function
        :rtype: Callable[..., Any]
        """
        from agilerl.training.train_multi_agent_off_policy import (
            train_multi_agent_off_policy,
        )

        return train_multi_agent_off_policy

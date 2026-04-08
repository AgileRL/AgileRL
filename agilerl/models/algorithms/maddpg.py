"""MADDPG algorithm specification."""

from collections.abc import Callable
from typing import Any, ClassVar

from pydantic import Field

from agilerl.algorithms import MADDPG
from agilerl.models.algo import MultiAgentRLAlgorithmSpec, off_policy, register
from agilerl.models.networks import DeterministicActorSpec
from agilerl.modules import ModuleDict
from agilerl.training.train_multi_agent_off_policy import train_multi_agent_off_policy


@register(arena=True)
@off_policy()
class MADDPGSpec(MultiAgentRLAlgorithmSpec):
    """Specification for MADDPG algorithm."""

    vect_noise_dim: int = Field(default=1, ge=1)
    lr_actor: float = Field(default=0.001, ge=0.0)
    lr_critic: float = Field(default=0.01, ge=0.0)
    tau: float = Field(default=0.01, ge=0.0, le=1.0)
    O_U_noise: bool = Field(default=True)
    expl_noise: float = Field(default=0.1)
    mean_noise: float = Field(default=0.0)
    theta: float = Field(default=0.15)
    dt: float = Field(default=0.01)
    torch_compiler: str | None = Field(default=None)
    net_config: DeterministicActorSpec | None = Field(default=None)
    actor_networks: ModuleDict | None = Field(default=None)
    critic_networks: ModuleDict | None = Field(default=None)

    algo_class: ClassVar[type[MADDPG]] = MADDPG

    @staticmethod
    def get_training_fn() -> Callable[..., Any]:
        """Get the training function for MADDPG.

        :return: Training function
        :rtype: Callable[..., Any]
        """
        return train_multi_agent_off_policy

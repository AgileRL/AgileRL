"""Rainbow DQN algorithm specification."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Self

from pydantic import Field, model_validator

from agilerl.models.algo import RLAlgorithmSpec, off_policy, register
from agilerl.models.networks import RainbowQNetworkSpec

if TYPE_CHECKING:
    from agilerl.modules import EvolvableModule
else:
    EvolvableModule = Any


@register(arena=True)
@off_policy()
class RainbowDQNSpec(RLAlgorithmSpec):
    """Specification for Rainbow DQN algorithm."""

    tau: float = Field(default=0.001)
    beta: float = Field(default=0.4)
    prior_eps: float = Field(default=1e-6)
    num_atoms: int = Field(default=51, ge=1)
    v_min: float = Field(default=-200)
    v_max: float = Field(default=200)
    noise_std: float = Field(default=0.5)
    n_step: int = Field(default=3, ge=1)
    combined_reward: bool = Field(default=False)
    lr: float = Field(default=0.0001, ge=0.0)
    net_config: RainbowQNetworkSpec | None = Field(default=None)
    actor_network: EvolvableModule | None = Field(default=None)

    @model_validator(mode="after")
    def _check_v_range(self) -> Self:
        if self.v_min >= self.v_max:
            msg = "v_min must be less than v_max."
            raise ValueError(msg)
        return self

    @staticmethod
    def get_training_fn() -> Callable[..., Any]:
        """Get the training function for Rainbow DQN.

        :return: Training function
        :rtype: Callable[..., Any]
        """
        from agilerl.training.train_off_policy import train_off_policy

        return train_off_policy

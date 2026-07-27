"""Rainbow DQN algorithm specification."""

from __future__ import annotations

from pydantic import Field, model_validator
from typing_extensions import Self

from agilerl.arena.models.algo import RLAlgorithmSpec, register
from agilerl.arena.models.networks import RainbowQNetworkSpec


@register()
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

    @model_validator(mode="after")
    def _check_v_range(self) -> Self:
        if self.v_min >= self.v_max:
            msg = "v_min must be less than v_max."
            raise ValueError(msg)
        return self

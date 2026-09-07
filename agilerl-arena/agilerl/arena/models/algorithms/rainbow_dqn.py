# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Rainbow DQN algorithm specification."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field, model_validator
from typing_extensions import Self

from agilerl.arena.models.algorithms.base import SingleAgentAlgorithmSpec
from agilerl.arena.models.descriptions import LR, NET_CONFIG, TAU
from agilerl.arena.models.networks import RainbowQNetworkSpec
from agilerl.arena.models.registry import register


@register("Rainbow DQN")
@register()
class RainbowDQNSpec(SingleAgentAlgorithmSpec):
    """Rainbow DQN."""

    tau: float = Field(default=0.001, description=TAU)
    beta: float = Field(
        default=0.4,
        description=(
            "Importance-sampling exponent correcting the bias prioritized "
            "replay introduces. 1.0 corrects it fully."
        ),
    )
    prior_eps: float = Field(
        default=1e-6,
        description=(
            "Floor added to each priority so a transition with zero TD error is "
            "still sampled occasionally."
        ),
    )
    num_atoms: int = Field(
        default=51,
        ge=1,
        description=(
            "Support points in the distributional value estimate. 1 collapses "
            "to a scalar Q-value."
        ),
    )
    v_min: float = Field(
        default=0,
        description="Lowest return the distributional value support covers.",
    )
    v_max: float = Field(
        default=200,
        description="Highest return the distributional value support covers.",
    )
    noise_std: float = Field(
        default=0.5,
        description=(
            "Initial standard deviation of the NoisyNet parameters that drive "
            "exploration in place of epsilon-greedy."
        ),
    )
    n_step: int = Field(
        default=3,
        ge=1,
        description="Steps of reward accumulated into each n-step return.",
    )
    combined_reward: bool = Field(
        default=False,
        description="Learn from both the one-step and the n-step return.",
    )
    lr: float = Field(default=0.0001, ge=0.0, description=LR)
    net_config: RainbowQNetworkSpec | None = Field(default=None, description=NET_CONFIG)

    off_policy: ClassVar[bool] = True
    supports_per_buffer: ClassVar[bool] = True

    @model_validator(mode="after")
    def _check_v_range(self) -> Self:
        if self.v_min >= self.v_max:
            msg = "v_min must be less than v_max."
            raise ValueError(msg)
        return self

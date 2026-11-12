# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""TD3 algorithm specification."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field, model_validator
from typing_extensions import Self

from agilerl.arena.models.algorithms.base import (
    SingleAgentAlgorithmSpec,
    reject_unstable_ou_noise,
)
from agilerl.arena.models.descriptions import (
    DT,
    EXPL_NOISE,
    LR_ACTOR,
    LR_CRITIC,
    MEAN_NOISE,
    NET_CONFIG,
    OU_NOISE,
    POLICY_FREQ,
    SHARE_ENCODERS,
    TAU,
    THETA,
    VECT_NOISE_DIM,
)
from agilerl.arena.models.networks import DeterministicActorSpec
from agilerl.arena.models.registry import register


@register()
class TD3Spec(SingleAgentAlgorithmSpec):
    """Twin Delayed DDPG."""

    vect_noise_dim: int = Field(default=1, ge=1, description=VECT_NOISE_DIM)
    lr_actor: float = Field(default=0.0001, gt=0.0, le=1.0, description=LR_ACTOR)
    lr_critic: float = Field(default=0.001, gt=0.0, le=1.0, description=LR_CRITIC)
    tau: float = Field(default=0.005, gt=0.0, le=1.0, description=TAU)
    policy_freq: int = Field(default=2, ge=1, le=1000, description=POLICY_FREQ)
    O_U_noise: bool = Field(default=True, description=OU_NOISE)
    expl_noise: float = Field(default=0.1, ge=0.0, le=5.0, description=EXPL_NOISE)
    mean_noise: float = Field(default=0.0, ge=-5.0, le=5.0, description=MEAN_NOISE)
    theta: float = Field(default=0.15, ge=0.0, le=10.0, description=THETA)
    dt: float = Field(default=0.01, gt=0.0, le=1.0, description=DT)
    share_encoders: bool = Field(default=False, description=SHARE_ENCODERS)
    net_config: DeterministicActorSpec | None = Field(
        default=None, description=NET_CONFIG
    )

    off_policy: ClassVar[bool] = True

    @model_validator(mode="after")
    def _ou_stable(self) -> Self:
        """The discrete OU step multiplies the deviation by (1 - theta * dt)."""
        reject_unstable_ou_noise(self.theta, self.dt, self.O_U_noise)
        return self

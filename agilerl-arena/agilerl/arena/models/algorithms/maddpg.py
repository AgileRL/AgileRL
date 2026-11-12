# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""MADDPG algorithm specification."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field, model_validator
from typing_extensions import Self

from agilerl.arena.models.algorithms.base import (
    MultiAgentAlgorithmSpec,
    reject_unstable_ou_noise,
)
from agilerl.arena.models.descriptions import (
    DT,
    EXPL_NOISE,
    LR_ACTOR,
    LR_CRITIC,
    MEAN_NOISE,
    NET_CONFIG_PER_AGENT,
    OU_NOISE,
    TAU,
    THETA,
    VECT_NOISE_DIM,
)
from agilerl.arena.models.networks import DeterministicActorSpec
from agilerl.arena.models.registry import register


@register()
class MADDPGSpec(MultiAgentAlgorithmSpec):
    """Multi-Agent DDPG."""

    vect_noise_dim: int = Field(default=1, ge=1, description=VECT_NOISE_DIM)
    lr_actor: float = Field(default=0.001, gt=0.0, le=1.0, description=LR_ACTOR)
    lr_critic: float = Field(default=0.01, gt=0.0, le=1.0, description=LR_CRITIC)
    # Off-policy multi-agent: the constructors learn every few steps with a
    # shorter horizon than IPPO's rollout-sized learn_step and 0.99.
    learn_step: int = Field(
        default=5,
        ge=1,
        description="Environment steps collected between learn steps.",
    )
    gamma: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Discount factor on future reward.",
    )
    tau: float = Field(default=0.01, gt=0.0, le=1.0, description=TAU)
    O_U_noise: bool = Field(default=True, description=OU_NOISE)
    expl_noise: float = Field(default=0.1, ge=0.0, le=5.0, description=EXPL_NOISE)
    mean_noise: float = Field(default=0.0, ge=-5.0, le=5.0, description=MEAN_NOISE)
    theta: float = Field(default=0.15, ge=0.0, le=10.0, description=THETA)
    dt: float = Field(default=0.01, gt=0.0, le=1.0, description=DT)
    net_config: DeterministicActorSpec | dict[str, DeterministicActorSpec] | None = (
        Field(default=None, description=NET_CONFIG_PER_AGENT)
    )

    off_policy: ClassVar[bool] = True

    @model_validator(mode="after")
    def _ou_stable(self) -> Self:
        """The discrete OU step multiplies the deviation by (1 - theta * dt)."""
        reject_unstable_ou_noise(self.theta, self.dt, self.O_U_noise)
        return self

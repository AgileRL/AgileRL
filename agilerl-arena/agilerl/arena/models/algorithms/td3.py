# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""TD3 algorithm specification."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from agilerl.arena.models.algorithms.base import SingleAgentAlgorithmSpec
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
    lr_actor: float = Field(default=0.0001, ge=0.0, description=LR_ACTOR)
    lr_critic: float = Field(default=0.001, ge=0.0, description=LR_CRITIC)
    tau: float = Field(default=0.005, description=TAU)
    policy_freq: int = Field(default=2, ge=1, description=POLICY_FREQ)
    O_U_noise: bool = Field(default=True, description=OU_NOISE)
    expl_noise: float = Field(default=0.1, description=EXPL_NOISE)
    mean_noise: float = Field(default=0.0, description=MEAN_NOISE)
    theta: float = Field(default=0.15, description=THETA)
    dt: float = Field(default=0.01, description=DT)
    share_encoders: bool = Field(default=False, description=SHARE_ENCODERS)
    net_config: DeterministicActorSpec | None = Field(
        default=None, description=NET_CONFIG
    )

    off_policy: ClassVar[bool] = True

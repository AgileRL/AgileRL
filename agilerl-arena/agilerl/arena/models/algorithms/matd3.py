# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""MATD3 algorithm specification."""

from __future__ import annotations

from pydantic import Field

from agilerl.arena.models.algo import MultiAgentRLAlgorithmSpec, register
from agilerl.arena.models.networks import DeterministicActorSpec


@register()
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

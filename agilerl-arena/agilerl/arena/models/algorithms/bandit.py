# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Shared bandit algorithm specification."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from agilerl.arena.models.algorithms.base import SingleAgentAlgorithmSpec
from agilerl.arena.models.descriptions import NET_CONFIG
from agilerl.arena.models.networks import QNetworkSpec
from agilerl.arena.models.registry import AgentType


class BanditSpec(SingleAgentAlgorithmSpec):
    """Shared surface for the neural contextual bandits."""

    gamma: float = Field(
        default=1.0,
        ge=0.0,
        description="Exploration scale on the predicted uncertainty of each arm.",
    )
    lamb: float = Field(
        default=1.0,
        description="Ridge parameter of the covariance estimate over arm features.",
    )
    reg: float = Field(
        default=0.000625,
        description="L2 regularization on the network weights.",
    )
    learn_step: int = Field(
        default=2, ge=1, description="Arm pulls between learn steps."
    )
    net_config: QNetworkSpec | None = Field(default=None, description=NET_CONFIG)

    bandit: ClassVar[bool] = True
    cluster_supported: ClassVar[bool] = False
    agent_type: ClassVar[AgentType] = AgentType.BanditAgent
    env_type: ClassVar[str] = "bandit"
    default_evo_steps: ClassVar[int] = 500

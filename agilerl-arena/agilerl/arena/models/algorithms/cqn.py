# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""CQN algorithm specification."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from agilerl.arena.models.algorithms.base import SingleAgentAlgorithmSpec
from agilerl.arena.models.descriptions import DOUBLE, LR, NET_CONFIG, TAU
from agilerl.arena.models.networks import QNetworkSpec
from agilerl.arena.models.registry import AgentType, register


@register()
class CQNSpec(SingleAgentAlgorithmSpec):
    """Conservative Q-Learning, trained from a fixed dataset."""

    tau: float = Field(default=0.001, description=TAU)
    double: bool = Field(default=False, description=DOUBLE)
    lr: float = Field(default=0.0001, ge=0.0, description=LR)
    net_config: QNetworkSpec | None = Field(default=None, description=NET_CONFIG)

    offline: ClassVar[bool] = True
    cluster_supported: ClassVar[bool] = False
    agent_type: ClassVar[AgentType] = AgentType.OfflineAgent
    env_type: ClassVar[str] = "offline"
    default_evo_steps: ClassVar[int] = 5_000

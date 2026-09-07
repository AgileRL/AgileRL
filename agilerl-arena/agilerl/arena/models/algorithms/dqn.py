# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""DQN algorithm specification."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from agilerl.arena.models.algorithms.base import SingleAgentAlgorithmSpec
from agilerl.arena.models.descriptions import DOUBLE, LR, NET_CONFIG, TAU
from agilerl.arena.models.networks import QNetworkSpec
from agilerl.arena.models.registry import register


@register()
class DQNSpec(SingleAgentAlgorithmSpec):
    """Deep Q-Network."""

    tau: float = Field(default=0.001, description=TAU)
    double: bool = Field(default=False, description=DOUBLE)
    lr: float = Field(default=0.0001, ge=0.0, description=LR)
    cudagraphs: bool = Field(
        default=False,
        description=(
            "Capture the update step as a CUDA graph. Cuts launch overhead for "
            "small networks; requires static shapes."
        ),
    )
    net_config: QNetworkSpec | None = Field(default=None, description=NET_CONFIG)

    off_policy: ClassVar[bool] = True

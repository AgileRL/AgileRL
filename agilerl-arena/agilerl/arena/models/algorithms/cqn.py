# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""CQN algorithm specification."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from agilerl.arena.models.algo import RLAlgorithmSpec, register
from agilerl.arena.models.networks import QNetworkSpec


@register()
class CQNSpec(RLAlgorithmSpec):
    """Specification for CQN algorithm."""

    tau: float = Field(default=0.001)
    double: bool = Field(default=False)
    lr: float = Field(default=0.0001, ge=0.0)
    net_config: QNetworkSpec | None = Field(default=None)

    default_evo_steps: ClassVar[int] = 5_000

# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""NeuralUCB algorithm specification."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from agilerl.arena.models.algo import RLAlgorithmSpec, register
from agilerl.arena.models.networks import QNetworkSpec


@register()
class NeuralUCBSpec(RLAlgorithmSpec):
    """Specification for NeuralUCB (Neural Upper Confidence Bound) algorithm."""

    gamma: float = Field(default=1.0, ge=0.0)
    lamb: float = Field(default=1.0)
    reg: float = Field(default=0.000625)
    lr: float = Field(default=0.001, ge=0.0)
    learn_step: int = Field(default=2, ge=1)
    net_config: QNetworkSpec | None = Field(default=None)

    default_evo_steps: ClassVar[int] = 500

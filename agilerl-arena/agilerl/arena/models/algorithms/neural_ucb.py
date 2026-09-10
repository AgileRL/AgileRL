# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""NeuralUCB algorithm specification."""

from __future__ import annotations

from pydantic import Field

from agilerl.arena.models.algorithms.bandit import BanditSpec
from agilerl.arena.models.descriptions import LR
from agilerl.arena.models.registry import register


@register()
class NeuralUCBSpec(BanditSpec):
    """Neural Upper Confidence Bound."""

    lr: float = Field(default=0.001, ge=0.0, description=LR)

# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""NeuralTS algorithm specification."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import Field

from agilerl.arena.models.algorithms.neural_ts import NeuralTSSpec as ArenaNeuralTSSpec
from agilerl.models.algo import RLAlgorithmSpec, bandit, register

if TYPE_CHECKING:
    from agilerl.modules import EvolvableModule
else:
    EvolvableModule = Any


@register()
@bandit()
class NeuralTSSpec(RLAlgorithmSpec, ArenaNeuralTSSpec):
    """Specification for NeuralTS (Neural Thompson Sampling) algorithm."""

    actor_network: EvolvableModule | None = Field(default=None)

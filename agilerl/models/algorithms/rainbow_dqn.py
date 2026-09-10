# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Rainbow DQN algorithm specification."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import Field

from agilerl.arena.models.algorithms.rainbow_dqn import (
    RainbowDQNSpec as ArenaRainbowDQNSpec,
)
from agilerl.models.algo import RLAlgorithmSpec, off_policy, register

if TYPE_CHECKING:
    from agilerl.modules import EvolvableModule
else:
    EvolvableModule = Any


@register()
@off_policy()
class RainbowDQNSpec(RLAlgorithmSpec, ArenaRainbowDQNSpec):
    """Specification for Rainbow DQN algorithm."""

    actor_network: EvolvableModule | None = Field(default=None)

# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""PPO algorithm specification."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import Field

from agilerl.arena.models.algorithms.ppo import PPOSpec as ArenaPPOSpec
from agilerl.models.algo import RLAlgorithmSpec, register

if TYPE_CHECKING:
    from agilerl.modules import EvolvableModule
else:
    EvolvableModule = Any


@register()
class PPOSpec(RLAlgorithmSpec, ArenaPPOSpec):
    """Specification for PPO algorithm."""

    actor_network: EvolvableModule | None = Field(default=None)
    critic_network: EvolvableModule | None = Field(default=None)

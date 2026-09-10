# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""DDPG algorithm specification."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import Field

from agilerl.arena.models.algorithms.ddpg import DDPGSpec as ArenaDDPGSpec
from agilerl.models.algo import RLAlgorithmSpec, off_policy, register

if TYPE_CHECKING:
    from agilerl.modules import EvolvableModule
else:
    EvolvableModule = Any


@register()
@off_policy()
class DDPGSpec(RLAlgorithmSpec, ArenaDDPGSpec):
    """Specification for DDPG algorithm."""

    actor_network: EvolvableModule | None = Field(default=None)
    critic_network: EvolvableModule | None = Field(default=None)

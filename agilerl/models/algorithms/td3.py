# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""TD3 algorithm specification."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import Field

from agilerl.arena.models.algorithms.td3 import TD3Spec as ArenaTD3Spec
from agilerl.models.algo import RLAlgorithmSpec, off_policy, register

if TYPE_CHECKING:
    from agilerl.modules import EvolvableModule
else:
    EvolvableModule = Any


@register()
@off_policy()
class TD3Spec(RLAlgorithmSpec, ArenaTD3Spec):
    """Specification for TD3 algorithm."""

    actor_network: EvolvableModule | None = Field(default=None)
    critic_networks: list[EvolvableModule] | None = Field(default=None)

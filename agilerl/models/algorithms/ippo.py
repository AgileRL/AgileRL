# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""IPPO algorithm specification."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import Field

from agilerl.arena.models.algorithms.ippo import IPPOSpec as ArenaIPPOSpec
from agilerl.models.algo import MultiAgentRLAlgorithmSpec, register
from agilerl.models.networks import StochasticActorSpec

if TYPE_CHECKING:
    from agilerl.modules import ModuleDict
else:
    ModuleDict = Any


@register()
class IPPOSpec(MultiAgentRLAlgorithmSpec, ArenaIPPOSpec):
    """Specification for IPPO algorithm."""

    net_config: StochasticActorSpec | dict[str, StochasticActorSpec] | None = Field(
        default=None
    )
    actor_networks: ModuleDict | None = Field(default=None)
    critic_networks: ModuleDict | None = Field(default=None)

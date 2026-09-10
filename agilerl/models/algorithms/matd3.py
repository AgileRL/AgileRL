# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""MATD3 algorithm specification."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import Field

from agilerl.arena.models.algorithms.matd3 import MATD3Spec as ArenaMATD3Spec
from agilerl.models.algo import MultiAgentRLAlgorithmSpec, off_policy, register
from agilerl.models.networks import DeterministicActorSpec

if TYPE_CHECKING:
    from agilerl.modules import ModuleDict
else:
    ModuleDict = Any


@register()
@off_policy()
class MATD3Spec(MultiAgentRLAlgorithmSpec, ArenaMATD3Spec):
    """Specification for MATD3 algorithm."""

    net_config: DeterministicActorSpec | dict[str, DeterministicActorSpec] | None = (
        Field(default=None)
    )
    actor_networks: ModuleDict | None = Field(default=None)
    critic_networks: list[ModuleDict] | None = Field(default=None)

# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""IPPO algorithm specification."""

from __future__ import annotations

from collections.abc import Callable
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

    @staticmethod
    def get_training_fn() -> Callable[..., Any]:
        """Get the training function for IPPO.

        :return: Training function
        :rtype: Callable[..., Any]
        """
        from agilerl.training.train_multi_agent_on_policy import (  # circular import with agilerl.training
            train_multi_agent_on_policy,
        )

        return train_multi_agent_on_policy

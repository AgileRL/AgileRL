# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""MADDPG algorithm specification."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from pydantic import Field

from agilerl.arena.models.algorithms.maddpg import MADDPGSpec as ArenaMADDPGSpec
from agilerl.models.algo import MultiAgentRLAlgorithmSpec, off_policy, register
from agilerl.models.networks import DeterministicActorSpec

if TYPE_CHECKING:
    from agilerl.modules import ModuleDict
else:
    ModuleDict = Any


@register()
@off_policy()
class MADDPGSpec(MultiAgentRLAlgorithmSpec, ArenaMADDPGSpec):
    """Specification for MADDPG algorithm."""

    net_config: DeterministicActorSpec | dict[str, DeterministicActorSpec] | None = (
        Field(default=None)
    )
    actor_networks: ModuleDict | None = Field(default=None)
    critic_networks: ModuleDict | None = Field(default=None)

    @staticmethod
    def get_training_fn() -> Callable[..., Any]:
        """Get the training function for MADDPG.

        :return: Training function
        :rtype: Callable[..., Any]
        """
        from agilerl.training.train_multi_agent_off_policy import (  # circular import with agilerl.training
            train_multi_agent_off_policy,
        )

        return train_multi_agent_off_policy

# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""PPO algorithm specification."""

from __future__ import annotations

from collections.abc import Callable
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

    @staticmethod
    def get_training_fn() -> Callable[..., Any]:
        """Get the training function for PPO.

        :return: Training function
        :rtype: Callable[..., Any]
        """
        from agilerl.training.train_on_policy import (  # circular import with agilerl.training
            train_on_policy,
        )

        return train_on_policy

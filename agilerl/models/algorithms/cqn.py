# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""CQN algorithm specification."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from pydantic import Field

from agilerl.arena.models.algorithms.cqn import CQNSpec as ArenaCQNSpec
from agilerl.models.algo import RLAlgorithmSpec, offline, register

if TYPE_CHECKING:
    from agilerl.modules.base import EvolvableModule
else:
    EvolvableModule = Any


@register()
@offline()
class CQNSpec(RLAlgorithmSpec, ArenaCQNSpec):
    """Specification for CQN algorithm."""

    actor_network: EvolvableModule | None = Field(default=None)

    @staticmethod
    def get_training_fn() -> Callable[..., Any]:
        """Get the training function for CQN.

        :return: Training function
        :rtype: Callable[..., Any]
        """
        from agilerl.training.train_offline import (  # circular import with agilerl.training
            train_offline,
        )

        return train_offline

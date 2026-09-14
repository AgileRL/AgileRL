# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""NeuralUCB algorithm specification."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from pydantic import Field

from agilerl.arena.models.algorithms.neural_ucb import (
    NeuralUCBSpec as ArenaNeuralUCBSpec,
)
from agilerl.models.algo import RLAlgorithmSpec, bandit, register

if TYPE_CHECKING:
    from agilerl.modules import EvolvableModule
else:
    EvolvableModule = Any


@register()
@bandit()
class NeuralUCBSpec(RLAlgorithmSpec, ArenaNeuralUCBSpec):
    """Specification for NeuralUCB (Neural Upper Confidence Bound) algorithm."""

    actor_network: EvolvableModule | None = Field(default=None)

    @staticmethod
    def get_training_fn() -> Callable[..., Any]:
        """Get the training function for NeuralUCB.

        :return: Training function
        :rtype: Callable[..., Any]
        """
        from agilerl.training.train_bandits import (  # circular import with agilerl.training
            train_bandits,
        )

        return train_bandits

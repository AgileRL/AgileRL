# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Training strategy for contextual bandit algorithms."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from agilerl.strategies.base import TrainingStrategy, rl_trainer_kwargs
from agilerl.training.train_bandits import train_bandits

if TYPE_CHECKING:
    from agilerl.arena.models.algorithms import AlgoSpec
    from agilerl.components.replay_buffer import BufferType
    from agilerl.models.training import TrainingSpec
    from agilerl.strategies.base import EnvSpecType, TrainingLoop


class BanditStrategy(TrainingStrategy):
    """Contextual bandit training (NeuralTS, NeuralUCB)."""

    default_loop: ClassVar[TrainingLoop | None] = staticmethod(train_bandits)

    def get_trainer_kwargs(
        self,
        spec: AlgoSpec,
        *,
        training: TrainingSpec,
        env_spec: EnvSpecType,
        memory: BufferType | None = None,
        n_step_memory: BufferType | None = None,
    ) -> dict[str, Any]:
        kwargs = rl_trainer_kwargs(spec, training=training, env_spec=env_spec)
        kwargs["memory"] = memory
        kwargs["episode_steps"] = training.episode_steps
        return kwargs

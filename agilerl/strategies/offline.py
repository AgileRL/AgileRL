# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Training strategy for offline algorithms."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import h5py

from agilerl.models.env import OfflineEnvSpec
from agilerl.strategies.base import TrainingStrategy, rl_trainer_kwargs
from agilerl.training.train_offline import train_offline

if TYPE_CHECKING:
    from agilerl.arena.models.algorithms import AlgoSpec
    from agilerl.components.replay_buffer import BufferType
    from agilerl.models.training import TrainingSpec
    from agilerl.strategies.base import EnvSpecType, TrainingLoop


class OfflineStrategy(TrainingStrategy):
    """Offline training from a fixed dataset (CQN)."""

    default_loop: ClassVar[TrainingLoop | None] = staticmethod(train_offline)

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
        if isinstance(env_spec, OfflineEnvSpec):
            if env_spec.minari_dataset_id is not None:
                kwargs["minari_dataset_id"] = env_spec.minari_dataset_id
                kwargs["remote"] = env_spec.remote
            elif env_spec.dataset_path is not None:
                kwargs["dataset"] = h5py.File(env_spec.dataset_path, "r")
        return kwargs

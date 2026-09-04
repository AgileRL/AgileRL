# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Training strategies for multi-agent reinforcement learning."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from agilerl.strategies.base import TrainingStrategy, rl_trainer_kwargs
from agilerl.strategies.single_agent import off_policy_trainer_kwargs
from agilerl.training.train_multi_agent_off_policy import (
    train_multi_agent_off_policy,
)
from agilerl.training.train_multi_agent_on_policy import (
    train_multi_agent_on_policy,
)

if TYPE_CHECKING:
    from agilerl.arena.models.algorithms import AlgoSpec
    from agilerl.components.replay_buffer import BufferType
    from agilerl.models.training import TrainingSpec
    from agilerl.strategies.base import EnvSpecType, TrainingLoop


class MultiAgentOnPolicyStrategy(TrainingStrategy):
    """On-policy multi-agent training (IPPO)."""

    default_loop: ClassVar[TrainingLoop | None] = staticmethod(
        train_multi_agent_on_policy
    )

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
        kwargs["sum_scores"] = training.sum_scores
        return kwargs


class MultiAgentOffPolicyStrategy(TrainingStrategy):
    """Off-policy multi-agent training (MADDPG, MATD3)."""

    default_loop: ClassVar[TrainingLoop | None] = staticmethod(
        train_multi_agent_off_policy
    )

    def get_trainer_kwargs(
        self,
        spec: AlgoSpec,
        *,
        training: TrainingSpec,
        env_spec: EnvSpecType,
        memory: BufferType | None = None,
        n_step_memory: BufferType | None = None,
    ) -> dict[str, Any]:
        kwargs = {
            **rl_trainer_kwargs(spec, training=training, env_spec=env_spec),
            **off_policy_trainer_kwargs(
                training, memory=memory, n_step_memory=n_step_memory
            ),
        }
        kwargs["sum_scores"] = training.sum_scores
        return kwargs

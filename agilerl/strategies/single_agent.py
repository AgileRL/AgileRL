# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Training strategies for single-agent reinforcement learning."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from agilerl.strategies.base import TrainingStrategy, rl_trainer_kwargs
from agilerl.training.train_off_policy import train_off_policy
from agilerl.training.train_on_policy import train_on_policy

if TYPE_CHECKING:
    from agilerl.arena.models.algorithms import AlgoSpec
    from agilerl.components.replay_buffer import BufferType
    from agilerl.models.training import TrainingSpec
    from agilerl.strategies.base import EnvSpecType, TrainingLoop


class SingleAgentOnPolicyStrategy(TrainingStrategy):
    """On-policy single-agent training (PPO)."""

    default_loop: ClassVar[TrainingLoop | None] = staticmethod(train_on_policy)

    def get_trainer_kwargs(
        self,
        spec: AlgoSpec,
        *,
        training: TrainingSpec,
        env_spec: EnvSpecType,
        memory: BufferType | None = None,
        n_step_memory: BufferType | None = None,
    ) -> dict[str, Any]:
        return rl_trainer_kwargs(spec, training=training, env_spec=env_spec)


def off_policy_trainer_kwargs(
    training: TrainingSpec,
    *,
    memory: BufferType | None,
    n_step_memory: BufferType | None,
) -> dict[str, Any]:
    """The replay-buffer and exploration kwargs the off-policy loops share.

    :param training: Training specification.
    :type training: TrainingSpec
    :param memory: Replay buffer instance.
    :type memory: BufferType | None
    :param n_step_memory: N-step replay buffer for combined PER + n-step setups.
    :type n_step_memory: BufferType | None
    :returns: The shared keyword arguments.
    :rtype: dict[str, Any]
    """
    kwargs: dict[str, Any] = {
        "memory": memory,
        "learning_delay": training.learning_delay,
    }
    maybe_kwargs = {
        "eps_start": training.eps_start,
        "eps_end": training.eps_end,
        "eps_decay": training.eps_decay,
        "n_step_memory": n_step_memory,
    }
    kwargs.update({k: v for k, v in maybe_kwargs.items() if v is not None})
    return kwargs


class SingleAgentOffPolicyStrategy(TrainingStrategy):
    """Off-policy single-agent training (DQN, Rainbow DQN, DDPG, TD3)."""

    default_loop: ClassVar[TrainingLoop | None] = staticmethod(train_off_policy)

    def get_trainer_kwargs(
        self,
        spec: AlgoSpec,
        *,
        training: TrainingSpec,
        env_spec: EnvSpecType,
        memory: BufferType | None = None,
        n_step_memory: BufferType | None = None,
    ) -> dict[str, Any]:
        return {
            **rl_trainer_kwargs(spec, training=training, env_spec=env_spec),
            **off_policy_trainer_kwargs(
                training, memory=memory, n_step_memory=n_step_memory
            ),
        }

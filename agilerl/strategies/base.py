# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Which loop trains a spec, and the extra kwargs that loop takes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from collections.abc import Sequence

    from agilerl.algorithms.core import EvolvableAlgorithm
    from agilerl.arena.models.algorithms import AlgoSpec
    from agilerl.components.replay_buffer import BufferType
    from agilerl.models.env import EnvSpec
    from agilerl.models.training import TrainingSpec

    EnvSpecType = EnvSpec
    # Loops return their concrete population list plus one fitness entry per
    # agent (a per-agent dict from the multi-agent loops); Sequence keeps the
    # alias covariant with those concrete types.
    TrainingLoop = Callable[
        ...,
        tuple[
            Sequence[EvolvableAlgorithm],
            Sequence[int | float | dict[str, int | float]],
        ],
    ]
else:
    TrainingLoop = Callable[..., Any]


class TrainingStrategy(ABC):
    """Paradigm-keyed run-time orchestration for an algorithm spec.

    Subclasses set :attr:`default_loop` and implement
    :meth:`get_trainer_kwargs`; a paradigm with more than one loop overrides
    :meth:`get_training_loop`.
    """

    default_loop: ClassVar[TrainingLoop | None] = None

    def get_training_loop(self, spec: AlgoSpec) -> TrainingLoop:
        """Select the training loop for *spec*.

        :param spec: The algorithm spec.
        :type spec: AlgoSpec
        :returns: The training function.
        :rtype: TrainingLoop
        :raises NotImplementedError: If the strategy names no loop.
        """
        loop = type(self).default_loop
        if loop is None:
            msg = "Training strategies must set default_loop or override get_training_loop."
            raise NotImplementedError(msg)
        return loop

    @abstractmethod
    def get_trainer_kwargs(
        self,
        spec: AlgoSpec,
        *,
        training: TrainingSpec,
        env_spec: EnvSpecType,
        memory: BufferType | None = None,
        n_step_memory: BufferType | None = None,
    ) -> dict[str, Any]:
        """Return the extra keyword arguments the training loop takes.

        :param spec: The algorithm spec.
        :type spec: AlgoSpec
        :param training: Training specification.
        :type training: TrainingSpec
        :param env_spec: Environment specification.
        :type env_spec: EnvSpecType
        :param memory: Replay buffer instance.
        :type memory: BufferType | None
        :param n_step_memory: N-step replay buffer for combined PER + n-step setups.
        :type n_step_memory: BufferType | None
        :returns: Extra keyword arguments for the training function.
        :rtype: dict[str, Any]
        """


def rl_trainer_kwargs(
    spec: AlgoSpec, *, training: TrainingSpec, env_spec: EnvSpecType
) -> dict[str, Any]:
    """The keyword arguments every classic RL loop takes.

    :param spec: The algorithm spec.
    :type spec: AlgoSpec
    :param training: Training specification.
    :type training: TrainingSpec
    :param env_spec: Environment specification.
    :type env_spec: EnvSpecType
    :returns: The shared keyword arguments.
    :rtype: dict[str, Any]
    """
    return {
        "env_name": env_spec.name,
        "algo": spec.name,
        "eval_steps": training.eval_steps,
        "eval_loop": training.eval_loop,
        "target": training.target_score,
        "checkpoint": training.checkpoint_steps,
        "checkpoint_path": training.checkpoint_path,
        "overwrite_checkpoints": training.overwrite_checkpoints,
    }

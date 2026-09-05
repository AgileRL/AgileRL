# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the LLM finetuning entry points.

Imported by both :mod:`agilerl.training.llm.rollout` (generative rollout
training) and :mod:`agilerl.training.llm.dataset` (teacher-forced dataset
training).
"""

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from agilerl.hpo.mutation import Mutations
from agilerl.protocols import SelectionStrategyProtocol
from agilerl.training.configs import LLMTrainCheckpointConfig
from agilerl.utils.utils import save_llm_checkpoint

if TYPE_CHECKING:
    from agilerl.llm_envs import DatasetEnv


def _validate_finetune_args(
    evo_steps: int | None,
    selection_strategy: SelectionStrategyProtocol | None,
    mutation: Mutations | None,
    num_epochs: int | None,
    max_steps: int | None,
    pop: list[Any],
    expected_type: type | tuple[type, ...],
    algorithm_type_error: str,
    *,
    checkpoint_steps: int | None = None,
) -> None:
    if evo_steps is not None and (selection_strategy is None or mutation is None):
        warnings.warn(
            "'evo_steps' is set but at least one of 'selection_strategy' or "
            "'mutation' is set to None. Evolution will not take place.",
            stacklevel=2,
        )
    if (selection_strategy is not None and mutation is not None) and evo_steps is None:
        msg = (
            "'evo_steps' must be set if 'selection_strategy' and 'mutation' "
            "are not None."
        )
        raise ValueError(msg)
    if num_epochs is not None and max_steps is not None:
        warnings.warn(
            "'num_epochs' is set but 'max_steps' is also set. "
            "'num_epochs' will take precedence over 'max_steps'.",
            stacklevel=2,
        )

    evo_active = (
        evo_steps is not None
        and selection_strategy is not None
        and mutation is not None
    )
    if checkpoint_steps is not None and evo_active:
        warnings.warn(
            "'checkpoint_steps' is set, but evolution is active ('evo_steps', "
            "'selection_strategy', and 'mutation'). Periodic step-based checkpoints "
            "are skipped while evolution is enabled.",
            stacklevel=2,
        )

    if mutation is not None:
        assert mutation.architecture_mut == 0, (
            "Probability of architecture mutation must be 0 for LLM finetuning."
        )
        assert mutation.new_layer_prob == 0, (
            "Probability of new layer mutation must be 0 for LLM finetuning."
        )
        assert mutation.parameters_mut == 0, (
            "Probability of network parameters mutation must be 0 for LLM finetuning."
        )
        assert mutation.activation_mut == 0, (
            "Probability of activation mutation must be 0 for LLM finetuning."
        )

    if not isinstance(pop[0], expected_type):
        raise ValueError(algorithm_type_error)


def _compute_training_steps(
    max_steps: int | None,
    num_epochs: int | None,
    env_len: int,
    effective_data_batch_size: int,
    pop_size: int = 1,
) -> tuple[int, int]:
    """Compute the number of training steps."""
    if max_steps is None and num_epochs is None:
        max_steps = env_len
    elif max_steps is None and num_epochs is not None:
        max_steps = num_epochs * env_len
    assert max_steps is not None
    steps_per_iteration = effective_data_batch_size * pop_size
    training_steps = -(max_steps // -steps_per_iteration)
    return max_steps, training_steps


def _resolve_training_envs(
    pop: list[Any],
    env: "DatasetEnv | None",
    env_fn: "Callable[[], DatasetEnv] | None",
) -> "tuple[list[DatasetEnv], bool]":
    """Resolve shared or per-agent training environments.

    :param pop: Population of agents being trained.
    :type pop: list[Any]
    :param env: Shared environment instance.
    :type env: DatasetEnv | None
    :param env_fn: Factory for creating one environment per agent.
    :type env_fn: Callable[[], DatasetEnv] | None
    :return: Environment list (aligned with population) and whether env_fn mode is active.
    :rtype: tuple[list[DatasetEnv], bool]
    """
    if env is not None and env_fn is not None:
        msg = "Provide exactly one of 'env' or 'env_fn', not both."
        raise ValueError(msg)
    if env is None and env_fn is None:
        msg = "Either 'env' or 'env_fn' must be provided."
        raise ValueError(msg)

    if env_fn is not None:
        return [env_fn() for _ in pop], True

    if len(pop) > 1:
        warnings.warn(
            "A shared 'env' is being used with multiple agents. This can introduce "
            "fairness bias; prefer 'env_fn' for per-agent environments.",
            stacklevel=2,
        )
    assert env is not None
    return [env], False


@dataclass
class LLMCheckpointProgress:
    """Step-based checkpoint cadence for LLM train loops without evolution."""

    next_checkpoint_step: int | None
    max_steps_checkpoint_saved: bool = False


def save_llm_elite_if_requested(
    agents: list[Any],
    checkpoint: LLMTrainCheckpointConfig,
) -> None:
    """Write the highest-fitness agent when elite export is configured."""
    if not checkpoint.save_elite or checkpoint.elite_path is None:
        return
    elite = max(
        agents,
        key=lambda agent: agent.fitness[-1] if agent.fitness else float("-inf"),
    )
    save_llm_checkpoint(elite, checkpoint.elite_path)


def maybe_save_llm_step_checkpoint(
    agents: list[Any],
    progress: LLMCheckpointProgress,
    checkpoint: LLMTrainCheckpointConfig,
    total_steps: int,
    max_steps: int,
) -> None:
    """Save a periodic or end-of-run checkpoint when evolution is off."""
    due = False
    if checkpoint.checkpoint_steps is not None:
        while (
            progress.next_checkpoint_step is not None
            and total_steps >= progress.next_checkpoint_step
        ):
            due = True
            progress.next_checkpoint_step += checkpoint.checkpoint_steps
    path_configured = (
        checkpoint.checkpoint_steps is not None
        or checkpoint.checkpoint_path is not None
        or checkpoint.elite_path is not None
    )
    if (
        total_steps >= max_steps
        and not progress.max_steps_checkpoint_saved
        and path_configured
    ):
        due = True
        progress.max_steps_checkpoint_saved = True
    if due:
        save_llm_checkpoint(
            agents[-1],
            checkpoint.checkpoint_path
            if checkpoint.checkpoint_path is not None
            else checkpoint.elite_path,
        )

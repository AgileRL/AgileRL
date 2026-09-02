# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the LLM finetuning entry points.

Imported by both :mod:`agilerl.training.llm.rollout` (generative rollout
training) and :mod:`agilerl.training.llm.dataset` (teacher-forced dataset
training).
"""

import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from agilerl.hpo.mutation import Mutations
from agilerl.protocols import SelectionStrategyProtocol

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

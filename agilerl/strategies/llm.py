# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Training strategies for LLM fine-tuning."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

from agilerl.models.env import LLMEnvSpec, LLMEnvType
from agilerl.strategies.base import TrainingStrategy

if TYPE_CHECKING:
    from agilerl.arena.models.algorithms import AlgoSpec
    from agilerl.components.replay_buffer import BufferType
    from agilerl.models.training import TrainingSpec
    from agilerl.strategies.base import EnvSpecType, TrainingLoop

# TrainingSpec fields with no equivalent in the LLM fine-tuning loops
UNSUPPORTED_TRAINING_FIELDS = (
    "target_score",
    "eval_steps",
    "eval_loop",
    "learning_delay",
    "eps_start",
    "eps_end",
    "eps_decay",
    "overwrite_checkpoints",
)


def _warn_ignored_training_fields(training: TrainingSpec) -> None:
    """Warn when explicitly-set TrainingSpec fields are ignored by LLM loops."""
    ignored = [
        name
        for name in UNSUPPORTED_TRAINING_FIELDS
        if name in training.model_fields_set
        and getattr(training, name) != type(training).model_fields[name].default
    ]
    if ignored:
        warnings.warn(
            "TrainingSpec fields not supported by LLM fine-tuning are ignored: "
            + ", ".join(ignored),
            UserWarning,
            stacklevel=4,
        )


class LLMStrategy(TrainingStrategy):
    """Shared orchestration for the LLM fine-tuning loops."""

    def get_trainer_kwargs(
        self,
        spec: AlgoSpec,
        *,
        training: TrainingSpec,
        env_spec: EnvSpecType,
        memory: BufferType | None = None,
        n_step_memory: BufferType | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "evaluation_interval": training.evaluation_interval,
        }
        maybe_kwargs: dict[str, Any] = {
            "checkpoint_steps": training.checkpoint_steps,
            "checkpoint_path": training.checkpoint_path,
        }
        if training.num_epochs is not None:
            if LLMEnvType(spec.env_type) == LLMEnvType.DATASET:
                maybe_kwargs["num_epochs"] = training.num_epochs
            else:
                warnings.warn(
                    "TrainingSpec.num_epochs only applies to dataset "
                    "fine-tuning (DPO/SFT) and is ignored for rollout "
                    "algorithms.",
                    UserWarning,
                    stacklevel=2,
                )
        if isinstance(env_spec, LLMEnvSpec):
            maybe_kwargs["max_reward"] = env_spec.max_reward
        kwargs.update({k: v for k, v in maybe_kwargs.items() if v is not None})
        _warn_ignored_training_fields(training)
        return kwargs


# The LLM loops are imported inside ``get_training_loop`` rather than at module
# load: ``agilerl.training.llm`` imports GRPO, DPO, SFT, ... which only exist
# with the ``[llm]`` extra, and an RL-only install still has to import
# ``LocalTrainer`` (and so this package).


class LLMRolloutStrategy(LLMStrategy):
    """Generative rollout fine-tuning (GRPO family, LLM PPO, LLM REINFORCE).

    One loop for every rollout regime: single-turn reasoning is
    ``max_turns=1``.
    """

    def get_training_loop(self, spec: AlgoSpec) -> TrainingLoop:
        # optional LLM extra
        from agilerl.training.llm import train_llm_rollout

        return train_llm_rollout


class LLMDatasetStrategy(LLMStrategy):
    """Teacher-forced fine-tuning over dataset rows (DPO and SFT).

    One loop for both objectives; the env's ``objective`` picks the loss.
    """

    def get_training_loop(self, spec: AlgoSpec) -> TrainingLoop:
        # optional LLM extra
        from agilerl.training.llm import train_llm_dataset

        return train_llm_dataset

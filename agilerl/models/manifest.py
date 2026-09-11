# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""The training manifest contract, plus a constructor from trainer specs.

:class:`TrainingManifest` is the arena class. This module re-exports it and
adds :func:`from_trainer_specs` for building one from the objects a trainer
already holds.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from pydantic import BaseModel

from agilerl.arena.models import TrainingManifest
from agilerl.arena.models.algorithms import (
    AlgoSpec,
    IPPOSpec,
    LLMAlgorithmSpec,
    MADDPGSpec,
    MATD3Spec,
    SingleAgentAlgorithmSpec,
)
from agilerl.models.hpo import (
    MultiFrequencySelectionSpec,
    MutationSpec,
    SelectionStrategySpec,
    TournamentSelectionSpec,
)
from agilerl.models.networks import FinetuningNetworkSpec, NetworkSpec
from agilerl.models.training import LLMRolloutBufferSpec, ReplayBufferSpec, TrainingSpec

__all__ = ["TrainingManifest", "from_trainer_specs"]

SpecT = TypeVar("SpecT", bound=BaseModel)


def from_trainer_specs(
    *,
    algorithm: AlgoSpec,
    environment: BaseModel,
    training: TrainingSpec | BaseModel,
    mutation: MutationSpec | None = None,
    replay_buffer: ReplayBufferSpec | LLMRolloutBufferSpec | None = None,
    selection_strategy: SelectionStrategySpec | None = None,
    **kwargs: Any,
) -> TrainingManifest:
    """Build a validated manifest from the trainer's component specs.

    :param algorithm: Framework algorithm spec.
    :type algorithm: AlgoSpec
    :param environment: Environment spec held on the trainer.
    :type environment: BaseModel
    :param training: Training loop parameters; a foreign model is dumped so the
        manifest rebuilds it as its own :class:`TrainingSpec`.
    :type training: TrainingSpec | BaseModel
    :param mutation: Optional mutation spec.
    :type mutation: MutationSpec | None
    :param replay_buffer: Optional replay-buffer spec.
    :type replay_buffer: ReplayBufferSpec | LLMRolloutBufferSpec | None
    :param selection_strategy: Tournament or multi-frequency selection.
    :type selection_strategy: SelectionStrategySpec | None
    :param kwargs: Accepts the deprecated tournament_selection alias.
    :returns: A validated :class:`TrainingManifest`.
    :rtype: TrainingManifest
    """
    from agilerl.utils.trainer_utils import resolve_deprecated_selection_kwargs

    selection_strategy = resolve_deprecated_selection_kwargs(
        selection_strategy,
        kwargs,
        deprecated_key="tournament_selection",
        caller="from_trainer_specs",
    )

    def _coerce(
        value: BaseModel | Mapping[str, Any] | None,
        accepted: type[SpecT] | tuple[type[SpecT], ...],
    ) -> SpecT | Mapping[str, Any] | None:
        """Dump a foreign section so the field can rebuild it as its own class."""
        if value is None or isinstance(value, accepted):
            return value
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json", exclude_none=True)
        return value

    return TrainingManifest(
        algorithm=algorithm,
        environment=environment.model_dump(mode="json", exclude_none=True),
        training=(
            training
            if isinstance(training, TrainingSpec)
            else training.model_dump(mode="json", exclude_none=True)
        ),
        network=_network_from_algorithm(algorithm),
        mutation=_coerce(mutation, MutationSpec),
        replay_buffer=_coerce(replay_buffer, (ReplayBufferSpec, LLMRolloutBufferSpec)),
        selection_strategy=_coerce(
            selection_strategy,
            (TournamentSelectionSpec, MultiFrequencySelectionSpec),
        ),
    )


def _network_from_algorithm(
    algorithm: AlgoSpec,
) -> NetworkSpec | FinetuningNetworkSpec | Mapping[str, NetworkSpec] | None:
    """Resolve the manifest ``network`` section from an algorithm spec."""
    if isinstance(algorithm, LLMAlgorithmSpec):
        return FinetuningNetworkSpec.model_validate(
            {
                "pretrained_model_name_or_path": (
                    algorithm.pretrained_model_name_or_path
                ),
                "max_context_length": algorithm.max_model_len,
                "lora_config": algorithm.lora_config,
            }
        )
    if isinstance(algorithm, SingleAgentAlgorithmSpec):
        return algorithm.net_config
    if isinstance(algorithm, (IPPOSpec, MADDPGSpec, MATD3Spec)):
        return algorithm.net_config
    return None

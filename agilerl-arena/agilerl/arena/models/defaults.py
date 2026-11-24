# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Arena's starting manifest for an LLM algorithm, built by the manifest model."""

from __future__ import annotations

import copy
from typing import Any

from pydantic import ValidationError

from agilerl.arena.models import LLMAlgorithmSpec
from agilerl.arena.models.algorithms.rollout_llm import RolloutLLMSpec
from agilerl.arena.models.env import LLMEnvSpec
from agilerl.arena.models.manifest import REWARD_FILE_AT_SUBMIT, TrainingManifest
from agilerl.arena.models.model_info import SUPPORTED_MODEL_INFO
from agilerl.arena.models.registry import MANIFEST_REGISTRY
from agilerl.arena.models.training import (
    DATASET_LLM_NUM_EPOCHS,
    LLM_EVALUATION_INTERVAL,
    LLM_REPORTING_INTERVAL,
    ROLLOUT_LLM_MAX_STEPS,
)
from agilerl.arena.models.verdict import errors, failure, verdict


def default_verdict(document: dict[str, Any]) -> dict[str, Any]:
    """Fill *document* with Arena's defaults for its LLM algorithm, as a verdict; the reward file may come at submit.

    :param document: A partial manifest: ``algorithm.name``, the environment,
        and ``network.pretrained_model_name_or_path``.
    :type document: dict[str, Any]
    :returns: The verdict; on success ``payload`` is the full starting manifest.
    :rtype: dict[str, Any]
    """
    context = {REWARD_FILE_AT_SUBMIT: True}
    draft = copy.deepcopy(document)
    sections = ("algorithm", "environment", "network", "training")
    if not all(isinstance(draft.get(section, {}), dict) for section in sections):
        return verdict(draft, context=context)
    algorithm_name = draft.get("algorithm", {}).get("name")
    try:
        spec_cls = MANIFEST_REGISTRY.get(algorithm_name)
    except KeyError:
        return verdict(draft, context=context)
    if not issubclass(spec_cls, LLMAlgorithmSpec):
        return failure(
            [
                {
                    "path": "algorithm.name",
                    "message": f"Arena builds defaults for LLM algorithms; {algorithm_name} is not one.",
                    "kind": "value_error",
                }
            ]
        )

    network = draft.setdefault("network", {})
    info = SUPPORTED_MODEL_INFO.get(network.get("pretrained_model_name_or_path"))
    if info is not None:
        network.setdefault("max_context_length", info.max_context_length)
    try:
        manifest = TrainingManifest.model_validate(draft, context=context)
    except (ValidationError, ValueError) as err:
        return failure(errors(err))
    training = draft.setdefault("training", {})
    for key, value in training_defaults(manifest).items():
        training.setdefault(key, value)
    return verdict(draft, context=context)


def training_defaults(manifest: TrainingManifest) -> dict[str, Any]:
    """Training values a new LLM run starts from, beyond the model's own defaults.

    :param manifest: The validated manifest.
    :type manifest: TrainingManifest
    :returns: Training fields to set where the document leaves them out.
    :rtype: dict[str, Any]
    """
    training = manifest.training
    defaults: dict[str, Any] = {
        "reporting_interval": LLM_REPORTING_INTERVAL,
        "evaluation_interval": LLM_EVALUATION_INTERVAL,
        "checkpoint_export": training.effective_checkpoint_export().model_dump(
            mode="json", exclude_none=True
        ),
    }
    if training.model_fields_set & {"num_epochs", "max_steps"}:
        return defaults
    environment = manifest.environment
    over_rows = environment.env_type == "dataset" or (
        isinstance(environment, LLMEnvSpec) and environment.dataset_backed_rollout
    )
    if over_rows:
        defaults["num_epochs"] = DATASET_LLM_NUM_EPOCHS
    elif isinstance(manifest.algorithm, RolloutLLMSpec):
        defaults["max_steps"] = ROLLOUT_LLM_MAX_STEPS
    return defaults

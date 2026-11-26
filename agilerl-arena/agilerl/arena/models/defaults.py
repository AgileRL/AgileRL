# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Arena's starting manifest for an LLM algorithm, built by the manifest model."""

from __future__ import annotations

import copy
from typing import Any

from pydantic import ValidationError

from agilerl.arena.models import LLMAlgorithmSpec, visibility
from agilerl.arena.models.algorithms.rollout_llm import RolloutLLMSpec
from agilerl.arena.models.env import LLMEnvSpec
from agilerl.arena.models.manifest import REWARD_FILE_AT_SUBMIT, TrainingManifest
from agilerl.arena.models.model_info import SUPPORTED_MODEL_INFO
from agilerl.arena.models.registry import MANIFEST_REGISTRY
from agilerl.arena.models.training import (
    LLM_EVALUATION_INTERVAL,
    LLM_REPORTING_INTERVAL,
    ROLLOUT_LLM_MAX_STEPS,
)
from agilerl.arena.models.verdict import errors, failure, is_local_model_path, verdict


def default_verdict(
    document: dict[str, Any], user_view: bool = False
) -> dict[str, Any]:
    """Fill *document* with Arena's defaults for its LLM algorithm, as a verdict; the reward file may come at submit.

    :param document: A partial manifest: ``algorithm.name``, the environment,
        and ``network.pretrained_model_name_or_path``.
    :type document: dict[str, Any]
    :param user_view: Leave out the fields Arena chooses (:mod:`~agilerl.arena.models.visibility`).
    :type user_view: bool
    :returns: The verdict; on success ``payload`` is the starting manifest.
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

    model = draft.get("network", {}).get("pretrained_model_name_or_path")
    if (
        isinstance(model, str)
        and model not in SUPPORTED_MODEL_INFO
        and not is_local_model_path(model)
    ):
        return failure(
            [
                {
                    "path": "network.pretrained_model_name_or_path",
                    "message": f"Arena does not support the model '{model}'.",
                    "kind": "value_error",
                }
            ]
        )

    try:
        manifest = TrainingManifest.model_validate(draft, context=context)
    except (ValidationError, ValueError) as err:
        return failure(errors(err))
    training = draft.setdefault("training", {})
    for key, value in training_defaults(manifest).items():
        training.setdefault(key, value)
    record = verdict(draft, context=context)
    if user_view and record["ok"]:
        record["payload"] = template_view(record["payload"])
    return record


def template_view(spec: dict[str, Any], section: str | None = None) -> dict[str, Any]:
    """A copy of *spec* without the fields Arena chooses, as a user starts from it.

    :param spec: A whole manifest, or one section of it.
    :type spec: dict[str, Any]
    :param section: The section *spec* holds, or ``None`` for a whole manifest.
    :type section: str | None
    :returns: The spec without Arena-chosen fields.
    :rtype: dict[str, Any]
    """
    fields = visibility.marked_fields(TrainingManifest)
    if section is None:
        return visibility.user_view(spec, fields)
    return visibility.user_view({section: spec}, fields).get(section, {})


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
    # Unset over dataset rows trains one pass, which the trainer sizes from the rows.
    environment = manifest.environment
    over_rows = environment.env_type == "dataset" or (
        isinstance(environment, LLMEnvSpec) and environment.dataset_backed_rollout
    )
    if (
        "max_steps" not in training.model_fields_set
        and isinstance(manifest.algorithm, RolloutLLMSpec)
        and not over_rows
    ):
        defaults["max_steps"] = ROLLOUT_LLM_MAX_STEPS
    return defaults

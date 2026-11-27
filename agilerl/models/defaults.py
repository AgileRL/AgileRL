# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Arena's starting manifest for an LLM algorithm, built by the manifest model."""

from __future__ import annotations

import copy
from typing import Any

from pydantic import ValidationError

from agilerl.models import LLMAlgorithmSpec, visibility
from agilerl.models.algorithms.opd import OPDSpec
from agilerl.models.algorithms.rollout_llm import RolloutLLMSpec
from agilerl.models.bundled_envs import manifest_model
from agilerl.models.env import LLMEnvSpec
from agilerl.models.manifest import REWARD_FILE_AT_SUBMIT, TrainingManifest
from agilerl.models.model_info import DEFAULT_MODEL
from agilerl.models.prompt_template import DEFAULT_PROMPT_TEMPLATE
from agilerl.models.registry import MANIFEST_REGISTRY
from agilerl.models.traces import TRACES_DATASET_FORMAT
from agilerl.models.training import (
    LLM_EVALUATION_INTERVAL,
    LLM_REPORTING_INTERVAL,
    ROLLOUT_LLM_MAX_STEPS,
)
from agilerl.models.verdict import (
    errors,
    failure,
    unsupported_model_failures,
    verdict,
)


def default_verdict(
    document: dict[str, Any], user_view: bool = False
) -> dict[str, Any]:
    """Fill *document* with Arena's defaults for its LLM algorithm, as a verdict; the reward file may come at submit.

    :param document: A partial manifest: ``algorithm.name``, the environment,
        and optionally a model on ``network`` or ``algorithm``.
    :type document: dict[str, Any]
    :param user_view: Leave out the fields Arena chooses (:mod:`~agilerl.models.visibility`).
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

    if manifest_model(draft) is None:
        network = draft.setdefault("network", {})
        network["pretrained_model_name_or_path"] = DEFAULT_MODEL
    environment = draft.get("environment", {})
    prompts_rows = (
        environment.get("env_type") == "rollout"
        and any(
            environment.get(key) is not None
            for key in ("dataset", "dataset_path", "hf_dataset_id")
        )
        and environment.get("dataset_format") != TRACES_DATASET_FORMAT
    )
    if prompts_rows:
        environment.setdefault("prompt_template", dict(DEFAULT_PROMPT_TEMPLATE))
        # The built-in rewards score a correct answer 1; the success metric counts those.
        environment.setdefault("max_reward", 1.0)

    unsupported = unsupported_model_failures(draft)
    if unsupported:
        return failure(unsupported)
    if issubclass(spec_cls, OPDSpec):
        reference_answer_column(draft)

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


def reference_answer_column(draft: dict[str, Any]) -> None:
    """Point OPD's self-distilling teacher at the dataset's answer column when the draft names no reference.

    :param draft: An OPD draft whose sections are mappings; edited in place.
    :type draft: dict[str, Any]
    """
    columns = draft.get("environment", {}).get("columns")
    if not isinstance(columns, dict) or "answer" not in columns:
        return
    teacher = draft["algorithm"].get("teacher", {})
    if not isinstance(teacher, dict) or any(
        teacher.get(key) for key in ("model", "adapter", "base", "context_key")
    ):
        return
    draft["algorithm"]["teacher"] = {**teacher, "context_key": "answer"}


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

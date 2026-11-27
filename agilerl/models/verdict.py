# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Validate a manifest and report the result as JSON-safe data."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from agilerl.models import (
    FinetuningNetworkSpec,
    LLMAlgorithmSpec,
    __version__,
)
from agilerl.models.algorithms.rollout_llm import RolloutLLMSpec
from agilerl.models.manifest import (
    TrainingManifest,
    explicit_colocated_rollout,
    mutates,
    validated_run_manifest,
)
from agilerl.models.model_info import SUPPORTED_MODEL_INFO
from agilerl.models.networks import target_modules_warnings
from agilerl.models.registry import did_you_mean
from agilerl.models.training import classic_only_key_warnings
from agilerl.models.visibility import arena_chosen_paths

EXIT_OK = 0
EXIT_INVALID = 1
EXIT_FAILED = 2

HPO_UNAVAILABLE_MESSAGE = (
    "HPO and populations aren't available for LLM training on Arena. "
    "Remove training.hpo and the mutation section, and set training.pop_size to 1."
)
ARENA_CHOSEN_MESSAGE = "Arena chooses this field; remove it from the manifest."
CLASSIC_RL_UNAVAILABLE_MESSAGE = (
    "Classic RL is no longer supported in Arena. Use the open-source AgileRL v2 "
    "release: https://github.com/AgileRL/AgileRL/releases/tag/v2.67.0"
)
# Registered by AgileRL 2.x; the 3.0 CLI names them rather than calling them unknown.
CLASSIC_RL_ALGORITHMS = frozenset(
    {
        "CQN",
        "DDPG",
        "DQN",
        "IPPO",
        "MADDPG",
        "MATD3",
        "NeuralTS",
        "NeuralUCB",
        "PPO",
        "Rainbow DQN",
        "RainbowDQN",
        "Recurrent PPO",
        "RecurrentPPO",
        "TD3",
    }
)


def loc_to_path(loc: tuple[Any, ...]) -> str:
    """Render a pydantic error location as a dotted path.

    :param loc: A pydantic error ``loc`` tuple.
    :type loc: tuple[Any, ...]
    :returns: A dotted path such as ``algorithm.vllm_config.max_num_seqs``.
    :rtype: str
    """
    parts: list[str] = []
    for entry in loc:
        if isinstance(entry, int):
            if parts:
                parts[-1] = f"{parts[-1]}[{entry}]"
            continue
        text = str(entry)
        if "[" in text or text.endswith("]"):
            continue
        parts.append(text)
    return ".".join(parts)


def describe(error: dict[str, str]) -> str:
    """Render one error record as ``path: message``, or the message alone without a path.

    :param error: A record from :func:`errors` or :func:`submit_errors`.
    :type error: dict[str, str]
    :rtype: str
    """
    return f"{error['path']}: {error['message']}" if error["path"] else error["message"]


def errors(err: ValidationError | ValueError) -> list[dict[str, str]]:
    """Flatten a validation failure into JSON-safe error records.

    :param err: The failure.
    :type err: ValidationError | ValueError
    :returns: One record per error, with ``path``, ``message`` and ``kind``.
    :rtype: list[dict[str, str]]
    """
    if isinstance(err, ValidationError):
        return [
            {
                "path": loc_to_path(e["loc"]),
                "message": e["msg"],
                "kind": e["type"],
            }
            for e in err.errors()
        ]
    return [{"path": "", "message": str(err), "kind": "value_error"}]


def read_manifest(source: str) -> dict[str, Any]:
    """Load a manifest document from a path, or from stdin when *source* is ``-``.

    :param source: A path, or ``-``.
    :type source: str
    :returns: The raw document.
    :rtype: dict[str, Any]
    """
    if source == "-":
        return yaml.safe_load(sys.stdin.read())
    return TrainingManifest.load(source)


def is_local_model_path(model: str) -> bool:
    """Whether *model* is a filesystem path or a checkout, not a Hub id.

    :param model: ``pretrained_model_name_or_path``.
    :type model: str
    :returns: True for a ``.`` / ``/`` / ``~`` prefix, or a directory with ``config.json``.
    :rtype: bool
    """
    return (
        model.startswith((".", "/", "~"))
        or (Path(model).expanduser() / "config.json").is_file()
    )


def unsupported_model_failures(document: dict[str, Any]) -> list[dict[str, str]]:
    """Report a Hub id outside the Arena catalog as one error.

    :param document: The raw manifest document.
    :type document: dict[str, Any]
    :returns: Zero or one error record.
    :rtype: list[dict[str, str]]
    """
    network = document.get("network")
    model = (
        network.get("pretrained_model_name_or_path")
        if isinstance(network, dict)
        else None
    )
    if (
        not isinstance(model, str)
        or model in SUPPORTED_MODEL_INFO
        or is_local_model_path(model)
    ):
        return []
    return [
        {
            "path": "network.pretrained_model_name_or_path",
            "message": (
                f"Arena does not support the model '{model}'."
                f"{did_you_mean(model, SUPPORTED_MODEL_INFO)}"
            ),
            "kind": "value_error",
        }
    ]


def unsupported_hub_id_failures(document: dict[str, Any]) -> list[dict[str, str]]:
    """Report an ``org/name`` Hub id outside the Arena catalog as one error.

    :param document: The raw manifest document.
    :type document: dict[str, Any]
    :returns: Zero or one error record.
    :rtype: list[dict[str, str]]
    """
    # A name without "/" is a model registered on Arena, which the platform resolves.
    return [
        record
        for record in unsupported_model_failures(document)
        if "/" in document["network"]["pretrained_model_name_or_path"]
    ]


def unsupported_model_symptom_failures(
    document: dict[str, Any], err: ValidationError | ValueError
) -> list[dict[str, str]]:
    """Report an ``org/name`` Hub id outside the catalog instead of the context error it causes.

    An unknown Hub id has no catalog window, so an unset context stays unset.
    A name without ``/`` is registered on Arena and keeps the context error.

    :param document: The raw manifest document.
    :type document: dict[str, Any]
    :param err: The error validating *document* raised.
    :type err: ValidationError | ValueError
    :returns: The one unsupported-model error, or none when *err* is not a context error for a Hub id.
    :rtype: list[dict[str, str]]
    """
    if all(record["path"] != "network.max_context_length" for record in errors(err)):
        return []
    return unsupported_hub_id_failures(document)


def model_warnings(manifest: TrainingManifest) -> list[str]:
    """Warn when an LLM manifest's model is outside the Arena catalog, deprecated or in preview.

    :param manifest: A validated manifest.
    :type manifest: TrainingManifest
    :returns: Zero or one warning.
    :rtype: list[str]
    """
    algorithm = manifest.algorithm
    if not isinstance(algorithm, LLMAlgorithmSpec):
        return []
    hub_id = algorithm.pretrained_model_name_or_path
    if hub_id is None:
        return []
    info = SUPPORTED_MODEL_INFO.get(hub_id)
    if info is not None:
        notice = info.status_notice
        return [notice] if notice else []
    if is_local_model_path(hub_id):
        return []
    msg = (
        f"Model {hub_id} is not in the Arena supported model catalog; "
        f"see 'arena models supported'.{did_you_mean(hub_id, SUPPORTED_MODEL_INFO)}"
    )
    return [msg]


def model_status_errors(manifest: TrainingManifest) -> list[dict[str, str]]:
    """Report a catalog model that is deprecated or in preview, which Arena refuses on submit.

    :param manifest: A validated manifest.
    :type manifest: TrainingManifest
    :returns: Zero or one error record.
    :rtype: list[dict[str, str]]
    """
    algorithm = manifest.algorithm
    if not isinstance(algorithm, LLMAlgorithmSpec):
        return []
    hub_id = algorithm.pretrained_model_name_or_path
    info = SUPPORTED_MODEL_INFO.get(hub_id) if hub_id else None
    notice = info.status_notice if info else None
    if notice is None:
        return []
    return [
        {
            "path": "network.pretrained_model_name_or_path",
            "message": notice,
            "kind": "model_status",
        }
    ]


def context_warnings(manifest: TrainingManifest) -> list[str]:
    """Warn when Arena filled the context with its default because the manifest left it unset.

    :param manifest: A validated manifest.
    :type manifest: TrainingManifest
    :returns: Zero or one warning.
    :rtype: list[str]
    """
    network = manifest.network
    if not manifest.context_defaulted or not isinstance(network, FinetuningNetworkSpec):
        return []
    message = (
        f"network.max_context_length is unset, so Arena used a default "
        f"context of {network.max_context_length} tokens. The run and its "
        "memory and cost estimates are sized for that, and rows that do not "
        "fit are dropped; set network.max_context_length to fit your longest "
        "prompt and completion."
    )
    return [message]


def catalog_gpu_share_warnings(
    manifest: TrainingManifest, document: dict[str, Any]
) -> list[str]:
    """Warn when a catalog rollout share replaces the share *document* sets.

    :param manifest: *document*, validated.
    :type manifest: TrainingManifest
    :param document: The raw manifest document.
    :type document: dict[str, Any]
    :returns: Zero or one warning.
    :rtype: list[str]
    """
    algorithm = manifest.algorithm
    if not isinstance(algorithm, RolloutLLMSpec) or algorithm.vllm_config is None:
        return []
    chosen = algorithm.vllm_config.gpu_memory_utilization
    raw_algorithm = document.get("algorithm")
    if not isinstance(raw_algorithm, dict):
        return []
    raw_vllm = raw_algorithm.get("vllm_config")
    if not isinstance(raw_vllm, dict) or "gpu_memory_utilization" not in raw_vllm:
        return []
    set_share = raw_vllm["gpu_memory_utilization"]
    if set_share is None or set_share == chosen:
        return []
    return [
        (
            "algorithm.vllm_config.gpu_memory_utilization is "
            f"{set_share} in the manifest; Arena uses the catalog share {chosen}."
        )
    ]


def override_warnings(
    manifest: TrainingManifest,
    document: dict[str, Any],
    context: dict[str, Any] | None = None,
) -> list[str]:
    """Warn for each advanced field *document* sets to a value Arena would not choose.

    :param manifest: *document*, validated.
    :type manifest: TrainingManifest
    :param document: The raw manifest document.
    :type document: dict[str, Any]
    :param context: The pydantic validation context *manifest* was validated with.
    :type context: dict[str, Any] | None
    :returns: One warning per field.
    :rtype: list[str]
    """
    return [
        f"{path} overrides the value Arena would choose for it. Leave it out to use Arena's."
        for path in arena_chosen_paths(
            TrainingManifest, document, manifest, context, tier="advanced"
        )
    ]


def manifest_warnings(
    manifest: TrainingManifest,
    document: dict[str, Any],
    context: dict[str, Any] | None = None,
) -> list[str]:
    """Every warning a valid manifest gets: model, context, overrides, vLLM share and LoRA targets.

    :param manifest: *document*, validated.
    :type manifest: TrainingManifest
    :param document: The raw manifest document.
    :type document: dict[str, Any]
    :param context: The pydantic validation context *manifest* was validated with.
    :type context: dict[str, Any] | None
    :returns: The warnings.
    :rtype: list[str]
    """
    return [
        *model_warnings(manifest),
        *context_warnings(manifest),
        *override_warnings(manifest, document, context),
        *catalog_gpu_share_warnings(manifest, document),
        *target_modules_warnings(document),
    ]


def hpo_fields(manifest: TrainingManifest) -> list[str]:
    """List the manifest fields that turn on HPO, or a population for an LLM, which Arena does not run.

    A mutation section that mutates nothing is allowed: ``arena manifest
    template`` prints one.

    :param manifest: A validated manifest.
    :type manifest: TrainingManifest
    :returns: Any of ``training.hpo``, ``mutation`` and ``training.pop_size``.
    :rtype: list[str]
    """
    fields = ["training.hpo"] if manifest.training.hpo else []
    if manifest.mutation is not None and mutates(manifest.mutation):
        fields.append("mutation")
    if (
        isinstance(manifest.algorithm, LLMAlgorithmSpec)
        and manifest.training.pop_size > 1
    ):
        fields.append("training.pop_size")
    return fields


def is_classic_rl(algorithm_name: str) -> bool:
    """Whether *algorithm_name* names an AgileRL 2.x classic RL algorithm.

    Arena trains only the LLM algorithms; the rest run with open-source AgileRL.

    :param algorithm_name: The manifest's ``algorithm.name``.
    :type algorithm_name: str
    :returns: True for classic RL algorithms such as PPO or DQN.
    :rtype: bool
    """
    return algorithm_name in CLASSIC_RL_ALGORITHMS


def names_classic_rl(document: dict[str, Any]) -> bool:
    """Whether the raw manifest *document* names a classic RL algorithm."""
    algorithm = document.get("algorithm")
    return isinstance(algorithm, dict) and is_classic_rl(str(algorithm.get("name")))


def classic_rl_failures() -> list[dict[str, str]]:
    """The failure record for a manifest that names a classic RL algorithm."""
    return [
        {
            "path": "algorithm.name",
            "message": CLASSIC_RL_UNAVAILABLE_MESSAGE,
            "kind": "classic_rl_unavailable",
        }
    ]


def arena_failures(manifest: TrainingManifest) -> list[dict[str, str]]:
    """List what stops a valid manifest from running on Arena.

    :param manifest: A validated manifest.
    :type manifest: TrainingManifest
    :returns: One record per failure, with ``path``, ``message`` and ``kind``.
    :rtype: list[dict[str, str]]
    """
    if not isinstance(manifest.algorithm, LLMAlgorithmSpec):
        return classic_rl_failures()
    return [
        *(
            {
                "path": field,
                "message": HPO_UNAVAILABLE_MESSAGE,
                "kind": "hpo_unavailable",
            }
            for field in hpo_fields(manifest)
        ),
        *batch_size_errors(manifest),
        *context_length_errors(manifest),
    ]


def not_allowed(path: str, requirement: str, options: list[int]) -> dict[str, str]:
    """An error record for a value outside a fixed list of options, worded as Arena words it."""
    allowed = ", ".join(str(option) for option in options)
    return {
        "path": path,
        "message": f"must be {requirement}. Allowed: {allowed}",
        "kind": "not_allowed",
    }


def batch_size_errors(manifest: TrainingManifest) -> list[dict[str, str]]:
    """Arena takes an ``algorithm.batch_size`` that is a power of 2 from 1 to 1024.

    :param manifest: A validated manifest.
    :type manifest: TrainingManifest
    :returns: Zero or one error record.
    :rtype: list[dict[str, str]]
    """
    maximum = 1024
    options = [1 << n for n in range(maximum.bit_length())]
    if manifest.algorithm.batch_size in options:
        return []
    requirement = f"a power of 2 between 1 and {maximum}"
    return [not_allowed("algorithm.batch_size", requirement, options)]


def context_length_errors(manifest: TrainingManifest) -> list[dict[str, str]]:
    """Arena takes an LLM ``max_context_length`` from the catalog model's options.

    The options are the powers of 2 from 1024 up to the model's context window,
    plus the window itself; a window below 1024 is the only option.

    :param manifest: A validated manifest.
    :type manifest: TrainingManifest
    :returns: Zero or one error record.
    :rtype: list[dict[str, str]]
    """
    network = manifest.network
    if not isinstance(manifest.algorithm, LLMAlgorithmSpec) or not isinstance(
        network, FinetuningNetworkSpec
    ):
        return []
    entry = SUPPORTED_MODEL_INFO.get(network.pretrained_model_name_or_path)
    if entry is None:
        return []
    catalog_max = entry.max_context_length
    floor = 1024
    if catalog_max < floor:
        options = [catalog_max]
        requirement = f"{catalog_max}"
    else:
        options = [floor << n for n in range((catalog_max // floor).bit_length())]
        requirement = f"a power of 2 between {floor} and {catalog_max}"
        if options[-1] != catalog_max:
            options.append(catalog_max)
            requirement += f", or {catalog_max}"
    if network.max_context_length in options:
        return []
    return [not_allowed("network.max_context_length", requirement, options)]


def run_errors(manifest: TrainingManifest) -> list[dict[str, str]]:
    """Arena runs every rollout LLM job async; check the manifest as that run.

    Async rollout needs the LLM replay buffer, so a manifest that names another
    buffer, or ``rollout_mode: colocated``, fails here and not in the schema.

    :param manifest: A validated manifest.
    :type manifest: TrainingManifest
    :returns: The error records, or an empty list.
    :rtype: list[dict[str, str]]
    """
    if explicit_colocated_rollout(manifest):
        return [
            {
                "path": "training.rollout_mode",
                "message": (
                    "colocated runs only locally; Arena runs rollout LLM jobs "
                    "async. Set rollout_mode: async or drop the key."
                ),
                "kind": "colocated_rollout",
            }
        ]
    try:
        validated_run_manifest(manifest)
    except ValidationError as err:
        return errors(err)
    return []


def arena_chosen_errors(
    manifest: TrainingManifest,
    document: dict[str, Any],
    context: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Fields Arena chooses that *document* sets to another value, less those :func:`hpo_fields` reports.

    :param manifest: *document*, validated.
    :type manifest: TrainingManifest
    :param document: The raw manifest document.
    :type document: dict[str, Any]
    :param context: The pydantic validation context *manifest* was validated with.
    :type context: dict[str, Any] | None
    :returns: One error record per field.
    :rtype: list[dict[str, str]]
    """
    reported = set(hpo_fields(manifest))
    return [
        {"path": path, "message": ARENA_CHOSEN_MESSAGE, "kind": "arena_chosen_field"}
        for path in arena_chosen_paths(TrainingManifest, document, manifest, context)
        if path not in reported
    ]


def submit_errors(
    manifest: TrainingManifest,
    document: dict[str, Any],
    context: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """What Arena refuses on submit that the manifest schema allows.

    ``arena experiments submit`` stops on these before it uploads, and
    ``arena manifest validate`` reports them.

    :param manifest: *document*, validated.
    :type manifest: TrainingManifest
    :param document: The raw manifest document.
    :type document: dict[str, Any]
    :param context: The pydantic validation context *manifest* was validated with.
    :type context: dict[str, Any] | None
    :returns: The error records, or an empty list.
    :rtype: list[dict[str, str]]
    """
    return [
        *run_errors(manifest),
        *unsupported_hub_id_failures(document),
        *model_status_errors(manifest),
        *arena_chosen_errors(manifest, document, context),
    ]


def failure(failures: list[dict[str, str]]) -> dict[str, Any]:
    """A verdict record for a manifest that failed.

    :param failures: Error records, as :func:`errors` builds them.
    :type failures: list[dict[str, str]]
    :returns: The verdict.
    :rtype: dict[str, Any]
    """
    return {
        "ok": False,
        "errors": failures,
        "warnings": [],
        "manifest_version": __version__,
    }


def verdict(
    document: dict[str, Any],
    submit: bool = False,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate *document* and report the outcome as one JSON-safe record.

    :param document: The raw manifest document.
    :type document: dict[str, Any]
    :param submit: Also report :func:`submit_errors`, as ``arena manifest
        validate`` does. Arena validates a stored run without them.
    :type submit: bool
    :param context: Pydantic validation context.
    :type context: dict[str, Any] | None
    :returns: The verdict.
    :rtype: dict[str, Any]
    """
    if names_classic_rl(document):
        return failure(classic_rl_failures())
    try:
        manifest = TrainingManifest.model_validate(document, context=context)
    except (ValidationError, ValueError) as err:
        failures = unsupported_model_symptom_failures(document, err) or errors(err)
    else:
        failures = arena_failures(manifest)
        if submit:
            failures += submit_errors(manifest, document, context)
        if not failures:
            return {
                "ok": True,
                "payload": manifest.to_payload(),
                "warnings": [
                    *manifest_warnings(manifest, document, context),
                    *classic_only_key_warnings(document),
                ],
                "manifest_version": __version__,
            }
    return failure(failures)


__all__ = [
    "ARENA_CHOSEN_MESSAGE",
    "CLASSIC_RL_ALGORITHMS",
    "CLASSIC_RL_UNAVAILABLE_MESSAGE",
    "EXIT_FAILED",
    "EXIT_INVALID",
    "EXIT_OK",
    "HPO_UNAVAILABLE_MESSAGE",
    "arena_failures",
    "catalog_gpu_share_warnings",
    "classic_rl_failures",
    "context_warnings",
    "describe",
    "errors",
    "failure",
    "hpo_fields",
    "is_classic_rl",
    "is_local_model_path",
    "loc_to_path",
    "manifest_warnings",
    "model_status_errors",
    "model_warnings",
    "names_classic_rl",
    "override_warnings",
    "read_manifest",
    "submit_errors",
    "target_modules_warnings",
    "unsupported_hub_id_failures",
    "unsupported_model_failures",
    "unsupported_model_symptom_failures",
    "verdict",
]

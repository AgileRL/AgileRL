# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Validate a manifest and report the result as JSON-safe data."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from agilerl.arena.models import (
    FinetuningNetworkSpec,
    LLMAlgorithmSpec,
    __version__,
)
from agilerl.arena.models.algorithms.rollout_llm import RolloutLLMSpec
from agilerl.arena.models.manifest import (
    TrainingManifest,
    explicit_colocated_rollout,
    mutates,
    validated_run_manifest,
)
from agilerl.arena.models.model_info import SUPPORTED_MODEL_INFO
from agilerl.arena.models.networks import target_modules_warnings
from agilerl.arena.models.registry import MANIFEST_REGISTRY, did_you_mean
from agilerl.arena.models.visibility import arena_chosen_paths

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


def context_warnings(manifest: TrainingManifest) -> list[str]:
    """Warn when Arena set the context to the model's whole window because the manifest left it unset.

    :param manifest: A validated manifest.
    :type manifest: TrainingManifest
    :returns: Zero or one warning.
    :rtype: list[str]
    """
    network = manifest.network
    if not manifest.context_from_catalog or not isinstance(
        network, FinetuningNetworkSpec
    ):
        return []
    message = (
        f"network.max_context_length is unset, so Arena used "
        f"{network.pretrained_model_name_or_path}'s full context window "
        f"({network.max_context_length} tokens). The run and its memory "
        "estimate are sized for that; set a smaller network.max_context_length "
        "if your prompts and completions are shorter."
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
    """Whether *algorithm_name* is a registered algorithm that is not an LLM one.

    Arena trains only the LLM algorithms; the rest run with open-source AgileRL.
    An unregistered name is not classic: the server decides what it is.

    :param algorithm_name: The manifest's ``algorithm.name``.
    :type algorithm_name: str
    :returns: True for classic RL algorithms such as PPO or DQN.
    :rtype: bool
    """
    try:
        spec_cls = MANIFEST_REGISTRY.get(algorithm_name)
    except KeyError:
        return False
    return not issubclass(spec_cls, LLMAlgorithmSpec)


def arena_failures(manifest: TrainingManifest) -> list[dict[str, str]]:
    """List what stops a valid manifest from running on Arena.

    :param manifest: A validated manifest.
    :type manifest: TrainingManifest
    :returns: One record per failure, with ``path``, ``message`` and ``kind``.
    :rtype: list[dict[str, str]]
    """
    if not isinstance(manifest.algorithm, LLMAlgorithmSpec):
        return [
            {
                "path": "algorithm.name",
                "message": CLASSIC_RL_UNAVAILABLE_MESSAGE,
                "kind": "classic_rl_unavailable",
            }
        ]
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
    try:
        manifest = TrainingManifest.model_validate(document, context=context)
    except (ValidationError, ValueError) as err:
        failures = errors(err)
    else:
        failures = arena_failures(manifest)
        if submit:
            failures += submit_errors(manifest, document, context)
        if not failures:
            return {
                "ok": True,
                "payload": manifest.to_payload(),
                "warnings": [
                    *model_warnings(manifest),
                    *context_warnings(manifest),
                    *override_warnings(manifest, document, context),
                    *catalog_gpu_share_warnings(manifest, document),
                    *target_modules_warnings(document),
                ],
                "manifest_version": __version__,
            }
    return failure(failures)


__all__ = [
    "ARENA_CHOSEN_MESSAGE",
    "CLASSIC_RL_UNAVAILABLE_MESSAGE",
    "EXIT_FAILED",
    "EXIT_INVALID",
    "EXIT_OK",
    "HPO_UNAVAILABLE_MESSAGE",
    "arena_failures",
    "catalog_gpu_share_warnings",
    "context_warnings",
    "describe",
    "errors",
    "failure",
    "hpo_fields",
    "is_classic_rl",
    "is_local_model_path",
    "loc_to_path",
    "model_warnings",
    "override_warnings",
    "read_manifest",
    "submit_errors",
    "target_modules_warnings",
    "verdict",
]

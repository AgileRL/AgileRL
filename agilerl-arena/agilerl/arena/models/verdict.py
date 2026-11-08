# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Validate a manifest and report the result as JSON-safe data."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from agilerl.arena.models import LLMAlgorithmSpec, __version__
from agilerl.arena.models.manifest import TrainingManifest
from agilerl.arena.models.model_info import SUPPORTED_MODEL_INFO
from agilerl.arena.models.registry import did_you_mean

EXIT_OK = 0
EXIT_INVALID = 1
EXIT_FAILED = 2


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
    # A path prefix, or a checkout that already has config.json, is not a Hub id.
    if (
        hub_id.startswith((".", "/", "~"))
        or (Path(hub_id).expanduser() / "config.json").is_file()
    ):
        return []
    msg = (
        f"Model {hub_id} is not in the Arena supported model catalog; "
        f"see 'arena models supported'.{did_you_mean(hub_id, SUPPORTED_MODEL_INFO)}"
    )
    return [msg]


def verdict(document: dict[str, Any]) -> dict[str, Any]:
    """Validate *document* and report the outcome as one JSON-safe record.

    :param document: The raw manifest document.
    :type document: dict[str, Any]
    :returns: The verdict.
    :rtype: dict[str, Any]
    """
    try:
        manifest = TrainingManifest.model_validate(document)
    except (ValidationError, ValueError) as err:
        return {
            "ok": False,
            "errors": errors(err),
            "warnings": [],
            "manifest_version": __version__,
        }
    return {
        "ok": True,
        "payload": manifest.to_payload(),
        "warnings": model_warnings(manifest),
        "manifest_version": __version__,
    }


__all__ = [
    "EXIT_FAILED",
    "EXIT_INVALID",
    "EXIT_OK",
    "errors",
    "loc_to_path",
    "model_warnings",
    "read_manifest",
    "verdict",
]

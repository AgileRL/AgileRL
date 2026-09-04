# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Validate a manifest and report the result as JSON-safe data."""

from __future__ import annotations

import sys
from typing import Any

import yaml
from pydantic import ValidationError

from agilerl.arena.models import __version__
from agilerl.arena.models.manifest import TrainingManifest

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
        return {"ok": False, "errors": errors(err), "manifest_version": __version__}
    return {
        "ok": True,
        "payload": manifest.to_payload(),
        "manifest_version": __version__,
    }


__all__ = [
    "EXIT_FAILED",
    "EXIT_INVALID",
    "EXIT_OK",
    "errors",
    "loc_to_path",
    "read_manifest",
    "verdict",
]

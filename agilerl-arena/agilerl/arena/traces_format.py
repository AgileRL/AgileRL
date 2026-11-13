# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Names of the ``traces`` dataset format (v1) and its snapshot ``manifest.json``.

This module imports only the standard library. Importing it still runs the
``agilerl.arena`` package init, which loads the pydantic manifest models but
not torch; a reader without ``agilerl-arena`` installed can copy this file. It lives in ``agilerl-arena``, which ``agilerl``
depends on, so the standalone CLI, the inference pod and the snapshot export
read one definition.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar, get_args, get_type_hints

__all__ = [
    "ANSWER",
    "BASE_MODEL",
    "CALL_ID",
    "CHECKPOINT_ID",
    "COMPLETION",
    "FINISH_REASON",
    "JSON_TEXT_COLUMNS",
    "MANIFEST_FIELDS",
    "MANIFEST_FILE",
    "METADATA",
    "MODEL_VERSION",
    "PROMPT",
    "REASONING_KEYS",
    "RECEIVED_AT",
    "REQUIRED_COLUMNS",
    "REQUIRED_MANIFEST_FIELDS",
    "SCORES",
    "SCORE_FIELDS",
    "SEGMENT_INDEX",
    "SESSION_ID",
    "SNAPSHOT_CALLS_FILE",
    "TOOLS",
    "TRACES_FORMAT",
    "TRACES_SCHEMA_VERSION",
    "TRACE_COLUMNS",
    "ExcludedCounts",
    "ModelVersionCount",
    "SnapshotFile",
    "SnapshotManifest",
    "TimeRange",
    "read_manifest",
    "write_manifest",
]

T = TypeVar("T")

TRACES_FORMAT = "agilerl-traces"
"""``format`` of a snapshot ``manifest.json``. The dataset category and ``environment.dataset_format`` are ``traces``."""
TRACES_SCHEMA_VERSION = 1

PROMPT = "prompt"
COMPLETION = "completion"
TOOLS = "tools"
ANSWER = "answer"
CALL_ID = "call_id"
SESSION_ID = "session_id"
SEGMENT_INDEX = "segment_index"
MODEL_VERSION = "model_version"
CHECKPOINT_ID = "checkpoint_id"
BASE_MODEL = "base_model"
RECEIVED_AT = "received_at"
SCORES = "scores"
FINISH_REASON = "finish_reason"
METADATA = "metadata"

TRACE_COLUMNS = (
    PROMPT,
    COMPLETION,
    TOOLS,
    ANSWER,
    CALL_ID,
    SESSION_ID,
    SEGMENT_INDEX,
    MODEL_VERSION,
    CHECKPOINT_ID,
    BASE_MODEL,
    RECEIVED_AT,
    SCORES,
    FINISH_REASON,
    METADATA,
)
"""Every column of a traces row, in the spec's order."""

REQUIRED_COLUMNS = (PROMPT, COMPLETION)
JSON_TEXT_COLUMNS = (PROMPT, COMPLETION, TOOLS)
"""Columns a Parquet traces file holds as JSON text."""

SCORE_FIELDS = ("name", "value", "source", "label", "comment")
"""Keys of one entry in a row's ``scores`` list."""

REASONING_KEYS = ("reasoning_content", "thinking", "reasoning")
"""Message keys an assistant turn's reasoning is read from; writers use the first."""

MANIFEST_FILE = "manifest.json"
SNAPSHOT_CALLS_FILE = "calls.parquet"


@dataclass(frozen=True)
class ModelVersionCount:
    """Rows a snapshot holds from one model version."""

    model_version: str
    base_model: str
    checkpoint_id: str | None
    row_count: int


@dataclass(frozen=True)
class TimeRange:
    """The first and last ``received_at`` in a snapshot."""

    start: datetime
    end: datetime


@dataclass(frozen=True)
class ExcludedCounts:
    """Calls left out of a snapshot because they cannot be traces rows."""

    no_content: int = 0
    error_or_cancelled: int = 0


@dataclass(frozen=True)
class SnapshotFile:
    """One data file of a snapshot, relative to its manifest."""

    path: str
    row_count: int
    sha256: str


@dataclass(frozen=True)
class SnapshotManifest:
    """A snapshot's ``manifest.json``."""

    snapshot_id: str
    row_count: int
    session_count: int
    model_versions: list[ModelVersionCount]
    score_names: list[str]
    label_score: str | None
    files: list[SnapshotFile]
    created_at: datetime | None = None
    created_by: str | None = None
    time_range: TimeRange | None = None
    filter: dict[str, Any] = dataclasses.field(default_factory=dict)
    parent_snapshot_id: str | None = None
    excluded: ExcludedCounts = dataclasses.field(default_factory=ExcludedCounts)
    organization_id: str | None = None
    deployment_ids: list[str] = dataclasses.field(default_factory=list)
    format: str = TRACES_FORMAT
    schema_version: int = TRACES_SCHEMA_VERSION


MANIFEST_FIELDS = tuple(item.name for item in dataclasses.fields(SnapshotManifest))
"""Every ``manifest.json`` key this format defines."""

REQUIRED_MANIFEST_FIELDS = (
    "format",
    "schema_version",
    "snapshot_id",
    "row_count",
    "session_count",
    "model_versions",
    "score_names",
    "label_score",
    "files",
)
"""The keys the platform reads."""


def write_manifest(path: str | Path, manifest: SnapshotManifest) -> None:
    """Write ``manifest`` as JSON, timestamps as ISO 8601.

    :param path: The ``manifest.json`` to write.
    :param manifest: The manifest.
    """
    Path(path).write_text(
        json.dumps(dataclasses.asdict(manifest), default=_iso, indent=2, sort_keys=True)
    )


def read_manifest(path: str | Path) -> SnapshotManifest:
    """Read a ``manifest.json``; keys this format does not define are ignored.

    :param path: The ``manifest.json`` to read.
    :return: The manifest; optional keys it lacks take their defaults.
    :raises ValueError: If ``format`` is not :data:`TRACES_FORMAT`, ``schema_version``
        is not :data:`TRACES_SCHEMA_VERSION`, a required key is missing, or a key
        has the wrong type.
    """
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        msg = "manifest is not a JSON object."
        raise ValueError(msg)
    if data.get("format") != TRACES_FORMAT:
        msg = f"manifest format is {data.get('format')!r}, not {TRACES_FORMAT!r}."
        raise ValueError(msg)
    if data.get("schema_version") != TRACES_SCHEMA_VERSION:
        msg = (
            f"manifest schema_version is {data.get('schema_version')!r}; this "
            f"reader understands {TRACES_SCHEMA_VERSION}."
        )
        raise ValueError(msg)
    missing = [name for name in REQUIRED_MANIFEST_FIELDS if name not in data]
    if missing:
        msg = f"manifest is missing {', '.join(missing)}."
        raise ValueError(msg)
    fields: dict[str, Any] = {
        "snapshot_id": _typed(data, "snapshot_id", str),
        "row_count": _typed(data, "row_count", int),
        "session_count": _typed(data, "session_count", int),
        "model_versions": [
            _record(ModelVersionCount, item, "model_versions")
            for item in _typed(data, "model_versions", list)
        ],
        "score_names": _typed(data, "score_names", list),
        "label_score": _optional(data, "label_score", str),
        "files": [
            _record(SnapshotFile, item, "files") for item in _typed(data, "files", list)
        ],
    }
    optional: dict[str, Any] = {
        "created_by": _optional(data, "created_by", str),
        "filter": _optional(data, "filter", dict),
        "parent_snapshot_id": _optional(data, "parent_snapshot_id", str),
        "organization_id": _optional(data, "organization_id", str),
        "deployment_ids": _optional(data, "deployment_ids", list),
    }
    if data.get("created_at") is not None:
        optional["created_at"] = _timestamp(_typed(data, "created_at", str))
    time_range = _optional(data, "time_range", dict)
    if time_range is not None:
        optional["time_range"] = TimeRange(
            _timestamp(_typed(time_range, "start", str)),
            _timestamp(_typed(time_range, "end", str)),
        )
    excluded = _optional(data, "excluded", dict)
    if excluded is not None:
        optional["excluded"] = _record(ExcludedCounts, excluded, "excluded")
    fields.update(
        {name: value for name, value in optional.items() if value is not None}
    )
    return SnapshotManifest(**fields)


def _typed(data: dict[str, Any], name: str, kind: type[T]) -> T:
    """``data[name]`` if it is a ``kind``."""
    value = data.get(name)
    if not _is_kind(value, kind):
        msg = f"manifest {name} is {type(value).__name__}, not {kind.__name__}."
        raise ValueError(msg)
    return value


def _is_kind(value: object, kind: type) -> bool:
    """Whether ``value`` is a ``kind``; bool is an int subclass, but a count of True is a writer bug."""
    return isinstance(value, kind) and not (kind is int and isinstance(value, bool))


def _optional(data: dict[str, Any], name: str, kind: type[T]) -> T | None:
    """``data[name]`` if it is a ``kind``, or ``None`` when absent or null."""
    return None if data.get(name) is None else _typed(data, name, kind)


def _record(cls: type[T], item: object, name: str) -> T:
    """One nested manifest object as ``cls``; keys ``cls`` does not define are ignored."""
    if not isinstance(item, dict):
        msg = f"manifest {name} entry is {type(item).__name__}, not an object."
        raise ValueError(msg)
    hints = get_type_hints(cls)
    values = {key: value for key, value in item.items() if key in hints}
    for key, value in values.items():
        kinds = get_args(hints[key]) or (hints[key],)
        if not any(
            value is None if kind is type(None) else _is_kind(value, kind)
            for kind in kinds
        ):
            msg = f"manifest {name} entry {key} is {type(value).__name__}."
            raise ValueError(msg)
    try:
        return cls(**values)
    except TypeError as error:
        msg = f"manifest {name} entry {item!r}: {error}"
        raise ValueError(msg) from error


def _timestamp(text: str) -> datetime:
    """An ISO 8601 timestamp; a trailing ``Z`` is UTC on Python 3.10 too."""
    return datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith("Z") else text)


def _iso(value: object) -> str:
    """JSON text for a timestamp."""
    if isinstance(value, datetime):
        return value.isoformat()
    msg = f"{type(value).__name__} is not JSON serializable"
    raise TypeError(msg)

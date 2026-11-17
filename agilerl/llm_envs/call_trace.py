# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Call records as Parquet, and ``traces`` dataset snapshots of append-only segments."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from agilerl.arena.models.traces import TracesSegmentRow
from agilerl.arena.traces_format import (
    ANSWER,
    BASE_MODEL,
    CALL_IDS,
    CHECKPOINT_ID,
    COMPLETION_LOGPROBS,
    COMPLETION_TOKEN_IDS,
    FINISH_REASONS,
    MANIFEST_FILE,
    MESSAGES,
    METADATA,
    MODEL_VERSION,
    REASONING_KEYS,
    RECEIVED_AT,
    SCORES,
    SEGMENT_INDEX,
    SESSION_ID,
    SNAPSHOT_SEGMENTS_FILE,
    TOOLS,
    ExcludedCounts,
    ModelVersionCount,
    SnapshotFile,
    SnapshotManifest,
    TimeRange,
    manifest_json,
)
from agilerl.llm_envs.call_record import (
    CallRecord,
    ScoreRecord,
    message_hash,
    new_call_id,
    to_row,
)
from agilerl.llm_envs.call_sessions import Session

__all__ = [
    "CALL_RECORD_SCHEMA",
    "SNAPSHOT_SCHEMA",
    "SnapshotSegment",
    "calls_table",
    "snapshot_segments",
    "snapshot_table",
    "write_snapshot",
]

CALL_RECORD_SCHEMA = pa.schema(
    [
        ("call_id", pa.string()),
        ("request_id", pa.string()),
        ("schema_version", pa.int32()),
        ("organization_id", pa.string()),
        ("deployment_id", pa.string()),
        ("instance_id", pa.string()),
        ("user_id", pa.string()),
        ("end_user", pa.string()),
        ("client_session_id", pa.string()),
        ("metadata", pa.string()),
        ("base_model", pa.string()),
        ("checkpoint_id", pa.string()),
        ("model_version", pa.string()),
        ("chat_template_hash", pa.string()),
        ("sampling", pa.string()),
        ("received_at", pa.timestamp("us", tz="UTC")),
        ("queue_ms", pa.int64()),
        ("first_token_ms", pa.int64()),
        ("total_ms", pa.int64()),
        ("prompt_tokens", pa.int64()),
        ("completion_tokens", pa.int64()),
        ("cached_tokens", pa.int64()),
        ("finish_reason", pa.string()),
        ("error", pa.string()),
        ("stream", pa.bool_()),
        ("prompt_prefix_hashes", pa.list_(pa.string())),
        ("response_prefix_hash", pa.string()),
        ("messages", pa.string()),
        ("tools", pa.string()),
        ("tool_choice", pa.string()),
        ("response_message", pa.string()),
        ("raw_completion", pa.string()),
        ("prompt_token_ids", pa.list_(pa.int32())),
        ("completion_token_ids", pa.list_(pa.int32())),
        ("completion_logprobs", pa.list_(pa.float32())),
    ]
)
"""Arrow schema of a ``calls/`` Parquet file, one row per :class:`CallRecord`."""

SNAPSHOT_SCHEMA = pa.schema(
    [
        pa.field(MESSAGES, pa.string()),
        pa.field(TOOLS, pa.string()),
        pa.field(CALL_IDS, pa.list_(pa.string())),
        pa.field(FINISH_REASONS, pa.list_(pa.string())),
        pa.field(COMPLETION_TOKEN_IDS, pa.list_(pa.list_(pa.int32()))),
        pa.field(COMPLETION_LOGPROBS, pa.list_(pa.list_(pa.float32()))),
        pa.field(SESSION_ID, pa.string()),
        pa.field(SEGMENT_INDEX, pa.int32()),
        pa.field(MODEL_VERSION, pa.string()),
        pa.field(CHECKPOINT_ID, pa.string()),
        pa.field(BASE_MODEL, pa.string()),
        pa.field(RECEIVED_AT, pa.timestamp("us", tz="UTC")),
        pa.field(METADATA, pa.string()),
        pa.field(ANSWER, pa.string()),
        pa.field(
            SCORES,
            pa.list_(
                pa.struct(
                    [
                        ("name", pa.string()),
                        ("value", pa.float64()),
                        ("source", pa.string()),
                        ("label", pa.string()),
                        ("comment", pa.string()),
                        ("turn", pa.int32()),
                    ]
                )
            ),
        ),
    ]
)
"""Arrow schema of a snapshot's ``segments.parquet``, one row per append-only segment."""


@dataclass(frozen=True)
class SnapshotSegment:
    """One append-only segment of a session, with the messages its row holds."""

    session_key: str
    calls: list[CallRecord]
    messages: list[dict[str, Any]]
    """The row's ``messages``, see :func:`snapshot_table`."""
    turns: list[CallRecord | None]
    """Per assistant message, the call that produced it, ``None`` for context."""


def calls_table(records: Iterable[CallRecord]) -> pa.Table:
    """Call records as an Arrow table of :data:`CALL_RECORD_SCHEMA`.

    :param records: Call records.
    :return: The table, one row per record.
    """
    return pa.Table.from_pylist(
        [to_row(record) for record in records], schema=CALL_RECORD_SCHEMA
    )


def snapshot_segments(
    sessions: Iterable[Session],
) -> tuple[list[SnapshotSegment], ExcludedCounts]:
    """Split stitched sessions into the append-only segments a snapshot stores.

    Calls with no content and calls that ended in ``error`` or ``cancelled``
    are left out and counted. A kept call starts a new snapshot segment where
    stitching split, or when its request is not the previous kept call's
    request plus response, its ``model_version`` or ``tools`` differ, or its
    request has an assistant message after that response (one the harness
    inserted, which no recorded call produced). A segment whose row breaks the
    traces spec is left out, its calls counted as ``invalid``.

    :param sessions: Sessions from :func:`~agilerl.llm_envs.call_sessions.stitch_sessions`.
    :return: The segments, in session then call order, and the counts left out.
    """
    segments: list[tuple[str, list[CallRecord]]] = []
    no_content = error_or_cancelled = 0
    for session in sessions:
        for stitched in session.segments:
            current: list[CallRecord] = []
            for call in stitched:
                if call.finish_reason in ("error", "cancelled"):
                    error_or_cancelled += 1
                    continue
                if call.messages is None or call.response_message is None:
                    no_content += 1
                    continue
                if current and not _appends(current[-1], call):
                    segments.append((session.key, current))
                    current = []
                current.append(call)
            if current:
                segments.append((session.key, current))
    valid: list[SnapshotSegment] = []
    invalid = 0
    for key, calls in segments:
        segment = _traces_segment(key, calls)
        if segment is None:
            invalid += len(calls)
        else:
            valid.append(segment)
    return valid, ExcludedCounts(no_content, error_or_cancelled, invalid)


def _traces_segment(key: str, calls: list[CallRecord]) -> SnapshotSegment | None:
    """``calls`` as a segment of session ``key``, or ``None`` when its row breaks the traces spec."""
    try:
        messages, turns = _segment_messages(calls)
        TracesSegmentRow.model_validate(
            {
                MESSAGES: messages,
                TOOLS: calls[0].tools or None,
                METADATA: calls[0].metadata or None,
                **_per_turn_columns(turns),
            }
        )
    except ValueError:
        return None
    return SnapshotSegment(key, calls, messages, turns)


def snapshot_table(
    sessions: Iterable[Session], scores: Iterable[ScoreRecord] = ()
) -> tuple[pa.Table, ExcludedCounts]:
    """Stitched sessions as traces segment rows of :data:`SNAPSHOT_SCHEMA`, with their scores joined.

    A row's ``messages`` is its last call's request and response, with each
    earlier call's reply as recorded, so reasoning a harness dropped when
    resending is kept. Assistant messages keep reasoning under
    ``reasoning_content`` and tool-call arguments as JSON text. ``call_ids``,
    ``finish_reasons``, ``completion_token_ids`` and ``completion_logprobs``
    have one entry per assistant message, ``None`` for one no call in the
    segment produced. Row-level columns come from the
    segment's first call.

    :param sessions: Sessions from :func:`~agilerl.llm_envs.call_sessions.stitch_sessions`.
    :param scores: Scores to join by ``call_id``, with ``turn`` that call's
        assistant index, or by ``session_key`` onto every row of that session.
    :return: The table, one row per :func:`snapshot_segments` segment, and the
        counts left out.
    :raises ValueError: If a score does not set exactly one of ``call_id``
        and ``session_key``.
    """
    segments, excluded = snapshot_segments(sessions)
    rows = _segment_rows(segments, scores)
    return pa.Table.from_pylist(rows, schema=SNAPSHOT_SCHEMA), excluded


def _segment_rows(
    segments: list[SnapshotSegment], scores: Iterable[ScoreRecord]
) -> list[dict[str, Any]]:
    """``segments`` as rows of :data:`SNAPSHOT_SCHEMA`, see :func:`snapshot_table`."""
    by_call, by_session = _scores_by_target(scores)
    rows: list[dict[str, Any]] = []
    index_in_session: Counter[str] = Counter()
    for segment in segments:
        key = segment.session_key
        joined = [
            _score_entry(score, turn)
            for turn, call in enumerate(segment.turns)
            if call is not None
            for score in by_call.get(call.call_id, [])
        ] + [_score_entry(score, None) for score in by_session.get(key, [])]
        first = segment.calls[0]
        rows.append(
            {
                MESSAGES: json.dumps(segment.messages, ensure_ascii=False),
                TOOLS: None if not first.tools else json.dumps(first.tools),
                **_per_turn_columns(segment.turns),
                SESSION_ID: key,
                SEGMENT_INDEX: index_in_session[key],
                MODEL_VERSION: first.model_version,
                CHECKPOINT_ID: first.checkpoint_id,
                BASE_MODEL: first.base_model,
                RECEIVED_AT: first.received_at,
                METADATA: json.dumps(first.metadata) if first.metadata else None,
                ANSWER: None,
                SCORES: joined or None,
            }
        )
        index_in_session[key] += 1
    return rows


def _scores_by_target(
    scores: Iterable[ScoreRecord],
) -> tuple[dict[str, list[ScoreRecord]], dict[str, list[ScoreRecord]]]:
    """``scores`` keyed by ``call_id``, and by ``session_key``.

    :raises ValueError: If a score does not set exactly one of the two.
    """
    by_call: dict[str, list[ScoreRecord]] = {}
    by_session: dict[str, list[ScoreRecord]] = {}
    for score in scores:
        if score.call_id is not None and score.session_key is None:
            by_call.setdefault(score.call_id, []).append(score)
        elif score.session_key is not None and score.call_id is None:
            by_session.setdefault(score.session_key, []).append(score)
        else:
            msg = (
                f"score {score.name!r} must set exactly one of call_id and "
                "session_key, to attach it to one call or one session."
            )
            raise ValueError(msg)
    return by_call, by_session


def _per_turn_columns(turns: list[CallRecord | None]) -> dict[str, list[Any]]:
    """The per-assistant-message columns, ``None`` for a context turn."""
    columns: dict[str, list[Any]] = {
        CALL_IDS: [],
        FINISH_REASONS: [],
        COMPLETION_TOKEN_IDS: [],
        COMPLETION_LOGPROBS: [],
    }
    for call in turns:
        values = (
            (None,) * len(columns)
            if call is None
            else (
                call.call_id,
                call.finish_reason,
                call.completion_token_ids,
                call.completion_logprobs,
            )
        )
        for column, value in zip(columns.values(), values, strict=True):
            column.append(value)
    return columns


def write_snapshot(
    directory: str | Path,
    sessions: Iterable[Session],
    created_by: str,
    query: dict[str, Any],
    scores: Iterable[ScoreRecord] = (),
    parent_snapshot_id: str | None = None,
    label_score: str | None = None,
) -> SnapshotManifest:
    """Write a snapshot: ``segments.parquet`` from :func:`snapshot_table`, then its ``manifest.json``.

    Both files are encoded before either is written, so a rejected snapshot
    writes nothing.

    :param directory: The snapshot's directory; created if missing.
    :param sessions: Stitched sessions to export.
    :param created_by: User id of whoever took the snapshot.
    :param query: The trace-query filter that selected the calls.
    :param scores: Scores to join, see :func:`snapshot_table`.
    :param parent_snapshot_id: The snapshot this one grew from, if any.
    :param label_score: The score name used as a filter, if any.
    :return: The manifest written.
    :raises ValueError: If ``label_score`` names no score in the rows, or the
        calls come from more than one organisation.
    :raises TypeError: If ``query`` is not JSON.
    """
    segments, excluded = snapshot_segments(sessions)
    rows = _segment_rows(segments, scores)
    score_names = sorted({entry["name"] for row in rows for entry in row[SCORES] or []})
    _check_snapshot(segments, score_names, label_score)
    table = pa.Table.from_pylist(rows, schema=SNAPSHOT_SCHEMA)
    buffer = pa.BufferOutputStream()
    pq.write_table(table, buffer, compression="zstd")
    segments_bytes = buffer.getvalue()
    manifest = _manifest(
        segments,
        score_names,
        excluded,
        SnapshotFile(
            SNAPSHOT_SEGMENTS_FILE,
            table.num_rows,
            hashlib.sha256(segments_bytes).hexdigest(),
        ),
        created_by,
        query,
        parent_snapshot_id,
        label_score,
    )
    manifest_text = manifest_json(manifest)
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    (root / SNAPSHOT_SEGMENTS_FILE).write_bytes(segments_bytes)
    (root / MANIFEST_FILE).write_text(manifest_text)
    return manifest


def _appends(previous: CallRecord, call: CallRecord) -> bool:
    """Whether ``call`` continues ``previous``'s segment: same policy and tools, ``previous``'s reply next, no unrecorded reply after it."""
    if (call.model_version, call.tools or None) != (
        previous.model_version,
        previous.tools or None,
    ):
        return False
    history = call.messages or []
    reply = len(previous.messages or [])
    # Stitching checks this slot only when ``previous`` has a response hash.
    if [message_hash(message) for message in history[reply : reply + 1]] != [
        message_hash(previous.response_message or {})
    ]:
        return False
    return not any(_is_assistant(message) for message in history[reply + 1 :])


def _segment_messages(
    calls: Sequence[CallRecord],
) -> tuple[list[dict[str, Any]], list[CallRecord | None]]:
    """A segment's messages, and per assistant message the call that produced it, ``None`` for context."""
    messages = [*(calls[-1].messages or []), calls[-1].response_message or {}]
    producers: dict[int, CallRecord] = {}
    for call in calls:
        # Each call's reply sits right after its request in the last call's history.
        position = len(call.messages or [])
        messages[position] = call.response_message or {}
        producers[position] = call
    turns = [producers.get(i) for i, m in enumerate(messages) if _is_assistant(m)]
    return [
        _completion_message(message) if _is_assistant(message) else message
        for message in messages
    ], turns


def _is_assistant(message: dict[str, Any]) -> bool:
    """Whether ``message`` is an assistant turn."""
    return message.get("role") == "assistant"


def _score_entry(score: ScoreRecord, turn: int | None) -> dict[str, Any]:
    """One ``scores`` entry, on assistant message ``turn`` or the whole row."""
    return {
        "name": score.name,
        "value": score.value,
        "source": score.source,
        "label": score.label,
        "comment": score.comment,
        "turn": turn,
    }


def _check_snapshot(
    segments: list[SnapshotSegment],
    score_names: list[str],
    label_score: str | None,
) -> None:
    """Raise unless ``label_score`` is among ``score_names`` and the calls come from at most one organisation."""
    if label_score is not None and label_score not in score_names:
        msg = f"label_score {label_score!r} is not among the scores {score_names}."
        raise ValueError(msg)
    organizations = sorted(
        {
            call.organization_id
            for segment in segments
            for call in segment.calls
            if call.organization_id is not None
        }
    )
    if len(organizations) > 1:
        msg = f"a snapshot holds one organisation's calls, got {organizations}."
        raise ValueError(msg)


def _manifest(
    segments: list[SnapshotSegment],
    score_names: list[str],
    excluded: ExcludedCounts,
    segments_file: SnapshotFile,
    created_by: str,
    query: dict[str, Any],
    parent_snapshot_id: str | None,
    label_score: str | None,
) -> SnapshotManifest:
    """The manifest of a snapshot holding one row per segment, whose rows carry ``score_names``."""
    firsts = [segment.calls[0] for segment in segments]
    versions = Counter(
        (call.model_version, call.base_model, call.checkpoint_id) for call in firsts
    )
    calls = [call for segment in segments for call in segment.calls]
    organizations = {
        call.organization_id for call in calls if call.organization_id is not None
    }
    received = [call.received_at for call in firsts]
    return SnapshotManifest(
        snapshot_id=new_call_id(),
        created_at=datetime.now(timezone.utc),
        created_by=created_by,
        row_count=len(segments),
        session_count=len({segment.session_key for segment in segments}),
        model_versions=[
            ModelVersionCount(version, base, checkpoint, count)
            for (version, base, checkpoint), count in sorted(
                versions.items(), key=lambda item: (item[0][0], item[0][2] or "")
            )
        ],
        time_range=TimeRange(min(received), max(received)) if received else None,
        filter=query,
        parent_snapshot_id=parent_snapshot_id,
        score_names=score_names,
        label_score=label_score,
        excluded=excluded,
        organization_id=next(iter(organizations), None),
        deployment_ids=sorted({call.deployment_id for call in calls}),
        turn_count=sum(len(segment.turns) for segment in segments),
        files=[segments_file],
    )


def _completion_message(message: dict[str, Any]) -> dict[str, Any]:
    """An assistant message with reasoning under ``reasoning_content`` and tool-call arguments as JSON text."""
    reasoning = {key: message[key] for key in REASONING_KEYS if key in message}
    if len(reasoning) > 1 and len(set(map(json.dumps, reasoning.values()))) > 1:
        msg = (
            f"An assistant message sets {sorted(reasoning)} to different "
            "reasoning; keep one."
        )
        raise ValueError(msg)
    kept = {key: value for key, value in message.items() if key not in REASONING_KEYS}
    if reasoning:
        kept["reasoning_content"] = next(iter(reasoning.values()))
    if kept.get("tool_calls"):
        kept["tool_calls"] = [
            _tool_call_with_text_arguments(call) for call in kept["tool_calls"]
        ]
    return kept


def _tool_call_with_text_arguments(call: dict[str, Any]) -> dict[str, Any]:
    """A tool call whose ``function.arguments`` is JSON text."""
    function = call.get("function")
    if (
        isinstance(function, dict)
        and "arguments" in function
        and not isinstance(function["arguments"], str)
    ):
        arguments = json.dumps(function["arguments"], ensure_ascii=False)
        return {**call, "function": {**function, "arguments": arguments}}
    return call

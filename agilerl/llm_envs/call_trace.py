# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Call records as Parquet, ``traces`` dataset snapshots, and snapshots as SFT rows on agent traces."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from agilerl.arena.traces_format import (
    ANSWER,
    BASE_MODEL,
    CHECKPOINT_ID,
    COMPLETION,
    MANIFEST_FILE,
    MODEL_VERSION,
    PROMPT,
    REASONING_KEYS,
    RECEIVED_AT,
    SCORES,
    SEGMENT_INDEX,
    SESSION_ID,
    SNAPSHOT_CALLS_FILE,
    ExcludedCounts,
    ModelVersionCount,
    SnapshotFile,
    SnapshotManifest,
    TimeRange,
    write_manifest,
)
from agilerl.llm_envs.call_record import (
    CallRecord,
    ScoreRecord,
    from_row,
    new_call_id,
    to_row,
)
from agilerl.llm_envs.call_sessions import Session

__all__ = [
    "CALL_RECORD_SCHEMA",
    "FEEDBACK_SCHEMA",
    "SNAPSHOT_SCHEMA",
    "calls_table",
    "read_snapshot",
    "sft_rows",
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

FEEDBACK_SCHEMA = pa.schema(
    [
        ("feedback_id", pa.string()),
        ("call_id", pa.string()),
        ("session_key", pa.string()),
        ("score", pa.float64()),
        ("label", pa.string()),
        ("comment", pa.string()),
        ("source", pa.string()),
        ("user_id", pa.string()),
        ("created_at", pa.timestamp("us", tz="UTC")),
    ]
)
"""Arrow schema of a ``feedback/`` Parquet file, one row per :class:`FeedbackRecord`."""

SNAPSHOT_SCHEMA = pa.schema(
    [
        # The recorder's columns a snapshot renames to the traces format's.
        pa.field(
            {"messages": PROMPT, "response_message": COMPLETION}.get(
                item.name, item.name
            ),
            item.type,
        )
        for item in CALL_RECORD_SCHEMA
    ]
    + [
        pa.field(SESSION_ID, pa.string()),
        pa.field(SEGMENT_INDEX, pa.int32()),
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
                    ]
                )
            ),
        ),
        pa.field(ANSWER, pa.string()),
    ]
)
"""Arrow schema of a snapshot's ``calls.parquet``: the traces format's columns beside the recorder's."""


def calls_table(records: Iterable[CallRecord]) -> pa.Table:
    """Call records as an Arrow table of :data:`CALL_RECORD_SCHEMA`.

    :param records: Call records.
    :return: The table, one row per record.
    """
    return pa.Table.from_pylist(
        [to_row(record) for record in records], schema=CALL_RECORD_SCHEMA
    )


def snapshot_table(
    sessions: Iterable[Session], scores: Iterable[ScoreRecord] = ()
) -> tuple[pa.Table, ExcludedCounts]:
    """Stitched sessions as traces rows of :data:`SNAPSHOT_SCHEMA`, with their scores joined.

    Calls with no content and calls that ended in ``error`` or ``cancelled``
    are left out and counted. Assistant messages in the prompt and the
    completion keep reasoning under ``reasoning_content`` and tool-call
    arguments as JSON text.

    :param sessions: Sessions from :func:`~agilerl.llm_envs.call_sessions.stitch_sessions`.
    :param scores: Scores to join by ``call_id``, or by ``session_key`` onto
        every row of that session.
    :return: The table, one row per kept call in session then call order, and
        the counts left out.
    :raises ValueError: If a score does not set exactly one of ``call_id``
        and ``session_key``.
    """
    by_call: dict[str, list[dict[str, Any]]] = {}
    by_session: dict[str, list[dict[str, Any]]] = {}
    for score in scores:
        entry = {
            "name": score.name,
            "value": score.value,
            "source": score.source,
            "label": score.label,
            "comment": score.comment,
        }
        if score.call_id is not None and score.session_key is None:
            by_call.setdefault(score.call_id, []).append(entry)
        elif score.session_key is not None and score.call_id is None:
            by_session.setdefault(score.session_key, []).append(entry)
        else:
            msg = (
                f"score {score.name!r} must set exactly one of call_id and "
                "session_key, to attach it to one call or one session."
            )
            raise ValueError(msg)
    rows: list[dict[str, Any]] = []
    no_content = error_or_cancelled = 0
    for session in sessions:
        for index, segment in enumerate(session.segments):
            for call in segment:
                if call.finish_reason in ("error", "cancelled"):
                    error_or_cancelled += 1
                    continue
                if call.messages is None or call.response_message is None:
                    no_content += 1
                    continue
                row = to_row(call)
                del row["messages"], row["response_message"]
                prompt = [
                    _completion_message(message)
                    if message.get("role") == "assistant"
                    else message
                    for message in call.messages
                ]
                joined = [
                    *by_call.get(call.call_id, []),
                    *by_session.get(session.key, []),
                ]
                rows.append(
                    {
                        **row,
                        PROMPT: json.dumps(prompt, ensure_ascii=False),
                        COMPLETION: json.dumps(
                            [_completion_message(call.response_message)],
                            ensure_ascii=False,
                        ),
                        SESSION_ID: session.key,
                        SEGMENT_INDEX: index,
                        SCORES: joined or None,
                        ANSWER: None,
                    }
                )
    table = pa.Table.from_pylist(rows, schema=SNAPSHOT_SCHEMA)
    return table, ExcludedCounts(no_content, error_or_cancelled)


def write_snapshot(
    directory: str | Path,
    sessions: Iterable[Session],
    created_by: str,
    query: dict[str, Any],
    scores: Iterable[ScoreRecord] = (),
    parent_snapshot_id: str | None = None,
    label_score: str | None = None,
) -> SnapshotManifest:
    """Write a snapshot: ``calls.parquet`` from :func:`snapshot_table`, then its ``manifest.json``.

    :param directory: The snapshot's directory; created if missing.
    :param sessions: Stitched sessions to export.
    :param created_by: User id of whoever took the snapshot.
    :param query: The trace-query filter that selected the calls.
    :param scores: Scores to join, see :func:`snapshot_table`.
    :param parent_snapshot_id: The snapshot this one grew from, if any.
    :param label_score: The score name used as a filter, if any.
    :return: The manifest written.
    :raises ValueError: If ``label_score`` names no score in the rows, or the
        rows come from more than one organisation.
    """
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    table, excluded = snapshot_table(sessions, scores)
    calls_path = root / SNAPSHOT_CALLS_FILE
    pq.write_table(table, calls_path, compression="zstd")
    manifest = _manifest(
        table.to_pylist(),
        excluded,
        SnapshotFile(
            SNAPSHOT_CALLS_FILE,
            table.num_rows,
            hashlib.sha256(calls_path.read_bytes()).hexdigest(),
        ),
        created_by,
        query,
        parent_snapshot_id,
        label_score,
    )
    write_manifest(root / MANIFEST_FILE, manifest)
    return manifest


def read_snapshot(path: str | Path) -> list[Session]:
    """The sessions a snapshot's ``calls.parquet`` holds, as it stitched them.

    :param path: The snapshot's ``calls.parquet``.
    :return: Sessions in first-row order, segments by ``segment_index``, calls
        by ``received_at`` then ``call_id``.
    """
    segments: dict[str, dict[int, list[CallRecord]]] = {}
    for row in pq.read_table(path).to_pylist():
        (completion,) = json.loads(row.pop(COMPLETION))
        record = from_row(
            {**row, "messages": row.pop(PROMPT), "response_message": completion}
        )
        session = segments.setdefault(row[SESSION_ID], {})
        session.setdefault(row[SEGMENT_INDEX], []).append(record)
    return [
        Session(
            key,
            [
                sorted(by_index[index], key=lambda c: (c.received_at, c.call_id))
                for index in sorted(by_index)
            ],
        )
        for key, by_index in segments.items()
    ]


def _manifest(
    rows: list[dict[str, Any]],
    excluded: ExcludedCounts,
    calls_file: SnapshotFile,
    created_by: str,
    query: dict[str, Any],
    parent_snapshot_id: str | None,
    label_score: str | None,
) -> SnapshotManifest:
    """The manifest of a snapshot holding ``rows``."""
    versions = Counter(
        (row[MODEL_VERSION], row[BASE_MODEL], row[CHECKPOINT_ID]) for row in rows
    )
    score_names = sorted({entry["name"] for row in rows for entry in row[SCORES] or []})
    if label_score is not None and label_score not in score_names:
        msg = f"label_score {label_score!r} is not among the scores {score_names}."
        raise ValueError(msg)
    organizations = {row["organization_id"] for row in rows} - {None}
    if len(organizations) > 1:
        msg = f"a snapshot holds one organisation's calls, got {sorted(organizations)}."
        raise ValueError(msg)
    received = [row[RECEIVED_AT] for row in rows]
    return SnapshotManifest(
        snapshot_id=new_call_id(),
        created_at=datetime.now(timezone.utc),
        created_by=created_by,
        row_count=len(rows),
        session_count=len({row[SESSION_ID] for row in rows}),
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
        deployment_ids=sorted({row["deployment_id"] for row in rows}),
        files=[calls_file],
    )


def _completion_message(message: dict[str, Any]) -> dict[str, Any]:
    """An assistant message with reasoning under ``reasoning_content`` and tool-call arguments as JSON text."""
    reasoning = {key: message[key] for key in REASONING_KEYS if key in message}
    if len(set(map(json.dumps, reasoning.values()))) > 1:
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


def sft_rows(sessions: Iterable[Session]) -> list[dict[str, Any]]:
    """One SFT ``messages`` row per segment, the agent-trace form :class:`DatasetEnv` trains on.

    A segment's transcript is its last call's messages and response. Each
    earlier call's reply in that transcript is the recorded
    ``response_message``, so reasoning a harness dropped when resending is kept.
    Trailing calls with no response (errors, cancellations) are left out, and
    a segment with none left is skipped.

    :param sessions: Stitched sessions.
    :return: Rows of ``messages`` and ``tools`` as JSON text, plus
        ``session_id`` and ``segment_index``.
    :raises ValueError: If a call was captured without content.
    """
    rows: list[dict[str, Any]] = []
    for session in sessions:
        for index, segment in enumerate(session.segments):
            answered = _through_last_response(segment)
            if not answered:
                continue
            messages = _segment_transcript(answered)
            tools = answered[-1].tools
            rows.append(
                {
                    "messages": json.dumps(messages, ensure_ascii=False),
                    "tools": None if not tools else json.dumps(tools),
                    SESSION_ID: session.key,
                    SEGMENT_INDEX: index,
                }
            )
    return rows


def _through_last_response(segment: Sequence[CallRecord]) -> list[CallRecord]:
    """``segment`` through its last call with a response, checking every call has content."""
    for call in segment:
        if call.messages is None:
            msg = (
                f"call {call.call_id} has no messages; SFT needs calls captured "
                "at the 'full' level."
            )
            raise ValueError(msg)
    last = max(
        (i for i, call in enumerate(segment) if call.response_message is not None),
        default=-1,
    )
    return list(segment[: last + 1])


def _segment_transcript(segment: Sequence[CallRecord]) -> list[dict[str, Any]]:
    """The last call's messages and response, with earlier replies as recorded."""
    messages = [dict(message) for message in segment[-1].messages or []]
    for call in segment:
        if call.response_message is None:
            continue
        position = len(call.messages or [])
        # Stitching puts an earlier call's reply at this position of the last prompt.
        if position < len(messages):
            messages[position] = call.response_message
        else:
            messages.append(call.response_message)
    return messages

# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""The trace call record: one model call, as the inference pod and training both write it.

Standard library only, so the inference pod imports it without the LLM extras.
:func:`prompt_prefix_hashes` and :func:`model_version` are the values both
sides must compute the same way; :func:`to_row` / :func:`from_row` convert to
and from the Parquet row of :mod:`agilerl.llm_envs.call_trace`.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, get_args

from agilerl.arena.traces_format import REASONING_KEYS

__all__ = [
    "CAPTURE_OFF",
    "FINISH_REASONS",
    "SCHEMA_VERSION",
    "CallOrigin",
    "CallRecord",
    "FeedbackRecord",
    "FinishReason",
    "ScoreRecord",
    "chat_template_hash",
    "directory_digest",
    "feedback_score",
    "from_row",
    "message_hash",
    "model_version",
    "new_call_id",
    "prompt_prefix_hashes",
    "to_row",
]

SCHEMA_VERSION = 1

# The ``X-AgileRL-Capture`` value that skips recording a call.
CAPTURE_OFF = "off"

FinishReason = Literal["stop", "length", "tool_calls", "error", "cancelled"]

FINISH_REASONS: tuple[FinishReason, ...] = get_args(FinishReason)

CALL_JSON_COLUMNS = (
    "messages",
    "tools",
    "tool_choice",
    "response_message",
    "sampling",
    "metadata",
)


@dataclass(frozen=True)
class CallOrigin:
    """The call record fields fixed by who serves the calls, not by each call."""

    organization_id: str | None
    deployment_id: str
    instance_id: str
    user_id: str | None
    base_model: str
    chat_template_hash: str


@dataclass(frozen=True)
class CallRecord:
    """One model call and its response, the unit the trace store holds.

    Field names are the Parquet column names. Content and token fields are
    ``None`` for a call captured at the ``metadata`` level.
    ``organization_id`` and ``user_id`` are ``None`` when the server runs
    without an auth server.
    """

    # Identity
    call_id: str
    request_id: str
    organization_id: str | None
    deployment_id: str
    instance_id: str
    user_id: str | None
    end_user: str | None
    client_session_id: str | None
    # Model
    base_model: str
    checkpoint_id: str | None
    model_version: str
    chat_template_hash: str
    # Timing and size
    received_at: datetime
    total_ms: int
    finish_reason: FinishReason
    stream: bool
    prompt_prefix_hashes: list[str]
    response_prefix_hash: str | None
    queue_ms: int | None = None
    first_token_ms: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    error: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    sampling: dict[str, Any] = field(default_factory=dict)
    # Content
    messages: list[dict[str, Any]] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    response_message: dict[str, Any] | None = None
    raw_completion: str | None = None
    # Tokens
    prompt_token_ids: list[int] | None = None
    completion_token_ids: list[int] | None = None
    completion_logprobs: list[float] | None = None
    schema_version: int = SCHEMA_VERSION


@dataclass(frozen=True)
class FeedbackRecord:
    """A score, label or comment attached to one call or one session."""

    feedback_id: str
    call_id: str | None
    session_key: str | None
    score: float | None
    label: str | None
    comment: str | None
    source: str
    user_id: str | None
    created_at: datetime


@dataclass(frozen=True)
class ScoreRecord:
    """A named score on one call (``call_id``) or on every call of a session (``session_key``)."""

    call_id: str | None
    session_key: str | None
    name: str
    value: float | None
    source: str
    label: str | None = None
    comment: str | None = None


def feedback_score(record: FeedbackRecord) -> ScoreRecord:
    """A feedback row as a score named ``feedback``.

    :param record: The feedback row.
    :return: The score, attached where the feedback was.
    """
    return ScoreRecord(
        call_id=record.call_id,
        session_key=record.session_key,
        name="feedback",
        value=record.score,
        source=record.source,
        label=record.label,
        comment=record.comment,
    )


def to_row(record: CallRecord) -> dict[str, Any]:
    """The Parquet row of ``record``: JSON-valued columns as JSON text.

    :param record: The call record.
    :return: Column name to value.
    """
    row = dataclasses.asdict(record)
    for name in CALL_JSON_COLUMNS:
        if row[name] is not None:
            row[name] = json.dumps(row[name], ensure_ascii=False)
    return row


def from_row(row: dict[str, Any]) -> CallRecord:
    """The call record of a Parquet row from :func:`to_row`; extra columns are ignored.

    :param row: Column name to value.
    :return: The call record.
    :raises ValueError: If the row's ``schema_version`` is not :data:`SCHEMA_VERSION`,
        or its ``finish_reason`` is not one of :data:`FINISH_REASONS`.
    """
    if row["schema_version"] != SCHEMA_VERSION:
        msg = (
            f"call record schema_version {row['schema_version']} is not "
            f"{SCHEMA_VERSION}, the version this reader understands."
        )
        raise ValueError(msg)
    if row["finish_reason"] not in FINISH_REASONS:
        msg = (
            f"call {row['call_id']} has finish_reason {row['finish_reason']!r}; "
            f"expected one of {', '.join(FINISH_REASONS)}."
        )
        raise ValueError(msg)
    names = {item.name for item in dataclasses.fields(CallRecord)}
    values = {name: row[name] for name in names if name in row}
    for name in CALL_JSON_COLUMNS:
        if isinstance(values.get(name), str):
            values[name] = json.loads(values[name])
    return CallRecord(**values)


def new_call_id(now_ms: int | None = None) -> str:
    """A new ULID: 48 bits of millisecond time then 80 random bits, Crockford base32.

    :param now_ms: Unix time in milliseconds; the current time when ``None``.
    :return: The 26-character id, which sorts by time.
    """
    stamp = time.time_ns() // 1_000_000 if now_ms is None else now_ms
    value = (stamp << 80) | int.from_bytes(os.urandom(10), "big")
    crockford = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
    return "".join(crockford[(value >> shift) & 31] for shift in range(125, -1, -5))


def message_hash(message: dict[str, Any]) -> str:
    """Hash of one chat message as it would be resent, see :func:`prompt_prefix_hashes`.

    :param message: An OpenAI-style chat message.
    :return: Hex SHA-256 of the message's canonical JSON.
    """
    return _sha256(_json_text(_canonical_message(message)))


def prompt_prefix_hashes(
    messages: list[dict[str, Any]], start: str | None = None
) -> list[str]:
    """Rolling hashes of a request's messages, one per message.

    Entry ``i`` hashes messages ``0..i``, so two requests share entry ``i``
    exactly when their first ``i + 1`` messages match. A message is compared
    as harnesses resend it: ``None`` fields, empty lists and reasoning are
    dropped, and content made only of text parts is joined into one string.

    :param messages: The request's messages, in order.
    :param start: The hash to extend, so a response can continue a request's
        hashes; ``None`` starts a new chain.
    :return: One hex SHA-256 per message.
    """
    hashes: list[str] = []
    previous = "" if start is None else start
    for message in messages:
        previous = _sha256(previous + message_hash(message))
        hashes.append(previous)
    return hashes


def chat_template_hash(chat_template: str) -> str:
    """Hex SHA-256 of a chat template's source.

    :param chat_template: The Jinja source the tokenizer renders prompts with.
    :return: The hash.
    """
    return _sha256(chat_template)


def directory_digest(path: str | Path) -> str:
    """Hex SHA-256 over every file under ``path``, by sorted relative path.

    :param path: A saved adapter directory.
    :return: The digest.
    """
    root = Path(path)
    digest = hashlib.sha256()
    for file in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(file.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(file.read_bytes())
    return digest.hexdigest()


def model_version(
    base_model: str, adapter_digest: str | None, chat_template: str
) -> str:
    """The policy id two calls share exactly when they were sampled from the same policy.

    :param base_model: Hub id of the base model.
    :param adapter_digest: :func:`directory_digest` of the served adapter, or
        ``None`` when the base model is served alone.
    :param chat_template: The chat template's source.
    :return: Hex SHA-256 of the three.
    """
    return _sha256(
        _json_text([base_model, adapter_digest, chat_template_hash(chat_template)])
    )


def _canonical_message(message: dict[str, Any]) -> dict[str, Any]:
    """``message`` as compared for hashing, see :func:`prompt_prefix_hashes`."""
    kept = {
        key: _without_empty(value)
        for key, value in message.items()
        if key not in REASONING_KEYS and value is not None and value != []
    }
    content = kept.get("content")
    if isinstance(content, list) and all(
        isinstance(part, dict) and part.get("type") == "text" for part in content
    ):
        kept["content"] = "".join(part.get("text", "") for part in content)
    return kept


def _without_empty(value: Any) -> Any:  # noqa: ANN401 -- any JSON value
    """``value`` without ``None`` or empty-list dict entries at any depth."""
    if isinstance(value, dict):
        return {
            key: _without_empty(item)
            for key, item in value.items()
            if item is not None and item != []
        }
    if isinstance(value, list):
        return [_without_empty(item) for item in value]
    return value


def _json_text(value: Any) -> str:  # noqa: ANN401 -- any JSON value
    """Canonical JSON: sorted keys, no spaces, non-ASCII kept."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(text: str) -> str:
    """Hex SHA-256 of ``text`` as UTF-8."""
    return hashlib.sha256(text.encode()).hexdigest()

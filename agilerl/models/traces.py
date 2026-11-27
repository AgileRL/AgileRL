# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""The ``traces`` dataset format: OpenAI Chat Completions messages in TRL ``messages`` rows (torch-free).

A row is a segment: one append-only stretch of a session in ``messages``,
trained on every assistant message without ``weight: 0``.
``tools`` holds the function schemas the model could call. See the Arena
docs page on traces datasets.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from datetime import datetime
from itertools import pairwise
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from typing_extensions import Self

from agilerl.traces_format import (
    JSON_TEXT_COLUMNS,
    REASONING_KEYS,
    WEIGHT,
)

TRACES_DATASET_FORMAT = "traces"
"""``environment.dataset_format`` and ``arena datasets create --category`` value."""

MAX_REPORTED_ERRORS = 20


class ContentPart(BaseModel):
    """One part of a multi-part message ``content``, such as ``{"type": "text", "text": ...}``."""

    model_config = ConfigDict(extra="ignore")

    type: str = Field(min_length=1)


MessageContent = str | list[ContentPart]
"""Content of a system, developer or user message; assistant and tool content is a string."""


class ToolCallFunction(BaseModel):
    """The function an assistant tool call names, with its arguments as a JSON object string."""

    model_config = ConfigDict(extra="ignore")

    name: str = Field(min_length=1)
    arguments: str

    @field_validator("arguments")
    @classmethod
    def _arguments_are_a_json_object(cls, arguments: str) -> str:
        try:
            decoded = json.loads(arguments)
        except json.JSONDecodeError as error:
            msg = f"function.arguments is not valid JSON ({error})."
            raise ValueError(msg) from error
        if not isinstance(decoded, dict):
            msg = (
                "function.arguments must encode a JSON object, got "
                f"{type(decoded).__name__}."
            )
            raise ValueError(msg)
        return arguments


class ToolCall(BaseModel):
    """One call in an assistant message's ``tool_calls``."""

    model_config = ConfigDict(extra="ignore")

    id: str = Field(min_length=1)
    type: Literal["function"] = "function"
    function: ToolCallFunction


class _UnweightedMessage(BaseModel):
    """A message that is never a training target, so it carries no ``weight``."""

    model_config = ConfigDict(extra="ignore")

    @model_validator(mode="before")
    @classmethod
    def _no_weight(cls, data: Any) -> Any:  # noqa: ANN401 -- raw message
        if isinstance(data, Mapping) and data.get(WEIGHT) is not None:
            msg = (
                f"A {data.get('role')} message has weight; only assistant "
                "messages are weighted."
            )
            raise ValueError(msg)
        return data


class SystemMessage(_UnweightedMessage):
    """A ``system`` or ``developer`` instruction message."""

    role: Literal["system", "developer"]
    content: MessageContent
    name: str | None = None


class UserMessage(_UnweightedMessage):
    """A ``user`` message."""

    role: Literal["user"]
    content: MessageContent
    name: str | None = None


class AssistantMessage(BaseModel):
    """An ``assistant`` turn: content, tool calls, or both, with optional reasoning.

    ``weight`` 0 marks the turn as context, never trained; unset means 1.
    """

    model_config = ConfigDict(extra="ignore")

    role: Literal["assistant"]
    content: str | None = None
    reasoning_content: str | None = None
    reasoning: str | None = None
    thinking: str | None = None
    tool_calls: list[ToolCall] | None = None
    name: str | None = None
    weight: Literal[0, 1] | None = None

    @field_validator("weight", mode="before")
    @classmethod
    def _weight_is_zero_or_one(cls, weight: object) -> object:
        if weight is not None and (type(weight) is not int or weight not in (0, 1)):
            msg = f"weight must be 0 or 1, got {weight!r}."
            raise ValueError(msg)
        return weight

    @property
    def is_target(self) -> bool:
        """Whether the turn is a training target: ``weight`` is not 0."""
        return self.weight != 0

    @model_validator(mode="after")
    def _says_something(self) -> Self:
        if not self.content and not self.tool_calls:
            msg = "An assistant message needs content or tool_calls."
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _one_reasoning(self) -> Self:
        present = {
            key: getattr(self, key) for key in REASONING_KEYS if getattr(self, key)
        }
        if len(set(present.values())) > 1:
            msg = (
                f"An assistant message sets {sorted(present)} to different "
                "reasoning; keep one (reasoning_content)."
            )
            raise ValueError(msg)
        return self

    @property
    def reasoning_text(self) -> str | None:
        """The turn's reasoning under whichever of :data:`REASONING_KEYS` holds it."""
        return next(
            (getattr(self, key) for key in REASONING_KEYS if getattr(self, key)), None
        )


class ToolMessage(_UnweightedMessage):
    """A ``tool`` result answering one earlier tool call."""

    role: Literal["tool"]
    tool_call_id: str = Field(min_length=1)
    content: str | None = None
    name: str | None = None


Message = Annotated[
    SystemMessage | UserMessage | AssistantMessage | ToolMessage,
    Field(discriminator="role"),
]


class FunctionDefinition(BaseModel):
    """A callable function: name, description and JSON Schema parameters."""

    model_config = ConfigDict(extra="ignore")

    name: str = Field(min_length=1)
    description: str | None = None
    parameters: dict[str, Any] | None = None
    strict: bool | None = None


class ToolDefinition(BaseModel):
    """One OpenAI tool definition, ``{"type": "function", "function": {...}}``."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["function"] = "function"
    function: FunctionDefinition


class TraceScore(BaseModel):
    """A named score on a row: user feedback, or a judge id plus rubric hash.

    ``turn`` is a 0-based index into a segment's assistant messages, for a
    score on one turn; unset scores the whole row.
    """

    model_config = ConfigDict(extra="ignore")

    name: str = Field(min_length=1)
    value: float | None = None
    source: str | None = None
    label: str | None = None
    comment: str | None = None
    turn: int | None = Field(default=None, ge=0)


class TracesRowBase(BaseModel):
    """Columns both row forms share; unknown extra columns are ignored."""

    model_config = ConfigDict(extra="ignore")

    tools: list[ToolDefinition] | None = None
    answer: str | None = None
    session_id: str | None = None
    segment_index: int | None = None
    model_version: str | None = None
    checkpoint_id: str | None = None
    base_model: str | None = None
    received_at: datetime | None = None
    scores: list[TraceScore] | None = None
    metadata: dict[str, str] | None = None

    @model_validator(mode="before")
    @classmethod
    def _decode_json_columns(cls, data: Any) -> Any:  # noqa: ANN401 -- raw row
        if not isinstance(data, Mapping):
            return data
        decoded = dict(data)
        for column in JSON_TEXT_COLUMNS:
            value = decoded.get(column)
            if isinstance(value, str):
                try:
                    decoded[column] = json.loads(value)
                except json.JSONDecodeError as error:
                    msg = f"{column} is a string but not valid JSON ({error})."
                    raise ValueError(msg) from error
        return decoded

    def _tool_names(self) -> set[str] | None:
        if self.tools is None:
            return None
        return {tool.function.name for tool in self.tools}


def _check_conversation(
    messages: list[Any], tool_names: set[str] | None, label: str
) -> None:
    """Raise unless every tool result answers an earlier call and every call names a tool."""
    open_calls: set[str] = set()
    for index, message in enumerate(messages):
        if isinstance(message, AssistantMessage):
            for call in message.tool_calls or []:
                if tool_names is not None and call.function.name not in tool_names:
                    msg = (
                        f"{label}[{index}] calls {call.function.name!r}, which is "
                        f"not in tools ({sorted(tool_names)})."
                    )
                    raise ValueError(msg)
                open_calls.add(call.id)
        elif isinstance(message, ToolMessage):
            if message.tool_call_id not in open_calls:
                msg = (
                    f"{label}[{index}] answers tool_call_id "
                    f"{message.tool_call_id!r}, which no earlier assistant "
                    "message called."
                )
                raise ValueError(msg)


class TracesSegmentRow(TracesRowBase):
    """One append-only stretch of a session in ``messages``: a stored traces row.

    ``call_ids``, ``finish_reasons``, ``completion_token_ids`` and
    ``completion_logprobs`` hold one entry per assistant message.
    An assistant message with ``weight: 0`` is context, not a training
    target; ``call_ids`` is lineage only.
    """

    messages: list[Message] = Field(min_length=2)
    call_ids: list[str | None] | None = None
    finish_reasons: list[str | None] | None = None
    completion_token_ids: list[list[int] | None] | None = None
    completion_logprobs: list[list[float] | None] | None = None

    @model_validator(mode="after")
    def _check_turns(self) -> Self:
        if isinstance(self.messages[0], AssistantMessage):
            msg = "messages starts with an assistant turn; it needs a prompt first."
            raise ValueError(msg)
        turns = [
            index
            for index, message in enumerate(self.messages)
            if isinstance(message, AssistantMessage)
        ]
        if not turns:
            msg = "messages has no assistant turn to train on."
            raise ValueError(msg)
        _check_conversation(self.messages, self._tool_names(), "messages")
        for index, message in enumerate(self.messages[: turns[-1]]):
            if isinstance(message, ToolMessage) and message.content is None:
                msg = (
                    f"messages[{index}] is a tool result without content before "
                    "an assistant turn; training renders its text."
                )
                raise ValueError(msg)
        targets = [self.messages[at].is_target for at in turns]
        if not any(targets):
            msg = "every assistant message has weight 0, so none is a training target."
            raise ValueError(msg)
        self._check_per_turn_columns(targets)
        _check_calls_answered(self.messages, turns, targets)
        for score in self.scores or []:
            if score.turn is not None and score.turn >= len(turns):
                msg = (
                    f"score {score.name!r} is on turn {score.turn}; messages has "
                    f"{len(turns)} assistant messages."
                )
                raise ValueError(msg)
        return self

    def _check_per_turn_columns(self, targets: list[bool]) -> None:
        """Raise unless the per-turn lists have one entry per assistant message and none is set on a context turn.

        :param targets: Per assistant message, whether it is a training target.
        """
        count = len(targets)
        for column in (
            "call_ids",
            "finish_reasons",
            "completion_token_ids",
            "completion_logprobs",
        ):
            values = getattr(self, column)
            if values is not None and len(values) != count:
                msg = (
                    f"{column} has {len(values)} entries; messages has "
                    f"{count} assistant messages."
                )
                raise ValueError(msg)
        for position, (ids, logprobs) in enumerate(
            zip(
                self.completion_token_ids or [None] * count,
                self.completion_logprobs or [None] * count,
                strict=True,
            )
        ):
            if ids is not None and logprobs is not None and len(ids) != len(logprobs):
                msg = (
                    f"turn {position} has {len(ids)} completion_token_ids but "
                    f"{len(logprobs)} completion_logprobs."
                )
                raise ValueError(msg)
            if not targets[position] and (ids is not None or logprobs is not None):
                msg = (
                    f"turn {position} is context (weight 0), so its "
                    "completion_token_ids and completion_logprobs must be null."
                )
                raise ValueError(msg)


def _check_calls_answered(
    messages: list[Any], turns: list[int], targets: list[bool]
) -> None:
    """Raise unless each assistant message's tool calls are answered before the next one.

    The final assistant message's calls may be unanswered. Two assistant
    messages in a row are allowed only when the later one is context
    (``weight: 0``): a target's prompt must not end with an assistant turn.

    :param targets: Per assistant message, whether it is a training target.
    """
    for position, (previous, turn) in enumerate(pairwise(turns), start=1):
        if turn == previous + 1 and targets[position]:
            msg = (
                f"messages[{previous}] and messages[{turn}] are back-to-back "
                f"assistant messages, and messages[{turn}] is a training target, "
                "so its prompt would end with an assistant turn. Mark it context "
                "with weight 0."
            )
            raise ValueError(msg)
        answered = {
            message.tool_call_id
            for message in messages[previous + 1 : turn]
            if isinstance(message, ToolMessage)
        }
        unanswered = [
            call.id
            for call in messages[previous].tool_calls or []
            if call.id not in answered
        ]
        if unanswered:
            msg = (
                f"messages[{previous}] calls {unanswered}, which no tool message "
                f"answers before the next assistant message, messages[{turn}]."
            )
            raise ValueError(msg)


class TracesValidationError(ValueError):
    """A traces file breaks the spec; ``errors`` lists ``row N: ...`` lines."""

    def __init__(self, source: str, errors: list[str], total: int) -> None:
        self.errors = errors
        self.total = total
        shown = "\n".join(f"  {line}" for line in errors)
        more = "" if total <= len(errors) else f"\n  ... and {total - len(errors)} more"
        super().__init__(
            f"{source} is not a valid traces dataset ({total} bad rows):\n{shown}{more}"
        )


def _error_lines(row_number: int, error: ValidationError) -> str:
    details = "; ".join(
        f"{'.'.join(str(part) for part in item['loc']) or 'row'}: {item['msg']}"
        for item in error.errors(include_url=False)
    )
    return f"row {row_number}: {details}"


def _row_error(row_number: int, row: Mapping[str, Any]) -> str | None:
    """Why row ``row_number`` breaks the spec, or ``None`` if it is a valid segment."""
    try:
        TracesSegmentRow.model_validate(row)
    except ValidationError as error:
        return _error_lines(row_number, error)
    return None


def validate_traces_rows(
    rows: Iterable[Mapping[str, Any]], source: str = "dataset"
) -> int:
    """Validate rows against the traces spec.

    Rows are numbered from 1 in file order (the line number of a JSONL file
    without blank lines).

    :param rows: Raw rows.
    :param source: Name used in the error.
    :return: Number of rows checked.
    :raises TracesValidationError: If any row breaks the spec, or there are none.
    """
    errors: list[str] = []
    bad = 0
    count = 0
    for count, row in enumerate(rows, start=1):
        error = _row_error(count, row)
        if error is not None:
            bad += 1
            if len(errors) < MAX_REPORTED_ERRORS:
                errors.append(error)
    if count == 0:
        raise TracesValidationError(source, ["the file has no rows"], 1)
    if bad:
        raise TracesValidationError(source, errors, bad)
    return count


def _jsonl_rows(path: Path) -> Iterator[Mapping[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise TracesValidationError(
                    str(path), [f"row {line_number}: not valid JSON ({error})"], 1
                ) from error
            if not isinstance(row, dict):
                raise TracesValidationError(
                    str(path), [f"row {line_number}: not a JSON object"], 1
                )
            yield row


def _without_none_keys(value: Any) -> Any:  # noqa: ANN401 -- any nested Arrow value
    """``value`` without ``None`` dict entries; Arrow fills every struct key."""
    if isinstance(value, dict):
        return {
            key: _without_none_keys(item)
            for key, item in value.items()
            if item is not None
        }
    if isinstance(value, list):
        return [_without_none_keys(item) for item in value]
    return value


def _parquet_rows(path: Path) -> Iterator[Mapping[str, Any]]:
    # Optional: pyarrow is not an agilerl dependency; only parquet checks need it.
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        msg = (
            "Validating a parquet traces file needs pyarrow; install it "
            "(pip install pyarrow) or upload JSONL."
        )
        raise ImportError(msg) from error
    files = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
    for file in files:
        for batch in pq.ParquetFile(file).iter_batches():
            for row in batch.to_pylist():
                yield _without_none_keys(row)


def iter_traces_file(path: str | Path) -> Iterator[Mapping[str, Any]]:
    """Rows of a JSONL file, or a parquet file or folder of shards.

    :param path: The dataset path.
    :return: Raw rows; parquet rows drop the ``None`` keys Arrow fills in.
    """
    path = Path(path)
    if path.is_dir() or path.suffix.lower() == ".parquet":
        return _parquet_rows(path)
    if path.suffix.lower() in (".jsonl", ".ndjson", ".json"):
        return _jsonl_rows(path)
    msg = f"A traces dataset is JSONL or parquet; got {path.name!r}."
    raise ValueError(msg)


def validate_traces_file(path: str | Path) -> int:
    """Validate a local traces file as stored rows (segments).

    :param path: JSONL file, parquet file, or folder of parquet shards.
    :return: Number of rows checked.
    :raises TracesValidationError: If any row breaks the spec.
    """
    return validate_traces_rows(iter_traces_file(path), source=str(path))

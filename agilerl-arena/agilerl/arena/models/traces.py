# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""The ``traces`` dataset format: OpenAI Chat Completions messages in TRL prompt-completion rows (torch-free).

One row is one model call: ``prompt`` holds the messages the model was given,
``completion`` the one assistant message it produced, ``tools`` the function
schemas it could call. The CL loop's conversational ``messages`` rows are the
same format grouped per conversation. See the Arena docs page on traces datasets.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from datetime import datetime
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

from agilerl.arena.traces_format import (
    JSON_TEXT_COLUMNS,
    METADATA,
    REASONING_KEYS,
    REQUIRED_COLUMNS,
    SCORES,
)

TRACES_DATASET_FORMAT = "traces"
"""``environment.dataset_format`` and ``arena datasets create --category`` value."""

MESSAGES = "messages"
"""The TRL conversational column a CL ``sft_rows`` export writes instead of prompt / completion."""

JSON_COLUMNS = (*JSON_TEXT_COLUMNS, MESSAGES, SCORES, METADATA)
"""Columns read from JSON text as well as nested values."""

MAX_REPORTED_ERRORS = 20


class ContentPart(BaseModel):
    """One part of a multi-part message ``content``, such as ``{"type": "text", "text": ...}``."""

    model_config = ConfigDict(extra="ignore")

    type: str = Field(min_length=1)


MessageContent = str | list[ContentPart]


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


class SystemMessage(BaseModel):
    """A ``system`` or ``developer`` instruction message."""

    model_config = ConfigDict(extra="ignore")

    role: Literal["system", "developer"]
    content: MessageContent
    name: str | None = None


class UserMessage(BaseModel):
    """A ``user`` message."""

    model_config = ConfigDict(extra="ignore")

    role: Literal["user"]
    content: MessageContent
    name: str | None = None


class AssistantMessage(BaseModel):
    """An ``assistant`` turn: content, tool calls, or both, with optional reasoning."""

    model_config = ConfigDict(extra="ignore")

    role: Literal["assistant"]
    content: MessageContent | None = None
    reasoning_content: str | None = None
    reasoning: str | None = None
    thinking: str | None = None
    tool_calls: list[ToolCall] | None = None
    name: str | None = None

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


class ToolMessage(BaseModel):
    """A ``tool`` result answering one earlier tool call."""

    model_config = ConfigDict(extra="ignore")

    role: Literal["tool"]
    tool_call_id: str = Field(min_length=1)
    content: MessageContent
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
    """A named score on a call: user feedback, or a judge id plus rubric hash."""

    model_config = ConfigDict(extra="ignore")

    name: str = Field(min_length=1)
    value: float | None = None
    source: str | None = None
    label: str | None = None
    comment: str | None = None


class TracesRowBase(BaseModel):
    """Columns both row forms share; unknown extra columns are ignored."""

    model_config = ConfigDict(extra="ignore")

    tools: list[ToolDefinition] | None = None
    answer: str | None = None
    call_id: str | None = None
    session_id: str | None = None
    segment_index: int | None = None
    model_version: str | None = None
    checkpoint_id: str | None = None
    base_model: str | None = None
    received_at: datetime | None = None
    scores: list[TraceScore] | None = None
    finish_reason: str | None = None
    metadata: dict[str, str] | None = None

    @model_validator(mode="before")
    @classmethod
    def _decode_json_columns(cls, data: Any) -> Any:  # noqa: ANN401 -- raw row
        if not isinstance(data, Mapping):
            return data
        decoded = dict(data)
        for column in JSON_COLUMNS:
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


class TracesRow(TracesRowBase):
    """One model call: the ``prompt`` it saw and its one-message ``completion``."""

    prompt: list[Message] = Field(min_length=1)
    completion: list[AssistantMessage] = Field(min_length=1, max_length=1)

    @model_validator(mode="after")
    def _check_turns(self) -> Self:
        if isinstance(self.prompt[-1], AssistantMessage):
            msg = (
                "prompt ends with an assistant message; the completion is the "
                "next assistant turn, so the prompt ends before it."
            )
            raise ValueError(msg)
        tool_names = self._tool_names()
        _check_conversation(self.prompt, tool_names, "prompt")
        _check_conversation(self.completion, tool_names, "completion")
        return self

    @property
    def completion_message(self) -> AssistantMessage:
        """The reference assistant turn."""
        return self.completion[0]


class TracesMessagesRow(TracesRowBase):
    """One conversation in TRL ``messages`` form, trained on every assistant turn."""

    messages: list[Message] = Field(min_length=2)

    @model_validator(mode="after")
    def _check_turns(self) -> Self:
        if isinstance(self.messages[0], AssistantMessage):
            msg = "messages starts with an assistant turn; it needs a prompt first."
            raise ValueError(msg)
        if not any(isinstance(m, AssistantMessage) for m in self.messages):
            msg = "messages has no assistant turn to train on."
            raise ValueError(msg)
        _check_conversation(self.messages, self._tool_names(), "messages")
        return self


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


def row_form(columns: Iterable[str]) -> type[TracesRow | TracesMessagesRow]:
    """The row model a dataset's columns select: ``prompt``/``completion``, else ``messages``.

    :param columns: Column names of the dataset.
    :return: :class:`TracesRow` or :class:`TracesMessagesRow`.
    :raises ValueError: If the columns name neither form.
    """
    names = set(columns)
    if set(REQUIRED_COLUMNS) <= names:
        return TracesRow
    if MESSAGES in names:
        return TracesMessagesRow
    msg = (
        "A traces dataset needs 'prompt' and 'completion' columns, or a "
        f"'messages' column; got {sorted(names)}."
    )
    raise ValueError(msg)


def _error_lines(row_number: int, error: ValidationError) -> str:
    details = "; ".join(
        f"{'.'.join(str(part) for part in item['loc']) or 'row'}: {item['msg']}"
        for item in error.errors(include_url=False)
    )
    return f"row {row_number}: {details}"


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
    model: type[TracesRow | TracesMessagesRow] | None = None
    for count, row in enumerate(rows, start=1):
        if model is None:
            try:
                model = row_form(row.keys())
            except ValueError as error:
                raise TracesValidationError(source, [f"row 1: {error}"], 1) from error
        try:
            model.model_validate(row)
        except ValidationError as error:
            bad += 1
            if len(errors) < MAX_REPORTED_ERRORS:
                errors.append(_error_lines(count, error))
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
    # Optional: pyarrow is not an agilerl-arena dependency; only parquet checks need it.
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
    """Validate a local traces file before upload.

    :param path: JSONL file, parquet file, or folder of parquet shards.
    :return: Number of rows checked.
    :raises TracesValidationError: If any row breaks the spec.
    """
    return validate_traces_rows(iter_traces_file(path), source=str(path))

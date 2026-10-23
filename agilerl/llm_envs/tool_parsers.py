# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Token-ID tool-call parsers, one per model family's tool-call format.

Each parser finds its family's special-token delimiters in the generation IDs
and decodes only the segments between them. A tag typed as prose tokenizes to
ordinary IDs, never the special-token ID, so it cannot false-positive.
Detection picks the first registered parser whose required tokens are all in
the tokenizer vocab.

- :class:`ToolCallTagParser`: Qwen, Hermes, Granite 4, Nemotron
- :class:`Granite3ToolParser`: Granite 3
- :class:`Gemma4ToolParser`: Gemma 4
- :class:`HarmonyToolParser`: GPT-OSS
"""

from __future__ import annotations

import enum
import json
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from agilerl.utils.algo_utils import is_str_keyed_dict

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

__all__ = [
    "TOOL_PARSERS",
    "Gemma4ToolParser",
    "Granite3ToolParser",
    "HarmonyToolParser",
    "ParsedToolCall",
    "ToolCallParseStatus",
    "ToolCallTagParser",
    "ToolParser",
    "detect_tool_parser",
    "get_tool_parser",
]

GEMMA_STRING_DELIM = '<|"|>'
XML_FUNCTION_RE = re.compile(r"<function=([^>]+)>")
XML_PARAMETER_RE = re.compile(r"<parameter=([^>]+)>\n?(.*?)\n?</parameter>", re.DOTALL)
HARMONY_RECIPIENT_RE = re.compile(r"to=([^\s<]+)")
HARMONY_FUNCTION_PREFIX = "functions."


class ToolCallParseStatus(str, enum.Enum):
    """Per-attempt outcome of parsing one tool-call block."""

    OK = "ok"
    INVALID_JSON = "invalid_json"
    UNCLOSED_BLOCK = "unclosed_block"
    MISSING_NAME = "missing_name"
    MALFORMED_STRUCTURE = "malformed_structure"


@dataclass
class ParsedToolCall:
    """One tool-call block as parsed, successful or malformed."""

    raw: str
    name: str | None = None
    arguments: dict[str, Any] | None = None
    token_span: tuple[int, int] | None = None
    status: ToolCallParseStatus = ToolCallParseStatus.OK
    turn: tuple[ParsedToolCall, ...] = ()
    """Every valid call of this call's turn, in order; set when the harness queues it."""
    turn_index: int = 0
    """This call's position in :attr:`turn`."""


class ToolParser(ABC):
    """Extracts tool calls from completion token IDs.

    Subclasses declare the special tokens they need; construction fails when
    the tokenizer lacks a required one.
    """

    REQUIRED_TOKENS: ClassVar[tuple[str, ...]]
    OPTIONAL_TOKENS: ClassVar[tuple[str, ...]] = ()

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        """Resolve this parser's delimiter IDs in ``tokenizer``."""
        self._tokenizer = tokenizer
        self._ids = self.token_ids(tokenizer)
        missing = [token for token in self.REQUIRED_TOKENS if token not in self._ids]
        if missing:
            msg = f"{type(self).__name__} needs {', '.join(missing)} in the tokenizer vocab."
            raise ValueError(msg)

    @classmethod
    def token_ids(cls, tokenizer: PreTrainedTokenizerBase) -> dict[str, int]:
        """IDs of this parser's delimiter tokens that are in ``tokenizer``'s vocab."""
        ids: dict[str, int] = {}
        for token in (*cls.REQUIRED_TOKENS, *cls.OPTIONAL_TOKENS):
            token_id = tokenizer.convert_tokens_to_ids(token)
            # Hugging Face maps a token missing from the vocab to the unk ID.
            if isinstance(token_id, int) and token_id != tokenizer.unk_token_id:
                ids[token] = token_id
        return ids

    @abstractmethod
    def extract(
        self, token_ids: list[int], tools: list[Any] | None = None, /
    ) -> tuple[list[int], list[ParsedToolCall]]:
        """Split IDs into the content before the first call plus every parse attempt.

        :param token_ids: One generation's token IDs.
        :param tools: Tool schemas, for parsers whose bodies need value types.
        :return: Content IDs and one entry per call block, malformed ones included.
        """

    def _decode(self, token_ids: list[int]) -> str:
        """Decode IDs to stripped text, keeping special tokens inside the segment."""
        if not token_ids:
            return ""
        text = self._tokenizer.decode(token_ids, skip_special_tokens=False)
        if not isinstance(text, str):
            msg = "decode() of one sequence returns str"
            raise TypeError(msg)
        return text.strip()

    def _tagged_calls(
        self,
        token_ids: list[int],
        start: str,
        end: str,
        parse_body: Callable[[str, tuple[int, int]], ParsedToolCall],
        scan_from: int = 0,
    ) -> tuple[list[int], list[ParsedToolCall]]:
        """Parse every ``start`` ... ``end`` block at or after ``scan_from``."""
        start_id, end_id = self._ids[start], self._ids.get(end)
        first = _find(token_ids, start_id, scan_from)
        if first == -1:
            return token_ids, []
        calls: list[ParsedToolCall] = []
        index = first
        while index != -1:
            close = -1 if end_id is None else _find(token_ids, end_id, index + 1)
            if close == -1:
                raw = self._decode(token_ids[index + 1 :])
                span = (index, len(token_ids))
                calls.append(_failed(raw, span, ToolCallParseStatus.UNCLOSED_BLOCK))
                break
            body = self._decode(token_ids[index + 1 : close])
            calls.append(parse_body(body, (index, close + 1)))
            index = _find(token_ids, start_id, close + 1)
        return token_ids[:first], calls


class ToolCallTagParser(ToolParser):
    """``<tool_call>`` blocks with JSON or XML bodies.

    JSON bodies hold one object with name and arguments (Qwen, Hermes,
    Granite 4). XML bodies hold one function block with parameter values as
    raw text (Nemotron). Blocks inside a thinking trace are skipped.
    """

    CALL_START = "<tool_call>"
    CALL_END = "</tool_call>"
    THINK_END = "</think>"
    REQUIRED_TOKENS: ClassVar[tuple[str, ...]] = (CALL_START,)
    OPTIONAL_TOKENS: ClassVar[tuple[str, ...]] = (CALL_END, THINK_END)

    def extract(
        self, token_ids: list[int], tools: list[Any] | None = None, /
    ) -> tuple[list[int], list[ParsedToolCall]]:
        """Parse tool blocks after any thinking trace."""
        think_end = _find(token_ids, self._ids.get(self.THINK_END))
        params = _param_schemas(tools)

        def parse_body(raw: str, span: tuple[int, int]) -> ParsedToolCall:
            if raw.startswith("<function="):
                return _parsed_xml_call(raw, span, params)
            return _parsed_json_call(raw, span)

        return self._tagged_calls(
            token_ids, self.CALL_START, self.CALL_END, parse_body, think_end + 1
        )


class Granite3ToolParser(ToolParser):
    """One ``<|tool_call|>`` tag, then a JSON array of calls to end of turn.

    Granite 3 emits no closing tag and no leading prose. Array items share the
    block span.
    """

    CALL_START = "<|tool_call|>"
    REQUIRED_TOKENS: ClassVar[tuple[str, ...]] = (CALL_START,)

    def extract(
        self, token_ids: list[int], _tools: list[Any] | None = None, /
    ) -> tuple[list[int], list[ParsedToolCall]]:
        """Parse the trailing call array."""
        tag = _find(token_ids, self._ids[self.CALL_START])
        if tag == -1:
            return token_ids, []
        content = self._decode(token_ids[:tag])
        if content:
            failed = _failed(content, (0, tag), ToolCallParseStatus.MALFORMED_STRUCTURE)
            return token_ids, [failed]
        raw = self._decode(token_ids[tag + 1 :])
        span = (tag, len(token_ids))
        if not raw.startswith("["):
            return token_ids, [
                _failed(raw, span, ToolCallParseStatus.MALFORMED_STRUCTURE)
            ]
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return token_ids, [_failed(raw, span, ToolCallParseStatus.INVALID_JSON)]
        return token_ids[:tag], [_call_from_object(raw, item, span) for item in payload]


class Gemma4ToolParser(ToolParser):
    """``<|tool_call>call:name{key:value,...}<tool_call|>`` blocks.

    Argument values are Gemma's own ``key:value`` text, not JSON: strings wrap
    in ``<|"|>``, bare scalars stay strings for the executor to coerce, and
    braces nest.
    """

    CALL_START = "<|tool_call>"
    CALL_END = "<tool_call|>"
    RESPONSE_START = "<|tool_response>"
    RESPONSE_END = "<tool_response|>"
    REQUIRED_TOKENS: ClassVar[tuple[str, ...]] = (
        CALL_START,
        CALL_END,
        RESPONSE_START,
        RESPONSE_END,
    )

    def extract(
        self, token_ids: list[int], _tools: list[Any] | None = None, /
    ) -> tuple[list[int], list[ParsedToolCall]]:
        """Parse native Gemma calls."""
        return self._tagged_calls(
            token_ids, self.CALL_START, self.CALL_END, _parsed_gemma_call
        )

    def format_responses(self, messages: list[dict[str, str]]) -> str | None:
        """One ``<|tool_response>response:name{value:...}<tool_response|>`` block per result.

        The template skips standalone tool messages, so results render mid-turn here instead of through a turn frame.
        """
        if not messages or any(message["role"] != "tool" for message in messages):
            return None
        parts = []
        for message in messages:
            name = message.get("name") or "unknown"
            value = f"{GEMMA_STRING_DELIM}{message['content']}{GEMMA_STRING_DELIM}"
            parts.append(
                f"{self.RESPONSE_START}response:{name}{{value:{value}}}{self.RESPONSE_END}"
            )
        return "".join(parts)


class HarmonyToolParser(ToolParser):
    """Channel blocks addressed ``to=functions.name`` (GPT-OSS).

    Each ``<|start|>`` block names its recipient in the header. Function blocks
    hold JSON arguments and close with ``<|call|>``. Other blocks are content.
    Parsing stops at the first ``<|return|>``.
    """

    BLOCK_START = "<|start|>"
    MESSAGE = "<|message|>"
    CALL_END = "<|call|>"
    BLOCK_END = "<|end|>"
    RETURN = "<|return|>"
    REQUIRED_TOKENS: ClassVar[tuple[str, ...]] = (
        BLOCK_START,
        MESSAGE,
        CALL_END,
        BLOCK_END,
    )
    OPTIONAL_TOKENS: ClassVar[tuple[str, ...]] = (RETURN,)

    def extract(
        self, token_ids: list[int], _tools: list[Any] | None = None, /
    ) -> tuple[list[int], list[ParsedToolCall]]:
        """Parse function-addressed blocks."""
        start_id = self._ids[self.BLOCK_START]
        message_id = self._ids[self.MESSAGE]
        stop_ids = (start_id, self._ids[self.BLOCK_END], self._ids[self.CALL_END])
        returned = _find(token_ids, self._ids.get(self.RETURN))
        ids = token_ids if returned == -1 else token_ids[:returned]
        calls: list[ParsedToolCall] = []
        content_end = len(ids)
        index = _find(ids, start_id)
        while index != -1:
            message_at = _find(ids, message_id, index + 1)
            if message_at == -1:
                break
            stops = [_find(ids, stop, message_at + 1) for stop in stop_ids]
            body_end = min((s for s in stops if s != -1), default=len(ids))
            closed = body_end < len(ids) and ids[body_end] != start_id
            next_index = body_end + 1 if closed else body_end
            match = HARMONY_RECIPIENT_RE.search(
                self._decode(ids[index + 1 : message_at])
            )
            if match and match.group(1).startswith(HARMONY_FUNCTION_PREFIX):
                content_end = min(content_end, index)
                body = self._decode(ids[message_at + 1 : body_end])
                name = match.group(1).removeprefix(HARMONY_FUNCTION_PREFIX)
                calls.append(
                    _parsed_harmony_call(body, name, (index, next_index), closed)
                )
            index = _find(ids, start_id, next_index)
        return ids[:content_end], calls


TOOL_PARSERS: dict[str, type[ToolParser]] = {
    "tool_call": ToolCallTagParser,
    "granite3": Granite3ToolParser,
    "gemma4": Gemma4ToolParser,
    "harmony": HarmonyToolParser,
}


def get_tool_parser(name: str, tokenizer: PreTrainedTokenizerBase) -> ToolParser:
    """Build the named parser over ``tokenizer``.

    :param name: Parser name from :data:`TOOL_PARSERS`.
    :param tokenizer: Tokenizer defining the delimiter IDs.
    :return: The parser.
    """
    try:
        parser_cls = TOOL_PARSERS[name]
    except KeyError:
        available = ", ".join(sorted(TOOL_PARSERS))
        msg = f"Unknown tool_parser {name!r}. Available: {available}."
        raise ValueError(msg) from None
    return parser_cls(tokenizer)


def detect_tool_parser(tokenizer: PreTrainedTokenizerBase) -> ToolParser | None:
    """First registered parser whose required tokens are all in the vocab, or ``None``.

    :param tokenizer: Tokenizer to probe.
    :return: The matching parser, or ``None`` when no parser matches.
    """
    for parser_cls in TOOL_PARSERS.values():
        if set(parser_cls.REQUIRED_TOKENS) <= parser_cls.token_ids(tokenizer).keys():
            return parser_cls(tokenizer)
    return None


def _failed(
    raw: str,
    span: tuple[int, int],
    status: ToolCallParseStatus,
    name: str | None = None,
) -> ParsedToolCall:
    """A parse attempt that did not yield a runnable call."""
    return ParsedToolCall(raw=raw, name=name, token_span=span, status=status)


def _call_from_object(
    raw: str, payload: object, span: tuple[int, int]
) -> ParsedToolCall:
    """One ``{"name", "arguments"}`` object; ``arguments`` may be a JSON string."""
    if not is_str_keyed_dict(payload):
        return _failed(raw, span, ToolCallParseStatus.MALFORMED_STRUCTURE)
    name = payload.get("name")
    if not isinstance(name, str) or not name:
        return _failed(raw, span, ToolCallParseStatus.MISSING_NAME)
    arguments = payload.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return _failed(raw, span, ToolCallParseStatus.MALFORMED_STRUCTURE)
    if not is_str_keyed_dict(arguments):
        return _failed(raw, span, ToolCallParseStatus.MALFORMED_STRUCTURE)
    return ParsedToolCall(raw=raw, name=name, arguments=arguments, token_span=span)


def _parsed_json_call(raw: str, span: tuple[int, int]) -> ParsedToolCall:
    """One JSON tool-call body."""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return _failed(raw, span, ToolCallParseStatus.INVALID_JSON)
    return _call_from_object(raw, payload, span)


def _parsed_harmony_call(
    body: str, name: str, span: tuple[int, int], closed: bool
) -> ParsedToolCall:
    """One function-addressed Harmony block with JSON arguments."""
    try:
        arguments = json.loads(body)
    except json.JSONDecodeError:
        return _failed(body, span, ToolCallParseStatus.INVALID_JSON, name or None)
    if not closed:
        return _failed(body, span, ToolCallParseStatus.UNCLOSED_BLOCK, name or None)
    if not name:
        return _failed(body, span, ToolCallParseStatus.MISSING_NAME)
    if not is_str_keyed_dict(arguments):
        return _failed(body, span, ToolCallParseStatus.MALFORMED_STRUCTURE)
    return ParsedToolCall(raw=body, name=name, arguments=arguments, token_span=span)


def _parsed_xml_call(
    raw: str,
    span: tuple[int, int],
    params: dict[str, dict[str, dict[str, Any]]],
) -> ParsedToolCall:
    """One XML function body, values coerced to their schema types."""
    name_match = XML_FUNCTION_RE.search(raw)
    if not name_match:
        return _failed(raw, span, ToolCallParseStatus.MALFORMED_STRUCTURE)
    name = name_match.group(1)
    schemas = params.get(name, {})
    arguments: dict[str, Any] = {}
    any_invalid = False
    for match in XML_PARAMETER_RE.finditer(raw):
        key, text = match.group(1), match.group(2).strip()
        arguments[key], invalid = _coerce_xml_value(text, schemas.get(key))
        any_invalid = any_invalid or invalid
    status = ToolCallParseStatus.INVALID_JSON if any_invalid else ToolCallParseStatus.OK
    return ParsedToolCall(
        raw=raw, name=name, arguments=arguments, token_span=span, status=status
    )


def _coerce_xml_value(text: str, schema: dict[str, Any] | None) -> tuple[Any, bool]:
    """Coerce one raw XML parameter value to its schema type.

    String-typed params stay verbatim; anything else is JSON-decoded.
    :return: The value, and whether it is invalid: not JSON and the schema
        does not accept a string.
    """
    string_allowed = False
    if schema is not None:
        if schema.get("type") in ("string", ["string"]):
            return text, False
        branches = schema.get("anyOf") or schema.get("oneOf") or []
        string_allowed = any(
            isinstance(branch, dict) and branch.get("type") == "string"
            for branch in branches
        )
    try:
        return json.loads(text), False
    except json.JSONDecodeError:
        return text, not string_allowed


def _param_schemas(tools: list[Any] | None) -> dict[str, dict[str, dict[str, Any]]]:
    """Map tool name to parameter name to JSON-schema fragment."""
    index: dict[str, dict[str, dict[str, Any]]] = {}
    for tool in tools or []:
        spec = tool.get("function", tool) if isinstance(tool, dict) else None
        if not isinstance(spec, dict) or not isinstance(spec.get("name"), str):
            continue
        parameters = spec.get("parameters")
        properties = (
            parameters.get("properties") if isinstance(parameters, dict) else None
        )
        if isinstance(properties, dict):
            index[spec["name"]] = {
                key: value
                for key, value in properties.items()
                if isinstance(value, dict)
            }
    return index


def _parsed_gemma_call(raw: str, span: tuple[int, int]) -> ParsedToolCall:
    """One Gemma body: ``call:name{args}``."""
    malformed = _failed(raw, span, ToolCallParseStatus.MALFORMED_STRUCTURE)
    if not raw.startswith("call:"):
        return malformed
    brace = raw.find("{", len("call:"))
    if brace == -1:
        return malformed
    name = raw[len("call:") : brace].strip()
    if not name:
        return _failed(raw, span, ToolCallParseStatus.MISSING_NAME)
    close = _matching_brace(raw, brace)
    if close == -1 or raw[close + 1 :].strip():
        return malformed
    try:
        arguments = _parse_gemma_args(raw[brace + 1 : close])
    except ValueError:
        return malformed
    return ParsedToolCall(raw=raw, name=name, arguments=arguments, token_span=span)


def _matching_brace(text: str, opening: int) -> int:
    """Index of the brace matching ``text[opening]``, skipping delimited strings."""
    depth = 0
    index = opening
    while index < len(text):
        if text.startswith(GEMMA_STRING_DELIM, index):
            end = text.find(GEMMA_STRING_DELIM, index + len(GEMMA_STRING_DELIM))
            if end == -1:
                return -1
            index = end + len(GEMMA_STRING_DELIM)
            continue
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return -1


def _parse_gemma_args(text: str) -> dict[str, Any]:
    """Parse ``key:value`` args; bare scalars stay strings."""
    args: dict[str, Any] = {}
    index = 0
    while index < len(text):
        while index < len(text) and text[index] in " ,\n\t":
            index += 1
        if index >= len(text):
            break
        colon = text.find(":", index)
        if colon == -1:
            msg = "Gemma args end mid-key."
            raise ValueError(msg)
        args[text[index:colon].strip()], index = _parse_gemma_value(text, colon + 1)
    return args


def _parse_gemma_value(text: str, index: int) -> tuple[Any, int]:
    """Parse one value at ``index``; return it plus the next index."""
    while index < len(text) and text[index] in " \n\t":
        index += 1
    if text.startswith(GEMMA_STRING_DELIM, index):
        start = index + len(GEMMA_STRING_DELIM)
        end = text.find(GEMMA_STRING_DELIM, start)
        if end == -1:
            msg = "Gemma args hold an unterminated string."
            raise ValueError(msg)
        return text[start:end], end + len(GEMMA_STRING_DELIM)
    if text.startswith("{", index):
        close = _matching_brace(text, index)
        if close == -1:
            msg = "Gemma args hold unbalanced braces."
            raise ValueError(msg)
        return _parse_gemma_args(text[index + 1 : close]), close + 1
    if text.startswith("[", index):
        return _parse_gemma_array(text, index + 1)
    start = index
    while index < len(text) and text[index] not in ",}]":
        index += 1
    return text[start:index].strip(), index


def _parse_gemma_array(text: str, index: int) -> tuple[list[Any], int]:
    """Parse array items from just past its ``[``."""
    items: list[Any] = []
    while True:
        while index < len(text) and text[index] in " ,\n\t":
            index += 1
        if index >= len(text):
            msg = "Gemma args hold an unterminated array."
            raise ValueError(msg)
        if text[index] == "]":
            return items, index + 1
        value, index = _parse_gemma_value(text, index)
        items.append(value)


def _find(token_ids: list[int], target: int | None, start: int = 0) -> int:
    """Index of ``target`` at or after ``start``; ``-1`` when absent or ``None``."""
    if target is None:
        return -1
    try:
        return token_ids.index(target, start)
    except ValueError:
        return -1

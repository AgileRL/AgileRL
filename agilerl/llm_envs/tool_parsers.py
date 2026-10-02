# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Token-ID tool-call parsers, one per delimiter family.

Each parser scans generation IDs for its family's delimiter IDs, then decodes
only the segments between them. No regex on decoded text: a literal tag typed
as prose tokenizes to ordinary IDs, never the special delimiter ID, so it
cannot false-positive. Detection probes each family's required tags against
the tokenizer vocab and picks the first full match.
"""

from __future__ import annotations

import enum
import json
import re
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "Gemma4ToolParser",
    "Granite3ToolParser",
    "HarmonyToolParser",
    "ParsedToolCall",
    "ToolCallParseStatus",
    "ToolCallTagParser",
    "ToolParser",
    "ToolTokenizer",
    "detect_tool_parser",
    "get_tool_parser",
]


class ToolCallParseStatus(str, enum.Enum):
    """Per-attempt outcome of parsing one tool-call block."""

    OK = "ok"
    INVALID_JSON = "invalid_json"
    UNCLOSED_BLOCK = "unclosed_block"
    MISSING_NAME = "missing_name"
    MALFORMED_STRUCTURE = "malformed_structure"
    UNKNOWN_TOOL = "unknown_tool"


@dataclass
class ParsedToolCall:
    """One tool-call block as parsed, successful or malformed."""

    raw: str
    name: str | None = None
    arguments: dict[str, Any] | None = None
    token_span: tuple[int, int] | None = None
    status: ToolCallParseStatus = ToolCallParseStatus.OK


@runtime_checkable
class ToolTokenizer(Protocol):
    """Tokenizer surface tool parsers need: vocab lookup plus decode."""

    def convert_tokens_to_ids(self, token: str) -> int | None: ...

    def decode(
        self, token_ids: list[int], skip_special_tokens: bool = False
    ) -> str: ...


@runtime_checkable
class ToolParser(Protocol):
    """Extracts tool calls from completion token IDs."""

    REQUIRED_TOKENS: tuple[str, ...]

    def __init__(self, tokenizer: ToolTokenizer) -> None: ...

    def extract(
        self,
        token_ids: list[int],
        tools: list[Any] | None = None,
    ) -> tuple[list[int], list[ParsedToolCall]]:
        """Split IDs into content prefix plus every parse attempt."""
        ...


class ToolCallTagParser:
    """Angle-bracket tool blocks with JSON or XML bodies.

    JSON bodies hold one object with name and arguments (Qwen, Hermes,
    Granite 4 templates). XML bodies hold one function block with parameter
    values as raw text (Nemotron templates). Bodies dispatch on their first
    token: an object brace parses as JSON, a function tag as XML.
    """

    REQUIRED_TOKENS: tuple[str, ...] = ("<tool_call>",)

    def __init__(self, tokenizer: ToolTokenizer) -> None:
        """Cache this tokenizer's delimiter IDs."""
        self._tokenizer = tokenizer
        tag_id = _token_id(tokenizer, "<tool_call>")
        if tag_id is None:
            msg = "Tool-call parsing needs <tool_call> in the tokenizer vocab."
            raise ValueError(msg)
        self._tc_id = tag_id
        self._tc_end_id = _token_id(tokenizer, "</tool_call>")
        self._think_end_id = _token_id(tokenizer, "</think>")

    def extract(
        self,
        token_ids: list[int],
        tools: list[Any] | None = None,
    ) -> tuple[list[int], list[ParsedToolCall]]:
        """Parse tool blocks from IDs, scanning past any thinking trace."""
        scan_from = 0
        if self._think_end_id is not None:
            think_end = _find(token_ids, self._think_end_id)
            if think_end != -1:
                scan_from = think_end + 1
        tc_start = _find(token_ids, self._tc_id, scan_from)
        if tc_start == -1:
            return token_ids, []
        content_ids = token_ids[:tc_start]
        params = _build_param_type_index(tools)
        calls: list[ParsedToolCall] = []
        index = tc_start
        while index < len(token_ids):
            if token_ids[index] != self._tc_id:
                index += 1
                continue
            end = (
                _find(token_ids, self._tc_end_id, index + 1)
                if self._tc_end_id is not None
                else -1
            )
            if end == -1:
                raw = _decode(self._tokenizer, token_ids[index + 1 :]).strip()
                calls.append(
                    ParsedToolCall(
                        raw=raw,
                        token_span=(index, len(token_ids)),
                        status=ToolCallParseStatus.UNCLOSED_BLOCK,
                    )
                )
                break
            raw = _decode(self._tokenizer, token_ids[index + 1 : end]).strip()
            span = (index, end + 1)
            if raw.startswith("<function="):
                calls.append(_parsed_xml_call(raw, span, params))
            else:
                calls.append(_parsed_json_call(raw, span))
            index = end + 1
        return content_ids, calls


class Granite3ToolParser:
    """Bare calls: ``<|tool_call|>`` plus a JSON array to end of turn.

    Granite 3 emits calls with no closing tag and no leading prose: one tag,
    then ``[{"name", "arguments"}, ...]``. Array items share the block span;
    per-item offsets are not cheaply recoverable.
    """

    REQUIRED_TOKENS: tuple[str, ...] = ("<|tool_call|>",)

    def __init__(self, tokenizer: ToolTokenizer) -> None:
        """Cache this tokenizer's call-tag ID."""
        self._tokenizer = tokenizer
        tag_id = _token_id(tokenizer, "<|tool_call|>")
        if tag_id is None:
            msg = "Granite parsing needs <|tool_call|> in the tokenizer vocab."
            raise ValueError(msg)
        self._tc_id = tag_id

    def extract(
        self,
        token_ids: list[int],
        _tools: list[Any] | None = None,
    ) -> tuple[list[int], list[ParsedToolCall]]:
        """Parse the trailing call array from ``token_ids``."""
        tag = _find(token_ids, self._tc_id)
        if tag == -1:
            return token_ids, []
        content = _decode(self._tokenizer, token_ids[:tag]).strip()
        if content:
            return token_ids, [
                ParsedToolCall(
                    raw=content,
                    token_span=(0, tag),
                    status=ToolCallParseStatus.MALFORMED_STRUCTURE,
                )
            ]
        raw = _decode(self._tokenizer, token_ids[tag + 1 :]).strip()
        span = (tag, len(token_ids))
        if not raw.startswith("["):
            return token_ids, [
                ParsedToolCall(
                    raw=raw,
                    token_span=span,
                    status=ToolCallParseStatus.MALFORMED_STRUCTURE,
                )
            ]
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return token_ids, [
                ParsedToolCall(
                    raw=raw,
                    token_span=span,
                    status=ToolCallParseStatus.INVALID_JSON,
                )
            ]
        if not isinstance(payload, list):
            return token_ids, [
                ParsedToolCall(
                    raw=raw,
                    token_span=span,
                    status=ToolCallParseStatus.MALFORMED_STRUCTURE,
                )
            ]
        calls = [_parsed_array_item(raw, item, span) for item in payload]
        return token_ids[:tag], calls


class Gemma4ToolParser:
    """Native calls: ``<|tool_call>call:name{key:value,...}<tool_call|>``.

    Gemma 4 argument values are custom ``key:value`` text, not JSON: strings
    wrap in ``<|"|>`` delimiters, bare scalars stay strings for the executor
    to coerce, and braces nest. Thought channels never nest call tags.
    """

    REQUIRED_TOKENS: tuple[str, ...] = ("<|tool_call>", "<tool_call|>")

    def __init__(self, tokenizer: ToolTokenizer) -> None:
        """Cache this tokenizer's delimiter IDs."""
        self._tokenizer = tokenizer
        start_id = _token_id(tokenizer, "<|tool_call>")
        end_id = _token_id(tokenizer, "<tool_call|>")
        if start_id is None or end_id is None:
            msg = "Gemma parsing needs <|tool_call> tags in the vocab."
            raise ValueError(msg)
        self._tc_id = start_id
        self._tc_end_id = end_id

    def extract(
        self,
        token_ids: list[int],
        _tools: list[Any] | None = None,
    ) -> tuple[list[int], list[ParsedToolCall]]:
        """Parse native Gemma calls from ``token_ids``."""
        tc_start = _find(token_ids, self._tc_id)
        if tc_start == -1:
            return token_ids, []
        content_ids = token_ids[:tc_start]
        calls: list[ParsedToolCall] = []
        index = tc_start
        while index < len(token_ids):
            if token_ids[index] != self._tc_id:
                index += 1
                continue
            end = _find(token_ids, self._tc_end_id, index + 1)
            if end == -1:
                raw = _decode(self._tokenizer, token_ids[index + 1 :]).strip()
                calls.append(
                    ParsedToolCall(
                        raw=raw,
                        token_span=(index, len(token_ids)),
                        status=ToolCallParseStatus.UNCLOSED_BLOCK,
                    )
                )
                break
            raw = _decode(self._tokenizer, token_ids[index + 1 : end]).strip()
            calls.append(_parsed_gemma_call(raw, (index, end + 1)))
            index = end + 1
        return content_ids, calls


class HarmonyToolParser:
    """Channel blocks with ``to=functions.name`` recipients (GPT-OSS).

    Each ``<|start|>`` block carries its recipient in the header; blocks
    addressed to a function hold JSON arguments and close with ``<|call|>``.
    Analysis and final channels are content, never calls. Parsing truncates
    at the first ``<|return|>``.
    """

    REQUIRED_TOKENS: tuple[str, ...] = (
        "<|start|>",
        "<|message|>",
        "<|call|>",
        "<|end|>",
    )

    def __init__(self, tokenizer: ToolTokenizer) -> None:
        """Cache this tokenizer's channel IDs."""
        self._tokenizer = tokenizer
        ids = {token: _token_id(tokenizer, token) for token in self.REQUIRED_TOKENS}
        if any(value is None for value in ids.values()):
            msg = "Harmony parsing needs channel tags in the tokenizer vocab."
            raise ValueError(msg)
        self._start_id = ids["<|start|>"]
        self._message_id = ids["<|message|>"]
        self._call_id = ids["<|call|>"]
        self._end_id = ids["<|end|>"]
        self._return_id = _token_id(tokenizer, "<|return|>")

    def extract(
        self,
        token_ids: list[int],
        _tools: list[Any] | None = None,
    ) -> tuple[list[int], list[ParsedToolCall]]:
        """Parse function-addressed blocks from ``token_ids``."""
        ids = token_ids
        if self._return_id is not None:
            terminal = _find(ids, self._return_id)
            if terminal != -1:
                ids = ids[:terminal]
        calls: list[ParsedToolCall] = []
        first_call_start: int | None = None
        index = 0
        while index < len(ids):
            if ids[index] != self._start_id:
                index += 1
                continue
            block_start = index
            message_at = _find(ids, self._message_id, index + 1)
            if message_at == -1:
                break
            header = _decode(self._tokenizer, ids[index + 1 : message_at])
            body_end = len(ids)
            for stop in (
                _find(ids, self._start_id, message_at + 1),
                _find(ids, self._end_id, message_at + 1),
                _find(ids, self._call_id, message_at + 1),
            ):
                if stop != -1:
                    body_end = min(body_end, stop)
            closed = body_end < len(ids) and ids[body_end] in (
                self._end_id,
                self._call_id,
            )
            body = _decode(self._tokenizer, ids[message_at + 1 : body_end])
            recipient = _harmony_recipient(header)
            if recipient is not None and recipient.startswith("functions."):
                if first_call_start is None:
                    first_call_start = block_start
                span = (block_start, body_end + 1 if closed else body_end)
                calls.append(_parsed_harmony_call(body, recipient, span, closed))
            index = body_end
            if index < len(ids) and ids[index] in (self._end_id, self._call_id):
                index += 1
        content_end = first_call_start if first_call_start is not None else len(ids)
        return ids[:content_end], calls


TOOL_PARSERS: dict[str, type[ToolParser]] = {
    "tool_call": ToolCallTagParser,
    "granite3": Granite3ToolParser,
    "gemma4": Gemma4ToolParser,
    "harmony": HarmonyToolParser,
}


def get_tool_parser(name: str, tokenizer: ToolTokenizer) -> ToolParser:
    """Build the named parser over ``tokenizer``.

    :param name: Parser name from the registry.
    :param tokenizer: Tokenizer defining the delimiter IDs.
    :return: The parser.
    :rtype: ToolParser
    """
    try:
        parser_cls = TOOL_PARSERS[name]
    except KeyError:
        available = ", ".join(sorted(TOOL_PARSERS))
        msg = f"Unknown tool_parser {name!r}. Available: {available}."
        raise ValueError(msg) from None
    return parser_cls(tokenizer)


def detect_tool_parser(tokenizer: ToolTokenizer) -> ToolParser | None:
    """First registry parser whose tags all resolve, or ``None``.

    :param tokenizer: Tokenizer to probe.
    :return: The matching parser, or ``None`` when no grammar applies.
    :rtype: ToolParser | None
    """
    for parser_cls in TOOL_PARSERS.values():
        if all(
            _token_id(tokenizer, token) is not None
            for token in parser_cls.REQUIRED_TOKENS
        ):
            return parser_cls(tokenizer)
    return None


def _parsed_array_item(raw: str, item: object, span: tuple[int, int]) -> ParsedToolCall:
    """One Granite array item: ``OK``, ``MISSING_NAME``, or malformed."""
    if not isinstance(item, dict):
        return ParsedToolCall(
            raw=raw,
            token_span=span,
            status=ToolCallParseStatus.MALFORMED_STRUCTURE,
        )
    name = item.get("name")
    arguments = item.get("arguments")
    if not isinstance(name, str) or not name:
        return ParsedToolCall(
            raw=raw,
            token_span=span,
            status=ToolCallParseStatus.MISSING_NAME,
        )
    if not isinstance(arguments, dict):
        return ParsedToolCall(
            raw=raw,
            token_span=span,
            status=ToolCallParseStatus.MALFORMED_STRUCTURE,
        )
    return ParsedToolCall(
        raw=raw,
        name=name,
        arguments=arguments,
        token_span=span,
        status=ToolCallParseStatus.OK,
    )


def _parsed_gemma_call(raw: str, span: tuple[int, int]) -> ParsedToolCall:
    """One Gemma body: ``call:name{args}`` with custom-format args."""
    malformed = ParsedToolCall(
        raw=raw, token_span=span, status=ToolCallParseStatus.MALFORMED_STRUCTURE
    )
    if not raw.startswith("call:"):
        return malformed
    brace = raw.find("{", len("call:"))
    if brace == -1:
        return malformed
    name = raw[len("call:") : brace].strip()
    if not name:
        return ParsedToolCall(
            raw=raw, token_span=span, status=ToolCallParseStatus.MISSING_NAME
        )
    close = _matching_brace(raw, brace)
    if close == -1 or raw[close + 1 :].strip():
        return malformed
    try:
        arguments = _parse_gemma_args(raw[brace + 1 : close])
    except ValueError:
        return malformed
    return ParsedToolCall(
        raw=raw,
        name=name,
        arguments=arguments,
        token_span=span,
        status=ToolCallParseStatus.OK,
    )


GEMMA_STRING_DELIM = '<|"|>'


def _matching_brace(text: str, opening: int) -> int:
    """Index of the brace matching ``text[opening]``, skipping delim strings."""
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
    """Parse custom ``key:value`` args; strings stay bare for the executor."""
    args: dict[str, Any] = {}
    index = 0
    while index < len(text):
        while index < len(text) and text[index] in " ,\n\t":
            index += 1
        if index >= len(text):
            break
        key_start = index
        while index < len(text) and text[index] != ":":
            index += 1
        if index >= len(text):
            msg = "Gemma args end mid-key."
            raise ValueError(msg)
        key = text[key_start:index].strip()
        index += 1
        args[key], index = _parse_gemma_value(text, index)
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
    if index < len(text) and text[index] == "{":
        close = _matching_brace(text, index)
        if close == -1:
            msg = "Gemma args hold unbalanced braces."
            raise ValueError(msg)
        return _parse_gemma_args(text[index + 1 : close]), close + 1
    if index < len(text) and text[index] == "[":
        return _parse_gemma_array(text, index)
    start = index
    while index < len(text) and text[index] not in ",}]":
        index += 1
    return text[start:index].strip(), index


def _parse_gemma_array(text: str, index: int) -> tuple[list[Any], int]:
    """Parse one bracketed array at ``index``."""
    index += 1
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


def _harmony_recipient(header: str) -> str | None:
    """Addressee after ``to=`` in a Harmony header, or ``None``."""
    match = re.search(r"to=([^\s<]+)", header)
    return match.group(1) if match else None


def _parsed_harmony_call(
    body: str, recipient: str, span: tuple[int, int], closed: bool
) -> ParsedToolCall:
    """One function-addressed block: JSON args or a precise failure."""
    name = recipient[len("functions.") :]
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return ParsedToolCall(
            raw=body,
            name=name or None,
            token_span=span,
            status=ToolCallParseStatus.INVALID_JSON,
        )
    if not closed:
        return ParsedToolCall(
            raw=body,
            name=name or None,
            token_span=span,
            status=ToolCallParseStatus.UNCLOSED_BLOCK,
        )
    if not name:
        return ParsedToolCall(
            raw=body,
            token_span=span,
            status=ToolCallParseStatus.MISSING_NAME,
        )
    if not isinstance(payload, dict):
        return ParsedToolCall(
            raw=body,
            token_span=span,
            status=ToolCallParseStatus.MALFORMED_STRUCTURE,
        )
    return ParsedToolCall(
        raw=body,
        name=name,
        arguments=payload,
        token_span=span,
        status=ToolCallParseStatus.OK,
    )


def _parsed_json_call(raw: str, span: tuple[int, int]) -> ParsedToolCall:
    """One JSON body: ``OK``, ``MISSING_NAME``, or malformed."""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return ParsedToolCall(
            raw=raw,
            token_span=span,
            status=ToolCallParseStatus.INVALID_JSON,
        )
    if not isinstance(payload, dict):
        return ParsedToolCall(
            raw=raw,
            token_span=span,
            status=ToolCallParseStatus.MALFORMED_STRUCTURE,
        )
    name = payload.get("name")
    arguments = payload.get("arguments")
    if not isinstance(name, str) or not name:
        return ParsedToolCall(
            raw=raw,
            token_span=span,
            status=ToolCallParseStatus.MISSING_NAME,
        )
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = None
    if not isinstance(arguments, dict):
        return ParsedToolCall(
            raw=raw,
            token_span=span,
            status=ToolCallParseStatus.MALFORMED_STRUCTURE,
        )
    return ParsedToolCall(
        raw=raw,
        name=name,
        arguments=arguments,
        token_span=span,
        status=ToolCallParseStatus.OK,
    )


def _parsed_xml_call(
    raw: str,
    span: tuple[int, int],
    params: dict[str, dict[str, dict[str, Any]]],
) -> ParsedToolCall:
    """One XML function body, with schema-aware value coercion."""
    name_match = re.search(r"<function=([^>]+)>", raw)
    if not name_match:
        return ParsedToolCall(
            raw=raw,
            token_span=span,
            status=ToolCallParseStatus.MALFORMED_STRUCTURE,
        )
    name = name_match.group(1)
    schemas = params.get(name, {})
    arguments: dict[str, Any] = {}
    suspect_fallback = False
    for match in re.finditer(PARAMETER_RE, raw):
        value, used_fallback = _coerce_arg_value(
            match.group(2).strip(), schemas.get(match.group(1))
        )
        arguments[match.group(1)] = value
        suspect_fallback = suspect_fallback or used_fallback
    return ParsedToolCall(
        raw=raw,
        name=name,
        arguments=arguments,
        token_span=span,
        status=(
            ToolCallParseStatus.INVALID_JSON
            if suspect_fallback
            else ToolCallParseStatus.OK
        ),
    )


PARAMETER_RE = re.compile(r"<parameter=([^>]+)>\n?(.*?)\n?</" + "parameter>", re.DOTALL)


def _coerce_arg_value(
    text: str, param_schema: dict[str, Any] | None
) -> tuple[Any, bool]:
    """Coerce one raw XML value to its declared type.

    String-typed params stay verbatim; anything else tries JSON first and
    keeps the raw text on failure. Returns the value plus whether that
    fallback is suspect (schema bars strings).
    """
    string_allowed = False
    if param_schema is not None:
        declared = param_schema.get("type")
        if declared == "string" or declared == ["string"]:
            return text, False
        branches = param_schema.get("anyOf") or param_schema.get("oneOf") or []
        for branch in branches:
            if isinstance(branch, dict) and branch.get("type") == "string":
                string_allowed = True
    try:
        return json.loads(text), False
    except (json.JSONDecodeError, ValueError):
        return text, not string_allowed


def _build_param_type_index(
    tools: list[Any] | None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Map tool name to its parameter name to JSON-schema fragment."""
    if not tools:
        return {}
    index: dict[str, dict[str, dict[str, Any]]] = {}
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        spec = tool.get("function", tool)
        if not isinstance(spec, dict):
            continue
        name = spec.get("name")
        if not isinstance(name, str):
            continue
        parameters = spec.get("parameters") or {}
        properties = (
            parameters.get("properties") if isinstance(parameters, dict) else None
        )
        if isinstance(properties, dict):
            index[name] = {
                key: value
                for key, value in properties.items()
                if isinstance(value, dict)
            }
    return index


def _find(token_ids: list[int], target: int, start: int = 0) -> int:
    """Index of ``target`` at or after ``start``, or ``-1``."""
    for index in range(start, len(token_ids)):
        if token_ids[index] == target:
            return index
    return -1


def _decode(tokenizer: ToolTokenizer, token_ids: list[int]) -> str:
    """Decode IDs to text, keeping special tokens inside the segment."""
    if not token_ids:
        return ""
    return tokenizer.decode(token_ids, skip_special_tokens=False)


def _token_id(tokenizer: ToolTokenizer, token: str) -> int | None:
    """ID for ``token``, or ``None`` when the vocab lacks it."""
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if convert is None:
        return None
    token_id = convert(token)
    if token_id is None or token_id == getattr(tokenizer, "unk_token_id", None):
        return None
    return int(token_id)

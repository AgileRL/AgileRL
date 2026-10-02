# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Token-ID tool-call parsers: one grammar per delimiter family."""

from __future__ import annotations

import json
from typing import Any

import pytest

from agilerl.llm_envs.tool_parsers import (
    Gemma4ToolParser,
    Granite3ToolParser,
    HarmonyToolParser,
    ParsedToolCall,
    ToolCallParseStatus,
    ToolCallTagParser,
    detect_tool_parser,
    get_tool_parser,
)

TC, TC_END, THINK_END = 9000, 9001, 9002
GTC = 9100
MSTART, MEND = 9200, 9201
HSTART, HMSG, HCALL, HEND, HRET, HCHAN = 9300, 9301, 9302, 9303, 9304, 9305

_PARAM_CLOSE = "</" + "parameter>"


class FakeTokenizer:
    """Char IDs plus a tag table, looked up by token string."""

    def __init__(self, tags: dict[str, int]) -> None:
        self._table = dict(tags)
        self._reverse = {value: key for key, value in tags.items()}

    def convert_tokens_to_ids(self, token: str) -> int | None:
        return self._table.get(token)

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return "".join(self._reverse.get(i, chr(i)) for i in ids)


TAG_TOKENS = {"<tool_call>": TC, "</tool_call>": TC_END, "</think>": THINK_END}
GRANITE_TOKENS = {"<|tool_call|>": GTC}
GEMMA_TOKENS = {"<|tool_call>": MSTART, "<tool_call|>": MEND}
HARMONY_TOKENS = {
    "<|start|>": HSTART,
    "<|message|>": HMSG,
    "<|call|>": HCALL,
    "<|end|>": HEND,
    "<|return|>": HRET,
    "<|channel|>": HCHAN,
}


def chars(text: str) -> list[int]:
    """Text as char IDs."""
    return [ord(c) for c in text]


def json_block(name: str, arguments: dict[str, Any]) -> list[int]:
    """One JSON tool block as IDs."""
    payload = json.dumps({"name": name, "arguments": arguments})
    return [TC, *chars(payload), TC_END]


def xml_body(name: str, params: list[tuple[str, str]]) -> str:
    """One XML function body."""
    parts = [f"<function={name}>"]
    for key, value in params:
        parts.append(f"<parameter={key}>\n{value}\n{_PARAM_CLOSE}")
    parts.append("</function>")
    return "".join(parts)


def xml_block(name: str, params: list[tuple[str, str]]) -> list[int]:
    """One XML tool block as IDs."""
    return [TC, *chars(xml_body(name, params)), TC_END]


def granite_block(calls: list[Any]) -> list[int]:
    """One Granite array block as IDs."""
    return [GTC, *chars(json.dumps(calls))]


def gemma_block(name: str, args: str) -> list[int]:
    """One Gemma call as IDs."""
    return [MSTART, *chars(f"call:{name}{args}"), MEND]


def harmony_block(header: str, body: str, closer: int) -> list[int]:
    """One Harmony message block as IDs."""
    return [HSTART, *chars(header), HMSG, *chars(body), closer]


class TestToolCallTagParserJson:
    def test_parses_single_call(self) -> None:
        parser = ToolCallTagParser(FakeTokenizer(TAG_TOKENS))
        ids = json_block("add", {"a": 2})

        _, calls = parser.extract(ids)

        assert calls == [
            ParsedToolCall(
                raw='{"name": "add", "arguments": {"a": 2}}',
                name="add",
                arguments={"a": 2},
                token_span=(0, len(ids)),
                status=ToolCallParseStatus.OK,
            )
        ]

    def test_returns_content_before_first_call(self) -> None:
        parser = ToolCallTagParser(FakeTokenizer(TAG_TOKENS))
        ids = chars("hi") + json_block("add", {})

        content, calls = parser.extract(ids)

        assert content == chars("hi")
        assert len(calls) == 1

    def test_no_tags_returns_empty(self) -> None:
        parser = ToolCallTagParser(FakeTokenizer(TAG_TOKENS))

        content, calls = parser.extract(chars("ab"))

        assert (content, calls) == (chars("ab"), [])

    def test_literal_tag_text_does_not_parse(self) -> None:
        parser = ToolCallTagParser(FakeTokenizer(TAG_TOKENS))
        ids = chars("<tool_call>not a call</tool_call>")

        _, calls = parser.extract(ids)

        assert calls == []

    def test_parses_several_calls_in_order(self) -> None:
        parser = ToolCallTagParser(FakeTokenizer(TAG_TOKENS))
        ids = json_block("a", {}) + json_block("b", {"x": 1})

        _, calls = parser.extract(ids)

        assert [call.name for call in calls] == ["a", "b"]
        assert all(call.status is ToolCallParseStatus.OK for call in calls)

    def test_invalid_json_reports_status(self) -> None:
        parser = ToolCallTagParser(FakeTokenizer(TAG_TOKENS))
        ids = [TC, *chars("{bad}"), TC_END]

        _, calls = parser.extract(ids)

        assert len(calls) == 1
        assert calls[0].status is ToolCallParseStatus.INVALID_JSON
        assert calls[0].raw == "{bad}"

    def test_unclosed_block_reports_status(self) -> None:
        parser = ToolCallTagParser(FakeTokenizer(TAG_TOKENS))
        ids = [TC, *chars('{"a": 1}')]

        _, calls = parser.extract(ids)

        assert len(calls) == 1
        assert calls[0].status is ToolCallParseStatus.UNCLOSED_BLOCK

    def test_missing_name_reports_status(self) -> None:
        parser = ToolCallTagParser(FakeTokenizer(TAG_TOKENS))
        ids = [TC, *chars(json.dumps({"arguments": {}})), TC_END]

        _, calls = parser.extract(ids)

        assert calls[0].status is ToolCallParseStatus.MISSING_NAME

    def test_non_object_json_reports_malformed(self) -> None:
        parser = ToolCallTagParser(FakeTokenizer(TAG_TOKENS))
        ids = [TC, *chars("[1, 2]"), TC_END]

        _, calls = parser.extract(ids)

        assert calls[0].status is ToolCallParseStatus.MALFORMED_STRUCTURE

    def test_non_object_arguments_reports_malformed(self) -> None:
        parser = ToolCallTagParser(FakeTokenizer(TAG_TOKENS))
        payload = json.dumps({"name": "g", "arguments": "7"})
        ids = [TC, *chars(payload), TC_END]

        _, calls = parser.extract(ids)

        assert calls[0].status is ToolCallParseStatus.MALFORMED_STRUCTURE

    def test_json_string_arguments_are_loaded(self) -> None:
        parser = ToolCallTagParser(FakeTokenizer(TAG_TOKENS))
        payload = json.dumps({"name": "add", "arguments": json.dumps({"a": 2})})
        ids = [TC, *chars(payload), TC_END]

        _, calls = parser.extract(ids)

        assert calls[0].status is ToolCallParseStatus.OK
        assert calls[0].name == "add"
        assert calls[0].arguments == {"a": 2}

    def test_unparseable_string_arguments_report_malformed(self) -> None:
        parser = ToolCallTagParser(FakeTokenizer(TAG_TOKENS))
        payload = json.dumps({"name": "g", "arguments": "{"})
        ids = [TC, *chars(payload), TC_END]

        _, calls = parser.extract(ids)

        assert calls[0].status is ToolCallParseStatus.MALFORMED_STRUCTURE

    def test_think_drafts_are_skipped(self) -> None:
        parser = ToolCallTagParser(FakeTokenizer(TAG_TOKENS))
        ids = [*json_block("draft", {}), THINK_END, *json_block("real", {})]

        _, calls = parser.extract(ids)

        assert [call.name for call in calls] == ["real"]

    def test_missing_tag_vocab_raises(self) -> None:
        with pytest.raises(ValueError, match="<tool_call>"):
            ToolCallTagParser(FakeTokenizer({}))


class TestToolCallTagParserXml:
    def test_parses_xml_body(self) -> None:
        parser = ToolCallTagParser(FakeTokenizer(TAG_TOKENS))
        ids = xml_block("add", [("a", "2"), ("b", "3")])

        _, calls = parser.extract(ids)

        assert len(calls) == 1
        assert calls[0].name == "add"
        assert calls[0].arguments == {"a": 2, "b": 3}
        assert calls[0].status is ToolCallParseStatus.OK

    def test_plain_values_stay_strings(self) -> None:
        parser = ToolCallTagParser(FakeTokenizer(TAG_TOKENS))
        ids = xml_block("greet", [("name", "ada")])
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "greet",
                    "parameters": {"properties": {"name": {"type": "string"}}},
                },
            }
        ]

        _, calls = parser.extract(ids, tools)

        assert calls[0].arguments == {"name": "ada"}
        assert calls[0].status is ToolCallParseStatus.OK

    def test_untyped_string_without_schema_reports_invalid(self) -> None:
        # Without a schema a failed load is indistinguishable from breakage.
        parser = ToolCallTagParser(FakeTokenizer(TAG_TOKENS))
        ids = xml_block("greet", [("name", "ada")])

        _, calls = parser.extract(ids)

        assert calls[0].status is ToolCallParseStatus.INVALID_JSON

    def test_schema_keeps_string_true_verbatim(self) -> None:
        parser = ToolCallTagParser(FakeTokenizer(TAG_TOKENS))
        ids = xml_block("set", [("flag", "true")])
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "set",
                    "parameters": {"properties": {"flag": {"type": "string"}}},
                },
            }
        ]

        _, calls = parser.extract(ids, tools)

        assert calls[0].arguments == {"flag": "true"}
        assert calls[0].status is ToolCallParseStatus.OK

    def test_schema_violation_reports_invalid(self) -> None:
        parser = ToolCallTagParser(FakeTokenizer(TAG_TOKENS))
        ids = xml_block("add", [("a", "not-a-number")])
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "add",
                    "parameters": {"properties": {"a": {"type": "integer"}}},
                },
            }
        ]

        _, calls = parser.extract(ids, tools)

        assert calls[0].status is ToolCallParseStatus.INVALID_JSON

    def test_missing_function_tag_reports_malformed(self) -> None:
        parser = ToolCallTagParser(FakeTokenizer(TAG_TOKENS))
        ids = [TC, *chars("<function=oops"), TC_END]

        _, calls = parser.extract(ids)

        assert calls[0].status is ToolCallParseStatus.MALFORMED_STRUCTURE


class TestGranite3ToolParser:
    def test_parses_array_of_calls(self) -> None:
        parser = Granite3ToolParser(FakeTokenizer(GRANITE_TOKENS))
        ids = granite_block(
            [
                {"name": "a", "arguments": {}},
                {"name": "b", "arguments": {"x": 1}},
            ]
        )

        _, calls = parser.extract(ids)

        assert [call.name for call in calls] == ["a", "b"]
        assert all(call.status is ToolCallParseStatus.OK for call in calls)

    def test_no_tag_returns_empty(self) -> None:
        parser = Granite3ToolParser(FakeTokenizer(GRANITE_TOKENS))

        _, calls = parser.extract(chars("answer"))

        assert calls == []

    def test_leading_prose_reports_malformed(self) -> None:
        parser = Granite3ToolParser(FakeTokenizer(GRANITE_TOKENS))
        ids = chars("hmm ") + granite_block([{"name": "a", "arguments": {}}])

        _, calls = parser.extract(ids)

        assert calls[0].status is ToolCallParseStatus.MALFORMED_STRUCTURE

    def test_non_array_reports_malformed(self) -> None:
        parser = Granite3ToolParser(FakeTokenizer(GRANITE_TOKENS))
        ids = [GTC, *chars(json.dumps({"name": "a"}))]

        _, calls = parser.extract(ids)

        assert calls[0].status is ToolCallParseStatus.MALFORMED_STRUCTURE

    def test_bad_json_reports_invalid(self) -> None:
        parser = Granite3ToolParser(FakeTokenizer(GRANITE_TOKENS))
        ids = [GTC, *chars("[{bad}]")]

        _, calls = parser.extract(ids)

        assert calls[0].status is ToolCallParseStatus.INVALID_JSON

    def test_missing_tag_vocab_raises(self) -> None:
        with pytest.raises(ValueError, match=r"<\|tool_call\|>"):
            Granite3ToolParser(FakeTokenizer({}))


class TestGemma4ToolParser:
    def test_parses_call_with_string_and_bare_args(self) -> None:
        parser = Gemma4ToolParser(FakeTokenizer(GEMMA_TOKENS))
        ids = gemma_block("add", '{a:<|"|>x<|"|>,b:3}')

        _, calls = parser.extract(ids)

        assert len(calls) == 1
        assert calls[0].name == "add"
        assert calls[0].arguments == {"a": "x", "b": "3"}
        assert calls[0].status is ToolCallParseStatus.OK

    def test_parses_nested_args(self) -> None:
        parser = Gemma4ToolParser(FakeTokenizer(GEMMA_TOKENS))
        ids = gemma_block("f", '{o:{k:<|"|>v<|"|>},l:[<|"|>a<|"|>]}')

        _, calls = parser.extract(ids)

        assert calls[0].arguments == {"o": {"k": "v"}, "l": ["a"]}
        assert calls[0].status is ToolCallParseStatus.OK

    def test_no_tags_returns_empty(self) -> None:
        parser = Gemma4ToolParser(FakeTokenizer(GEMMA_TOKENS))

        _, calls = parser.extract(chars("answer"))

        assert calls == []

    def test_missing_call_prefix_reports_malformed(self) -> None:
        parser = Gemma4ToolParser(FakeTokenizer(GEMMA_TOKENS))
        ids = [MSTART, *chars("add{a:1}"), MEND]

        _, calls = parser.extract(ids)

        assert calls[0].status is ToolCallParseStatus.MALFORMED_STRUCTURE

    def test_missing_name_reports_status(self) -> None:
        parser = Gemma4ToolParser(FakeTokenizer(GEMMA_TOKENS))
        ids = [MSTART, *chars("call:{a:1}"), MEND]

        _, calls = parser.extract(ids)

        assert calls[0].status is ToolCallParseStatus.MISSING_NAME

    def test_unbalanced_braces_reports_malformed(self) -> None:
        parser = Gemma4ToolParser(FakeTokenizer(GEMMA_TOKENS))
        ids = [MSTART, *chars("call:add{a:{b:1}"), MEND]

        _, calls = parser.extract(ids)

        assert calls[0].status is ToolCallParseStatus.MALFORMED_STRUCTURE

    def test_unclosed_block_reports_status(self) -> None:
        parser = Gemma4ToolParser(FakeTokenizer(GEMMA_TOKENS))
        ids = [MSTART, *chars("call:add{a:1}")]

        _, calls = parser.extract(ids)

        assert calls[0].status is ToolCallParseStatus.UNCLOSED_BLOCK

    def test_missing_tag_vocab_raises(self) -> None:
        with pytest.raises(ValueError, match="Gemma"):
            Gemma4ToolParser(FakeTokenizer({}))


class TestHarmonyToolParser:
    def test_parses_function_block(self) -> None:
        parser = HarmonyToolParser(FakeTokenizer(HARMONY_TOKENS))
        ids = harmony_block("assistant to=functions.add", '{"a": 2}', HCALL)

        _, calls = parser.extract(ids)

        assert len(calls) == 1
        assert calls[0].name == "add"
        assert calls[0].arguments == {"a": 2}
        assert calls[0].status is ToolCallParseStatus.OK

    def test_ignores_non_function_channels(self) -> None:
        parser = HarmonyToolParser(FakeTokenizer(HARMONY_TOKENS))
        analysis = harmony_block("assistant to=commentary", "thinking", HEND)
        final = harmony_block("assistant to=user", "answer", HEND)
        call = harmony_block("assistant to=functions.add", '{"a": 1}', HCALL)

        _, calls = parser.extract(analysis + final + call)

        assert [call.name for call in calls] == ["add"]

    def test_channel_tag_in_header_still_parses(self) -> None:
        parser = HarmonyToolParser(FakeTokenizer(HARMONY_TOKENS))
        ids = [
            HSTART,
            *chars("assistant to=functions.add "),
            HCHAN,
            *chars("commentary"),
            HMSG,
            *chars('{"a": 1}'),
            HCALL,
        ]

        _, calls = parser.extract(ids)

        assert len(calls) == 1
        assert calls[0].name == "add"

    def test_invalid_json_reports_status_with_name(self) -> None:
        parser = HarmonyToolParser(FakeTokenizer(HARMONY_TOKENS))
        ids = harmony_block("assistant to=functions.add", "{bad}", HCALL)

        _, calls = parser.extract(ids)

        assert calls[0].status is ToolCallParseStatus.INVALID_JSON
        assert calls[0].name == "add"

    def test_unclosed_block_reports_status(self) -> None:
        parser = HarmonyToolParser(FakeTokenizer(HARMONY_TOKENS))
        ids = [HSTART, *chars("assistant to=functions.add"), HMSG, *chars('{"a": 1}')]

        _, calls = parser.extract(ids)

        assert calls[0].status is ToolCallParseStatus.UNCLOSED_BLOCK

    def test_return_truncates_later_blocks(self) -> None:
        parser = HarmonyToolParser(FakeTokenizer(HARMONY_TOKENS))
        first = harmony_block("assistant to=functions.a", "{}", HCALL)
        second = harmony_block("assistant to=functions.b", "{}", HCALL)

        _, calls = parser.extract([*first, HRET, *second])

        assert [call.name for call in calls] == ["a"]

    def test_missing_tag_vocab_raises(self) -> None:
        with pytest.raises(ValueError, match="Harmony"):
            HarmonyToolParser(FakeTokenizer({}))


class TestDetectToolParser:
    def test_detects_each_family(self) -> None:
        assert isinstance(
            detect_tool_parser(FakeTokenizer(TAG_TOKENS)), ToolCallTagParser
        )
        assert isinstance(
            detect_tool_parser(FakeTokenizer(GRANITE_TOKENS)), Granite3ToolParser
        )
        assert isinstance(
            detect_tool_parser(FakeTokenizer(GEMMA_TOKENS)), Gemma4ToolParser
        )
        assert isinstance(
            detect_tool_parser(FakeTokenizer(HARMONY_TOKENS)), HarmonyToolParser
        )

    def test_returns_none_without_tags(self) -> None:
        assert detect_tool_parser(FakeTokenizer({})) is None

    def test_first_registry_match_wins(self) -> None:
        merged = {**TAG_TOKENS, **GRANITE_TOKENS}

        assert isinstance(detect_tool_parser(FakeTokenizer(merged)), ToolCallTagParser)


class TestGetToolParser:
    def test_builds_each_registered_parser(self) -> None:
        assert isinstance(
            get_tool_parser("tool_call", FakeTokenizer(TAG_TOKENS)),
            ToolCallTagParser,
        )
        assert isinstance(
            get_tool_parser("granite3", FakeTokenizer(GRANITE_TOKENS)),
            Granite3ToolParser,
        )
        assert isinstance(
            get_tool_parser("gemma4", FakeTokenizer(GEMMA_TOKENS)), Gemma4ToolParser
        )
        assert isinstance(
            get_tool_parser("harmony", FakeTokenizer(HARMONY_TOKENS)),
            HarmonyToolParser,
        )

    def test_unknown_name_lists_available(self) -> None:
        with pytest.raises(ValueError, match="tool_call"):
            get_tool_parser("llama", FakeTokenizer(TAG_TOKENS))

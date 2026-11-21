# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Token-ID tool-call parsers: one per model family's tool-call format."""

from __future__ import annotations

import json
import re
from typing import Any

import pytest
from jinja2.exceptions import TemplateError
from transformers.utils.chat_template_utils import render_jinja_template

from agilerl.llm_envs.chat_formats import (
    DSML,
    DSML_REASONING_EFFORT_PROMPTS,
    DSML_TOOLS_PROMPT_HEAD,
    DSML_TOOLS_PROMPT_TAIL,
    DsmlFormat,
    Gemma4Format,
    Glm47Format,
    Granite3Format,
    HarmonyFormat,
    ParsedToolCall,
    ToolCallParseStatus,
    ToolCallTagFormat,
    action_text,
    detect_chat_format,
    get_chat_format,
    resolve_chat_format,
)

TC, TC_END, THINK_END = 9000, 9001, 9002
GTC, GEOT = 9100, 9101
MSTART, MEND, MRESP_START, MRESP_END, MCHAN_END = 9200, 9201, 9202, 9203, 9204
HSTART, HMSG, HCALL, HEND, HRET, HCHAN = 9300, 9301, 9302, 9303, 9304, 9305
GKEY, GKEY_END, GVALUE, GVALUE_END = 9400, 9401, 9402, 9403
DSML_ID, DEOS = 9500, 9501
USER, ASSISTANT = "<\uff5cUser\uff5c>", "<\uff5cAssistant\uff5c>"
DSML_BOS = "<\uff5cbegin▁of▁sentence\uff5c>"
DSML_EOS = "<\uff5cend▁of▁sentence\uff5c>"

_PARAM_CLOSE = "</" + "parameter>"


class FakeTokenizer:
    """Char IDs plus a tag table, looked up by token string."""

    unk_token_id = None

    def __init__(self, tags: dict[str, int]) -> None:
        self._table = dict(tags)
        self._reverse = {value: key for key, value in tags.items()}

    def convert_tokens_to_ids(self, token: str) -> int | None:
        return self._table.get(token)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        tags = sorted(self._table, key=len, reverse=True)
        parts = (
            re.split(f"({'|'.join(map(re.escape, tags))})", text) if tags else [text]
        )
        return [
            token_id
            for part in parts
            for token_id in (
                [self._table[part]] if part in self._table else chars(part)
            )
        ]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        parts = []
        for i in ids:
            if i not in self._reverse:
                parts.append(chr(i))
            elif not skip_special_tokens:
                parts.append(self._reverse[i])
        return "".join(parts)


TAG_TOKENS = {"<tool_call>": TC, "</tool_call>": TC_END, "</think>": THINK_END}
GRANITE_TOKENS = {"<|tool_call|>": GTC, "<|end_of_text|>": GEOT}
GEMMA_TOKENS = {
    "<|tool_call>": MSTART,
    "<tool_call|>": MEND,
    "<|tool_response>": MRESP_START,
    "<tool_response|>": MRESP_END,
    "<channel|>": MCHAN_END,
}
HARMONY_TOKENS = {
    "<|start|>": HSTART,
    "<|message|>": HMSG,
    "<|call|>": HCALL,
    "<|end|>": HEND,
    "<|return|>": HRET,
    "<|channel|>": HCHAN,
}
GLM_TOKENS = {
    **TAG_TOKENS,
    "<arg_key>": GKEY,
    "</arg_key>": GKEY_END,
    "<arg_value>": GVALUE,
    "</arg_value>": GVALUE_END,
}
DSML_TOKENS = {DSML: DSML_ID, "</think>": THINK_END, DSML_EOS: DEOS}
WEATHER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer"},
                },
            },
        },
    }
]


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


def glm_ids(text: str) -> list[int]:
    """GLM text as IDs, each tag one token."""
    return FakeTokenizer(GLM_TOKENS).encode(text)


def dsml_invoke(name: str, params: list[tuple[str, str, str]]) -> str:
    """One DSML invoke; each param is ``(key, string flag, value)``."""
    lines = [
        f'<{DSML}parameter name="{key}" string="{flag}">{value}</{DSML}parameter>'
        for key, flag, value in params
    ]
    return f'<{DSML}invoke name="{name}">\n' + "\n".join(lines) + f"\n</{DSML}invoke>"


def dsml_block(*invokes: str) -> str:
    """One DSML tool-call block."""
    return f"<{DSML}tool_calls>\n" + "\n".join(invokes) + f"\n</{DSML}tool_calls>"


def dsml_ids(text: str) -> list[int]:
    """DeepSeek-V4 text as IDs: :data:`DSML` is one token, the rest of each tag is chars."""
    return FakeTokenizer(DSML_TOKENS).encode(text)


def render_dsml(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> str:
    """Render ``messages`` through the DeepSeek-V4 chat template with the generation prompt."""
    rendered, _ = render_jinja_template(
        conversations=[messages],
        tools=tools,
        chat_template=DsmlFormat.CHAT_TEMPLATE,
        add_generation_prompt=True,
        **kwargs,
    )
    return rendered[0]


class TestToolCallTagFormatJson:
    def test_parses_single_call(self) -> None:
        parser = ToolCallTagFormat(FakeTokenizer(TAG_TOKENS))
        ids = json_block("add", {"a": 2})

        _, calls = parser.extract_tool_calls(ids)

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
        parser = ToolCallTagFormat(FakeTokenizer(TAG_TOKENS))
        ids = chars("hi") + json_block("add", {})

        content, calls = parser.extract_tool_calls(ids)

        assert content == chars("hi")
        assert len(calls) == 1

    def test_no_tags_returns_empty(self) -> None:
        parser = ToolCallTagFormat(FakeTokenizer(TAG_TOKENS))

        content, calls = parser.extract_tool_calls(chars("ab"))

        assert (content, calls) == (chars("ab"), [])

    def test_literal_tag_text_does_not_parse(self) -> None:
        parser = ToolCallTagFormat(FakeTokenizer(TAG_TOKENS))
        ids = chars("<tool_call>not a call</tool_call>")

        _, calls = parser.extract_tool_calls(ids)

        assert calls == []

    def test_parses_several_calls_in_order(self) -> None:
        parser = ToolCallTagFormat(FakeTokenizer(TAG_TOKENS))
        ids = json_block("a", {}) + json_block("b", {"x": 1})

        _, calls = parser.extract_tool_calls(ids)

        assert [call.name for call in calls] == ["a", "b"]
        assert all(call.status is ToolCallParseStatus.OK for call in calls)

    def test_prose_between_and_after_calls_is_skipped(self) -> None:
        parser = ToolCallTagFormat(FakeTokenizer(TAG_TOKENS))
        ids = json_block("a", {}) + chars(" and ") + json_block("b", {}) + chars("!")

        _, calls = parser.extract_tool_calls(ids)

        assert [call.name for call in calls] == ["a", "b"]
        assert all(call.status is ToolCallParseStatus.OK for call in calls)

    def test_invalid_json_reports_status(self) -> None:
        parser = ToolCallTagFormat(FakeTokenizer(TAG_TOKENS))
        ids = [TC, *chars("{bad}"), TC_END]

        _, calls = parser.extract_tool_calls(ids)

        assert len(calls) == 1
        assert calls[0].status is ToolCallParseStatus.INVALID_JSON
        assert calls[0].raw == "{bad}"

    def test_unclosed_block_reports_status(self) -> None:
        parser = ToolCallTagFormat(FakeTokenizer(TAG_TOKENS))
        ids = [TC, *chars('{"a": 1}')]

        _, calls = parser.extract_tool_calls(ids)

        assert len(calls) == 1
        assert calls[0].status is ToolCallParseStatus.UNCLOSED_BLOCK

    def test_missing_name_reports_status(self) -> None:
        parser = ToolCallTagFormat(FakeTokenizer(TAG_TOKENS))
        ids = [TC, *chars(json.dumps({"arguments": {}})), TC_END]

        _, calls = parser.extract_tool_calls(ids)

        assert calls[0].status is ToolCallParseStatus.MISSING_NAME

    def test_non_object_json_reports_malformed(self) -> None:
        parser = ToolCallTagFormat(FakeTokenizer(TAG_TOKENS))
        ids = [TC, *chars("[1, 2]"), TC_END]

        _, calls = parser.extract_tool_calls(ids)

        assert calls[0].status is ToolCallParseStatus.MALFORMED_STRUCTURE

    def test_non_object_arguments_reports_malformed(self) -> None:
        parser = ToolCallTagFormat(FakeTokenizer(TAG_TOKENS))
        payload = json.dumps({"name": "g", "arguments": "7"})
        ids = [TC, *chars(payload), TC_END]

        _, calls = parser.extract_tool_calls(ids)

        assert calls[0].status is ToolCallParseStatus.MALFORMED_STRUCTURE

    def test_json_string_arguments_are_loaded(self) -> None:
        parser = ToolCallTagFormat(FakeTokenizer(TAG_TOKENS))
        payload = json.dumps({"name": "add", "arguments": json.dumps({"a": 2})})
        ids = [TC, *chars(payload), TC_END]

        _, calls = parser.extract_tool_calls(ids)

        assert calls[0].status is ToolCallParseStatus.OK
        assert calls[0].name == "add"
        assert calls[0].arguments == {"a": 2}

    def test_unparseable_string_arguments_report_malformed(self) -> None:
        parser = ToolCallTagFormat(FakeTokenizer(TAG_TOKENS))
        payload = json.dumps({"name": "g", "arguments": "{"})
        ids = [TC, *chars(payload), TC_END]

        _, calls = parser.extract_tool_calls(ids)

        assert calls[0].status is ToolCallParseStatus.MALFORMED_STRUCTURE

    def test_think_drafts_are_skipped(self) -> None:
        parser = ToolCallTagFormat(FakeTokenizer(TAG_TOKENS))
        ids = [*json_block("draft", {}), THINK_END, *json_block("real", {})]

        _, calls = parser.extract_tool_calls(ids)

        assert [call.name for call in calls] == ["real"]

    def test_missing_tag_vocab_raises(self) -> None:
        with pytest.raises(ValueError, match="<tool_call>"):
            ToolCallTagFormat(FakeTokenizer({}))


class TestToolCallTagFormatXml:
    def test_parses_xml_body(self) -> None:
        parser = ToolCallTagFormat(FakeTokenizer(TAG_TOKENS))
        ids = xml_block("add", [("a", "2"), ("b", "3")])

        _, calls = parser.extract_tool_calls(ids)

        assert len(calls) == 1
        assert calls[0].name == "add"
        assert calls[0].arguments == {"a": 2, "b": 3}
        assert calls[0].status is ToolCallParseStatus.OK

    def test_plain_values_stay_strings(self) -> None:
        parser = ToolCallTagFormat(FakeTokenizer(TAG_TOKENS))
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

        _, calls = parser.extract_tool_calls(ids, tools)

        assert calls[0].arguments == {"name": "ada"}
        assert calls[0].status is ToolCallParseStatus.OK

    def test_untyped_string_without_schema_reports_invalid(self) -> None:
        # Without a schema a failed load is indistinguishable from breakage.
        parser = ToolCallTagFormat(FakeTokenizer(TAG_TOKENS))
        ids = xml_block("greet", [("name", "ada")])

        _, calls = parser.extract_tool_calls(ids)

        assert calls[0].status is ToolCallParseStatus.INVALID_JSON

    def test_schema_keeps_string_true_verbatim(self) -> None:
        parser = ToolCallTagFormat(FakeTokenizer(TAG_TOKENS))
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

        _, calls = parser.extract_tool_calls(ids, tools)

        assert calls[0].arguments == {"flag": "true"}
        assert calls[0].status is ToolCallParseStatus.OK

    def test_schema_violation_reports_invalid(self) -> None:
        parser = ToolCallTagFormat(FakeTokenizer(TAG_TOKENS))
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

        _, calls = parser.extract_tool_calls(ids, tools)

        assert calls[0].status is ToolCallParseStatus.INVALID_JSON

    def test_missing_function_tag_reports_malformed(self) -> None:
        parser = ToolCallTagFormat(FakeTokenizer(TAG_TOKENS))
        ids = [TC, *chars("<function=oops"), TC_END]

        _, calls = parser.extract_tool_calls(ids)

        assert calls[0].status is ToolCallParseStatus.MALFORMED_STRUCTURE

    def test_any_of_string_branch_keeps_value_verbatim(self) -> None:
        parser = ToolCallTagFormat(FakeTokenizer(TAG_TOKENS))
        ids = xml_block("greet", [("name", "ada")])
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "greet",
                    "parameters": {
                        "properties": {
                            "name": {"anyOf": [{"type": "integer"}, {"type": "string"}]}
                        }
                    },
                },
            }
        ]

        _, calls = parser.extract_tool_calls(ids, tools)

        assert calls[0].arguments == {"name": "ada"}
        assert calls[0].status is ToolCallParseStatus.OK

    def test_non_dict_tool_entries_are_ignored(self) -> None:
        parser = ToolCallTagFormat(FakeTokenizer(TAG_TOKENS))
        ids = xml_block("add", [("a", "2")])
        tools: list[Any] = [42, {"function": "xx"}, {"function": {"name": 42}}]

        _, calls = parser.extract_tool_calls(ids, tools)

        assert calls[0].arguments == {"a": 2}
        assert calls[0].status is ToolCallParseStatus.OK


class UnkEndOfTextTokenizer(FakeTokenizer):
    """Granite's vocab: ``<|end_of_text|>`` is also the unk token."""

    unk_token = "<|end_of_text|>"
    unk_token_id = GEOT


class TestGranite3Format:
    def test_end_of_text_that_is_the_unk_token_still_ends_the_array(self) -> None:
        parser = Granite3Format(UnkEndOfTextTokenizer(GRANITE_TOKENS))
        ids = [*granite_block([{"name": "a", "arguments": {"x": 1}}]), GEOT]

        _, calls = parser.extract_tool_calls(ids)

        assert [(call.name, call.status) for call in calls] == [
            ("a", ToolCallParseStatus.OK)
        ]

    def test_formats_calls_as_one_tagged_json_array(self) -> None:
        parser = Granite3Format(FakeTokenizer(GRANITE_TOKENS))

        text = parser.format_tool_calls([{"name": "a", "arguments": {"x": 1}}])

        assert text == '<|tool_call|>[{"name": "a", "arguments": {"x": 1}}]'

    def test_parses_array_of_calls(self) -> None:
        parser = Granite3Format(FakeTokenizer(GRANITE_TOKENS))
        ids = granite_block(
            [
                {"name": "a", "arguments": {}},
                {"name": "b", "arguments": {"x": 1}},
            ]
        )

        _, calls = parser.extract_tool_calls(ids)

        assert [call.name for call in calls] == ["a", "b"]
        assert all(call.status is ToolCallParseStatus.OK for call in calls)

    def test_end_of_turn_token_ends_the_array(self) -> None:
        parser = Granite3Format(FakeTokenizer(GRANITE_TOKENS))
        ids = [*granite_block([{"name": "a", "arguments": {"x": 1}}]), GEOT]

        _, calls = parser.extract_tool_calls(ids)

        assert [(call.name, call.status) for call in calls] == [
            ("a", ToolCallParseStatus.OK)
        ]

    def test_no_tag_returns_empty(self) -> None:
        parser = Granite3Format(FakeTokenizer(GRANITE_TOKENS))

        _, calls = parser.extract_tool_calls(chars("answer"))

        assert calls == []

    def test_leading_prose_reports_malformed(self) -> None:
        parser = Granite3Format(FakeTokenizer(GRANITE_TOKENS))
        ids = chars("hmm ") + granite_block([{"name": "a", "arguments": {}}])

        _, calls = parser.extract_tool_calls(ids)

        assert calls[0].status is ToolCallParseStatus.MALFORMED_STRUCTURE

    def test_non_array_reports_malformed(self) -> None:
        parser = Granite3Format(FakeTokenizer(GRANITE_TOKENS))
        ids = [GTC, *chars(json.dumps({"name": "a"}))]

        _, calls = parser.extract_tool_calls(ids)

        assert calls[0].status is ToolCallParseStatus.MALFORMED_STRUCTURE

    def test_bad_json_reports_invalid(self) -> None:
        parser = Granite3Format(FakeTokenizer(GRANITE_TOKENS))
        ids = [GTC, *chars("[{bad}]")]

        _, calls = parser.extract_tool_calls(ids)

        assert calls[0].status is ToolCallParseStatus.INVALID_JSON

    def test_non_dict_item_reports_malformed(self) -> None:
        parser = Granite3Format(FakeTokenizer(GRANITE_TOKENS))
        ids = granite_block([{"name": "a", "arguments": {}}, "junk"])

        _, calls = parser.extract_tool_calls(ids)

        assert calls[0].status is ToolCallParseStatus.OK
        assert calls[1].status is ToolCallParseStatus.MALFORMED_STRUCTURE

    def test_item_missing_name_reports_status(self) -> None:
        parser = Granite3Format(FakeTokenizer(GRANITE_TOKENS))
        ids = granite_block([{"arguments": {}}])

        _, calls = parser.extract_tool_calls(ids)

        assert calls[0].status is ToolCallParseStatus.MISSING_NAME

    def test_non_dict_arguments_report_malformed(self) -> None:
        parser = Granite3Format(FakeTokenizer(GRANITE_TOKENS))
        ids = granite_block([{"name": "a", "arguments": [1]}])

        _, calls = parser.extract_tool_calls(ids)

        assert calls[0].status is ToolCallParseStatus.MALFORMED_STRUCTURE

    def test_missing_tag_vocab_raises(self) -> None:
        with pytest.raises(ValueError, match=r"<\|tool_call\|>"):
            Granite3Format(FakeTokenizer({}))


class TestGemma4Format:
    def test_calls_inside_the_thought_channel_are_skipped(self) -> None:
        parser = Gemma4Format(FakeTokenizer(GEMMA_TOKENS))
        ids = [
            *gemma_block("draft", "{}"),
            MCHAN_END,
            *gemma_block("real", "{}"),
        ]

        _, calls = parser.extract_tool_calls(ids)

        assert [call.name for call in calls] == ["real"]

    def test_parses_call_with_string_and_bare_args(self) -> None:
        parser = Gemma4Format(FakeTokenizer(GEMMA_TOKENS))
        ids = gemma_block("add", '{a:<|"|>x<|"|>,b:3}')

        _, calls = parser.extract_tool_calls(ids)

        assert len(calls) == 1
        assert calls[0].name == "add"
        assert calls[0].arguments == {"a": "x", "b": "3"}
        assert calls[0].status is ToolCallParseStatus.OK

    def test_parses_nested_args(self) -> None:
        parser = Gemma4Format(FakeTokenizer(GEMMA_TOKENS))
        ids = gemma_block("f", '{o:{k:<|"|>v<|"|>},l:[<|"|>a<|"|>]}')

        _, calls = parser.extract_tool_calls(ids)

        assert calls[0].arguments == {"o": {"k": "v"}, "l": ["a"]}
        assert calls[0].status is ToolCallParseStatus.OK

    def test_no_tags_returns_empty(self) -> None:
        parser = Gemma4Format(FakeTokenizer(GEMMA_TOKENS))

        _, calls = parser.extract_tool_calls(chars("answer"))

        assert calls == []

    def test_missing_call_prefix_reports_malformed(self) -> None:
        parser = Gemma4Format(FakeTokenizer(GEMMA_TOKENS))
        ids = [MSTART, *chars("add{a:1}"), MEND]

        _, calls = parser.extract_tool_calls(ids)

        assert calls[0].status is ToolCallParseStatus.MALFORMED_STRUCTURE

    def test_missing_name_reports_status(self) -> None:
        parser = Gemma4Format(FakeTokenizer(GEMMA_TOKENS))
        ids = [MSTART, *chars("call:{a:1}"), MEND]

        _, calls = parser.extract_tool_calls(ids)

        assert calls[0].status is ToolCallParseStatus.MISSING_NAME

    def test_unbalanced_braces_reports_malformed(self) -> None:
        parser = Gemma4Format(FakeTokenizer(GEMMA_TOKENS))
        ids = [MSTART, *chars("call:add{a:{b:1}"), MEND]

        _, calls = parser.extract_tool_calls(ids)

        assert calls[0].status is ToolCallParseStatus.MALFORMED_STRUCTURE

    def test_unclosed_block_reports_status(self) -> None:
        parser = Gemma4Format(FakeTokenizer(GEMMA_TOKENS))
        ids = [MSTART, *chars("call:add{a:1}")]

        _, calls = parser.extract_tool_calls(ids)

        assert calls[0].status is ToolCallParseStatus.UNCLOSED_BLOCK

    def test_prose_between_calls_is_skipped(self) -> None:
        parser = Gemma4Format(FakeTokenizer(GEMMA_TOKENS))
        ids = gemma_block("a", "{x:1}") + chars(" then ") + gemma_block("b", "{y:2}")

        _, calls = parser.extract_tool_calls(ids)

        assert [call.name for call in calls] == ["a", "b"]
        assert all(call.status is ToolCallParseStatus.OK for call in calls)

    def test_call_without_braces_reports_malformed(self) -> None:
        parser = Gemma4Format(FakeTokenizer(GEMMA_TOKENS))
        ids = [MSTART, *chars("call:foo"), MEND]

        _, calls = parser.extract_tool_calls(ids)

        assert calls[0].status is ToolCallParseStatus.MALFORMED_STRUCTURE

    def test_lone_string_delimiter_reports_malformed(self) -> None:
        parser = Gemma4Format(FakeTokenizer(GEMMA_TOKENS))
        ids = [MSTART, *chars('call:f{<|"|>abc}'), MEND]

        _, calls = parser.extract_tool_calls(ids)

        assert calls[0].status is ToolCallParseStatus.MALFORMED_STRUCTURE

    def test_trailing_separator_is_ignored(self) -> None:
        parser = Gemma4Format(FakeTokenizer(GEMMA_TOKENS))
        ids = gemma_block("f", "{a:1,}")

        _, calls = parser.extract_tool_calls(ids)

        assert calls[0].arguments == {"a": "1"}
        assert calls[0].status is ToolCallParseStatus.OK

    def test_args_ending_mid_key_report_malformed(self) -> None:
        parser = Gemma4Format(FakeTokenizer(GEMMA_TOKENS))
        ids = gemma_block("f", "{abc}")

        _, calls = parser.extract_tool_calls(ids)

        assert calls[0].status is ToolCallParseStatus.MALFORMED_STRUCTURE

    def test_spaced_values_parse(self) -> None:
        parser = Gemma4Format(FakeTokenizer(GEMMA_TOKENS))
        ids = gemma_block("f", "{a: 1}")

        _, calls = parser.extract_tool_calls(ids)

        assert calls[0].arguments == {"a": "1"}
        assert calls[0].status is ToolCallParseStatus.OK

    def test_bare_value_swallowing_delimiter_reports_malformed(self) -> None:
        parser = Gemma4Format(FakeTokenizer(GEMMA_TOKENS))
        ids = gemma_block("f", '{a:x<|"|>y,b:<|"|>z}')

        _, calls = parser.extract_tool_calls(ids)

        assert calls[0].status is ToolCallParseStatus.MALFORMED_STRUCTURE

    def test_swallowed_delimiter_breaks_nested_braces(self) -> None:
        parser = Gemma4Format(FakeTokenizer(GEMMA_TOKENS))
        ids = gemma_block("f", '{a:Q<|"|>W,b:{Y<|"|>}')

        _, calls = parser.extract_tool_calls(ids)

        assert calls[0].status is ToolCallParseStatus.MALFORMED_STRUCTURE

    def test_array_separators_are_skipped(self) -> None:
        parser = Gemma4Format(FakeTokenizer(GEMMA_TOKENS))
        ids = gemma_block("f", "{a:[1, 2]}")

        _, calls = parser.extract_tool_calls(ids)

        assert calls[0].arguments == {"a": ["1", "2"]}
        assert calls[0].status is ToolCallParseStatus.OK

    def test_unterminated_array_reports_malformed(self) -> None:
        parser = Gemma4Format(FakeTokenizer(GEMMA_TOKENS))
        ids = gemma_block("f", "{a:[1}")

        _, calls = parser.extract_tool_calls(ids)

        assert calls[0].status is ToolCallParseStatus.MALFORMED_STRUCTURE

    def test_missing_tag_vocab_raises(self) -> None:
        with pytest.raises(ValueError, match="Gemma"):
            Gemma4Format(FakeTokenizer({}))


class TestGemma4FormatResponses:
    def test_formats_one_string_result(self) -> None:
        parser = Gemma4Format(FakeTokenizer(GEMMA_TOKENS))

        text = parser.format_tool_results(
            [{"role": "tool", "name": "guess", "content": "higher"}]
        )

        assert text == (
            '<|tool_response>response:guess{value:<|"|>higher<|"|>}<tool_response|>'
        )

    def test_concatenates_one_block_per_result(self) -> None:
        parser = Gemma4Format(FakeTokenizer(GEMMA_TOKENS))

        text = parser.format_tool_results(
            [
                {"role": "tool", "name": "a", "content": "1"},
                {"role": "tool", "name": "b", "content": "2"},
            ]
        )

        assert text == (
            '<|tool_response>response:a{value:<|"|>1<|"|>}<tool_response|>'
            '<|tool_response>response:b{value:<|"|>2<|"|>}<tool_response|>'
        )

    def test_missing_name_renders_unknown(self) -> None:
        parser = Gemma4Format(FakeTokenizer(GEMMA_TOKENS))

        text = parser.format_tool_results([{"role": "tool", "content": "oops"}])

        assert text is not None
        assert "response:unknown{" in text

    def test_non_tool_turn_uses_the_template_frame(self) -> None:
        parser = Gemma4Format(FakeTokenizer(GEMMA_TOKENS))

        assert parser.format_tool_results([{"role": "user", "content": "hi"}]) is None
        assert parser.format_tool_results([]) is None


class TestHarmonyFormatResponses:
    def test_formats_each_result_then_the_assistant_header(self) -> None:
        parser = HarmonyFormat(FakeTokenizer(HARMONY_TOKENS))

        text = parser.format_tool_results(
            [
                {"role": "tool", "name": "add", "content": "3"},
                {"role": "tool", "name": "mul", "content": 'say "hi"\n'},
            ]
        )

        assert text == (
            "<|start|>functions.add to=assistant<|channel|>commentary"
            '<|message|>"3"<|end|>'
            "<|start|>functions.mul to=assistant<|channel|>commentary"
            '<|message|>"say \\"hi\\"\\n"<|end|>'
            "<|start|>assistant"
        )

    def test_missing_name_renders_unknown(self) -> None:
        parser = HarmonyFormat(FakeTokenizer(HARMONY_TOKENS))

        text = parser.format_tool_results([{"role": "tool", "content": "oops"}])

        assert text is not None
        assert text.startswith("<|start|>functions.unknown to=assistant")

    def test_non_tool_turn_uses_the_template_frame(self) -> None:
        parser = HarmonyFormat(FakeTokenizer(HARMONY_TOKENS))

        assert parser.format_tool_results([{"role": "user", "content": "hi"}]) is None
        assert parser.format_tool_results([]) is None


class TestChatFormatFormatResponses:
    def test_tag_format_uses_the_template_frame(self) -> None:
        parser = ToolCallTagFormat(FakeTokenizer(TAG_TOKENS))

        assert parser.format_tool_results([{"role": "tool", "content": "3"}]) is None


class TestChatFormatClosesTurn:
    def test_same_token_closes_the_turn(self) -> None:
        parser = ToolCallTagFormat(FakeTokenizer(TAG_TOKENS))

        assert parser.closes_turn(TC_END, TC_END) is True
        assert parser.closes_turn(TC_END, TC) is False

    @pytest.mark.parametrize("sampled", [HRET, HCALL, HEND])
    def test_harmony_message_ends_close_an_end_frame(self, sampled: int) -> None:
        parser = HarmonyFormat(FakeTokenizer(HARMONY_TOKENS))

        assert parser.closes_turn(sampled, HEND) is True

    def test_harmony_return_does_not_close_another_frame_token(self) -> None:
        parser = HarmonyFormat(FakeTokenizer(HARMONY_TOKENS))

        assert parser.closes_turn(HRET, HSTART) is False


class TestHarmonyFormat:
    def test_parses_function_block(self) -> None:
        parser = HarmonyFormat(FakeTokenizer(HARMONY_TOKENS))
        ids = harmony_block("assistant to=functions.add", '{"a": 2}', HCALL)

        _, calls = parser.extract_tool_calls(ids)

        assert len(calls) == 1
        assert calls[0].name == "add"
        assert calls[0].arguments == {"a": 2}
        assert calls[0].status is ToolCallParseStatus.OK

    def test_parses_a_call_in_the_block_the_generation_prompt_opened(self) -> None:
        parser = HarmonyFormat(FakeTokenizer(HARMONY_TOKENS))
        ids = harmony_block(" to=functions.add", '{"a": 2}', HCALL)[1:]

        content, calls = parser.extract_tool_calls(ids)

        assert content == []
        assert [(call.name, call.arguments) for call in calls] == [("add", {"a": 2})]

    def test_answer_in_the_opened_block_is_content(self) -> None:
        parser = HarmonyFormat(FakeTokenizer(HARMONY_TOKENS))
        ids = [HCHAN, *chars("final"), HMSG, *chars("4"), HRET]

        content, calls = parser.extract_tool_calls(ids)

        assert calls == []
        assert content == ids[:-1]

    def test_ignores_non_function_channels(self) -> None:
        parser = HarmonyFormat(FakeTokenizer(HARMONY_TOKENS))
        analysis = harmony_block("assistant to=commentary", "thinking", HEND)
        final = harmony_block("assistant to=user", "answer", HEND)
        call = harmony_block("assistant to=functions.add", '{"a": 1}', HCALL)

        _, calls = parser.extract_tool_calls(analysis + final + call)

        assert [call.name for call in calls] == ["add"]

    def test_channel_tag_in_header_still_parses(self) -> None:
        parser = HarmonyFormat(FakeTokenizer(HARMONY_TOKENS))
        ids = [
            HSTART,
            *chars("assistant to=functions.add "),
            HCHAN,
            *chars("commentary"),
            HMSG,
            *chars('{"a": 1}'),
            HCALL,
        ]

        _, calls = parser.extract_tool_calls(ids)

        assert len(calls) == 1
        assert calls[0].name == "add"

    def test_invalid_json_reports_status_with_name(self) -> None:
        parser = HarmonyFormat(FakeTokenizer(HARMONY_TOKENS))
        ids = harmony_block("assistant to=functions.add", "{bad}", HCALL)

        _, calls = parser.extract_tool_calls(ids)

        assert calls[0].status is ToolCallParseStatus.INVALID_JSON
        assert calls[0].name == "add"

    def test_unclosed_block_reports_status(self) -> None:
        parser = HarmonyFormat(FakeTokenizer(HARMONY_TOKENS))
        ids = [HSTART, *chars("assistant to=functions.add"), HMSG, *chars('{"a": 1}')]

        _, calls = parser.extract_tool_calls(ids)

        assert calls[0].status is ToolCallParseStatus.UNCLOSED_BLOCK

    def test_return_truncates_later_blocks(self) -> None:
        parser = HarmonyFormat(FakeTokenizer(HARMONY_TOKENS))
        first = harmony_block("assistant to=functions.a", "{}", HCALL)
        second = harmony_block("assistant to=functions.b", "{}", HCALL)

        _, calls = parser.extract_tool_calls([*first, HRET, *second])

        assert [call.name for call in calls] == ["a"]

    def test_leading_content_returns_as_content(self) -> None:
        parser = HarmonyFormat(FakeTokenizer(HARMONY_TOKENS))
        ids = chars("note ") + harmony_block(
            "assistant to=functions.add", '{"a": 1}', HCALL
        )

        content, calls = parser.extract_tool_calls(ids)

        assert content == chars("note ")
        assert [call.name for call in calls] == ["add"]

    def test_start_without_message_yields_no_calls(self) -> None:
        parser = HarmonyFormat(FakeTokenizer(HARMONY_TOKENS))
        ids = [HSTART, *chars("assistant to=user"), *chars("hi")]

        content, calls = parser.extract_tool_calls(ids)

        assert (content, calls) == (ids, [])

    def test_missing_return_token_keeps_full_sequence(self) -> None:
        tokens = {k: v for k, v in HARMONY_TOKENS.items() if k != "<|return|>"}
        parser = HarmonyFormat(FakeTokenizer(tokens))
        ids = harmony_block("assistant to=functions.add", '{"a": 1}', HCALL)

        _, calls = parser.extract_tool_calls(ids)

        assert [call.name for call in calls] == ["add"]
        assert calls[0].status is ToolCallParseStatus.OK

    def test_empty_function_name_reports_status(self) -> None:
        parser = HarmonyFormat(FakeTokenizer(HARMONY_TOKENS))
        ids = harmony_block("assistant to=functions.", '{"a": 1}', HCALL)

        _, calls = parser.extract_tool_calls(ids)

        assert calls[0].status is ToolCallParseStatus.MISSING_NAME

    def test_non_object_body_reports_malformed(self) -> None:
        parser = HarmonyFormat(FakeTokenizer(HARMONY_TOKENS))
        ids = harmony_block("assistant to=functions.add", "[1, 2]", HCALL)

        _, calls = parser.extract_tool_calls(ids)

        assert calls[0].status is ToolCallParseStatus.MALFORMED_STRUCTURE

    def test_missing_tag_vocab_raises(self) -> None:
        with pytest.raises(ValueError, match="Harmony"):
            HarmonyFormat(FakeTokenizer({}))


class TestGlm47Format:
    def test_parses_args_typed_by_the_schema(self) -> None:
        parser = Glm47Format(FakeTokenizer(GLM_TOKENS))
        ids = glm_ids(
            "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value>"
            "<arg_key>days</arg_key><arg_value>3</arg_value></tool_call>"
        )

        _, calls = parser.extract_tool_calls(ids, WEATHER_TOOLS)

        assert len(calls) == 1
        assert calls[0].name == "get_weather"
        assert calls[0].arguments == {"city": "Paris", "days": 3}
        assert calls[0].token_span == (0, len(ids))
        assert calls[0].status is ToolCallParseStatus.OK

    def test_returns_content_before_first_call(self) -> None:
        parser = Glm47Format(FakeTokenizer(GLM_TOKENS))
        ids = glm_ids("plan</think>hi<tool_call>ping</tool_call>")

        content, calls = parser.extract_tool_calls(ids)

        assert content == glm_ids("plan</think>hi")
        assert [(call.name, call.arguments) for call in calls] == [("ping", {})]

    def test_parses_several_calls_with_json_values(self) -> None:
        parser = Glm47Format(FakeTokenizer(GLM_TOKENS))
        ids = glm_ids(
            '<tool_call>a<arg_key>o</arg_key><arg_value>{"k": [1]}</arg_value></tool_call>'
            "<tool_call>b<arg_key>n</arg_key> <arg_value>null</arg_value></tool_call>"
        )

        _, calls = parser.extract_tool_calls(ids)

        assert [(call.name, call.arguments) for call in calls] == [
            ("a", {"o": {"k": [1]}}),
            ("b", {"n": None}),
        ]
        assert all(call.status is ToolCallParseStatus.OK for call in calls)

    def test_string_value_keeps_its_whitespace(self) -> None:
        parser = Glm47Format(FakeTokenizer(GLM_TOKENS))
        ids = glm_ids(
            "<tool_call>get_weather<arg_key>city</arg_key>"
            "<arg_value> New York\n</arg_value></tool_call>"
        )

        _, calls = parser.extract_tool_calls(ids, WEATHER_TOOLS)

        assert calls[0].arguments == {"city": " New York\n"}

    def test_tag_text_in_a_value_does_not_end_it(self) -> None:
        parser = Glm47Format(FakeTokenizer(GLM_TOKENS))
        ids = [
            *glm_ids("<tool_call>get_weather<arg_key>city</arg_key><arg_value>"),
            *chars("a</arg_value>b"),
            *glm_ids("</arg_value></tool_call>"),
        ]

        _, calls = parser.extract_tool_calls(ids, WEATHER_TOOLS)

        assert calls[0].arguments == {"city": "a</arg_value>b"}
        assert calls[0].status is ToolCallParseStatus.OK

    def test_think_drafts_are_skipped(self) -> None:
        parser = Glm47Format(FakeTokenizer(GLM_TOKENS))
        ids = glm_ids("<tool_call>draft</tool_call></think><tool_call>real</tool_call>")

        _, calls = parser.extract_tool_calls(ids)

        assert [call.name for call in calls] == ["real"]

    def test_schema_violation_reports_invalid(self) -> None:
        parser = Glm47Format(FakeTokenizer(GLM_TOKENS))
        ids = glm_ids(
            "<tool_call>get_weather<arg_key>days</arg_key>"
            "<arg_value>soon</arg_value></tool_call>"
        )

        _, calls = parser.extract_tool_calls(ids, WEATHER_TOOLS)

        assert calls[0].arguments == {"days": "soon"}
        assert calls[0].status is ToolCallParseStatus.INVALID_JSON

    def test_missing_name_reports_status(self) -> None:
        parser = Glm47Format(FakeTokenizer(GLM_TOKENS))
        ids = glm_ids(
            "<tool_call><arg_key>a</arg_key><arg_value>1</arg_value></tool_call>"
        )

        _, calls = parser.extract_tool_calls(ids)

        assert calls[0].status is ToolCallParseStatus.MISSING_NAME

    @pytest.mark.parametrize(
        "body",
        [
            "f<arg_key>a</arg_key></tool_call>",
            "f<arg_key>a</arg_key>x<arg_value>1</arg_value></tool_call>",
            "f<arg_key>a<arg_value>1</arg_value></tool_call>",
        ],
    )
    def test_broken_pair_reports_malformed(self, body: str) -> None:
        parser = Glm47Format(FakeTokenizer(GLM_TOKENS))

        _, calls = parser.extract_tool_calls(glm_ids(f"<tool_call>{body}"))

        assert calls[0].name == "f"
        assert calls[0].status is ToolCallParseStatus.MALFORMED_STRUCTURE

    def test_unclosed_block_reports_status(self) -> None:
        parser = Glm47Format(FakeTokenizer(GLM_TOKENS))
        ids = glm_ids("<tool_call>f<arg_key>a</arg_key><arg_value>1</arg_value>")

        _, calls = parser.extract_tool_calls(ids)

        assert calls[0].status is ToolCallParseStatus.UNCLOSED_BLOCK

    def test_missing_tag_vocab_raises(self) -> None:
        with pytest.raises(ValueError, match="<arg_key>"):
            Glm47Format(FakeTokenizer(TAG_TOKENS))


class TestDsmlFormat:
    def test_parses_typed_parameters(self) -> None:
        parser = DsmlFormat(FakeTokenizer(DSML_TOKENS))
        block = dsml_ids(
            dsml_block(
                dsml_invoke(
                    "get_weather",
                    [
                        ("city", "true", "Paris"),
                        ("days", "false", "3"),
                        ("opts", "false", '{"u": ["c", 1]}'),
                    ],
                )
            )
        )

        _, calls = parser.extract_tool_calls([*block, DEOS])

        assert len(calls) == 1
        assert calls[0].name == "get_weather"
        assert calls[0].arguments == {
            "city": "Paris",
            "days": 3,
            "opts": {"u": ["c", 1]},
        }
        assert calls[0].token_span == (0, len(block))
        assert calls[0].status is ToolCallParseStatus.OK

    def test_returns_content_before_the_block(self) -> None:
        parser = DsmlFormat(FakeTokenizer(DSML_TOKENS))
        ids = dsml_ids(
            "plan</think>Checking.\n\n" + dsml_block(dsml_invoke("ping", []))
        )

        content, calls = parser.extract_tool_calls(ids)

        assert content == dsml_ids("plan</think>Checking.\n\n")
        assert [(call.name, call.arguments) for call in calls] == [("ping", {})]

    def test_invokes_in_one_block_share_its_span(self) -> None:
        parser = DsmlFormat(FakeTokenizer(DSML_TOKENS))
        ids = dsml_ids(
            dsml_block(
                dsml_invoke("a", [("x", "false", "1")]),
                dsml_invoke("b", [("y", "true", "two")]),
            )
        )

        _, calls = parser.extract_tool_calls(ids)

        assert [(call.name, call.arguments) for call in calls] == [
            ("a", {"x": 1}),
            ("b", {"y": "two"}),
        ]
        assert {call.token_span for call in calls} == {(0, len(ids))}

    def test_parameter_without_string_flag_follows_the_schema(self) -> None:
        parser = DsmlFormat(FakeTokenizer(DSML_TOKENS))
        invoke = (
            f'<{DSML}invoke name="get_weather">\n'
            f'<{DSML}parameter name="city">true</{DSML}parameter>\n'
            f'<{DSML}parameter name="days">2</{DSML}parameter>\n'
            f"</{DSML}invoke>"
        )

        _, calls = parser.extract_tool_calls(
            dsml_ids(dsml_block(invoke)), WEATHER_TOOLS
        )

        assert calls[0].arguments == {"city": "true", "days": 2}

    def test_unclosed_parameter_ends_at_the_next_one(self) -> None:
        parser = DsmlFormat(FakeTokenizer(DSML_TOKENS))
        invoke = (
            f'<{DSML}invoke name="f">\n'
            f'<{DSML}parameter name="a" string="true">x\n'
            f'<{DSML}parameter name="b" string="false">1</{DSML}parameter>\n'
            f"</{DSML}invoke>"
        )

        _, calls = parser.extract_tool_calls(dsml_ids(dsml_block(invoke)))

        assert calls[0].arguments == {"a": "x\n", "b": 1}

    def test_think_drafts_are_skipped(self) -> None:
        parser = DsmlFormat(FakeTokenizer(DSML_TOKENS))
        ids = dsml_ids(
            dsml_block(dsml_invoke("draft", []))
            + "</think>"
            + dsml_block(dsml_invoke("real", []))
        )

        _, calls = parser.extract_tool_calls(ids)

        assert [call.name for call in calls] == ["real"]

    def test_tag_text_without_the_dsml_token_does_not_parse(self) -> None:
        parser = DsmlFormat(FakeTokenizer(DSML_TOKENS))
        ids = chars(dsml_block(dsml_invoke("f", [])))

        content, calls = parser.extract_tool_calls(ids)

        assert (content, calls) == (ids, [])

    def test_invoke_outside_a_block_does_not_parse(self) -> None:
        parser = DsmlFormat(FakeTokenizer(DSML_TOKENS))
        ids = dsml_ids(dsml_invoke("f", []))

        _, calls = parser.extract_tool_calls(ids)

        assert calls == []

    def test_unclosed_block_reports_status(self) -> None:
        parser = DsmlFormat(FakeTokenizer(DSML_TOKENS))
        ids = dsml_ids(f"<{DSML}tool_calls>\n" + dsml_invoke("f", []))

        _, calls = parser.extract_tool_calls(ids)

        assert len(calls) == 1
        assert calls[0].status is ToolCallParseStatus.UNCLOSED_BLOCK

    def test_block_without_invoke_reports_malformed(self) -> None:
        parser = DsmlFormat(FakeTokenizer(DSML_TOKENS))

        _, calls = parser.extract_tool_calls(dsml_ids(dsml_block("call f")))

        assert calls[0].status is ToolCallParseStatus.MALFORMED_STRUCTURE

    def test_empty_name_reports_status(self) -> None:
        parser = DsmlFormat(FakeTokenizer(DSML_TOKENS))

        _, calls = parser.extract_tool_calls(dsml_ids(dsml_block(dsml_invoke("", []))))

        assert calls[0].status is ToolCallParseStatus.MISSING_NAME

    def test_non_json_value_reports_invalid(self) -> None:
        parser = DsmlFormat(FakeTokenizer(DSML_TOKENS))
        ids = dsml_ids(dsml_block(dsml_invoke("f", [("a", "false", "abc")])))

        _, calls = parser.extract_tool_calls(ids)

        assert calls[0].arguments == {"a": "abc"}
        assert calls[0].status is ToolCallParseStatus.INVALID_JSON

    def test_missing_tag_vocab_raises(self) -> None:
        with pytest.raises(ValueError, match="DSML"):
            DsmlFormat(FakeTokenizer(TAG_TOKENS))

    def test_tag_end_without_a_closing_bracket_is_the_output_end(self) -> None:
        parser = DsmlFormat(FakeTokenizer(DSML_TOKENS))
        ids = dsml_ids(f"</{DSML}tool_calls")

        end = parser._tag_end(ids, 0)

        assert end == len(ids)


class TestDsmlFormatChatTemplate:
    def test_renders_a_tool_call_and_its_result(self) -> None:
        # Arrange
        messages = [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Weather?"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "need tool",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": {"city": "Paris", "days": 2},
                        },
                    }
                ],
            },
            {"role": "tool", "content": "sunny"},
        ]

        # Act
        rendered = render_dsml(messages, WEATHER_TOOLS, reasoning_effort="low")

        # Assert
        schema = json.dumps(WEATHER_TOOLS[0]["function"], ensure_ascii=False)
        assert rendered == (
            f"{DSML_BOS}Be brief.\n\n{DSML_TOOLS_PROMPT_HEAD}{schema}{DSML_TOOLS_PROMPT_TAIL}"
            f"{USER}Weather?{ASSISTANT}<think>need tool</think>\n\n"
            + dsml_block(
                dsml_invoke(
                    "get_weather", [("city", "true", "Paris"), ("days", "false", "2")]
                )
            )
            + f"{DSML_EOS}{USER}<tool_result>sunny</tool_result>{ASSISTANT}<think>"
        )

    def test_rendered_call_parses_back(self) -> None:
        tokenizer = FakeTokenizer(DSML_TOKENS)
        arguments = {"city": "Paris", "days": 2}
        call = {
            "type": "function",
            "function": {"name": "get_weather", "arguments": arguments},
        }
        rendered = render_dsml(
            [
                {"role": "user", "content": "Weather?"},
                {"role": "assistant", "content": "", "tool_calls": [call]},
            ],
            WEATHER_TOOLS,
        )
        turn = rendered[
            rendered.rindex(f"{ASSISTANT}<think>") + len(f"{ASSISTANT}<think>") :
        ]

        _, calls = DsmlFormat(tokenizer).extract_tool_calls(
            tokenizer.encode(turn), WEATHER_TOOLS
        )

        assert [(c.name, c.arguments, c.status) for c in calls] == [
            ("get_weather", arguments, ToolCallParseStatus.OK)
        ]

    def test_without_tools_drops_earlier_reasoning(self) -> None:
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello", "reasoning_content": "greet"},
            {"role": "user", "content": "capital?"},
        ]

        rendered = render_dsml(messages)

        assert rendered == (
            f"{DSML_BOS}{DSML_REASONING_EFFORT_PROMPTS['high']}"
            f"{USER}hi{ASSISTANT}</think>hello{DSML_EOS}"
            f"{USER}capital?{ASSISTANT}<think>"
        )

    def test_tools_without_a_system_message_lead_the_prompt(self) -> None:
        rendered = render_dsml(
            [{"role": "user", "content": "hi"}], WEATHER_TOOLS, reasoning_effort="max"
        )

        assert rendered.startswith(
            f"{DSML_BOS}{DSML_REASONING_EFFORT_PROMPTS['max']}\n\n{DSML_TOOLS_PROMPT_HEAD}"
        )
        assert rendered.endswith(f"{DSML_TOOLS_PROMPT_TAIL}{USER}hi{ASSISTANT}<think>")

    def test_consecutive_tool_results_share_one_user_turn(self) -> None:
        messages = [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "ok"},
            {"role": "tool", "content": "a"},
            {"role": "tool", "content": "b"},
        ]

        rendered = render_dsml(messages, thinking=False)

        assert rendered == (
            f"{DSML_BOS}{USER}go{ASSISTANT}</think>ok{DSML_EOS}"
            f"{USER}<tool_result>a</tool_result>\n\n<tool_result>b</tool_result>"
            f"{ASSISTANT}</think>"
        )

    def test_unsupported_role_raises(self) -> None:
        with pytest.raises(TemplateError, match="not developer"):
            render_dsml([{"role": "developer", "content": "x"}])


class TestDecodeGuard:
    def test_non_string_decode_raises(self) -> None:
        class BytesTokenizer(FakeTokenizer):
            def decode(self, ids: list[int], skip_special_tokens: bool = True) -> Any:
                del ids, skip_special_tokens
                return b"bytes"

        parser = ToolCallTagFormat(BytesTokenizer(TAG_TOKENS))

        with pytest.raises(TypeError, match="returns str"):
            parser.extract_tool_calls(json_block("add", {}))


class TestDetectChatFormat:
    def test_detects_each_family(self) -> None:
        assert isinstance(
            detect_chat_format(FakeTokenizer(TAG_TOKENS)), ToolCallTagFormat
        )
        assert isinstance(
            detect_chat_format(FakeTokenizer(GRANITE_TOKENS)), Granite3Format
        )
        assert isinstance(detect_chat_format(FakeTokenizer(GEMMA_TOKENS)), Gemma4Format)
        assert isinstance(
            detect_chat_format(FakeTokenizer(HARMONY_TOKENS)), HarmonyFormat
        )
        assert isinstance(detect_chat_format(FakeTokenizer(DSML_TOKENS)), DsmlFormat)

    def test_glm_vocab_picks_glm_over_the_tag_format(self) -> None:
        assert isinstance(detect_chat_format(FakeTokenizer(GLM_TOKENS)), Glm47Format)

    def test_returns_none_without_tags(self) -> None:
        assert detect_chat_format(FakeTokenizer({})) is None

    def test_first_registry_match_wins(self) -> None:
        merged = {**TAG_TOKENS, **GRANITE_TOKENS}

        assert isinstance(detect_chat_format(FakeTokenizer(merged)), ToolCallTagFormat)


class TestGetChatFormat:
    def test_builds_each_registered_format(self) -> None:
        assert isinstance(
            get_chat_format("tool_call", FakeTokenizer(TAG_TOKENS)),
            ToolCallTagFormat,
        )
        assert isinstance(
            get_chat_format("granite3", FakeTokenizer(GRANITE_TOKENS)),
            Granite3Format,
        )
        assert isinstance(
            get_chat_format("gemma4", FakeTokenizer(GEMMA_TOKENS)), Gemma4Format
        )
        assert isinstance(
            get_chat_format("harmony", FakeTokenizer(HARMONY_TOKENS)),
            HarmonyFormat,
        )
        assert isinstance(
            get_chat_format("glm47", FakeTokenizer(GLM_TOKENS)), Glm47Format
        )
        assert isinstance(
            get_chat_format("dsml", FakeTokenizer(DSML_TOKENS)), DsmlFormat
        )

    def test_unknown_name_lists_available(self) -> None:
        with pytest.raises(ValueError, match="tool_call"):
            get_chat_format("llama", FakeTokenizer(TAG_TOKENS))


class TestResolveChatFormat:
    def test_builds_the_named_format(self) -> None:
        merged = {**TAG_TOKENS, **GRANITE_TOKENS}

        parser = resolve_chat_format(FakeTokenizer(merged), "granite3")

        assert isinstance(parser, Granite3Format)

    def test_detects_when_no_name_is_set(self) -> None:
        parser = resolve_chat_format(FakeTokenizer(GEMMA_TOKENS), None)

        assert isinstance(parser, Gemma4Format)

    def test_returns_none_when_nothing_matches(self) -> None:
        assert resolve_chat_format(FakeTokenizer({}), None) is None


def harmony_turn(analysis: str, final: str) -> list[int]:
    """One GPT-OSS reply: an analysis message, then the final answer."""
    return [
        HCHAN,
        *chars("analysis"),
        HMSG,
        *chars(analysis),
        HEND,
        HSTART,
        *chars("assistant"),
        HCHAN,
        *chars("final"),
        HMSG,
        *chars(final),
        HRET,
    ]


class TestChatFormatAnswerStart:
    def test_tag_answer_starts_after_the_think_close(self) -> None:
        parser = ToolCallTagFormat(FakeTokenizer(TAG_TOKENS))
        ids = [*chars("plan"), THINK_END, *chars("go")]

        assert parser.answer_start(ids) == 5

    def test_gemma_answer_starts_after_the_thought_channel(self) -> None:
        parser = Gemma4Format(FakeTokenizer(GEMMA_TOKENS))
        ids = [*chars("thought\nplan"), MCHAN_END, *chars("go")]

        assert parser.answer_start(ids) == 13

    def test_harmony_answer_starts_after_the_final_channel_header(self) -> None:
        parser = HarmonyFormat(FakeTokenizer(HARMONY_TOKENS))
        ids = harmony_turn("plan", "go")

        assert ids[parser.answer_start(ids) :] == [*chars("go"), HRET]

    def test_is_zero_without_a_close(self) -> None:
        parser = ToolCallTagFormat(FakeTokenizer(TAG_TOKENS))

        assert parser.answer_start(chars("still thinking")) == 0

    def test_is_zero_for_a_format_with_no_reasoning_token(self) -> None:
        parser = Granite3Format(FakeTokenizer(GRANITE_TOKENS))

        assert parser.reasoning_end_ids == ()
        assert parser.answer_start(chars("a</think>b")) == 0

    def test_granite_closes_at_a_think_token_in_the_vocab(self) -> None:
        parser = Granite3Format(
            FakeTokenizer({**GRANITE_TOKENS, "</think>": THINK_END})
        )

        assert parser.reasoning_end_ids == (THINK_END,)
        assert parser.answer_start([*chars("a"), THINK_END, *chars("b")]) == 2

    def test_harmony_close_is_the_whole_final_channel_header(self) -> None:
        parser = HarmonyFormat(FakeTokenizer(HARMONY_TOKENS))

        assert parser.reasoning_end_ids == (HCHAN, *chars("final"), HMSG)

    def test_harmony_without_a_channel_token_has_no_close(self) -> None:
        tokens = {k: v for k, v in HARMONY_TOKENS.items() if k != "<|channel|>"}

        parser = HarmonyFormat(FakeTokenizer(tokens))

        assert parser.reasoning_end_ids == ()


class TestActionText:
    def test_drops_the_think_block(self) -> None:
        tokenizer = FakeTokenizer(TAG_TOKENS)
        ids = [*chars("plan"), THINK_END, *chars("\nclick('12') ")]

        action = action_text(tokenizer.decode, ids, ToolCallTagFormat(tokenizer))

        assert action == "click('12')"

    def test_gemma_drops_the_thought_channel(self) -> None:
        tokenizer = FakeTokenizer(GEMMA_TOKENS)
        ids = [*chars("thought\nIt is 4."), MCHAN_END, *chars("4")]

        action = action_text(tokenizer.decode, ids, Gemma4Format(tokenizer))

        assert action == "4"

    def test_harmony_sends_only_the_final_channel(self) -> None:
        tokenizer = FakeTokenizer(HARMONY_TOKENS)
        ids = harmony_turn("It is 4.", "4")

        action = action_text(tokenizer.decode, ids, HarmonyFormat(tokenizer))

        assert action == "4"

    def test_harmony_final_channel_without_analysis_sends_the_answer(self) -> None:
        tokenizer = FakeTokenizer(HARMONY_TOKENS)
        ids = [HCHAN, *chars("final"), HMSG, *chars("go"), HRET]

        action = action_text(tokenizer.decode, ids, HarmonyFormat(tokenizer))

        assert action == "go"

    def test_harmony_reasoning_naming_final_without_a_final_channel_passes_through(
        self,
    ) -> None:
        tokenizer = FakeTokenizer(HARMONY_TOKENS)
        ids = [HCHAN, *chars("analysis"), HMSG, *chars("The final answer."), HEND]

        action = action_text(tokenizer.decode, ids, HarmonyFormat(tokenizer))

        assert action == "analysisThe final answer."

    def test_plain_text_think_close_splits_without_a_close_token(self) -> None:
        tokenizer = FakeTokenizer(GRANITE_TOKENS)
        ids = chars("plan</think> go")

        action = action_text(tokenizer.decode, ids, Granite3Format(tokenizer))

        assert action == "go"

    def test_granite_think_token_dropped_by_decode_still_splits(self) -> None:
        tokenizer = FakeTokenizer({**GRANITE_TOKENS, "</think>": THINK_END})
        ids = [*chars("plan"), THINK_END, *chars(" go")]

        action = action_text(tokenizer.decode, ids, Granite3Format(tokenizer))

        assert action == "go"

    def test_splits_the_decoded_text_without_a_format(self) -> None:
        tokenizer = FakeTokenizer({})

        action = action_text(tokenizer.decode, chars("plan</think> go"), None)

        assert action == "go"

    def test_unclosed_reasoning_passes_through_whole(self) -> None:
        tokenizer = FakeTokenizer(TAG_TOKENS)
        ids = chars(" still thinking")

        action = action_text(tokenizer.decode, ids, ToolCallTagFormat(tokenizer))

        assert action == " still thinking"

    def test_non_string_decode_raises(self) -> None:
        with pytest.raises(TypeError, match="returns str"):
            action_text(lambda _ids, **_kwargs: b"bytes", chars("go"), None)

# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Chat formats: how each model family spells tool calls, reasoning and turn ends in tokens.

A format finds its family's special-token delimiters in generation IDs and
decodes only the segments between them, so a tag typed as prose (ordinary IDs,
never the special-token ID) cannot false-positive. It says where reasoning
ends, how tool results render back to the model, and which sampled token
already closes a turn. Detection picks the first registered format whose
required tokens are all in the tokenizer vocab.

- :class:`Glm47Format`: GLM-4.7, GLM-5
- :class:`ToolCallTagFormat`: Qwen, Hermes, Granite 4, Nemotron
- :class:`Granite3Format`: Granite 3
- :class:`Gemma4Format`: Gemma 4
- :class:`HarmonyFormat`: GPT-OSS
- :class:`DsmlFormat`: DeepSeek-V4
"""

from __future__ import annotations

import enum
import json
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Collection
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from agilerl.llm_envs.reasoning import (
    REASONING_END_MARKER,
    SUBSEQUENCE_NOT_FOUND,
    find_token_subsequence,
)
from agilerl.utils.algo_utils import is_str_keyed_dict

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

__all__ = [
    "CHAT_FORMATS",
    "ChatFormat",
    "DsmlFormat",
    "Gemma4Format",
    "Glm47Format",
    "Granite3Format",
    "HarmonyFormat",
    "ParsedToolCall",
    "ToolCallParseStatus",
    "ToolCallTagFormat",
    "action_span",
    "action_text",
    "detect_chat_format",
    "get_chat_format",
    "resolve_chat_format",
]

GEMMA_STRING_DELIM = '<|"|>'
XML_FUNCTION_RE = re.compile(r"<function=([^>]+)>")
XML_PARAMETER_RE = re.compile(r"<parameter=([^>]+)>\n?(.*?)\n?</parameter>", re.DOTALL)
HARMONY_RECIPIENT_RE = re.compile(r"to=([^\s<]+)")
HARMONY_FUNCTION_PREFIX = "functions."
DSML = "\uff5cDSML\uff5c"
# Block names vLLM's DeepSeek-V4 parser accepts; ``tool_calls`` is the one the encoder writes.
DSML_BLOCK_RE = re.compile(r"(?:tool_calls|toolcalls|tool)>")
DSML_INVOKE_RE = re.compile(
    rf'<{DSML}invoke\s+name="([^"]*)"\s*>(.*?)</{DSML}invoke>', re.DOTALL
)
DSML_PARAMETER_RE = re.compile(
    rf'<{DSML}parameter\s+name="([^"]+)"(?:\s+string="([^"]*)")?\s*>'
    rf"(.*?)(?:</{DSML}parameter>|(?=<{DSML}parameter\s+name=))",
    re.DOTALL,
)
DSML_TOOLS_PROMPT_HEAD = f"""## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<{DSML}tool_calls>" block like the following:

<{DSML}tool_calls>
<{DSML}invoke name="$TOOL_NAME">
<{DSML}parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</{DSML}parameter>
...
</{DSML}invoke>
<{DSML}invoke name="$TOOL_NAME2">
...
</{DSML}invoke>
</{DSML}tool_calls>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.

Otherwise, output directly after </think> with tool calls or final response.

### Available Tool Schemas

"""
DSML_TOOLS_PROMPT_TAIL = """

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
"""
DSML_REASONING_EFFORT_PROMPTS = {
    "high": (
        "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n"
        "You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\n"
        "Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n"
    ),
    "max": (
        "Reasoning Effort: Beyond maximum — exhaustive, relentless, and uncompromising.\n"
        "You MUST reason with the utmost depth and rigor, leaving absolutely nothing to chance: exhaustively decompose the problem into its most fundamental components, trace every causal chain to its root, and resolve the underlying cause rather than any surface symptom.\n"
        "Do not stop reasoning until you have independently verified the solution from multiple angles and are certain that no assumption remains unchecked and no error remains undiscovered.\n\n"
    ),
}
# The DeepSeek-V4 encoder as a Jinja chat template. Consecutive user and tool
# messages merge into one user turn; earlier turns keep their reasoning only
# when tools are set. Thinking defaults on and effort to ``high``, as vLLM serves it.
DSML_CHAT_TEMPLATE = (
    r"""{%- set dsml = '\uff5cDSML\uff5c' -%}
{%- if thinking is defined or enable_thinking is defined -%}
{%- set thinking_on = (thinking is defined and thinking) or (enable_thinking is defined and enable_thinking) -%}
{%- else -%}
{%- set thinking_on = true -%}
{%- endif -%}
{%- set effort = reasoning_effort if reasoning_effort is defined and reasoning_effort is string else none -%}
{%- if effort == 'none' -%}
{%- set thinking_on = false -%}
{%- elif effort in ['low', 'minimal', 'medium'] -%}
{%- set effort = 'low' -%}
{%- elif effort != 'max' and (effort is not none or thinking_on) -%}
{%- set effort = 'high' -%}
{%- endif -%}
{%- set effort_prompts = """
    + json.dumps(DSML_REASONING_EFFORT_PROMPTS)
    + r""" -%}
{%- set has_tools = tools is defined and tools -%}
{%- set ns = namespace(system_idx=-1, last_user=-1) -%}
{%- for m in messages -%}
{%- if m.role == 'system' and ns.system_idx == -1 -%}{%- set ns.system_idx = loop.index0 -%}{%- endif -%}
{%- if m.role in ['user', 'tool'] -%}{%- set ns.last_user = loop.index0 -%}{%- endif -%}
{%- endfor -%}
{%- macro render_tools() -%}
{{- '\n\n' + """
    + json.dumps(DSML_TOOLS_PROMPT_HEAD)
    + r""" -}}
{%- for tool in tools -%}
{%- if not loop.first -%}{{- '\n' -}}{%- endif -%}
{{- (tool['function'] if 'function' in tool else tool) | tojson -}}
{%- endfor -%}
{{- """
    + json.dumps(DSML_TOOLS_PROMPT_TAIL)
    + r""" -}}
{%- endmacro -%}
{{- '<\uff5cbegin▁of▁sentence\uff5c>' -}}
{%- if thinking_on and effort in effort_prompts -%}{{- effort_prompts[effort] -}}{%- endif -%}
{%- if has_tools and ns.system_idx == -1 -%}{{- render_tools() -}}{%- endif -%}
{%- for m in messages -%}
{%- if m.role == 'system' -%}
{{- m.content or '' -}}
{%- if has_tools and loop.index0 == ns.system_idx -%}{{- render_tools() -}}{%- endif -%}
{%- elif m.role in ['user', 'tool'] -%}
{%- if not loop.first and loop.previtem.role in ['user', 'tool'] -%}{{- '\n\n' -}}{%- else -%}{{- '<\uff5cUser\uff5c>' -}}{%- endif -%}
{%- if m.role == 'tool' -%}{{- '<tool_result>' + (m.content or '') + '</tool_result>' -}}{%- else -%}{{- m.content or '' -}}{%- endif -%}
{%- if (loop.last and add_generation_prompt) or (not loop.last and loop.nextitem.role == 'assistant') -%}
{{- '<\uff5cAssistant\uff5c>' -}}
{%- if thinking_on and (has_tools or loop.index0 >= ns.last_user) -%}{{- '<think>' -}}{%- else -%}{{- '</think>' -}}{%- endif -%}
{%- endif -%}
{%- elif m.role == 'assistant' -%}
{%- if thinking_on and (has_tools or loop.index0 > ns.last_user) -%}{{- (m.reasoning_content or '') + '</think>' -}}{%- endif -%}
{{- m.content or '' -}}
{%- if m.tool_calls -%}
{{- '\n\n<' + dsml + 'tool_calls>\n' -}}
{%- for tool_call in m.tool_calls -%}
{%- set call = tool_call['function'] if 'function' in tool_call else tool_call -%}
{%- if not loop.first -%}{{- '\n' -}}{%- endif -%}
{{- '<' + dsml + 'invoke name="' + call.name + '">\n' -}}
{%- for key, value in call.arguments.items() -%}
{%- if not loop.first -%}{{- '\n' -}}{%- endif -%}
{%- if value is string -%}
{{- '<' + dsml + 'parameter name="' + key + '" string="true">' + value + '</' + dsml + 'parameter>' -}}
{%- else -%}
{{- '<' + dsml + 'parameter name="' + key + '" string="false">' + (value | tojson) + '</' + dsml + 'parameter>' -}}
{%- endif -%}
{%- endfor -%}
{{- '\n</' + dsml + 'invoke>' -}}
{%- endfor -%}
{{- '\n</' + dsml + 'tool_calls>' -}}
{%- endif -%}
{{- '<\uff5cend▁of▁sentence\uff5c>' -}}
{%- else -%}
{{- raise_exception('DeepSeek-V4 prompts take system, user, tool and assistant messages, not ' + m.role) -}}
{%- endif -%}
{%- endfor -%}"""
)


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


class ChatFormat(ABC):
    """One model family's tool-call, reasoning and turn-boundary tokens.

    Subclasses declare the special tokens they need; construction fails when
    the tokenizer lacks a required one.
    """

    REQUIRED_TOKENS: ClassVar[tuple[str, ...]]
    OPTIONAL_TOKENS: ClassVar[tuple[str, ...]] = ()
    REASONING_END: ClassVar[str | None] = None
    # Jinja template for a family whose tokenizer ships none.
    CHAT_TEMPLATE: ClassVar[str | None] = None

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        """Resolve this format's delimiter IDs in ``tokenizer``."""
        self._tokenizer = tokenizer
        self._ids = self.token_ids(tokenizer)
        missing = [token for token in self.REQUIRED_TOKENS if token not in self._ids]
        if missing:
            msg = f"{type(self).__name__} needs {', '.join(missing)} in the tokenizer vocab."
            raise ValueError(msg)
        end_id = self._ids.get(self.REASONING_END) if self.REASONING_END else None
        self.reasoning_end_ids: tuple[int, ...] = () if end_id is None else (end_id,)

    def answer_start(self, token_ids: list[int]) -> int:
        """Index just past the first reasoning close, or ``0`` when there is none.

        :param token_ids: One generation's token IDs.
        :return: Where the answer starts.
        """
        at = find_token_subsequence(token_ids, self.reasoning_end_ids)
        if at == SUBSEQUENCE_NOT_FOUND:
            return 0
        return at + len(self.reasoning_end_ids)

    def format_tool_results(self, _messages: list[dict[str, str]], /) -> str | None:
        """Tool results in this format's own blocks, or ``None`` to frame them with the chat template.

        :param _messages: The turn's feedback messages.
        :return: Text to append after the sampled turn, or ``None``.
        """
        return None

    def format_tool_calls(self, _calls: list[dict[str, Any]], /) -> str | None:
        """Assistant text for structured tool calls, or ``None`` when the chat template renders them.

        :param _calls: The turn's ``{"name", "arguments"}`` calls.
        :return: The turn's content, or ``None``.
        """
        return None

    def closes_turn(self, sampled_id: int, frame_id: int) -> bool:
        """Whether a turn that ended in ``sampled_id`` already holds the frame's leading ``frame_id``.

        :param sampled_id: Last token the model sampled.
        :param frame_id: First token of the feedback frame.
        :return: ``True`` when the frame's leading token is redundant.
        """
        return sampled_id == frame_id

    @classmethod
    def token_ids(cls, tokenizer: PreTrainedTokenizerBase) -> dict[str, int]:
        """IDs of this format's delimiter tokens that are in ``tokenizer``'s vocab."""
        ids: dict[str, int] = {}
        for token in (*cls.REQUIRED_TOKENS, *cls.OPTIONAL_TOKENS):
            token_id = tokenizer.convert_tokens_to_ids(token)
            # Hugging Face maps a token missing from the vocab to the unk ID,
            # which Granite also gives its own <|end_of_text|>.
            if isinstance(token_id, int) and (
                token_id != tokenizer.unk_token_id or token == tokenizer.unk_token
            ):
                ids[token] = token_id
        return ids

    @abstractmethod
    def extract_tool_calls(
        self, token_ids: list[int], tools: list[Any] | None = None, /
    ) -> tuple[list[int], list[ParsedToolCall]]:
        """Split IDs into the content before the first call plus every parse attempt.

        :param token_ids: One generation's token IDs.
        :param tools: Tool schemas, for formats whose bodies need value types.
        :return: Content IDs and one entry per call block, malformed ones included.
        """

    def _decode(self, token_ids: list[int], strip: bool = True) -> str:
        """Decode IDs to text, keeping special tokens inside the segment."""
        if not token_ids:
            return ""
        text = self._tokenizer.decode(token_ids, skip_special_tokens=False)
        if not isinstance(text, str):
            msg = "decode() of one sequence returns str"
            raise TypeError(msg)
        return text.strip() if strip else text

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


class Glm47Format(ChatFormat):
    """``<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value>...</tool_call>`` blocks.

    Every arg tag is its own special token, so a body splits at token IDs.
    Values are raw text: strings verbatim, other types as JSON, coerced to the
    tool schema. Blocks inside the thinking trace are skipped.
    """

    CALL_START = "<tool_call>"
    CALL_END = "</tool_call>"
    KEY_START = "<arg_key>"
    KEY_END = "</arg_key>"
    VALUE_START = "<arg_value>"
    VALUE_END = "</arg_value>"
    REASONING_END = REASONING_END_MARKER
    REQUIRED_TOKENS: ClassVar[tuple[str, ...]] = (
        CALL_START,
        CALL_END,
        KEY_START,
        KEY_END,
        VALUE_START,
        VALUE_END,
    )
    OPTIONAL_TOKENS: ClassVar[tuple[str, ...]] = (REASONING_END_MARKER,)

    def extract_tool_calls(
        self, token_ids: list[int], tools: list[Any] | None = None, /
    ) -> tuple[list[int], list[ParsedToolCall]]:
        """Parse tool blocks after any thinking trace."""
        params = _param_schemas(tools)

        def parse_body(raw: str, span: tuple[int, int]) -> ParsedToolCall:
            body_ids = token_ids[span[0] + 1 : span[1] - 1]
            return self._parsed_call(raw, body_ids, span, params)

        return self._tagged_calls(
            token_ids,
            self.CALL_START,
            self.CALL_END,
            parse_body,
            self.answer_start(token_ids),
        )

    def _parsed_call(
        self,
        raw: str,
        body_ids: list[int],
        span: tuple[int, int],
        params: dict[str, dict[str, dict[str, Any]]],
    ) -> ParsedToolCall:
        """One body: the name, then key/value tag pairs."""
        key_start, key_end, value_start, value_end = (
            self._ids[tag]
            for tag in (self.KEY_START, self.KEY_END, self.VALUE_START, self.VALUE_END)
        )
        index = _find(body_ids, key_start)
        name = self._decode(body_ids if index == -1 else body_ids[:index])
        if not name:
            return _failed(raw, span, ToolCallParseStatus.MISSING_NAME)
        schemas = params.get(name, {})
        arguments: dict[str, Any] = {}
        any_invalid = False
        while index != -1:
            key_close = _find(body_ids, key_end, index + 1)
            value_open = _find(body_ids, value_start, key_close + 1)
            value_close = _find(body_ids, value_end, value_open + 1)
            if -1 in (key_close, value_open, value_close) or self._decode(
                body_ids[key_close + 1 : value_open]
            ):
                return _failed(raw, span, ToolCallParseStatus.MALFORMED_STRUCTURE, name)
            key = self._decode(body_ids[index + 1 : key_close])
            text = self._decode(body_ids[value_open + 1 : value_close], strip=False)
            arguments[key], invalid = _coerce_xml_value(text, schemas.get(key))
            any_invalid = any_invalid or invalid
            index = _find(body_ids, key_start, value_close + 1)
        status = (
            ToolCallParseStatus.INVALID_JSON if any_invalid else ToolCallParseStatus.OK
        )
        return ParsedToolCall(
            raw=raw, name=name, arguments=arguments, token_span=span, status=status
        )


class ToolCallTagFormat(ChatFormat):
    """``<tool_call>`` blocks with JSON or XML bodies.

    JSON bodies hold one object with name and arguments (Qwen, Hermes,
    Granite 4). XML bodies hold one function block with parameter values as
    raw text (Nemotron). Blocks inside a thinking trace are skipped.
    """

    CALL_START = "<tool_call>"
    CALL_END = "</tool_call>"
    REASONING_END = REASONING_END_MARKER
    REQUIRED_TOKENS: ClassVar[tuple[str, ...]] = (CALL_START,)
    OPTIONAL_TOKENS: ClassVar[tuple[str, ...]] = (CALL_END, REASONING_END_MARKER)

    def extract_tool_calls(
        self, token_ids: list[int], tools: list[Any] | None = None, /
    ) -> tuple[list[int], list[ParsedToolCall]]:
        """Parse tool blocks after any thinking trace."""
        params = _param_schemas(tools)

        def parse_body(raw: str, span: tuple[int, int]) -> ParsedToolCall:
            if raw.startswith("<function="):
                return _parsed_xml_call(raw, span, params)
            return _parsed_json_call(raw, span)

        return self._tagged_calls(
            token_ids,
            self.CALL_START,
            self.CALL_END,
            parse_body,
            self.answer_start(token_ids),
        )


class Granite3Format(ChatFormat):
    """One ``<|tool_call|>`` tag, then a JSON array of calls to end of turn.

    Granite 3 emits no closing tag and no leading prose. Array items share the
    block span.
    """

    CALL_START = "<|tool_call|>"
    TURN_END = "<|end_of_text|>"
    REASONING_END = REASONING_END_MARKER
    REQUIRED_TOKENS: ClassVar[tuple[str, ...]] = (CALL_START,)
    OPTIONAL_TOKENS: ClassVar[tuple[str, ...]] = (REASONING_END_MARKER, TURN_END)

    def extract_tool_calls(
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
        # Generated ids keep the end-of-turn stop token after the array.
        turn_end = _find(token_ids, self._ids.get(self.TURN_END), tag + 1)
        raw = self._decode(token_ids[tag + 1 : None if turn_end == -1 else turn_end])
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

    def format_tool_calls(self, calls: list[dict[str, Any]], /) -> str | None:
        """``<|tool_call|>`` then the calls as a JSON array.

        The template writes no structured ``tool_calls``; its system prompt asks for this form.
        """
        return self.CALL_START + json.dumps(calls)


class Gemma4Format(ChatFormat):
    """``<|tool_call>call:name{key:value,...}<tool_call|>`` blocks.

    Argument values are Gemma's own ``key:value`` text, not JSON: strings wrap
    in ``<|"|>``, bare scalars stay strings for the executor to coerce, and
    braces nest. Reasoning is a ``<|channel>thought ... <channel|>`` block.
    """

    CALL_START = "<|tool_call>"
    CALL_END = "<tool_call|>"
    RESPONSE_START = "<|tool_response>"
    RESPONSE_END = "<tool_response|>"
    REASONING_END = "<channel|>"
    REQUIRED_TOKENS: ClassVar[tuple[str, ...]] = (
        CALL_START,
        CALL_END,
        RESPONSE_START,
        RESPONSE_END,
    )
    OPTIONAL_TOKENS: ClassVar[tuple[str, ...]] = (REASONING_END,)

    def extract_tool_calls(
        self, token_ids: list[int], _tools: list[Any] | None = None, /
    ) -> tuple[list[int], list[ParsedToolCall]]:
        """Parse native Gemma calls after any thought channel."""
        return self._tagged_calls(
            token_ids,
            self.CALL_START,
            self.CALL_END,
            _parsed_gemma_call,
            self.answer_start(token_ids),
        )

    def format_tool_results(self, messages: list[dict[str, str]], /) -> str | None:
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


class HarmonyFormat(ChatFormat):
    """Channel blocks addressed ``to=functions.name`` (GPT-OSS).

    Each ``<|start|>`` block names its recipient in the header. Function blocks
    hold JSON arguments and close with ``<|call|>``. Other blocks are content.
    Parsing stops at the first ``<|return|>``. Reasoning is the analysis
    channel; the answer follows ``<|channel|>final<|message|>``.
    """

    BLOCK_START = "<|start|>"
    MESSAGE = "<|message|>"
    CALL_END = "<|call|>"
    BLOCK_END = "<|end|>"
    RETURN = "<|return|>"
    CHANNEL = "<|channel|>"
    FINAL_CHANNEL = "final"
    REQUIRED_TOKENS: ClassVar[tuple[str, ...]] = (
        BLOCK_START,
        MESSAGE,
        CALL_END,
        BLOCK_END,
    )
    OPTIONAL_TOKENS: ClassVar[tuple[str, ...]] = (RETURN, CHANNEL)

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        """Resolve delimiter IDs and the ``<|channel|>final<|message|>`` answer marker."""
        super().__init__(tokenizer)
        if self.CHANNEL in self._ids:
            # Encoded whole, so ``final`` tokenizes as it does between the two special tokens.
            self.reasoning_end_ids = tuple(
                tokenizer.encode(
                    f"{self.CHANNEL}{self.FINAL_CHANNEL}{self.MESSAGE}",
                    add_special_tokens=False,
                )
            )
        self._message_end_ids = frozenset(
            self._ids[token]
            for token in (self.BLOCK_END, self.CALL_END, self.RETURN)
            if token in self._ids
        )

    def format_tool_results(self, messages: list[dict[str, str]], /) -> str | None:
        """One ``functions.name`` commentary block per result, then the assistant header.

        The template renders a tool result only after a structured assistant
        ``tool_calls`` entry, which the harness does not keep. Content is JSON
        encoded, as the template's ``tojson`` filter does.
        """
        if not messages or any(message["role"] != "tool" for message in messages):
            return None
        blocks = "".join(
            f"{self.BLOCK_START}{HARMONY_FUNCTION_PREFIX}{message.get('name') or 'unknown'}"
            f" to=assistant{self.CHANNEL}commentary{self.MESSAGE}"
            f"{json.dumps(message['content'], ensure_ascii=False)}{self.BLOCK_END}"
            for message in messages
        )
        return f"{blocks}{self.BLOCK_START}assistant"

    def closes_turn(self, sampled_id: int, frame_id: int) -> bool:
        """``<|return|>`` and ``<|call|>`` end a message as ``<|end|>`` does.

        The sampled token stays in the transcript, so a final answer reads
        ``<|return|><|start|>`` where a template re-render has ``<|end|><|start|>``.
        """
        return sampled_id == frame_id or (
            frame_id == self._ids[self.BLOCK_END]
            and sampled_id in self._message_end_ids
        )

    def extract_tool_calls(
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
        # The generation prompt opens the first block, so its header may start at 0.
        index = 0 if ids else -1
        while index != -1:
            header_start = index + 1 if ids[index] == start_id else index
            message_at = _find(ids, message_id, header_start)
            if message_at == -1:
                break
            next_start = _find(ids, start_id, header_start)
            if next_start != -1 and next_start < message_at:
                index = next_start
                continue
            stops = [_find(ids, stop, message_at + 1) for stop in stop_ids]
            body_end = min((s for s in stops if s != -1), default=len(ids))
            closed = body_end < len(ids) and ids[body_end] != start_id
            next_index = body_end + 1 if closed else body_end
            match = HARMONY_RECIPIENT_RE.search(
                self._decode(ids[header_start:message_at])
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


class DsmlFormat(ChatFormat):
    """DSML ``tool_calls`` blocks of ``invoke`` elements (DeepSeek-V4).

    Tags span several tokens around the one :data:`DSML` special token, so a
    block tag is that token with ``<`` or ``</`` before it and a block name
    after it. Each ``parameter`` says whether its value is a string; other
    values are JSON. Tool results return as ``<tool_result>`` user turns. The
    tokenizer ships no chat template; :attr:`CHAT_TEMPLATE` renders the encoder's.
    """

    CHAT_TEMPLATE = DSML_CHAT_TEMPLATE
    REASONING_END = REASONING_END_MARKER
    REQUIRED_TOKENS: ClassVar[tuple[str, ...]] = (DSML,)
    OPTIONAL_TOKENS: ClassVar[tuple[str, ...]] = (REASONING_END_MARKER,)

    def extract_tool_calls(
        self, token_ids: list[int], tools: list[Any] | None = None, /
    ) -> tuple[list[int], list[ParsedToolCall]]:
        """Parse DSML blocks after any thinking trace."""
        params = _param_schemas(tools)
        first = self._block_tag(token_ids, "<", self.answer_start(token_ids))
        if first == -1:
            return token_ids, []
        calls: list[ParsedToolCall] = []
        start = first
        while start != -1:
            close = self._block_tag(token_ids, "</", start + 1)
            if close == -1:
                raw = self._decode(token_ids[start:])
                span = (start, len(token_ids))
                calls.append(_failed(raw, span, ToolCallParseStatus.UNCLOSED_BLOCK))
                break
            end = self._tag_end(token_ids, close)
            calls.extend(
                _parsed_dsml_calls(
                    self._decode(token_ids[start:end]), (start, end), params
                )
            )
            start = self._block_tag(token_ids, "<", end)
        return token_ids[:first], calls

    def _block_tag(self, token_ids: list[int], opener: str, start: int) -> int:
        """Index of the first ``opener`` block tag's leading token at or after ``start``, or ``-1``."""
        dsml_id = self._ids[DSML]
        at = _find(token_ids, dsml_id, start + 1)
        while at != -1:
            following = _find(token_ids, dsml_id, at + 1)
            after = self._decode(
                token_ids[at + 1 : None if following == -1 else following]
            )
            if DSML_BLOCK_RE.match(after):
                # The opener may itself span tokens, e.g. ``<`` then ``/``.
                for lead in range(at - 1, max(start, at - len(opener)) - 1, -1):
                    if self._decode(token_ids[lead:at], strip=False).endswith(opener):
                        return lead
            at = following
        return -1

    def _tag_end(self, token_ids: list[int], start: int) -> int:
        """Index just past the first token after ``start`` holding ``>``."""
        for index in range(start + 1, len(token_ids)):
            if ">" in self._decode(token_ids[index : index + 1]):
                return index + 1
        return len(token_ids)


CHAT_FORMATS: dict[str, type[ChatFormat]] = {
    # Ahead of tool_call: the GLM vocab also holds <tool_call>.
    "glm47": Glm47Format,
    "tool_call": ToolCallTagFormat,
    "granite3": Granite3Format,
    "gemma4": Gemma4Format,
    "harmony": HarmonyFormat,
    "dsml": DsmlFormat,
}


def get_chat_format(name: str, tokenizer: PreTrainedTokenizerBase) -> ChatFormat:
    """Build the named format over ``tokenizer``.

    :param name: Format name from :data:`CHAT_FORMATS`.
    :param tokenizer: Tokenizer defining the delimiter IDs.
    :return: The format.
    """
    try:
        format_cls = CHAT_FORMATS[name]
    except KeyError:
        available = ", ".join(sorted(CHAT_FORMATS))
        msg = f"Unknown chat_format {name!r}. Available: {available}."
        raise ValueError(msg) from None
    return format_cls(tokenizer)


def detect_chat_format(tokenizer: PreTrainedTokenizerBase) -> ChatFormat | None:
    """First registered format whose required tokens are all in the vocab, or ``None``.

    :param tokenizer: Tokenizer to probe.
    :return: The matching format, or ``None`` when none matches.
    """
    for format_cls in CHAT_FORMATS.values():
        if set(format_cls.REQUIRED_TOKENS) <= format_cls.token_ids(tokenizer).keys():
            return format_cls(tokenizer)
    return None


def resolve_chat_format(
    tokenizer: PreTrainedTokenizerBase, name: str | None
) -> ChatFormat | None:
    """The named format, else the one the vocab matches, else ``None``.

    :param tokenizer: Tokenizer defining the delimiter IDs.
    :param name: Format name from :data:`CHAT_FORMATS`, or ``None`` to detect.
    :return: The format, or ``None`` when ``name`` is unset and none matches.
    """
    if name is None:
        return detect_chat_format(tokenizer)
    return get_chat_format(name, tokenizer)


def frame_repeats_terminator(
    last_id: int,
    frame_id: int,
    special_ids: Collection[int],
    chat_format: ChatFormat | None,
) -> bool:
    """Whether a feedback frame's first token repeats the transcript's end-of-turn token.

    :param last_id: Last token of the transcript.
    :param frame_id: First token of the feedback frame.
    :param special_ids: The tokenizer's special-token ids.
    :param chat_format: The model family's chat format, or ``None``.
    :return: ``True`` when the frame's first token should be dropped.
    """
    if last_id not in special_ids:
        return False
    if chat_format is None:
        return last_id == frame_id
    return chat_format.closes_turn(last_id, frame_id)


def action_text(
    decode: Callable[..., object],
    token_ids: list[int],
    chat_format: ChatFormat | None,
) -> str:
    """Text the env acts on: what follows the reasoning close, else the whole generation.

    A generation with no close token still splits at a plain-text ``</think>``,
    for models taught that format by the prompt.

    :param decode: The sampling tokenizer's ``decode``.
    :param token_ids: One generation's token IDs.
    :param chat_format: The family's chat format, or ``None`` when none matched.
    :return: The action text, special tokens removed.
    """
    start = 0 if chat_format is None else chat_format.answer_start(token_ids)
    text = decode(token_ids[start:], skip_special_tokens=True)
    if not isinstance(text, str):
        msg = "decode() of one sequence returns str"
        raise TypeError(msg)
    return text.strip() if start else action_span(text)


def action_span(gen_text: str) -> str:
    """Text after the first ``</think>``, else the whole generation.

    A template that prefills an open ``<think>`` yields ``reasoning</think>action``
    with no opening tag. A reasoning block that never closes passes through whole.
    """
    _reasoning, closed, action = gen_text.partition(REASONING_END_MARKER)
    return action.strip() if closed else gen_text


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


def _parsed_dsml_calls(
    raw: str,
    span: tuple[int, int],
    params: dict[str, dict[str, dict[str, Any]]],
) -> list[ParsedToolCall]:
    """Every ``invoke`` in one DSML block, sharing its span; a block with none is malformed."""
    calls: list[ParsedToolCall] = []
    for invoke in DSML_INVOKE_RE.finditer(raw):
        call_raw, name, body = invoke.group(0), invoke.group(1).strip(), invoke.group(2)
        if not name:
            calls.append(_failed(call_raw, span, ToolCallParseStatus.MISSING_NAME))
            continue
        schemas = params.get(name, {})
        arguments: dict[str, Any] = {}
        any_invalid = False
        for parameter in DSML_PARAMETER_RE.finditer(body):
            key, is_string, text = parameter.groups()
            if is_string == "true":
                arguments[key] = text
                continue
            arguments[key], invalid = _coerce_xml_value(text, schemas.get(key))
            any_invalid = any_invalid or invalid
        status = (
            ToolCallParseStatus.INVALID_JSON if any_invalid else ToolCallParseStatus.OK
        )
        calls.append(
            ParsedToolCall(
                raw=call_raw,
                name=name,
                arguments=arguments,
                token_span=span,
                status=status,
            )
        )
    return calls or [_failed(raw, span, ToolCallParseStatus.MALFORMED_STRUCTURE)]


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

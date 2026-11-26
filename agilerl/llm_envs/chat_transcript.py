# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Render chat transcripts with a chat template and mark the assistant tokens to train.

Shared by SFT and DPO rows written as OpenAI-style messages with optional ``tools``.
"""

from __future__ import annotations

import json
import os
import re
import warnings
from bisect import bisect_right
from dataclasses import dataclass
from functools import cache, lru_cache
from itertools import pairwise
from typing import TYPE_CHECKING, Any

from agilerl.arena.traces_format import REASONING_KEYS, WEIGHT
from agilerl.llm_envs.chat_formats import (
    ChatFormat,
    ToolCallParseStatus,
    frame_repeats_terminator,
)
from agilerl.llm_envs.reasoning import REASONING_END_MARKER, REASONING_START_MARKER
from agilerl.utils.chat_template import (
    feedback_frame_parts,
    special_token_ids,
)

if TYPE_CHECKING:
    from collections.abc import Collection

    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

TURN_OPENER = {"role": "user", "content": "."}


class UnplacedTurnError(ValueError):
    """The template render holds no span an assistant turn can be cut from."""


SLOW_TOKENIZER = (
    "Cutting turns out of the chat template's render needs a fast tokenizer, "
    "which gives each token's character span."
)
UNPLACED_TURN = (
    "The chat template renders an assistant turn that does not extend "
    "its generation prompt, so the turn's sampled tokens cannot be cut out."
)


def chat_messages(value: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize one row's OpenAI-style ``messages`` for a chat template.

    Accepts a JSON string or a list. Drops ``None`` fields and the training
    ``weight``, which is not message content, joins a list of text content
    parts into one string, and decodes string tool-call arguments, which HF
    chat templates expect as a dict. A list is Arrow data, which fills
    every struct key at every depth, so ``None`` is also dropped inside its
    struct arguments; nulls in JSON text are real and kept.

    :param value: The row's ``messages`` value.
    :return: Messages ready for ``apply_chat_template``.
    :raises ValueError: If the row or a call's arguments are not valid JSON, a
        call's arguments are not an object, or a content part is not text.
    """
    from_arrow = not isinstance(value, str)
    raw = (
        _parsed_json(value, "The messages column") if isinstance(value, str) else value
    )
    messages = []
    for raw_message in raw:
        message = {
            key: item
            for key, item in raw_message.items()
            if item is not None and key != WEIGHT
        }
        if isinstance(message.get("content"), list):
            message["content"] = _joined_text_parts(message["content"], message["role"])
        if "tool_calls" in message:
            message["tool_calls"] = [
                _decoded_tool_call(call, from_arrow) for call in message["tool_calls"]
            ]
        messages.append(message)
    return messages


def _joined_text_parts(parts: list[dict[str, Any]], role: str) -> str:
    """The ``text`` of each content part, joined by newlines as vLLM joins them.

    :raises ValueError: If a part is not a text part.
    """
    texts = []
    for part in parts:
        text = part.get("text")
        if part.get("type") != "text" or not isinstance(text, str):
            msg = (
                f"A {role} message has a {part.get('type')!r} content part; "
                "only text parts can be trained."
            )
            raise ValueError(msg)
        texts.append(text)
    return "\n".join(texts)


REASONING_READ = re.compile(
    r"""\.get\(\s*['"](reasoning_content|thinking|reasoning)['"]\s*\)"""
    r"""|\.(reasoning_content|thinking|reasoning)\b"""
    r"""|\[['"](reasoning_content|thinking|reasoning)['"]\]"""
)
"""A chat template reading one of :data:`REASONING_KEYS` off a message."""


def template_reasoning_key(tokenizer: PreTrainedTokenizerBase) -> str | None:
    """The reasoning key the tokenizer's chat template reads off a message, if any.

    :param tokenizer: Tokenizer whose ``chat_template`` is searched.
    :return: The first of :data:`REASONING_KEYS` the template reads, or ``None``.
    """
    # Tokenizers that render without a Jinja template carry no ``chat_template``.
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str):
        return None
    return _reasoning_key_read_by(template)


@lru_cache(maxsize=8)
def _reasoning_key_read_by(template: str) -> str | None:
    """The first of :data:`REASONING_KEYS` a chat template source reads, or ``None``."""
    read = {
        group
        for match in REASONING_READ.finditer(template)
        for group in match.groups()
        if group
    }
    return next((key for key in REASONING_KEYS if key in read), None)


def _with_reasoning_key(message: dict[str, Any], key: str | None) -> dict[str, Any]:
    """An assistant message with its reasoning under ``key``, the key the template reads.

    :param message: Normalized message.
    :param key: Target key, or ``None`` when the template reads none: the
        reasoning is then dropped, with a warning once per process.
    :return: The message, with any other reasoning key moved to ``key``.
    :raises ValueError: If the message sets two reasoning keys to different text.
    """
    if message.get("role") != "assistant":
        return message
    present = {name: message[name] for name in REASONING_KEYS if message.get(name)}
    if key is None:
        if present:
            _warn_reasoning_dropped()
        return {
            name: item for name, item in message.items() if name not in REASONING_KEYS
        }
    values = list(present.values())
    if any(value != values[0] for value in values[1:]):
        msg = (
            f"An assistant message sets {sorted(present)} to different reasoning; "
            "keep one."
        )
        raise ValueError(msg)
    if not present or set(present) == {key}:
        return message
    value = next(iter(present.values()))
    kept = {name: item for name, item in message.items() if name not in REASONING_KEYS}
    return {**kept, key: value}


@cache
def _warn_reasoning_dropped() -> None:
    """Warn, once, that a chat template without a reasoning key drops assistant reasoning."""
    warnings.warn(
        "The chat template reads none of "
        f"{list(REASONING_KEYS)}, so assistant reasoning is dropped from SFT "
        "turns, prompts and the OPSD reference.",
        stacklevel=4,
    )


def _as_messages(value: str | list[dict[str, Any]], role: str) -> list[dict[str, Any]]:
    """A string as one ``role`` message, or a list of messages normalized."""
    if isinstance(value, str):
        return [{"role": role, "content": value}]
    return chat_messages(value)


def _parsed_json(text: str, label: str) -> Any:  # noqa: ANN401 -- any JSON value
    """Decode JSON ``text``, naming ``label`` in the error."""
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        msg = f"{label} is not valid JSON ({error}): {text[:200]!r}"
        raise ValueError(msg) from error


def without_none_keys(value: object) -> object:
    """``value`` without ``None`` dict entries at any depth; ``None`` list items are data."""
    if isinstance(value, dict):
        return {
            key: without_none_keys(item)
            for key, item in value.items()
            if item is not None
        }
    if isinstance(value, list):
        return [without_none_keys(item) for item in value]
    return value


def json_cell(value: Any, label: str) -> Any:  # noqa: ANN401 -- any nested cell
    """A nested traces cell: JSON text decoded, Arrow data without the ``None`` keys it fills.

    :param value: The cell, JSON text or a native nested value.
    :param label: Column name for the error.
    :return: The decoded value.
    :raises ValueError: If a string cell is not valid JSON.
    """
    if isinstance(value, str):
        return _parsed_json(value, f"The {label} column")
    return without_none_keys(value)


def _decoded_tool_call(call: dict[str, Any], from_arrow: bool) -> dict[str, Any]:
    """Tool call without ``None`` fields, with its arguments decoded to an object."""
    call = {key: item for key, item in call.items() if item is not None}
    if "function" in call:
        return {**call, "function": _decoded_tool_call(call["function"], from_arrow)}
    if "arguments" not in call:
        return call
    arguments = call["arguments"]
    label = f"The arguments of tool call {call.get('name')!r}"
    if isinstance(arguments, str):
        arguments = _parsed_json(arguments, label)
    elif from_arrow:
        arguments = without_none_keys(arguments)
    if not isinstance(arguments, dict):
        msg = f"{label} must be an object, got {type(arguments).__name__}."
        raise ValueError(msg)
    return {**call, "arguments": arguments}


def encode_chat_transcript(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    chat_format: ChatFormat | None,
    trained_from: int = 0,
    skipped: Collection[int] = (),
) -> tuple[list[int], list[bool]]:
    """Token ids of a multi-turn conversation, with an assistant mask.

    The ids follow the chat template's render of the messages through the last
    assistant turn. A trailing newline after the final end-of-turn token is
    dropped, and a turn with no end-of-turn token gains the eos token. Between
    assistant turns, trailing whitespace of the generation prompt that the
    completed turn does not continue stays in the gap, outside the assistant
    mask. Each assistant turn runs from where that render leaves the generation
    prompt through its end-of-turn token. A generation prompt that is not a
    prefix of the completed turn is not spliced on. Tool results a format
    renders inside an assistant turn stay there. Two assistant turns in a row
    are allowed only when the later one is context. Messages after the last
    assistant turn are dropped. Assistant messages' reasoning is moved to the
    key the chat template reads, see :func:`template_reasoning_key`.
    Assistant turns before ``trained_from`` are context: masked out, and their
    tool calls are not checked against the chat format's parser. With
    ``trained_from`` set, the context is the template's render of the messages
    before the first trained turn with the generation prompt, the text a
    rollout samples that turn after. Assistant turns at ``skipped`` are context
    too.

    :param tokenizer: Tokenizer whose chat template renders the conversation.
    :param messages: Normalized messages, see :func:`chat_messages`.
    :param tools: Tool schemas passed to the template, or ``None``.
    :param chat_format: The model family's chat format, see
        :func:`~agilerl.llm_envs.chat_formats.detect_chat_format`.
    :param trained_from: Index of the first message whose assistant turns train.
    :param skipped: Indices of assistant messages that do not train.
    :return: Token ids and, per id, whether it is a trained assistant token.
    :raises ValueError: If the conversation has no prompt or no assistant
        turn, has a trained assistant turn right after another, the template renders an
        assistant turn that does not extend the generation prompt, the chat
        format does not parse a turn's rendered tool calls back to its calls,
        or the trained turn does not start where the generation prompt ends.
    """
    reasoning_key = template_reasoning_key(tokenizer)
    messages = [_with_reasoning_key(message, reasoning_key) for message in messages]
    render_kwargs: dict[str, Any] = {} if tools is None else {"tools": tools}
    assistant_indices = [
        idx for idx, message in enumerate(messages) if message["role"] == "assistant"
    ]
    if not assistant_indices:
        msg = "An SFT conversation needs at least one assistant message."
        raise ValueError(msg)
    if assistant_indices[0] == 0:
        msg = "An SFT conversation needs a message before its first assistant turn."
        raise ValueError(msg)
    skipped = set(skipped)
    trained_indices = [
        idx for idx in assistant_indices if idx >= trained_from and idx not in skipped
    ]
    trained = set(trained_indices)
    if not trained_indices:
        msg = "An SFT conversation needs an assistant turn to train after its prompt."
        raise ValueError(msg)
    for previous, idx in pairwise(assistant_indices):
        if idx == previous + 1 and idx in trained:
            msg = f"Messages {previous} and {idx} are back-to-back assistant turns."
            raise ValueError(msg)

    render_messages = _messages_for_render(messages, chat_format)
    ids, mask = _transcript_ids(
        tokenizer,
        render_messages,
        messages,
        assistant_indices,
        trained,
        render_kwargs,
        chat_format,
        tools,
    )
    if trained_from == 0:
        return ids, mask
    return _after_prompt_render(
        tokenizer, render_messages[: trained_indices[0]], ids, mask, render_kwargs
    )


def _transcript_ids(
    tokenizer: PreTrainedTokenizerBase,
    render_messages: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    assistant_indices: list[int],
    trained: Collection[int],
    render_kwargs: dict[str, Any],
    chat_format: ChatFormat | None,
    tools: list[dict[str, Any]] | None,
) -> tuple[list[int], list[bool]]:
    """Ids and assistant mask of the conversation through its last assistant turn.

    Cut from the template's render, else built turn by turn.

    :raises ValueError: If a rollout could not frame the messages between two
        turns, or neither way places every turn.
    """
    for previous, idx in pairwise(assistant_indices):
        if idx > previous + 1:
            _feedback_frame(
                tokenizer,
                render_messages[previous + 1 : idx],
                render_kwargs,
                chat_format,
            )
    kept = assistant_indices[-1] + 1
    canonical = _render(tokenizer, render_messages[:kept], False, render_kwargs)
    try:
        if not tokenizer.is_fast:
            raise UnplacedTurnError(SLOW_TOKENIZER)
        ids, mask = _assemble_from_template(
            tokenizer,
            render_messages[:kept],
            messages[:kept],
            canonical,
            render_kwargs,
            chat_format,
            tools,
            trained,
        )
    except UnplacedTurnError:
        # A slow tokenizer gives no character offsets, and a turn with no
        # header of its own, after a tool result the chat format writes into
        # the turn before (Gemma 4), has nothing to cut at: both are built
        # turn by turn.
        spliced = _spliced_turn_ids(
            tokenizer,
            render_messages,
            messages,
            assistant_indices,
            trained,
            render_kwargs,
            chat_format,
            tools,
        )
        if spliced is None or not _decodes_template(tokenizer, spliced[0], canonical):
            raise
        return spliced
    return ids, mask


def _render_kwargs(
    tools: list[dict[str, Any]] | None, chat_template_kwargs: dict[str, Any] | None
) -> dict[str, Any]:
    """``apply_chat_template`` kwargs: ``chat_template_kwargs`` plus ``tools`` when there are any."""
    kwargs = dict(chat_template_kwargs or {})
    if tools:
        kwargs["tools"] = tools
    return kwargs


def render_chat_prompt(
    tokenizer: PreTrainedTokenizerBase,
    messages: str | list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    chat_format: ChatFormat | None,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> str:
    """Chat-template text of OpenAI-style ``messages``, ending in the generation prompt.

    This is the context SFT trains the next assistant turn after, see
    ``trained_from`` in :func:`encode_chat_transcript`. Messages are normalized
    by :func:`chat_messages`, reasoning moves to the key the template reads,
    past tool calls are written the way the chat format parses them, and tool
    results are named after the call they answer.

    :param tokenizer: Tokenizer whose chat template renders the prompt.
    :param messages: The messages, as JSON text or a list.
    :param tools: Tool schemas offered on the turn; ``None`` or empty for none.
    :param chat_format: The model family's chat format, which writes past tool calls.
    :param chat_template_kwargs: Extra ``apply_chat_template`` kwargs.
    :return: The prompt text the model samples its next turn after.
    :raises ValueError: If there are no messages or the last is an assistant turn.
    """
    reasoning_key = template_reasoning_key(tokenizer)
    normalized = [
        _with_reasoning_key(message, reasoning_key)
        for message in chat_messages(messages)
    ]
    if not normalized or normalized[-1]["role"] == "assistant":
        msg = "A chat prompt needs messages and must not end with an assistant turn."
        raise ValueError(msg)
    rendered = _messages_for_render(normalized, chat_format)
    last = max(
        (
            idx
            for idx, message in enumerate(normalized)
            if message["role"] == "assistant"
        ),
        default=None,
    )
    if last is not None:
        rendered[last + 1 :] = _named_tool_results(
            normalized[last], rendered[last + 1 :]
        )
    return _render(
        tokenizer, rendered, True, _render_kwargs(tools, chat_template_kwargs)
    )


def assistant_turn_text(
    tokenizer: PreTrainedTokenizerBase,
    message: dict[str, Any],
    tools: list[dict[str, Any]] | None,
    chat_format: ChatFormat | None,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> str:
    """One assistant turn as the model samples it: reasoning, content and tool calls.

    Includes a reasoning opener the generation prompt prefills, keeps special
    tokens inside the turn, and drops its end-of-turn token. A template that
    reads no reasoning key gets no reasoning.

    :param tokenizer: Tokenizer whose chat template renders the turn.
    :param message: A normalized assistant message, see :func:`chat_messages`.
    :param tools: Tool schemas offered on the turn, or ``None``.
    :param chat_format: The model family's chat format, which writes tool calls.
    :param chat_template_kwargs: Extra ``apply_chat_template`` kwargs.
    :return: The turn's text.
    """
    render_kwargs = _render_kwargs(tools, chat_template_kwargs)
    turn = _written_tool_calls(
        _with_reasoning_key(message, template_reasoning_key(tokenizer)), chat_format
    )
    special_ids = special_token_ids(tokenizer)
    scaffold = _turn_scaffold(tokenizer, render_kwargs, special_ids)
    turn_ids = _assistant_turn_ids(
        tokenizer, turn, render_kwargs, special_ids, scaffold
    )
    # Keep special tokens inside the turn (tool-call and channel markers); drop its end-of-turn token.
    text = tokenizer.decode(turn_ids[:-1], skip_special_tokens=False)
    if not isinstance(text, str):
        msg = "tokenizer.decode must return a string"
        raise TypeError(msg)
    opener = _prefilled_opener(
        tokenizer,
        turn,
        render_kwargs,
        scaffold[0],
        _generation_addition(tokenizer, render_kwargs),
    )
    return (opener + text).strip()


def _prefilled_opener(
    tokenizer: PreTrainedTokenizerBase,
    message: dict[str, Any],
    render_kwargs: dict[str, Any],
    prompt: str,
    addition: str,
) -> str:
    """The open reasoning block the generation prompt prefills and ``message``'s render continues, else ``""``.

    :param prompt: Render of :data:`TURN_OPENER` with the generation prompt.
    :param addition: The text the generation prompt adds to ``prompt``.
    """
    at = addition.rfind(REASONING_START_MARKER)
    if at < 0 or REASONING_END_MARKER in addition[at:]:
        return ""
    rendered = _render(tokenizer, [TURN_OPENER, message], False, render_kwargs)
    shared = len(os.path.commonprefix([prompt, rendered]))
    return prompt[len(prompt) - len(addition) + at : shared]


def _spliced_turn_ids(
    tokenizer: PreTrainedTokenizerBase,
    render_messages: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    assistant_indices: list[int],
    trained_indices: Collection[int],
    render_kwargs: dict[str, Any],
    chat_format: ChatFormat | None,
    tools: list[dict[str, Any]] | None,
) -> tuple[list[int], list[bool]] | None:
    """Ids and assistant mask built turn by turn: the prompt render, then each turn and the feedback before it.

    :return: The ids and mask, or ``None`` when a context turn follows another
        with no feedback between them to splice.
    """
    prompt = _render(
        tokenizer, render_messages[: assistant_indices[0]], True, render_kwargs
    )
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    mask = [False] * len(ids)
    special_ids = special_token_ids(tokenizer)
    scaffold = _turn_scaffold(tokenizer, render_kwargs, special_ids)
    for previous, idx in zip(
        [None, *assistant_indices], assistant_indices, strict=False
    ):
        if previous is not None:
            between = render_messages[previous + 1 : idx]
            if not between:
                return None
            feedback = _feedback_ids(tokenizer, between, render_kwargs, chat_format)
            if len(feedback) > 1 and frame_repeats_terminator(
                ids[-1], feedback[0], special_ids, chat_format
            ):
                feedback = feedback[1:]
            ids += feedback
            mask += [False] * len(feedback)
        turn = _assistant_turn_ids(
            tokenizer, render_messages[idx], render_kwargs, special_ids, scaffold
        )
        trained = idx in trained_indices
        if trained:
            _check_tool_calls_parse(messages[idx], turn, tools, chat_format)
        ids += turn
        mask += [trained] * len(turn)
    return ids, mask


def _after_prompt_render(
    tokenizer: PreTrainedTokenizerBase,
    prompt_messages: list[dict[str, Any]],
    ids: list[int],
    mask: list[bool],
    render_kwargs: dict[str, Any],
) -> tuple[list[int], list[bool]]:
    """``ids`` with the context before the first trained token replaced by the prompt's render.

    A template may rewrite earlier turns once a later one follows (gpt-oss drops
    their reasoning), but a rollout samples the turn after the prompt alone.

    :param prompt_messages: Messages before the first trained turn, as rendered.
    :param ids: Transcript ids from the template's render of the whole conversation.
    :param mask: Their assistant mask.
    :return: The prompt's ids with the generation prompt, then the trained ids.
    :raises ValueError: If the trained turn does not start where the
        generation prompt ends, as when it closes a thinking block the turn's
        reasoning goes in.
    """
    prompt = _render(tokenizer, prompt_messages, True, render_kwargs)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    start = mask.index(True)
    if ids[:start] == prompt_ids:
        return ids, mask
    plain = _render(tokenizer, prompt_messages, False, render_kwargs)
    addition = prompt[len(os.path.commonprefix([prompt, plain])) :]
    context = str(tokenizer.decode(ids[:start]))
    if not context.endswith(addition):
        msg = (
            f"The template starts the trained turn after {context[-80:]!r}, "
            f"not after the generation prompt {addition!r}. A completion with "
            "reasoning needs a generation prompt that leaves the thinking block "
            "open (Qwen3.5: chat_template_kwargs enable_thinking: true)."
        )
        raise ValueError(msg)
    return prompt_ids + ids[start:], [False] * len(prompt_ids) + mask[start:]


def _messages_for_render(
    messages: list[dict[str, Any]], chat_format: ChatFormat | None
) -> list[dict[str, Any]]:
    """``messages`` with tool results named and tool calls written into content."""
    rendered: list[dict[str, Any]] = []
    previous: int | None = None
    for message in messages:
        if message["role"] != "assistant":
            rendered.append(message)
            continue
        if previous is not None:
            rendered[previous + 1 :] = _named_tool_results(
                messages[previous], rendered[previous + 1 :]
            )
        rendered.append(_written_tool_calls(message, chat_format))
        previous = len(rendered) - 1
    return rendered


def _decodes_template(
    tokenizer: PreTrainedTokenizerBase, ids: list[int], rendered: str
) -> bool:
    """Whether ``ids`` are ``rendered``, aside from a final newline or a trailing eos."""
    decoded = tokenizer.decode(ids)
    if not isinstance(decoded, str):
        return False
    accepted = {rendered, rendered.removesuffix("\n")}
    eos = tokenizer.eos_token
    if isinstance(eos, str) and eos:
        accepted.update(item + eos for item in tuple(accepted))
    return decoded in accepted


def generation_prompt_addition(
    tokenizer: PreTrainedTokenizerBase,
    tools: list[dict[str, Any]] | None,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> str:
    """Text ``add_generation_prompt`` appends after one user message.

    :param tokenizer: Tokenizer whose chat template renders the prompt.
    :param tools: Tool schemas offered on the turn, or ``None``.
    :param chat_template_kwargs: Extra ``apply_chat_template`` kwargs.
    :return: The generation prompt's text.
    """
    return _generation_addition(tokenizer, _render_kwargs(tools, chat_template_kwargs))


def _generation_addition(
    tokenizer: PreTrainedTokenizerBase, render_kwargs: dict[str, Any]
) -> str:
    """Text ``add_generation_prompt`` appends after one user message."""
    opener = [TURN_OPENER]
    prompted = _render(tokenizer, opener, True, render_kwargs)
    plain = _render(tokenizer, opener, False, render_kwargs)
    return prompted[len(os.path.commonprefix([prompted, plain])) :]


def _common_length(left: str, right: str) -> int:
    """Length of the longest common prefix of two strings, by halving slice compares."""
    low, high = 0, min(len(left), len(right))
    while low < high:
        middle = (low + high + 1) // 2
        if left[low:middle] == right[low:middle]:
            low = middle
        else:
            high = middle - 1
    return low


@dataclass(frozen=True)
class TemplateRender:
    """The template's render of the conversation: text, ids, and each id's character span."""

    text: str
    ids: list[int]
    starts: list[int]
    ends: list[int]

    def token_at(self, char: int) -> int:
        """Index of the first id that ends after character ``char``."""
        return bisect_right(self.ends, char)

    def on_boundary(self, char: int) -> bool:
        """Whether no id spans character ``char`` from before it."""
        at = self.token_at(char)
        return at == len(self.ids) or self.starts[at] >= char

    def ids_between(
        self, tokenizer: PreTrainedTokenizerBase, begin: int, end: int
    ) -> list[int]:
        """Ids of ``text[begin:end]``: the render's own ids when no id crosses either end."""
        if self.on_boundary(begin) and self.on_boundary(end):
            return self.ids[self.token_at(begin) : self.token_at(end)]
        return tokenizer.encode(self.text[begin:end], add_special_tokens=False)


def _rendered(tokenizer: PreTrainedTokenizerBase, text: str) -> TemplateRender:
    """``text`` with its ids and their character spans, from one fast-tokenizer pass."""
    encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    spans = encoding["offset_mapping"]
    if len(spans) != len(encoding["input_ids"]):
        raise UnplacedTurnError(SLOW_TOKENIZER)
    return TemplateRender(
        text,
        list(encoding["input_ids"]),
        [begin for begin, _ in spans],
        [end for _, end in spans],
    )


@dataclass(frozen=True)
class TurnFrame:
    """What every assistant turn the template renders opens with."""

    # Text ``add_generation_prompt`` appends after one user message.
    prompt: str
    # ``prompt`` through its last special token.
    header: str
    # Text an assistant turn opens with: the block id and the assistant's
    # name (``<|im_start|>assistant``), else ``prompt`` stripped.
    opener: str
    # The special id that opens ``prompt`` and every other block, and the id
    # after it that names the assistant; ``None`` when ``prompt`` opens with
    # no special token.
    block: int | None
    role: int | None


def _turn_frame(
    tokenizer: PreTrainedTokenizerBase,
    render_kwargs: dict[str, Any],
    specials: set[int],
) -> TurnFrame:
    """The generation prompt, its header, and the ids that open an assistant block."""
    prompt = _generation_addition(tokenizer, render_kwargs)
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    last = max((pos for pos, token in enumerate(ids) if token in specials), default=-1)
    header = str(tokenizer.decode(ids[: last + 1]))
    if not ids or ids[0] not in specials:
        return TurnFrame(prompt, header, prompt.strip(), None, None)
    opener = str(tokenizer.decode(ids[:2]))
    return TurnFrame(prompt, header, opener, ids[0], ids[1] if len(ids) > 1 else None)


class PrefixRenders:
    """Renders of the messages before each index, with and without the generation prompt, made once."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        messages: list[dict[str, Any]],
        render_kwargs: dict[str, Any],
    ) -> None:
        self._tokenizer = tokenizer
        self._messages = messages
        self.render_kwargs = render_kwargs
        self._made: dict[tuple[int, bool], str] = {}

    def message(self, index: int) -> dict[str, Any]:
        """Message ``index``, as passed to the template."""
        return self._messages[index]

    def before(self, index: int, add_generation_prompt: bool) -> str:
        """Render of the messages before ``index``."""
        key = (index, add_generation_prompt)
        if key not in self._made:
            self._made[key] = _render(
                self._tokenizer,
                self._messages[:index],
                add_generation_prompt,
                self.render_kwargs,
            )
        return self._made[key]


def _not_inside_added_token(
    tokenizer: PreTrainedTokenizerBase, render: TemplateRender, char: int
) -> int:
    """``char``, or the start of the added token it falls strictly inside.

    A turn opens with a whole added token: ``<t`` is shared by ``<think>`` and
    ``<tool_call>``, but a turn that opens with ``<tool_call>`` holds all of it.
    """
    at = render.token_at(char)
    if (
        at < len(render.ids)
        and render.starts[at] < char
        and render.ids[at] in tokenizer.added_tokens_decoder
    ):
        return render.starts[at]
    return char


def _content_end(text: str, message: dict[str, Any], cursor: int) -> int:
    """Where ``message``'s content ends in ``text`` from ``cursor``, else ``0``."""
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        return 0
    at = text.find(content.strip(), cursor)
    return 0 if at < 0 else at + len(content.strip())


def _after_content(prompt: str, content: str) -> str:
    """Text ``prompt`` holds after ``content``: the frame up to the turn."""
    at = prompt.rfind(content)
    return "" if at < 0 else prompt[at + len(content) :]


def _turn_start(
    tokenizer: PreTrainedTokenizerBase,
    render: TemplateRender,
    prefixes: PrefixRenders,
    roles: list[str],
    index: int,
    cursor: int,
    frame: TurnFrame,
) -> tuple[int, str]:
    """Where assistant message ``index``'s sampled text starts in ``render``, and the prompt's unmatched tail.

    After a non-assistant message, the turn starts where the render leaves the
    render of the messages before it with the generation prompt, when that is
    past the message's content. Otherwise (a turn right after another
    assistant turn, or a template that renders earlier messages differently
    once later ones follow) it starts after the next assistant opener from
    character ``cursor`` and the generation prompt it shares, or, with no
    opener, after the message's content and the frame the template writes
    after it.

    :raises ValueError: If the render does not open the turn with the
        generation prompt's header.
    """
    if roles[index - 1] != "assistant":
        prompt = prefixes.before(index, True)
        anchor = _content_end(render.text, prefixes.message(index - 1), cursor)
        opened = _common_length(prefixes.before(index, False), render.text)
        start = _not_inside_added_token(
            tokenizer, render, _common_length(prompt, render.text)
        )
        if start >= max(cursor, anchor) and start - opened >= len(frame.header):
            return start, prompt[start:]
        cursor = max(cursor, anchor)
        if not frame.opener and anchor > 0:
            # A positive anchor means the message has string content.
            tail = _after_content(
                prompt, prefixes.message(index - 1)["content"].strip()
            )
            start = _not_inside_added_token(
                tokenizer, render, anchor + _common_length(tail, render.text[anchor:])
            )
            return start, tail[start - anchor :]
    at = render.text.find(frame.opener, cursor) if frame.opener else -1
    if at >= 0:
        shared = _common_length(frame.prompt, render.text[at:])
        start = _not_inside_added_token(tokenizer, render, at + shared)
        if start - at >= len(frame.header):
            return start, frame.prompt[start - at :]
    raise UnplacedTurnError(UNPLACED_TURN)


def _turn_limit(
    tokenizer: PreTrainedTokenizerBase,
    render: TemplateRender,
    prefixes: PrefixRenders,
    render_messages: list[dict[str, Any]],
    index: int,
    next_index: int | None,
    start: int,
    frame: TurnFrame,
    chat_format: ChatFormat | None,
) -> tuple[int, bool]:
    """Index of the first id after ``start`` that cannot belong to assistant message ``index``, and whether the turn ends right before it.

    Before a tool result the chat format writes into the turn, that is the
    result's first id. Before other messages, it is the id after the leading
    terminator of the frame the template writes between an assistant turn and
    those messages (Qwen's ``<|im_end|>`` before the user block), else the
    next block that is not an assistant block. Before an assistant turn right
    after it, it is that turn's opener.
    """
    if next_index is None:
        return len(render.ids), False
    first = render.token_at(start)
    between = render_messages[index + 1 : next_index]
    if not between:
        after = _answer_end(render.text, render_messages[index], start)
        at = render.text.find(frame.opener, after) if frame.opener else -1
        if at < 0:
            raise UnplacedTurnError(UNPLACED_TURN)
        return render.token_at(at), False
    following = render.token_at(
        _common_length(prefixes.before(next_index, False), render.text)
    )
    if following <= first:
        # The template renders this turn differently once later messages follow.
        following = len(render.ids)
    in_turn = None if chat_format is None else chat_format.format_tool_results(between)
    if in_turn is not None:
        written = tokenizer.encode(in_turn, add_special_tokens=False)
        at = following - len(written)
        if at >= first and render.ids[at:following] == written:
            return at, False
    roles = tuple(message["role"] for message in between)
    parts = feedback_frame_parts(tokenizer, roles, prefixes.render_kwargs)
    if parts is not None:
        at = render.text.find(
            parts[0], _answer_end(render.text, render_messages[index], start)
        )
        if at >= 0 and render.token_at(at) < following:
            terminator = _terminator_length(tokenizer, parts[0])
            if terminator == 0:
                return render.token_at(at), False
            # The id holding the terminator's last character: Phi-3.5's
            # ``<|end|>`` also holds the newline after it.
            return render.token_at(at + terminator - 1) + 1, True
    blocks = (
        pos
        for pos in range(first, following)
        if render.ids[pos] == frame.block
        and render.ids[pos + 1 : pos + 2] != [frame.role]
    )
    return next(blocks, following), False


def _terminator_length(tokenizer: PreTrainedTokenizerBase, frame: str) -> int:
    """Characters of ``frame`` through its first added or special token, or ``0`` when it has none.

    Phi-3.5 adds ``<|end|>`` without flagging it special.
    """
    ids = tokenizer.encode(frame, add_special_tokens=False)
    marks = special_token_ids(tokenizer) | set(tokenizer.added_tokens_decoder)
    first = next((pos for pos, token in enumerate(ids) if token in marks), None)
    if first is None:
        return 0
    return len(str(tokenizer.decode(ids[: first + 1])))


def _answer_end(text: str, message: dict[str, Any], start: int) -> int:
    """Where ``message``'s visible answer ends in ``text`` from ``start``, else ``start``."""
    answer = _visible_answer(message)
    at = text.find(answer, start) if answer else -1
    return start if at < 0 else at + len(answer)


def _visible_answer(message: dict[str, Any]) -> str:
    """The text an assistant message shows after any reasoning block, stripped."""
    content = message.get("content")
    if not isinstance(content, str):
        return ""
    return content.rsplit(REASONING_END_MARKER, 1)[-1].strip()


def _assemble_from_template(
    tokenizer: PreTrainedTokenizerBase,
    render_messages: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    canonical: str,
    render_kwargs: dict[str, Any],
    chat_format: ChatFormat | None,
    tools: list[dict[str, Any]] | None,
    trained_indices: Collection[int],
) -> tuple[list[int], list[bool]]:
    """Ids of ``canonical``, the template render, with an assistant mask.

    Each assistant turn is found by comparing ``canonical`` with renders of the
    messages before it, nothing being written into the messages, so a template
    may rewrite a message's content (split out a ``<think>`` block, trim it,
    drop earlier reasoning) and the turn is still cut out. A turn runs through
    its last special token before the next message; one with none gains eos.
    The context before a turn and the turn keep the render's ids; where an id
    crosses a boundary, each side is encoded alone, as a rollout encodes its
    prompt and samples the turn. A generation prompt the completed turn does
    not continue is not spliced on, but its unmatched trailing whitespace
    stays in the context before a later turn. Text after the last turn is
    dropped.

    :param render_messages: Messages as passed to the template.
    :param messages: The same turns, still carrying ``tool_calls`` for the parse check.
    :param trained_indices: Indices of the assistant messages that train.
    :return: Token ids and the assistant mask.
    :raises ValueError: If the render does not open a turn with the generation
        prompt's header, or a turn does not show its message's answer.
    """
    render = _rendered(tokenizer, canonical)
    prefixes = PrefixRenders(tokenizer, render_messages, render_kwargs)
    specials = special_token_ids(tokenizer)
    frame = _turn_frame(tokenizer, render_kwargs, specials)
    roles = [message["role"] for message in render_messages]
    assistant_indices = [idx for idx, role in enumerate(roles) if role == "assistant"]
    ids: list[int] = []
    mask: list[bool] = []
    cursor = 0
    for position, (idx, next_idx) in enumerate(
        zip(assistant_indices, [*assistant_indices[1:], None], strict=True)
    ):
        start, prompt_tail = _turn_start(
            tokenizer, render, prefixes, roles, idx, cursor, frame
        )
        limit, closed = _turn_limit(
            tokenizer,
            render,
            prefixes,
            render_messages,
            idx,
            next_idx,
            start,
            frame,
            chat_format,
        )
        first = render.token_at(start)
        closing = (
            limit - 1
            if closed
            else next(
                (
                    pos
                    for pos in range(limit - 1, first - 1, -1)
                    if render.ids[pos] in specials
                ),
                None,
            )
        )
        last = limit - 1 if closing is None else closing
        end = render.ends[last] if last >= first else start
        if _visible_answer(render_messages[idx]) not in canonical[start:end]:
            raise UnplacedTurnError(UNPLACED_TURN)
        gap_ids = render.ids_between(tokenizer, cursor, start)
        # The completed turn does not continue this whitespace. The rollout
        # prompt still ends with it, so it stays outside the assistant span.
        if position > 0 and prompt_tail.isspace():
            gap_ids = tokenizer.encode(
                canonical[cursor:start] + prompt_tail, add_special_tokens=False
            )
        turn_ids = render.ids_between(tokenizer, start, end)
        if closing is None:
            turn_ids = [*turn_ids, tokenizer.eos_token_id]
        trained = idx in trained_indices
        if trained:
            _check_tool_calls_parse(messages[idx], turn_ids, tools, chat_format)
        ids += gap_ids + turn_ids
        mask += [False] * len(gap_ids) + [trained] * len(turn_ids)
        cursor = end
    return ids, mask


def _render(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, Any]],
    add_generation_prompt: bool,
    render_kwargs: dict[str, Any],
) -> str:
    """Chat-template text of ``messages``."""
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        **render_kwargs,
    )
    if not isinstance(rendered, str):
        msg = f"apply_chat_template returned {type(rendered).__name__}, expected str."
        raise TypeError(msg)
    return rendered


def _turn_scaffold(
    tokenizer: PreTrainedTokenizerBase,
    render_kwargs: dict[str, Any],
    special_ids: set[int],
) -> tuple[str, int]:
    """Generation prompt after :data:`TURN_OPENER`, and its text length through the last special token."""
    prompt = _render(tokenizer, [TURN_OPENER], True, render_kwargs)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    last_special = max(
        (pos for pos, token in enumerate(prompt_ids) if token in special_ids),
        default=-1,
    )
    return prompt, len(tokenizer.decode(prompt_ids[: last_special + 1]))


def _assistant_turn_ids(
    tokenizer: PreTrainedTokenizerBase,
    message: dict[str, Any],
    render_kwargs: dict[str, Any],
    special_ids: set[int],
    scaffold: tuple[str, int],
) -> list[int]:
    """Ids the model samples for one assistant turn, through its end-of-turn token.

    A re-render may drop text the generation prompt prefills after its last
    special token (a ``<think>`` opener), so the turn starts where the two renders part.
    """
    prompt, header = scaffold
    rendered = _render(tokenizer, [TURN_OPENER, message], False, render_kwargs)
    shared = len(os.path.commonprefix([prompt, rendered]))
    if shared < header:
        raise ValueError(UNPLACED_TURN)
    turn = tokenizer.encode(rendered[shared:], add_special_tokens=False)
    last_special = max(
        (pos for pos, token in enumerate(turn) if token in special_ids), default=None
    )
    if last_special is None:
        return [*turn, tokenizer.eos_token_id]
    return turn[: last_special + 1]


def _written_tool_calls(
    message: dict[str, Any], chat_format: ChatFormat | None
) -> dict[str, Any]:
    """``message`` with its tool calls in its content when the chat format writes them."""
    calls = [call.get("function", call) for call in message.get("tool_calls", [])]
    if chat_format is None or not calls:
        return message
    if not all(call.get("name") for call in calls):
        msg = f"An assistant tool call has no name: {calls}."
        raise ValueError(msg)
    text = chat_format.format_tool_calls(
        [
            {"name": call["name"], "arguments": call.get("arguments", {})}
            for call in calls
        ]
    )
    if text is None:
        return message
    content = message.get("content") or ""
    if not isinstance(content, str):
        msg = f"An assistant tool-call turn needs string content, got {type(content).__name__}."
        raise ValueError(msg)
    written = {key: value for key, value in message.items() if key != "tool_calls"}
    written["content"] = content + text
    return written


def _check_tool_calls_parse(
    message: dict[str, Any],
    turn: list[int],
    tools: list[dict[str, Any]] | None,
    chat_format: ChatFormat | None,
) -> None:
    """Raise unless the chat format parses ``turn`` to ``message``'s tool calls.

    A rollout runs only the calls the chat format parses, so SFT must not train
    calls it would miss. Names are compared; Gemma's parser keeps scalar
    arguments as text for the executor to coerce.
    """
    expected = [
        call.get("function", call).get("name") for call in message.get("tool_calls", [])
    ]
    if chat_format is None:
        if expected:
            msg = (
                "No chat format parses tool calls for this tokenizer, so a rollout "
                f"could not run the calls {expected} this turn trains."
            )
            raise ValueError(msg)
        return
    _, calls = chat_format.extract_tool_calls(turn, tools)
    parsed = [call.name for call in calls]
    if parsed != expected or any(
        call.status is not ToolCallParseStatus.OK for call in calls
    ):
        statuses = [call.status.value for call in calls]
        msg = (
            f"{type(chat_format).__name__} parses the rendered turn as calls "
            f"{parsed} ({statuses}), but the message calls {expected}."
        )
        raise ValueError(msg)


def _named_tool_results(
    assistant: dict[str, Any], messages: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """``messages`` with each unnamed tool result named after the call it answers.

    A result answers the call its ``tool_call_id`` names; one with neither an id
    nor a name answers the next call no other result answers. Ids are ignored
    when the assistant's calls have none.

    :param assistant: The assistant turn the results follow.
    :param messages: The messages between that turn and the next.
    :return: The messages with tool results named.
    :raises ValueError: If a ``tool_call_id`` matches none of the turn's calls.
    """
    calls = [
        (call.get("id"), call.get("function", call).get("name"))
        for call in assistant.get("tool_calls", [])
    ]
    name_by_id = {call_id: name for call_id, name in calls if call_id is not None}
    unanswered = list(calls)
    for message in messages:
        if message["role"] != "tool":
            continue
        call_id = message.get("tool_call_id") if name_by_id else None
        if call_id is not None and call_id not in name_by_id:
            msg = f"Tool result tool_call_id {call_id!r} is not one of the calls {list(name_by_id)}."
            raise ValueError(msg)
        answered = next(
            (
                call
                for call in unanswered
                if (
                    call[0] == call_id
                    if call_id is not None
                    else call[1] == message.get("name")
                )
            ),
            None,
        )
        if answered is not None:
            unanswered.remove(answered)
    next_names = iter(name for _, name in unanswered)
    named = []
    for message in messages:
        if message["role"] != "tool" or message.get("name"):
            named.append(message)
            continue
        call_id = message.get("tool_call_id") if name_by_id else None
        name = name_by_id[call_id] if call_id is not None else next(next_names, None)
        named.append({**message, "name": name})
    return named


def _feedback_frame(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, Any]],
    render_kwargs: dict[str, Any],
    chat_format: ChatFormat | None,
) -> str | tuple[str, ...]:
    """How a rollout frames ``messages``, the non-assistant messages between two assistant turns.

    :return: The text the chat format writes for them inside the turn, else
        the template's frame parts around them (:func:`feedback_frame_parts`).
    :raises ValueError: If a message has no string content, or neither the
        chat format nor the template frames these roles after an assistant turn.
    """
    for message in messages:
        if not isinstance(message.get("content"), str):
            msg = (
                f"A {message['role']} message between assistant turns needs string "
                f"content, got {type(message.get('content')).__name__}."
            )
            raise ValueError(msg)
    in_turn = None if chat_format is None else chat_format.format_tool_results(messages)
    if in_turn is not None:
        return in_turn
    roles = tuple(message["role"] for message in messages)
    parts = feedback_frame_parts(tokenizer, roles, render_kwargs)
    if parts is None:
        msg = f"The chat template cannot render a {roles} turn after an assistant turn."
        raise ValueError(msg)
    return parts


def _feedback_ids(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, Any]],
    render_kwargs: dict[str, Any],
    chat_format: ChatFormat | None,
) -> list[int]:
    """Ids of the non-assistant messages between two assistant turns."""
    frame = _feedback_frame(tokenizer, messages, render_kwargs, chat_format)
    if isinstance(frame, str):
        return tokenizer.encode(frame, add_special_tokens=False)
    text = frame[0] + "".join(
        message["content"] + part
        for message, part in zip(messages, frame[1:], strict=True)
    )
    return tokenizer.encode(text, add_special_tokens=False)

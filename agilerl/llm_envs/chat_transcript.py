# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Render chat transcripts with a chat template and mark the assistant tokens to train.

Shared by SFT and DPO rows written as OpenAI-style messages with optional ``tools``.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from agilerl.arena.traces_format import REASONING_KEYS
from agilerl.llm_envs.chat_formats import (
    ChatFormat,
    ToolCallParseStatus,
    frame_repeats_terminator,
)
from agilerl.utils.chat_template import (
    feedback_frame_parts,
    special_token_ids,
)

if TYPE_CHECKING:
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

TURN_OPENER = {"role": "user", "content": "."}


def chat_messages(value: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize one row's OpenAI-style ``messages`` for a chat template.

    Accepts a JSON string or a list. Drops ``None`` fields and decodes string
    tool-call arguments, which HF chat templates expect as a dict. A list is
    Arrow data, which fills every struct key at every depth, so ``None`` is also
    dropped inside its struct arguments; nulls in JSON text are real and kept.

    :param value: The row's ``messages`` value.
    :return: Messages ready for ``apply_chat_template``.
    :raises ValueError: If the row or a call's arguments are not valid JSON, or
        a call's arguments are not an object.
    """
    from_arrow = not isinstance(value, str)
    raw = (
        _parsed_json(value, "The messages column") if isinstance(value, str) else value
    )
    messages = []
    for raw_message in raw:
        message = {key: item for key, item in raw_message.items() if item is not None}
        if "tool_calls" in message:
            message["tool_calls"] = [
                _decoded_tool_call(call, from_arrow) for call in message["tool_calls"]
            ]
        messages.append(message)
    return messages


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
    :param key: Target key, or ``None`` to leave the message as it is.
    :return: The message, with any other reasoning key moved to ``key``.
    :raises ValueError: If the message sets two reasoning keys to different text.
    """
    if key is None or message.get("role") != "assistant":
        return message
    present = {name: message[name] for name in REASONING_KEYS if message.get(name)}
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


def _without_none_keys(value: Any) -> Any:  # noqa: ANN401 -- any nested Arrow value
    """``value`` without ``None`` dict entries at any depth; ``None`` list items are data."""
    if isinstance(value, dict):
        return {
            key: _without_none_keys(item)
            for key, item in value.items()
            if item is not None
        }
    if isinstance(value, list):
        return [_without_none_keys(item) for item in value]
    return value


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
        arguments = _without_none_keys(arguments)
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
    renders inside an assistant turn stay there. Messages after the last
    assistant turn are dropped. Assistant messages' reasoning is moved to the
    key the chat template reads, see :func:`template_reasoning_key`.
    Assistant turns before ``trained_from`` are context: masked out, and their
    tool calls are not checked against the chat format's parser.

    :param tokenizer: Tokenizer whose chat template renders the conversation.
    :param messages: Normalized messages, see :func:`chat_messages`.
    :param tools: Tool schemas passed to the template, or ``None``.
    :param chat_format: The model family's chat format, see
        :func:`~agilerl.llm_envs.chat_formats.detect_chat_format`.
    :param trained_from: Index of the first message whose assistant turns train.
    :return: Token ids and, per id, whether it is a trained assistant token.
    :raises ValueError: If the conversation has no prompt or no assistant
        turn, has back-to-back assistant turns, the template renders an
        assistant turn that does not extend the generation prompt, or the chat
        format does not parse a turn's rendered tool calls back to its calls.
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
    if assistant_indices[-1] < trained_from:
        msg = "An SFT conversation needs an assistant turn to train after its prompt."
        raise ValueError(msg)

    render_messages = _messages_for_render(messages, chat_format)
    prompt = _render(
        tokenizer, render_messages[: assistant_indices[0]], True, render_kwargs
    )
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    mask = [False] * len(ids)
    special_ids = special_token_ids(tokenizer)
    scaffold = _turn_scaffold(tokenizer, render_kwargs, special_ids)
    previous = None
    for idx in assistant_indices:
        if previous is not None:
            between = render_messages[previous + 1 : idx]
            if not between:
                msg = f"Messages {previous} and {idx} are back-to-back assistant turns."
                raise ValueError(msg)
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
        trained = idx >= trained_from
        if trained:
            _check_tool_calls_parse(messages[idx], turn, tools, chat_format)
        ids += turn
        mask += [trained] * len(turn)
        previous = idx
    kept = assistant_indices[-1] + 1
    canonical = _render(tokenizer, render_messages[:kept], False, render_kwargs)
    if _decodes_template(tokenizer, ids, canonical):
        return ids, mask
    return _assemble_from_template(
        tokenizer,
        render_messages[:kept],
        messages[:kept],
        canonical,
        render_kwargs,
        chat_format,
        tools,
        trained_from,
    )


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


def _with_payload_sentinel(message: dict[str, Any], sentinel: str) -> dict[str, Any]:
    """``message`` with ``sentinel`` prepended to each non-empty string the template may emit."""
    marked = dict(message)
    content = marked.get("content")
    if isinstance(content, str) and content:
        marked["content"] = sentinel + content
    calls = marked.get("tool_calls")
    if calls:
        marked["tool_calls"] = [_call_with_sentinel(call, sentinel) for call in calls]
    for key in REASONING_KEYS:
        value = marked.get(key)
        if isinstance(value, str) and value.strip():
            marked[key] = sentinel + value
    return marked


def _call_with_sentinel(call: dict[str, Any], sentinel: str) -> dict[str, Any]:
    """``call`` with ``sentinel`` prepended to its function name."""
    copied = dict(call)
    function = copied.get("function")
    if isinstance(function, dict):
        function = dict(function)
        name = function.get("name")
        if isinstance(name, str) and name:
            function["name"] = sentinel + name
        copied["function"] = function
        return copied
    name = copied.get("name")
    if isinstance(name, str) and name:
        copied["name"] = sentinel + name
    return copied


def _payload_at(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, Any]],
    render_kwargs: dict[str, Any],
    text: str,
    index: int,
) -> int | None:
    """Index of message ``index``'s first emitted field in ``text``, or ``None``."""
    sentinel = f"\ue000{index:04d}"
    marked = [
        _with_payload_sentinel(message, sentinel) if j == index else message
        for j, message in enumerate(messages)
    ]
    probed = _render(tokenizer, marked, False, render_kwargs)
    pos = probed.find(sentinel)
    if pos < 0 or not text.startswith(probed[:pos]):
        return None
    return pos


def _generation_addition(
    tokenizer: PreTrainedTokenizerBase, render_kwargs: dict[str, Any]
) -> str:
    """Text ``add_generation_prompt`` appends after one user message."""
    opener = [TURN_OPENER]
    prompted = _render(tokenizer, opener, True, render_kwargs)
    plain = _render(tokenizer, opener, False, render_kwargs)
    return prompted[len(os.path.commonprefix([prompted, plain])) :]


def _block_token(tokenizer: PreTrainedTokenizerBase, addition: str) -> str:
    """The first special token of the generation prompt, or ``""`` when it has none."""
    ids = tokenizer.encode(addition, add_special_tokens=False)
    specials = special_token_ids(tokenizer)
    for token_id in ids:
        if token_id in specials:
            text = tokenizer.decode([token_id])
            return text if isinstance(text, str) else ""
    return ""


def _addition_end(
    tokenizer: PreTrainedTokenizerBase, text: str, start: int, end: int, addition: str
) -> tuple[int, str] | None:
    """Where ``addition`` stops matching ``text``, and the unmatched tail.

    The tokenizer never splits an added token, so a match that stops inside one
    backs up to its start: ``<t`` is shared by ``<think>`` and ``<tool_call>``,
    but a turn that opens with ``<tool_call>`` holds the whole token.
    """
    limit = max(start, end)
    for length in range(len(addition), 0, -1):
        prefix = addition[:length]
        at = text.rfind(prefix, start, limit)
        if at < 0:
            continue
        shared = len(os.path.commonprefix([addition, text[at:]]))
        cut = _added_token_start(tokenizer, text, at + shared)
        return cut, addition[cut - at :]
    return None


def _added_token_start(tokenizer: PreTrainedTokenizerBase, text: str, pos: int) -> int:
    """Start of the added token in ``text`` that ``pos`` falls strictly inside, else ``pos``."""
    for token in tokenizer.added_tokens_decoder.values():
        content = token.content
        for at in range(max(0, pos - len(content) + 1), pos):
            if text.startswith(content, at):
                return at
    return pos


def _cut_kept_turn(
    tokenizer: PreTrainedTokenizerBase, turn_text: str
) -> tuple[str, bool]:
    """Turn text through its last special token, and whether the turn needs an eos token."""
    ids = tokenizer.encode(turn_text, add_special_tokens=False)
    specials = special_token_ids(tokenizer)
    last = max(
        (pos for pos, token in enumerate(ids) if token in specials), default=None
    )
    if last is None:
        return turn_text, True
    encoding = tokenizer(
        turn_text, add_special_tokens=False, return_offsets_mapping=True
    )
    if list(encoding["input_ids"]) != ids:
        msg = (
            "Cutting a turn out of the template render needs a fast tokenizer "
            "whose offsets match its encoding of the turn."
        )
        raise ValueError(msg)
    return turn_text[: encoding["offset_mapping"][last][1]], False


def _assemble_from_template(
    tokenizer: PreTrainedTokenizerBase,
    render_messages: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    canonical: str,
    render_kwargs: dict[str, Any],
    chat_format: ChatFormat | None,
    tools: list[dict[str, Any]] | None,
    trained_from: int,
) -> tuple[list[int], list[bool]]:
    """Ids of ``canonical``, the template render, with an assistant mask.

    A generation prompt that is not a prefix of the completed turn is not
    spliced onto the transcript. Trailing whitespace of that prompt stays in
    the gap before the next assistant turn.

    :param render_messages: Messages as passed to the template.
    :param messages: The same turns, still carrying ``tool_calls`` for the parse check.
    :param trained_from: Index of the first message whose assistant turns train.
    :return: Token ids and the assistant mask.
    """
    addition = _generation_addition(tokenizer, render_kwargs)
    block = _block_token(tokenizer, addition)
    assistant_indices = [
        idx
        for idx, message in enumerate(render_messages)
        if message["role"] == "assistant"
    ]
    payloads = [
        _payload_at(tokenizer, render_messages, render_kwargs, canonical, idx)
        for idx in range(len(render_messages))
    ]
    ids: list[int] = []
    mask: list[bool] = []
    cursor = 0

    def _assistant_span(index: int, payload: int) -> tuple[int, str, int]:
        """Start, unmatched generation-prompt tail, and end of this assistant turn."""
        located = _addition_end(tokenizer, canonical, cursor, payload, addition)
        if located is None:
            turn_start = payload
            prompt_tail = ""
        else:
            turn_start, prompt_tail = located
        next_payload = None
        for later in range(index + 1, len(payloads)):
            if payloads[later] is not None:
                next_payload = payloads[later]
                break
        limit = len(canonical) if next_payload is None else next_payload
        if block and next_payload is not None:
            at = canonical.rfind(block, turn_start, next_payload)
            if at >= 0:
                limit = at
        return turn_start, prompt_tail, limit

    for position, idx in enumerate(assistant_indices):
        payload = payloads[idx]
        if payload is None:
            msg = (
                "The chat template renders an assistant turn that does not extend "
                "its generation prompt, so the turn's sampled tokens cannot be cut out."
            )
            raise ValueError(msg)
        turn_start, prompt_tail, limit = _assistant_span(idx, payload)
        kept, needs_eos = _cut_kept_turn(tokenizer, canonical[turn_start:limit])
        gap = canonical[cursor:turn_start]
        # The completed turn does not continue this whitespace. The rollout
        # prompt still ends with it, so it stays outside the assistant span.
        if position > 0 and prompt_tail.isspace():
            gap += prompt_tail
        gap_ids = tokenizer.encode(gap, add_special_tokens=False)
        ids += gap_ids
        mask += [False] * len(gap_ids)
        turn_ids = tokenizer.encode(kept, add_special_tokens=False)
        if needs_eos:
            turn_ids.append(tokenizer.eos_token_id)
        trained = idx >= trained_from
        if trained:
            _check_tool_calls_parse(messages[idx], turn_ids, tools, chat_format)
        ids += turn_ids
        mask += [trained] * len(turn_ids)
        cursor = turn_start + len(kept)
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
        msg = (
            "The chat template renders an assistant turn that does not extend "
            "its generation prompt, so the turn's sampled tokens cannot be cut out."
        )
        raise ValueError(msg)
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


def _feedback_ids(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, Any]],
    render_kwargs: dict[str, Any],
    chat_format: ChatFormat | None,
) -> list[int]:
    """Ids of the non-assistant messages between two assistant turns."""
    for message in messages:
        if not isinstance(message.get("content"), str):
            msg = (
                f"A {message['role']} message between assistant turns needs string "
                f"content, got {type(message.get('content')).__name__}."
            )
            raise ValueError(msg)
    in_turn = None if chat_format is None else chat_format.format_tool_results(messages)
    if in_turn is not None:
        return tokenizer.encode(in_turn, add_special_tokens=False)
    roles = tuple(message["role"] for message in messages)
    parts = feedback_frame_parts(tokenizer, roles, render_kwargs)
    if parts is None:
        msg = f"The chat template cannot render a {roles} turn after an assistant turn."
        raise ValueError(msg)
    text = parts[0] + "".join(
        message["content"] + part
        for message, part in zip(messages, parts[1:], strict=True)
    )
    return tokenizer.encode(text, add_special_tokens=False)

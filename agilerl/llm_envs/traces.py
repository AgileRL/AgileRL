# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Render ``traces`` dataset rows for a rollout: the prompt the student continues, the reference the teacher reads.

The text matches what SFT trains on the same row: messages are normalized by
:func:`~agilerl.llm_envs.dataset_env.chat_messages`, reasoning moves to the key
the chat template reads, tool calls are written the way the chat format parses
them, and tool results are named after the call they answer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from agilerl.arena.models.traces import MESSAGES, TracesRow, row_form
from agilerl.arena.traces_format import REASONING_KEYS
from agilerl.llm_envs.chat_transcript import (
    _assistant_turn_ids,
    _messages_for_render,
    _named_tool_results,
    _render,
    _turn_scaffold,
    _with_reasoning_key,
    _without_none_keys,
    _written_tool_calls,
    chat_messages,
    template_reasoning_key,
)
from agilerl.llm_envs.dataset_env import sft_tools
from agilerl.utils.chat_template import special_token_ids

if TYPE_CHECKING:
    from collections.abc import Mapping

    from datasets import Dataset
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    from agilerl.llm_envs.chat_formats import ChatFormat

TRAINED_COLUMNS = ("prompt", "completion", "messages", "tools")
"""Columns of a traces row that training reads; identity and score columns are ignored."""


def _render_kwargs(
    tools: list[dict[str, Any]] | None, chat_template_kwargs: dict[str, Any] | None
) -> dict[str, Any]:
    kwargs = dict(chat_template_kwargs or {})
    if tools:
        kwargs["tools"] = tools
    return kwargs


def traces_prompt_text(
    tokenizer: PreTrainedTokenizerBase,
    prompt: str | list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    chat_format: ChatFormat | None,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> str:
    """Chat-template text of a row's ``prompt`` messages, ending in the generation prompt.

    :param tokenizer: The student's tokenizer.
    :param prompt: The row's ``prompt`` (JSON text or a list of messages).
    :param tools: The row's tool schemas, or ``None``.
    :param chat_format: The student's chat format, which writes past tool calls.
    :param chat_template_kwargs: Extra ``apply_chat_template`` kwargs.
    :return: The prompt text the student samples its turn after.
    :raises ValueError: If the prompt ends with an assistant turn.
    """
    reasoning_key = template_reasoning_key(tokenizer)
    messages = [
        _with_reasoning_key(message, reasoning_key) for message in chat_messages(prompt)
    ]
    if not messages or messages[-1]["role"] == "assistant":
        msg = "A traces prompt needs messages and must not end with an assistant turn."
        raise ValueError(msg)
    rendered = _messages_for_render(messages, chat_format)
    last = max(
        (idx for idx, message in enumerate(messages) if message["role"] == "assistant"),
        default=None,
    )
    if last is not None:
        rendered[last + 1 :] = _named_tool_results(messages[last], rendered[last + 1 :])
    return _render(
        tokenizer, rendered, True, _render_kwargs(tools, chat_template_kwargs)
    )


def traces_reference_text(
    tokenizer: PreTrainedTokenizerBase,
    completion: str | list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    chat_format: ChatFormat | None,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> str:
    """The row's ``completion`` turn as text: reasoning, content and tool calls.

    The turn is rendered as the student would sample it, so a self-distilling
    teacher reads the reference in its own format. A template that drops
    reasoning gets the reasoning ahead of the turn.

    :param tokenizer: The student's tokenizer.
    :param completion: The row's ``completion`` (JSON text or a one-message list).
    :param tools: The row's tool schemas, or ``None``.
    :param chat_format: The student's chat format, which writes tool calls.
    :param chat_template_kwargs: Extra ``apply_chat_template`` kwargs.
    :return: The reference text for ``teacher.template``'s ``{reference}``.
    :raises ValueError: If the completion is not one assistant message.
    """
    messages = chat_messages(completion)
    if len(messages) != 1 or messages[0]["role"] != "assistant":
        msg = "A traces completion is exactly one assistant message."
        raise ValueError(msg)
    reasoning_key = template_reasoning_key(tokenizer)
    message = _with_reasoning_key(messages[0], reasoning_key)
    kwargs = _render_kwargs(tools, chat_template_kwargs)
    special_ids = special_token_ids(tokenizer)
    turn_ids = _assistant_turn_ids(
        tokenizer,
        _written_tool_calls(message, chat_format),
        kwargs,
        special_ids,
        _turn_scaffold(tokenizer, kwargs, special_ids),
    )
    # Keep special tokens inside the turn (tool-call and channel markers); drop its end-of-turn token.
    text = tokenizer.decode(turn_ids[:-1], skip_special_tokens=False)
    if not isinstance(text, str):
        msg = "tokenizer.decode must return a string"
        raise TypeError(msg)
    reasoning = next((message[key] for key in REASONING_KEYS if message.get(key)), None)
    if reasoning_key is None and reasoning:
        return f"{reasoning}\n\n{text.strip()}"
    return text.strip()


def check_traces_split(dataset: Dataset, label: str, rollout: bool) -> None:
    """Raise unless every row of a loaded split follows the traces spec.

    Only the columns training reads are checked; identity, score and extra
    columns are left to the upload validator.

    :param dataset: The loaded split.
    :param label: Split name for the error.
    :param rollout: Whether a rollout reads it, which needs ``prompt`` /
        ``completion`` rows; SFT also takes ``messages`` rows.
    :raises ValueError: Naming the first bad rows, numbered from 1.
    """
    model = row_form(dataset.column_names)
    if rollout and model is not TracesRow:
        msg = (
            f"A traces rollout reads 'prompt' and 'completion' columns; the {label} "
            f"has {sorted(dataset.column_names)}."
        )
        raise ValueError(msg)
    errors: list[str] = []
    for number, row in enumerate(dataset, start=1):
        trained = {
            column: _without_none_keys(row[column])
            for column in TRAINED_COLUMNS
            if column in row
        }
        try:
            model.model_validate(trained)
        except ValidationError as error:
            first = error.errors(include_url=False)[0]
            where = ".".join(str(part) for part in first["loc"])
            row_at = f"row {number}, {where}" if where else f"row {number}"
            errors.append(f"{row_at}: {first['msg']}")
            if len(errors) == 5:
                break
    if errors:
        msg = f"The {label} breaks the traces spec:\n  " + "\n  ".join(errors)
        raise ValueError(msg)


def traces_sft_split(dataset: Dataset, label: str) -> Dataset:
    """A checked SFT split; prompt/completion rows drop any ``messages`` column.

    :param dataset: The loaded split.
    :param label: Split name for errors.
    :return: The split SFT trains on.
    :raises ValueError: If a row breaks the traces spec.
    """
    check_traces_split(dataset, label, rollout=False)
    if row_form(dataset.column_names) is TracesRow and MESSAGES in dataset.column_names:
        return dataset.remove_columns(MESSAGES)
    return dataset


def traces_rollout_split(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    chat_format: ChatFormat | None,
    chat_template_kwargs: dict[str, Any] | None,
    reference_column: str | None,
) -> Dataset:
    """A traces split ready for a single-turn rollout over its rows.

    Adds an empty ``answer`` column when the rows have none, and with
    ``reference_column`` writes each row's rendered ``completion`` there, which
    the rollout copies into reset info for the teacher.

    :param dataset: The loaded, checked split.
    :param tokenizer: The student's tokenizer.
    :param chat_format: The student's chat format.
    :param chat_template_kwargs: Extra ``apply_chat_template`` kwargs.
    :param reference_column: Column ``teacher.context_key`` reads, or ``None``.
    :return: The split with those columns.
    """
    if "answer" not in dataset.column_names:
        dataset = dataset.add_column("answer", [None] * len(dataset))
    if reference_column is None:
        return dataset

    def with_reference(row: Mapping[str, Any]) -> dict[str, str]:
        return {
            reference_column: traces_reference_text(
                tokenizer,
                row["completion"],
                sft_tools(dict(row)),
                chat_format,
                chat_template_kwargs,
            )
        }

    return dataset.map(with_reference)

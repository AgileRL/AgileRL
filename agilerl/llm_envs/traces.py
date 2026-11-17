# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Render ``traces`` dataset rows for a rollout: the prompt the student continues, the reference the teacher reads.

A segment expands, in memory, into one prompt and completion per target
assistant turn. The text matches what SFT trains on the same turn: messages are normalized by
:func:`~agilerl.llm_envs.dataset_env.chat_messages`, reasoning moves to the key
the chat template reads, tool calls are written the way the chat format parses
them, and tool results are named after the call they answer.
"""

from __future__ import annotations

import json
import warnings
from functools import cache
from typing import TYPE_CHECKING, Any

from datasets import Dataset
from pydantic import ValidationError

from agilerl.arena.models.traces import TracesSegmentRow
from agilerl.arena.traces_format import (
    CALL_ID,
    CALL_IDS,
    COMPLETION,
    COMPLETION_LOGPROBS,
    COMPLETION_TOKEN_IDS,
    FINISH_REASON,
    FINISH_REASONS,
    MESSAGES,
    PROMPT,
    REASONING_KEYS,
    SCORES,
    TOOLS,
    input_shape,
)
from agilerl.llm_envs.chat_formats import ChatFormat
from agilerl.llm_envs.chat_transcript import (
    assistant_turn_text,
    chat_messages,
    generation_prompt_addition,
    json_cell,
    render_chat_prompt,
    template_reasoning_key,
    without_none_keys,
)
from agilerl.llm_envs.dataset_env import (
    CHAT_FORMAT,
    SKIPPED_TURNS,
    encode_sft_row,
    sft_tools,
)
from agilerl.llm_envs.reasoning import REASONING_END_MARKER, REASONING_START_MARKER
from agilerl.utils.chat_template import inject_chat_template_kwargs

if TYPE_CHECKING:
    from collections.abc import Mapping

    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

TRAINED_COLUMNS = (MESSAGES, TOOLS, CALL_IDS, FINISH_REASONS)
"""Columns of a traces row that training reads; identity and score columns are ignored."""

SEGMENT_COLUMNS = (
    MESSAGES,
    CALL_IDS,
    FINISH_REASONS,
    SCORES,
    PROMPT,
    COMPLETION,
    COMPLETION_TOKEN_IDS,
    COMPLETION_LOGPROBS,
)
"""Columns a segment's expansion rewrites per turn or drops, rather than copies."""


def traces_reference_text(
    tokenizer: PreTrainedTokenizerBase,
    completion: str | list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    chat_format: ChatFormat | None,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> str:
    """The row's ``completion`` turn as text: reasoning, content and tool calls.

    The turn is rendered as the student would sample it, so a self-distilling
    teacher reads the reference in its own format, see
    :func:`~agilerl.llm_envs.chat_transcript.assistant_turn_text`. A template
    that reads no reasoning key gets no reasoning, as SFT trains it.

    :param tokenizer: The student's tokenizer.
    :param completion: The row's ``completion`` (JSON text or a one-message list).
    :param tools: The row's tool schemas, or ``None``.
    :param chat_format: The student's chat format, which writes tool calls.
    :param chat_template_kwargs: Extra ``apply_chat_template`` kwargs.
    :return: The reference text for ``teacher.template``'s ``{reference}``.
    :raises ValueError: If the completion is not one assistant message, or has
        reasoning the generation prompt closes the thinking block before.
    """
    messages = chat_messages(completion)
    if len(messages) != 1 or messages[0]["role"] != "assistant":
        msg = "A traces completion is exactly one assistant message."
        raise ValueError(msg)
    addition = generation_prompt_addition(tokenizer, tools, chat_template_kwargs)
    trains_reasoning = template_reasoning_key(tokenizer) is not None and any(
        messages[0].get(key) for key in REASONING_KEYS
    )
    if _closes_thinking(addition) and trains_reasoning:
        raise ValueError(_closed_thinking_message(addition))
    return assistant_turn_text(
        tokenizer, messages[0], tools, chat_format, chat_template_kwargs
    )


def _closes_thinking(addition: str) -> bool:
    """Whether a generation prompt's ``addition`` opens and then closes a thinking block."""
    opener = addition.find(REASONING_START_MARKER)
    return opener >= 0 and REASONING_END_MARKER in addition[opener:]


def _closed_thinking_message(addition: str) -> str:
    """Error text for a completion with reasoning after a generation prompt that closes thinking."""
    return (
        f"The completion has reasoning, but the generation prompt {addition!r} "
        "closes the thinking block, so a rollout would sample without it. Set "
        "environment.chat_template_kwargs enable_thinking: true (or the "
        "template's thinking switch)."
    )


def _trains_reasoning(row: Mapping[str, Any]) -> bool:
    """Whether a trained assistant turn of ``row`` has reasoning; context turns do not count."""
    messages, targets = _target_turns(row)
    return any(messages[at].get(key) for at in targets for key in REASONING_KEYS)


def _sample_indices(count: int) -> range | list[int]:
    """Up to 16 row indices evenly spaced from the first row to the last, so a split's tail is checked."""
    sample = 16
    if count <= sample:
        return range(count)
    return sorted({at * (count - 1) // (sample - 1) for at in range(sample)})


def _per_turn(row: Mapping[str, Any], column: str) -> list[Any] | None:
    """A row's per-turn list column, decoded from JSON text when stored as text."""
    value = row.get(column)
    return None if value is None else json_cell(value, column)


def _skipped_indices(
    messages: list[dict[str, Any]],
    call_ids: list[str | None] | None,
    finish_reasons: list[str | None] | None,
) -> list[int]:
    """Indices of a segment's assistant messages that are not training targets.

    A turn with a null ``call_ids`` entry is context (inserted, or inherited
    from an earlier segment), and a turn cut off at the token limit
    (``finish_reasons`` entry ``length``) is skipped.

    :param messages: The segment's messages.
    :param call_ids: One call id per assistant message, or ``None`` (all targets).
    :param finish_reasons: One finish reason per assistant message, or ``None``.
    :return: Message indices of the skipped assistant turns.
    """
    turns = [
        at for at, message in enumerate(messages) if message["role"] == "assistant"
    ]
    return [
        at
        for position, at in enumerate(turns)
        if (call_ids is not None and call_ids[position] is None)
        or (finish_reasons is not None and finish_reasons[position] == "length")
    ]


def traces_skipped_turns(row: Mapping[str, Any]) -> list[int]:
    """Indices of a segment row's assistant messages that SFT does not train.

    :param row: A segment row; an expanded turn has none.
    :return: Message indices of the skipped assistant turns.
    """
    if row.get(MESSAGES) is None:
        return []
    return _skipped_indices(
        chat_messages(row[MESSAGES]),
        _per_turn(row, CALL_IDS),
        _per_turn(row, FINISH_REASONS),
    )


def _target_turns(row: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[int]]:
    """A row's normalized messages and the indices of its target assistant messages.

    :param row: A segment row, or an expanded turn with ``prompt`` and ``completion``.
    """
    if row.get(MESSAGES) is None:
        prompt = chat_messages(row[PROMPT])
        return [*prompt, *chat_messages(row[COMPLETION])], [len(prompt)]
    messages = chat_messages(row[MESSAGES])
    skipped = set(
        _skipped_indices(
            messages, _per_turn(row, CALL_IDS), _per_turn(row, FINISH_REASONS)
        )
    )
    return messages, [
        at
        for at, message in enumerate(messages)
        if message["role"] == "assistant" and at not in skipped
    ]


def _run_starts(mask: list[bool]) -> list[int]:
    """Start of each run of ``True`` in ``mask``."""
    return [
        at for at, kept in enumerate(mask) if kept and (at == 0 or not mask[at - 1])
    ]


def _render_row(
    tokenizer: PreTrainedTokenizerBase,
    row: Mapping[str, Any],
    chat_format: ChatFormat | None,
    chat_template_kwargs: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[int], list[str], tuple[str, str] | None]:
    """A row's messages, SFT ids, rollout prompt texts, and the first target whose prompt is not a prefix of its SFT context.

    :return: The normalized messages, the SFT token ids, each checked target's
        prompt text, and that prompt's text with the SFT context's text, or
        ``None`` when every target matches.
    """
    tools = sft_tools(dict(row))
    messages, targets = _target_turns(row)
    ids, mask = encode_sft_row(
        tokenizer,
        dict(row),
        COMPLETION,
        chat_format,
        True,
        [
            at
            for at, message in enumerate(messages)
            if message["role"] == "assistant" and at not in targets
        ],
    )
    texts: list[str] = []
    # A turn split into several runs misaligns the pairs, which reads as a mismatch: the safe answer.
    for at, start in zip(targets, _run_starts(mask), strict=False):
        prompt_text = render_chat_prompt(
            tokenizer, messages[:at], tools, chat_format, chat_template_kwargs
        )
        texts.append(prompt_text)
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        if len(prompt_ids) > start or ids[: len(prompt_ids)] != prompt_ids:
            return (
                messages,
                ids,
                texts,
                (prompt_text, str(tokenizer.decode(ids[:start]))),
            )
    return messages, ids, texts, None


def check_traces_render(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    chat_format: ChatFormat | None,
    chat_template_kwargs: dict[str, Any],
) -> None:
    """Raise unless a sample of rows renders the way a rollout sees them.

    The generation prompt must not close the thinking block when a completion
    has reasoning, no tool call's arguments may appear as an escaped JSON
    string, and each trained turn's prompt render, which a rollout samples
    after, must be a token prefix of the context SFT trains that turn after.
    Applies ``chat_template_kwargs`` to ``tokenizer`` as the SFT env does.

    :param dataset: A checked traces split.
    :param tokenizer: The student's tokenizer.
    :param chat_format: The student's chat format.
    :param chat_template_kwargs: The manifest's ``chat_template_kwargs``.
    :raises ValueError: Naming the first row, numbered from 1, that fails.
    """
    tokenizer = inject_chat_template_kwargs(tokenizer, chat_template_kwargs)
    _check_thinking_stays_open(dataset, tokenizer)
    for index in _sample_indices(len(dataset)):
        messages, ids, texts, mismatch = _render_row(
            tokenizer, dataset[index], chat_format, chat_template_kwargs
        )
        if mismatch is not None:
            prompt_text, context = mismatch
            msg = (
                f"Row {index + 1}: the prompt a rollout samples after is not a "
                "prefix of the context SFT trains the completion after: "
                f"{prompt_text[-200:]!r} vs {context[-200:]!r}."
            )
            raise ValueError(msg)
        _check_arguments_not_escaped(
            messages, [*texts, str(tokenizer.decode(ids))], index + 1
        )


def _check_thinking_stays_open(
    dataset: Dataset, tokenizer: PreTrainedTokenizerBase
) -> None:
    """Raise if the generation prompt closes the thinking block and a sampled row trains reasoning.

    :param tokenizer: The student's tokenizer, with ``chat_template_kwargs`` applied.
    """
    addition = generation_prompt_addition(tokenizer, None)
    if not _closes_thinking(addition):
        return
    for index in _sample_indices(len(dataset)):
        if _trains_reasoning(dataset[index]):
            msg = f"Row {index + 1}: {_closed_thinking_message(addition)}"
            raise ValueError(msg)


def template_rewrites_turns(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    chat_format: ChatFormat | None,
    chat_template_kwargs: dict[str, Any],
) -> bool:
    """Whether the template renders a sampled segment's earlier turns differently once later ones follow.

    gpt-oss drops the analysis of earlier turns once a final message follows,
    so one render of the whole segment trains a turn after a context no
    rollout samples it after.

    :param dataset: A checked split of segment rows.
    :param tokenizer: The student's tokenizer.
    :param chat_format: The student's chat format.
    :param chat_template_kwargs: The manifest's ``chat_template_kwargs``.
    :return: Whether a trained turn's prompt render is not a prefix of the segment's render.
    """
    tokenizer = inject_chat_template_kwargs(tokenizer, chat_template_kwargs)
    return any(
        _render_row(tokenizer, dataset[index], chat_format, chat_template_kwargs)[3]
        is not None
        for index in _sample_indices(len(dataset))
    )


def _check_arguments_not_escaped(
    messages: list[dict[str, Any]], texts: list[str], number: int
) -> None:
    """Raise if ``texts`` hold a tool call's arguments as an escaped JSON string.

    :param messages: Normalized messages, arguments decoded to objects.
    :param texts: Renders of those messages.
    :param number: Row number for the error.
    """
    for message in messages:
        for call in message.get("tool_calls", []):
            function = call.get("function", call)
            arguments = function.get("arguments") or {}
            spellings = {
                json.dumps(arguments, separators=separators, ensure_ascii=ascii_only)
                for separators in ((", ", ": "), (",", ":"))
                for ascii_only in (True, False)
            }
            for spelling in spellings:
                escaped = json.dumps(spelling)[1:-1]
                if escaped != spelling and any(escaped in text for text in texts):
                    msg = (
                        f"Row {number}: the chat template writes the arguments of "
                        f"tool call {function.get('name')!r} as an escaped JSON "
                        f"string ({escaped[:120]!r}); it expects them as an object."
                    )
                    raise ValueError(msg)


def check_traces_split(dataset: Dataset, label: str) -> Dataset:
    """The rows of a loaded split that are traces segments; warns with the count left out.

    Only the columns training reads are checked; identity, score and extra
    columns are left to the upload validator. A prompt/completion pair is
    left out: the platform converts pairs to segments on import.

    :param dataset: The loaded split.
    :param label: Split name for messages.
    :return: The split without the rows that break the spec.
    :raises ValueError: If the split has rows and none is a traces segment,
        naming the first bad row, numbered from 1.
    """
    kept: list[int] = []
    first_error = ""
    for number, row in enumerate(dataset, start=1):
        error = _segment_error(row, number)
        if error is None:
            kept.append(number - 1)
        elif not first_error:
            first_error = error
    if len(kept) == len(dataset):
        return dataset
    if not kept:
        msg = f"No row of the {label} is a traces segment. First error: {first_error}"
        raise ValueError(msg)
    warnings.warn(
        f"{len(dataset) - len(kept)} of {len(dataset)} rows of the {label} break "
        f"the traces spec and were dropped. First error: {first_error}",
        stacklevel=2,
    )
    return dataset.select(kept)


def _segment_error(row: Mapping[str, Any], number: int) -> str | None:
    """Why row ``number`` of a split is not a traces segment, or ``None`` if it is one."""
    try:
        shape = input_shape(c for c, value in row.items() if value is not None)
    except ValueError:
        return f"row {number}: has no messages column."
    if shape == "prompt_completion":
        return (
            f"row {number}: a prompt/completion pair, not a messages segment. "
            "Upload it as an Arena traces dataset, which converts pairs to "
            "segments on import."
        )
    trained = {
        column: without_none_keys(row[column])
        for column in TRAINED_COLUMNS
        if row.get(column) is not None
    }
    try:
        TracesSegmentRow.model_validate(trained)
    except ValidationError as error:
        first = error.errors(include_url=False)[0]
        where = ".".join(str(part) for part in first["loc"])
        row_at = f"row {number}, {where}" if where else f"row {number}"
        return f"{row_at}: {first['msg']}"
    return None


def expand_segments(dataset: Dataset) -> Dataset:
    """A checked split's segments as one row per target assistant turn.

    Assistant message ``k`` gives ``prompt = messages[:k]`` and ``completion =
    [messages[k]]``, as JSON text. Its ``call_ids`` and ``finish_reasons``
    entries become ``call_id`` and ``finish_reason``, and it keeps the
    segment's whole-row scores and its own turn's. Other columns are copied.
    Context turns (null ``call_ids``) and turns cut off at the token limit
    (``finish_reasons`` ``length``) give no row.

    :param dataset: A split checked by :func:`check_traces_split`.
    :return: One row per target turn.
    :raises ValueError: If no turn of the split is a target.
    """
    copied = [
        column
        for column in dataset.column_names
        if column not in (*SEGMENT_COLUMNS, CALL_ID, FINISH_REASON)
    ]
    has_scores = SCORES in dataset.column_names
    rows: list[dict[str, Any]] = []
    for row in dataset:
        messages = json_cell(row[MESSAGES], MESSAGES)
        turns = [
            at for at, message in enumerate(messages) if message["role"] == "assistant"
        ]
        call_ids = _per_turn(row, CALL_IDS)
        finish_reasons = _per_turn(row, FINISH_REASONS)
        skipped = set(_skipped_indices(messages, call_ids, finish_reasons))
        scores = _per_turn(row, SCORES)
        for position, at in enumerate(turns):
            if at in skipped:
                continue
            expanded = {column: row[column] for column in copied}
            expanded[PROMPT] = json.dumps(messages[:at])
            expanded[COMPLETION] = json.dumps([messages[at]])
            expanded[CALL_ID] = None if call_ids is None else call_ids[position]
            expanded[FINISH_REASON] = (
                None if finish_reasons is None else finish_reasons[position]
            )
            if has_scores:
                expanded[SCORES] = (
                    None
                    if scores is None
                    else [
                        score
                        for score in scores
                        if score.get("turn") in (None, position)
                    ]
                )
            rows.append(expanded)
    if not rows:
        raise ValueError(_no_target_message("traces split"))
    return Dataset.from_list(rows)


def traces_sft_split(
    dataset: Dataset,
    label: str,
    tokenizer: PreTrainedTokenizerBase,
    chat_format: ChatFormat | None,
    chat_template_kwargs: dict[str, Any],
) -> Dataset:
    """A checked SFT split whose render matches a rollout's.

    Each segment trains in one render, with its context and ``length`` turns
    listed in a :data:`~agilerl.llm_envs.dataset_env.SKIPPED_TURNS` column,
    unless the template rewrites earlier turns once later ones follow: then the split is expanded, see
    :func:`expand_segments`, warned once per process. Segments with no target
    turn are dropped. A :data:`~agilerl.llm_envs.dataset_env.CHAT_FORMAT`
    column names ``chat_format``, so SFT writes tool calls as the rollout does.

    :param dataset: The loaded split.
    :param label: Split name for errors.
    :param tokenizer: The student's tokenizer.
    :param chat_format: The student's chat format.
    :param chat_template_kwargs: The manifest's ``chat_template_kwargs``.
    :return: The split SFT trains on.
    :raises ValueError: If no row follows the traces spec, or a row renders
        differently from a rollout.
    """
    dataset = check_traces_split(dataset, label)
    dataset = dataset.remove_columns(
        [column for column in (PROMPT, COMPLETION) if column in dataset.column_names]
    )
    dataset = _with_trained_turns(dataset, label)
    _check_thinking_stays_open(
        dataset, inject_chat_template_kwargs(tokenizer, chat_template_kwargs)
    )
    if template_rewrites_turns(dataset, tokenizer, chat_format, chat_template_kwargs):
        _warn_per_turn_fallback()
        dataset = expand_segments(dataset)
    else:
        dataset = dataset.map(lambda row: {SKIPPED_TURNS: traces_skipped_turns(row)})
    if chat_format is not None:
        dataset = dataset.map(lambda _row: {CHAT_FORMAT: chat_format.NAME})
    check_traces_render(dataset, tokenizer, chat_format, chat_template_kwargs)
    return dataset


def _with_trained_turns(dataset: Dataset, label: str) -> Dataset:
    """Segments with an assistant turn to train; warns how many are dropped.

    :raises ValueError: If no segment is left.
    """
    kept = dataset.filter(lambda row: bool(_target_turns(row)[1]))
    if len(kept) == 0:
        raise ValueError(_no_target_message(label))
    if len(kept) < len(dataset):
        warnings.warn(
            f"{len(dataset) - len(kept)} segments of the {label} have no turn to "
            "train (every turn is context or finish_reasons length) and were "
            "dropped.",
            stacklevel=3,
        )
    return kept


def _no_target_message(label: str) -> str:
    """Error text for a split with no target assistant turn."""
    return (
        f"No assistant turn of the {label} is a target: every turn is context "
        "(null call_ids) or cut off at the token limit (finish_reasons length)."
    )


@cache
def _warn_per_turn_fallback() -> None:
    """Warn, once, that segments train per turn because the template rewrites earlier turns."""
    warnings.warn(
        "The chat template renders earlier assistant turns differently once "
        "later ones follow (gpt-oss drops their analysis), so each traces "
        "segment is expanded and trained one assistant turn per row.",
        stacklevel=4,
    )


def traces_rollout_split(
    dataset: Dataset,
    label: str,
    tokenizer: PreTrainedTokenizerBase,
    chat_format: ChatFormat | None,
    chat_template_kwargs: dict[str, Any],
    reference_column: str | None,
) -> Dataset:
    """A loaded traces split as single-turn rollout rows: one per target assistant turn.

    Checks the split (:func:`check_traces_split`), expands its segments
    (:func:`expand_segments`) and checks their render
    (:func:`check_traces_render`). Adds an empty ``answer`` column when the
    rows have none, and with ``reference_column`` writes each turn's rendered
    ``completion`` there, which the rollout copies into reset info for the
    teacher.

    :param dataset: The loaded split of segment rows.
    :param label: Split name for errors.
    :param tokenizer: The student's tokenizer.
    :param chat_format: The student's chat format.
    :param chat_template_kwargs: The manifest's ``chat_template_kwargs``.
    :param reference_column: Column ``teacher.context_key`` reads, or ``None``.
    :return: The expanded split with those columns.
    :raises ValueError: If a row breaks the traces spec, no turn is a target,
        or a turn renders differently from SFT.
    """
    dataset = check_traces_split(dataset, label)
    dataset = expand_segments(dataset)
    check_traces_render(dataset, tokenizer, chat_format, chat_template_kwargs)
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

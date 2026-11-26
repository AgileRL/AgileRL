# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Traces rows rendered for a rollout and an OPSD teacher, and trained by SFT on the completion only."""

from __future__ import annotations

import copy
import json
import warnings
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

pytest.importorskip("datasets", reason="LLM dependencies not installed")

import huggingface_hub.constants as hub_constants
from datasets import Dataset as HFDataset
from huggingface_hub import snapshot_download
from huggingface_hub.errors import (
    GatedRepoError,
    HfHubHTTPError,
    LocalEntryNotFoundError,
    OfflineModeIsEnabled,
)
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from agilerl.llm_envs import DatasetEnv
from agilerl.llm_envs.chat_formats import (
    HarmonyFormat,
    detect_chat_format,
)
from agilerl.llm_envs.chat_transcript import _warn_reasoning_dropped, render_chat_prompt
from agilerl.llm_envs.dataset_env import (
    chat_messages,
    encode_chat_transcript,
    encode_sft_row,
)
from agilerl.llm_envs.traces import (
    _warn_per_turn_fallback,
    check_traces_render,
    check_traces_split,
    expand_segments,
    template_rewrites_turns,
    traces_reference_text,
    traces_rollout_split,
    traces_sft_split,
    traces_skipped_turns,
)
from tests import TINY_LLM_FIXTURE_PATH
from tests.helpers.transcripts import assistant_runs

LOOKUP_TOOL = {
    "type": "function",
    "function": {
        "name": "lookup_user",
        "description": "Find a user",
        "parameters": {
            "type": "object",
            "properties": {"email": {"type": "string"}},
            "required": ["email"],
        },
    },
}

# ChatML that renders an assistant's reasoning_content in <think> tags.
REASONING_TEMPLATE = (
    "{% for message in messages %}<|im_start|>{{ message.role }}\n"
    "{% if message.role == 'assistant' and message.reasoning_content %}"
    "<think>\n{{ message.reasoning_content }}\n</think>\n\n{% endif %}"
    "{{ message.content or '' }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)

# Qwen3.5 layout: the generation prompt closes the think block unless enable_thinking.
THINKING_SWITCH_TEMPLATE = (
    "{% for message in messages %}<|im_start|>{{ message.role }}\n"
    "{% if message.role == 'assistant' %}"
    "<think>\n{{ message.reasoning_content or '' }}\n</think>\n\n{% endif %}"
    "{{ message.content or '' }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n<think>\n"
    "{% if not enable_thinking %}{{ '\\n' }}</think>\n\n{% endif %}{% endif %}"
)

# gpt-oss layout: an assistant turn's analysis renders only when no later turn is final.
DROPS_PAST_ANALYSIS_TEMPLATE = (
    "{% set ns = namespace(final=-1) %}"
    "{% for m in messages %}{% if m.role == 'assistant' and m.content %}"
    "{% set ns.final = loop.index0 %}{% endif %}{% endfor %}"
    "{% for m in messages %}"
    "{% if m.role == 'assistant' %}<|start|>assistant"
    "{% if m.thinking and loop.index0 >= ns.final %}"
    "<|channel|>analysis<|message|>{{ m.thinking }}<|end|><|start|>assistant{% endif %}"
    "{% if m.tool_calls %} to=functions.{{ m.tool_calls[0].function.name }}"
    "<|channel|>commentary json<|message|>"
    "{{ m.tool_calls[0].function.arguments | tojson }}<|call|>"
    "{% else %}<|channel|>final<|message|>{{ m.content }}"
    "{% if loop.last %}<|return|>{% else %}<|end|>{% endif %}{% endif %}"
    "{% elif m.role == 'tool' %}<|start|>functions.lookup_user to=assistant"
    "<|channel|>commentary<|message|>{{ m.content }}<|end|>"
    "{% else %}<|start|>{{ m.role }}<|message|>{{ m.content }}<|end|>{% endif %}"
    "{% endfor %}{% if add_generation_prompt %}<|start|>assistant{% endif %}"
)

HARMONY_SPECIALS = [
    "<|start|>",
    "<|message|>",
    "<|channel|>",
    "<|call|>",
    "<|end|>",
    "<|return|>",
]


def template_tokenizer(template: str, specials: list[str] | None = None) -> Any:
    """Tiny tokenizer rendering with ``template``, with ``specials`` as special tokens."""
    tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
    if specials:
        tokenizer.add_tokens(specials, special_tokens=True)
    tokenizer.chat_template = template
    return tokenizer


def lookup_call(call_id: str, email: str) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": "lookup_user",
            "arguments": json.dumps({"email": email}),
        },
    }


def helpdesk_prompt() -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": "You are the IT helpdesk agent."},
        {"role": "user", "content": "I can't sign in."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [lookup_call("c1", "a@b.com")],
        },
        {"role": "tool", "tool_call_id": "c1", "content": '{"locked": true}'},
    ]


def helpdesk_completion() -> list[dict[str, Any]]:
    return [
        {
            "role": "assistant",
            "reasoning_content": "The account is locked.",
            "content": "I've sent an unlock link.",
        }
    ]


def helpdesk_row() -> dict[str, Any]:
    return {
        "prompt": helpdesk_prompt(),
        "completion": helpdesk_completion(),
        "tools": [LOOKUP_TOOL],
    }


def helpdesk_segment() -> dict[str, Any]:
    """The helpdesk call as one segment: the tool-call turn, its result, then the reply."""
    return {
        "messages": [*helpdesk_prompt(), *helpdesk_completion()],
        "tools": [LOOKUP_TOOL],
        "call_ids": ["c-a", "c-b"],
        "finish_reasons": ["tool_calls", "stop"],
        "scores": [
            {"name": "feedback", "value": 1.0, "turn": None},
            {"name": "judge", "value": 0.5, "turn": 1},
        ],
        "session_id": "s-1",
    }


def as_context(segment: dict[str, Any], *turns: int) -> dict[str, Any]:
    """``segment`` with ``weight: 0`` on its assistant messages at ``turns``, 0-based among them."""
    segment = copy.deepcopy(segment)
    assistants = [m for m in segment["messages"] if m["role"] == "assistant"]
    for turn in turns:
        assistants[turn]["weight"] = 0
    return segment


@pytest.fixture
def tokenizer() -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)


@pytest.fixture
def reasoning_tokenizer() -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
    tokenizer.chat_template = REASONING_TEMPLATE
    return tokenizer


class TestTracesPromptText:
    def test_renders_tools_tool_calls_and_tool_results(
        self, tokenizer: PreTrainedTokenizerBase
    ) -> None:
        # Act
        text = render_chat_prompt(
            tokenizer, helpdesk_prompt(), [LOOKUP_TOOL], detect_chat_format(tokenizer)
        )

        # Assert
        expected = tokenizer.apply_chat_template(
            chat_messages(helpdesk_prompt()),
            tools=[LOOKUP_TOOL],
            tokenize=False,
            add_generation_prompt=True,
        )
        assert text == expected
        assert '"name": "lookup_user", "description": "Find a user"' in text
        assert (
            '<tool_call>\n{"name": "lookup_user", "arguments": {"email": "a@b.com"}}'
            in text
        )
        assert '<tool_response>\n{"locked": true}\n</tool_response>' in text
        assert text.endswith("<|im_start|>assistant\n")

    def test_reads_json_text_prompts(self, tokenizer: PreTrainedTokenizerBase) -> None:
        chat_format = detect_chat_format(tokenizer)

        text = render_chat_prompt(
            tokenizer, json.dumps(helpdesk_prompt()), [LOOKUP_TOOL], chat_format
        )

        assert text == render_chat_prompt(
            tokenizer, helpdesk_prompt(), [LOOKUP_TOOL], chat_format
        )

    def test_moves_past_reasoning_to_the_key_the_template_reads(
        self, reasoning_tokenizer: PreTrainedTokenizerBase
    ) -> None:
        prompt = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "thinking": "greet back", "content": "hello"},
            {"role": "user", "content": "bye"},
        ]

        text = render_chat_prompt(reasoning_tokenizer, prompt, None, None)

        assert "<think>\ngreet back\n</think>\n\nhello<|im_end|>" in text

    def test_passes_chat_template_kwargs(
        self, reasoning_tokenizer: PreTrainedTokenizerBase
    ) -> None:
        reasoning_tokenizer.chat_template = "{{ greeting }}" + REASONING_TEMPLATE

        text = render_chat_prompt(
            reasoning_tokenizer,
            [{"role": "user", "content": "hi"}],
            None,
            None,
            {"greeting": "HELLO "},
        )

        assert text.startswith("HELLO <|im_start|>user\nhi")

    def test_rejects_a_prompt_ending_with_an_assistant_turn(
        self, tokenizer: PreTrainedTokenizerBase
    ) -> None:
        prompt = [*helpdesk_prompt(), {"role": "assistant", "content": "x"}]

        with pytest.raises(ValueError, match="must not end with an assistant turn"):
            render_chat_prompt(tokenizer, prompt, None, None)


class TestTracesReferenceText:
    def test_renders_reasoning_and_content_through_the_template(
        self, reasoning_tokenizer: PreTrainedTokenizerBase
    ) -> None:
        text = traces_reference_text(
            reasoning_tokenizer, helpdesk_completion(), None, None
        )

        assert text == (
            "<think>\nThe account is locked.\n</think>\n\nI've sent an unlock link."
        )

    def test_renders_tool_calls_as_the_chat_format_writes_them(
        self, tokenizer: PreTrainedTokenizerBase
    ) -> None:
        # Arrange
        completion = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [lookup_call("c2", "x@y.com")],
            }
        ]

        # Act
        text = traces_reference_text(
            tokenizer, completion, [LOOKUP_TOOL], detect_chat_format(tokenizer)
        )

        # Assert
        assert text == (
            '<tool_call>\n{"name": "lookup_user", "arguments": {"email": "x@y.com"}}'
            "\n</tool_call>"
        )

    def test_drops_reasoning_a_template_reads_no_key_for_as_sft_does(
        self, tokenizer: PreTrainedTokenizerBase
    ) -> None:
        # Arrange
        _warn_reasoning_dropped.cache_clear()
        row = helpdesk_row()

        # Act
        with pytest.warns(UserWarning, match="assistant reasoning is dropped"):
            text = traces_reference_text(tokenizer, row["completion"], None, None)
        ids, mask = encode_sft_row(
            tokenizer, row, "completion", detect_chat_format(tokenizer), True
        )

        # Assert
        assert text == "I've sent an unlock link."
        assert trained_text(tokenizer, ids, mask) == [f"{text}<|im_end|>"]

    def test_warns_once_about_dropped_reasoning(
        self, tokenizer: PreTrainedTokenizerBase
    ) -> None:
        _warn_reasoning_dropped.cache_clear()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(3):
                traces_reference_text(tokenizer, helpdesk_completion(), None, None)

        assert [str(w.message) for w in caught].count(str(caught[0].message)) == 1

    def test_includes_the_think_opener_the_generation_prompt_prefills(self) -> None:
        tokenizer = template_tokenizer(THINKING_SWITCH_TEMPLATE)

        text = traces_reference_text(
            tokenizer, helpdesk_completion(), None, None, {"enable_thinking": True}
        )

        assert text == (
            "<think>\nThe account is locked.\n</think>\n\nI've sent an unlock link."
        )

    def test_rejects_reasoning_after_a_closed_think_prefill(self) -> None:
        tokenizer = template_tokenizer(THINKING_SWITCH_TEMPLATE)

        with pytest.raises(ValueError, match="closes the thinking block"):
            traces_reference_text(tokenizer, helpdesk_completion(), None, None)

    def test_a_closed_prefilled_think_block_is_not_an_opener(self) -> None:
        tokenizer = template_tokenizer(THINKING_SWITCH_TEMPLATE)
        completion = [{"role": "assistant", "content": "Done."}]

        text = traces_reference_text(tokenizer, completion, None, None)

        assert text == "Done."

    def test_keeps_special_tokens_inside_the_turn(self) -> None:
        # Arrange
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        tokenizer.add_special_tokens({"additional_special_tokens": ["<|call|>"]})
        tokenizer.chat_template = (
            "{% for message in messages %}<|im_start|>{{ message.role }}\n"
            "{% if message.role == 'assistant' %}<|call|>{% endif %}"
            "{{ message.content or '' }}<|im_end|>\n{% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
        )

        # Act
        text = traces_reference_text(
            tokenizer, [{"role": "assistant", "content": "lookup"}], None, None
        )

        # Assert
        assert text == "<|call|>lookup"

    def test_rejects_a_decode_that_is_not_text(
        self, tokenizer: PreTrainedTokenizerBase
    ) -> None:
        tokenizer.decode = lambda *_args, **_kwargs: ["not", "text"]

        with pytest.raises(TypeError, match="decode must return a string"):
            traces_reference_text(tokenizer, helpdesk_completion(), None, None)

    def test_rejects_more_than_one_message(
        self, tokenizer: PreTrainedTokenizerBase
    ) -> None:
        with pytest.raises(ValueError, match="exactly one assistant message"):
            traces_reference_text(tokenizer, helpdesk_completion() * 2, None, None)


class TestCheckTracesSplit:
    def test_accepts_arrow_segments(self) -> None:
        dataset = HFDataset.from_list([helpdesk_segment(), helpdesk_segment()])

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            checked = check_traces_split(dataset, "train split")

        assert len(checked) == 2

    def test_drops_a_bad_row_and_warns_with_the_count_and_first_error(self) -> None:
        # Arrange
        bad = helpdesk_segment()
        bad["messages"][3]["tool_call_id"] = "c9"
        dataset = HFDataset.from_list([helpdesk_segment(), bad, bad])

        # Act
        with pytest.warns(UserWarning, match=r"2 of 3 rows .* row 2: .*'c9'"):
            checked = check_traces_split(dataset, "train split")

        # Assert
        assert checked["messages"] == [dataset[0]["messages"]]

    def test_raises_when_no_row_is_a_segment(self) -> None:
        bad = helpdesk_segment()
        bad["messages"][3]["tool_call_id"] = "c9"
        dataset = HFDataset.from_list([bad] * 7)

        with pytest.raises(
            ValueError, match=r"No row of the train split .* row 1: .*'c9'"
        ):
            check_traces_split(dataset, "train split")

    def test_keeps_and_trains_a_trailing_tool_result_without_content(
        self, tokenizer: PreTrainedTokenizerBase
    ) -> None:
        # Arrange
        segment = helpdesk_segment()
        segment["messages"] += [
            {"role": "user", "content": "And b@c.com?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [lookup_call("c2", "b@c.com")],
            },
            {"role": "tool", "tool_call_id": "c2", "content": None},
        ]
        segment["call_ids"].append("c-c")
        segment["finish_reasons"].append("tool_calls")
        dataset = HFDataset.from_list([segment])

        # Act
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            checked = check_traces_split(dataset, "train split")

        # Assert
        ids, mask = encode_sft_row(
            tokenizer, checked[0], "completion", detect_chat_format(tokenizer), True
        )
        assert len(trained_text(tokenizer, ids, mask)) == 3

    def test_drops_a_tool_result_without_content_before_a_turn(self) -> None:
        segment = helpdesk_segment()
        segment["messages"][3]["content"] = None
        dataset = HFDataset.from_list([segment, helpdesk_segment()])

        with pytest.warns(UserWarning, match=r"messages\[3\] is a tool result"):
            checked = check_traces_split(dataset, "train split")

        assert len(checked) == 1

    def test_an_empty_split_passes(self) -> None:
        dataset = HFDataset.from_list([helpdesk_segment()]).select([])

        checked = check_traces_split(dataset, "train split")

        assert len(checked) == 0

    def test_names_the_field_of_a_bad_message(self) -> None:
        bad = helpdesk_segment()
        bad["messages"][4]["role"] = "robot"
        dataset = HFDataset.from_list([bad])

        with pytest.raises(ValueError, match=r"row 1, messages\.4: "):
            check_traces_split(dataset, "train split")

    def test_drops_a_pair_and_points_to_import_conversion(self) -> None:
        segment = {**helpdesk_segment(), "prompt": None, "completion": None}
        pair = {**helpdesk_row(), "messages": None}
        dataset = HFDataset.from_list([segment, pair])

        with pytest.warns(
            UserWarning, match=r"row 2: a prompt/completion pair.*converts pairs"
        ):
            checked = check_traces_split(dataset, "train split")

        assert len(checked) == 1

    def test_reads_a_row_with_both_pair_columns_as_a_pair(self) -> None:
        row = {
            **helpdesk_row(),
            "messages": [*helpdesk_prompt(), *helpdesk_completion()],
        }
        dataset = HFDataset.from_list([row])

        with pytest.raises(ValueError, match="row 1: a prompt/completion pair"):
            check_traces_split(dataset, "train split")

    def test_rejects_a_row_without_messages(self) -> None:
        dataset = HFDataset.from_list([{"prompt": "hi", "answer": "a"}])

        with pytest.raises(ValueError, match="row 1: has no messages column"):
            check_traces_split(dataset, "train split")

    def test_names_a_per_turn_list_of_the_wrong_length(self) -> None:
        dataset = HFDataset.from_list(
            [{**helpdesk_segment(), "finish_reasons": ["stop"]}]
        )

        with pytest.raises(ValueError, match="finish_reasons has 1 entries"):
            check_traces_split(dataset, "train split")

    def test_rejects_a_segment_with_no_target(self) -> None:
        dataset = HFDataset.from_list([as_context(helpdesk_segment(), 0, 1)])

        with pytest.raises(ValueError, match="every assistant message has weight 0"):
            check_traces_split(dataset, "train split")


def json_text_row() -> dict[str, str]:
    """The helpdesk row as a snapshot ``calls.parquet`` stores it: JSON text cells."""
    return {key: json.dumps(value) for key, value in helpdesk_row().items()}


def reply_segment(**columns: Any) -> dict[str, Any]:
    """The helpdesk segment whose tool-call turn is context, so the reply is the one target."""
    return {**as_context(helpdesk_segment(), 0), **columns}


class TestTracesRolloutSplit:
    def test_reads_json_text_cells(
        self, reasoning_tokenizer: PreTrainedTokenizerBase
    ) -> None:
        dataset = HFDataset.from_list([json_text_segment()])

        split = traces_rollout_split(
            dataset, "train split", reasoning_tokenizer, None, {}, "reference_trace"
        )

        assert split[0]["reference_trace"].endswith("I've sent an unlock link.")
        assert render_chat_prompt(
            reasoning_tokenizer, split[0]["prompt"], None, None
        ) == render_chat_prompt(reasoning_tokenizer, helpdesk_prompt(), None, None)

    def test_adds_an_answer_and_the_rendered_reference(
        self, reasoning_tokenizer: PreTrainedTokenizerBase
    ) -> None:
        dataset = HFDataset.from_list([reply_segment()])

        split = traces_rollout_split(
            dataset, "train split", reasoning_tokenizer, None, {}, "reference_trace"
        )

        assert split[0]["answer"] is None
        assert split[0]["reference_trace"] == (
            "<think>\nThe account is locked.\n</think>\n\nI've sent an unlock link."
        )

    def test_a_teacher_model_without_a_context_key_ignores_the_completion(
        self, tokenizer: PreTrainedTokenizerBase
    ) -> None:
        dataset = HFDataset.from_list([reply_segment(answer="unlocked")])

        split = traces_rollout_split(
            dataset, "train split", tokenizer, detect_chat_format(tokenizer), {}, None
        )

        assert "reference_trace" not in split.column_names
        assert split["answer"] == ["unlocked"]

    def test_rejects_a_split_that_breaks_the_spec(
        self, tokenizer: PreTrainedTokenizerBase
    ) -> None:
        dataset = HFDataset.from_list([helpdesk_row()])

        with pytest.raises(ValueError, match="row 1: a prompt/completion pair"):
            traces_rollout_split(dataset, "train split", tokenizer, None, {}, None)


def trained_text(
    tokenizer: PreTrainedTokenizerBase, ids: list[int], mask: list[bool]
) -> list[str]:
    return [tokenizer.decode(ids[start:end]) for start, end in assistant_runs(mask)]


class TestTracesSft:
    def test_completion_only_loss_trains_the_completion_alone(
        self, tokenizer: PreTrainedTokenizerBase
    ) -> None:
        # Arrange
        row = HFDataset.from_list([helpdesk_row()])[0]
        chat_format = detect_chat_format(tokenizer)

        # Act
        ids, mask = encode_sft_row(tokenizer, row, "completion", chat_format, True)
        all_ids, all_mask = encode_sft_row(
            tokenizer, row, "completion", chat_format, False
        )

        # Assert
        assert ids == all_ids
        assert trained_text(tokenizer, ids, mask) == [
            "I've sent an unlock link.<|im_end|>"
        ]
        assert len(assistant_runs(all_mask)) == 2

    def test_json_text_cells_encode_like_native_lists(
        self, tokenizer: PreTrainedTokenizerBase
    ) -> None:
        # Arrange
        native = HFDataset.from_list([helpdesk_row()])[0]
        text = {key: json.dumps(value) for key, value in helpdesk_row().items()}
        chat_format = detect_chat_format(tokenizer)

        # Act
        text_ids, text_mask = encode_sft_row(
            tokenizer, text, "completion", chat_format, True
        )
        native_ids, native_mask = encode_sft_row(
            tokenizer, native, "completion", chat_format, True
        )

        # Assert
        assert (text_ids, text_mask) == (native_ids, native_mask)

    def test_prompt_completion_wins_over_a_messages_column(
        self, tokenizer: PreTrainedTokenizerBase
    ) -> None:
        # Arrange
        row = helpdesk_row()
        item = {**row, "messages": row["prompt"][:2]}
        chat_format = detect_chat_format(tokenizer)

        # Act
        ids, mask = encode_sft_row(tokenizer, item, "completion", chat_format, True)

        # Assert
        assert trained_text(tokenizer, ids, mask) == [
            "I've sent an unlock link.<|im_end|>"
        ]

    def test_sft_split_keeps_segments_and_drops_null_pair_columns(
        self, tokenizer: PreTrainedTokenizerBase
    ) -> None:
        row = {**helpdesk_segment(), "prompt": None, "completion": None}
        dataset = HFDataset.from_list([row])

        split = traces_sft_split(
            dataset, "train split", tokenizer, detect_chat_format(tokenizer), {}
        )

        assert "messages" in split.column_names
        assert "prompt" not in split.column_names

    def test_rejects_a_prompt_with_no_assistant_turn_after_it(
        self, tokenizer: PreTrainedTokenizerBase
    ) -> None:
        messages = chat_messages([*helpdesk_prompt(), {"role": "user", "content": "?"}])

        with pytest.raises(
            ValueError, match="assistant turn to train after its prompt"
        ):
            encode_chat_transcript(
                tokenizer, messages, None, detect_chat_format(tokenizer), len(messages)
            )

    def test_messages_rows_train_every_assistant_turn(
        self, tokenizer: PreTrainedTokenizerBase
    ) -> None:
        row = helpdesk_row()
        item = {"messages": [*row["prompt"], *row["completion"]], "tools": row["tools"]}

        _, mask = encode_sft_row(
            tokenizer, item, "completion", detect_chat_format(tokenizer), True
        )

        assert len(assistant_runs(mask)) == 2

    def test_dataset_env_batches_mask_only_the_completion(
        self, tokenizer: PreTrainedTokenizerBase
    ) -> None:
        # Arrange
        dataset = HFDataset.from_list([helpdesk_row()])
        env = DatasetEnv(
            train_dataset=dataset,
            test_dataset=dataset,
            tokenizer=tokenizer,
            objective="sft",
            response_column="completion",
            completion_only_loss=True,
            data_batch_size_per_gpu=1,
        )

        # Act
        batch = env.reset()

        # Assert
        ids = batch["input_ids"][0].tolist()
        mask = batch["loss_mask"][0].tolist()
        trained = trained_text(tokenizer, ids, mask)
        assert len(trained) == 1
        assert "I've sent an unlock link." in trained[0]
        assert "lookup_user" not in trained[0]


def as_text_parts(text: str) -> list[dict[str, str]]:
    return [{"type": "text", "text": text}]


class TestTracesSftContentParts:
    @pytest.mark.parametrize(
        ("column", "index"),
        [("prompt", 1), ("prompt", 3), ("completion", 0)],
        ids=["user", "tool_result", "completion"],
    )
    def test_text_parts_render_like_a_string(
        self, tokenizer: PreTrainedTokenizerBase, column: str, index: int
    ) -> None:
        # Arrange
        chat_format = detect_chat_format(tokenizer)
        parts_row = helpdesk_row()
        message = parts_row[column][index]
        message["content"] = as_text_parts(message["content"])

        # Act
        parts = encode_sft_row(tokenizer, parts_row, "completion", chat_format, True)
        strings = encode_sft_row(
            tokenizer, helpdesk_row(), "completion", chat_format, True
        )
        parts_prompt = render_chat_prompt(
            tokenizer, parts_row["prompt"], [LOOKUP_TOOL], chat_format
        )

        # Assert
        assert parts == strings
        assert parts_prompt == render_chat_prompt(
            tokenizer, helpdesk_prompt(), [LOOKUP_TOOL], chat_format
        )
        assert trained_text(tokenizer, *parts) == [
            "I've sent an unlock link.<|im_end|>"
        ]


class TestTracesSftContext:
    def test_keeps_past_reasoning_a_later_final_turn_would_drop(self) -> None:
        # Arrange
        tokenizer = template_tokenizer(DROPS_PAST_ANALYSIS_TEMPLATE, HARMONY_SPECIALS)
        row = helpdesk_row()
        row["prompt"][2]["thinking"] = "Look the user up first."
        row["completion"] = [
            {"role": "assistant", "thinking": "Locked.", "content": "Unlocked."}
        ]

        # Act
        ids, mask = encode_sft_row(tokenizer, row, "completion", None, True)

        # Assert
        start = mask.index(True)
        context = tokenizer.decode(ids[:start])
        assert context == render_chat_prompt(tokenizer, row["prompt"], None, None)
        assert "Look the user up first." in context
        assert trained_text(tokenizer, ids, mask) == [
            (
                "<|channel|>analysis<|message|>Locked.<|end|><|start|>assistant"
                "<|channel|>final<|message|>Unlocked.<|return|>"
            )
        ]

    def test_trains_reasoning_after_an_open_think_prefill(self) -> None:
        tokenizer = template_tokenizer(THINKING_SWITCH_TEMPLATE)
        tokenizer.chat_template = tokenizer.chat_template.replace(
            "not enable_thinking", "false"
        )

        ids, mask = encode_sft_row(tokenizer, helpdesk_row(), "completion", None, True)

        assert tokenizer.decode(ids[: mask.index(True)]).endswith(
            "<|im_start|>assistant\n<think>\n"
        )
        assert trained_text(tokenizer, ids, mask) == [
            "The account is locked.\n</think>\n\nI've sent an unlock link.<|im_end|>"
        ]

    def test_rejects_reasoning_after_a_closed_think_prefill(self) -> None:
        tokenizer = template_tokenizer(THINKING_SWITCH_TEMPLATE)
        # Qwen3.5 holds the think markers as single, non-special added tokens.
        tokenizer.add_tokens(["<think>", "</think>"])
        row = {"prompt": helpdesk_prompt()[1:2], "completion": helpdesk_completion()}

        with pytest.raises(ValueError, match="enable_thinking: true"):
            encode_sft_row(tokenizer, row, "completion", None, True)


class TestCheckTracesRender:
    @pytest.mark.parametrize(
        "row",
        [
            helpdesk_row(),
            {
                "messages": [*helpdesk_prompt(), *helpdesk_completion()],
                "tools": [LOOKUP_TOOL],
            },
        ],
        ids=["prompt_completion", "messages"],
    )
    def test_accepts_rows_that_render_as_a_rollout_sees_them(
        self, tokenizer: PreTrainedTokenizerBase, row: dict[str, Any]
    ) -> None:
        dataset = HFDataset.from_list([row] * 20)

        check_traces_render(dataset, tokenizer, detect_chat_format(tokenizer), {})

    @pytest.mark.parametrize("count", [16, 20, 31, 40])
    def test_checks_the_last_row(self, count: int) -> None:
        # Arrange
        tokenizer = template_tokenizer(THINKING_SWITCH_TEMPLATE)
        plain = {
            **helpdesk_row(),
            "completion": [{"role": "assistant", "content": "ok"}],
        }
        dataset = HFDataset.from_list([plain] * (count - 1) + [helpdesk_row()])

        # Act / Assert
        with pytest.raises(ValueError, match=f"Row {count}: .*closes the thinking"):
            check_traces_render(dataset, tokenizer, None, {})

    def test_rejects_arguments_written_as_an_escaped_json_string(self) -> None:
        tokenizer = template_tokenizer(
            REASONING_TEMPLATE.replace(
                "{{ message.content or '' }}",
                "{{ message.content or '' }}{% for c in message.tool_calls or [] %}"
                "{{ c.function.arguments | tojson | tojson }}{% endfor %}",
            )
        )
        dataset = HFDataset.from_list([helpdesk_row()])

        with pytest.raises(ValueError, match=r"Row 1: .*'lookup_user'.*escaped JSON"):
            check_traces_render(dataset, tokenizer, None, {})

    def test_rejects_reasoning_when_the_generation_prompt_closes_thinking(
        self,
    ) -> None:
        tokenizer = template_tokenizer(THINKING_SWITCH_TEMPLATE)
        dataset = HFDataset.from_list([helpdesk_row()])

        with pytest.raises(ValueError, match="closes the thinking block"):
            check_traces_render(dataset, tokenizer, None, {})

        check_traces_render(dataset, tokenizer, None, {"enable_thinking": True})

    def test_rejects_a_prompt_that_is_not_a_prefix_of_the_sft_context(self) -> None:
        tokenizer = template_tokenizer(
            "{% if messages | length > 2 %}[answered]{% endif %}" + REASONING_TEMPLATE
        )
        row = {"messages": [*helpdesk_prompt()[:2], *helpdesk_completion()]}
        dataset = HFDataset.from_list([row])

        with pytest.raises(
            ValueError,
            match="Row 1: SFT renders a trained turn or its context differently",
        ):
            check_traces_render(dataset, tokenizer, None, {})


# Same commits as test_feedback_frames. An unpinned cache entry can load with no chat template.
TOKENIZER_REVISIONS = {
    "Qwen/Qwen3-4B": "1cfa9a7208912126459214e8b04321603b3df60c",
    "openai/gpt-oss-20b": "6cee5e81ee83917806bbde320786a8fb61efebee",
}
TOKENIZER_FILES = [
    "tokenizer*",
    "special_tokens_map.json",
    "chat_template*",
    "*.jinja",
]


def cached_tokenizer(model: str) -> PreTrainedTokenizerBase:
    """Tokenizer files at :data:`TOKENIZER_REVISIONS`.

    Skips when that revision cannot be loaded or its chat template is missing.
    """
    revision = TOKENIZER_REVISIONS[model]
    try:
        # tests/conftest.py forces the Hub offline; these tests download the pin.
        with patch.object(hub_constants, "HF_HUB_OFFLINE", False):
            path = Path(
                snapshot_download(
                    model, revision=revision, allow_patterns=TOKENIZER_FILES
                )
            )
        tokenizer = AutoTokenizer.from_pretrained(path)
    except (
        OSError,
        GatedRepoError,
        HfHubHTTPError,
        LocalEntryNotFoundError,
        OfflineModeIsEnabled,
    ) as error:
        pytest.skip(f"{model} tokenizer at {revision} is not available: {error}")
    if path.name != revision:
        msg = f"{model} tokenizer resolved to {path.name}, not {revision}"
        raise AssertionError(msg)
    if not tokenizer.chat_template:
        pytest.skip(f"{model} tokenizer at {revision} has no chat template")
    return tokenizer


def per_turn_targets(
    ids: list[int], mask: list[bool]
) -> list[tuple[list[int], list[int]]]:
    """Per trained run: the ids before it and its ids."""
    return [(ids[:start], ids[start:end]) for start, end in assistant_runs(mask)]


class TestExpandSegments:
    def test_expands_a_segment_into_one_row_per_turn(self) -> None:
        # Arrange
        dataset = HFDataset.from_list([helpdesk_segment()])

        # Act
        rows = expand_segments(dataset)

        # Assert
        messages = chat_messages(helpdesk_segment()["messages"])
        assert len(rows) == 2
        assert [chat_messages(row["prompt"]) for row in rows] == [
            messages[:2],
            messages[:4],
        ]
        assert [chat_messages(row["completion"]) for row in rows] == [
            [messages[2]],
            [messages[4]],
        ]
        assert rows["call_id"] == ["c-a", "c-b"]
        assert rows["finish_reason"] == ["tool_calls", "stop"]
        assert rows["session_id"] == ["s-1", "s-1"]
        assert [[score["name"] for score in row["scores"]] for row in rows] == [
            ["feedback"],
            ["feedback", "judge"],
        ]
        assert "messages" not in rows.column_names

    def test_drops_token_ids_and_logprobs(self) -> None:
        segment = {
            **helpdesk_segment(),
            "completion_token_ids": [[1, 2], [3]],
            "completion_logprobs": [[-0.5, -0.1], [-0.2]],
        }

        rows = expand_segments(HFDataset.from_list([segment]))

        assert "completion_token_ids" not in rows.column_names
        assert "completion_logprobs" not in rows.column_names

    def test_a_weight_zero_turn_is_context_not_a_rollout(self) -> None:
        segment = as_context(helpdesk_segment(), 0)

        rows = expand_segments(HFDataset.from_list([segment]))

        assert rows["call_id"] == ["c-b"]
        assert len(chat_messages(rows[0]["prompt"])) == 4

    def test_drops_weight_from_the_served_messages(self) -> None:
        segment = as_context(helpdesk_segment(), 0)
        segment["messages"][4]["weight"] = 1

        rows = expand_segments(HFDataset.from_list([segment]))

        served = [*json.loads(rows[0]["prompt"]), *json.loads(rows[0]["completion"])]
        assert [message["role"] for message in served] == [
            "system",
            "user",
            "assistant",
            "tool",
            "assistant",
        ]
        assert all(message.get("weight") is None for message in served)

    def test_null_call_ids_are_lineage_not_context(self) -> None:
        segment = {**helpdesk_segment(), "call_ids": [None, None]}

        rows = expand_segments(HFDataset.from_list([segment]))

        assert rows["call_id"] == [None, None]

    def test_skips_a_turn_cut_off_at_the_token_limit(self) -> None:
        segment = {**helpdesk_segment(), "finish_reasons": ["length", "stop"]}

        rows = expand_segments(HFDataset.from_list([segment]))

        assert rows["call_id"] == ["c-b"]

    def test_rejects_a_split_with_no_target_turn(self) -> None:
        segment = {
            **as_context(helpdesk_segment(), 0),
            "finish_reasons": ["stop", "length"],
        }

        with pytest.raises(ValueError, match="No assistant turn of the traces split"):
            expand_segments(HFDataset.from_list([segment]))

    def test_reads_json_text_and_no_per_turn_columns(self) -> None:
        segment = helpdesk_segment()
        row = {
            "messages": json.dumps(segment["messages"]),
            "scores": json.dumps(segment["scores"]),
        }

        rows = expand_segments(HFDataset.from_list([row]))

        assert rows["call_id"] == [None, None]
        assert [len(json.loads(row["completion"])) for row in rows] == [1, 1]
        assert [len(row["scores"]) for row in rows] == [1, 2]

    def test_keeps_null_scores_null(self) -> None:
        segment = helpdesk_segment()
        other = {**segment, "scores": None}

        rows = expand_segments(HFDataset.from_list([other, segment]))

        assert rows["scores"][:2] == [None, None]


def json_text_segment() -> dict[str, Any]:
    """The helpdesk segment as the platform stores an upload: every nested cell as JSON text."""
    segment = {
        **as_context(helpdesk_segment(), 0),
        "call_ids": [None, "c-b"],
        "finish_reasons": ["tool_calls", "stop"],
    }
    return {
        key: json.dumps(value) if isinstance(value, (list, dict)) else value
        for key, value in segment.items()
    }


class TestJsonTextSegments:
    def test_split_expansion_and_sft_read_json_text_cells(
        self, tokenizer: PreTrainedTokenizerBase
    ) -> None:
        # Arrange
        dataset = HFDataset.from_list([json_text_segment()])

        # Act
        check_traces_split(dataset, "train split")
        rows = expand_segments(dataset)
        ids, mask = encode_sft_row(
            tokenizer,
            dataset[0],
            "completion",
            detect_chat_format(tokenizer),
            True,
            traces_skipped_turns(dataset[0]),
        )

        # Assert
        assert rows["call_id"] == ["c-b"]
        assert [score["name"] for score in rows[0]["scores"]] == ["feedback", "judge"]
        assert trained_text(tokenizer, ids, mask) == [
            "I've sent an unlock link.<|im_end|>"
        ]

    def test_names_a_per_turn_cell_that_is_not_json(
        self, tokenizer: PreTrainedTokenizerBase
    ) -> None:
        row = {**json_text_segment(), "finish_reasons": "[oops"}

        with pytest.raises(
            ValueError, match="The finish_reasons column is not valid JSON"
        ):
            traces_skipped_turns(row)


class TestTracesSegmentRollout:
    def test_each_turn_is_a_prompt_and_its_reference(
        self, reasoning_tokenizer: PreTrainedTokenizerBase
    ) -> None:
        # Arrange
        segment = helpdesk_segment()
        segment["messages"][2] = {"role": "assistant", "content": "Looking you up."}
        segment["messages"][3] = {"role": "user", "content": "Go on."}
        dataset = HFDataset.from_list([segment])

        # Act
        split = traces_rollout_split(
            dataset, "train split", reasoning_tokenizer, None, {}, "reference_trace"
        )

        # Assert
        assert split["reference_trace"] == [
            "Looking you up.",
            "<think>\nThe account is locked.\n</think>\n\nI've sent an unlock link.",
        ]
        assert render_chat_prompt(
            reasoning_tokenizer, split[1]["prompt"], None, None
        ) == render_chat_prompt(
            reasoning_tokenizer, segment["messages"][:4], None, None
        )


class TestTracesSkippedTurns:
    @pytest.mark.parametrize(
        ("context", "columns", "skipped"),
        [
            ((), {}, []),
            ((), {"finish_reasons": ["length", "stop"]}, [2]),
            ((0,), {}, [2]),
            ((0,), {"finish_reasons": ["stop", "length"]}, [2, 4]),
            ((), {"call_ids": [None, None]}, []),
        ],
        ids=["none", "length", "context", "both", "null_call_ids"],
    )
    def test_skips_context_and_length_turns(
        self, context: tuple[int, ...], columns: dict[str, Any], skipped: list[int]
    ) -> None:
        segment = {
            **as_context(helpdesk_segment(), *context),
            "call_ids": None,
            "finish_reasons": None,
        }

        assert traces_skipped_turns({**segment, **columns}) == skipped

    def test_an_expanded_turn_skips_nothing(self) -> None:
        assert traces_skipped_turns(helpdesk_row()) == []


class TestTracesSegmentSft:
    def test_trains_every_assistant_turn_in_one_render(
        self, tokenizer: PreTrainedTokenizerBase
    ) -> None:
        row = HFDataset.from_list([helpdesk_segment()])[0]

        ids, mask = encode_sft_row(
            tokenizer, row, "completion", detect_chat_format(tokenizer), True
        )

        trained = trained_text(tokenizer, ids, mask)
        assert len(trained) == 2
        assert "lookup_user" in trained[0]
        assert trained[1] == "I've sent an unlock link.<|im_end|>"

    @pytest.mark.parametrize(
        ("context", "columns"),
        [((0,), {}), ((), {"finish_reasons": ["length", "stop"]})],
        ids=["weight_zero", "length"],
    )
    def test_a_skipped_turn_is_context_not_a_target(
        self,
        tokenizer: PreTrainedTokenizerBase,
        context: tuple[int, ...],
        columns: dict[str, Any],
    ) -> None:
        row = {**as_context(helpdesk_segment(), *context), **columns}

        ids, mask = encode_sft_row(
            tokenizer,
            row,
            "completion",
            detect_chat_format(tokenizer),
            True,
            traces_skipped_turns(row),
        )

        assert trained_text(tokenizer, ids, mask) == [
            "I've sent an unlock link.<|im_end|>"
        ]
        assert "lookup_user" in tokenizer.decode(ids[: mask.index(True)])

    def test_matches_the_expanded_turns(
        self, tokenizer: PreTrainedTokenizerBase
    ) -> None:
        # Arrange
        dataset = HFDataset.from_list([helpdesk_segment()])
        chat_format = detect_chat_format(tokenizer)

        # Act
        segment = encode_sft_row(tokenizer, dataset[0], "completion", chat_format, True)
        turns = [
            encode_sft_row(tokenizer, row, "completion", chat_format, True)
            for row in expand_segments(dataset)
        ]

        # Assert
        assert per_turn_targets(*segment) == [
            per_turn_targets(*turn)[0] for turn in turns
        ]

    def test_qwen3_segment_matches_the_expanded_turns(self) -> None:
        # Arrange
        tokenizer = cached_tokenizer("Qwen/Qwen3-4B")
        segment_row = helpdesk_segment()
        segment_row["messages"][2]["reasoning_content"] = "Look the user up."
        dataset = HFDataset.from_list([segment_row])
        chat_format = detect_chat_format(tokenizer)

        # Act
        split = traces_sft_split(dataset, "train split", tokenizer, chat_format, {})
        segment = encode_sft_row(tokenizer, split[0], "completion", chat_format, True)
        turns = [
            encode_sft_row(tokenizer, row, "completion", chat_format, True)
            for row in expand_segments(dataset)
        ]

        # Assert
        assert "messages" in split.column_names
        targets = per_turn_targets(*segment)
        assert targets == [per_turn_targets(*turn)[0] for turn in turns]
        assert "Look the user up." in tokenizer.decode(targets[1][0])

    def test_sft_split_drops_segments_with_no_target(
        self, tokenizer: PreTrainedTokenizerBase
    ) -> None:
        cut = {**helpdesk_segment(), "finish_reasons": ["length", "length"]}
        dataset = HFDataset.from_list([helpdesk_segment(), cut])

        with pytest.warns(UserWarning, match="1 segments of the train split"):
            split = traces_sft_split(
                dataset, "train split", tokenizer, detect_chat_format(tokenizer), {}
            )

        assert len(split) == 1

    def test_sft_split_rejects_a_split_with_nothing_to_train(
        self, tokenizer: PreTrainedTokenizerBase
    ) -> None:
        cut = {**helpdesk_segment(), "finish_reasons": ["length", "length"]}
        dataset = HFDataset.from_list([cut])

        with pytest.raises(ValueError, match="No assistant turn of the train split"):
            traces_sft_split(dataset, "train split", tokenizer, None, {})

    def test_dataset_env_masks_context_turns(
        self, tokenizer: PreTrainedTokenizerBase
    ) -> None:
        # Arrange
        dataset = traces_sft_split(
            HFDataset.from_list([as_context(helpdesk_segment(), 0)]),
            "train split",
            tokenizer,
            detect_chat_format(tokenizer),
            {},
        )
        env = DatasetEnv(
            train_dataset=dataset,
            test_dataset=dataset,
            tokenizer=tokenizer,
            objective="sft",
            response_column="completion",
            completion_only_loss=True,
            data_batch_size_per_gpu=1,
        )

        # Act
        batch = env.reset()

        # Assert
        ids = batch["input_ids"][0].tolist()
        mask = batch["loss_mask"][0].tolist()
        assert trained_text(tokenizer, ids, mask) == [
            "I've sent an unlock link.<|im_end|>"
        ]


def inserted_turn_segment() -> dict[str, Any]:
    """A segment with a harness-inserted assistant message right after a target turn."""
    return {
        "messages": [
            {"role": "system", "content": "You are the IT helpdesk agent."},
            {"role": "user", "content": "I can't sign in."},
            {"role": "assistant", "content": "Let me check."},
            {"role": "assistant", "content": "[harness] session resumed", "weight": 0},
            {"role": "user", "content": "Any news?"},
            {"role": "assistant", "content": "I've sent an unlock link."},
        ],
        "call_ids": ["a", None, "b"],
    }


class TestTracesSegmentAdjacentContext:
    def test_trains_around_an_inserted_context_turn(
        self, tokenizer: PreTrainedTokenizerBase
    ) -> None:
        # Arrange
        dataset = HFDataset.from_list([inserted_turn_segment()])
        chat_format = detect_chat_format(tokenizer)

        # Act
        check_traces_split(dataset, "train split")
        ids, mask = encode_sft_row(
            tokenizer,
            dataset[0],
            "completion",
            chat_format,
            True,
            traces_skipped_turns(dataset[0]),
        )
        turns = [
            encode_sft_row(tokenizer, row, "completion", chat_format, True)
            for row in expand_segments(dataset)
        ]

        # Assert
        assert trained_text(tokenizer, ids, mask) == [
            "Let me check.<|im_end|>",
            "I've sent an unlock link.<|im_end|>",
        ]
        assert "[harness] session resumed" in tokenizer.decode(ids)
        assert per_turn_targets(ids, mask) == [
            per_turn_targets(*turn)[0] for turn in turns
        ]

    def test_rejects_a_trained_turn_after_an_earlier_context_pair(
        self, tokenizer: PreTrainedTokenizerBase
    ) -> None:
        messages = chat_messages(
            [
                *inserted_turn_segment()["messages"][:4],
                {"role": "assistant", "content": "Trained too soon."},
            ]
        )

        with pytest.raises(ValueError, match="Messages 3 and 4 are back-to-back"):
            encode_chat_transcript(tokenizer, messages, None, None, skipped=[3])

    def test_rejects_a_trained_turn_right_after_another(
        self, tokenizer: PreTrainedTokenizerBase
    ) -> None:
        messages = chat_messages(inserted_turn_segment()["messages"])

        with pytest.raises(ValueError, match="back-to-back assistant turns"):
            encode_chat_transcript(tokenizer, messages, None, None)


def thinking_segment() -> dict[str, Any]:
    """A segment whose tool-call turn has analysis a later final turn makes gpt-oss drop."""
    segment = helpdesk_segment()
    segment["messages"][2]["thinking"] = "Look the user up first."
    segment["messages"][4] = {
        "role": "assistant",
        "thinking": "Locked.",
        "content": "Unlocked.",
    }
    return segment


def two_question_segment(reasoning_key: str) -> dict[str, Any]:
    """A plain two-question chat whose turns both reason."""
    return {
        "messages": [
            {"role": "user", "content": "What is 2+3?"},
            {"role": "assistant", reasoning_key: "Two plus three.", "content": "5."},
            {"role": "user", "content": "And 4+4?"},
            {"role": "assistant", reasoning_key: "Four plus four.", "content": "8."},
        ]
    }


class TestTracesSegmentSftRewritingTemplate:
    def test_detects_a_template_that_drops_a_trained_turns_own_analysis(
        self,
    ) -> None:
        # Arrange: the first turn's analysis is gone once the second final follows.
        dataset = HFDataset.from_list([two_question_segment("thinking")])
        harmony = template_tokenizer(DROPS_PAST_ANALYSIS_TEMPLATE, HARMONY_SPECIALS)

        # Act / Assert
        assert template_rewrites_turns(dataset, harmony, HarmonyFormat(harmony), {})

    def test_qwen3_trains_each_turn_with_its_reasoning(self) -> None:
        # Arrange
        tokenizer = cached_tokenizer("Qwen/Qwen3-4B")
        dataset = HFDataset.from_list([two_question_segment("reasoning_content")])
        chat_format = detect_chat_format(tokenizer)

        # Act
        split = traces_sft_split(dataset, "train split", tokenizer, chat_format, {})
        trained = [
            tokenizer.decode(target)
            for row in split
            for _, target in per_turn_targets(
                *encode_sft_row(tokenizer, row, "completion", chat_format, True)
            )
        ]

        # Assert
        assert len(trained) == 2
        assert "Two plus three." in trained[0]
        assert "Four plus four." in trained[1]

    def test_detects_a_template_that_drops_past_analysis(
        self, tokenizer: PreTrainedTokenizerBase
    ) -> None:
        dataset = HFDataset.from_list([thinking_segment()])
        harmony = template_tokenizer(DROPS_PAST_ANALYSIS_TEMPLATE, HARMONY_SPECIALS)

        assert template_rewrites_turns(dataset, harmony, HarmonyFormat(harmony), {})
        assert not template_rewrites_turns(
            HFDataset.from_list([helpdesk_segment()]),
            tokenizer,
            detect_chat_format(tokenizer),
            {},
        )

    @pytest.mark.parametrize(
        ("context", "kept"), [((), 2), ((0,), 1)], ids=["all", "context"]
    )
    def test_falls_back_to_per_turn_rows_and_warns_once(
        self, context: tuple[int, ...], kept: int
    ) -> None:
        # Arrange
        _warn_per_turn_fallback.cache_clear()
        harmony = template_tokenizer(DROPS_PAST_ANALYSIS_TEMPLATE, HARMONY_SPECIALS)
        dataset = HFDataset.from_list([as_context(thinking_segment(), *context)])
        chat_format = HarmonyFormat(harmony)

        # Act
        with pytest.warns(UserWarning, match="expanded and trained one assistant"):
            split = traces_sft_split(dataset, "train split", harmony, chat_format, {})
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            traces_sft_split(dataset, "test split", harmony, chat_format, {})

        # Assert
        assert len(split) == kept
        assert "prompt" in split.column_names
        ids, mask = encode_sft_row(
            harmony, split[kept - 1], "completion", chat_format, True
        )
        assert "Look the user up first." in harmony.decode(ids[: mask.index(True)])
        assert trained_text(harmony, ids, mask) == [
            (
                "<|channel|>analysis<|message|>Locked.<|end|><|start|>assistant"
                "<|channel|>final<|message|>Unlocked.<|return|>"
            )
        ]

    def test_gpt_oss_falls_back_to_per_turn_rows(self) -> None:
        # Arrange
        _warn_per_turn_fallback.cache_clear()
        tokenizer = cached_tokenizer("openai/gpt-oss-20b")
        dataset = HFDataset.from_list([thinking_segment()])
        chat_format = detect_chat_format(tokenizer)

        # Act
        with pytest.warns(UserWarning, match="expanded and trained one assistant"):
            split = traces_sft_split(dataset, "train split", tokenizer, chat_format, {})

        # Assert
        assert "prompt" in split.column_names
        ids, mask = encode_sft_row(tokenizer, split[1], "completion", chat_format, True)
        assert "Look the user up first." in tokenizer.decode(ids[: mask.index(True)])


class TestCheckTracesRenderSegments:
    @pytest.mark.parametrize(
        ("context", "columns"),
        [((0,), {}), ((), {"finish_reasons": ["length", "stop"]})],
        ids=["context", "length"],
    )
    def test_reasoning_on_a_turn_that_does_not_train_is_allowed(
        self, context: tuple[int, ...], columns: dict[str, Any]
    ) -> None:
        # Arrange
        tokenizer = template_tokenizer(THINKING_SWITCH_TEMPLATE)
        segment = helpdesk_segment()
        segment["messages"][2]["reasoning_content"] = "Look the user up."
        segment["messages"][4] = {"role": "assistant", "content": "Unlocked."}
        dataset = HFDataset.from_list([{**as_context(segment, *context), **columns}])

        # Act / Assert
        check_traces_render(dataset, tokenizer, None, {})

    def test_reasoning_on_a_trained_turn_is_rejected(self) -> None:
        tokenizer = template_tokenizer(THINKING_SWITCH_TEMPLATE)
        dataset = HFDataset.from_list([helpdesk_segment()])

        with pytest.raises(ValueError, match="closes the thinking block"):
            check_traces_render(dataset, tokenizer, None, {})

    def test_checks_every_trained_turn(self) -> None:
        harmony = template_tokenizer(DROPS_PAST_ANALYSIS_TEMPLATE, HARMONY_SPECIALS)
        dataset = HFDataset.from_list([thinking_segment()])

        with pytest.raises(
            ValueError,
            match="Row 1: SFT renders a trained turn or its context differently",
        ):
            check_traces_render(dataset, harmony, HarmonyFormat(harmony), {})

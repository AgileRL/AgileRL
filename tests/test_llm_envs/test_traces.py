# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Traces rows rendered for a rollout and an OPSD teacher, and trained by SFT on the completion only."""

from __future__ import annotations

import json
from typing import Any

import pytest

pytest.importorskip("datasets", reason="LLM dependencies not installed")

from datasets import Dataset as HFDataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from agilerl.llm_envs import DatasetEnv
from agilerl.llm_envs.chat_formats import detect_chat_format
from agilerl.llm_envs.dataset_env import (
    chat_messages,
    encode_chat_transcript,
    encode_sft_row,
)
from agilerl.llm_envs.traces import (
    check_traces_split,
    traces_prompt_text,
    traces_reference_text,
    traces_rollout_split,
    traces_sft_split,
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
        text = traces_prompt_text(
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

        text = traces_prompt_text(
            tokenizer, json.dumps(helpdesk_prompt()), [LOOKUP_TOOL], chat_format
        )

        assert text == traces_prompt_text(
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

        text = traces_prompt_text(reasoning_tokenizer, prompt, None, None)

        assert "<think>\ngreet back\n</think>\n\nhello<|im_end|>" in text

    def test_passes_chat_template_kwargs(
        self, reasoning_tokenizer: PreTrainedTokenizerBase
    ) -> None:
        reasoning_tokenizer.chat_template = "{{ greeting }}" + REASONING_TEMPLATE

        text = traces_prompt_text(
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
            traces_prompt_text(tokenizer, prompt, None, None)


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
                "reasoning": "Check the account first.",
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
            "Check the account first.\n\n<tool_call>\n"
            '{"name": "lookup_user", "arguments": {"email": "x@y.com"}}\n</tool_call>'
        )

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
    def test_accepts_arrow_rows(self) -> None:
        dataset = HFDataset.from_list([helpdesk_row(), helpdesk_row()])

        check_traces_split(dataset, "train split", rollout=True)

    def test_names_the_bad_row(self) -> None:
        bad = helpdesk_row()
        bad["prompt"][3]["tool_call_id"] = "c9"
        dataset = HFDataset.from_list([helpdesk_row(), bad])

        with pytest.raises(ValueError, match=r"row 2: .*'c9'"):
            check_traces_split(dataset, "train split", rollout=True)

    def test_reports_at_most_five_bad_rows(self) -> None:
        bad = helpdesk_row()
        bad["prompt"][3]["tool_call_id"] = "c9"
        dataset = HFDataset.from_list([bad] * 7)

        with pytest.raises(ValueError, match="spec") as raised:
            check_traces_split(dataset, "train split", rollout=True)

        assert str(raised.value).count("row ") == 5

    def test_names_the_field_of_a_bad_message(self) -> None:
        bad = helpdesk_row()
        bad["completion"][0]["role"] = "robot"
        dataset = HFDataset.from_list([bad])

        with pytest.raises(ValueError, match=r"row 1, completion\.0\.role: "):
            check_traces_split(dataset, "train split", rollout=True)

    def test_rollout_rejects_messages_rows(self) -> None:
        row = helpdesk_row()
        dataset = HFDataset.from_list(
            [{"messages": [*row["prompt"], *row["completion"]]}]
        )

        with pytest.raises(ValueError, match="reads 'prompt' and 'completion'"):
            check_traces_split(dataset, "train split", rollout=True)

        check_traces_split(dataset, "train split", rollout=False)


def json_text_row() -> dict[str, str]:
    """The helpdesk row as a snapshot ``calls.parquet`` stores it: JSON text cells."""
    return {key: json.dumps(value) for key, value in helpdesk_row().items()}


class TestTracesRolloutSplit:
    def test_reads_json_text_cells(
        self, reasoning_tokenizer: PreTrainedTokenizerBase
    ) -> None:
        dataset = HFDataset.from_list([json_text_row()])

        check_traces_split(dataset, "train split", rollout=True)
        split = traces_rollout_split(
            dataset, reasoning_tokenizer, None, None, "reference_trace"
        )

        assert split[0]["reference_trace"].endswith("I've sent an unlock link.")
        assert traces_prompt_text(
            reasoning_tokenizer, split[0]["prompt"], None, None
        ) == traces_prompt_text(reasoning_tokenizer, helpdesk_prompt(), None, None)

    def test_adds_an_answer_and_the_rendered_reference(
        self, reasoning_tokenizer: PreTrainedTokenizerBase
    ) -> None:
        dataset = HFDataset.from_list([helpdesk_row()])

        split = traces_rollout_split(
            dataset, reasoning_tokenizer, None, None, "reference_trace"
        )

        assert split[0]["answer"] is None
        assert split[0]["reference_trace"] == (
            "<think>\nThe account is locked.\n</think>\n\nI've sent an unlock link."
        )

    def test_a_teacher_model_without_a_context_key_ignores_the_completion(
        self, tokenizer: PreTrainedTokenizerBase
    ) -> None:
        dataset = HFDataset.from_list([{**helpdesk_row(), "answer": "unlocked"}])

        split = traces_rollout_split(dataset, tokenizer, None, None, None)

        assert split.column_names == ["prompt", "completion", "tools", "answer"]
        assert split[0]["answer"] == "unlocked"


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

    def test_sft_split_drops_messages_beside_prompt_completion(self) -> None:
        row = helpdesk_row()
        dataset = HFDataset.from_list([{**row, "messages": row["prompt"][:2]}])

        split = traces_sft_split(dataset, "train split")

        assert split.column_names == ["prompt", "completion", "tools"]

    def test_sft_split_keeps_messages_rows(self) -> None:
        row = helpdesk_row()
        dataset = HFDataset.from_list(
            [{"messages": [*row["prompt"], *row["completion"]]}]
        )

        split = traces_sft_split(dataset, "train split")

        assert split.column_names == ["messages"]

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

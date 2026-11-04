# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for :mod:`agilerl.llm_envs` (preference and SFT dataset envs, rollout env)."""

import json
from typing import Any

import pytest
import torch

pytest.importorskip("datasets", reason="LLM dependencies not installed")

from datasets import Dataset as HFDataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding

from agilerl.llm_envs import (
    DatasetEnv,
    apply_chat_template,
)
from agilerl.llm_envs.chat_formats import (
    Gemma4Format,
    Granite3Format,
    HarmonyFormat,
    detect_chat_format,
)
from agilerl.llm_envs.dataset_env import (
    _named_tool_results,
    _written_tool_calls,
    chat_messages,
    encode_chat_transcript,
    sft_eval_sample,
    sft_messages,
    sft_tools,
)
from agilerl.utils.chat_template import (
    DEFAULT_CHAT_TEMPLATE,
    resolve_chat_template_kwargs,
)
from tests import TINY_LLM_FIXTURE_PATH
from tests.helpers.rollout_doubles import RolloutEnvDoubleMixin
from tests.helpers.transcripts import assistant_runs


def DummyPreferenceDataset(num_samples: int) -> HFDataset:
    """Real HF preference dataset, so tests exercise the production code paths."""
    dataset = HFDataset.from_dict(
        {
            "prompt": [f"This is prompt {i}." for i in range(num_samples)],
            "chosen": [f"This is chosen {i}." for i in range(num_samples)],
            "rejected": [f"This is rejected {i}." for i in range(num_samples)],
        }
    )
    dataset.info.dataset_name = "dummy_dataset"
    return dataset


def DummySFTDataset(num_samples: int) -> HFDataset:
    """Real HF SFT dataset keyed by the default ``response_column`` ("response")."""
    dataset = HFDataset.from_dict(
        {
            "prompt": [f"This is prompt {i}." for i in range(num_samples)],
            "response": [f"This is response {i}." for i in range(num_samples)],
        }
    )
    dataset.info.dataset_name = "dummy_sft_dataset"
    return dataset


@pytest.fixture
def preference_dataset(num_samples):
    train_dataset = DummyPreferenceDataset(int(num_samples * 0.8))
    test_dataset = DummyPreferenceDataset(int(num_samples * 0.2))
    return train_dataset, test_dataset


@pytest.fixture
def sft_dataset(num_samples):
    train_dataset = DummySFTDataset(int(num_samples * 0.8))
    test_dataset = DummySFTDataset(int(num_samples * 0.2))
    return train_dataset, test_dataset


class TestDatasetEnvPreferenceInit:
    @pytest.mark.parametrize("num_samples", [20])
    def test_preference_init(
        self,
        preference_dataset,
        num_samples,
    ):
        train_dataset, test_dataset = preference_dataset
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        data_batch_size = 8
        env = DatasetEnv(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            tokenizer=tokenizer,
            objective="preference",
            data_batch_size_per_gpu=data_batch_size,
        )
        assert isinstance(env, DatasetEnv)
        assert env.name == "dummy_dataset"
        assert hasattr(env, "tokenizer")
        assert isinstance(env.train_dataloader, DataLoader)
        assert isinstance(env.test_dataloader, DataLoader)
        assert list(next(env.train_dataloader_iter).keys()) == [
            "prompt",
            "prompt_lengths",
            "chosen",
            "rejected",
            "chosen_input_ids",
            "chosen_attention_mask",
            "rejected_input_ids",
            "rejected_attention_mask",
        ]
        assert env.dataloader == env.train_dataloader_iter
        assert not hasattr(env, "reset_called")
        assert not env.evaluation_mode
        assert env.data_batch_size_per_gpu == data_batch_size

    def test_preference_max_context_length_error(self):
        train_dataset = HFDataset.from_dict(
            {
                "prompt": [
                    "This is a prompt that is longer than the max context length."
                ],
                "chosen": ["This is an answer."],
                "rejected": ["This is an answer."],
            },
        )
        test_dataset = HFDataset.from_dict(
            {
                "prompt": ["This is a normal length prompt"],
                "chosen": ["This is an answer."],
                "rejected": ["This is an answer."],
            },
        )
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        data_batch_size = 8
        with pytest.raises(
            ValueError,
            match=r"No samples left in the train dataset after filtering by the max context length constraint, use a larger max context length.",
        ):
            DatasetEnv(
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                tokenizer=tokenizer,
                objective="preference",
                data_batch_size_per_gpu=data_batch_size,
                max_context_length=5,
                min_completion_length=1,
            )

    def test_preference_max_context_length_warning(self):
        train_dataset = HFDataset.from_dict(
            {
                "prompt": [
                    "This is a prompt that is longer than the max context length. This prompt really is a lot longer than the other one.",
                    "This is a prompt that is shorter.",
                ],
                "chosen": ["This is an answer.", "This is an answer."],
                "rejected": ["This is an answer.", "This is an answer."],
            },
        )
        test_dataset = HFDataset.from_dict(
            {
                "prompt": ["This is a normal length prompt"],
                "chosen": ["This is an answer."],
                "rejected": ["This is an answer."],
            },
        )
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        data_batch_size = 8
        with pytest.warns(
            UserWarning,
            match=r"1 samples were filtered out of the train dataset due to the max context length constraint.",
        ):
            env = DatasetEnv(
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                tokenizer=tokenizer,
                objective="preference",
                data_batch_size_per_gpu=data_batch_size,
                max_context_length=10,
                min_completion_length=1,
            )
        assert len(env.train_dataloader) == 1
        assert len(env.test_dataloader) == 1

    def test_preference_init_missing_features(self):
        """A preference ``DatasetEnv`` raises when the dataset lacks required features."""
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        good_dataset = HFDataset.from_dict(
            {"prompt": ["p"], "chosen": ["c"], "rejected": ["r"]},
        )
        # Has "prompt" (so the column filter in __init__ works) but missing "chosen"/"rejected"
        bad_dataset = HFDataset.from_dict({"prompt": ["p"], "other": ["o"]})
        with pytest.raises(ValueError, match="must contain columns"):
            DatasetEnv(
                train_dataset=bad_dataset,
                test_dataset=good_dataset,
                tokenizer=tokenizer,
                objective="preference",
            )
        with pytest.raises(ValueError, match="must contain columns"):
            DatasetEnv(
                train_dataset=good_dataset,
                test_dataset=bad_dataset,
                tokenizer=tokenizer,
                objective="preference",
            )


class TestDatasetEnvShardValidation:
    @pytest.mark.parametrize(
        ("rank", "world_size", "match"),
        [(0, 0, "world_size must be"), (2, 2, "rank must be")],
    )
    def test_invalid_shard_geometry_is_rejected(self, rank, world_size, match):
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        with pytest.raises(ValueError, match=match):
            DatasetEnv(
                train_dataset=DummySFTDataset(4),
                test_dataset=DummySFTDataset(2),
                tokenizer=tokenizer,
                objective="sft",
                rank=rank,
                world_size=world_size,
            )


class TestDatasetEnvCollateEos:
    def test_preference_completions_end_with_eos(self):
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        env = DatasetEnv(
            train_dataset=DummyPreferenceDataset(4),
            test_dataset=DummyPreferenceDataset(2),
            tokenizer=tokenizer,
            objective="preference",
            data_batch_size_per_gpu=2,
        )
        batch = next(env.train_dataloader_iter)
        for key in ("chosen", "rejected"):
            ids = batch[f"{key}_input_ids"]
            masks = batch[f"{key}_attention_mask"]
            for row_ids, row_mask in zip(ids, masks, strict=True):
                last_real = row_ids[row_mask.bool()][-1]
                assert int(last_real) == tokenizer.eos_token_id


class TestDatasetEnvSplitValidation:
    @pytest.mark.parametrize("empty_split", ["train", "test"])
    def test_empty_split_is_rejected(self, empty_split):
        """Either split being empty is a build-time error, not a silent no-op."""
        populated = DummyPreferenceDataset(4)
        empty = DummyPreferenceDataset(0)
        train_dataset = empty if empty_split == "train" else populated
        test_dataset = empty if empty_split == "test" else populated
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)

        with pytest.raises(ValueError, match="each split needs at least one row"):
            DatasetEnv(
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                tokenizer=tokenizer,
                objective="preference",
            )


class TestDatasetEnvPreferenceReset:
    @pytest.mark.parametrize("num_samples", [20])
    def test_preference_reset(
        self,
        preference_dataset,
        num_samples,
    ):
        train_dataset, test_dataset = preference_dataset
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        data_batch_size = 8
        env = DatasetEnv(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            tokenizer=tokenizer,
            objective="preference",
            data_batch_size_per_gpu=data_batch_size,
        )
        prompts = env.reset()
        assert isinstance(prompts, dict)
        assert set(prompts.keys()) == {
            "prompt",
            "prompt_lengths",
            "chosen",
            "rejected",
            "chosen_input_ids",
            "chosen_attention_mask",
            "rejected_input_ids",
            "rejected_attention_mask",
        }
        assert len(prompts["prompt"]) == data_batch_size
        assert len(prompts["prompt_lengths"]) == data_batch_size
        assert len(prompts["chosen"]) == data_batch_size
        assert len(prompts["rejected"]) == data_batch_size
        assert len(prompts["chosen_input_ids"]) == data_batch_size
        assert len(prompts["chosen_attention_mask"]) == data_batch_size
        assert len(prompts["rejected_input_ids"]) == data_batch_size
        assert len(prompts["rejected_attention_mask"]) == data_batch_size
        assert isinstance(prompts["prompt"], list)
        assert isinstance(prompts["prompt"][0], str)
        assert isinstance(prompts["prompt_lengths"][0], int)
        assert isinstance(prompts["prompt_lengths"], list)
        assert isinstance(prompts["chosen"], list)
        assert isinstance(prompts["rejected"], list)
        assert isinstance(prompts["chosen_input_ids"], torch.Tensor)
        assert isinstance(prompts["chosen_attention_mask"], torch.Tensor)
        assert isinstance(prompts["rejected_input_ids"], torch.Tensor)
        assert isinstance(prompts["rejected_attention_mask"], torch.Tensor)

    @pytest.mark.parametrize("num_samples", [20])
    def test_preference_reset_reset_dataloaders_warning(
        self,
        preference_dataset,
        num_samples,
    ):
        train_dataset, test_dataset = preference_dataset
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        data_batch_size = 1
        env = DatasetEnv(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            tokenizer=tokenizer,
            objective="preference",
            data_batch_size_per_gpu=data_batch_size,
        )
        env.reset()
        env.reset()
        env.reset()
        prompts = env.reset(reset_dataloaders=True)
        assert env.num_epochs == 0
        assert len(prompts["prompt"]) == data_batch_size
        assert isinstance(prompts, dict)
        assert set(prompts.keys()) == {
            "prompt",
            "prompt_lengths",
            "chosen",
            "rejected",
            "chosen_input_ids",
            "chosen_attention_mask",
            "rejected_input_ids",
            "rejected_attention_mask",
        }
        assert len(prompts["prompt"]) == data_batch_size
        assert len(prompts["prompt_lengths"]) == data_batch_size
        assert len(prompts["chosen"]) == data_batch_size
        assert len(prompts["rejected"]) == data_batch_size
        assert len(prompts["chosen_input_ids"]) == data_batch_size
        assert len(prompts["chosen_attention_mask"]) == data_batch_size
        assert len(prompts["rejected_input_ids"]) == data_batch_size
        assert len(prompts["rejected_attention_mask"]) == data_batch_size
        assert isinstance(prompts["prompt"], list)
        assert isinstance(prompts["prompt"][0], str)
        assert isinstance(prompts["prompt_lengths"][0], int)
        assert isinstance(prompts["prompt_lengths"], list)
        assert isinstance(prompts["chosen"], list)
        assert isinstance(prompts["rejected"], list)
        assert isinstance(prompts["chosen_input_ids"], torch.Tensor)
        assert isinstance(prompts["chosen_attention_mask"], torch.Tensor)
        assert isinstance(prompts["rejected_input_ids"], torch.Tensor)
        assert isinstance(prompts["rejected_attention_mask"], torch.Tensor)

    @pytest.mark.parametrize("num_samples", [20])
    def test_preference_reset_num_epochs(
        self,
        preference_dataset,
        num_samples,
    ):
        train_dataset, test_dataset = preference_dataset
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        data_batch_size = 1
        env = DatasetEnv(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            tokenizer=tokenizer,
            objective="preference",
            data_batch_size_per_gpu=data_batch_size,
        )
        while env.num_epochs == 0:
            env.reset()
        assert env.num_epochs == 1

    @pytest.mark.parametrize("num_samples", [20])
    def test_preference_reset_drives_epoch_rollover(
        self,
        preference_dataset,
        num_samples,
    ):
        """Epoch rollover is driven purely by ``reset``: the (N+1)th fetch rolls."""
        train_dataset, test_dataset = preference_dataset
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        env = DatasetEnv(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            tokenizer=tokenizer,
            objective="preference",
            data_batch_size_per_gpu=1,
        )
        batches_per_epoch = len(env.train_dataloader)
        for _ in range(batches_per_epoch):
            env.reset()
        assert env.num_epochs == 0  # full epoch consumed, not yet rolled
        env.reset()
        assert env.num_epochs == 1  # (N+1)th fetch triggers rollover


class TestDatasetEnvPreferenceCreateCollateFn:
    def test_preference_collate_max_context_length_branch(self):
        """Exercise ``max_context_length is not None`` tokenisation in a preference ``DatasetEnv``."""
        tokenizer = AutoTokenizer.from_pretrained(
            TINY_LLM_FIXTURE_PATH,
        )
        train_ds = HFDataset.from_dict(
            {
                "prompt": ["hello"],
                "chosen": ["yes"],
                "rejected": ["no"],
            },
        )
        test_ds = HFDataset.from_dict(
            {
                "prompt": ["hello"],
                "chosen": ["yes"],
                "rejected": ["no"],
            },
        )
        env = DatasetEnv(
            train_dataset=train_ds,
            test_dataset=test_ds,
            tokenizer=tokenizer,
            objective="preference",
            data_batch_size_per_gpu=1,
            max_context_length=64,
        )
        collate = env.create_collate_fn(tokenizer)
        batch = [
            {"prompt": "hello", "chosen": "yes please", "rejected": "no thanks"},
        ]
        out = collate(batch)
        assert "chosen_input_ids" in out
        assert out["chosen_input_ids"].shape[1] == 64


class TestDatasetEnvSFTInit:
    def test_sft_max_context_length_warning(self):
        train_dataset = HFDataset.from_dict(
            {
                "prompt": ["word " * 200, "short"],
                "response": ["a", "b"],
            },
        )
        test_dataset = HFDataset.from_dict(
            {
                "prompt": ["ok"],
                "response": ["a"],
            },
        )
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        with pytest.warns(
            UserWarning,
            match=r"1 samples were filtered out of the train dataset due to the max context length constraint.",
        ):
            env = DatasetEnv(
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                tokenizer=tokenizer,
                objective="sft",
                data_batch_size_per_gpu=8,
                max_context_length=64,
            )
        assert len(env.train_dataloader) == 1

    def test_sft_init_missing_features(self):
        """An SFT ``DatasetEnv`` raises when the dataset lacks required features."""
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        good_dataset = HFDataset.from_dict({"prompt": ["p"], "response": ["r"]})
        # Has "prompt" (so the column filter in __init__ works) but missing "response"
        bad_dataset = HFDataset.from_dict({"prompt": ["p"], "other": ["o"]})
        with pytest.raises(ValueError, match="must contain"):
            DatasetEnv(
                train_dataset=bad_dataset,
                test_dataset=good_dataset,
                tokenizer=tokenizer,
                objective="sft",
            )
        with pytest.raises(ValueError, match="must contain"):
            DatasetEnv(
                train_dataset=good_dataset,
                test_dataset=bad_dataset,
                tokenizer=tokenizer,
                objective="sft",
            )

    def test_sft_init_tokenizer_without_pad_token(self):
        # Arrange
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        tokenizer.pad_token = None
        dataset = HFDataset.from_dict({"prompt": ["p"], "response": ["r"]})

        # Act / Assert
        with pytest.raises(ValueError, match="tokenizer has no pad token"):
            DatasetEnv(
                train_dataset=dataset,
                test_dataset=dataset,
                tokenizer=tokenizer,
                objective="sft",
            )


class TestDatasetEnvSFTStep:
    @pytest.mark.parametrize("num_samples", [20])
    def test_sft_num_epochs_increment(
        self,
        sft_dataset,
        num_samples,
    ):
        train_dataset, test_dataset = sft_dataset
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        env = DatasetEnv(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            tokenizer=tokenizer,
            objective="sft",
            data_batch_size_per_gpu=1,
        )
        while env.num_epochs == 0:
            env.reset()
        assert env.num_epochs == 1


class TestDatasetEnvSFTReset:
    @pytest.mark.parametrize("num_samples", [20])
    def test_sft_reset_returns_batch(
        self,
        sft_dataset,
        num_samples,
    ):
        train_dataset, test_dataset = sft_dataset
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        data_batch_size = 8
        env = DatasetEnv(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            tokenizer=tokenizer,
            objective="sft",
            data_batch_size_per_gpu=data_batch_size,
        )
        batch = env.reset()
        assert set(batch.keys()) == {"input_ids", "attention_mask", "loss_mask"}
        assert batch["input_ids"].shape[0] == data_batch_size

    @pytest.mark.parametrize("num_samples", [20])
    def test_sft_reset_dataloaders_warning(
        self,
        sft_dataset,
        num_samples,
    ):
        train_dataset, test_dataset = sft_dataset
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        env = DatasetEnv(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            tokenizer=tokenizer,
            objective="sft",
            data_batch_size_per_gpu=1,
        )
        env.reset()
        env.reset()
        env.reset(reset_dataloaders=True)
        assert env.num_epochs == 0


WEATHER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        },
    }
]


def weather_conversation() -> list[dict]:
    """System, user, tool-calling assistant, tool result, final assistant answer."""
    return [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Weather in Paris?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "Paris"}',
                    },
                }
            ],
        },
        {"role": "tool", "content": "sunny"},
        {"role": "assistant", "content": "Sunny."},
    ]


class TestEncodeChatTranscript:
    def test_matches_template_render_and_masks_assistant_turns(self):
        # Arrange
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        messages = chat_messages(weather_conversation())

        # Act
        ids, mask = encode_chat_transcript(
            tokenizer, messages, WEATHER_TOOLS, detect_chat_format(tokenizer)
        )

        # Assert
        rendered = tokenizer.apply_chat_template(
            messages, tools=WEATHER_TOOLS, tokenize=False
        )
        assert tokenizer.decode(ids) == rendered.removesuffix("\n")
        trained = tokenizer.decode([i for i, m in zip(ids, mask, strict=True) if m])
        assert trained == (
            '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n'
            "</tool_call><|im_end|>Sunny.<|im_end|>"
        )

    def test_messages_after_last_assistant_turn_are_dropped(self):
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        messages = chat_messages(weather_conversation())

        full_ids, _ = encode_chat_transcript(
            tokenizer, messages, WEATHER_TOOLS, detect_chat_format(tokenizer)
        )
        trailing_ids, _ = encode_chat_transcript(
            tokenizer,
            [*messages, {"role": "user", "content": "Thanks"}],
            WEATHER_TOOLS,
            detect_chat_format(tokenizer),
        )

        assert trailing_ids == full_ids

    @pytest.mark.parametrize(
        ("messages", "match"),
        [
            ([{"role": "user", "content": "hi"}], "at least one assistant"),
            (
                [{"role": "assistant", "content": "hi"}],
                "before its first assistant turn",
            ),
            (
                [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "a"},
                    {"role": "assistant", "content": "b"},
                ],
                "back-to-back assistant turns",
            ),
        ],
    )
    def test_rejects_conversations_without_a_trainable_layout(self, messages, match):
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)

        with pytest.raises(ValueError, match=match):
            encode_chat_transcript(tokenizer, messages, None, None)


GEMMA_STYLE_TEMPLATE = (
    "{% for m in messages %}"
    "{% if m.role == 'user' %}<|turn>user\n{{ m.content }}<turn|>\n"
    "{% elif m.role == 'assistant' %}"
    "{% if loop.previtem is undefined or loop.previtem.role != 'tool' %}<|turn>model\n{% endif %}"
    "{% if m.tool_calls %}"
    "{% for c in m.tool_calls %}<|tool_call>call:{{ c.function.name }}{{ '{' }}"
    '{% for k, v in c.function.arguments.items() %}{{ k }}:<|"|>{{ v }}<|"|>{% endfor %}'
    "{{ '}' }}<tool_call|>{% endfor %}"
    "<|tool_response>"
    "{% else %}{{ m.content }}<turn|>\n{% endif %}"
    "{% elif m.role == 'tool' %}"
    "response:{{ loop.previtem.tool_calls[0].function.name }}"
    '{value:<|"|>{{ m.content }}<|"|>}<tool_response|>'
    "{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}<|turn>model\n{% endif %}"
)
"""Gemma 4 layout: tool results render inside the model turn that called the tool."""

HARMONY_STYLE_TEMPLATE = (
    "{% for m in messages %}"
    "{% if m.role == 'user' %}<|start|>user<|message|>{{ m.content }}<|end|>"
    "{% elif m.role == 'assistant' %}<|start|>assistant"
    "{% if m.tool_calls %}"
    " to=functions.{{ m.tool_calls[0].function.name }}<|channel|>commentary json"
    "<|message|>{{ m.tool_calls[0].function.arguments | tojson }}<|call|>"
    "{% else %}<|channel|>final<|message|>{{ m.content }}"
    "{% if loop.last %}<|return|>{% else %}<|end|>{% endif %}{% endif %}"
    "{% elif m.role == 'tool' %}"
    "<|start|>functions.{{ loop.previtem.tool_calls[0].function.name }} to=assistant"
    "<|channel|>commentary<|message|>{{ m.content | tojson }}<|end|>"
    "{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}<|start|>assistant{% endif %}"
)
"""gpt-oss layout: a tool result is its own block, addressed from the function it answers."""

GEMMA_STYLE_SPECIALS = [
    "<|turn>",
    "<turn|>",
    "<|tool_call>",
    "<tool_call|>",
    "<|tool_response>",
    "<tool_response|>",
]
HARMONY_STYLE_SPECIALS = [
    "<|start|>",
    "<|message|>",
    "<|channel|>",
    "<|call|>",
    "<|end|>",
    "<|return|>",
]


def in_turn_tokenizer(template: str, specials: list[str]) -> Any:
    """Tiny tokenizer with ``template`` and its turn markers as special tokens."""
    tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
    tokenizer.add_tokens(specials, special_tokens=True)
    tokenizer.chat_template = template
    return tokenizer


def trained_text(tokenizer: Any, ids: list[int], mask: list[bool]) -> list[str]:
    """Decoded text of each run of assistant tokens."""
    return [tokenizer.decode(ids[start:end]) for start, end in assistant_runs(mask)]


HI_HELLO = [
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello"},
]


class TestEncodeChatTranscriptInTurnToolResults:
    @pytest.mark.parametrize(
        ("template", "specials", "chat_format", "expected_turns"),
        [
            pytest.param(
                GEMMA_STYLE_TEMPLATE,
                GEMMA_STYLE_SPECIALS,
                Gemma4Format,
                [
                    '<|tool_call>call:get_weather{city:<|"|>Paris<|"|>}<tool_call|><|tool_response>',
                    "Sunny.<turn|>",
                ],
                id="gemma",
            ),
            pytest.param(
                HARMONY_STYLE_TEMPLATE,
                HARMONY_STYLE_SPECIALS,
                HarmonyFormat,
                [
                    (
                        " to=functions.get_weather<|channel|>commentary json<|message|>"
                        '{"city": "Paris"}<|call|>'
                    ),
                    "<|channel|>final<|message|>Sunny.<|return|>",
                ],
                id="harmony",
            ),
        ],
    )
    def test_layout_matches_the_template(
        self, template, specials, chat_format, expected_turns
    ):
        # Arrange
        tokenizer = in_turn_tokenizer(template, specials)
        messages = chat_messages(weather_conversation())

        # Act
        ids, mask = encode_chat_transcript(
            tokenizer, messages, WEATHER_TOOLS, chat_format(tokenizer)
        )

        # Assert
        rendered = tokenizer.apply_chat_template(
            messages, tools=WEATHER_TOOLS, tokenize=False
        )
        assert tokenizer.decode(ids) == rendered.removesuffix("\n")
        assert trained_text(tokenizer, ids, mask) == expected_turns


PREFILL_TEMPLATE = (
    "{% for m in messages %}"
    "{% if m.role == 'user' %}<|im_start|>user\n{{ m.content }}<|im_end|>\n"
    "{% else %}<|im_start|>assistant\n<think></think>{{ m.content }}<|im_end|>\n"
    "{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n<think>\n{% endif %}"
)
"""Nemotron 3 layout: the generation prompt prefills ``<think>\\n``; history writes ``<think></think>``."""

GRANITE3_STYLE_TEMPLATE = (
    "{% for m in messages %}"
    "<|start_of_role|>{{ m.role }}<|end_of_role|>{{ m.content }}<|end_of_text|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|start_of_role|>assistant<|end_of_role|>{% endif %}"
)
"""Granite 3 layout: the template writes no structured ``tool_calls``."""


class TestEncodeChatTranscriptTemplateQuirks:
    def test_turn_starts_after_the_reasoning_prefill_the_render_drops(self):
        # Arrange
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        tokenizer.chat_template = PREFILL_TEMPLATE

        # Act
        ids, mask = encode_chat_transcript(tokenizer, HI_HELLO, None, None)

        # Assert
        assert tokenizer.decode(ids).endswith(
            "<|im_start|>assistant\n<think>\n</think>Hello<|im_end|>"
        )
        assert trained_text(tokenizer, ids, mask) == ["</think>Hello<|im_end|>"]

    def test_chat_format_writes_calls_the_template_drops(self):
        # Arrange
        tokenizer = in_turn_tokenizer(
            GRANITE3_STYLE_TEMPLATE,
            [
                "<|start_of_role|>",
                "<|end_of_role|>",
                "<|end_of_text|>",
                "<|tool_call|>",
            ],
        )
        messages = chat_messages(weather_conversation()[1:3])

        # Act
        ids, mask = encode_chat_transcript(
            tokenizer, messages, None, Granite3Format(tokenizer)
        )

        # Assert
        assert trained_text(tokenizer, ids, mask) == [
            (
                '<|tool_call|>[{"name": "get_weather", "arguments": {"city": "Paris"}}]'
                "<|end_of_text|>"
            )
        ]


class TestEncodeChatTranscriptLayouts:
    def test_template_without_an_end_of_turn_token_ends_the_turn_with_eos(self):
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

        ids, mask = encode_chat_transcript(tokenizer, HI_HELLO, None, None)

        assert (
            tokenizer.decode(ids)
            == f"User: Hi\n\nAssistant: Hello\n\n{tokenizer.eos_token}"
        )
        assert trained_text(tokenizer, ids, mask) == [f"Hello\n\n{tokenizer.eos_token}"]

    def test_turn_that_rewrites_the_generation_prompt_header_is_rejected(self):
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        tokenizer.chat_template = (
            "{% for m in messages %}<|im_start|>{{ 'bot' if m.role == 'assistant' "
            "else m.role }}\n{{ m.content }}<|im_end|>{% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant<|im_end|>{% endif %}"
        )

        with pytest.raises(ValueError, match="does not extend its generation prompt"):
            encode_chat_transcript(tokenizer, HI_HELLO, None, None)

    def test_render_that_is_not_text_is_rejected(self):
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        tokenizer.apply_chat_template = lambda *_args, **_kwargs: [1, 2]

        with pytest.raises(TypeError, match="returned list, expected str"):
            encode_chat_transcript(tokenizer, HI_HELLO, None, None)

    def test_feedback_role_the_template_cannot_frame_is_rejected(self):
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        tokenizer.chat_template = (
            "{% for m in messages %}"
            "{% if m.role == 'system' and not loop.first %}"
            "{{ raise_exception('no mid-conversation system message') }}{% endif %}"
            "<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
        )
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
            {"role": "system", "content": "Be brief."},
            {"role": "assistant", "content": "Ok"},
        ]

        with pytest.raises(ValueError, match=r"cannot render a \('system',\) turn"):
            encode_chat_transcript(tokenizer, messages, None, None)


class TestEncodeChatTranscriptToolCallCheck:
    def test_tool_calls_without_a_chat_format_are_rejected(self):
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)

        with pytest.raises(
            ValueError, match=r"No chat format parses tool calls.*get_weather"
        ):
            encode_chat_transcript(
                tokenizer, chat_messages(weather_conversation()), WEATHER_TOOLS, None
            )

    def test_tool_call_text_without_tool_calls_is_rejected(self):
        # Arrange: the call is written into content, so the trace has no tool_calls.
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        call = '<tool_call>\n{"name": "get_weather", "arguments": {}}\n</tool_call>'
        messages = [
            {"role": "user", "content": "Weather?"},
            {"role": "assistant", "content": call},
        ]

        # Act / Assert
        with pytest.raises(
            ValueError,
            match=r"parses the rendered turn as calls \['get_weather'\].*message calls \[\]",
        ):
            encode_chat_transcript(
                tokenizer, messages, None, detect_chat_format(tokenizer)
            )


class TestChatMessages:
    def test_dict_tool_call_arguments_are_kept(self):
        row = [
            {"role": "user", "content": "Hi"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "f", "arguments": {"x": 1}}}],
            },
        ]

        assert chat_messages(row)[1]["tool_calls"] == [
            {"function": {"name": "f", "arguments": {"x": 1}}}
        ]

    def test_decodes_json_rows_and_drops_null_fields(self):
        row = json.dumps(
            [
                {"role": "user", "content": "q", "tool_calls": None},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": None,
                            "type": "function",
                            "function": {"name": "f", "arguments": '{"x": 1}'},
                        }
                    ],
                },
            ]
        )

        messages = chat_messages(row)

        assert messages == [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "f", "arguments": {"x": 1}},
                    }
                ],
            },
        ]


class TestSftMessages:
    def test_string_prompt_and_response_are_one_user_and_one_assistant_turn(self):
        row = {"prompt": "Hi", "target": "Hello"}

        assert sft_messages(row, "target") == [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]

    def test_message_list_prompt_is_kept(self):
        prompt = [
            {"role": "system", "content": "Be brief.", "name": None},
            {"role": "user", "content": "Hi"},
        ]

        assert sft_messages({"prompt": prompt, "response": "Yo"}, "response") == [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Yo"},
        ]

    def test_messages_column_wins(self):
        row = {"messages": weather_conversation(), "prompt": "ignored"}

        assert sft_messages(row, "response") == chat_messages(weather_conversation())


class TestDatasetEnvSFTCollate:
    @pytest.mark.parametrize("response_column", ["response", "chosen"])
    def test_prompt_rows_train_the_assistant_turn_of_their_transcript(
        self, response_column
    ):
        # Arrange
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        dataset = HFDataset.from_dict({"prompt": ["Hi"], response_column: ["Hello"]})
        env = DatasetEnv(
            train_dataset=dataset,
            test_dataset=dataset,
            tokenizer=tokenizer,
            objective="sft",
            data_batch_size_per_gpu=1,
            response_column=response_column,
        )
        ids, mask = encode_chat_transcript(
            tokenizer,
            HI_HELLO,
            None,
            detect_chat_format(tokenizer),
        )

        # Act
        batch = env.reset()

        # Assert
        assert batch["input_ids"][0].tolist() == ids
        assert batch["loss_mask"][0].tolist() == mask
        trained = tokenizer.decode(batch["input_ids"][0][batch["loss_mask"][0]])
        assert trained == "Hello<|im_end|>"

    def test_batches_conversations_with_assistant_loss_mask(self):
        # Arrange
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        conversation = weather_conversation()
        dataset = HFDataset.from_dict(
            {
                "messages": [conversation, conversation[:3]],
                "tools": [json.dumps(WEATHER_TOOLS)] * 2,
            }
        )
        expected = [
            encode_chat_transcript(
                tokenizer,
                chat_messages(rows),
                WEATHER_TOOLS,
                detect_chat_format(tokenizer),
            )
            for rows in (conversation, conversation[:3])
        ]
        env = DatasetEnv(
            train_dataset=dataset,
            test_dataset=dataset,
            tokenizer=tokenizer,
            objective="sft",
            data_batch_size_per_gpu=2,
        )

        # Act
        batch = env.create_collate_fn(tokenizer)([dataset[0], dataset[1]])

        # Assert
        assert set(batch) == {"input_ids", "attention_mask", "loss_mask"}
        for row, (ids, mask) in enumerate(expected):
            length = int(batch["attention_mask"][row].sum())
            assert batch["input_ids"][row, :length].tolist() == ids
            assert batch["loss_mask"][row, :length].tolist() == mask
            assert not batch["loss_mask"][row, length:].any()

    def test_cuts_at_the_last_complete_assistant_turn_that_fits(self):
        # Arrange
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        conversation = weather_conversation()
        ids, mask = encode_chat_transcript(
            tokenizer,
            chat_messages(conversation),
            None,
            detect_chat_format(tokenizer),
        )
        first_turn_end = assistant_runs(mask)[0][1]
        dataset = HFDataset.from_dict({"messages": [conversation]})
        env = DatasetEnv(
            train_dataset=dataset,
            test_dataset=dataset,
            tokenizer=tokenizer,
            objective="sft",
            data_batch_size_per_gpu=1,
            max_context_length=len(ids) - 1,
        )

        # Act
        batch = env.reset()

        # Assert
        assert batch["input_ids"][0].tolist() == ids[:first_turn_end]
        assert batch["loss_mask"][0].tolist() == mask[:first_turn_end]

    def test_collate_rejects_a_row_with_no_turn_inside_the_context(self):
        # Arrange
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        dataset = HFDataset.from_dict({"prompt": ["Hi"], "response": ["Hello"]})
        env = DatasetEnv(
            train_dataset=dataset,
            test_dataset=dataset,
            tokenizer=tokenizer,
            objective="sft",
            data_batch_size_per_gpu=1,
        )
        env.max_context_length = 1

        # Act / Assert
        with pytest.raises(ValueError, match="no complete assistant turn"):
            env.create_collate_fn(tokenizer)([dataset[0]])


def _sft_env(train_rows: dict, tokenizer: Any, **kwargs: Any) -> DatasetEnv:
    """SFT env over ``train_rows`` with one valid held-out row."""
    return DatasetEnv(
        train_dataset=HFDataset.from_dict(train_rows),
        test_dataset=HFDataset.from_dict({"messages": [HI_HELLO]}),
        tokenizer=tokenizer,
        objective="sft",
        data_batch_size_per_gpu=1,
        **kwargs,
    )


class TestDatasetEnvSFTRowSelection:
    def test_row_whose_first_turn_does_not_fit_is_dropped(self):
        # Arrange
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        ids, _ = encode_chat_transcript(tokenizer, HI_HELLO, None, None)
        long_row = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello " * 200},
        ]

        # Act
        with pytest.warns(UserWarning, match="1 samples were filtered out"):
            env = _sft_env(
                {"messages": [HI_HELLO, long_row]},
                tokenizer,
                max_context_length=len(ids),
            )

        # Assert
        assert env.dataset_size["train"] == 1
        assert env.reset()["loss_mask"].any()

    def test_min_completion_length_reserves_room_after_the_prompt(self):
        # Arrange
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        ids, mask = encode_chat_transcript(tokenizer, HI_HELLO, None, None)
        long_prompt = [
            {"role": "user", "content": "Hi " * 200},
            {"role": "assistant", "content": "Hello"},
        ]

        # Act
        with pytest.warns(UserWarning, match="1 samples were filtered out"):
            env = _sft_env(
                {"messages": [HI_HELLO, long_prompt]},
                tokenizer,
                max_context_length=len(ids),
                min_completion_length=len(ids) - mask.index(True),
            )

        # Assert
        assert env.dataset_size["train"] == 1

    def test_no_row_fitting_raises(self):
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)

        with pytest.raises(ValueError, match="No samples left in the train dataset"):
            _sft_env({"messages": [HI_HELLO]}, tokenizer, max_context_length=1)

    @pytest.mark.parametrize(
        ("bad_row", "reason"),
        [
            pytest.param(
                [
                    *weather_conversation()[:3],
                    {"role": "tool", "content": None},
                    {"role": "assistant", "content": "Sunny."},
                ],
                "needs string content, got NoneType",
                id="null-tool-content",
            ),
            pytest.param(
                [
                    {"role": "user", "content": "q"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "f", "arguments": "{broken"}}
                        ],
                    },
                ],
                "arguments of tool call 'f' is not valid JSON",
                id="bad-json-arguments",
            ),
            pytest.param(
                [
                    {"role": "user", "content": "q"},
                    {
                        "role": "assistant",
                        "content": '<tool_call>{"name": "f", "arguments": {}}</tool_call>',
                    },
                ],
                "parses the rendered turn as calls",
                id="text-quoting-tool-syntax",
            ),
        ],
    )
    def test_invalid_row_is_dropped_with_its_reason(self, bad_row, reason):
        # Arrange
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        rows = [json.dumps(HI_HELLO), json.dumps(bad_row)]

        # Act
        with pytest.warns(UserWarning, match=f"1 invalid samples.*{reason}"):
            env = _sft_env({"messages": rows}, tokenizer)

        # Assert
        assert env.dataset_size["train"] == 1
        assert tokenizer.decode(env.reset()["input_ids"][0]).endswith("Hello<|im_end|>")

    def test_list_content_between_turns_gets_a_clear_error(self):
        # Arrange
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        messages = [
            *chat_messages(weather_conversation()[:3]),
            {"role": "tool", "content": [{"type": "text", "text": "sunny"}]},
            {"role": "assistant", "content": "Sunny."},
        ]

        # Act / Assert
        with pytest.raises(ValueError, match="needs string content, got list"):
            encode_chat_transcript(
                tokenizer, messages, None, detect_chat_format(tokenizer)
            )

    def test_every_row_invalid_raises_with_the_first_error(self):
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)

        with pytest.raises(ValueError, match=r"No valid samples.*not valid JSON"):
            _sft_env({"messages": ["{broken"]}, tokenizer)


class TestArrowFilledNoneKeys:
    def test_struct_arguments_lose_the_keys_arrow_filled_in(self):
        # Arrange
        rows = [
            [
                {"role": "user", "content": "q"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"function": {"name": "f", "arguments": arguments}}],
                },
            ]
            for arguments in ({"city": "Paris"}, {"unit": "C"})
        ]
        dataset = HFDataset.from_dict({"messages": rows})

        # Act
        messages = chat_messages(dataset[0]["messages"])

        # Assert
        assert dataset[0]["messages"][1]["tool_calls"][0]["function"]["arguments"] == {
            "city": "Paris",
            "unit": None,
        }
        assert messages[1]["tool_calls"][0]["function"]["arguments"] == {
            "city": "Paris"
        }

    def test_nested_struct_arguments_keep_none_list_items(self):
        row = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "f",
                            "arguments": {"a": {"b": None, "c": [1, None]}},
                        }
                    }
                ],
            }
        ]

        call = chat_messages(row)[0]["tool_calls"][0]

        assert call["function"]["arguments"] == {"a": {"c": [1, None]}}

    @pytest.mark.parametrize(
        "row",
        [
            pytest.param(
                [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {"function": {"name": "f", "arguments": '{"x": null}'}}
                        ],
                    }
                ],
                id="json-string-arguments",
            ),
            pytest.param(
                json.dumps(
                    [
                        {
                            "role": "assistant",
                            "tool_calls": [
                                {"function": {"name": "f", "arguments": {"x": None}}}
                            ],
                        }
                    ]
                ),
                id="json-string-row",
            ),
        ],
    )
    def test_nulls_written_in_json_are_kept(self, row):
        call = chat_messages(row)[0]["tool_calls"][0]

        assert call["function"]["arguments"] == {"x": None}

    def test_call_without_arguments_stays_without_them(self):
        row = [{"role": "assistant", "tool_calls": [{"function": {"name": "f"}}]}]

        call = chat_messages(row)[0]["tool_calls"][0]

        assert call == {"function": {"name": "f"}}

    def test_non_object_arguments_are_rejected(self):
        row = [
            {
                "role": "assistant",
                "tool_calls": [{"function": {"name": "f", "arguments": "[1]"}}],
            }
        ]

        with pytest.raises(ValueError, match="'f' must be an object, got list"):
            chat_messages(row)

    def test_struct_tool_schemas_lose_the_keys_arrow_filled_in(self):
        # Arrange
        schemas = [
            [{"function": {"name": "f", "parameters": {"properties": properties}}}]
            for properties in ({"a": {"type": "string"}}, {"b": {"type": "integer"}})
        ]
        dataset = HFDataset.from_dict({"tools": schemas})

        # Act
        tools = sft_tools(dataset[0])

        # Assert
        assert tools == [schemas[0][0]]

    def test_missing_tools_are_none_and_json_tools_are_decoded(self):
        assert sft_tools({"tools": None}) is None
        assert sft_tools({}) is None
        assert sft_tools({"tools": '[{"default": null}]'}) == [{"default": None}]

    def test_bad_json_tools_get_a_clear_error(self):
        with pytest.raises(ValueError, match="tools column is not valid JSON"):
            sft_tools({"tools": "{broken"})


class TestNamedToolResults:
    @staticmethod
    def _assistant() -> dict:
        return {
            "role": "assistant",
            "tool_calls": [
                {"id": "a", "function": {"name": "first"}},
                {"id": "b", "function": {"name": "second"}},
            ],
        }

    def test_out_of_order_results_are_named_by_tool_call_id(self):
        results = [
            {"role": "tool", "tool_call_id": "b", "content": "2"},
            {"role": "tool", "tool_call_id": "a", "content": "1"},
        ]

        named = _named_tool_results(self._assistant(), results)

        assert [message["name"] for message in named] == ["second", "first"]

    def test_unnamed_result_takes_the_call_a_named_result_left(self):
        results = [
            {"role": "tool", "content": "2"},
            {"role": "tool", "name": "first", "content": "1"},
        ]

        named = _named_tool_results(self._assistant(), results)

        assert [message["name"] for message in named] == ["second", "first"]

    def test_unnamed_result_takes_the_call_an_id_result_left(self):
        results = [
            {"role": "tool", "content": "2"},
            {"role": "tool", "tool_call_id": "a", "content": "1"},
        ]

        named = _named_tool_results(self._assistant(), results)

        assert [message["name"] for message in named] == ["second", "first"]

    def test_results_without_ids_are_named_in_call_order(self):
        results = [
            {"role": "tool", "content": "1"},
            {"role": "user", "content": "more"},
            {"role": "tool", "content": "2"},
        ]

        named = _named_tool_results(self._assistant(), results)

        assert named[0]["name"] == "first"
        assert "name" not in named[1]
        assert named[2]["name"] == "second"

    def test_ids_are_ignored_when_the_calls_have_none(self):
        assistant = {
            "role": "assistant",
            "tool_calls": [{"function": {"name": "only"}}],
        }
        results = [{"role": "tool", "tool_call_id": "x", "content": "1"}]

        assert _named_tool_results(assistant, results)[0]["name"] == "only"

    def test_unknown_tool_call_id_is_rejected(self):
        results = [{"role": "tool", "tool_call_id": "z", "content": "1"}]

        with pytest.raises(ValueError, match="tool_call_id 'z' is not one of"):
            _named_tool_results(self._assistant(), results)


class TestWrittenToolCalls:
    def test_call_without_a_name_is_rejected(self):
        tokenizer = in_turn_tokenizer(GEMMA_STYLE_TEMPLATE, GEMMA_STYLE_SPECIALS)
        message = {"role": "assistant", "tool_calls": [{"function": {"arguments": {}}}]}

        with pytest.raises(ValueError, match="tool call has no name"):
            _written_tool_calls(message, Gemma4Format(tokenizer))

    def test_list_content_with_written_calls_is_rejected(self):
        tokenizer = in_turn_tokenizer(
            GRANITE3_STYLE_TEMPLATE,
            [
                "<|start_of_role|>",
                "<|end_of_role|>",
                "<|end_of_text|>",
                "<|tool_call|>",
            ],
        )
        message = {
            "role": "assistant",
            "content": [{"type": "text", "text": "x"}],
            "tool_calls": [{"function": {"name": "f", "arguments": {}}}],
        }

        with pytest.raises(ValueError, match="needs string content, got list"):
            _written_tool_calls(message, Granite3Format(tokenizer))


class TestSftEvalSample:
    def test_prompt_is_the_rendered_context_and_chosen_the_first_turn(self):
        # Arrange
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        item = {
            "messages": [
                *HI_HELLO,
                {"role": "user", "content": "Bye"},
                {"role": "assistant", "content": "Goodbye"},
            ]
        }

        # Act
        prompt, chosen = sft_eval_sample(tokenizer, item)

        # Assert
        assert prompt == tokenizer.apply_chat_template(
            [HI_HELLO[0]], tokenize=False, add_generation_prompt=True
        )
        assert chosen == "Hello"

    def test_decode_that_is_not_text_is_rejected(self):
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        tokenizer.decode = lambda *_args, **_kwargs: ["not", "text"]

        with pytest.raises(TypeError, match="decode must return a string"):
            sft_eval_sample(tokenizer, {"messages": HI_HELLO})

    def test_single_turn_row_ends_at_the_transcript_end(self):
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)

        _, chosen = sft_eval_sample(tokenizer, {"messages": HI_HELLO})

        assert chosen == "Hello"


def test_apply_chat_template():
    """Directly test the apply_chat_template helper."""
    tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
    template = [
        {"role": "user", "content": "Q: {question}"},
        {"role": "assistant", "content": "{answer}"},
    ]
    result = apply_chat_template(template, "What is 2+2?", "4", tokenizer)
    assert isinstance(result, BatchEncoding)
    assert "input_ids" in result
    assert "attention_mask" in result
    assert isinstance(result["input_ids"], torch.Tensor)
    assert result["input_ids"].ndim == 2

    decoded = tokenizer.decode(result["input_ids"][0], skip_special_tokens=False)
    assert "2+2" in decoded


def _collect_batch_rows(vec, num_resets, seed):
    """Reset ``vec`` ``num_resets`` times and return the per-batch-row dataset rows.

    Returns a list (one entry per reset) of lists (one per batch row) of the
    distinct dataset row each group resolved to. Asserts group row-consistency:
    every trajectory within a batch row shares the same row.
    """
    per_reset_rows: list[list[int]] = []
    for _ in range(num_resets):
        vec.reset(seed=seed)
        reset_rows: list[int] = []
        for batch_idx in range(vec.batch_size):
            group_rows = {
                vec.envs[batch_idx * vec.group_size + g]._last_row
                for g in range(vec.group_size)
            }
            assert len(group_rows) == 1, "group envs must share one row"
            reset_rows.append(group_rows.pop())
        per_reset_rows.append(reset_rows)
    return per_reset_rows


def test_batch_rollout_env_shuffle_is_group_consistent_full_permutation():
    """RolloutCollector resolves one shuffled row per group; epochs are full perms."""
    from agilerl.llm_envs import RolloutCollector

    dataset_size = 6

    class _RowRecordingEnv(RolloutEnvDoubleMixin):
        """Minimal pooled env recording the ``row_index`` RolloutCollector assigns.

        This test exercises only the shuffle / cursor, so it needs the
        ``dataset_size`` RolloutCollector shuffles over and a ``reset`` that records
        its row — not the full token machinery.
        """

        def __init__(self):
            self.dataset_size = dataset_size
            self._last_row = None
            self.done = False
            # A live (not-done) env always holds a token prompt; an empty
            # prompt is the terminal sentinel.
            self.current_prompt = {"input_ids": torch.ones(1, 2, dtype=torch.long)}

        def reset(self, seed=None, *, row_index=None):
            del seed
            self._last_row = row_index
            self.done = False
            self.current_prompt = {"input_ids": torch.ones(1, 2, dtype=torch.long)}
            return self.current_prompt, {}

    def _factory():
        return _RowRecordingEnv()

    # batch_size * resets spans two full epochs of the 6-row dataset.
    batch_size, group_size = 3, 2
    vec = RolloutCollector(
        env_factory=_factory,
        batch_size=batch_size,
        group_size=group_size,
        adaptive_task_sampling=False,
    )
    per_reset_rows = _collect_batch_rows(vec, num_resets=4, seed=7)

    # Flatten in cursor order: batch rows within a reset are consumed in order.
    flat = [row for reset_rows in per_reset_rows for row in reset_rows]
    assert len(flat) == 4 * batch_size  # 12 == two epochs of 6 rows
    for start in range(0, len(flat), dataset_size):
        epoch = flat[start : start + dataset_size]
        assert sorted(epoch) == list(range(dataset_size)), "each epoch is a full perm"

    # Same seed reproduces the same row sequence.
    vec_again = RolloutCollector(
        env_factory=_factory,
        batch_size=batch_size,
        group_size=group_size,
        adaptive_task_sampling=False,
    )
    again = _collect_batch_rows(vec_again, num_resets=4, seed=7)
    assert again == per_reset_rows

    # A different seed yields a different ordering.
    vec_other = RolloutCollector(
        env_factory=_factory,
        batch_size=batch_size,
        group_size=group_size,
        adaptive_task_sampling=False,
    )
    other = _collect_batch_rows(vec_other, num_resets=4, seed=8)
    assert other != per_reset_rows


def test_dataset_env_len_reflects_active_split():
    """``__len__`` follows the active split inside and outside ``eval_mode``."""
    train_dataset = DummyPreferenceDataset(6)
    test_dataset = DummyPreferenceDataset(2)
    tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
    env = DatasetEnv(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        objective="preference",
        data_batch_size_per_gpu=2,
    )

    assert len(env) == 6  # train split length (evaluation_mode is False)
    with env.eval_mode():
        assert env.evaluation_mode is True
        assert len(env) == 2  # held-out split length
    assert env.evaluation_mode is False
    assert len(env) == 6  # restored to the train split
    env.close()


def test_dataset_env_eval_mode_restores_prior_mode_when_nested():
    """``eval_mode`` restores the mode active on entry, so a nested probe
    (e.g. ``agent.test`` inside an outer eval context) doesn't flip the env
    back to the train split early.
    """
    tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
    env = DatasetEnv(
        train_dataset=DummyPreferenceDataset(6),
        test_dataset=DummyPreferenceDataset(2),
        tokenizer=tokenizer,
        objective="preference",
        data_batch_size_per_gpu=2,
    )
    with env.eval_mode():
        with env.eval_mode():
            assert env.evaluation_mode is True
        # Inner exit stays on the held-out split for the outer block.
        assert env.evaluation_mode is True
        assert env.dataloader is env.test_dataloader_iter
    assert env.evaluation_mode is False
    assert env.dataloader is env.train_dataloader_iter


def test_dataset_env_applies_chat_template_kwargs_in_collate():
    """Collate injects ``chat_template_kwargs`` onto the tokenizer (empty by default)."""
    tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
    dataset = DummyPreferenceDataset(4)
    env = DatasetEnv(
        train_dataset=dataset,
        test_dataset=dataset,
        tokenizer=tokenizer,
        objective="preference",
        chat_template_kwargs={"enable_thinking": False},
    )
    assert env.chat_template_kwargs == {"enable_thinking": False}
    assert resolve_chat_template_kwargs(env.tokenizer) == {"enable_thinking": False}

    default_tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
    default_env = DatasetEnv(
        train_dataset=dataset,
        test_dataset=dataset,
        tokenizer=default_tokenizer,
        objective="preference",
    )
    assert default_env.chat_template_kwargs == {}
    assert resolve_chat_template_kwargs(default_env.tokenizer) == {}


def test_dataset_env_rejects_unknown_kind():
    """``objective`` must be ``"preference"`` or ``"sft"``; anything else raises."""
    tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
    dataset = DummyPreferenceDataset(4)
    with pytest.raises(ValueError, match="Unknown dataset objective"):
        DatasetEnv(
            train_dataset=dataset,
            test_dataset=dataset,
            tokenizer=tokenizer,
            objective="bogus",
        )


class TestDatasetEnvSharding:
    @pytest.mark.parametrize("num_samples", [20])
    def test_ranks_get_disjoint_equal_shards(self, sft_dataset, num_samples):
        train_dataset, test_dataset = sft_dataset
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        envs = [
            DatasetEnv(
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                tokenizer=tokenizer,
                objective="sft",
                data_batch_size_per_gpu=4,
                rank=rank,
                world_size=2,
            )
            for rank in range(2)
        ]
        assert [env.dataset_size["train"] for env in envs] == [8, 8]
        row_sets = []
        for env in envs:
            rows: set[tuple[int, ...]] = set()
            for batch in env.train_dataloader:
                for ids, mask in zip(
                    batch["input_ids"], batch["attention_mask"], strict=True
                ):
                    rows.add(tuple(ids[mask.bool()].tolist()))
            row_sets.append(rows)
        assert row_sets[0].isdisjoint(row_sets[1])
        assert len(row_sets[0] | row_sets[1]) == len(train_dataset)

    @pytest.mark.parametrize("num_samples", [20])
    def test_world_size_beyond_rows_is_rejected(self, sft_dataset, num_samples):
        train_dataset, test_dataset = sft_dataset
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        with pytest.raises(ValueError, match="empty shard"):
            DatasetEnv(
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                tokenizer=tokenizer,
                objective="sft",
                data_batch_size_per_gpu=4,
                rank=0,
                world_size=num_samples,
            )

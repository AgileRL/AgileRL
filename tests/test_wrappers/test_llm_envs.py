# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for :mod:`agilerl.llm_envs` (reasoning, preference, and SFT gyms)."""

import importlib
import sys
from unittest.mock import MagicMock

import pytest
import torch
from accelerate import Accelerator
from accelerate.state import AcceleratorState

pytest.importorskip("datasets", reason="LLM dependencies not installed")

from datasets import Dataset as HFDataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding

from agilerl.llm_envs import (
    IterablePromptBatchGym,
    PreferenceGym,
    ReasoningGym,
    SFTGym,
    TokenObservationWrapper,
    apply_chat_template,
)
from tests import TINY_LLM_FIXTURE_PATH

DUMMY_CONVERSATION_TEMPLATE = [
    {
        "role": "system",
        "content": "question: {question}\nanswer: {answer}",
    },
]


def test_wrappers_llm_envs_compat_module_warns_and_reexports():
    sys.modules.pop("agilerl.wrappers.llm_envs", None)
    with pytest.warns(FutureWarning, match="deprecated"):
        compat_module = importlib.import_module("agilerl.wrappers.llm_envs")
    from agilerl.llm_envs import ReasoningGym as NewReasoningGym

    assert compat_module.ReasoningGym is NewReasoningGym


def dummy_reward_fn(*args, **kwargs):
    return 1.0


class TestApplyChatTemplate:
    def _tokenizer(self):
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "<rendered>"
        return tokenizer

    def test_final_user_message_opens_a_generation_prompt(self):
        tokenizer = self._tokenizer()
        template = [{"role": "user", "content": "{question}"}]
        apply_chat_template(template, "q", "a", tokenizer)
        assert tokenizer.apply_chat_template.call_args.kwargs == {
            "tokenize": False,
            "add_generation_prompt": True,
        }

    def test_assistant_prefill_continues_the_final_message(self):
        tokenizer = self._tokenizer()
        template = [
            {"role": "user", "content": "{question}"},
            {"role": "assistant", "content": "Let me think:"},
        ]
        apply_chat_template(template, "q", "a", tokenizer)
        assert tokenizer.apply_chat_template.call_args.kwargs == {
            "tokenize": False,
            "continue_final_message": True,
        }

    def test_empty_assistant_prefill_opens_a_fresh_turn(self):
        tokenizer = self._tokenizer()
        template = [
            {"role": "user", "content": "{question}"},
            {"role": "assistant", "content": ""},
        ]
        apply_chat_template(template, "q", "a", tokenizer)
        assert tokenizer.apply_chat_template.call_args.kwargs == {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        (messages,) = tokenizer.apply_chat_template.call_args.args
        assert messages == [{"role": "user", "content": "q"}]

    def test_rendered_text_is_encoded_without_special_tokens(self):
        tokenizer = self._tokenizer()
        template = [{"role": "user", "content": "{question}"}]
        apply_chat_template(template, "q", "a", tokenizer)
        assert tokenizer.call_args.kwargs["add_special_tokens"] is False


class Info:
    def __init__(self, name: str) -> None:
        self.dataset_name = name


class DummyPreferenceDataset(Dataset):
    def __init__(self, num_samples: int) -> None:
        self.prompt = [f"This is prompt {i}." for i in range(num_samples)]
        self.chosen = [f"This is chosen {i}." for i in range(num_samples)]
        self.rejected = [f"This is rejected {i}." for i in range(num_samples)]
        self.features = {
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
        }
        self.info = Info("dummy_dataset")

    def __len__(self) -> int:
        return len(self.prompt)

    def __getitem__(self, index: int) -> dict[str, str]:
        return {
            "prompt": self.prompt[index],
            "chosen": self.chosen[index],
            "rejected": self.rejected[index],
        }


class DummySFTDataset(Dataset):
    def __init__(self, num_samples: int) -> None:
        self.prompt = [f"This is prompt {i}." for i in range(num_samples)]
        self.target = [f"This is response {i}." for i in range(num_samples)]
        # SFTGym's default ``response_column`` is "target"; the output
        # batch is still keyed under "response" regardless of input name.
        self.features = {
            "prompt": self.prompt,
            "target": self.target,
        }
        self.info = Info("dummy_sft_dataset")

    def __len__(self) -> int:
        return len(self.prompt)

    def __getitem__(self, index: int) -> dict[str, str]:
        return {
            "prompt": self.prompt[index],
            "target": self.target[index],
        }


class DummyReasoningDataset(Dataset):
    def __init__(self, num_samples: int) -> None:
        self.questions = [f"This is question {i}?" for i in range(num_samples)]
        self.answers = [f"This is answer {i}." for i in range(num_samples)]
        self.features = {"question": self.questions, "answer": self.answers}
        self.info = Info("dummy_dataset")

    def __len__(self) -> int:
        return len(self.questions)

    def __getitem__(self, index: int) -> dict[str, str]:
        return {"question": self.questions[index], "answer": self.answers[index]}


@pytest.fixture
def accelerator_factory():
    def generate_accelerator(use_accelerator: bool):
        AcceleratorState._reset_state(True)
        return Accelerator() if use_accelerator else None

    return generate_accelerator


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


@pytest.fixture
def reasoning_dataset(num_samples):
    train_dataset = DummyReasoningDataset(int(num_samples * 0.8))
    test_dataset = DummyReasoningDataset(int(num_samples * 0.2))
    return train_dataset, test_dataset


class TestReasoningGymInit:
    @pytest.mark.parametrize("num_samples", [200])
    @pytest.mark.parametrize("use_accelerator", [True, False])
    def test_reasoning_gym_init(
        self,
        reasoning_dataset,
        accelerator_factory,
        num_samples,
        use_accelerator,
    ):
        train_dataset, test_dataset = reasoning_dataset
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        data_batch_size = 8
        env = ReasoningGym(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            tokenizer=tokenizer,
            reward_fn=dummy_reward_fn,
            conversation_template=DUMMY_CONVERSATION_TEMPLATE,
            data_batch_size_per_gpu=data_batch_size,
            accelerator=accelerator_factory(use_accelerator),
        )
        assert env.name == "dummy_dataset"
        assert callable(env.reward_fn)
        assert hasattr(env, "tokenizer")
        assert env.tokenizer is not None
        assert isinstance(env.train_dataloader, DataLoader)
        assert isinstance(env.test_dataloader, DataLoader)
        assert list(next(env.train_dataloader_iter).keys()) == [
            "question",
            "answer",
            "tokenized_prompts",
        ]
        assert env.dataloader == env.train_dataloader_iter
        assert not env.reset_called
        assert not env.evaluation_mode
        assert env.data_batch_size_per_gpu == data_batch_size

    def test_reasoning_gym_max_context_length_warning(self):
        train_dataset = HFDataset.from_dict(
            {
                "question": [
                    "This is a prompt that is longer than the max context length. This prompt really is a lot longer than the other one.",
                    "This is a prompt that is shorter.",
                ],
                "answer": ["This is an answer.", "This is an answer."],
            },
        )
        test_dataset = HFDataset.from_dict(
            {
                "question": ["This is a normal length prompt"],
                "answer": ["This is an answer."],
            },
        )
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        data_batch_size = 8
        with pytest.warns(
            UserWarning,
            match=r"1 samples were filtered out of the train dataset due to the max context length constraint.",
        ):
            env = ReasoningGym(
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                tokenizer=tokenizer,
                reward_fn=dummy_reward_fn,
                conversation_template=DUMMY_CONVERSATION_TEMPLATE,
                data_batch_size_per_gpu=data_batch_size,
                max_context_length=10,
            )
        assert len(env.train_dataloader) == 1
        assert len(env.test_dataloader) == 1

    def test_filter_dataset_non_string_early_return(self):
        """_filter_dataset_by_max_context_length returns early when values are not strings."""
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        train_dataset = HFDataset.from_dict(
            {
                "question": [["token1", "token2"], ["token3"]],
                "answer": ["a", "b"],
            },
        )
        test_dataset = HFDataset.from_dict(
            {
                "question": [["token1"]],
                "answer": ["a"],
            },
        )
        env = ReasoningGym(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            tokenizer=tokenizer,
            reward_fn=dummy_reward_fn,
            conversation_template=DUMMY_CONVERSATION_TEMPLATE,
            data_batch_size_per_gpu=1,
            max_context_length=5,
        )
        assert len(env.train_dataloader.dataset) == 2
        assert len(env.test_dataloader.dataset) == 1

    def test_reasoning_gym_init_missing_features(self):
        """ReasoningGym raises AssertionError when dataset lacks required features."""
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        good_dataset = HFDataset.from_dict({"question": ["q"], "answer": ["a"]})
        bad_dataset = HFDataset.from_dict({"text": ["t"]})
        with pytest.raises(AssertionError, match="'question' and 'answer'"):
            ReasoningGym(
                train_dataset=bad_dataset,
                test_dataset=good_dataset,
                tokenizer=tokenizer,
                reward_fn=dummy_reward_fn,
                conversation_template=DUMMY_CONVERSATION_TEMPLATE,
            )
        with pytest.raises(AssertionError, match="'question' and 'answer'"):
            ReasoningGym(
                train_dataset=good_dataset,
                test_dataset=bad_dataset,
                tokenizer=tokenizer,
                reward_fn=dummy_reward_fn,
                conversation_template=DUMMY_CONVERSATION_TEMPLATE,
            )


class TestReasoningGymStep:
    @pytest.mark.parametrize("num_samples", [200])
    @pytest.mark.parametrize("eval_mode", [True, False])
    @pytest.mark.parametrize("return_raw_completions", [True, False])
    def test_reasoning_gym_step(
        self,
        reasoning_dataset,
        num_samples,
        eval_mode,
        return_raw_completions,
    ):
        train_dataset, test_dataset = reasoning_dataset
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        data_batch_size = 8
        env = ReasoningGym(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            tokenizer=tokenizer,
            reward_fn=dummy_reward_fn,
            conversation_template=DUMMY_CONVERSATION_TEMPLATE,
            data_batch_size_per_gpu=data_batch_size,
            return_raw_completions=return_raw_completions,
        )
        env.evaluation_mode = eval_mode
        env.reset()
        completions = [
            torch.randint(0, 1000, (10, 356)) for _ in range(data_batch_size)
        ]
        tokenized_prompts, rewards = env.step(completions)
        assert isinstance(tokenized_prompts, list)
        assert isinstance(rewards, torch.Tensor)

        for prompts in tokenized_prompts:
            assert sorted(prompts.keys()) == ["attention_mask", "input_ids", "text"]
            for key, val in prompts.items():
                match key:
                    case "attention_mask":
                        assert isinstance(val, torch.Tensor)
                    case "input_ids":
                        assert isinstance(val, torch.Tensor)
                    case "text":
                        if return_raw_completions:
                            assert isinstance(val, str)
                        else:
                            assert val is None


class TestReasoningGymReset:
    @pytest.mark.parametrize("num_samples", [200])
    @pytest.mark.parametrize("reset_dataloaders", [True, False])
    @pytest.mark.parametrize("return_raw_completions", [True, False])
    def test_reasoning_gym_reset(
        self,
        reasoning_dataset,
        num_samples,
        reset_dataloaders,
        return_raw_completions,
    ):
        train_dataset, test_dataset = reasoning_dataset
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        data_batch_size = 8
        env = ReasoningGym(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            tokenizer=tokenizer,
            reward_fn=dummy_reward_fn,
            conversation_template=DUMMY_CONVERSATION_TEMPLATE,
            data_batch_size_per_gpu=data_batch_size,
            return_raw_completions=return_raw_completions,
        )
        tokenized_prompts = env.reset(reset_dataloaders)
        assert isinstance(tokenized_prompts, list)

        for prompts in tokenized_prompts:
            assert sorted(prompts.keys()) == ["attention_mask", "input_ids", "text"]
            for key, val in prompts.items():
                match key:
                    case "attention_mask":
                        assert isinstance(val, torch.Tensor)
                    case "input_ids":
                        assert isinstance(val, torch.Tensor)
                    case "text":
                        if return_raw_completions:
                            assert isinstance(val, str)
                        else:
                            assert val is None

    @pytest.mark.parametrize("num_samples", [200])
    def test_reasoning_gym_reset_warning(self, reasoning_dataset, num_samples):
        train_dataset, test_dataset = reasoning_dataset
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        data_batch_size = 8
        env = ReasoningGym(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            tokenizer=tokenizer,
            reward_fn=dummy_reward_fn,
            conversation_template=DUMMY_CONVERSATION_TEMPLATE,
            data_batch_size_per_gpu=data_batch_size,
        )
        env.reset()
        with pytest.warns(
            UserWarning,
            match=r"env\.reset\(\) called more than once sequentially",
        ):
            env.reset()


class TestReasoningGymResetDataloaders:
    @pytest.mark.parametrize("num_samples", [200])
    @pytest.mark.parametrize("reset_dataloaders", [True, False])
    def test_reasoning_gym_reset_dataloaders(
        self,
        reasoning_dataset,
        num_samples,
        reset_dataloaders,
    ):
        train_dataset, test_dataset = reasoning_dataset
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        data_batch_size = 8
        env = ReasoningGym(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            tokenizer=tokenizer,
            reward_fn=dummy_reward_fn,
            conversation_template=DUMMY_CONVERSATION_TEMPLATE,
            data_batch_size_per_gpu=data_batch_size,
        )
        first_data_point = next(
            env.test_dataloader_iter,
        )  # use test_dataloader_iter as it is not shuffled
        env._reset_dataloaders()
        first_data_point_reset = next(env.test_dataloader_iter)
        for key1, _ in zip(
            first_data_point.keys(),
            first_data_point_reset.keys(),
            strict=False,
        ):
            if key1 == "tokenized_prompts":
                for item1, item2 in zip(
                    first_data_point["tokenized_prompts"],
                    first_data_point_reset["tokenized_prompts"],
                    strict=False,
                ):
                    for key3, key4 in zip(item1.keys(), item2.keys(), strict=False):
                        assert torch.equal(item1[key3], item2[key4])
            else:
                assert first_data_point[key1] == first_data_point_reset[key1]


class TestReasoningGymLen:
    @pytest.mark.parametrize("num_samples", [200])
    def test_reasoning_gym_len(self, reasoning_dataset, num_samples):
        train_dataset, test_dataset = reasoning_dataset
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        data_batch_size = 8
        env = ReasoningGym(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            tokenizer=tokenizer,
            reward_fn=dummy_reward_fn,
            conversation_template=DUMMY_CONVERSATION_TEMPLATE,
            data_batch_size_per_gpu=data_batch_size,
        )
        env.reset()
        assert len(env) == 200 * 0.8  # Length returns the training length
        with env.eval_mode():
            assert len(env) == 200 * 0.2


class TestReasoningGymCreateCollateFn:
    @pytest.mark.parametrize("num_samples", [20])
    def test_reasoning_gym_create_collate_fn(self, reasoning_dataset, num_samples):
        """Test ReasoningGym.create_collate_fn."""
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)

        train_dataset, test_dataset = reasoning_dataset
        data_batch_size = 8

        env = ReasoningGym(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            tokenizer=tokenizer,
            reward_fn=dummy_reward_fn,
            conversation_template=DUMMY_CONVERSATION_TEMPLATE,
            data_batch_size_per_gpu=data_batch_size,
        )

        collate_fn = env.create_collate_fn(tokenizer)

        batch = [
            {"question": "What is 2+2?", "answer": "4"},
            {"question": "What is 3+3?", "answer": "6"},
        ]

        result = collate_fn(batch)

        assert isinstance(result, dict)
        assert "question" in result
        assert "answer" in result
        assert "tokenized_prompts" in result

        assert result["question"] == ["What is 2+2?", "What is 3+3?"]
        assert result["answer"] == ["4", "6"]
        assert len(result["tokenized_prompts"]) == 2

        for prompt in result["tokenized_prompts"]:
            assert isinstance(prompt, BatchEncoding)
            assert "input_ids" in prompt
            assert "attention_mask" in prompt
            assert isinstance(prompt["input_ids"], torch.Tensor)
            assert isinstance(prompt["attention_mask"], torch.Tensor)

    @pytest.mark.parametrize("num_samples", [20])
    def test_reasoning_gym_collate_fn_requires_conversation_template(
        self, reasoning_dataset, num_samples
    ):
        """collate_fn raises when no conversation template is configured."""
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        train_dataset, test_dataset = reasoning_dataset

        env = ReasoningGym(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            tokenizer=tokenizer,
            reward_fn=dummy_reward_fn,
            conversation_template=DUMMY_CONVERSATION_TEMPLATE,
            data_batch_size_per_gpu=8,
        )
        collate_fn = env.create_collate_fn(tokenizer)
        env.conversation_template = None

        with pytest.raises(ValueError, match="requires a conversation template"):
            collate_fn([{"question": "What is 2+2?", "answer": "4"}])


class TestReasoningGymGetNextBatch:
    @pytest.mark.parametrize("num_samples", [20])
    @pytest.mark.parametrize("data_batch_size", [8, 10])
    def test_reasoning_gym_reset_dataloaders_when_train_dataloader_exhausted(
        self,
        reasoning_dataset,
        num_samples,
        data_batch_size,
    ):
        train_dataset, test_dataset = reasoning_dataset
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        env = ReasoningGym(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            tokenizer=tokenizer,
            reward_fn=dummy_reward_fn,
            conversation_template=DUMMY_CONVERSATION_TEMPLATE,
            data_batch_size_per_gpu=data_batch_size,
        )
        for _ in range(3):
            env._get_next_batch()

        assert env.num_epochs == 1

    @pytest.mark.parametrize("num_samples", [20])
    @pytest.mark.parametrize("data_batch_size", [8, 10])
    def test_reasoning_gym_not_reset_dataloaders_when_test_dataloader_exhausted(
        self,
        reasoning_dataset,
        num_samples,
        data_batch_size,
    ):
        train_dataset, test_dataset = reasoning_dataset
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        env = ReasoningGym(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            tokenizer=tokenizer,
            reward_fn=dummy_reward_fn,
            conversation_template=DUMMY_CONVERSATION_TEMPLATE,
            data_batch_size_per_gpu=data_batch_size,
        )
        env.reset()
        for _ in range(10):
            with env.eval_mode():
                env._get_next_batch()

        assert env.num_epochs == 0


class TestReasoningGymEvalMode:
    @pytest.mark.parametrize("num_samples", [20])
    def test_eval_mode_preserves_last_tokenized_prompts(
        self, reasoning_dataset, num_samples
    ):
        """eval_mode() should save and restore last_tokenized_prompts."""
        train_dataset, test_dataset = reasoning_dataset
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        env = ReasoningGym(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            tokenizer=tokenizer,
            reward_fn=dummy_reward_fn,
            conversation_template=DUMMY_CONVERSATION_TEMPLATE,
            data_batch_size_per_gpu=4,
        )
        env.reset()
        saved_ids = [p["input_ids"].clone() for p in env.last_tokenized_prompts]

        with env.eval_mode():
            env.reset()
            assert env.evaluation_mode

        assert not env.evaluation_mode
        for original, restored in zip(
            saved_ids, env.last_tokenized_prompts, strict=False
        ):
            assert torch.equal(original, restored["input_ids"])

    def test_eval_mode_restores_train_questions_and_answers(self):
        """The first post-eval train step must be scored against train targets.

        Regression test: eval used to leave ``questions``/``answers`` pointing
        at the last test batch, so a remainder test batch shorter than the
        train batch truncated the rewards and crashed ``learn()`` with a
        rewards/completions length mismatch.
        """
        train_dataset = HFDataset.from_dict(
            {
                "question": [f"train question {i}?" for i in range(8)],
                "answer": [f"train answer {i}" for i in range(8)],
            },
        )
        # 6 test samples with batch size 4 -> the second test batch is a
        # remainder of 2, shorter than the train batch.
        test_dataset = HFDataset.from_dict(
            {
                "question": [f"test question {i}?" for i in range(6)],
                "answer": [f"test answer {i}" for i in range(6)],
            },
        )
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        seen_answers = []

        def recording_reward_fn(completion, answer, question):
            seen_answers.append(answer)
            return 1.0

        env = ReasoningGym(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            tokenizer=tokenizer,
            reward_fn=recording_reward_fn,
            conversation_template=DUMMY_CONVERSATION_TEMPLATE,
            data_batch_size_per_gpu=4,
        )
        group_size = 8

        def completions_for(prompt_batch):
            return [torch.randint(0, 1000, (group_size, 356)) for _ in prompt_batch]

        prompts = env.reset()
        prompts, _ = env.step(completions_for(prompts))
        train_answers = list(env.answers)
        train_questions = list(env.questions)

        # One eval pass as run by GRPO.test with eval_loop=1: reset scores the
        # full first test batch, step advances to the remainder batch.
        with env.eval_mode():
            eval_prompts = env.reset()
            _, eval_rewards = env.step(completions_for(eval_prompts))
            assert eval_rewards.shape == (4, group_size)
            assert len(env.answers) == 2

        assert env.answers == train_answers
        assert env.questions == train_questions

        seen_answers.clear()
        _, rewards = env.step(completions_for(prompts))
        assert rewards.shape == (len(prompts), group_size)
        assert set(seen_answers) == set(train_answers)


class TestPreferenceGymInit:
    @pytest.mark.parametrize("use_accelerator", [True, False])
    @pytest.mark.parametrize("num_samples", [20])
    def test_preference_gym_init(
        self,
        preference_dataset,
        accelerator_factory,
        use_accelerator,
        num_samples,
    ):
        train_dataset, test_dataset = preference_dataset
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        data_batch_size = 8
        env = PreferenceGym(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            tokenizer=tokenizer,
            data_batch_size_per_gpu=data_batch_size,
            accelerator=accelerator_factory(use_accelerator),
        )
        assert isinstance(env, IterablePromptBatchGym)
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
        assert not env.reset_called
        assert not env.evaluation_mode
        assert env.data_batch_size_per_gpu == data_batch_size

    def test_preference_gym_max_context_length_error(self):
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
            PreferenceGym(
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                tokenizer=tokenizer,
                data_batch_size_per_gpu=data_batch_size,
                max_context_length=5,
                min_completion_length=1,
            )

    def test_preference_gym_max_context_length_warning(self):
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
            env = PreferenceGym(
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                tokenizer=tokenizer,
                data_batch_size_per_gpu=data_batch_size,
                max_context_length=10,
                min_completion_length=1,
            )
        assert len(env.train_dataloader) == 1
        assert len(env.test_dataloader) == 1

    def test_preference_gym_init_missing_features(self):
        """PreferenceGym raises AssertionError when dataset lacks required features."""
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        good_dataset = HFDataset.from_dict(
            {"prompt": ["p"], "chosen": ["c"], "rejected": ["r"]},
        )
        # Has "prompt" (so super().__init__ filter works) but missing "chosen"/"rejected"
        bad_dataset = HFDataset.from_dict({"prompt": ["p"], "other": ["o"]})
        with pytest.raises(AssertionError, match="'prompt', 'chosen', and 'rejected'"):
            PreferenceGym(
                train_dataset=bad_dataset,
                test_dataset=good_dataset,
                tokenizer=tokenizer,
            )
        with pytest.raises(AssertionError, match="'prompt', 'chosen', and 'rejected'"):
            PreferenceGym(
                train_dataset=good_dataset,
                test_dataset=bad_dataset,
                tokenizer=tokenizer,
            )


class TestPreferenceGymStep:
    @pytest.mark.parametrize("use_accelerator", [True, False])
    @pytest.mark.parametrize("num_samples", [20])
    def test_preference_gym_step(
        self,
        preference_dataset,
        accelerator_factory,
        use_accelerator,
        num_samples,
    ):
        train_dataset, test_dataset = preference_dataset
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        data_batch_size = 8
        env = PreferenceGym(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            tokenizer=tokenizer,
            data_batch_size_per_gpu=data_batch_size,
            accelerator=accelerator_factory(use_accelerator),
        )
        prompts = env.step()
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
        assert not env.reset_called


class TestPreferenceGymReset:
    @pytest.mark.parametrize("use_accelerator", [True, False])
    @pytest.mark.parametrize("num_samples", [20])
    def test_preference_gym_reset(
        self,
        preference_dataset,
        accelerator_factory,
        use_accelerator,
        num_samples,
    ):
        train_dataset, test_dataset = preference_dataset
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        data_batch_size = 8
        env = PreferenceGym(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            tokenizer=tokenizer,
            data_batch_size_per_gpu=data_batch_size,
            accelerator=accelerator_factory(use_accelerator),
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
        assert env.reset_called

    @pytest.mark.parametrize("use_accelerator", [True, False])
    @pytest.mark.parametrize("num_samples", [20])
    def test_preference_gym_reset_reset_dataloaders_warning(
        self,
        preference_dataset,
        accelerator_factory,
        use_accelerator,
        num_samples,
    ):
        train_dataset, test_dataset = preference_dataset
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        data_batch_size = 1
        env = PreferenceGym(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            tokenizer=tokenizer,
            data_batch_size_per_gpu=data_batch_size,
            accelerator=accelerator_factory(use_accelerator),
        )
        env.reset()
        env.step()
        env.step()
        with pytest.warns(
            UserWarning,
            match=r"env\.reset\(\) called with reset_dataloaders=True, this will reset the dataloaders to the beginning of the dataset, proceed with caution\.",
        ):
            prompts = env.reset(reset_dataloaders=True)
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
        assert env.reset_called

    @pytest.mark.parametrize("use_accelerator", [True, False])
    @pytest.mark.parametrize("num_samples", [20])
    def test_preference_gym_reset_reset_called_warning(
        self,
        preference_dataset,
        accelerator_factory,
        use_accelerator,
        num_samples,
    ):
        train_dataset, test_dataset = preference_dataset
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        data_batch_size = 1
        env = PreferenceGym(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            tokenizer=tokenizer,
            data_batch_size_per_gpu=data_batch_size,
            accelerator=accelerator_factory(use_accelerator),
        )
        env.reset_called = True
        with pytest.warns(
            UserWarning,
            match=r"env\.reset\(\) called more than once sequentially, it should typically follow with env\.step\(\)\.",
        ):
            prompts = env.reset()
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
        assert env.reset_called

    @pytest.mark.parametrize("num_samples", [20])
    @pytest.mark.parametrize("use_accelerator", [True, False])
    def test_preference_gym_reset_num_epochs(
        self,
        preference_dataset,
        num_samples,
        accelerator_factory,
        use_accelerator,
    ):
        train_dataset, test_dataset = preference_dataset
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        data_batch_size = 1
        env = PreferenceGym(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            tokenizer=tokenizer,
            data_batch_size_per_gpu=data_batch_size,
            accelerator=accelerator_factory(use_accelerator),
        )
        while env.num_epochs == 0:
            env.step()
        assert env.num_epochs == 1


class TestPreferenceGymCreateCollateFn:
    def test_preference_gym_collate_max_context_length_branch(self):
        """Exercise ``max_context_length is not None`` tokenisation in PreferenceGym."""
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
        env = PreferenceGym(
            train_dataset=train_ds,
            test_dataset=test_ds,
            tokenizer=tokenizer,
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

    def test_preference_completions_end_with_eos(self):
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        ds = HFDataset.from_dict(
            {
                "prompt": ["hello", "hi"],
                "chosen": ["yes please", "sure"],
                "rejected": ["no thanks", "never"],
            },
        )
        env = PreferenceGym(
            train_dataset=ds,
            test_dataset=ds,
            tokenizer=tokenizer,
            data_batch_size_per_gpu=2,
        )
        batch = next(env.train_dataloader_iter)
        for key in ("chosen", "rejected"):
            ids = batch[f"{key}_input_ids"]
            masks = batch[f"{key}_attention_mask"]
            for row_ids, row_mask in zip(ids, masks, strict=True):
                last_real = row_ids[row_mask.bool()][-1]
                assert int(last_real) == tokenizer.eos_token_id
        assert all(not c.endswith(tokenizer.eos_token) for c in batch["chosen"])
        assert all(not r.endswith(tokenizer.eos_token) for r in batch["rejected"])


class TestSFTGymInit:
    @pytest.mark.parametrize("use_accelerator", [True, False])
    @pytest.mark.parametrize("num_samples", [20])
    def test_sft_gym_init(
        self,
        sft_dataset,
        accelerator_factory,
        use_accelerator,
        num_samples,
    ):
        train_dataset, test_dataset = sft_dataset
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        data_batch_size = 8
        env = SFTGym(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            tokenizer=tokenizer,
            data_batch_size_per_gpu=data_batch_size,
            accelerator=accelerator_factory(use_accelerator),
        )
        assert isinstance(env, IterablePromptBatchGym)
        assert env.name == "dummy_sft_dataset"
        assert list(next(env.train_dataloader_iter).keys()) == [
            "prompt",
            "prompt_lengths",
            "response",
            "input_ids",
            "attention_mask",
        ]
        assert env.dataloader == env.train_dataloader_iter
        assert not env.reset_called
        assert env.data_batch_size_per_gpu == data_batch_size

    def test_sft_gym_max_context_length_warning(self):
        train_dataset = HFDataset.from_dict(
            {
                "prompt": [
                    "This is a prompt that is longer than the max context length. This prompt really is a lot longer than the other one.",
                    "short",
                ],
                "target": ["a", "b"],
            },
        )
        test_dataset = HFDataset.from_dict(
            {
                "prompt": ["ok"],
                "target": ["a"],
            },
        )
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        with pytest.warns(
            UserWarning,
            match=r"1 samples were filtered out of the train dataset due to the max context length constraint.",
        ):
            env = SFTGym(
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                tokenizer=tokenizer,
                data_batch_size_per_gpu=8,
                max_context_length=10,
            )
        assert len(env.train_dataloader) == 1

    def test_sft_gym_init_missing_features(self):
        """SFTGym raises AssertionError when dataset lacks required features."""
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        good_dataset = HFDataset.from_dict({"prompt": ["p"], "target": ["r"]})
        # Has "prompt" (so super().__init__ filter works) but missing "target"
        bad_dataset = HFDataset.from_dict({"prompt": ["p"], "other": ["o"]})
        with pytest.raises(AssertionError, match="must contain"):
            SFTGym(
                train_dataset=bad_dataset,
                test_dataset=good_dataset,
                tokenizer=tokenizer,
            )
        with pytest.raises(AssertionError, match="must contain"):
            SFTGym(
                train_dataset=good_dataset,
                test_dataset=bad_dataset,
                tokenizer=tokenizer,
            )

    def test_sft_targets_end_with_eos(self):
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        ds = HFDataset.from_dict(
            {
                "prompt": ["hello", "hi"],
                "target": ["yes please", "sure"],
            },
        )
        env = SFTGym(
            train_dataset=ds,
            test_dataset=ds,
            tokenizer=tokenizer,
            data_batch_size_per_gpu=2,
        )
        batch = next(env.train_dataloader_iter)
        for row_ids, row_mask in zip(
            batch["input_ids"], batch["attention_mask"], strict=True
        ):
            last_real = row_ids[row_mask.bool()][-1]
            assert int(last_real) == tokenizer.eos_token_id
        assert all(not r.endswith(tokenizer.eos_token) for r in batch["response"])


class TestSFTGymStep:
    @pytest.mark.parametrize("num_samples", [20])
    @pytest.mark.parametrize("use_accelerator", [True, False])
    def test_sft_gym_num_epochs_increment(
        self,
        sft_dataset,
        num_samples,
        accelerator_factory,
        use_accelerator,
    ):
        train_dataset, test_dataset = sft_dataset
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        env = SFTGym(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            tokenizer=tokenizer,
            data_batch_size_per_gpu=1,
            accelerator=accelerator_factory(use_accelerator),
        )
        while env.num_epochs == 0:
            env.step()
        assert env.num_epochs == 1


class TestSFTGymReset:
    @pytest.mark.parametrize("use_accelerator", [True, False])
    @pytest.mark.parametrize("num_samples", [20])
    def test_sft_gym_step_and_reset(
        self,
        sft_dataset,
        accelerator_factory,
        use_accelerator,
        num_samples,
    ):
        train_dataset, test_dataset = sft_dataset
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        data_batch_size = 8
        env = SFTGym(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            tokenizer=tokenizer,
            data_batch_size_per_gpu=data_batch_size,
            accelerator=accelerator_factory(use_accelerator),
        )
        batch = env.step()
        assert set(batch.keys()) == {
            "prompt",
            "prompt_lengths",
            "response",
            "input_ids",
            "attention_mask",
        }
        assert not env.reset_called

        batch2 = env.reset()
        assert set(batch2.keys()) == {
            "prompt",
            "prompt_lengths",
            "response",
            "input_ids",
            "attention_mask",
        }
        assert env.reset_called
        assert len(batch2["prompt"]) == data_batch_size

    @pytest.mark.parametrize("use_accelerator", [True, False])
    @pytest.mark.parametrize("num_samples", [20])
    def test_sft_gym_reset_warnings_match_iterable_base(
        self,
        sft_dataset,
        accelerator_factory,
        use_accelerator,
        num_samples,
    ):
        train_dataset, test_dataset = sft_dataset
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        env = SFTGym(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            tokenizer=tokenizer,
            data_batch_size_per_gpu=1,
            accelerator=accelerator_factory(use_accelerator),
        )
        env.reset()
        env.step()
        env.step()
        with pytest.warns(
            UserWarning,
            match=r"env\.reset\(\) called with reset_dataloaders=True",
        ):
            env.reset(reset_dataloaders=True)

        env.reset_called = True
        with pytest.warns(
            UserWarning,
            match=r"env\.reset\(\) called more than once sequentially",
        ):
            env.reset()

    def test_sft_gym_response_column_chosen(self):
        """``response_column`` can point at a DPO-style ``chosen`` column."""
        train_dataset = HFDataset.from_dict(
            {
                "prompt": ["p"],
                "chosen": ["c"],
                "rejected": ["r"],
            },
        )
        test_dataset = HFDataset.from_dict(
            {
                "prompt": ["p"],
                "chosen": ["c"],
                "rejected": ["r"],
            },
        )
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        env = SFTGym(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            tokenizer=tokenizer,
            data_batch_size_per_gpu=1,
            response_column="chosen",
        )
        b = env.reset()
        assert b["response"] == ["c"]


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


class TestTokenObservationWrapperStepGuard:
    def test_step_before_reset_raises(self):
        """Stepping before reset() has populated the prompt state raises."""
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        wrapper = TokenObservationWrapper(
            env=None,
            tokenizer=tokenizer,
            max_turns=1,
            apply_chat_template=False,
        )
        assert wrapper.full_ids is None
        with pytest.raises(RuntimeError, match="reset\\(\\) was never called"):
            wrapper._step(torch.ones(1, 2, dtype=torch.long), "gen")

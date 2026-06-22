"""Tests for :mod:`agilerl.llm_envs` (reasoning, preference, and SFT gyms)."""

import importlib
import sys
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


@pytest.fixture(scope="function")
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
        with pytest.warns():
            env.reset()
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
            match="No samples left in the train dataset after filtering by the max context length constraint, use a larger max context length.",
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
        with pytest.raises(AssertionError, match="must contain columns"):
            PreferenceGym(
                train_dataset=bad_dataset,
                test_dataset=good_dataset,
                tokenizer=tokenizer,
            )
        with pytest.raises(AssertionError, match="must contain columns"):
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
        with pytest.warns(
            UserWarning,
            match=r"env\.reset\(\) called more than once sequentially, it should typically follow with env\.step\(\)\.",
        ):
            env.reset_called = True
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

        with pytest.warns(
            UserWarning,
            match=r"env\.reset\(\) called more than once sequentially",
        ):
            env.reset_called = True
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


def test_dataloader_shuffle_order_is_deterministic_permutation():
    """The shuffle index list is deterministic and a full per-epoch permutation."""
    from agilerl.llm_envs import dataloader_shuffle_order

    dataset_size = 16
    seed = 7
    epochs = 3
    order = dataloader_shuffle_order(dataset_size, seed, epochs)
    assert len(order) == dataset_size * epochs
    # Each epoch slice is a full permutation of the dataset indices.
    for start in range(0, len(order), dataset_size):
        assert sorted(order[start : start + dataset_size]) == list(range(dataset_size))

    # Deterministic per seed; a different seed produces a different shuffle.
    assert order == dataloader_shuffle_order(dataset_size, seed, epochs)
    assert order != dataloader_shuffle_order(dataset_size, seed + 1, epochs)


def test_batch_iteration_state_seed_maps_group_to_same_row():
    """A reused per-row seed maps every group member to the same dataset row."""
    from agilerl.llm_envs import BatchIterationState, dataloader_shuffle_order

    state = BatchIterationState(
        shuffle_order=[5, 3, 9, 1],
        seed=0,
        dataset_size=4,
    )
    # First sighting of a seed binds it to the current cursor; repeats return it.
    first = state.position_for_seed(100)
    repeat = state.position_for_seed(100)
    assert first == repeat == 0
    # A new seed advances the cursor.
    second = state.position_for_seed(101)
    assert second == 1
    # ``row_index`` extends the order across epoch boundaries deterministically.
    assert state.row_index(0) == 5
    expected_epoch_two = dataloader_shuffle_order(4, 0, 2)
    assert state.row_index(4) == expected_epoch_two[4]


class TestReasoningRolloutEquivalence:
    """Bit-for-bit oracle: old ``ReasoningGym`` vs the single-turn ``RolloutEnv``.

    Seeding the new env's shared shuffle order from the exact dataset rows the
    batched reasoning env produced, and stepping with identical appended
    generation tokens in ``(batch_idx, group_idx)`` order, must reproduce the old
    path's dataset rows, group layout, decoded completions, and rewards exactly.
    """

    @staticmethod
    def _length_reward_fn(completion, answer, question):
        # Depends on all three arguments so any row/grouping/decode divergence
        # changes the reward — making the equivalence check sensitive.
        return float(len(completion) * 0.01 + len(answer) + len(question))

    @pytest.mark.parametrize("num_samples", [80])
    @pytest.mark.parametrize("group_size", [1, 4])
    def test_reasoning_rollout_matches_reasoning_gym(
        self, reasoning_dataset, num_samples, group_size
    ):
        from agilerl.llm_envs import (
            BatchIterationState,
            BatchRolloutEnv,
            make_reasoning_rollout_env,
        )

        train_dataset, test_dataset = reasoning_dataset
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        data_batch_size = 4
        seed = 123
        gen_base = [101, 102, 103]

        old_env = ReasoningGym(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            tokenizer=tokenizer,
            reward_fn=self._length_reward_fn,
            conversation_template=DUMMY_CONVERSATION_TEMPLATE,
            data_batch_size_per_gpu=data_batch_size,
            seed=seed,
        )
        old_prompts = old_env.reset()
        old_questions = list(old_env.questions)

        # Build grouped completions: prompt ids + identical appended gen tokens.
        old_completions = []
        old_decoded = []
        for batch_idx in range(data_batch_size):
            prompt_ids = old_prompts[batch_idx]["input_ids"]
            prompt_len = prompt_ids.shape[1]
            group_rows = []
            for grp in range(group_size):
                gen = torch.tensor(
                    [[gen_base[0] + batch_idx, gen_base[1] + grp, gen_base[2]]]
                )
                group_rows.append(torch.cat([prompt_ids, gen], dim=1))
            grouped = torch.cat(group_rows, dim=0)
            old_completions.append(grouped)
            old_decoded.append(
                tokenizer.batch_decode(
                    grouped[:, prompt_len:], skip_special_tokens=True
                )
            )
        _, old_rewards = old_env.step(old_completions)
        assert tuple(old_rewards.shape) == (data_batch_size, group_size)

        # Recover the exact dataset rows the old env drew (its explicit order).
        question_to_index = {
            f"This is question {i}?": i for i in range(len(train_dataset))
        }
        old_rows = [question_to_index[q] for q in old_questions]

        # New path: the BatchRolloutEnv owns the dataset cursor. Seed its shared
        # iteration order from old's exact rows so row selection matches.
        state = BatchIterationState(
            shuffle_order=list(old_rows),
            seed=seed,
            dataset_size=len(train_dataset),
        )

        def env_factory(**_kwargs):
            return make_reasoning_rollout_env(
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                tokenizer=tokenizer,
                reward_fn=self._length_reward_fn,
                conversation_template=DUMMY_CONVERSATION_TEMPLATE,
                seed=seed,
            )

        vec_env = BatchRolloutEnv(
            env_factory, batch_size=data_batch_size, group_size=group_size
        )
        vec_env._iteration_state = state
        vec_env.reset(seed=1000)
        active = vec_env.trajectories.get_active_trajectories(sorted_by_index=True)
        assert len(active) == data_batch_size * group_size

        # Group layout matches: (batch_idx, group_idx) contiguous, row shared.
        for batch_idx in range(data_batch_size):
            group = [t for t in active if t.batch_idx == batch_idx]
            assert {t.group_idx for t in group} == set(range(group_size))
            rows = {t.env._env._question for t in group}
            assert rows == {f"This is question {old_rows[batch_idx]}?"}

        # Step with identical appended gen tokens, in (batch, group) order.
        new_completions = []
        for traj in active:
            prompt_ids = traj.prompt["input_ids"]
            gen = torch.tensor(
                [
                    [
                        gen_base[0] + traj.batch_idx,
                        gen_base[1] + traj.group_idx,
                        gen_base[2],
                    ]
                ]
            )
            new_completions.append(torch.cat([prompt_ids, gen], dim=1))
        vec_env.step(new_completions)

        (_ids, _masks, _turn_ids, all_rewards, _steps, _logps) = (
            vec_env.get_trajectories()
        )

        # Decoded generated text parity, per (batch, group).
        idx = 0
        new_matrix = torch.zeros(data_batch_size, group_size)
        for batch_idx in range(data_batch_size):
            for grp in range(group_size):
                traj = active[idx]
                wrapper = traj.env
                assert wrapper._gen_texts[0] == old_decoded[batch_idx][grp]
                new_matrix[batch_idx, grp] = all_rewards[idx].sum()
                idx += 1

        assert torch.equal(new_matrix, old_rewards.float())

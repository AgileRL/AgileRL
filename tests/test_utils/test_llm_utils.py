import sys
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from datasets import Dataset as Datasets
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding

from agilerl.utils.algo_utils import DummyOptimizer
from agilerl.utils.llm_utils import (
    PreferenceGym,
    ReasoningGym,
    gather_if_zero3,
    get_state_dict,
)

pytestmark = pytest.mark.llm

DUMMY_CONVERSATION_TEMPLATE = [
    {
        "role": "system",
        "content": "question: {question}\nanswer: {answer}",
    },
]


class DummyTokenizer:
    def __init__(self, *args, **kwargs):
        self.vocab_size = 1000

    def batch_decode(self, *args, **kwargs):
        return ["This is a test completion."]

    def apply_chat_template(self, *args, **kwargs):
        return "This is a test completion."

    def __call__(self, *args, **kwargs):
        return torch.tensor([1, 2, 3, 4, 5])


class Info:
    def __init__(self, name):
        self.dataset_name = name


class DummyReasoningDataset(Dataset):
    def __init__(self, num_samples):
        # Create dummy questions and answers
        self.questions = [f"This is question {i}?" for i in range(num_samples)]
        self.answers = [f"This is answer {i}." for i in range(num_samples)]
        self.features = {"question": self.questions, "answer": self.answers}
        self.info = Info("dummy_dataset")

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, index):
        return {"question": self.questions[index], "answer": self.answers[index]}


class DummyPreferenceDataset(Dataset):
    def __init__(self, num_samples):
        self.prompt = [f"This is prompt {i}." for i in range(num_samples)]
        self.chosen = [f"This is chosen {i}." for i in range(num_samples)]
        self.rejected = [f"This is rejected {i}." for i in range(num_samples)]
        self.features = {
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
        }
        self.info = Info("dummy_dataset")

    def __len__(self):
        return len(self.prompt)

    def __getitem__(self, index):
        return {
            "prompt": self.prompt[index],
            "chosen": self.chosen[index],
            "rejected": self.rejected[index],
        }


def dummy_reward_fn(*args, **kwargs):
    return 1.0


def dummy_chat_template_fn_custom(q, a, tokenizer):
    """
    Chat template function for test_reasoning_gym_reset_dataloaders, gives unique input_ids for each question so
    we can test equality.
    """
    index = int(q.split(" ")[-1][0])
    return {
        "input_ids": torch.tensor([index]),
        "attention_mask": torch.ones(1),
    }


def dummy_chat_template_fn(q, a, tokenizer):
    return {
        "input_ids": torch.randint(0, 1000, (1, 356)),
        "attention_mask": torch.ones(1, 356),
    }


@pytest.fixture(scope="function")
def accelerator_factory():
    def generate_accelerator(use_accelerator):
        AcceleratorState._reset_state(True)
        return Accelerator() if use_accelerator else None

    return generate_accelerator


@pytest.fixture
def reasoning_dataset(num_samples):
    train_dataset = DummyReasoningDataset(int(num_samples * 0.8))
    test_dataset = DummyReasoningDataset(int(num_samples * 0.2))
    return train_dataset, test_dataset


@pytest.fixture
def preference_dataset(num_samples):
    train_dataset = DummyPreferenceDataset(int(num_samples * 0.8))
    test_dataset = DummyPreferenceDataset(int(num_samples * 0.2))
    return train_dataset, test_dataset


@pytest.mark.parametrize("num_samples", [200])
@pytest.mark.parametrize("use_accelerator", [True, False])
def test_reasoning_gym_init(
    reasoning_dataset, accelerator_factory, num_samples, use_accelerator
):
    train_dataset, test_dataset = reasoning_dataset
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
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


@pytest.mark.parametrize("num_samples", [200])
@pytest.mark.parametrize("eval_mode", [True, False])
@pytest.mark.parametrize("return_raw_completions", [True, False])
def test_reasoning_gym_step(
    reasoning_dataset, num_samples, eval_mode, return_raw_completions
):
    train_dataset, test_dataset = reasoning_dataset
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
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
    completions = [torch.randint(0, 1000, (10, 356)) for _ in range(data_batch_size)]
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


@pytest.mark.parametrize("num_samples", [200])
@pytest.mark.parametrize("reset_dataloaders", [True, False])
@pytest.mark.parametrize("return_raw_completions", [True, False])
def test_reasoning_gym_reset(
    reasoning_dataset, num_samples, reset_dataloaders, return_raw_completions
):
    train_dataset, test_dataset = reasoning_dataset
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
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
@pytest.mark.parametrize("reset_dataloaders", [True, False])
def test_reasoning_gym_reset_dataloaders(
    reasoning_dataset, num_samples, reset_dataloaders
):
    train_dataset, test_dataset = reasoning_dataset
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
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
        env.test_dataloader_iter
    )  # use test_dataloader_iter as it is not shuffled
    env._reset_dataloaders()
    first_data_point_reset = next(env.test_dataloader_iter)
    for key1, key2 in zip(first_data_point.keys(), first_data_point_reset.keys()):
        if key1 == "tokenized_prompts":
            for item1, item2 in zip(
                first_data_point["tokenized_prompts"],
                first_data_point_reset["tokenized_prompts"],
            ):
                for key3, key4 in zip(item1.keys(), item2.keys()):
                    assert torch.equal(item1[key3], item2[key4])
        else:
            assert first_data_point[key1] == first_data_point_reset[key1]


@pytest.mark.parametrize("num_samples", [200])
def test_reset_warning(reasoning_dataset, num_samples):
    train_dataset, test_dataset = reasoning_dataset
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
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


@pytest.mark.parametrize("num_samples", [200])
def test_reasoning_gym_len(reasoning_dataset, num_samples):
    train_dataset, test_dataset = reasoning_dataset
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
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


@pytest.mark.parametrize("num_samples", [20])
def test_create_chat_collate_fn(reasoning_dataset, num_samples):
    """Test the create_chat_collate_fn method."""
    # Create a mock tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

    train_dataset, test_dataset = reasoning_dataset
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    data_batch_size = 8

    env = ReasoningGym(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        reward_fn=dummy_reward_fn,
        conversation_template=DUMMY_CONVERSATION_TEMPLATE,
        data_batch_size_per_gpu=data_batch_size,
    )

    # Create the collate function
    collate_fn = env.create_collate_fn(tokenizer)

    # Create a sample batch
    batch = [
        {"question": "What is 2+2?", "answer": "4"},
        {"question": "What is 3+3?", "answer": "6"},
    ]

    # Apply the collate function
    result = collate_fn(batch)

    # Verify the result structure
    assert isinstance(result, dict)
    assert "question" in result
    assert "answer" in result
    assert "tokenized_prompts" in result

    # Verify the content
    assert result["question"] == ["What is 2+2?", "What is 3+3?"]
    assert result["answer"] == ["4", "6"]
    assert len(result["tokenized_prompts"]) == 2

    # Verify each tokenized prompt
    for prompt in result["tokenized_prompts"]:
        assert isinstance(prompt, BatchEncoding)
        assert "input_ids" in prompt
        assert "attention_mask" in prompt
        assert isinstance(prompt["input_ids"], torch.Tensor)
        assert isinstance(prompt["attention_mask"], torch.Tensor)


@pytest.mark.parametrize("num_samples", [20])
@pytest.mark.parametrize("data_batch_size", [8, 10])
def test_reset_dataloaders_when_train_dataloader_exhausted(
    reasoning_dataset, num_samples, data_batch_size
):
    train_dataset, test_dataset = reasoning_dataset
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    env = ReasoningGym(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        reward_fn=dummy_reward_fn,
        conversation_template=DUMMY_CONVERSATION_TEMPLATE,
        data_batch_size_per_gpu=data_batch_size,
    )
    total_sampled = 0
    for _ in range(3):
        env._get_next_batch()
        total_sampled += data_batch_size

    assert env.num_epochs == 1


@pytest.mark.parametrize("num_samples", [20])
@pytest.mark.parametrize("data_batch_size", [8, 10])
def test_not_reset_dataloaders_when_test_dataloader_exhausted(
    reasoning_dataset, num_samples, data_batch_size
):
    train_dataset, test_dataset = reasoning_dataset
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    env = ReasoningGym(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        reward_fn=dummy_reward_fn,
        conversation_template=DUMMY_CONVERSATION_TEMPLATE,
        data_batch_size_per_gpu=data_batch_size,
    )
    total_sampled = 0
    env.reset()
    for _ in range(10):
        with env.eval_mode():
            env._get_next_batch()
            total_sampled += data_batch_size

    assert env.num_epochs == 0


def test_dummy_optimizer_init():
    """Test DummyOptimizer initialization."""
    params = [torch.tensor([1.0, 2.0, 3.0])]
    lr = 0.001
    optimizer = DummyOptimizer(params, lr)
    assert optimizer is not None


def test_dummy_optimizer_step():
    """Test DummyOptimizer step method raises RuntimeError."""
    params = [torch.tensor([1.0, 2.0, 3.0])]
    lr = 0.001
    optimizer = DummyOptimizer(params, lr)

    with pytest.raises(RuntimeError) as exc_info:
        optimizer.step()

    expected_message = (
        "DummyOptimizer is a placeholder optimizer and should not be used."
        "Please ensure you are calling accelerator.prepare() on the optimizer."
    )
    assert str(exc_info.value) == expected_message


def test_dummy_optimizer_zero_grad():
    """Test DummyOptimizer zero_grad method raises RuntimeError."""
    params = [torch.tensor([1.0, 2.0, 3.0])]
    lr = 0.001
    optimizer = DummyOptimizer(params, lr)

    with pytest.raises(RuntimeError) as exc_info:
        optimizer.zero_grad()

    expected_message = (
        "DummyOptimizer is a placeholder optimizer and should not be used."
        "Please ensure you are calling accelerator.prepare() on the optimizer."
    )
    assert str(exc_info.value) == expected_message


def test_dummy_optimizer_state_dict():
    """Test DummyOptimizer state_dict method raises RuntimeError."""
    params = [torch.tensor([1.0, 2.0, 3.0])]
    lr = 0.001
    optimizer = DummyOptimizer(params, lr)

    with pytest.raises(RuntimeError) as exc_info:
        optimizer.state_dict()

    expected_message = (
        "DummyOptimizer is a placeholder optimizer and should not be used."
        "Please ensure you are calling accelerator.prepare() on the optimizer."
    )
    assert str(exc_info.value) == expected_message


def test_dummy_optimizer_load_state_dict():
    """Test DummyOptimizer load_state_dict method raises RuntimeError."""
    params = [torch.tensor([1.0, 2.0, 3.0])]
    lr = 0.001
    optimizer = DummyOptimizer(params, lr)

    with pytest.raises(RuntimeError) as exc_info:
        optimizer.load_state_dict({})

    expected_message = (
        "DummyOptimizer is a placeholder optimizer and should not be used."
        "Please ensure you are calling accelerator.prepare() on the optimizer."
    )
    assert str(exc_info.value) == expected_message


@pytest.mark.parametrize("zero_stage", [0, 1, 2, 3])
def test_gather_if_zero3(zero_stage):
    """Test gather_if_zero3 context manager."""
    params = [torch.tensor([1.0, 2.0, 3.0])]

    @contextmanager
    def dummy_gather_parameters(*args, **kwargs):
        yield

    with patch(
        "deepspeed.zero.GatheredParameters", side_effect=dummy_gather_parameters
    ) as mock_gathered_parameters:
        with gather_if_zero3(zero_stage, params):
            assert mock_gathered_parameters.call_count == (zero_stage == 3)


@pytest.mark.parametrize("use_accelerator", [True, False])
@pytest.mark.parametrize("num_samples", [20])
def test_preference_gym_init(
    preference_dataset, accelerator_factory, use_accelerator, num_samples
):
    train_dataset, test_dataset = preference_dataset
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    data_batch_size = 8
    env = PreferenceGym(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        data_batch_size_per_gpu=data_batch_size,
        accelerator=accelerator_factory(use_accelerator),
    )
    assert env.name == "dummy_dataset"
    assert hasattr(env, "tokenizer")
    assert isinstance(env.train_dataloader, DataLoader)
    assert isinstance(env.test_dataloader, DataLoader)
    print("KEYS", next(env.train_dataloader_iter).keys())
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


@pytest.mark.parametrize("use_accelerator", [True, False])
@pytest.mark.parametrize("num_samples", [20])
def test_preference_gym_step(
    preference_dataset, accelerator_factory, use_accelerator, num_samples
):
    train_dataset, test_dataset = preference_dataset
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
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


@pytest.mark.parametrize("use_accelerator", [True, False])
@pytest.mark.parametrize("num_samples", [20])
def test_preference_gym_reset(
    preference_dataset, accelerator_factory, use_accelerator, num_samples
):
    train_dataset, test_dataset = preference_dataset
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
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
    preference_dataset, accelerator_factory, use_accelerator, num_samples
):
    train_dataset, test_dataset = preference_dataset
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
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
    preference_dataset, accelerator_factory, use_accelerator, num_samples
):
    train_dataset, test_dataset = preference_dataset
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
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
    preference_dataset, num_samples, accelerator_factory, use_accelerator
):
    train_dataset, test_dataset = preference_dataset
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
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


def test_get_state_dict():
    model = nn.Linear(10, 10)
    state_dict = get_state_dict(model)
    assert isinstance(state_dict, dict)
    for key, value in state_dict.items():
        assert isinstance(key, str)
        assert isinstance(value, torch.Tensor)


def test_preference_gym_max_context_length_error():
    train_dataset = Datasets.from_dict(
        {
            "prompt": ["This is a prompt that is longer than the max context length."],
            "chosen": ["This is an answer."],
            "rejected": ["This is an answer."],
        }
    )
    test_dataset = Datasets.from_dict(
        {
            "prompt": ["This is a normal length prompt"],
            "chosen": ["This is an answer."],
            "rejected": ["This is an answer."],
        }
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
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


def test_preference_gym_max_context_length_warning():
    train_dataset = Datasets.from_dict(
        {
            "prompt": [
                "This is a prompt that is longer than the max context length. This prompt really is a lot longer than the other one.",
                "This is a prompt that is shorter.",
            ],
            "chosen": ["This is an answer.", "This is an answer."],
            "rejected": ["This is an answer.", "This is an answer."],
        }
    )
    test_dataset = Datasets.from_dict(
        {
            "prompt": ["This is a normal length prompt"],
            "chosen": ["This is an answer."],
            "rejected": ["This is an answer."],
        }
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
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


def test_reasoning_gym_max_context_length_warning():
    train_dataset = Datasets.from_dict(
        {
            "question": [
                "This is a prompt that is longer than the max context length. This prompt really is a lot longer than the other one.",
                "This is a prompt that is shorter.",
            ],
            "answer": ["This is an answer.", "This is an answer."],
        }
    )
    test_dataset = Datasets.from_dict(
        {
            "question": ["This is a normal length prompt"],
            "answer": ["This is an answer."],
        }
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
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


def test_llm_utils_fallback_types_when_no_llm_dependencies():
    """Test that llm_utils sets type aliases to Any when HAS_LLM_DEPENDENCIES is False."""
    # Remove the module from cache to force reimport
    original_module = sys.modules.pop("agilerl.utils.llm_utils", None)

    try:
        # Patch HAS_LLM_DEPENDENCIES before reimporting
        with patch("agilerl.HAS_LLM_DEPENDENCIES", False):
            # Reimport the module - it will see HAS_LLM_DEPENDENCIES as False
            import agilerl.utils.llm_utils as llm_utils_reloaded

            # Verify the fallback type aliases are set to Any
            assert llm_utils_reloaded.AutoTokenizer is Any
            assert llm_utils_reloaded.PreTrainedModel is Any
            assert llm_utils_reloaded.BatchEncoding is Any
            assert llm_utils_reloaded.Dataset is Any
    finally:
        # Restore original module to avoid affecting other tests
        sys.modules["agilerl.utils.llm_utils"] = original_module

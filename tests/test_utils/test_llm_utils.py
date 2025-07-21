from unittest.mock import MagicMock

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from agilerl.utils.llm_utils import DummyOptimizer, HuggingFaceGym


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


class DummyDataset(Dataset):
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


def dummy_reward_fn(*args, **kwargs):
    return 1.0


def dummy_chat_template_fn_custom(q, a, tokenizer):
    """
    Chat template function for test_hugging_face_gym_reset_dataloaders, gives unique input_ids for each question so
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


@pytest.fixture
def dataset(num_samples):
    train_dataset = DummyDataset(int(num_samples * 0.8))
    test_dataset = DummyDataset(int(num_samples * 0.2))
    return train_dataset, test_dataset


@pytest.mark.parametrize("num_samples", [200])
def test_hugging_face_gym_init(dataset, num_samples):
    train_dataset, test_dataset = dataset
    tokenizer = DummyTokenizer()
    data_batch_size = 8
    env = HuggingFaceGym(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        reward_fn=dummy_reward_fn,
        apply_chat_template_fn=dummy_chat_template_fn,
        data_batch_size_per_gpu=data_batch_size,
    )
    assert env.name == "dummy_dataset"
    assert callable(env.reward_fn)
    assert isinstance(env.tokenizer, DummyTokenizer)
    assert isinstance(env.train_dataloader, DataLoader)
    assert isinstance(env.test_dataloader, DataLoader)
    assert list(next(env.train_dataloader_iter).keys()) == [
        "question",
        "answer",
        "tokenized_prompts",
    ]
    assert env.dataloader == env.train_dataloader_iter
    assert callable(env.apply_chat_template_fn)
    assert not env.reset_called
    assert isinstance(env.observation_space, gym.spaces.Space)
    assert np.all(env.observation_space.high == tokenizer.vocab_size - 1)
    assert isinstance(env.action_space, gym.spaces.Space)
    assert np.all(env.action_space.high == tokenizer.vocab_size - 1)
    assert not env.evaluation_mode
    assert env.data_batch_size_per_gpu == data_batch_size


@pytest.mark.parametrize("num_samples", [200])
@pytest.mark.parametrize("eval_mode", [True, False])
def test_hugging_face_gym_step(dataset, num_samples, eval_mode):
    train_dataset, test_dataset = dataset
    tokenizer = DummyTokenizer()
    data_batch_size = 8
    env = HuggingFaceGym(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        reward_fn=dummy_reward_fn,
        apply_chat_template_fn=dummy_chat_template_fn,
        data_batch_size_per_gpu=data_batch_size,
    )
    env.evaluation_mode = eval_mode
    env.reset()
    completions = [torch.randint(0, 1000, (10, 356)) for _ in range(data_batch_size)]
    tokenized_prompts, rewards = env.step(completions)
    assert isinstance(tokenized_prompts, list)
    assert isinstance(rewards, torch.Tensor)
    for prompt in tokenized_prompts:
        assert isinstance(prompt, dict)
        assert sorted(list(prompt.keys())) == ["attention_mask", "input_ids"]
        assert isinstance(prompt["attention_mask"], torch.Tensor)
        assert isinstance(prompt["input_ids"], torch.Tensor)


@pytest.mark.parametrize("num_samples", [200])
@pytest.mark.parametrize("reset_dataloaders", [True, False])
def test_hugging_face_gym_reset(dataset, num_samples, reset_dataloaders):
    train_dataset, test_dataset = dataset
    tokenizer = DummyTokenizer()
    data_batch_size = 8
    env = HuggingFaceGym(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        reward_fn=dummy_reward_fn,
        apply_chat_template_fn=dummy_chat_template_fn,
        data_batch_size_per_gpu=data_batch_size,
    )
    tokenized_prompts = env.reset(reset_dataloaders)
    assert isinstance(tokenized_prompts, list)
    for prompt in tokenized_prompts:
        assert isinstance(prompt, dict)
        assert sorted(list(prompt.keys())) == ["attention_mask", "input_ids"]
        assert isinstance(prompt["attention_mask"], torch.Tensor)
        assert isinstance(prompt["input_ids"], torch.Tensor)


@pytest.mark.parametrize("num_samples", [200])
@pytest.mark.parametrize("reset_dataloaders", [True, False])
def test_hugging_face_gym_reset_dataloaders(dataset, num_samples, reset_dataloaders):
    train_dataset, test_dataset = dataset
    tokenizer = DummyTokenizer()
    data_batch_size = 8
    env = HuggingFaceGym(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        reward_fn=dummy_reward_fn,
        apply_chat_template_fn=dummy_chat_template_fn_custom,
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
def test_reset_warning(dataset, num_samples):
    train_dataset, test_dataset = dataset
    tokenizer = DummyTokenizer()
    data_batch_size = 8
    env = HuggingFaceGym(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        reward_fn=dummy_reward_fn,
        apply_chat_template_fn=dummy_chat_template_fn,
        data_batch_size_per_gpu=data_batch_size,
    )
    with pytest.warns():
        env.reset()
        env.reset()


@pytest.mark.parametrize("num_samples", [200])
def test_hugging_face_gym_len(dataset, num_samples):
    train_dataset, test_dataset = dataset
    tokenizer = DummyTokenizer()
    data_batch_size = 8
    env = HuggingFaceGym(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        reward_fn=dummy_reward_fn,
        apply_chat_template_fn=dummy_chat_template_fn,
        data_batch_size_per_gpu=data_batch_size,
    )
    env.reset()
    assert len(env) == 200 * 0.8  # Length returns the training length
    with env.eval_mode():
        assert len(env) == 200 * 0.2


def test_create_chat_collate_fn():
    """Test the create_chat_collate_fn method."""
    # Create a mock tokenizer
    mock_tokenizer = MagicMock()

    # Create a mock chat template function
    def mock_chat_template(question, answer, tokenizer):
        return {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}

    # Create the collate function
    collate_fn = HuggingFaceGym.create_collate_fn(mock_tokenizer, mock_chat_template)

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
        assert isinstance(prompt, dict)
        assert "input_ids" in prompt
        assert "attention_mask" in prompt
        assert prompt["input_ids"] == [1, 2, 3]
        assert prompt["attention_mask"] == [1, 1, 1]


@pytest.mark.parametrize("num_samples", [20])
@pytest.mark.parametrize("data_batch_size", [8, 10])
def test_reset_dataloaders_when_dataloader_exhausted(
    dataset, num_samples, data_batch_size
):
    train_dataset, test_dataset = dataset
    tokenizer = DummyTokenizer()
    env = HuggingFaceGym(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        reward_fn=dummy_reward_fn,
        apply_chat_template_fn=dummy_chat_template_fn,
        data_batch_size_per_gpu=data_batch_size,
    )
    total_sampled = 0
    for _ in range(3):
        env._get_next_batch()
        total_sampled += data_batch_size

    assert env.num_dataset_passes == 1


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

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from agilerl.utils.llm_utils import HuggingFaceGym


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


def dummy_chat_template_fn(*args, **kwargs):
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
    assert list(next(env.train_dataloader_iter).keys()) == ["question", "answer"]
    assert next(env.test_dataloader_iter) == {
        "question": [f"This is question {i}?" for i in range(data_batch_size)],
        "answer": [f"This is answer {i}." for i in range(data_batch_size)],
    }
    assert env.dataloader == env.train_dataloader_iter
    assert callable(env.apply_chat_template_fn)
    assert not env.reset_called
    assert isinstance(env.observation_space, gym.spaces.Space)
    assert np.all(env.observation_space.high == tokenizer.vocab_size - 1)
    assert isinstance(env.action_space, gym.spaces.Space)
    assert np.all(env.action_space.high == tokenizer.vocab_size - 1)
    assert not env.eval_mode
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
    env.eval_mode = eval_mode
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
        apply_chat_template_fn=dummy_chat_template_fn,
        data_batch_size_per_gpu=data_batch_size,
    )
    first_data_point = next(
        env.test_dataloader_iter
    )  # use test_dataloader_iter as it is not shuffled
    env._reset_dataloaders()
    assert first_data_point == next(env.test_dataloader_iter)


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
    with env.eval():
        assert len(env) == 200 * 0.2

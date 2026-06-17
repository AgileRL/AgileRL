"""Tests for the gem-aligned V2 LLM envs (ReasoningGymV2, PreferenceGymV2)."""

import pytest
import torch

pytest.importorskip("datasets", reason="LLM dependencies not installed")

from torch.utils.data import Dataset
from transformers import AutoTokenizer

from agilerl.llm_envs import (
    PreferenceGym,
    PreferenceGymV2,
    ReasoningGym,
    ReasoningGymV2,
)
from tests import TINY_LLM_FIXTURE_PATH

DUMMY_CONVERSATION_TEMPLATE = [
    {
        "role": "system",
        "content": "question: {question}\nanswer: {answer}",
    },
]


def dummy_reward_fn(*args, **kwargs):
    return 1.0


class Info:
    def __init__(self, name: str) -> None:
        self.dataset_name = name


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


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)


@pytest.fixture
def reasoning_env(tokenizer):
    return ReasoningGymV2(
        train_dataset=DummyReasoningDataset(160),
        test_dataset=DummyReasoningDataset(40),
        tokenizer=tokenizer,
        reward_fn=dummy_reward_fn,
        conversation_template=DUMMY_CONVERSATION_TEMPLATE,
        data_batch_size_per_gpu=8,
    )


@pytest.fixture
def preference_env(tokenizer):
    return PreferenceGymV2(
        train_dataset=DummyPreferenceDataset(160),
        test_dataset=DummyPreferenceDataset(40),
        tokenizer=tokenizer,
        data_batch_size_per_gpu=8,
    )


def test_v2_classes_are_subclasses():
    assert issubclass(ReasoningGymV2, ReasoningGym)
    assert issubclass(PreferenceGymV2, PreferenceGym)


class TestReasoningGymV2:
    def test_reset_returns_prompts_info_tuple(self, reasoning_env):
        out = reasoning_env.reset()
        assert isinstance(out, tuple)
        assert len(out) == 2
        prompts, info = out
        assert isinstance(prompts, list)
        assert info == {}
        assert reasoning_env.reset_called is True

    def test_step_scores_and_terminates(self, reasoning_env):
        reasoning_env.reset()
        last_prompts_id = id(reasoning_env.last_tokenized_prompts)
        completions = [torch.randint(0, 1000, (10, 356)) for _ in range(8)]
        out = reasoning_env.step(completions)
        assert isinstance(out, tuple)
        assert len(out) == 5
        obs, rewards, terminated, truncated, info = out
        assert obs == {}
        assert info == {}
        assert terminated is True
        assert truncated is False
        assert isinstance(rewards, torch.Tensor)
        assert rewards.shape == (8, 10)
        assert torch.all(rewards == 1.0)
        assert reasoning_env.reset_called is False
        # step() must NOT advance the dataloader.
        assert id(reasoning_env.last_tokenized_prompts) == last_prompts_id

    def test_reset_advances_step_does_not(self, reasoning_env):
        reasoning_env.reset()
        questions_after_first_reset = list(reasoning_env.questions)
        completions = [torch.randint(0, 1000, (10, 356)) for _ in range(8)]
        reasoning_env.step(completions)
        # step() leaves the current batch in place.
        assert list(reasoning_env.questions) == questions_after_first_reset
        reasoning_env.reset()
        # reset() advances to a new batch.
        assert list(reasoning_env.questions) != questions_after_first_reset

    def test_reset_dataloaders_warns(self, reasoning_env):
        with pytest.warns(UserWarning, match="reset_dataloaders=True"):
            reasoning_env.reset(reset_dataloaders=True)


class TestPreferenceGymV2:
    def test_reset_returns_prompts_info_tuple(self, preference_env):
        out = preference_env.reset()
        assert isinstance(out, tuple)
        assert len(out) == 2
        prompts, info = out
        assert isinstance(prompts, dict)
        assert {"chosen_input_ids", "rejected_input_ids"}.issubset(prompts.keys())
        assert info == {}

    def test_step_returns_none_reward_and_terminates(self, preference_env):
        preference_env.reset()
        out = preference_env.step()
        assert isinstance(out, tuple)
        assert len(out) == 5
        obs, reward, terminated, truncated, info = out
        assert obs == {}
        assert reward is None
        assert terminated is True
        assert truncated is False
        assert info == {}
        assert preference_env.reset_called is False

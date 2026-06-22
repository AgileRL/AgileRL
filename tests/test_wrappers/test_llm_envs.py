"""Tests for :mod:`agilerl.llm_envs` (preference and SFT dataset envs, rollout env)."""

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
    BatchIterationState,
    DatasetEnv,
    PreferenceGym,
    RolloutEnv,
    SFTGym,
    apply_chat_template,
    dataloader_shuffle_order,
)
from tests import TINY_LLM_FIXTURE_PATH


def test_wrappers_llm_envs_compat_module_warns_and_reexports():
    sys.modules.pop("agilerl.wrappers.llm_envs", None)
    with pytest.warns(FutureWarning, match="deprecated"):
        compat_module = importlib.import_module("agilerl.wrappers.llm_envs")
    from agilerl.llm_envs import PreferenceGym as NewPreferenceGym

    assert compat_module.PreferenceGym is NewPreferenceGym


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
        assert isinstance(env, DatasetEnv)
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


def test_rollout_shuffle_order_extends_without_rewriting_first_epoch():
    """Row lookups past the first epoch append fresh epochs, never overwrite epoch 0."""
    dataset_size, seed = 11, 7
    state = BatchIterationState(
        shuffle_order=dataloader_shuffle_order(dataset_size, seed, 1),
        seed=seed,
        dataset_size=dataset_size,
    )
    first_epoch = list(state.shuffle_order)
    # Force the order to grow across the epoch boundary.
    _ = state.row_index(dataset_size + 3)
    assert state.shuffle_order[:dataset_size] == first_epoch
    assert (
        dataloader_shuffle_order(dataset_size, seed, state.epochs_built)
        == (state.shuffle_order[: dataset_size * state.epochs_built])
    )


def test_rollout_prompt_is_templated_and_reward_scores_once():
    """reset() returns the templated prompt; step() scores via reward_fn and ends."""
    questions, answers = ["2+2", "3+5"], ["4", "8"]
    state = BatchIterationState(shuffle_order=[0, 1], seed=0, dataset_size=2)

    def reward_fn(completion, answer, _question):
        return 1.0 if answer in completion else 0.0

    def prompt_builder(question):
        return f"Q: {question}\nA:"

    env = RolloutEnv(
        max_turns=1,
        questions=questions,
        answers=answers,
        reward_fn=reward_fn,
        prompt_builder=prompt_builder,
    )

    prompt, info = env.reset(seed=0, row_index=state.row_for_seed(0))
    assert prompt == "Q: 2+2\nA:"
    assert info == {}

    _, reward, terminated, truncated, _ = env.step("the answer is 4")
    assert reward == 1.0
    assert terminated is True
    assert truncated is False

    # A wrong completion on the next row scores zero, still one turn.
    next_prompt, _ = env.reset(seed=1, row_index=state.row_for_seed(1))
    assert next_prompt == "Q: 3+5\nA:"
    _, wrong_reward, terminated, _, _ = env.step("definitely 99")
    assert wrong_reward == 0.0
    assert terminated is True


def test_rollout_eval_mode_draws_from_held_out_split():
    """Under eval_mode the env serves the test split, restoring the train split after."""
    env = RolloutEnv(
        max_turns=1,
        questions=["train-q"],
        answers=["train-a"],
        reward_fn=lambda c, a, q: 0.0,
        prompt_builder=lambda q: q,
        test_questions=["eval-q"],
        test_answers=["eval-a"],
    )
    with env.eval_mode():
        eval_prompt, _ = env.reset(seed=0, row_index=0)
    assert eval_prompt == "eval-q"

    train_prompt, _ = env.reset(seed=1, row_index=0)
    assert train_prompt == "train-q"

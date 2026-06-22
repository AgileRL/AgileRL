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
    DatasetEnv,
    LLMEnv,
    PreferenceGym,
    RolloutEnv,
    SFTGym,
    apply_chat_template,
    dataloader_shuffle_order,
)
from agilerl.llm_envs.rollout_env import (
    _default_prompt_builder,
    _extract_question_answer_columns,
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
    """The shuffle index list is deterministic and a full permutation of the rows."""
    from agilerl.llm_envs import dataloader_shuffle_order

    dataset_size = 16
    seed = 7
    order = dataloader_shuffle_order(dataset_size, seed)
    assert len(order) == dataset_size
    # One epoch's order is a full permutation of the dataset indices.
    assert sorted(order) == list(range(dataset_size))

    # Deterministic per seed; a different seed produces a different shuffle.
    assert order == dataloader_shuffle_order(dataset_size, seed)
    assert order != dataloader_shuffle_order(dataset_size, seed + 1)


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
                vec.trajectories[batch_idx * vec.group_size + g].env._last_row
                for g in range(vec.group_size)
            }
            assert len(group_rows) == 1, "group trajectories must share one row"
            reset_rows.append(group_rows.pop())
        per_reset_rows.append(reset_rows)
    return per_reset_rows


def test_batch_rollout_env_shuffle_is_group_consistent_full_permutation():
    """BatchRolloutEnv resolves one shuffled row per group; epochs are full perms."""
    from agilerl.llm_envs import BatchRolloutEnv

    dataset_size = 6
    questions = [f"q{i}" for i in range(dataset_size)]
    answers = [f"a{i}" for i in range(dataset_size)]

    class _RowRecordingEnv(RolloutEnv):
        def reset(self, seed=None, *, row_index=None):
            prompt, info = super().reset(seed=seed, row_index=row_index)
            self._last_row = row_index
            return prompt, info

    def _factory():
        return _RowRecordingEnv(
            questions=list(questions),
            answers=list(answers),
            reward_fn=lambda c, a, q: 0.0,
            prompt_builder=lambda q: q,
        )

    # batch_size * resets spans two full epochs of the 6-row dataset.
    batch_size, group_size = 3, 2
    vec = BatchRolloutEnv(
        env_factory=_factory, batch_size=batch_size, group_size=group_size
    )
    per_reset_rows = _collect_batch_rows(vec, num_resets=4, seed=7)

    # Flatten in cursor order: batch rows within a reset are consumed in order.
    flat = [row for reset_rows in per_reset_rows for row in reset_rows]
    assert len(flat) == 4 * batch_size  # 12 == two epochs of 6 rows
    for start in range(0, len(flat), dataset_size):
        epoch = flat[start : start + dataset_size]
        assert sorted(epoch) == list(range(dataset_size)), "each epoch is a full perm"

    # Same seed reproduces the same row sequence.
    vec_again = BatchRolloutEnv(
        env_factory=_factory, batch_size=batch_size, group_size=group_size
    )
    again = _collect_batch_rows(vec_again, num_resets=4, seed=7)
    assert again == per_reset_rows

    # A different seed yields a different ordering.
    vec_other = BatchRolloutEnv(
        env_factory=_factory, batch_size=batch_size, group_size=group_size
    )
    other = _collect_batch_rows(vec_other, num_resets=4, seed=8)
    assert other != per_reset_rows


def test_rollout_prompt_is_templated_and_reward_scores_once():
    """reset() returns the templated prompt; step() scores via reward_fn and ends."""
    questions, answers = ["2+2", "3+5"], ["4", "8"]

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

    prompt, info = env.reset(seed=0, row_index=0)
    assert prompt == "Q: 2+2\nA:"
    assert info == {}

    _, reward, terminated, truncated, _ = env.step("the answer is 4")
    assert reward == 1.0
    assert terminated is True
    assert truncated is False

    # A wrong completion on the next row scores zero, still one turn.
    next_prompt, _ = env.reset(seed=1, row_index=1)
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


def test_dataloader_shuffle_order_rejects_empty_dataset():
    """A non-positive dataset size has no valid permutation, so it is rejected."""
    with pytest.raises(ValueError, match="dataset_size must be > 0"):
        dataloader_shuffle_order(0, seed=0)


def test_rollout_standalone_cursor_walks_split_and_resets_on_switch():
    """With no ``row_index`` the env walks its active split via an internal cursor,
    resetting the cursor when the train/eval split changes."""
    env = RolloutEnv(
        questions=["q0", "q1"],
        answers=["a0", "a1"],
        reward_fn=lambda c, a, q: 0.0,
        prompt_builder=lambda q: q,
        test_questions=["e0"],
        test_answers=["ea0"],
    )
    # Standalone resets (row_index omitted) consume the train split sequentially,
    # wrapping modulo the split length.
    assert env.reset()[0] == "q0"
    assert env.reset()[0] == "q1"
    assert env.reset()[0] == "q0"  # wrapped back to the start of the split

    # Switching to the eval split resets the per-split cursor to its start.
    with env.eval_mode():
        assert env.reset()[0] == "e0"

    # Back on the train split the cursor restarts from the beginning.
    assert env.reset()[0] == "q0"


def test_extract_question_answer_columns_from_hf_style_dataset():
    """Column access (``dataset["question"]``) is preferred for HF-style datasets."""

    class _ColumnDataset:
        def __init__(self):
            self._cols = {"question": ["q0", "q1"], "answer": ["a0", "a1"]}

        def __getitem__(self, key):
            return self._cols[key]

    questions, answers = _extract_question_answer_columns(_ColumnDataset())
    assert questions == ["q0", "q1"]
    assert answers == ["a0", "a1"]


def test_extract_question_answer_columns_from_torch_style_dataset():
    """A per-row ``torch``-style dataset (string indexing raises) is read row by row."""

    class _RowDataset(Dataset):
        def __init__(self):
            self._rows = [
                {"question": "q0", "answer": "a0"},
                {"question": "q1", "answer": "a1"},
            ]

        def __len__(self):
            return len(self._rows)

        def __getitem__(self, index):
            return self._rows[index]

    questions, answers = _extract_question_answer_columns(_RowDataset())
    assert questions == ["q0", "q1"]
    assert answers == ["a0", "a1"]


def test_default_prompt_builder_formats_and_joins_template():
    """The builder formats each template message's content with the question
    (answer blank, mirroring generation) and joins the non-empty parts."""
    template = [
        {"role": "system", "content": "Solve it."},
        {"role": "user", "content": "Q: {question} A: {answer}"},
        {"role": "assistant", "content": ""},  # empty render is dropped
    ]
    build = _default_prompt_builder(template)
    assert build("2+2") == "Solve it.\nQ: 2+2 A: "


def test_llm_env_close_is_a_noop_by_default():
    """The base ``close`` releases nothing by default and returns ``None``."""

    class _MinimalEnv(LLMEnv):
        def reset(self, *args, **kwargs):
            return None

        def step(self, *args, **kwargs):
            return None

    assert _MinimalEnv().close() is None


def test_preference_sft_back_compat_modules_reexport_dataset_env_subclasses():
    """The back-compat shim modules re-export the descriptor-configured
    :class:`DatasetEnv` subclasses."""
    from agilerl.llm_envs.preference import PreferenceGym as ShimPreferenceGym
    from agilerl.llm_envs.sft import SFTGym as ShimSFTGym

    assert issubclass(ShimPreferenceGym, DatasetEnv)
    assert issubclass(ShimSFTGym, DatasetEnv)
    assert ShimPreferenceGym is PreferenceGym
    assert ShimSFTGym is SFTGym


def test_dataset_env_len_and_eval_mode_preserve_tokenized_prompts():
    """``__len__`` reflects the active split and ``eval_mode`` saves/restores
    ``last_tokenized_prompts`` around the held-out block."""
    train_dataset = DummyPreferenceDataset(6)
    test_dataset = DummyPreferenceDataset(2)
    tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
    env = PreferenceGym(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        data_batch_size_per_gpu=2,
    )

    # Stand-in for prompts cached on a real training step; eval_mode must not
    # clobber it for the surrounding train loop.
    sentinel = {"input_ids": torch.tensor([[1, 2, 3]])}
    env.last_tokenized_prompts = sentinel

    assert len(env) == 6  # train split length (evaluation_mode is False)
    with env.eval_mode():
        assert env.evaluation_mode is True
        assert len(env) == 2  # held-out split length
    assert env.evaluation_mode is False
    assert len(env) == 6  # restored to the train split

    # The cached prompts survive the eval block (restored, equal by value).
    assert torch.equal(env.last_tokenized_prompts["input_ids"], sentinel["input_ids"])

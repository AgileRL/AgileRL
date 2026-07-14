"""Tests for :mod:`agilerl.llm_envs` (preference and SFT dataset envs, rollout env)."""

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
    apply_chat_template,
)
from tests import TINY_LLM_FIXTURE_PATH
from tests.helpers.rollout_doubles import RolloutEnvDoubleMixin


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
        # ``DatasetEnv(objective="sft")``'s default ``response_column`` is "target";
        # the output batch is still keyed under "response" regardless of input name.
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


class TestDatasetEnvPreferenceInit:
    @pytest.mark.parametrize("use_accelerator", [True, False])
    @pytest.mark.parametrize("num_samples", [20])
    def test_preference_init(
        self,
        preference_dataset,
        accelerator_factory,
        use_accelerator,
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
        """A preference ``DatasetEnv`` raises AssertionError when the dataset lacks required features."""
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        good_dataset = HFDataset.from_dict(
            {"prompt": ["p"], "chosen": ["c"], "rejected": ["r"]},
        )
        # Has "prompt" (so the column filter in __init__ works) but missing "chosen"/"rejected"
        bad_dataset = HFDataset.from_dict({"prompt": ["p"], "other": ["o"]})
        with pytest.raises(AssertionError, match="must contain columns"):
            DatasetEnv(
                train_dataset=bad_dataset,
                test_dataset=good_dataset,
                tokenizer=tokenizer,
                objective="preference",
            )
        with pytest.raises(AssertionError, match="must contain columns"):
            DatasetEnv(
                train_dataset=good_dataset,
                test_dataset=bad_dataset,
                tokenizer=tokenizer,
                objective="preference",
            )


class TestDatasetEnvPreferenceStep:
    @pytest.mark.parametrize("use_accelerator", [True, False])
    @pytest.mark.parametrize("num_samples", [20])
    def test_preference_step_is_noop(
        self,
        preference_dataset,
        accelerator_factory,
        use_accelerator,
        num_samples,
    ):
        """``step`` returns ``None`` and never advances the dataset cursor."""
        train_dataset, test_dataset = preference_dataset
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        data_batch_size = 8
        env = DatasetEnv(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            tokenizer=tokenizer,
            objective="preference",
            data_batch_size_per_gpu=data_batch_size,
            accelerator=accelerator_factory(use_accelerator),
        )
        assert env.step() is None
        assert env.step(completions=torch.zeros(1)) is None
        assert env.num_epochs == 0  # step never advances the cursor


class TestDatasetEnvPreferenceReset:
    @pytest.mark.parametrize("use_accelerator", [True, False])
    @pytest.mark.parametrize("num_samples", [20])
    def test_preference_reset(
        self,
        preference_dataset,
        accelerator_factory,
        use_accelerator,
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

    @pytest.mark.parametrize("use_accelerator", [True, False])
    @pytest.mark.parametrize("num_samples", [20])
    def test_preference_reset_reset_dataloaders_warning(
        self,
        preference_dataset,
        accelerator_factory,
        use_accelerator,
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
            accelerator=accelerator_factory(use_accelerator),
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
    @pytest.mark.parametrize("use_accelerator", [True, False])
    def test_preference_reset_num_epochs(
        self,
        preference_dataset,
        num_samples,
        accelerator_factory,
        use_accelerator,
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
            accelerator=accelerator_factory(use_accelerator),
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
        # Inert calls must not advance the cursor or trip the rollover.
        env.step()
        env.step()
        assert env.num_epochs == 0
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
    @pytest.mark.parametrize("use_accelerator", [True, False])
    @pytest.mark.parametrize("num_samples", [20])
    def test_sft_init(
        self,
        sft_dataset,
        accelerator_factory,
        use_accelerator,
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
        assert not hasattr(env, "reset_called")
        assert env.data_batch_size_per_gpu == data_batch_size

    def test_sft_max_context_length_warning(self):
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
            env = DatasetEnv(
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                tokenizer=tokenizer,
                objective="sft",
                data_batch_size_per_gpu=8,
                max_context_length=10,
            )
        assert len(env.train_dataloader) == 1

    def test_sft_init_missing_features(self):
        """An SFT ``DatasetEnv`` raises AssertionError when the dataset lacks required features."""
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        good_dataset = HFDataset.from_dict({"prompt": ["p"], "target": ["r"]})
        # Has "prompt" (so the column filter in __init__ works) but missing "target"
        bad_dataset = HFDataset.from_dict({"prompt": ["p"], "other": ["o"]})
        with pytest.raises(AssertionError, match="must contain"):
            DatasetEnv(
                train_dataset=bad_dataset,
                test_dataset=good_dataset,
                tokenizer=tokenizer,
                objective="sft",
            )
        with pytest.raises(AssertionError, match="must contain"):
            DatasetEnv(
                train_dataset=good_dataset,
                test_dataset=bad_dataset,
                tokenizer=tokenizer,
                objective="sft",
            )


class TestDatasetEnvSFTStep:
    @pytest.mark.parametrize("num_samples", [20])
    @pytest.mark.parametrize("use_accelerator", [True, False])
    def test_sft_num_epochs_increment(
        self,
        sft_dataset,
        num_samples,
        accelerator_factory,
        use_accelerator,
    ):
        train_dataset, test_dataset = sft_dataset
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        env = DatasetEnv(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            tokenizer=tokenizer,
            objective="sft",
            data_batch_size_per_gpu=1,
            accelerator=accelerator_factory(use_accelerator),
        )
        while env.num_epochs == 0:
            env.reset()
        assert env.num_epochs == 1


class TestDatasetEnvSFTReset:
    @pytest.mark.parametrize("use_accelerator", [True, False])
    @pytest.mark.parametrize("num_samples", [20])
    def test_sft_reset_returns_batch_step_is_noop(
        self,
        sft_dataset,
        accelerator_factory,
        use_accelerator,
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
            accelerator=accelerator_factory(use_accelerator),
        )
        assert env.step() is None

        batch = env.reset()
        assert set(batch.keys()) == {
            "prompt",
            "prompt_lengths",
            "response",
            "input_ids",
            "attention_mask",
        }
        assert len(batch["prompt"]) == data_batch_size

    @pytest.mark.parametrize("use_accelerator", [True, False])
    @pytest.mark.parametrize("num_samples", [20])
    def test_sft_reset_dataloaders_warning(
        self,
        sft_dataset,
        accelerator_factory,
        use_accelerator,
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
            accelerator=accelerator_factory(use_accelerator),
        )
        env.reset()
        env.reset()
        env.reset(reset_dataloaders=True)
        assert env.num_epochs == 0

    def test_sft_response_column_chosen(self):
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
        env = DatasetEnv(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            tokenizer=tokenizer,
            objective="sft",
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
    """BatchRolloutEnv resolves one shuffled row per group; epochs are full perms."""
    from agilerl.llm_envs import BatchRolloutEnv

    dataset_size = 6

    class _RowRecordingEnv(RolloutEnvDoubleMixin):
        """Minimal pooled env recording the ``row_index`` BatchRolloutEnv assigns.

        This test exercises only the shuffle / cursor, so it needs the
        ``dataset_size`` BatchRolloutEnv shuffles over and a ``reset`` that records
        its row — not the full token machinery.
        """

        def __init__(self):
            self.dataset_size = dataset_size
            self._last_row = None
            self.done = False
            self.current_prompt = {}

        def reset(self, seed=None, *, row_index=None):
            del seed
            self._last_row = row_index
            self.done = False
            self.current_prompt = {}
            return self.current_prompt, {}

    def _factory():
        return _RowRecordingEnv()

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

"""Dataset-backed (teacher-forced) LLM env — one class, configured by descriptors.

A ``DatasetEnv`` is the no-generation half of the env taxonomy (see
``docs/design/llm-env-taxonomy.md`` in agilerl-integration): the completions are dataset
labels, scored in a single teacher-forced forward (SFT cross-entropy, DPO preference) with
no autoregressive rollout. The training regimes (preference / SFT) differ only by the
*required columns* and the *collate function* — descriptors, not subclasses — so they share
one class. ``PreferenceGym`` / ``SFTGym`` remain as thin back-compat subclasses.
"""

from __future__ import annotations

import copy
import warnings
from collections.abc import Callable
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import torch
from torch.utils.data import DataLoader

from agilerl.llm_envs.base import LLMEnv

if TYPE_CHECKING:
    from collections.abc import Generator

    from accelerate import Accelerator
    from datasets import Dataset
    from transformers import AutoTokenizer

    CollateBuilder = Callable[
        ["DatasetEnv", "AutoTokenizer", "int | None"],
        Callable[[list[dict[str, Any]]], dict[str, Any]],
    ]


class DatasetEnv(LLMEnv, gym.Env):
    """Teacher-forced, dataset-backed LLM env (no generation).

    The no-generation half of the env taxonomy: completions are dataset labels
    scored in a single teacher-forced forward (SFT cross-entropy, DPO preference)
    with no autoregressive rollout. The training regimes (preference / SFT) differ
    only by the *required columns* and the *collate function* — descriptors, not
    subclasses — configured via ``required_columns`` + a ``collate_builder`` (and
    optional ``response_column``). ``reset`` / ``step`` advance the seeded
    ``DataLoader`` (completions ignored); ``PreferenceGym`` / ``SFTGym`` remain as
    thin back-compat subclasses.
    """

    def __init__(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        tokenizer: AutoTokenizer,
        *,
        required_columns: set[str],
        collate_builder: CollateBuilder,
        response_column: str | None = None,
        data_batch_size_per_gpu: int = 8,
        accelerator: Accelerator | None = None,
        max_context_length: int | None = None,
        min_completion_length: int | None = None,
        seed: int = 42,
    ) -> None:
        """Build a dataset env over ``required_columns`` collated by ``collate_builder``."""
        self.required_columns = set(required_columns)
        self.response_column = response_column
        self._collate_builder = collate_builder
        for label, dataset in (("Train", train_dataset), ("Test", test_dataset)):
            assert self.required_columns.issubset(set(dataset.features.keys())), (
                f"{label} dataset must contain columns {sorted(self.required_columns)}."
            )

        self.name = train_dataset.info.dataset_name
        self.tokenizer = tokenizer
        self.data_batch_size_per_gpu = data_batch_size_per_gpu
        self.accelerator = accelerator
        self.min_completion_length = (
            0 if min_completion_length is None else min_completion_length
        )
        self.max_context_length = max_context_length
        self.seed = seed
        generator = torch.Generator().manual_seed(seed)
        custom_collate_fn = self.create_collate_fn(tokenizer)
        dataloader_kwargs = {"collate_fn": custom_collate_fn}
        train_dataset = self._filter_dataset_by_max_context_length(
            train_dataset,
            "train dataset",
        )
        test_dataset = self._filter_dataset_by_max_context_length(
            test_dataset,
            "test dataset",
        )
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=data_batch_size_per_gpu,
            shuffle=True,
            **dataloader_kwargs,
            generator=generator,
        )
        self.test_dataloader = DataLoader(
            test_dataset,
            batch_size=data_batch_size_per_gpu,
            shuffle=False,
            **dataloader_kwargs,
            generator=generator,
        )
        self.dataset_size = {
            "train": len(train_dataset),
            "test": len(test_dataset),
        }
        if self.accelerator is not None:
            self.train_dataloader = self.accelerator.prepare(self.train_dataloader)
            self.test_dataloader = self.accelerator.prepare(self.test_dataloader)
        self.train_dataloader_iter = iter(self.train_dataloader)
        self.test_dataloader_iter = iter(self.test_dataloader)
        self.dataloader = self.train_dataloader_iter
        self.reset_called = False
        self.evaluation_mode = False
        self.num_epochs = 0

    def create_collate_fn(
        self,
        tokenizer: AutoTokenizer,
        max_context_length: int | None = None,
    ) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
        """Delegate to the injected collate builder."""
        return self._collate_builder(self, tokenizer, max_context_length)

    def reset(self, reset_dataloaders: bool = False) -> Any:
        """Reset the environment and get the next batch from the dataloader."""
        if reset_dataloaders:
            self._reset_dataloaders()
            warnings.warn(
                "env.reset() called with reset_dataloaders=True, this will reset "
                "the dataloaders to the beginning of the dataset, proceed with caution.",
                stacklevel=2,
            )
        if self.reset_called:
            warnings.warn(
                "env.reset() called more than once sequentially, it should typically "
                "follow with env.step().",
                stacklevel=2,
            )
        self.reset_called = True
        return self._get_next_batch()

    def step(self, completions: torch.Tensor | None = None) -> Any:
        """Advance the iterator and return the next batch (``completions`` unused)."""
        self.reset_called = False
        return self._get_next_batch()

    def _get_next_batch(self) -> Any:
        try:
            batch = next(self.dataloader)
        except StopIteration:
            if not self.evaluation_mode:
                self.num_epochs += 1
            self._reset_dataloaders(
                reset_train=not self.evaluation_mode,
                reset_test=self.evaluation_mode,
            )
            return self._get_next_batch()
        return batch

    @contextmanager
    def eval_mode(self) -> Generator[None, None, None]:
        """Context manager to switch to evaluation mode."""
        self.dataloader = self.test_dataloader_iter
        self.evaluation_mode = True
        last_tokenized_prompts = None
        if hasattr(self, "last_tokenized_prompts"):
            last_tokenized_prompts = copy.deepcopy(self.last_tokenized_prompts)
        try:
            yield
        finally:
            self.dataloader = self.train_dataloader_iter
            self.evaluation_mode = False
            if last_tokenized_prompts is not None:
                self.last_tokenized_prompts = last_tokenized_prompts

    def __len__(self) -> int:
        """Return the length of the dataset."""
        if self.evaluation_mode:
            return len(self.test_dataloader.dataset)
        return len(self.train_dataloader.dataset)

    def _reset_dataloaders(
        self, reset_train: bool = True, reset_test: bool = True
    ) -> None:
        """Reset the dataloaders to the beginning of the dataset."""
        if reset_train:
            self.train_dataloader_iter = iter(self.train_dataloader)
        if reset_test:
            self.test_dataloader_iter = iter(self.test_dataloader)
        self.dataloader = (
            self.test_dataloader_iter
            if self.evaluation_mode
            else self.train_dataloader_iter
        )

    def _filter_dataset_by_max_context_length(
        self,
        dataset: Dataset,
        dataset_type: str | None = None,
    ) -> Dataset:
        """Filter the dataset by the max context length."""
        dataset_type = "dataset" if dataset_type is None else dataset_type
        filter_keyword = "prompt" if "prompt" in dataset.features else "question"
        if self.max_context_length is None or not isinstance(
            dataset[0][filter_keyword],
            str,
        ):
            return dataset
        filtered_dataset = dataset.filter(
            lambda x: (
                len(self.tokenizer.encode(x[filter_keyword]))
                <= self.max_context_length - self.min_completion_length
            ),
        )
        if len(filtered_dataset) == 0:
            msg = f"No samples left in the {dataset_type} after filtering by the max context length constraint, use a larger max context length."
            raise ValueError(msg)
        if (dataset_difference := len(dataset) - len(filtered_dataset)) > 0:
            warnings.warn(
                f"{dataset_difference} samples were filtered out of the {dataset_type} due to the max context length constraint.",
                stacklevel=2,
            )
        return filtered_dataset


def preference_collate_builder(
    env: DatasetEnv,
    tokenizer: AutoTokenizer,
    max_context_length: int | None = None,
) -> Callable[[list[dict[str, str]]], dict[str, Any]]:
    """Collate ``(prompt, chosen, rejected)`` preference triples (DPO)."""

    def collate_fn(batch: list[dict[str, str]]) -> dict[str, Any]:
        prompts = [item["prompt"] for item in batch]
        chosen = [item["chosen"] for item in batch]
        rejected = [item["rejected"] for item in batch]

        prompt_encodings = tokenizer(
            prompts,
            truncation=True,
            padding=False,
            add_special_tokens=True,
        )
        prompt_lengths = [len(ids) for ids in prompt_encodings["input_ids"]]

        if env.max_context_length is not None:
            chosen_enc = tokenizer(
                prompts,
                chosen,
                max_length=env.max_context_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            rejected_enc = tokenizer(
                prompts,
                rejected,
                max_length=env.max_context_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
        else:
            chosen_ids = tokenizer(prompts, chosen, truncation=True, padding=False)
            rejected_ids = tokenizer(prompts, rejected, truncation=True, padding=False)
            max_len = max(
                *(len(ids) for ids in chosen_ids["input_ids"]),
                *(len(ids) for ids in rejected_ids["input_ids"]),
            )
            chosen_enc = tokenizer(
                prompts,
                chosen,
                truncation=True,
                padding="max_length",
                max_length=max_len,
                return_tensors="pt",
            )
            rejected_enc = tokenizer(
                prompts,
                rejected,
                truncation=True,
                padding="max_length",
                max_length=max_len,
                return_tensors="pt",
            )

        return {
            "prompt": prompts,
            "prompt_lengths": prompt_lengths,
            "chosen": chosen,
            "rejected": rejected,
            "chosen_input_ids": chosen_enc["input_ids"],
            "chosen_attention_mask": chosen_enc["attention_mask"].long(),
            "rejected_input_ids": rejected_enc["input_ids"],
            "rejected_attention_mask": rejected_enc["attention_mask"].long(),
        }

    return collate_fn


def sft_collate_builder(
    env: DatasetEnv,
    tokenizer: AutoTokenizer,
    max_context_length: int | None = None,
) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    """Collate ``(prompt, response)`` pairs (supervised fine-tuning)."""
    response_column = env.response_column

    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        prompts = [item["prompt"] for item in batch]
        responses = [item[response_column] for item in batch]

        prompt_encodings = tokenizer(
            prompts,
            truncation=True,
            padding=False,
            add_special_tokens=True,
        )
        prompt_lengths = [len(ids) for ids in prompt_encodings["input_ids"]]

        pair_enc = tokenizer(
            prompts,
            responses,
            max_length=env.max_context_length,
            truncation=True,
            padding="longest",
            return_tensors="pt",
        )

        return {
            "prompt": prompts,
            "prompt_lengths": prompt_lengths,
            "response": responses,
            "input_ids": pair_enc["input_ids"],
            "attention_mask": pair_enc["attention_mask"].long(),
        }

    return collate_fn


class PreferenceGym(DatasetEnv):
    """Back-compat thin subclass: a :class:`DatasetEnv` for preference (DPO) datasets."""

    def __init__(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        tokenizer: AutoTokenizer,
        data_batch_size_per_gpu: int = 8,
        accelerator: Accelerator | None = None,
        max_context_length: int | None = None,
        min_completion_length: int | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__(
            train_dataset,
            test_dataset,
            tokenizer,
            required_columns={"prompt", "chosen", "rejected"},
            collate_builder=preference_collate_builder,
            data_batch_size_per_gpu=data_batch_size_per_gpu,
            accelerator=accelerator,
            max_context_length=max_context_length,
            min_completion_length=min_completion_length,
            seed=seed,
        )

    def step(self, completions: torch.Tensor | None = None) -> Any:
        """Return the next batch (``completions`` is unused)."""
        return super().step(completions)


class SFTGym(DatasetEnv):
    """Back-compat thin subclass: a :class:`DatasetEnv` for SFT datasets."""

    def __init__(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        tokenizer: AutoTokenizer,
        data_batch_size_per_gpu: int = 8,
        response_column: str = "target",
        accelerator: Accelerator | None = None,
        max_context_length: int | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__(
            train_dataset,
            test_dataset,
            tokenizer,
            required_columns={"prompt", response_column},
            collate_builder=sft_collate_builder,
            response_column=response_column,
            data_batch_size_per_gpu=data_batch_size_per_gpu,
            accelerator=accelerator,
            max_context_length=max_context_length,
            seed=seed,
        )

    def step(self, completions: torch.Tensor | None = None) -> Any:
        """Advance the data iterator and return the next batch."""
        return super().step(completions)

# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Dataset-backed LLM env for SFT and DPO.

Scores the model on the dataset's own completions in a single forward pass (no
generation): cross-entropy for SFT, chosen-vs-rejected preference for DPO.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Literal

import torch
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from collections.abc import Generator

    from datasets import Dataset
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    CollateBuilder = Callable[
        ["DatasetEnv", "PreTrainedTokenizerBase"],
        Callable[[list[dict[str, Any]]], dict[str, Any]],
    ]


class DatasetEnv:
    """Dataset-backed LLM env for SFT and DPO (no generation).

    :ivar dataset_size: ``{"train": N, "test": M}`` row counts after filtering.
    :ivar num_epochs: Full passes completed over the train split.
    """

    def __init__(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        tokenizer: PreTrainedTokenizerBase,
        *,
        objective: Literal["preference", "sft"],
        response_column: str = "response",
        chat_template_kwargs: dict[str, Any] | None = None,
        data_batch_size_per_gpu: int = 8,
        max_context_length: int | None = None,
        min_completion_length: int | None = None,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        """Build a dataset env for the selected ``objective``.

        One contiguous equal-size shard per rank, taken after context-length
        filtering so every rank agrees on the rows and epoch boundaries.

        :param train_dataset: Training split of prompt/label rows.
        :param test_dataset: Held-out split used under :meth:`eval_mode`.
        :param tokenizer: Tokenizer used by the collate function.
        :param objective: ``"preference"`` (DPO) or ``"sft"``.
        :param response_column: Target column for SFT rows.
        :param chat_template_kwargs: Extra kwargs for any chat-template render of
            the dataset's prompts (e.g. ``{"enable_thinking": False}``).
        :param data_batch_size_per_gpu: Batch size for the train/test dataloaders.
        :param max_context_length: Optional max prompt+completion token budget.
        :param min_completion_length: Minimum reserved completion token budget.
        :param seed: Seed for dataloader shuffling.
        :param rank: This process's shard index in ``[0, world_size)``.
        :param world_size: Number of data-parallel shards (``1`` = no sharding).
        """
        if world_size < 1:
            msg = f"world_size must be >= 1, got {world_size}."
            raise ValueError(msg)
        if not 0 <= rank < world_size:
            msg = f"rank must be in [0, {world_size}), got {rank}."
            raise ValueError(msg)
        if objective == "preference":
            required_columns = {"prompt", "chosen", "rejected"}
            collate_builder: CollateBuilder = preference_collate_builder
        elif objective == "sft":
            required_columns = {"prompt", response_column}
            collate_builder = sft_collate_builder
        else:
            msg = f"Unknown dataset objective {objective!r}; expected 'preference' or 'sft'."
            raise ValueError(msg)
        self.objective = objective
        self.required_columns = set(required_columns)
        self.response_column = response_column
        self._collate_builder = collate_builder
        for label, dataset in (("Train", train_dataset), ("Test", test_dataset)):
            if len(dataset) == 0:
                msg = f"{label} dataset is empty; each split needs at least one row."
                raise ValueError(msg)
            if not self.required_columns.issubset(set(dataset.features.keys())):
                msg = (
                    f"{label} dataset must contain columns "
                    f"{sorted(self.required_columns)}."
                )
                raise ValueError(msg)

        self.name = train_dataset.info.dataset_name
        self.tokenizer = tokenizer
        self.chat_template_kwargs: dict[str, Any] = dict(chat_template_kwargs or {})
        self.data_batch_size_per_gpu = data_batch_size_per_gpu
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.min_completion_length = (
            0 if min_completion_length is None else min_completion_length
        )
        self.max_context_length = max_context_length
        self.seed = seed
        generator = torch.Generator().manual_seed(seed)
        custom_collate_fn = self.create_collate_fn(tokenizer)
        train_dataset = self._filter_dataset_by_max_context_length(
            train_dataset,
            "train dataset",
        )
        test_dataset = self._filter_dataset_by_max_context_length(
            test_dataset,
            "test dataset",
        )
        train_dataset = self._shard_for_rank(train_dataset, "train dataset")
        test_dataset = self._shard_for_rank(test_dataset, "test dataset")
        self.train_dataloader = DataLoader(
            train_dataset,  # ty: ignore[invalid-argument-type]
            batch_size=data_batch_size_per_gpu,
            shuffle=True,
            collate_fn=custom_collate_fn,
            generator=generator,
        )
        self.test_dataloader = DataLoader(
            test_dataset,  # ty: ignore[invalid-argument-type]
            batch_size=data_batch_size_per_gpu,
            shuffle=False,
            collate_fn=custom_collate_fn,
        )
        self.dataset_size = {
            "train": len(train_dataset),
            "test": len(test_dataset),
        }
        self.train_dataloader_iter = iter(self.train_dataloader)
        self.test_dataloader_iter = iter(self.test_dataloader)
        self.dataloader = self.train_dataloader_iter
        self.evaluation_mode = False
        self.num_epochs = 0

    def create_collate_fn(
        self,
        tokenizer: PreTrainedTokenizerBase,
    ) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
        """Build the row-collation callable for the current objective.

        :param tokenizer: Tokenizer used to encode prompts and labels.
        :return: Callable mapping rows to a tensor batch dict.
        """
        return self._collate_builder(self, tokenizer)

    def reset(self, reset_dataloaders: bool = False) -> dict[str, Any]:
        """Return the next batch (walks the dataset), optionally rewinding dataloaders first.

        :param reset_dataloaders: Whether to rewind train/test iterators first.
        :return: Objective-specific collated batch from the active split.
        """
        if reset_dataloaders:
            self._reset_dataloaders()
            self.num_epochs = 0
        return self._get_next_batch()

    def _get_next_batch(self) -> dict[str, Any]:
        """Read one batch and cycle dataloaders at split boundaries."""
        try:
            return next(self.dataloader)
        except StopIteration:
            if not self.evaluation_mode:
                self.num_epochs += 1
            self._reset_dataloaders(
                reset_train=not self.evaluation_mode,
                reset_test=self.evaluation_mode,
            )
            return next(self.dataloader)

    @contextmanager
    def eval_mode(self) -> Generator[None, None, None]:
        """Temporarily switch reads to the held-out split, restoring the prior mode.

        Saves/restores the entry mode so nested probes don't flip an outer eval context.
        """
        previous_mode = self.evaluation_mode
        self.dataloader = self.test_dataloader_iter
        self.evaluation_mode = True
        try:
            yield
        finally:
            self.evaluation_mode = previous_mode
            # Repoint at the live iterator; an inner reset may have rebuilt it.
            self.dataloader = (
                self.test_dataloader_iter
                if previous_mode
                else self.train_dataloader_iter
            )

    def __len__(self) -> int:
        """Return row count for the currently active split."""
        return self.dataset_size["test" if self.evaluation_mode else "train"]

    def _reset_dataloaders(
        self, reset_train: bool = True, reset_test: bool = True
    ) -> None:
        """Rebuild train/test iterators and refresh the active iterator pointer."""
        if reset_train:
            self.train_dataloader_iter = iter(self.train_dataloader)
        if reset_test:
            self.test_dataloader_iter = iter(self.test_dataloader)
        self.dataloader = (
            self.test_dataloader_iter
            if self.evaluation_mode
            else self.train_dataloader_iter
        )

    def _shard_for_rank(self, dataset: Dataset, dataset_type: str) -> Dataset:
        """This rank's contiguous, equal-size shard of ``dataset`` (identity when unsharded).

        Shards are truncated to ``len // world_size`` rows so every rank walks the same
        number of batches per epoch; the remainder rows (< ``world_size``) are dropped.
        """
        if self.world_size == 1:
            return dataset
        shard_size = len(dataset) // self.world_size
        if shard_size == 0:
            msg = (
                f"rank {self.rank} of {self.world_size} gets an empty shard of the "
                f"{len(dataset)}-row {dataset_type}; reduce world_size."
            )
            raise ValueError(msg)
        start = self.rank * shard_size
        return dataset.select(range(start, start + shard_size))

    def _filter_dataset_by_max_context_length(
        self,
        dataset: Dataset,
        dataset_type: str | None = None,
    ) -> Dataset:
        """Drop rows where ``prompt_len > max_context_length - min_completion_length``; warns if any."""
        dataset_type = "dataset" if dataset_type is None else dataset_type
        if self.max_context_length is None or not isinstance(
            dataset[0]["prompt"],
            str,
        ):
            return dataset
        max_prompt_tokens = self.max_context_length - self.min_completion_length
        filtered_dataset = dataset.filter(
            lambda x: len(self.tokenizer.encode(x["prompt"])) <= max_prompt_tokens,
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


def _prompt_lengths(
    tokenizer: PreTrainedTokenizerBase, prompts: list[str]
) -> list[int]:
    """Unpadded prompt token lengths, for masking the prompt span of a pair encoding."""
    enc = tokenizer(prompts, truncation=True, padding=False, add_special_tokens=True)
    return [len(ids) for ids in enc["input_ids"]]


def preference_collate_builder(
    env: DatasetEnv,
    tokenizer: PreTrainedTokenizerBase,
) -> Callable[[list[dict[str, str]]], dict[str, Any]]:
    """Build a collate function for ``(prompt, chosen, rejected)`` DPO rows."""

    def collate_fn(batch: list[dict[str, str]]) -> dict[str, Any]:
        """Tokenize preference triples into chosen/rejected paired tensors."""
        prompts = [item["prompt"] for item in batch]
        chosen = [item["chosen"] for item in batch]
        rejected = [item["rejected"] for item in batch]

        prompt_lengths = _prompt_lengths(tokenizer, prompts)

        eos = tokenizer.eos_token or ""
        chosen_eos = [completion + eos for completion in chosen]
        rejected_eos = [completion + eos for completion in rejected]

        if env.max_context_length is not None:
            chosen_enc = tokenizer(
                prompts,
                chosen_eos,
                max_length=env.max_context_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            rejected_enc = tokenizer(
                prompts,
                rejected_eos,
                max_length=env.max_context_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
        else:
            chosen_ids = tokenizer(prompts, chosen_eos, truncation=True, padding=False)
            rejected_ids = tokenizer(
                prompts, rejected_eos, truncation=True, padding=False
            )
            max_len = max(
                *(len(ids) for ids in chosen_ids["input_ids"]),
                *(len(ids) for ids in rejected_ids["input_ids"]),
            )
            chosen_enc = tokenizer.pad(
                chosen_ids,
                padding="max_length",
                max_length=max_len,
                return_tensors="pt",
            )
            rejected_enc = tokenizer.pad(
                rejected_ids,
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
    tokenizer: PreTrainedTokenizerBase,
) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    """Build a collate function for ``(prompt, response)`` SFT rows."""
    response_column = env.response_column

    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Tokenize SFT prompt/response rows into a padded tensor batch."""
        prompts = [item["prompt"] for item in batch]
        responses = [item[response_column] for item in batch]

        prompt_lengths = _prompt_lengths(tokenizer, prompts)

        eos = tokenizer.eos_token or ""
        pair_enc = tokenizer(
            prompts,
            [response + eos for response in responses],
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

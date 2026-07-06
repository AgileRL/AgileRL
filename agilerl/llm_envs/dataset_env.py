"""Dataset-backed LLM env for SFT and DPO — one class, selected by ``objective``.

A ``DatasetEnv`` serves batches straight from a labelled dataset: the model is scored
on the dataset's own completions in a single forward pass (cross-entropy for SFT,
chosen-vs-rejected preference for DPO), so nothing is generated — unlike a
``RolloutEnv``, where the model produces the text being scored. SFT and DPO differ
only by the *required columns* and the *collate function*, so they share one class and
are picked with the ``objective`` argument rather than separate subclasses.
"""

from __future__ import annotations

import copy
import warnings
from collections.abc import Callable
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Literal

import gymnasium as gym
import torch
from torch.utils.data import DataLoader

from agilerl.llm_envs.base import LLMEnv
from agilerl.utils.distributed import shard_dataloader_kwargs

if TYPE_CHECKING:
    from collections.abc import Generator

    from datasets import Dataset
    from transformers import AutoTokenizer

    CollateBuilder = Callable[
        ["DatasetEnv", "AutoTokenizer"],
        Callable[[list[dict[str, Any]]], dict[str, Any]],
    ]


class DatasetEnv(LLMEnv, gym.Env):
    """Dataset-backed LLM env for SFT and DPO (no generation).

    Serves batches straight from a labelled dataset: the model is scored on the
    dataset's own completions in a single forward pass (cross-entropy for SFT,
    chosen-vs-rejected preference for DPO) — nothing is generated. The two
    objectives differ only by the *required columns* and the *collate function*,
    selected with ``objective``:

    * ``objective="preference"`` (DPO) — requires ``prompt`` / ``chosen`` / ``rejected``.
    * ``objective="sft"`` — requires ``prompt`` and ``response_column`` (default
      ``"target"``).

    ``reset`` / ``step`` advance the seeded ``DataLoader`` (completions ignored).

    :ivar dataset_size: ``{"train": N, "test": M}`` row counts after filtering.
    :vartype dataset_size: dict[str, int]
    :ivar num_epochs: Number of full passes completed over the train split.
    :vartype num_epochs: int
    """

    def __init__(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        tokenizer: AutoTokenizer,
        *,
        objective: Literal["preference", "sft"],
        response_column: str = "target",
        data_batch_size_per_gpu: int = 8,
        max_context_length: int | None = None,
        min_completion_length: int | None = None,
        seed: int = 42,
    ) -> None:
        """Build a dataset env for the selected ``objective``.

        :param train_dataset: Training split containing prompt/label rows.
        :type train_dataset: Dataset
        :param test_dataset: Held-out split used under :meth:`eval_mode`.
        :type test_dataset: Dataset
        :param tokenizer: Tokenizer used by the collate function.
        :type tokenizer: AutoTokenizer
        :param objective: Dataset objective: ``"preference"`` (DPO) or ``"sft"``.
        :type objective: Literal["preference", "sft"]
        :param response_column: Target column for SFT rows.
        :type response_column: str
        :param data_batch_size_per_gpu: Batch size used by train/test dataloaders.
        :type data_batch_size_per_gpu: int
        :param max_context_length: Optional max prompt+completion token budget.
        :type max_context_length: int | None
        :param min_completion_length: Minimum reserved completion token budget.
        :type min_completion_length: int | None
        :param seed: Random seed used for dataloader shuffling.
        :type seed: int
        """
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
            assert self.required_columns.issubset(set(dataset.features.keys())), (
                f"{label} dataset must contain columns {sorted(self.required_columns)}."
            )

        self.name = train_dataset.info.dataset_name
        self.tokenizer = tokenizer
        self.data_batch_size_per_gpu = data_batch_size_per_gpu
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
        # In distributed runs the datasets are sharded across ranks; each rank
        # draws ``data_batch_size_per_gpu`` samples per step from its shard.
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=data_batch_size_per_gpu,
            **shard_dataloader_kwargs(train_dataset, shuffle=True),
            **dataloader_kwargs,
            generator=generator,
        )
        self.test_dataloader = DataLoader(
            test_dataset,
            batch_size=data_batch_size_per_gpu,
            **shard_dataloader_kwargs(test_dataset, shuffle=False),
            **dataloader_kwargs,
            generator=generator,
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
        tokenizer: AutoTokenizer,
    ) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
        """Build the row-collation callable for the current objective.

        Collation reads the context budget from ``self.max_context_length``.

        :param tokenizer: Tokenizer used to encode prompts and labels.
        :type tokenizer: AutoTokenizer
        :return: Callable that maps raw dataset rows to a tensor batch dict.
        :rtype: Callable[[list[dict[str, Any]]], dict[str, Any]]
        """
        return self._collate_builder(self, tokenizer)

    def reset(self, reset_dataloaders: bool = False) -> Any:
        """Return the next batch, optionally rewinding dataloaders first.

        ``reset`` is the sole data-advancing call, returning the next collated
        batch from the active split. Calling it repeatedly walks the dataset --
        this is the expected training pattern, mirroring how ``reset`` begins
        each iteration for a ``RolloutEnv``.

        :param reset_dataloaders: Whether to rewind train/test iterators first.
        :type reset_dataloaders: bool
        :return: Objective-specific collated batch from the active split.
        :rtype: Any
        """
        if reset_dataloaders:
            self._reset_dataloaders()
            warnings.warn(
                "env.reset() called with reset_dataloaders=True, this will reset "
                "the dataloaders to the beginning of the dataset, proceed with caution.",
                stacklevel=2,
            )
        return self._get_next_batch()

    def step(self, completions: torch.Tensor | None = None) -> None:
        """No-op: dataset training advances via :meth:`reset`.

        ``step`` exists only because :class:`~agilerl.llm_envs.base.LLMEnv`
        requires it, and returns ``None``. ``completions`` is accepted for
        trainer API parity and is ignored.

        :param completions: Unused; accepted for API parity.
        :type completions: torch.Tensor | None
        :return: Always ``None``.
        :rtype: None
        """

    def _get_next_batch(self) -> Any:
        """Read one batch and cycle dataloaders at split boundaries."""
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
        """Temporarily switch reads to the held-out split, restoring the prior mode.

        Restores whatever mode was active on entry (matching
        :meth:`LLMEnv.eval_mode`'s save/set/restore behaviour), so nested
        evaluation probes don't flip an outer eval context back to the train
        split. This also snapshots and
        restores ``last_tokenized_prompts`` when present, so train-loop prompt
        caches survive evaluation probes.
        """
        previous_mode = self.evaluation_mode
        self.dataloader = self.test_dataloader_iter
        self.evaluation_mode = True
        last_tokenized_prompts = None
        if hasattr(self, "last_tokenized_prompts"):
            last_tokenized_prompts = copy.deepcopy(self.last_tokenized_prompts)
        try:
            yield
        finally:
            self.evaluation_mode = previous_mode
            # Repoint at the live iterator for the restored mode (an inner reset
            # may have rebuilt either iterator, so don't restore a stale object).
            self.dataloader = (
                self.test_dataloader_iter
                if previous_mode
                else self.train_dataloader_iter
            )
            if last_tokenized_prompts is not None:
                self.last_tokenized_prompts = last_tokenized_prompts

    def __len__(self) -> int:
        """Return row count for the currently active split."""
        if self.evaluation_mode:
            return len(self.test_dataloader.dataset)
        return len(self.train_dataloader.dataset)

    def _reset_dataloaders(
        self, reset_train: bool = True, reset_test: bool = True
    ) -> None:
        """Rebuild train/test iterators and refresh the active iterator pointer."""
        if reset_train:
            # Reshuffle the distributed shard each epoch.
            sampler = getattr(self.train_dataloader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(self.num_epochs)
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
        """Drop rows whose tokenized prompt would exceed the context budget.

        The filter keeps rows where ``prompt_len <= max_context_length -
        min_completion_length`` and warns when any rows are removed.
        """
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
) -> Callable[[list[dict[str, str]]], dict[str, Any]]:
    """Build a collate function for ``(prompt, chosen, rejected)`` DPO rows."""

    def collate_fn(batch: list[dict[str, str]]) -> dict[str, Any]:
        """Tokenize preference triples into chosen/rejected paired tensors."""
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
) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    """Build a collate function for ``(prompt, response)`` SFT rows."""
    response_column = env.response_column

    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Tokenize SFT prompt/response rows into a padded tensor batch."""
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

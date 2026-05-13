"""Base helpers and classes for LLM gym-style environments."""

from __future__ import annotations

import copy
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import torch
from torch.utils.data import DataLoader

from agilerl.typing import ReasoningPrompts

if TYPE_CHECKING:
    from accelerate import Accelerator
    from datasets import Dataset
    from transformers import AutoTokenizer
    from transformers.tokenization_utils_base import BatchEncoding


def apply_chat_template(
    conversation_template: list[dict[str, str]],
    question: str,
    answer: str,
    tokenizer: AutoTokenizer,
) -> BatchEncoding:
    """Create and tokenize a chat template for a reasoning task.

    :param conversation_template: The conversation template to be tokenized.
    :type conversation_template: list[dict[str, str]]
    :param question: The question to be tokenized.
    :type question: str
    :param answer: The answer to be tokenized.
    :type answer: str
    :param tokenizer: The tokenizer to be used.
    :type tokenizer: AutoTokenizer
    :return: The tokenized prompt.
    :rtype: BatchEncoding
    """
    formatted_conversation = [
        {
            "role": msg["role"],
            "content": msg["content"].format(question=question, answer=answer),
        }
        for msg in conversation_template
    ]
    updated_prompt = tokenizer.apply_chat_template(
        formatted_conversation,
        tokenize=False,
        continue_final_message=True,
    )
    return tokenizer(
        [updated_prompt],
        return_tensors="pt",
        padding=True,
        padding_side="left",
        return_attention_mask=True,
    )


class HuggingFaceGym(gym.Env, ABC):
    """Abstract base class for HuggingFace Gymnasium environments."""

    def __init__(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        tokenizer: AutoTokenizer,
        conversation_template: list[dict[str, str]] | None,
        data_batch_size_per_gpu: int = 8,
        max_context_length: int | None = None,
        min_completion_length: int | None = None,
        accelerator: Accelerator | None = None,
        seed: int = 42,
    ) -> None:
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
        self.conversation_template = conversation_template
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

    @abstractmethod
    def reset(
        self,
        reset_dataloaders: bool = False,
    ) -> tuple[list[ReasoningPrompts], dict[str, Any]]:
        """Reset the environment and get the next batch of tokenized prompts."""

    @abstractmethod
    def step(
        self,
        completions: torch.Tensor,
    ) -> tuple[list[ReasoningPrompts], torch.Tensor]:
        """Take a step in a HuggingFaceGym environment."""

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

    @abstractmethod
    def create_collate_fn(
        self,
        tokenizer: AutoTokenizer,
        *args: Any,
        **kwargs: Any,
    ) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
        """Create a collate function for this environment."""

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


class IterablePromptBatchGym(HuggingFaceGym):
    """HuggingFaceGym whose ``step`` only advances the dataloader."""

    def reset(
        self,
        reset_dataloaders: bool = False,
    ) -> Any:
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

    def step(
        self,
        completions: torch.Tensor | None = None,
    ) -> Any:
        """Advance the iterator and return the next batch."""
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

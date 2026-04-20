"""Gymnasium-style environments for LLM training (SFT, preference, reasoning, multi-turn)."""

from __future__ import annotations

import copy
import os
import re
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import gymnasium as gym
import requests
import torch
from gem.tools.base_tool import BaseTool
from torch.utils.data import DataLoader

from agilerl.protocols import MultiTurnEnv
from agilerl.typing import PreferencePrompts, ReasoningPrompts, SFTPrompts
from agilerl.utils.algo_utils import stack_and_pad_experiences
from agilerl.utils.llm_utils import max_prompt_tokens_for_sliding_window

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
    """Abstract base class for HuggingFace Gymnasium environments.

    :param train_dataset: Train dataset to be loaded from HuggingFace datasets.
    :type train_dataset: Dataset
    :param test_dataset: Test dataset to be loaded from HuggingFace datasets.
    :type test_dataset: Dataset
    :param tokenizer: Tokenizer to be used for encoding and decoding the promåpts.
    :type tokenizer: AutoTokenizer
    :param custom_collate_fn: Custom collate function to be used for creating the batch, defaults to None
    :type custom_collate_fn: Callable, optional
    :param conversation_template: A structured conversation that acts as a base pattern for each data point.
    :type conversation_template: list[dict[str, str]]
    :param data_batch_size_per_gpu: DataLoader batch size, defaults to 8
    :type data_batch_size_per_gpu: int, optional
    :param max_context_length: Maximum context length, defaults to None
    :type max_context_length: int | None, optional
    :param min_completion_length: Minimum completion length, defaults to None
    :type min_completion_length: int, optional
    :param accelerator: Accelerator to be used for training, defaults to None
    :type accelerator: Accelerator, optional
    :param seed: Seed for the random number generator, defaults to 42
    :type seed: int, optional
    """

    def __init__(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        tokenizer: AutoTokenizer,
        conversation_template: list[dict[str, str]],
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
        """Take a step in a HuggingFaceGym environment, calculate rewards from completions generated from previous prompt and provide new batch
        of prompts.
        """

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
        """Create a collate function that applies the chat template to the batch of questions and answers."""

    def __len__(self) -> int:
        """Return the length of the dataset."""
        if self.evaluation_mode:
            return len(self.test_dataloader.dataset)
        return len(self.train_dataloader.dataset)

    def _reset_dataloaders(
        self, reset_train: bool = True, reset_test: bool = True
    ) -> None:
        """Reset the dataloaders to the beginning of the dataset.

        :param reset_train: Whether to reset the train dataloader, defaults to True
        :type reset_train: bool, optional
        :param reset_test: Whether to reset the test dataloader, defaults to True
        :type reset_test: bool, optional
        """
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
        """Filter the dataset by the max context length.

        :param dataset: Dataset to be filtered.
        :type dataset: Dataset
        :return: Filtered train and test datasets.
        :rtype: tuple[Dataset, Dataset]
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
            raise ValueError(
                msg,
            )
        if (dataset_difference := len(dataset) - len(filtered_dataset)) > 0:
            warnings.warn(
                f"{dataset_difference} samples were filtered out of the {dataset_type} due to the max context length constraint.",
                stacklevel=2,
            )
        return filtered_dataset


class ReasoningGym(HuggingFaceGym):
    """Class to convert HuggingFace datasets into Gymnasium style environment.

    :param train_dataset: Train dataset to be loaded from HuggingFace datasets.
    :type train_dataset: Dataset
    :param test_dataset: Test dataset to be loaded from HuggingFace datasets.
    :type test_dataset: Dataset
    :param tokenizer: Tokenizer to be used for encoding and decoding the prompts.
    :type tokenizer: AutoTokenizer
    :param reward_fn: Reward function for evaluating completions.
    :type reward_fn: Callable[..., float]
    :param conversation_template: A structured conversation that acts as a base pattern for each data point.
    :type conversation_template: list[dict[str, str]]
    :param data_batch_size_per_gpu: DataLoader batch size, defaults to 8
    :type data_batch_size_per_gpu: int, optional
    :param accelerator: Accelerator to be used for training, defaults to None
    :type accelerator: Accelerator, optional
    :param max_context_length: Maximum context length, defaults to None
    :type max_context_length: int | None, optional
    :param min_completion_length: Minimum completion length, defaults to 128
    :type min_completion_length: int, optional
    :param seed: Seed for the random number generator, defaults to 42
    :type seed: int, optional
    """

    def __init__(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        tokenizer: AutoTokenizer,
        reward_fn: Callable[[str, str, str], float],
        conversation_template: list[dict[str, str]],
        data_batch_size_per_gpu: int = 8,
        accelerator: Accelerator | None = None,
        return_raw_completions: bool = False,
        max_context_length: int | None = None,
        seed: int = 42,
    ) -> None:
        assert {"question", "answer"}.issubset(
            set(train_dataset.features.keys()),
        ), "Train dataset must contain 'question' and 'answer' features."
        assert {"question", "answer"}.issubset(
            set(test_dataset.features.keys()),
        ), "Train dataset must contain 'question' and 'answer' features."

        super().__init__(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            tokenizer=tokenizer,
            conversation_template=conversation_template,
            data_batch_size_per_gpu=data_batch_size_per_gpu,
            max_context_length=max_context_length,
            min_completion_length=0,
            accelerator=accelerator,
            seed=seed,
        )
        self.reward_fn = reward_fn
        self.return_raw_completions = return_raw_completions

    def step(
        self,
        completions: torch.Tensor,
    ) -> tuple[list[ReasoningPrompts], torch.Tensor]:
        """Take a step in the ReasoningGym environment, calculate rewards from completions generated from previous prompt and provide new batch
        of prompts.

        :param completions: Completion IDs generated by the agent.
        :type completions: torch.Tensor
        :return: New tokenized prompts and an information dictionary.
        :rtype: tuple[list[BatchEncoding], torch.Tensor]
        """
        self.reset_called = False
        rewards = self._decode_and_evaluate(completions)
        new_tokenized_prompts = self._get_next_batch()
        self.last_tokenized_prompts = new_tokenized_prompts
        return new_tokenized_prompts, rewards

    def reset(
        self,
        reset_dataloaders: bool = False,
    ) -> tuple[list[ReasoningPrompts], dict[str, Any]]:
        """Reset the environment and get the next batch of tokenized prompts.

        :param reset_dataloaders: Whether to reset the dataloaders, defaults to False
        :type reset_dataloaders: bool, optional
        :return: New tokenized prompts
        :rtype: tuple[list[BatchEncoding], dict[str, Any]]
        """
        if reset_dataloaders:
            self._reset_dataloaders()
            warnings.warn(
                "env.reset() called with reset_dataloaders=True, this will reset the dataloaders to the beginning of the dataset, proceed with caution.",
                stacklevel=2,
            )
        if self.reset_called:
            warnings.warn(
                "env.reset() called more than once sequentially, it should typically follow with env.step().",
                stacklevel=2,
            )
        self.reset_called = True
        new_tokenized_prompts = self._get_next_batch()
        self.last_tokenized_prompts = new_tokenized_prompts
        return new_tokenized_prompts

    def _decode_and_evaluate(self, completions: list[torch.Tensor]) -> torch.Tensor:
        """Decode the completions and evaluate the rewards.

        :param completions: Completion IDs generated by the agent.
        :type completions: list[torch.Tensor]
        :return: Rewards for the completions.
        :rtype: torch.Tensor
        """
        # This is for a batch of completions (prompt_batch x group_size), List of tensors of length batch size, each tensor is a group of answers
        total_rewards = []
        for idx, (group_completion, answer, question) in enumerate(
            zip(completions, self.answers, self.questions, strict=False),
        ):
            completion_to_decode = group_completion[
                :,
                self.last_tokenized_prompts[idx]["input_ids"].shape[1] :,
            ]

            # Vectorize this in the future
            decoded_group_completion = self.tokenizer.batch_decode(
                completion_to_decode,
                skip_special_tokens=True,
            )
            rewards = [
                self.reward_fn(completion, answer, question)
                for completion in decoded_group_completion
            ]
            total_rewards.append(rewards)
        return torch.tensor(total_rewards)

    def _get_next_batch(self) -> list[ReasoningPrompts]:
        """Get the next batch of tokenized prompts."""
        try:
            batch = next(self.dataloader)
            self.questions = batch["question"]
            self.answers = batch["answer"]

            returned_prompts = [
                {
                    "input_ids": returned_prompt["input_ids"],
                    "attention_mask": returned_prompt["attention_mask"],
                    "text": (
                        self.tokenizer.batch_decode(
                            returned_prompt["input_ids"],
                            skip_special_tokens=False,  # Needs to be False here as we need to provide context about user roles to the model
                            clean_up_tokenization_spaces=False,
                        )[0]
                        if self.return_raw_completions
                        else None
                    ),
                }
                for returned_prompt in batch["tokenized_prompts"]
            ]
        except StopIteration:
            if not self.evaluation_mode:
                self.num_epochs += 1

            self._reset_dataloaders(
                reset_train=not self.evaluation_mode,
                reset_test=self.evaluation_mode,
            )
            return self._get_next_batch()
        return returned_prompts

    def create_collate_fn(
        self,
        tokenizer: AutoTokenizer,
    ) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
        """Create a collate function that applies the chat template to the batch of questions and answers.

        :param tokenizer: Tokenizer to be used for encoding and decoding the prompts.
        :type tokenizer: AutoTokenizer
        :return: Collate function that applies the chat template to the batch of questions and answers.
        :rtype: Callable[[list[dict[str, Any]]], dict[str, Any]]
        """

        def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
            questions = [item["question"] for item in batch]
            answers = [item["answer"] for item in batch]

            # Apply chat template to all samples
            tokenized_prompts = [
                apply_chat_template(self.conversation_template, q, a, tokenizer)
                for q, a in zip(questions, answers, strict=False)
            ]

            return {
                "question": questions,
                "answer": answers,
                "tokenized_prompts": tokenized_prompts,  # Keep individual tokenized prompts
            }

        return collate_fn


class IterablePromptBatchGym(HuggingFaceGym):
    """HuggingFaceGym whose ``step`` only advances the dataloader (no reward decoding).

    Shared by Preference and SFT envs: same ``reset`` / ``step`` / epoch-wrapping
    iterator logic; subclasses differ in ``create_collate_fn`` and dataset checks.
    """

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
        """Advance the iterator and return the next batch.

        :param completions: Unused; kept for API compatibility with other Gym types.
        """
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


class PreferenceGym(IterablePromptBatchGym):
    """Class to convert HuggingFace preference datasets into Gymnasium style environment.

    :param dataset_name: Dataset name to be loaded from HuggingFace datasets.
    :type dataset_name: str
    :param tokenizer: Tokenizer to be used for encoding and decoding the prompts.
    :type tokenizer: AutoTokenizer
    :param reward_fn: Reward function for evaluating completions.
    :type reward_fn: Callable[..., float]
    :param data_batch_size_per_gpu: DataLoader batch size, defaults to 8
    :type data_batch_size_per_gpu: int, optional
    :param custom_collate_fn: Custom collate function to be used for creating the batch, defaults to None
    :type custom_collate_fn: Callable, optional
    :param accelerator: Accelerator to be used for training, defaults to None
    :type accelerator: Accelerator, optional
    :param max_context_length: Maximum context length, defaults to None
    :type max_context_length: int | None, optional
    :param min_completion_length: Minimum completion length, defaults to 128
    :type min_completion_length: int, optional
    """

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
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            tokenizer=tokenizer,
            conversation_template=None,
            data_batch_size_per_gpu=data_batch_size_per_gpu,
            max_context_length=max_context_length,
            min_completion_length=min_completion_length,
            accelerator=accelerator,
            seed=seed,
        )
        assert {"prompt", "chosen", "rejected"}.issubset(
            set(train_dataset.features.keys()),
        ), "Train dataset must contain 'prompt', 'chosen', and 'rejected' features."
        assert {"prompt", "chosen", "rejected"}.issubset(
            set(test_dataset.features.keys()),
        ), "Train dataset must contain 'prompt', 'chosen', and 'rejected' features."

    def reset(self, reset_dataloaders: bool = False) -> PreferencePrompts:
        """Reset the environment and get the next batch of tokenized prompts."""
        return super().reset(reset_dataloaders)

    def step(
        self,
        completions: torch.Tensor | None = None,
    ) -> PreferencePrompts:
        """Return the next batch (``completions`` is unused; API matches other gyms)."""
        return super().step(completions)

    def create_collate_fn(
        self,
        tokenizer: AutoTokenizer,
        max_context_length: int | None = None,
    ) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
        """Create a collate function that applies the chat template to the batch of questions and answers.

        :param tokenizer: Tokenizer to be used for encoding and decoding the prompts.
        :type tokenizer: AutoTokenizer
        :param max_context_length: Maximum context length, defaults to None
        :type max_context_length: int | None, optional
        :return: Collate function that applies the chat template to the batch of questions and answers.
        :rtype: Callable[[list[dict[str, Any]]], dict[str, Any]]
        """

        def collate_fn(batch: list[dict[str, str]]) -> dict[str, str]:
            prompts = [item["prompt"] for item in batch]
            chosen = [item["chosen"] for item in batch]
            rejected = [item["rejected"] for item in batch]

            # Tokenize prompts separately to get their lengths
            prompt_encodings = tokenizer(
                prompts,
                truncation=True,
                padding=False,
                add_special_tokens=True,
            )
            prompt_lengths = [len(ids) for ids in prompt_encodings["input_ids"]]

            # Tokenise chosen and rejected, padding both to the same length.
            # When max_context_length is set we pad directly to that fixed cap
            # (one __call__ each, no separate .pad() step).  When it is None we
            # need a first pass to find the joint max length before padding.
            if self.max_context_length is not None:
                chosen_enc = tokenizer(
                    prompts,
                    chosen,
                    max_length=self.max_context_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                rejected_enc = tokenizer(
                    prompts,
                    rejected,
                    max_length=self.max_context_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
            else:
                # First pass: get lengths without padding to compute joint max
                chosen_ids = tokenizer(prompts, chosen, truncation=True, padding=False)
                rejected_ids = tokenizer(
                    prompts, rejected, truncation=True, padding=False
                )
                max_len = max(
                    *(len(ids) for ids in chosen_ids["input_ids"]),
                    *(len(ids) for ids in rejected_ids["input_ids"]),
                )
                # Second pass: pad both to the joint max in a single __call__
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
                "chosen_input_ids": chosen_enc["input_ids"],  # [batch_size, max_len]
                "chosen_attention_mask": chosen_enc[
                    "attention_mask"
                ].long(),  # [batch_size, max_len]
                "rejected_input_ids": rejected_enc[
                    "input_ids"
                ],  # [batch_size, max_len]
                "rejected_attention_mask": rejected_enc[
                    "attention_mask"
                ].long(),  # [batch_size, max_len]
            }

        return collate_fn


class SFTGym(IterablePromptBatchGym):
    """Gymnasium-style environment for supervised fine-tuning (SFT) datasets.

    Each batch yields tokenised ``(prompt + response)`` pairs that can be
    consumed directly by :class:`~agilerl.algorithms.sft.SFT`.  The loss is
    computed on the response tokens only; prompt and padding positions are
    masked out inside :meth:`~agilerl.algorithms.sft.SFT.learn`.

    The dataset must have a ``"prompt"`` column and a response column (name
    set by ``response_column``, defaults to ``"response"``).  No
    negative/rejected response column is required — SFT only needs the target
    good response.

    .. tip::
        If your dataset uses DPO-style column names (``"chosen"`` /
        ``"rejected"``), pass ``response_column="chosen"`` to use it for SFT
        warm-up before a subsequent DPO fine-tuning stage.

    :param train_dataset: HuggingFace ``Dataset`` split used during training.
    :type train_dataset: Dataset
    :param test_dataset: HuggingFace ``Dataset`` split used for evaluation.
    :type test_dataset: Dataset
    :param tokenizer: Tokenizer for the target model.
    :type tokenizer: AutoTokenizer
    :param data_batch_size_per_gpu: DataLoader batch size per GPU, defaults to 8
    :type data_batch_size_per_gpu: int, optional
    :param response_column: Name of the dataset column containing the target
        response, defaults to ``"response"``
    :type response_column: str, optional
    :param accelerator: Accelerate handle for distributed training, defaults to
        None
    :type accelerator: Accelerator, optional
    :param max_context_length: Maximum tokenised sequence length.  Sequences
        longer than this are truncated, defaults to None
    :type max_context_length: int | None, optional
    :param seed: Random seed for dataset shuffling, defaults to 42
    :type seed: int, optional
    """

    def __init__(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        tokenizer: AutoTokenizer,
        data_batch_size_per_gpu: int = 8,
        response_column: str = "response",
        accelerator: Accelerator | None = None,
        max_context_length: int | None = None,
        seed: int = 42,
    ) -> None:
        self.response_column = response_column
        required = {"prompt", response_column}
        assert required.issubset(set(train_dataset.features.keys())), (
            f"Train dataset must contain {required} features."
        )
        assert required.issubset(set(test_dataset.features.keys())), (
            f"Test dataset must contain {required} features."
        )
        super().__init__(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            tokenizer=tokenizer,
            conversation_template=None,
            data_batch_size_per_gpu=data_batch_size_per_gpu,
            max_context_length=max_context_length,
            min_completion_length=None,
            accelerator=accelerator,
            seed=seed,
        )

    def reset(self, reset_dataloaders: bool = False) -> SFTPrompts:
        """Reset the environment and return the first batch of tokenised data."""
        return super().reset(reset_dataloaders)

    def step(
        self,
        completions: torch.Tensor | None = None,
    ) -> SFTPrompts:
        """Advance the data iterator and return the next batch."""
        return super().step(completions)

    def create_collate_fn(
        self,
        tokenizer: AutoTokenizer,
        max_context_length: int | None = None,
    ) -> Any:
        """Build a collate function that tokenises ``(prompt, response)`` pairs.

        :param tokenizer: Tokenizer to use.
        :type tokenizer: AutoTokenizer
        :param max_context_length: Truncation length, defaults to None
        :type max_context_length: int | None, optional
        :return: Collate function.
        :rtype: Callable
        """
        response_column = self.response_column

        def collate_fn(batch: list[dict[str, Any]]) -> SFTPrompts:
            prompts = [item["prompt"] for item in batch]
            responses = [item[response_column] for item in batch]

            # Tokenise prompts alone to measure prompt lengths (for loss masking)
            prompt_encodings = tokenizer(
                prompts,
                truncation=True,
                padding=False,
                add_special_tokens=True,
            )
            prompt_lengths = [len(ids) for ids in prompt_encodings["input_ids"]]

            # Tokenise prompt + response together in one __call__ to avoid the
            # "encode then pad" performance warning from fast tokenizers.
            # Use self.max_context_length (the instance attribute) rather than
            # the create_collate_fn parameter, which is never passed by the
            # base class and would silently default to None.
            pair_enc = tokenizer(
                prompts,
                responses,
                max_length=self.max_context_length,
                truncation=True,
                padding="longest",
                return_tensors="pt",
            )

            return {
                "prompt": prompts,
                "prompt_lengths": prompt_lengths,
                "response": responses,
                "input_ids": pair_enc["input_ids"],  # [B, max_len]
                "attention_mask": pair_enc["attention_mask"].long(),  # [B, max_len]
            }

        return collate_fn


class TokenObservationWrapper:
    """Token-level observation wrapper for multi-turn GEM environments.

    Wraps a GEM environment factory and provides a Gymnasium-like
    ``reset`` / ``step`` interface that operates on token IDs.  Maintains the
    growing token sequence, per-token action mask, and turn boundary tracking
    needed for LLMPPO's turn-level GAE.

    Each call to :meth:`reset` creates a fresh underlying environment via
    ``env_fn``, so the wrapper is reused across episodes just like a normal
    Gymnasium env.

    When ``apply_chat_template=True`` (the default for instruct models),
    observations are formatted as proper chat messages so the model receives
    input in the ``<|im_start|>user / assistant`` format it was trained on.

    :param env: GEM environment.
    :type env: GemEnv
    :param tokenizer: Tokenizer for encoding/decoding text.
    :type tokenizer: Any
    :param max_turns: Maximum number of interaction turns per episode.
    :type max_turns: int
    :param pad_id: Pad token ID used to mask padding positions, or ``None``.
    :type pad_id: int | None
    :param apply_chat_template: Whether to format observations using the
        tokenizer's chat template. Defaults to ``True``.
    :type apply_chat_template: bool
    :param max_model_len: Context length for sliding-window prompts; if ``None``,
        observations skip merging :meth:`build_model_prompt_fields`.
    :type max_model_len: int | None
    :param max_output_tokens: Max new tokens cap (same meaning as the policy);
        only used when ``max_model_len`` is set.
    :type max_output_tokens: int | None
    """

    def __init__(
        self,
        env: MultiTurnEnv,
        tokenizer: Any,
        max_turns: int,
        pad_id: int | None = None,
        apply_chat_template: bool = True,
        max_model_len: int | None = None,
        max_output_tokens: int | None = None,
    ) -> None:
        self._env = env
        self.tokenizer = tokenizer
        self.max_turns = max_turns
        self.pad_id = pad_id
        self.apply_chat_template = apply_chat_template
        self._sw_max_model_len = max_model_len
        self._sw_max_output_tokens = max_output_tokens
        self.full_ids: torch.Tensor | None = None
        self.turn_boundaries: list[tuple[int, int, int]] = []
        self.turn_rewards: list[float] = []
        self._turn_idx = 0
        self._prompt_text: str = ""
        self._gen_texts: list[str] = []
        self._feedback_texts: list[str] = []
        self._last_full_prompt_token_len: int | None = None

    @staticmethod
    def _format_obs(obs: str, info: dict[str, Any] | None) -> str:
        """Apply prefix/suffix from info dict to an observation string."""
        text = str(obs)
        if not info:
            return text
        prefix = info.get("prefix", "")
        suffix = info.get("suffix", "")
        if prefix:
            text = f"{prefix}{text}"
        if suffix:
            text = f"{text}\n{suffix}"
        return text

    def _tokenize_initial_prompt(self, obs_text: str) -> dict[str, torch.Tensor]:
        """Tokenize the initial observation, optionally with chat template."""
        if self.apply_chat_template:
            token_ids = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": obs_text}],
                tokenize=True,
                add_generation_prompt=True,
            )
            input_ids = torch.tensor([token_ids], dtype=torch.long)
            return {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
            }

        encoded = self.tokenizer(
            [obs_text],
            return_tensors="pt",
            padding=True,
            padding_side="left",
            return_attention_mask=True,
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }

    def _tokenize_feedback(self, feedback_text: str) -> torch.Tensor:
        """Tokenize feedback for the next turn, with chat turn boundaries."""
        if self.apply_chat_template:
            turn_boundary = (
                "<|im_end|>\n<|im_start|>user\n"
                + feedback_text
                + "<|im_end|>\n<|im_start|>assistant\n"
            )
            return torch.tensor(
                [self.tokenizer.encode(turn_boundary)],
                dtype=torch.long,
            )

        return torch.tensor(
            [self.tokenizer.encode(feedback_text)],
            dtype=torch.long,
        )

    def _policy_observation_from_state(self) -> dict[str, Any]:
        """Build observation dict for ``get_action`` from current ``full_ids``."""
        if self.full_ids is None:
            msg = "No prompt: reset() was never called"
            raise RuntimeError(
                msg,
            )
        self._last_full_prompt_token_len = int(self.full_ids.shape[1])
        prompt_ids_1d = self.full_ids[0]
        obs: dict[str, Any] = {
            "input_ids": self.full_ids,
            "attention_mask": torch.ones_like(self.full_ids),
            "text": self.tokenizer.decode(
                prompt_ids_1d.tolist(),
                skip_special_tokens=True,
            ),
        }
        if self._sw_max_model_len is not None:
            max_pt = max_prompt_tokens_for_sliding_window(
                self._sw_max_model_len,
                self._sw_max_output_tokens,
            )
            obs.update(self.build_model_prompt_fields(max_pt))
        return obs

    def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        """Create a fresh episode and return the policy-ready observation plus info.

        The observation includes ``input_ids``, ``attention_mask``, ``text``, and
        when ``max_model_len`` was set at construction, sliding-window fields from
        :meth:`build_model_prompt_fields`. Sets ``_initial_prompt_len`` and
        ``_last_full_prompt_token_len`` for :meth:`step`.

        :param seed: Optional RNG seed forwarded to the underlying env ``reset`` so
            parallel GRPO rollouts can share the same stochastic initial state.
        :type seed: int | None
        :return: ``(observation, info)`` with ``info`` from the underlying multi-turn env.
        :rtype: tuple[dict[str, Any], dict[str, Any]]
        """
        if seed is not None:
            try:
                obs_text, info = self._env.reset(seed=seed)
            except TypeError:
                obs_text, info = self._env.reset()
        else:
            obs_text, info = self._env.reset()
        obs_text = self._format_obs(obs_text, info)

        encoded = self._tokenize_initial_prompt(obs_text)
        self.full_ids = encoded["input_ids"]
        self._initial_prompt_len = int(encoded["input_ids"].shape[1])
        self.turn_boundaries = []
        self.turn_rewards = []
        self._turn_idx = 0
        self._prompt_text = obs_text
        self._gen_texts = []
        self._feedback_texts = []

        return self._policy_observation_from_state(), info

    def _step(
        self,
        full_completion_ids: torch.Tensor,
        gen_text: str,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Record a generation and step the underlying environment.

        :param full_completion_ids: Full token IDs (prompt + gen) of shape
            ``[1, seq_len]`` as returned by ``get_action``.
        :type full_completion_ids: torch.Tensor
        :param gen_text: Decoded generation text sent to the environment.
        :type gen_text: str
        :return: ``(next_observation, reward, terminated, truncated, info)``.
            ``next_observation`` is empty when the episode ended; otherwise it is
            the policy-ready dict from :meth:`_policy_observation_from_state`.
        :rtype: tuple[dict[str, Any], float, bool, bool, dict[str, Any]]
        """
        prompt_len = self.full_ids.shape[1]
        # Keep rollout state detached from any model computation graph.
        self.full_ids = full_completion_ids.detach().to(self.full_ids.device)
        gen_end = self.full_ids.shape[1]
        self.turn_boundaries.append((prompt_len, gen_end, self._turn_idx))
        self._gen_texts.append(gen_text)

        next_obs, reward, terminated, truncated, info = self._env.step(gen_text)
        self.turn_rewards.append(float(reward))
        self._turn_idx += 1

        prompt_dict: dict[str, Any] = {}
        if not (terminated or truncated):
            feedback_text = self._format_obs(next_obs, info)
            self._feedback_texts.append(feedback_text)
            feedback_ids = self._tokenize_feedback(feedback_text).to(
                self.full_ids.device
            )
            self.full_ids = torch.cat([self.full_ids, feedback_ids], dim=1)
            prompt_dict = self._policy_observation_from_state()

        return prompt_dict, reward, terminated, truncated, info

    def step(
        self,
        full_completion_ids: torch.Tensor,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Decode the generation from ``full_completion_ids`` and call :meth:`step`.

        Uses ``_last_full_prompt_token_len`` from the latest observation built by
        :meth:`reset` or :meth:`step`.

        :param full_completion_ids: Prompt + generated tokens, shape ``[1, seq]``.
        :type full_completion_ids: torch.Tensor
        :return: Same tuple as :meth:`_step`.
        :rtype: tuple[dict[str, Any], float, bool, bool, dict[str, Any]]
        """
        if self._last_full_prompt_token_len is None:
            msg = (
                "step() requires a prior reset() or step() "
                "that built a policy observation"
            )
            raise RuntimeError(
                msg,
            )
        pl = self._last_full_prompt_token_len
        gen_tokens = full_completion_ids[0, pl:]
        gen_text = self.tokenizer.decode(
            gen_tokens.tolist(),
            skip_special_tokens=True,
        )
        return self._step(full_completion_ids, gen_text)

    def build_model_prompt_fields(
        self,
        max_prompt_tokens: int,
    ) -> dict[str, Any]:
        """Build truncated prompt tensors for the LLM and a stitch prefix for full trajectories.

        Drops the oldest completed assistant+feedback turns after the initial
        user prompt until ``max_prompt_tokens`` is respected (or only the
        initial segment remains).

        :param max_prompt_tokens: Maximum number of tokens in the prompt passed
            to the model (after truncation).
        :type max_prompt_tokens: int
        :return: Dict with ``trajectory_input_ids``, ``trajectory_attention_mask``,
            ``trajectory_text``, ``stitch_prefix_ids`` (tokens removed between the
            initial segment and the kept tail), and ``model_window_initial_len``.
            Chronological reconstruction is
            ``cat(trajectory_input_ids[:, :I], stitch_prefix_ids, trajectory_input_ids[:, I:], dim=1)``
            where ``I`` is ``model_window_initial_len``.
        :rtype: dict[str, Any]
        """
        if self.full_ids is None:
            msg = "No prompt: reset() was never called"
            raise RuntimeError(
                msg,
            )

        full = self.full_ids
        initial_len = self._initial_prompt_len
        seq_len = full.shape[1]
        boundaries = self.turn_boundaries
        n = len(boundaries)

        if initial_len > max_prompt_tokens:
            msg = (
                f"Initial prompt ({initial_len} tokens) exceeds "
                f"max_prompt_tokens ({max_prompt_tokens})."
            )
            raise RuntimeError(
                msg,
            )

        k = 0
        while True:
            if k < n:
                drop_from = boundaries[k][0]
            elif n == 0:
                drop_from = initial_len
            else:
                drop_from = seq_len
            if drop_from >= seq_len:
                trunc = full[:, :initial_len].clone()
            else:
                trunc = torch.cat(
                    [full[:, :initial_len], full[:, drop_from:]],
                    dim=1,
                )
            if trunc.shape[1] <= max_prompt_tokens or k >= n:
                break
            k += 1

        if trunc.shape[1] > max_prompt_tokens:
            msg = (
                "Could not fit prompt even after dropping all post-initial turns; "
                f"trunc_len={trunc.shape[1]}, max_prompt_tokens={max_prompt_tokens}."
            )
            raise RuntimeError(
                msg,
            )

        if k < n:
            drop_from_final = boundaries[k][0]
        elif n == 0:
            drop_from_final = initial_len
        else:
            drop_from_final = seq_len
        stitch = full[:, initial_len:drop_from_final]

        prompt_ids_1d = trunc[0]
        trajectory_text = self.tokenizer.decode(
            prompt_ids_1d.tolist(),
            skip_special_tokens=True,
        )
        return {
            "trajectory_input_ids": trunc,
            "trajectory_attention_mask": torch.ones_like(trunc),
            "trajectory_text": trajectory_text,
            "stitch_prefix_ids": stitch,
            "initial_prompt_len": initial_len,
        }

    def get_episode_data(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build and return the full episode data for learning.

        :return: ``(full_ids, action_mask, turn_ids, turn_rewards)``
        :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        """
        if self.full_ids is None:
            msg = "No episode data: reset() was never called"
            raise RuntimeError(msg)

        seq_len = self.full_ids.shape[1]
        action_mask = torch.zeros(1, seq_len - 1, dtype=torch.bool)
        turn_ids = torch.full((1, seq_len - 1), -1, dtype=torch.long)

        for gen_start, gen_end, tidx in self.turn_boundaries:
            mask_start = gen_start - 1
            mask_end = gen_end - 1
            if mask_start >= 0 and mask_end <= seq_len - 1:
                action_mask[0, mask_start:mask_end] = True
                turn_ids[0, mask_start:mask_end] = tidx

        if self.pad_id is not None:
            pad_positions = self.full_ids[0, 1:] == self.pad_id
            action_mask[0, pad_positions] = False
            turn_ids[0, pad_positions] = -1

        turn_rewards = list(self.turn_rewards)
        while len(turn_rewards) < self.max_turns:
            turn_rewards.append(0.0)

        return (
            self.full_ids,
            action_mask,
            turn_ids,
            torch.tensor(turn_rewards, dtype=torch.float),
        )

    def close(self) -> None:
        """Close the wrapped environment when supported."""
        if hasattr(self._env, "close"):
            self._env.close()

    def get_debug_info(self) -> dict[str, Any]:
        """Return a dict of human-readable debug information for the episode."""
        if self.full_ids is None:
            return {"error": "No episode data"}

        full_ids, action_mask, _turn_ids, turn_rewards = self.get_episode_data()
        full_text = self.tokenizer.decode(
            full_ids[0].tolist(), skip_special_tokens=False
        )

        turn_details = []
        for gen_start, gen_end, tidx in self.turn_boundaries:
            gen_token_ids = full_ids[0, gen_start:gen_end].tolist()
            gen_text_decoded = self.tokenizer.decode(
                gen_token_ids, skip_special_tokens=True
            )
            turn_details.append(
                {
                    "turn": tidx,
                    "gen_start": gen_start,
                    "gen_end": gen_end,
                    "gen_len": gen_end - gen_start,
                    "gen_text_sent_to_env": self._gen_texts[tidx]
                    if tidx < len(self._gen_texts)
                    else None,
                    "gen_text_decoded_from_ids": gen_text_decoded,
                    "gen_token_ids": gen_token_ids[:50],
                    "reward": self.turn_rewards[tidx]
                    if tidx < len(self.turn_rewards)
                    else None,
                }
            )

        n_action_tokens = action_mask.sum().item()
        n_total_tokens = full_ids.shape[1]

        return {
            "n_turns": len(self.turn_boundaries),
            "n_total_tokens": n_total_tokens,
            "n_action_tokens": n_action_tokens,
            "action_fraction": n_action_tokens / max(n_total_tokens - 1, 1),
            "turn_rewards_raw": list(self.turn_rewards),
            "turn_rewards_padded": turn_rewards.tolist(),
            "prompt_text": self._prompt_text[:200],
            "full_text_preview": full_text[:500],
            "turn_details": turn_details,
            "feedback_texts": self._feedback_texts,
        }


# Adapted from https://github.com/PeterGriffinJin/Search-R1

# Timeout for search request in seconds
TIMEOUT = 5


class SearchTool(BaseTool):
    tool_type = "search"

    def __init__(self, num_workers=1, search_url=None, topk=3, timeout=TIMEOUT):
        super().__init__(num_workers)
        self.search_url = search_url
        self.topk = topk
        self.timeout = timeout
        self._search_url_resolved = self.search_url is not None

    def _parse_action(self, action: str) -> tuple[str, str, bool]:
        """Parse the action string to extract the <search> content and the full matched tag.
        Returns (content, parsed_action, is_valid)
        """
        # only take the first match
        pattern = r"<search>(.*?)</search>"
        match = re.search(pattern, action, re.DOTALL)
        if match:
            parsed_query = match.group(1).strip()
            parsed_action = action[: match.end()]  # including thinking process
            return parsed_query, parsed_action, True
        return "", "", False

    def _search(self, query: str):
        """Perform a search using the configured search_url.
        Returns a formatted string of search results.
        """
        if not self._search_url_resolved:
            self.search_url = self.search_url or os.environ.get("SEARCH_URL")
            self._search_url_resolved = True

        if not self.search_url:
            msg = "search_url must be provided for SearchTool."
            raise ValueError(msg)

        payload = {"q": query, "format": "json"}

        try:
            response = requests.get(
                self.search_url,
                params=payload,
                timeout=self.timeout,
            ).json()
            result = response["results"][: self.topk]
            response_string = ""
            for r in result:
                response_string += f"  {r.get('content', '')}\n"
            return response_string
        except Exception as e:
            return f"[SearchTool Error: {e}]"

    def _passages2string(self, result):
        format_reference = ""
        for idx, doc_item in enumerate(result):
            content = doc_item["document"]["contents"]
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx + 1}(Title: {title}) {text}\n"
        return format_reference

    def instruction_string(self) -> str:
        return (
            "You have access to a search engine to help answer questions.\n\n"
            "Additional instructions:\n"
            "- If your initial reasoning in <think> shows you lack some knowledge, explain what you need to find next inside a new <think> block.\n"
            "- Then issue a search query using:\n"
            "  <search> your query here </search>\n"
            "- The search engine will provide results inside:\n"
            "  <information> ... </information>\n"
            "- You may repeat the <think> and <search> steps as many times as needed.\n"
            "- When you are ready, give your final answer in:\n"
            "  <answer> your answer here </answer>"
        )

    def execute_action(self, action: str):
        """Execute the parsed action for the SearchTool.

        Args:
            action: The raw action string, typically containing a search query
                within <search>...</search> tags.

        Returns:
            observation: The formatted search result, or an empty string if invalid.
            done: Always False for search tool (search does not terminate the episode).
            valid: True if a valid search query was found and executed, False otherwise.
        """
        parsed_query, parsed_action, is_valid = self._parse_action(action)
        if not is_valid:
            # observation = "No valid search query found. Please provide your query within <search>...</search> tags."
            observation = ""
            valid = False
            has_error = True
        else:
            search_result = self._search(parsed_query)
            observation = f"\n\n<information>{search_result}</information>\n\n"
            valid = True
            has_error = "[SearchTool Error:" in search_result
        return valid, has_error, observation, parsed_action


class FormatRewardWrapper:
    """Wraps a multi-turn environment to give a small bonus for producing <answer> tags.

    Without this, the model gets zero reward when it answers without using
    the correct format, making it impossible to learn the format through RL
    alone (classic sparse-reward problem).
    """

    def __init__(self, env, format_bonus: float = 0.1):
        self._env = env
        self._format_bonus = format_bonus

    @property
    def format_bonus(self):
        return self._format_bonus

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def step(self, action: str, **kwargs):
        obs, reward, terminated, truncated, info = self._env.step(action, **kwargs)
        if terminated and "<answer>" in action and not info.get("correct", False):
            reward += self._format_bonus
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self._env.reset(**kwargs)


@dataclass
class Trajectory:
    """State for one environment rollout within a synchronized vector batch."""

    env: MultiTurnEnv
    batch_idx: int
    group_idx: int
    prompt: ReasoningPrompts
    done: bool


class TrajectoryBuffer:
    """Container for synchronized rollout trajectories.

    Keeps trajectory ordering by ``(batch_idx, group_idx)`` when preparing
    model prompts so generated completions can be mapped back deterministically.
    """

    def __init__(self, batch_size: int, group_size: int):
        """Initialize an empty trajectory buffer."""
        if batch_size <= 0:
            msg = f"batch_size must be > 0, got {batch_size}."
            raise ValueError(msg)
        if group_size <= 0:
            msg = f"group_size must be > 0, got {group_size}."
            raise ValueError(msg)
        self.batch_size = batch_size
        self.group_size = group_size
        self.trajectories: list[Trajectory] = []

    @property
    def is_initialized(self) -> bool:
        """Return ``True`` when the trajectory buffer is initialized."""
        return len(self.trajectories) == (self.batch_size * self.group_size)

    def add_trajectory(self, trajectory: Trajectory) -> None:
        """Append a trajectory to the buffer."""
        self.trajectories.append(trajectory)

    def clear(self) -> None:
        """Remove all stored trajectories."""
        self.trajectories.clear()

    def has_active(self) -> bool:
        """Return ``True`` when at least one trajectory is still active."""
        return any(not trajectory.done for trajectory in self.trajectories)

    def get_prompts(self) -> ReasoningPrompts | None:
        """Return stacked prompts for active trajectories.

        :return: Stacked prompt tensors for active trajectories, or ``None`` if
            all trajectories are complete.
        :rtype: ReasoningPrompts | None
        """
        active_trajectories = self.get_active_trajectories(sorted_by_index=True)
        if len(active_trajectories) == 0:
            return None
        return TrajectoryBuffer._stack_active_prompts(active_trajectories)

    def get_active_trajectories(
        self,
        *,
        sorted_by_index: bool = False,
    ) -> list[Trajectory]:
        """Get active (non-terminal) trajectories.

        :param sorted_by_index: If ``True``, sort by ``(batch_idx, group_idx)``.
        :type sorted_by_index: bool
        :return: Active trajectories.
        :rtype: list[Trajectory]
        """
        trajectories = [
            trajectory for trajectory in self.trajectories if not trajectory.done
        ]
        if sorted_by_index:
            trajectories.sort(key=lambda t: (t.batch_idx, t.group_idx))
        return trajectories

    def sort(self, key: Callable[[Trajectory], Any]) -> None:
        """Sort trajectories in place."""
        self.trajectories.sort(key=key)

    def __iter__(self) -> Iterator[Trajectory]:
        """Iterate over stored trajectories."""
        return iter(self.trajectories)

    def reset_trajectory(self, seed: int | None, env_idx: int) -> None:
        """Reset one trajectory in place.

        :param seed: Seed for the environment.
        :type seed: int | None
        :param env_idx: Index of the environment to reset.
        :type env_idx: int
        :return: None
        :rtype: None
        """
        if env_idx < 0 or env_idx >= len(self.trajectories):
            msg = (
                "env_idx out of bounds for trajectory buffer: "
                f"{env_idx} not in [0, {len(self.trajectories) - 1}]"
            )
            raise IndexError(msg)
        prompt_dict, _ = self.trajectories[env_idx].env.reset(seed=seed)
        self.trajectories[env_idx].prompt = prompt_dict
        self.trajectories[env_idx].done = False

    @staticmethod
    def _stack_active_prompts(trajectories: list[Trajectory]) -> ReasoningPrompts:
        """Stack prompt fields across active trajectories.

        :param trajectories: Active trajectories in the same order as the model
            input batch.
        :type trajectories: list[Trajectory]
        :return: Batched prompt tensors compatible with ``agent.get_action``.
        :rtype: ReasoningPrompts
        """
        if len(trajectories) == 0:
            msg = "Cannot stack prompts from an empty trajectory list."
            raise ValueError(msg)
        prompt_batch = [trajectory.prompt for trajectory in trajectories]
        stacked = cast("ReasoningPrompts", {})
        tensor_keys = (
            "input_ids",
            "attention_mask",
            "trajectory_input_ids",
            "trajectory_attention_mask",
            "stitch_prefix_ids",
        )
        for key in tensor_keys:
            values = [prompt.get(key) for prompt in prompt_batch]
            if all(v is None for v in values):
                continue
            if any(v is None for v in values):
                msg = (
                    f"Inconsistent prompt field '{key}' across active trajectories; "
                    "field must be present for all trajectories or none."
                )
                raise ValueError(msg)
            (stacked_tensor,) = stack_and_pad_experiences(values, padding_values=[0])
            if "attention_mask" in key:
                stacked_tensor = stacked_tensor.long()
            stacked[key] = stacked_tensor

        for required_key in ("input_ids", "attention_mask"):
            if required_key not in stacked:
                msg = f"Missing required prompt field '{required_key}'."
                raise ValueError(msg)

        initial_prompt_lengths = [
            prompt.get("initial_prompt_len") for prompt in prompt_batch
        ]
        if all(v is not None for v in initial_prompt_lengths):
            stacked["initial_prompt_len"] = torch.tensor(
                initial_prompt_lengths,
                dtype=torch.long,
            )

        return stacked

    def __getitem__(self, index: int) -> Trajectory:
        return self.trajectories[index]

    def __len__(self) -> int:
        return len(self.trajectories)


class SyncMultiTurnVecEnv:
    """Synchronous multi-turn vector environment for LLM rollouts."""

    def __init__(
        self,
        env_factory: Callable[..., MultiTurnEnv],
        batch_size: int,
        group_size: int,
        env_config: dict[str, Any] | None = None,
    ):
        """Create ``batch_size * group_size`` independent environments."""
        if batch_size <= 0:
            msg = f"batch_size must be > 0, got {batch_size}."
            raise ValueError(msg)
        if group_size <= 0:
            msg = f"group_size must be > 0, got {group_size}."
            raise ValueError(msg)
        if env_config is None:
            env_config = {}
        self.env_factory = env_factory
        self.env_config = env_config
        self.num_envs = batch_size * group_size
        self.batch_size = batch_size
        self.group_size = group_size
        self.trajectories = TrajectoryBuffer(batch_size, group_size)

    def reset(
        self,
        seed: int | None = None,
    ) -> ReasoningPrompts | None:
        """Reset all environments and initialize trajectories.

        Seeds are shared within each batch group (same seed for all members in
        a group), and incremented per batch index.
        """
        seed_base = seed
        for batch_idx in range(self.batch_size):
            batch_seed = None if seed_base is None else seed_base + batch_idx
            for group_idx in range(self.group_size):
                env_idx = batch_idx * self.group_size + group_idx
                if not self.trajectories.is_initialized:
                    env_i = self.env_factory(**self.env_config)
                    prompt_dict, _ = env_i.reset(seed=batch_seed)
                    self.trajectories.add_trajectory(
                        Trajectory(
                            env=env_i,
                            batch_idx=batch_idx,
                            group_idx=group_idx,
                            prompt=prompt_dict,
                            done=False,
                        )
                    )
                else:
                    self.trajectories.reset_trajectory(env_idx=env_idx, seed=batch_seed)
        return self.trajectories.get_prompts()

    def step(self, completion_ids: list[torch.Tensor]) -> ReasoningPrompts | None:
        """Step each active trajectory with its corresponding completion."""
        active = self.trajectories.get_active_trajectories(sorted_by_index=True)
        if len(completion_ids) != len(active):
            msg = (
                "Number of completions does not match number of active trajectories: "
                f"{len(completion_ids)} != {len(active)}"
            )
            raise RuntimeError(msg)
        for traj, completion in zip(active, completion_ids, strict=False):
            full_completion = completion
            if full_completion.dim() == 1:
                full_completion = full_completion.unsqueeze(0)
            next_prompt, _reward, terminated, truncated, _info = traj.env.step(
                full_completion,
            )
            traj.done = bool(terminated or truncated)
            if not traj.done:
                traj.prompt = next_prompt
        return self.trajectories.get_prompts()

    def close(self) -> None:
        """Close all underlying environments."""
        seen: set[int] = set()
        for traj in self.trajectories:
            env = traj.env
            env_id = id(env)
            if env_id in seen:
                continue
            seen.add(env_id)
            if hasattr(env, "close"):
                env.close()

    def get_trajectories(
        self,
    ) -> tuple[
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
        int,
    ]:
        """Collect complete episode tensors from all trajectories."""
        completion_ids_list: list[torch.Tensor] = []
        action_masks_list: list[torch.Tensor] = []
        all_turn_ids: list[torch.Tensor] = []
        all_rewards: list[torch.Tensor] = []
        batch_steps = 0
        self.trajectories.sort(key=lambda t: (t.batch_idx, t.group_idx))
        for traj in self.trajectories:
            ep_ids, action_mask, turn_ids, turn_rewards_t = traj.env.get_episode_data()
            completion_ids_list.append(ep_ids)
            action_masks_list.append(action_mask)
            all_turn_ids.append(turn_ids)
            all_rewards.append(turn_rewards_t)
            batch_steps += len(getattr(traj.env, "turn_boundaries", []))

        return (
            completion_ids_list,
            action_masks_list,
            all_turn_ids,
            all_rewards,
            batch_steps,
        )

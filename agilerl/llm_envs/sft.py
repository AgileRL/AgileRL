# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""SFT LLM Gym environment."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from agilerl.llm_envs.base import IterablePromptBatchGym
from agilerl.typing import SFTPrompts

if TYPE_CHECKING:
    import torch
    from datasets import Dataset
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class SFTGym(IterablePromptBatchGym[SFTPrompts]):
    """Gymnasium-style environment for supervised fine-tuning (SFT) datasets.

    :param train_dataset: The training dataset.
    :type train_dataset: Dataset
    :param test_dataset: The test dataset.
    :type test_dataset: Dataset
    :param tokenizer: The tokenizer.
    :type tokenizer: PreTrainedTokenizerBase
    :param data_batch_size_per_gpu: The batch size per GPU.
    :type data_batch_size_per_gpu: int
    :param response_column: The column name for the response in the dataset.
    :type response_column: str
    :param max_context_length: The maximum context length for the LLM model.
    :type max_context_length: int | None
    :param seed: The seed for the random number generator for the environment and the dataloaders.
    :type seed: int
    """

    def __init__(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        tokenizer: PreTrainedTokenizerBase,
        data_batch_size_per_gpu: int = 8,
        response_column: str = "target",
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
            seed=seed,
        )

    def reset(self, reset_dataloaders: bool = False) -> SFTPrompts:
        """Reset the environment and return the first batch of tokenised data.

        :param reset_dataloaders: Whether to reset the dataloaders.
        :type reset_dataloaders: bool
        :return: The first batch of tokenised data.
        :rtype: SFTPrompts
        """
        return super().reset(reset_dataloaders)

    def step(
        self,
        completions: torch.Tensor | None = None,
    ) -> SFTPrompts:
        """Advance the data iterator and return the next batch of tokenised data

        :param completions: The completions from the LLM model.
        :type completions: torch.Tensor | None
        :return: The next batch of tokenised data.
        :rtype: SFTPrompts
        """
        return super().step(completions)

    def create_collate_fn(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_context_length: int | None = None,
    ) -> Callable[[list[dict[str, Any]]], SFTPrompts]:
        """Build a collate function that tokenises ``(prompt, response)`` pairs.

        :param tokenizer: The tokenizer.
        :type tokenizer: PreTrainedTokenizerBase
        :param max_context_length: The maximum context length for the LLM model.
        :type max_context_length: int | None
        :return: The collate function.
        :rtype: Callable[[list[dict[str, Any]]], SFTPrompts]
        """
        response_column = self.response_column

        def collate_fn(batch: list[dict[str, Any]]) -> SFTPrompts:
            prompts = [item["prompt"] for item in batch]
            responses = [item[response_column] for item in batch]

            prompt_encodings = tokenizer(
                prompts,
                truncation=True,
                padding=False,
                add_special_tokens=True,
            )
            prompt_lengths = [len(ids) for ids in prompt_encodings["input_ids"]]

            eos = tokenizer.eos_token or ""
            pair_enc = tokenizer(
                prompts,
                [response + eos for response in responses],
                max_length=self.max_context_length,
                truncation=True,
                padding="longest",
                return_tensors="pt",
            )

            result: SFTPrompts = {
                "prompt": prompts,
                "prompt_lengths": prompt_lengths,
                "response": responses,
                "input_ids": pair_enc["input_ids"],
                "attention_mask": pair_enc["attention_mask"].long(),
            }
            return result

        return collate_fn

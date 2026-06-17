"""Reasoning LLM Gym environment."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch

from agilerl.llm_envs.base import HuggingFaceGym, apply_chat_template
from agilerl.typing import ReasoningPrompts

if TYPE_CHECKING:
    from accelerate import Accelerator
    from datasets import Dataset
    from transformers import AutoTokenizer


class ReasoningGym(HuggingFaceGym):
    """Class to convert HuggingFace datasets into Gymnasium style environment."""

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
        """Take a step in the ReasoningGym environment."""
        self.reset_called = False
        rewards = self._decode_and_evaluate(completions)
        new_tokenized_prompts = self._get_next_batch()
        self.last_tokenized_prompts = new_tokenized_prompts
        return new_tokenized_prompts, rewards

    def reset(
        self,
        reset_dataloaders: bool = False,
    ) -> tuple[list[ReasoningPrompts], dict[str, Any]]:
        """Reset the environment and get the next batch of tokenized prompts."""
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
        """Decode the completions and evaluate the rewards."""
        total_rewards = []
        for idx, (group_completion, answer, question) in enumerate(
            zip(completions, self.answers, self.questions, strict=False),
        ):
            completion_to_decode = group_completion[
                :,
                self.last_tokenized_prompts[idx]["input_ids"].shape[1] :,
            ]
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
                            skip_special_tokens=False,
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
        """Create a collate function that applies the chat template."""

        def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
            questions = [item["question"] for item in batch]
            answers = [item["answer"] for item in batch]
            tokenized_prompts = [
                apply_chat_template(self.conversation_template, q, a, tokenizer)
                for q, a in zip(questions, answers, strict=False)
            ]
            return {
                "question": questions,
                "answer": answers,
                "tokenized_prompts": tokenized_prompts,
            }

        return collate_fn


class ReasoningGymV2(HuggingFaceGym):
    """Gem-aligned reasoning gym with a split ``reset``/``step`` API.

    This is a standalone sibling of :class:`ReasoningGym` (it does **not**
    subclass it). The behavioural difference is how ``step`` works: instead of
    both scoring the current completions *and* advancing the dataloader to the
    next batch (as :class:`ReasoningGym` does), this variant follows the
    Gymnasium / multi-turn convention:

    - ``reset`` advances the dataloader and returns ``(prompts, info)``.
    - ``step`` only scores the current completions and terminates, returning
      ``(obs, rewards, terminated, truncated, info)``. It does **not** fetch the
      next batch.

    The next batch of prompts is produced by the following ``reset`` call, so a
    training loop alternates ``reset`` -> ``get_action`` -> ``step`` each
    iteration. Dataset iteration, epoch counting and GRPO grouping (via
    ``agent.get_action(repeat_prompts=True)``) match :class:`ReasoningGym`.
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

    def reset(
        self,
        seed: int | None = None,
        reset_dataloaders: bool = False,
    ) -> tuple[list[ReasoningPrompts], dict[str, Any]]:
        """Advance to the next batch and return it alongside an info dict.

        :param seed: Unused; accepted for Gymnasium-style API compatibility.
        :type seed: int | None
        :param reset_dataloaders: Whether to reset the dataloaders to the start.
        :type reset_dataloaders: bool
        :return: The next batch of tokenized prompts and an (empty) info dict.
        :rtype: tuple[list[ReasoningPrompts], dict[str, Any]]
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
        return new_tokenized_prompts, {}

    def step(
        self,
        completions: torch.Tensor,
    ) -> tuple[dict[str, Any], torch.Tensor, bool, bool, dict[str, Any]]:
        """Score the completions for the current batch and terminate.

        The next batch is *not* fetched here; call :meth:`reset` to obtain it.

        :param completions: Per-group completion id tensors for the current
            batch of prompts.
        :type completions: torch.Tensor
        :return: ``(obs, rewards, terminated, truncated, info)`` where ``obs``
            and ``info`` are empty dicts, ``terminated`` is ``True`` and
            ``truncated`` is ``False``.
        :rtype: tuple[dict[str, Any], torch.Tensor, bool, bool, dict[str, Any]]
        """
        self.reset_called = False
        rewards = self._decode_and_evaluate(completions)
        return {}, rewards, True, False, {}

    def _decode_and_evaluate(self, completions: list[torch.Tensor]) -> torch.Tensor:
        """Decode the completions and evaluate the rewards."""
        total_rewards = []
        for idx, (group_completion, answer, question) in enumerate(
            zip(completions, self.answers, self.questions, strict=False),
        ):
            completion_to_decode = group_completion[
                :,
                self.last_tokenized_prompts[idx]["input_ids"].shape[1] :,
            ]
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
                            skip_special_tokens=False,
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
        """Create a collate function that applies the chat template."""

        def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
            questions = [item["question"] for item in batch]
            answers = [item["answer"] for item in batch]
            tokenized_prompts = [
                apply_chat_template(self.conversation_template, q, a, tokenizer)
                for q, a in zip(questions, answers, strict=False)
            ]
            return {
                "question": questions,
                "answer": answers,
                "tokenized_prompts": tokenized_prompts,
            }

        return collate_fn

"""Preference LLM Gym environment."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from agilerl.llm_envs.base import HuggingFaceGym, IterablePromptBatchGym
from agilerl.typing import PreferencePrompts

if TYPE_CHECKING:
    import torch
    from accelerate import Accelerator
    from datasets import Dataset
    from transformers import AutoTokenizer


class PreferenceGym(IterablePromptBatchGym):
    """Class to convert HuggingFace preference datasets into Gymnasium style environment."""

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
        """Return the next batch (``completions`` is unused)."""
        return super().step(completions)

    def create_collate_fn(
        self,
        tokenizer: AutoTokenizer,
        max_context_length: int | None = None,
    ) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
        """Create a collate function for preference prompts."""

        def collate_fn(batch: list[dict[str, str]]) -> dict[str, str]:
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
                chosen_ids = tokenizer(prompts, chosen, truncation=True, padding=False)
                rejected_ids = tokenizer(
                    prompts, rejected, truncation=True, padding=False
                )
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


class PreferenceGymV2(HuggingFaceGym):
    """Gem-aligned preference gym with a split ``reset``/``step`` API.

    This is a standalone sibling of :class:`PreferenceGym` (it does **not**
    subclass it). Preference learning (DPO) has no generation and no environment
    reward, so this variant aligns the API for completeness rather than
    producing a meaningful reward:

    - ``reset`` advances the dataloader and returns ``(prompts, info)``.
    - ``step`` does not advance the dataloader and returns
      ``(obs, None, terminated, truncated, info)`` with a ``None`` reward.

    The next batch is produced by the following ``reset`` call.
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

    def reset(
        self,
        seed: int | None = None,
        reset_dataloaders: bool = False,
    ) -> tuple[PreferencePrompts, dict[str, Any]]:
        """Advance to the next batch and return it alongside an info dict.

        :param seed: Unused; accepted for Gymnasium-style API compatibility.
        :type seed: int | None
        :param reset_dataloaders: Whether to reset the dataloaders to the start.
        :type reset_dataloaders: bool
        :return: The next batch of preference prompts and an (empty) info dict.
        :rtype: tuple[PreferencePrompts, dict[str, Any]]
        """
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
        return self._get_next_batch(), {}

    def step(
        self,
        completions: torch.Tensor | None = None,
    ) -> tuple[dict[str, Any], None, bool, bool, dict[str, Any]]:
        """Terminate without scoring or advancing the dataloader.

        :param completions: Unused; preference learning has no env reward.
        :type completions: torch.Tensor | None
        :return: ``(obs, None, terminated, truncated, info)`` with empty dicts,
            a ``None`` reward, ``terminated=True`` and ``truncated=False``.
        :rtype: tuple[dict[str, Any], None, bool, bool, dict[str, Any]]
        """
        self.reset_called = False
        return {}, None, True, False, {}

    def _get_next_batch(self) -> PreferencePrompts:
        """Get the next batch from the dataloader, wrapping epochs."""
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

    def create_collate_fn(
        self,
        tokenizer: AutoTokenizer,
        max_context_length: int | None = None,
    ) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
        """Create a collate function for preference prompts."""

        def collate_fn(batch: list[dict[str, str]]) -> dict[str, str]:
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
                chosen_ids = tokenizer(prompts, chosen, truncation=True, padding=False)
                rejected_ids = tokenizer(
                    prompts, rejected, truncation=True, padding=False
                )
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

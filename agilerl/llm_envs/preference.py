"""Preference LLM Gym environment."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from agilerl.llm_envs.base import IterablePromptBatchGym
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

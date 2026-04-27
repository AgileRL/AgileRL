"""SFT LLM Gym environment."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agilerl.llm_envs.base import IterablePromptBatchGym
from agilerl.typing import SFTPrompts

if TYPE_CHECKING:
    import torch
    from accelerate import Accelerator
    from datasets import Dataset
    from transformers import AutoTokenizer


class SFTGym(IterablePromptBatchGym):
    """Gymnasium-style environment for supervised fine-tuning (SFT) datasets."""

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
        """Build a collate function that tokenises ``(prompt, response)`` pairs."""
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
                "input_ids": pair_enc["input_ids"],
                "attention_mask": pair_enc["attention_mask"].long(),
            }

        return collate_fn

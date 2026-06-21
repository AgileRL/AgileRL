"""Dataset-backed (teacher-forced) LLM env — one class, configured by descriptors.

A ``DatasetEnv`` is the no-generation half of the env taxonomy (see
``docs/design/llm-env-taxonomy.md`` in agilerl-integration): the completions are dataset
labels, scored in a single teacher-forced forward (SFT cross-entropy, DPO preference) with
no autoregressive rollout. The training regimes (preference / SFT) differ only by the
*required columns* and the *collate function* — descriptors, not subclasses — so they share
one class. ``PreferenceGym`` / ``SFTGym`` remain as thin back-compat subclasses.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agilerl.llm_envs.base import IterablePromptBatchGym

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch
    from accelerate import Accelerator
    from datasets import Dataset
    from transformers import AutoTokenizer

    CollateBuilder = Callable[
        ["DatasetEnv", "AutoTokenizer", "int | None"],
        Callable[[list[dict[str, Any]]], dict[str, Any]],
    ]


class DatasetEnv(IterablePromptBatchGym):
    """Teacher-forced, dataset-backed LLM env (no generation).

    Configured by ``required_columns`` + a ``collate_builder`` (and optional
    ``response_column``) instead of a subclass per training regime. ``reset``/``step`` (the
    dataloader advance, completions ignored) are inherited from
    :class:`~agilerl.llm_envs.base.IterablePromptBatchGym`.
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

    def create_collate_fn(
        self,
        tokenizer: AutoTokenizer,
        max_context_length: int | None = None,
    ) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
        """Delegate to the injected collate builder."""
        return self._collate_builder(self, tokenizer, max_context_length)


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

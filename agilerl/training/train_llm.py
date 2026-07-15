"""Compatibility exports for moved LLM finetuning helpers."""

from __future__ import annotations

import warnings

from agilerl.training.llm import (
    finetune_llm_multiturn,
    finetune_llm_preference,
    finetune_llm_reasoning,
    finetune_llm_sft,
)

warnings.warn(
    (
        "Importing from agilerl.training.train_llm is deprecated and will be removed "
        "in a future release. Import from agilerl.training.llm instead."
    ),
    FutureWarning,
    stacklevel=2,
)

__all__ = [
    "finetune_llm_multiturn",
    "finetune_llm_preference",
    "finetune_llm_reasoning",
    "finetune_llm_sft",
]

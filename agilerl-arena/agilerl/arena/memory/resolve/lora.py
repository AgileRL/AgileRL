# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Omitted vLLM ``max_lora_rank``."""

from __future__ import annotations

# Ranks vLLM will accept as ``max_lora_rank``.
VLLM_ACCEPTED_MAX_LORA_RANKS: tuple[int, ...] = (8, 16, 32, 64, 128, 256, 320, 512)
OMITTED_MAX_LORA_RANK = 16


def resolve_lora_rank_ceiling(configured: int | None, adapter_rank: int) -> int:
    """Ceiling at least as large as the adapter rank.

    :param configured: Manifest ``max_lora_rank``. ``None`` counts as 0.
    :param adapter_rank: LoRA rank the engine serves.
    :return: ``max(configured, adapter_rank)``.
    """
    return max(int(configured or 0), int(adapter_rank))


def resolve_max_lora_rank(configured: int | None, adapter_rank: int) -> int:
    """vLLM-accepted ``max_lora_rank`` for an async engine.

    :param configured: Manifest ceiling. ``None`` counts as 0.
    :param adapter_rank: LoRA rank the engine serves.
    :return: Smallest accepted rank that covers the ceiling.
    :raises ValueError: When the ceiling is above every accepted rank.
    """
    target = resolve_lora_rank_ceiling(configured, adapter_rank)
    largest = VLLM_ACCEPTED_MAX_LORA_RANKS[-1]
    if target > largest:
        driver = (
            f"configured max_lora_rank {int(configured or 0)}"
            if int(configured or 0) > int(adapter_rank)
            else f"LoRA adapter rank {int(adapter_rank)}"
        )
        msg = (
            f"{driver} needs a vLLM max_lora_rank of {target}, "
            f"above the largest value vLLM accepts ({largest})"
        )
        raise ValueError(msg)
    return next(value for value in VLLM_ACCEPTED_MAX_LORA_RANKS if value >= target)

# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""vLLM ``max_lora_rank``."""

from __future__ import annotations

# Ranks vLLM will accept as ``max_lora_rank``.
VLLM_ACCEPTED_MAX_LORA_RANKS: tuple[int, ...] = (8, 16, 32, 64, 128, 256, 320, 512)
# Rank an estimate sizes when it names no adapter.
OMITTED_MAX_LORA_RANK = 16


def resolve_max_lora_rank(adapter_rank: int) -> int:
    """vLLM ``max_lora_rank`` for an engine serving one adapter.

    :param adapter_rank: LoRA rank the engine serves.
    :return: Smallest accepted rank that covers the adapter rank.
    :raises ValueError: When the adapter rank is above every accepted rank.
    """
    largest = VLLM_ACCEPTED_MAX_LORA_RANKS[-1]
    if adapter_rank > largest:
        msg = (
            f"LoRA adapter rank {adapter_rank} needs a vLLM max_lora_rank above "
            f"the largest value vLLM accepts ({largest})"
        )
        raise ValueError(msg)
    return next(
        value for value in VLLM_ACCEPTED_MAX_LORA_RANKS if value >= adapter_rank
    )

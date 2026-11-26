# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Shapes and inputs shared by the Qwen3 and Qwen2 parallel tests."""

from __future__ import annotations

import torch

HIDDEN = 16
BATCH = 2
SEQ = 8
LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def eager_mask(length: int = SEQ) -> torch.Tensor:
    """Additive causal mask for eager attention."""
    allowed = torch.ones(length, length, dtype=torch.bool).tril()
    blocked = torch.finfo(torch.float32).min
    return torch.where(allowed, 0.0, blocked)[None, None]


def shard_data(dp_rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Input hidden states and an output probe for one data shard."""
    generator = torch.Generator().manual_seed(100 + dp_rank)
    hidden = torch.randn(BATCH, SEQ, HIDDEN, generator=generator)
    probe = torch.randn(BATCH, SEQ, HIDDEN, generator=generator)
    return hidden, probe

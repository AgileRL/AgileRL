# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""CPU contracts for Ulysses attention helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from agilerl.distributed.context_parallel import (
    ULYSSES_PARAMS,
    PublishedUlyssesParams,
    attention_head_counts,
    clear_ulysses_params,
    model_attention_backend,
    replicate_kv_heads,
    update_ulysses_params,
)


class TestReplicateKvHeads:
    def test_repeats_heads_to_cp_size(self) -> None:
        torch.manual_seed(0)
        data = torch.arange(2 * 3 * 2 * 4).reshape(2, 3, 2, 4).float()

        out = replicate_kv_heads(data, 4)

        assert out.shape == (2, 3, 4, 4)
        assert torch.equal(out[..., 0, :], data[..., 0, :])
        assert torch.equal(out[..., 1, :], data[..., 0, :])
        assert torch.equal(out[..., 2, :], data[..., 1, :])
        assert torch.equal(out[..., 3, :], data[..., 1, :])

    def test_rejects_when_heads_do_not_divide_cp(self) -> None:
        with pytest.raises(ValueError, match="must divide cp_size"):
            replicate_kv_heads(torch.zeros(1, 2, 3, 4), 4)


class TestPublishedUlyssesParams:
    def test_restores_previous_params(self) -> None:
        update_ulysses_params(torch.tensor([0, 4]), 4)

        with PublishedUlyssesParams(
            {"cu_seqlens": torch.tensor([0, 8]), "max_seqlen": 8}
        ):
            assert ULYSSES_PARAMS["max_seqlen"] == 8

        assert ULYSSES_PARAMS["max_seqlen"] == 4
        clear_ulysses_params()

    def test_clear_drops_published_params(self) -> None:
        update_ulysses_params(torch.tensor([0, 2]), 2)

        clear_ulysses_params()

        assert ULYSSES_PARAMS == {}


class TestAttentionHeadCounts:
    def test_reads_config_heads(self) -> None:
        model = nn.Linear(2, 2)
        model.config = SimpleNamespace(num_attention_heads=8, num_key_value_heads=2)

        assert attention_head_counts(model) == (8, 2)

    def test_missing_heads_raises(self) -> None:
        model = nn.Linear(2, 2)
        model.config = SimpleNamespace()

        with pytest.raises(ValueError, match="num_attention_heads"):
            attention_head_counts(model)


class TestModelAttentionBackend:
    def test_reads_attn_implementation(self) -> None:
        model = nn.Linear(2, 2)
        model.config = SimpleNamespace(_attn_implementation="flash_attention_2")

        assert model_attention_backend(model) == "flash_attention_2"

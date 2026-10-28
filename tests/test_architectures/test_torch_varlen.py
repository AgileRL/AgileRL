# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``agilerl.architectures.torch_varlen``."""

from __future__ import annotations

import itertools

import pytest
import torch
from transformers import LlamaConfig
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.modeling_llama import LlamaAttention

from agilerl.architectures.torch_varlen import (
    TORCH_VARLEN_IMPLEMENTATION,
    register_torch_varlen_attention,
    torch_varlen_attention_forward,
)

cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="torch varlen needs CUDA"
)


def _attention(config_kwargs=None):
    config = LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        **(config_kwargs or {}),
    )
    return LlamaAttention(config, layer_idx=0)


def _packed_inputs(segments, heads=4, kv_heads=2, dim=16, dtype=None, device="cpu"):
    torch.manual_seed(0)
    total = sum(segments)
    query = torch.randn(1, heads, total, dim, dtype=dtype, device=device)
    key = torch.randn(1, kv_heads, total, dim, dtype=dtype, device=device)
    value = torch.randn(1, kv_heads, total, dim, dtype=dtype, device=device)
    cu = torch.tensor(
        [0, *itertools.accumulate(segments)], dtype=torch.int32, device=device
    )
    return query, key, value, cu, max(segments)


def _repeat_kv_heads(key, value, heads):
    batches, _, rows, dim = key.shape
    repeats = heads // key.shape[1]
    key = (
        key[:, :, None, :, :]
        .expand(batches, -1, repeats, rows, dim)
        .reshape(batches, heads, rows, dim)
    )
    value = (
        value[:, :, None, :, :]
        .expand(batches, -1, repeats, rows, dim)
        .reshape(batches, heads, rows, dim)
    )
    return key, value


def _block_causal_mask(segments, window=None):
    total = sum(segments)
    mask = torch.full((total, total), float("-inf"))
    start = 0
    for length in segments:
        causal = torch.tril(torch.ones(length, length, dtype=torch.bool))
        if window is not None:
            rows = torch.arange(length).unsqueeze(1)
            cols = torch.arange(length).unsqueeze(0)
            causal = causal & (rows - cols < window)
        mask[start : start + length, start : start + length][causal] = 0.0
        start += length
    return mask


class TestRegisterTorchVarlenAttention:
    def test_registration_exposes_the_interface(self) -> None:
        register_torch_varlen_attention()
        register_torch_varlen_attention()

        assert TORCH_VARLEN_IMPLEMENTATION in ALL_ATTENTION_FUNCTIONS.valid_keys()
        assert (
            ALL_ATTENTION_FUNCTIONS.get_interface(
                TORCH_VARLEN_IMPLEMENTATION, sdpa_attention_forward
            )
            is torch_varlen_attention_forward
        )

    def test_from_config_accepts_the_implementation(self) -> None:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_config(
            LlamaConfig(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                head_dim=16,
            ),
            attn_implementation=TORCH_VARLEN_IMPLEMENTATION,
        )

        assert model.config._attn_implementation == TORCH_VARLEN_IMPLEMENTATION


class TestTorchVarlenUnpackedFallback:
    def test_unpacked_rows_match_sdpa(self) -> None:
        module = _attention()
        query = torch.randn(2, 4, 6, 16)
        key = torch.randn(2, 2, 6, 16)
        value = torch.randn(2, 2, 6, 16)
        mask = torch.tril(torch.ones(6, 6)).unsqueeze(0).unsqueeze(0)
        mask = torch.where(mask.bool(), 0.0, float("-inf"))

        out, weights = torch_varlen_attention_forward(
            module, query, key, value, mask, is_causal=True
        )
        expected, _ = sdpa_attention_forward(
            module, query, key, value, mask, is_causal=True
        )

        assert weights is None
        assert torch.equal(out, expected)

    def test_non_causal_rows_match_sdpa(self) -> None:
        module = _attention()
        query = torch.randn(1, 4, 5, 16)
        key = torch.randn(1, 4, 5, 16)
        value = torch.randn(1, 4, 5, 16)

        out, _ = torch_varlen_attention_forward(
            module, query, key, value, None, is_causal=False
        )
        expected, _ = sdpa_attention_forward(
            module, query, key, value, None, is_causal=False
        )

        assert torch.equal(out, expected)


class TestTorchVarlenRejects:
    def test_softcap_raises(self) -> None:
        module = _attention()
        query = torch.randn(1, 4, 5, 16)
        key = torch.randn(1, 2, 5, 16)
        value = torch.randn(1, 2, 5, 16)

        with pytest.raises(ValueError, match="softcap"):
            torch_varlen_attention_forward(
                module, query, key, value, None, softcap=50.0
            )

    def test_dropout_on_packed_rows_raises(self) -> None:
        module = _attention()
        query, key, value, cu, longest = _packed_inputs([3, 2])

        with pytest.raises(ValueError, match="dropout"):
            torch_varlen_attention_forward(
                module,
                query,
                key,
                value,
                None,
                dropout=0.1,
                cu_seq_lens_q=cu,
                cu_seq_lens_k=cu,
                max_length_q=longest,
                max_length_k=longest,
            )

    def test_attention_sink_on_packed_rows_raises(self) -> None:
        module = _attention()
        query, key, value, cu, longest = _packed_inputs([3, 2])

        with pytest.raises(ValueError, match="attention-sink"):
            torch_varlen_attention_forward(
                module,
                query,
                key,
                value,
                None,
                cu_seq_lens_q=cu,
                cu_seq_lens_k=cu,
                max_length_q=longest,
                max_length_k=longest,
                s_aux=torch.zeros(1),
            )


@pytest.mark.gpu
@cuda_required
class TestTorchVarlenPackedParity:
    @pytest.mark.parametrize("kv_heads", [4, 2])
    def test_packed_rows_match_masked_reference(self, kv_heads) -> None:
        device = torch.device("cuda")
        module = _attention().to(device)
        segments = [7, 5, 3]
        query, key, value, cu, longest = _packed_inputs(
            segments, kv_heads=kv_heads, dtype=torch.bfloat16, device=device
        )
        mask = _block_causal_mask(segments).to(device)

        out, weights = torch_varlen_attention_forward(
            module,
            query,
            key,
            value,
            None,
            cu_seq_lens_q=cu,
            cu_seq_lens_k=cu,
            max_length_q=longest,
            max_length_k=longest,
        )
        ref_key, ref_value = _repeat_kv_heads(key.float(), value.float(), 4)
        expected = torch.nn.functional.scaled_dot_product_attention(
            query.float(),
            ref_key,
            ref_value,
            attn_mask=mask,
            is_causal=False,
        ).transpose(1, 2)

        assert weights is None
        # bf16 flash kernel against an fp32 reference.
        assert torch.allclose(out.float(), expected, rtol=1e-2, atol=1e-2)

    def test_packed_rows_match_with_sliding_window(self) -> None:
        device = torch.device("cuda")
        module = _attention().to(device)
        segments = [7, 5]
        query, key, value, cu, longest = _packed_inputs(
            segments, dtype=torch.bfloat16, device=device
        )
        mask = _block_causal_mask(segments, window=4).to(device)

        out, _ = torch_varlen_attention_forward(
            module,
            query,
            key,
            value,
            None,
            sliding_window=4,
            cu_seq_lens_q=cu,
            cu_seq_lens_k=cu,
            max_length_q=longest,
            max_length_k=longest,
        )
        ref_key, ref_value = _repeat_kv_heads(key.float(), value.float(), 4)
        expected = torch.nn.functional.scaled_dot_product_attention(
            query.float(),
            ref_key,
            ref_value,
            attn_mask=mask,
            is_causal=False,
        ).transpose(1, 2)

        assert torch.allclose(out.float(), expected, rtol=1e-2, atol=1e-2)

    def test_position_ids_derive_the_same_segments(self) -> None:
        device = torch.device("cuda")
        module = _attention().to(device)
        query, key, value, cu, longest = _packed_inputs(
            [4, 6], dtype=torch.bfloat16, device=device
        )
        position_ids = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3, 4, 5]], device=device)

        explicit, _ = torch_varlen_attention_forward(
            module,
            query,
            key,
            value,
            None,
            cu_seq_lens_q=cu,
            cu_seq_lens_k=cu,
            max_length_q=longest,
            max_length_k=longest,
        )
        derived, _ = torch_varlen_attention_forward(
            module, query, key, value, None, position_ids=position_ids
        )

        assert torch.equal(explicit, derived)

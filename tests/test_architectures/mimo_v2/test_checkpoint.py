# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for the FP8 / MXFP4 dequantizers and the pre-sharded ``qkv_proj`` split."""

from __future__ import annotations

import pytest
import torch

from agilerl.architectures.mimo_v2.checkpoint import (
    Fp8BlockDequantize,
    Mxfp4Dequantize,
    ParameterDtypeCast,
    PreshardedQkvDequantize,
    dequant_fp8_block,
    dequant_fp8_presharded,
    dequant_mxfp4,
    split_presharded_qkv,
)
from tests.test_architectures.mimo_v2.checkpoint_helpers import (
    fp8_quantize,
    fp8_quantize_presharded,
    interleave_qkv,
    mxfp4_quantize,
)


def random_weight(rows: int, cols: int, seed: int) -> torch.Tensor:
    return torch.randn(rows, cols, generator=torch.Generator().manual_seed(seed))


class TestDequantFp8Block:
    def test_matches_reference_on_a_padded_grid(self):
        # Arrange
        quantized, scale, expected = fp8_quantize(random_weight(200, 150, seed=0))

        # Act
        dequantized = dequant_fp8_block(quantized, scale)

        # Assert
        assert scale.shape == (2, 2)
        assert dequantized.dtype == torch.float32
        assert torch.equal(dequantized, expected)

    def test_rejects_a_grid_that_does_not_cover_the_weight(self):
        quantized, scale, _ = fp8_quantize(random_weight(200, 150, seed=0))

        with pytest.raises(ValueError, match=r"needs a \(2, 2\) scale grid"):
            dequant_fp8_block(quantized, scale[:1])


class TestDequantFp8Presharded:
    def test_matches_reference_with_a_padded_grid_per_chunk(self):
        # Arrange
        quantized, scale, expected = fp8_quantize_presharded(
            random_weight(352, 64, seed=1), chunks=2
        )

        # Act
        dequantized = dequant_fp8_presharded(quantized, scale, chunks=2)

        # Assert
        assert scale.shape == (4, 1)
        assert torch.equal(dequantized, expected)

    def test_rejects_scales_for_a_different_chunk_count(self):
        quantized, scale, _ = fp8_quantize_presharded(
            random_weight(352, 64, seed=1), chunks=2
        )

        with pytest.raises(ValueError, match="zip"):
            dequant_fp8_presharded(quantized, scale, chunks=1)


class TestSplitPreshardedQkv:
    @pytest.mark.parametrize(("kv_heads", "chunks"), [(2, 2), (4, 2), (8, 4)])
    def test_recovers_contiguous_projections(self, kv_heads, chunks):
        # Arrange
        query = random_weight(8 * 48, 64, seed=2)
        key = random_weight(kv_heads * 48, 64, seed=3)
        value = random_weight(kv_heads * 32, 64, seed=4)
        fused = interleave_qkv(query, key, value, chunks)

        # Act
        projections = split_presharded_qkv(
            fused,
            num_heads=8,
            num_kv_heads=kv_heads,
            head_dim=48,
            v_head_dim=32,
            chunks=chunks,
        )

        # Assert
        for projection, expected in zip(projections, (query, key, value), strict=True):
            assert torch.equal(projection, expected)


class TestDequantMxfp4:
    def test_matches_transformers_mxfp4_decoder(self):
        # Arrange
        packed, scale, expected = mxfp4_quantize(random_weight(32, 128, seed=5))

        # Act
        dequantized = dequant_mxfp4(packed, scale)

        # Assert
        assert packed.dtype == torch.uint8
        assert packed.shape == (32, 64)
        assert scale.shape == (32, 4)
        assert dequantized.dtype == torch.float32
        assert torch.equal(dequantized, expected)

    def test_decodes_known_nibbles_and_exponents(self):
        # Arrange: low nibble 0b0011 = 1.5, high nibble 0b1101 = -3.0; exponent 128 - 127 = 1.
        packed = torch.full((1, 16), 0xD3, dtype=torch.uint8)
        scale = torch.tensor([[128]], dtype=torch.uint8)

        # Act
        dequantized = dequant_mxfp4(packed, scale)

        # Assert
        assert torch.equal(dequantized, torch.tensor([[3.0, -6.0] * 16]))


@pytest.mark.parametrize(
    "operation",
    [
        ParameterDtypeCast,
        Fp8BlockDequantize,
        PreshardedQkvDequantize,
        Mxfp4Dequantize,
    ],
)
class TestLoadOnlyReverse:
    def test_saving_back_to_the_hub_format_raises(self, operation):
        reverse = operation().reverse_op

        with pytest.raises(NotImplementedError, match="convert on load only"):
            reverse.convert({}, [], [])

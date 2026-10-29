# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from agilerl.lora.moe import grouped_gemm as moe_gemm


class TestExpertRowCounts:
    @pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
    def test_counts_rows_per_expert(self, dtype):
        expert_ids = torch.tensor([2, 0, 2, 3, 2], dtype=dtype)

        counts = moe_gemm.expert_row_counts(expert_ids, num_experts=5)

        assert torch.equal(counts, torch.tensor([1, 0, 3, 1, 0]))

    def test_empty_ids_give_zero_counts(self):
        counts = moe_gemm.expert_row_counts(torch.zeros(0, dtype=torch.long), 3)

        assert torch.equal(counts, torch.zeros(3, dtype=torch.long))


class TestRowChunkOffsets:
    def test_chunks_cut_through_experts(self):
        counts = torch.tensor([3, 0, 4, 2])

        sizes, offsets = moe_gemm.row_chunk_offsets(counts, 9, max_rows=4)

        assert sizes == [4, 4, 1]
        assert offsets.dtype == torch.int32
        assert torch.equal(
            offsets,
            torch.tensor([[3, 3, 4, 4], [0, 0, 3, 4], [0, 0, 0, 1]], dtype=torch.int32),
        )

    def test_one_chunk_holds_every_row(self):
        counts = torch.tensor([3, 0, 4, 2])

        sizes, offsets = moe_gemm.row_chunk_offsets(counts, 9, max_rows=100)

        assert sizes == [9]
        assert torch.equal(offsets, torch.tensor([[3, 3, 7, 9]], dtype=torch.int32))

    def test_no_rows_give_no_chunks(self):
        sizes, offsets = moe_gemm.row_chunk_offsets(torch.zeros(4), 0, max_rows=4)

        assert sizes == []
        assert offsets.shape == (0, 4)

    def test_last_offset_of_each_chunk_is_its_row_count(self):
        torch.manual_seed(0)
        counts = torch.randint(0, 7, (16,))
        total = int(counts.sum())

        sizes, offsets = moe_gemm.row_chunk_offsets(counts, total, max_rows=5)

        assert offsets[:, -1].tolist() == sizes
        assert sum(sizes) == total


class TestOffsetCounts:
    def test_returns_rows_per_group(self):
        offs = torch.tensor([2, 2, 5], dtype=torch.int32)

        assert moe_gemm.offset_counts(offs) == [2, 0, 3]


class TestGroupedOperand:
    def test_returns_the_in_out_layout(self):
        weight = torch.randn(3, 4, 8)

        operand = moe_gemm.grouped_operand(weight)

        assert operand.shape == (3, 8, 4)
        assert torch.equal(operand, weight.transpose(-2, -1))

    def test_dtensor_weight_returns_its_local_shard_layout(self, monkeypatch):
        # Arrange
        class LocalShardTensor:
            def __init__(self, local: torch.Tensor) -> None:
                self.local = local

            def to_local(self) -> torch.Tensor:
                return self.local

        monkeypatch.setattr(moe_gemm, "DTensor", LocalShardTensor)
        local = torch.randn(2, 4, 8)

        # Act
        operand = moe_gemm.grouped_operand(LocalShardTensor(local))

        # Assert
        assert torch.equal(operand, local.transpose(-2, -1))


class TestGroupedMatmul:
    @pytest.mark.parametrize("use_grouped_mm", [False, True], ids=["loop", "gmm"])
    def test_matches_per_expert_reference_with_empty_groups(
        self, monkeypatch, use_grouped_mm
    ):
        # Arrange
        torch.manual_seed(0)
        counts = [2, 0, 3, 0]
        x = torch.randn(sum(counts), 8, requires_grad=True)
        weight = torch.randn(4, 8, 4, requires_grad=True)
        offs = torch.tensor([2, 2, 5, 5], dtype=torch.int32)
        if use_grouped_mm:
            monkeypatch.setattr(moe_gemm, "_use_grouped_mm", lambda _x: True)
        expected = torch.cat(
            [rows @ weight[e] for e, rows in enumerate(x.split(counts))]
        )
        expected_grads = torch.autograd.grad(expected.square().sum(), (x, weight))

        # Act
        out = moe_gemm.grouped_matmul(x, weight, offs)
        grads = torch.autograd.grad(out.square().sum(), (x, weight))

        # Assert
        # fp32 sums of 8 products; only the accumulation order differs.
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-6)
        for grad, expected_grad in zip(grads, expected_grads, strict=True):
            torch.testing.assert_close(grad, expected_grad, rtol=1e-5, atol=1e-6)
        assert torch.equal(grads[1][1], torch.zeros(8, 4))


class TestPadLoraRank:
    def test_zero_pads_an_unaligned_bf16_rank_on_the_grouped_path(self, monkeypatch):
        # Arrange
        monkeypatch.setattr(moe_gemm, "_use_grouped_mm", lambda _x: True)
        torch.manual_seed(0)
        lora_a = torch.randn(3, 2, 16, dtype=torch.bfloat16)
        lora_b = torch.randn(3, 8, 2, dtype=torch.bfloat16)

        # Act
        padded_a, padded_b = moe_gemm.pad_lora_rank(lora_a, lora_b)

        # Assert
        assert padded_a.shape == (3, 8, 16)
        assert padded_b.shape == (3, 8, 8)
        assert torch.equal(padded_a[:, :2], lora_a)
        assert torch.equal(padded_b[..., :2], lora_b)
        assert not padded_a[:, 2:].any()
        assert not padded_b[..., 2:].any()

    def test_gradients_reach_the_unpadded_factors(self, monkeypatch):
        # Arrange
        monkeypatch.setattr(moe_gemm, "_use_grouped_mm", lambda _x: True)
        lora_a = torch.ones(2, 3, 16, dtype=torch.bfloat16, requires_grad=True)
        lora_b = torch.ones(2, 8, 3, dtype=torch.bfloat16, requires_grad=True)

        # Act
        padded_a, padded_b = moe_gemm.pad_lora_rank(lora_a, lora_b)
        (padded_a.sum() + padded_b.sum()).backward()

        # Assert
        assert torch.equal(lora_a.grad, torch.ones_like(lora_a))
        assert torch.equal(lora_b.grad, torch.ones_like(lora_b))

    def test_keeps_an_aligned_rank(self, monkeypatch):
        monkeypatch.setattr(moe_gemm, "_use_grouped_mm", lambda _x: True)
        lora_a = torch.randn(3, 8, 16, dtype=torch.bfloat16)
        lora_b = torch.randn(3, 8, 8, dtype=torch.bfloat16)

        padded_a, padded_b = moe_gemm.pad_lora_rank(lora_a, lora_b)

        assert padded_a is lora_a
        assert padded_b is lora_b

    def test_keeps_factors_off_the_grouped_path(self):
        lora_a = torch.randn(3, 2, 16, dtype=torch.bfloat16)
        lora_b = torch.randn(3, 8, 2, dtype=torch.bfloat16)

        padded_a, padded_b = moe_gemm.pad_lora_rank(lora_a, lora_b)

        assert padded_a is lora_a
        assert padded_b is lora_b

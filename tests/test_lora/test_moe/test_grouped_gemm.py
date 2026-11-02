# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

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


class TestCountsList:
    def test_device_counts_become_python_ints(self):
        counts = moe_gemm.counts_list(torch.tensor([3, 0, 2]))

        assert counts == [3, 0, 2]
        assert all(type(count) is int for count in counts)


@pytest.fixture
def fresh_device_backends():
    """Forget cached per-device backends before and after the test."""
    moe_gemm.device_expert_backend.cache_clear()
    yield
    moe_gemm.device_expert_backend.cache_clear()


class TestExpertKernels:
    def test_cpu_rows_stay_on_torch_ops(self):
        x = torch.randn(4, 8, dtype=torch.bfloat16)

        assert moe_gemm.expert_kernels(x) is None

    def test_triton_backend_takes_the_triton_kernels(
        self, monkeypatch, fresh_device_backends
    ):
        # Arrange: bf16 on an sm80 GPU runs the Triton kernels.
        kernels = moe_gemm.ExpertKernels(None, None, None)
        monkeypatch.setattr(moe_gemm, "HAS_TRITON", True)
        monkeypatch.setattr(moe_gemm, "TRITON_EXPERT_KERNELS", kernels)
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _index: (8, 0))
        x = SimpleNamespace(
            is_cuda=True, device=torch.device("cuda", 0), dtype=torch.bfloat16
        )

        # Act
        chosen = moe_gemm.expert_kernels(x)

        # Assert
        assert chosen is kernels


class TestGroupedMatmul:
    @pytest.mark.parametrize("backend", ["gmm", "kernels"])
    def test_matches_per_expert_reference_with_empty_groups(self, request, backend):
        # Arrange
        torch.manual_seed(0)
        counts = [2, 0, 3, 0]
        x = torch.randn(sum(counts), 8, requires_grad=True)
        weight = torch.randn(4, 8, 4, requires_grad=True)
        offs = torch.tensor([2, 2, 5, 5], dtype=torch.int32)
        if backend == "kernels":
            request.getfixturevalue("reference_expert_kernels")
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

    def test_weight_follows_the_activation_dtype(self):
        # Arrange
        torch.manual_seed(0)
        x = torch.randn(4, 8, dtype=torch.bfloat16)
        weight = torch.randn(2, 8, 8)
        offs = torch.tensor([1, 4], dtype=torch.int32)
        expected = torch.cat(
            [x[:1] @ weight[0].bfloat16(), x[1:] @ weight[1].bfloat16()]
        )

        # Act
        out = moe_gemm.grouped_matmul(x, weight, offs)

        # Assert
        assert out.dtype == torch.bfloat16
        torch.testing.assert_close(out, expected)

    def test_unaligned_dims_off_cuda_raise(self):
        # Arrange: 6 fp32 columns are 24 bytes, off the 16-byte boundary.
        x = torch.randn(3, 6)
        weight = torch.randn(2, 6, 4)
        offs = torch.tensor([1, 3], dtype=torch.int32)

        # Act / Assert
        with pytest.raises(RuntimeError, match="MoE expert GEMMs in float32 on cpu"):
            moe_gemm.grouped_matmul(x, weight, offs)

    def test_frozen_weight_skips_the_weight_grad(self, reference_expert_kernels):
        # Arrange
        torch.manual_seed(0)
        counts = [3, 0, 2]
        x = torch.randn(sum(counts), 8, requires_grad=True)
        weight = torch.randn(3, 8, 4)
        offs = torch.tensor([3, 3, 5], dtype=torch.int32)
        expected = torch.cat(
            [rows @ weight[e] for e, rows in enumerate(x.split(counts))]
        )
        (expected_grad,) = torch.autograd.grad(expected.square().sum(), x)

        # Act
        out = moe_gemm.grouped_matmul(x, weight, offs)
        (grad,) = torch.autograd.grad(out.square().sum(), x)

        # Assert
        # fp32 sums of 8 products; only the accumulation order differs.
        torch.testing.assert_close(grad, expected_grad, rtol=1e-5, atol=1e-6)
        assert reference_expert_kernels.used == {"matmul"}


class TestUnalignedGemmKernels:
    def test_cuda_with_triton_takes_the_triton_kernels(self, monkeypatch):
        kernels = moe_gemm.ExpertKernels(None, None, None)
        monkeypatch.setattr(moe_gemm, "TRITON_EXPERT_KERNELS", kernels)
        x = SimpleNamespace(is_cuda=True)

        assert moe_gemm.unaligned_gemm_kernels(x) is kernels

    def test_cuda_without_triton_raises(self, monkeypatch):
        monkeypatch.setattr(moe_gemm, "TRITON_EXPERT_KERNELS", None)
        x = SimpleNamespace(
            is_cuda=True, dtype=torch.bfloat16, device=torch.device("cuda", 0)
        )

        with pytest.raises(RuntimeError, match="in bfloat16 on cuda have no kernel"):
            moe_gemm.unaligned_gemm_kernels(x)


class TestGroupedLinear:
    def test_kernels_match_per_expert_linear(self, reference_expert_kernels):
        # Arrange
        torch.manual_seed(0)
        counts = [0, 4, 1]
        x = torch.randn(sum(counts), 8, requires_grad=True)
        weight = torch.randn(3, 4, 8, requires_grad=True)
        expected = torch.cat(
            [
                rows @ weight[e].T
                for e, rows in enumerate(x.split(counts))
                if rows.shape[0]
            ]
        )
        expected_grads = torch.autograd.grad(expected.square().sum(), (x, weight))

        # Act
        out = moe_gemm.grouped_linear(x, weight, counts)
        grads = torch.autograd.grad(out.square().sum(), (x, weight))

        # Assert
        # fp32 sums of 8 products; only the accumulation order differs.
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-6)
        for grad, expected_grad in zip(grads, expected_grads, strict=True):
            torch.testing.assert_close(grad, expected_grad, rtol=1e-5, atol=1e-6)
        assert reference_expert_kernels.used == {"matmul", "weight_grad"}


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

    def test_keeps_factors_off_the_grouped_path(self, monkeypatch):
        monkeypatch.setattr(moe_gemm, "_use_grouped_mm", lambda _x: False)
        lora_a = torch.randn(3, 2, 16, dtype=torch.bfloat16)
        lora_b = torch.randn(3, 8, 2, dtype=torch.bfloat16)

        padded_a, padded_b = moe_gemm.pad_lora_rank(lora_a, lora_b)

        assert padded_a is lora_a
        assert padded_b is lora_b


@pytest.mark.skipif(
    not (torch.cuda.is_available() and moe_gemm.HAS_TRITON),
    reason="Triton expert kernels need a CUDA device",
)
class TestGroupedMatmulTriton:
    @pytest.mark.parametrize("rank", [1, 2, 3, 8, 16])
    def test_lora_factors_match_per_expert_reference(self, monkeypatch, rank):
        # Arrange
        monkeypatch.setattr(moe_gemm, "expert_backend", lambda _x: "triton")
        torch.manual_seed(0)
        counts = [5, 0, 3, 9]
        offs = torch.tensor([5, 5, 8, 17], dtype=torch.int32, device="cuda")
        options = {"device": "cuda", "dtype": torch.bfloat16}
        x = torch.randn(sum(counts), 64, **options, requires_grad=True)
        lora_a = torch.randn(4, 64, rank, **options, requires_grad=True)
        lora_b = torch.randn(4, rank, 32, **options, requires_grad=True)
        expected = torch.cat(
            [
                (rows @ lora_a[e]) @ lora_b[e]
                for e, rows in enumerate(x.split(counts))
                if rows.shape[0]
            ]
        )
        expected_grads = torch.autograd.grad(
            expected.float().square().sum(), (x, lora_a, lora_b)
        )

        # Act
        out = moe_gemm.grouped_matmul(
            moe_gemm.grouped_matmul(x, lora_a, offs), lora_b, offs
        )
        grads = torch.autograd.grad(out.float().square().sum(), (x, lora_a, lora_b))

        # Assert
        # bf16 keeps 8 mantissa bits; both sides accumulate in fp32 but round
        # sums of up to 64 products in a different order.
        tolerance = {"rtol": 2e-2, "atol": 2e-2}
        torch.testing.assert_close(out, expected, **tolerance)
        for grad, expected_grad in zip(grads, expected_grads, strict=True):
            torch.testing.assert_close(grad, expected_grad, **tolerance)

    def test_unaligned_dims_run_on_triton(self):
        # Arrange: 12 bf16 output columns are 24 bytes, off grouped_mm's
        # 16-byte boundary on every backend.
        torch.manual_seed(0)
        counts = [3, 0, 5]
        offs = torch.tensor([3, 3, 8], dtype=torch.int32, device="cuda")
        options = {"device": "cuda", "dtype": torch.bfloat16}
        x = torch.randn(sum(counts), 64, **options)
        weight = torch.randn(3, 64, 12, **options)
        expected = torch.cat(
            [
                rows @ weight[e]
                for e, rows in enumerate(x.split(counts))
                if rows.shape[0]
            ]
        )

        # Act
        out = moe_gemm.grouped_matmul(x, weight, offs)

        # Assert
        # bf16 sums of 64 products, rounded in a different order.
        torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-2)

    def test_zero_padded_rank_matches_the_unpadded_rank(self, monkeypatch):
        # Arrange
        monkeypatch.setattr(moe_gemm, "expert_backend", lambda _x: "triton")
        torch.manual_seed(0)
        offs = torch.tensor([4, 4, 11], dtype=torch.int32, device="cuda")
        options = {"device": "cuda", "dtype": torch.bfloat16}
        x = torch.randn(11, 64, **options)
        lora_a = torch.randn(3, 64, 2, **options)
        lora_b = torch.randn(3, 2, 32, **options)
        padded_a = torch.nn.functional.pad(lora_a, (0, 6))
        padded_b = torch.nn.functional.pad(lora_b, (0, 0, 0, 6))

        # Act
        unpadded = moe_gemm.grouped_matmul(
            moe_gemm.grouped_matmul(x, lora_a, offs), lora_b, offs
        )
        padded = moe_gemm.grouped_matmul(
            moe_gemm.grouped_matmul(x, padded_a, offs), padded_b, offs
        )

        # Assert: the zero rank columns add exact zeros.
        assert torch.equal(padded, unpadded)

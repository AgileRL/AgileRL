# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest
import torch

from tests.utils import spawn_new_process_for_each_test

triton_kernels = pytest.importorskip("agilerl.lora.moe.triton_kernels")

# bf16 outputs of fp32 accumulators: one rounding, at most half a bf16 ulp
# (2**-8 relative); atol covers sums that cancel to near zero.
BF16_TOLERANCE = {"rtol": 4e-3, "atol": 1e-3}
# The interpreter runs the kernels in numpy fp32; only the summation order
# differs from torch's.
FP32_TOLERANCE = {"rtol": 1e-5, "atol": 1e-5}


def per_expert_matmul(x, weight, counts):
    pieces = [
        rows.float() @ weight[e].float() for e, rows in enumerate(x.split(counts))
    ]
    return torch.cat(pieces)


def offsets(counts, device="cuda"):
    return torch.tensor(counts, device=device).cumsum(0).to(torch.int32)


def run_once(kernel_call, warmup, rep, quantiles):
    """Autotune timing stub: autotune timing needs a GPU driver."""
    kernel_call()
    return [1.0] * len(quantiles)


@pytest.fixture
def triton_interpreter(monkeypatch):
    """Run the kernels on CPU tensors through the Triton interpreter.

    Triton reads ``TRITON_INTERPRET`` when it first decorates its own library
    kernels, so the test must run in a fresh process that starts with it set.
    """
    monkeypatch.setenv("TRITON_INTERPRET", "1")


class TestGroupedMatmul:
    @pytest.mark.gpu
    @pytest.mark.parametrize(
        ("counts", "inner", "cols"),
        [
            pytest.param([70, 0, 129, 1, 0], 96, 160, id="empty_and_partial_tiles"),
            pytest.param([0, 0, 300, 0], 64, 16, id="all_to_one_rank_wide"),
            pytest.param([33, 17, 5], 16, 72, id="rank_deep"),
        ],
    )
    def test_matches_per_expert_matmul(self, counts, inner, cols):
        # Arrange
        torch.manual_seed(0)
        x = torch.randn(sum(counts), inner, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(
            len(counts), inner, cols, device="cuda", dtype=torch.bfloat16
        )

        # Act
        out = triton_kernels.grouped_matmul(x, weight, offsets(counts))

        # Assert
        torch.testing.assert_close(
            out.float(), per_expert_matmul(x, weight, counts), **BF16_TOLERANCE
        )

    @pytest.mark.gpu
    def test_transposed_weight_view_matches_contiguous(self):
        # Arrange
        torch.manual_seed(0)
        counts = [40, 0, 90]
        x = torch.randn(sum(counts), 64, device="cuda", dtype=torch.bfloat16)
        stacked = torch.randn(3, 48, 64, device="cuda", dtype=torch.bfloat16)
        weight = stacked.transpose(-2, -1)

        # Act
        out = triton_kernels.grouped_matmul(x, weight, offsets(counts))

        # Assert
        torch.testing.assert_close(
            out.float(), per_expert_matmul(x, weight, counts), **BF16_TOLERANCE
        )

    @spawn_new_process_for_each_test
    @pytest.mark.usefixtures("triton_interpreter")
    @patch("triton.testing.do_bench", run_once)
    def test_interpreted_kernel_matches_per_expert_matmul(self):
        # Arrange
        torch.manual_seed(0)
        counts = [5, 0, 9, 1]
        x = torch.randn(sum(counts), 16)
        weight = torch.randn(len(counts), 16, 24)

        # Act
        out = triton_kernels.grouped_matmul(x, weight, offsets(counts, "cpu"))

        # Assert
        torch.testing.assert_close(
            out, per_expert_matmul(x, weight, counts), **FP32_TOLERANCE
        )


class TestGroupedWeightGrad:
    @pytest.mark.gpu
    def test_matches_per_expert_outer_products(self):
        # Arrange
        torch.manual_seed(0)
        counts = [70, 0, 129, 1]
        x = torch.randn(sum(counts), 96, device="cuda", dtype=torch.bfloat16)
        grad = torch.randn(sum(counts), 40, device="cuda", dtype=torch.bfloat16)
        expected = torch.stack(
            [
                rows.float().T @ grads.float()
                for rows, grads in zip(x.split(counts), grad.split(counts), strict=True)
            ]
        )

        # Act
        out = triton_kernels.grouped_weight_grad(x, grad, offsets(counts))

        # Assert
        torch.testing.assert_close(out.float(), expected, **BF16_TOLERANCE)
        assert torch.equal(out[1], torch.zeros_like(out[1]))

    @spawn_new_process_for_each_test
    @pytest.mark.usefixtures("triton_interpreter")
    @patch("triton.testing.do_bench", run_once)
    def test_interpreted_kernel_matches_per_expert_outer_products(self):
        # Arrange
        torch.manual_seed(0)
        counts = [5, 0, 9, 1]
        x = torch.randn(sum(counts), 16)
        grad = torch.randn(sum(counts), 24)
        expected = torch.stack(
            [
                rows.T @ grads
                for rows, grads in zip(x.split(counts), grad.split(counts), strict=True)
            ]
        )

        # Act
        out = triton_kernels.grouped_weight_grad(x, grad, offsets(counts, "cpu"))

        # Assert
        torch.testing.assert_close(out, expected, **FP32_TOLERANCE)
        assert torch.equal(out[1], torch.zeros_like(out[1]))


class TestScatterAddRows:
    @pytest.mark.gpu
    @pytest.mark.parametrize("weighted", [True, False], ids=["weighted", "unweighted"])
    def test_matches_index_add(self, weighted):
        # Arrange
        torch.manual_seed(0)
        rows = torch.randn(300, 1100, device="cuda", dtype=torch.bfloat16)
        index = torch.randint(0, 64, (300,), device="cuda")
        weights = (
            torch.rand(300, 1, device="cuda", dtype=torch.bfloat16)
            if weighted
            else None
        )
        values = rows.float() if weights is None else rows.float() * weights.float()
        expected = torch.zeros(64, 1100, device="cuda").index_add_(0, index, values)
        out = torch.zeros(64, 1100, device="cuda")

        # Act
        triton_kernels.scatter_add_rows(out, index, rows, weights)

        # Assert
        # fp32 sums of the same products; only the atomic order differs.
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)

    @spawn_new_process_for_each_test
    @pytest.mark.usefixtures("triton_interpreter")
    @pytest.mark.parametrize("weighted", [True, False], ids=["weighted", "unweighted"])
    def test_interpreted_kernel_matches_index_add(self, weighted):
        # Arrange: 600 columns span two scatter blocks.
        torch.manual_seed(0)
        rows = torch.randn(7, 600)
        index = torch.randint(0, 4, (7,))
        weights = torch.rand(7, 1) if weighted else None
        values = rows if weights is None else rows * weights
        expected = torch.zeros(4, 600).index_add_(0, index, values)
        out = torch.zeros(4, 600)

        # Act
        triton_kernels.scatter_add_rows(out, index, rows, weights)

        # Assert
        torch.testing.assert_close(out, expected, **FP32_TOLERANCE)

    def test_no_rows_leaves_out_untouched(self):
        # Arrange
        out = torch.ones(4, 600)

        # Act
        triton_kernels.scatter_add_rows(
            out, torch.empty(0, dtype=torch.long), torch.empty(0, 600), None
        )

        # Assert
        torch.testing.assert_close(out, torch.ones(4, 600))

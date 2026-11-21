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
# DeepSeek-V4-Flash and GLM-5.3-Flash ``swiglu_limit``.
SWIGLU_LIMIT = 10.0


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

    @pytest.mark.gpu
    @pytest.mark.parametrize("rank", [8, 3], ids=["rank_8", "rank_padded"])
    def test_lora_adds_low_rank_product_per_expert(self, rank):
        # Arrange
        torch.manual_seed(0)
        counts = [70, 0, 129, 1]
        x = torch.randn(sum(counts), 96, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(4, 96, 160, device="cuda", dtype=torch.bfloat16)
        xa = torch.randn(sum(counts), rank, device="cuda", dtype=torch.bfloat16)
        lora_b = torch.randn(4, rank, 160, device="cuda", dtype=torch.bfloat16)
        expected = per_expert_matmul(x, weight, counts) + per_expert_matmul(
            xa, lora_b, counts
        )

        # Act
        out = triton_kernels.grouped_matmul(x, weight, offsets(counts), (xa, lora_b))

        # Assert
        torch.testing.assert_close(out.float(), expected, **BF16_TOLERANCE)

    @pytest.mark.gpu
    def test_lora_with_transposed_operand_views(self):
        # Arrange
        torch.manual_seed(0)
        counts = [40, 0, 90]
        x = torch.randn(sum(counts), 64, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(3, 48, 64, device="cuda", dtype=torch.bfloat16)
        xa = torch.randn(sum(counts), 8, device="cuda", dtype=torch.bfloat16)
        lora_b = torch.randn(3, 48, 8, device="cuda", dtype=torch.bfloat16)
        weight_t, lora_b_t = weight.transpose(-2, -1), lora_b.transpose(-2, -1)
        expected = per_expert_matmul(x, weight_t, counts) + per_expert_matmul(
            xa, lora_b_t, counts
        )

        # Act
        out = triton_kernels.grouped_matmul(
            x, weight_t, offsets(counts), (xa, lora_b_t)
        )

        # Assert
        torch.testing.assert_close(out.float(), expected, **BF16_TOLERANCE)

    @spawn_new_process_for_each_test
    @pytest.mark.usefixtures("triton_interpreter")
    @patch("triton.testing.do_bench", run_once)
    def test_interpreted_lora_kernel_matches_per_expert_matmul(self):
        # Arrange
        torch.manual_seed(0)
        counts = [5, 0, 9, 1]
        x = torch.randn(sum(counts), 16)
        weight = torch.randn(len(counts), 16, 24)
        xa = torch.randn(sum(counts), 3)
        lora_b = torch.randn(len(counts), 3, 24)
        expected = per_expert_matmul(x, weight, counts) + per_expert_matmul(
            xa, lora_b, counts
        )

        # Act
        out = triton_kernels.grouped_matmul(
            x, weight, offsets(counts, "cpu"), (xa, lora_b)
        )

        # Assert
        torch.testing.assert_close(out, expected, **FP32_TOLERANCE)


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


def scaled_gather_reference(grad, index, rows, weights):
    gathered = grad.index_select(0, index)
    return gathered * weights.float(), (gathered * rows.float()).sum(-1, keepdim=True)


class TestGatherScaledRows:
    @pytest.mark.gpu
    def test_matches_scaled_index_select(self):
        # Arrange
        torch.manual_seed(0)
        grad = torch.randn(64, 1100, device="cuda")
        index = torch.randint(0, 64, (300,), device="cuda")
        rows = torch.randn(300, 1100, device="cuda", dtype=torch.bfloat16)
        weights = torch.rand(300, 1, device="cuda", dtype=torch.bfloat16)
        expected_rows, expected_weights = scaled_gather_reference(
            grad, index, rows, weights
        )

        # Act
        grad_rows, grad_weights = triton_kernels.gather_scaled_rows(
            grad, index, rows, weights
        )

        # Assert
        assert grad_rows.dtype == torch.bfloat16
        assert grad_weights.dtype == torch.float32
        torch.testing.assert_close(grad_rows.float(), expected_rows, **BF16_TOLERANCE)
        # fp32 sums of the same 1100 products; only the summation order differs.
        torch.testing.assert_close(grad_weights, expected_weights, rtol=1e-4, atol=1e-4)

    @spawn_new_process_for_each_test
    @pytest.mark.usefixtures("triton_interpreter")
    def test_interpreted_kernel_matches_scaled_index_select(self):
        # Arrange: 600 columns span two gather blocks.
        torch.manual_seed(0)
        grad = torch.randn(4, 600)
        index = torch.randint(0, 4, (7,))
        rows = torch.randn(7, 600)
        weights = torch.rand(7, 1)
        expected_rows, expected_weights = scaled_gather_reference(
            grad, index, rows, weights
        )

        # Act
        grad_rows, grad_weights = triton_kernels.gather_scaled_rows(
            grad, index, rows, weights
        )

        # Assert
        torch.testing.assert_close(grad_rows, expected_rows, **FP32_TOLERANCE)
        torch.testing.assert_close(grad_weights, expected_weights, **FP32_TOLERANCE)

    def test_no_rows_returns_empty_grads(self):
        # Act
        grad_rows, grad_weights = triton_kernels.gather_scaled_rows(
            torch.ones(4, 600),
            torch.empty(0, dtype=torch.long),
            torch.empty(0, 600),
            torch.empty(0, 1),
        )

        # Assert
        assert grad_rows.shape == (0, 600)
        assert grad_weights.shape == (0, 1)


def swiglu_reference(projected):
    gate, up = projected.float().chunk(2, dim=-1)
    return torch.nn.functional.silu(gate) * up


def swiglu_grad_reference(grad, projected):
    projected = projected.float().requires_grad_(True)
    swiglu_reference(projected).backward(grad.float())
    return projected.grad


def clamped_swiglu_reference(projected):
    """DeepSeek-V4 / GLM-5.3 experts gate: gate clamped above, up on both sides."""
    gate, up = projected.float().chunk(2, dim=-1)
    gate = gate.clamp(max=SWIGLU_LIMIT)
    up = up.clamp(min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
    return torch.nn.functional.silu(gate) * up


def clamped_swiglu_grad_reference(grad, projected):
    projected = projected.float().requires_grad_(True)
    clamped_swiglu_reference(projected).backward(grad.float())
    return projected.grad


def clamped_inputs(rows, inter, device, dtype):
    """``[gate | up]`` mostly past ``±limit``, with columns exactly on ``limit`` and ``-limit``."""
    projected = torch.randn(rows, 2 * inter, device=device) * 3 * SWIGLU_LIMIT
    projected[:, ::7] = SWIGLU_LIMIT
    projected[:, 3::11] = -SWIGLU_LIMIT
    return projected.to(dtype)


class TestSwiglu:
    @pytest.mark.gpu
    @pytest.mark.parametrize(
        ("rows", "inter"),
        [
            pytest.param(4099, 512, id="qwen36_expert_width_partial_row_tile"),
            pytest.param(37, 600, id="partial_column_tile"),
        ],
    )
    def test_matches_silu_gate_times_up(self, rows, inter):
        # Arrange
        torch.manual_seed(0)
        projected = torch.randn(rows, 2 * inter, device="cuda", dtype=torch.bfloat16)

        # Act
        out = triton_kernels.swiglu(projected, None)

        # Assert
        assert out.dtype == torch.bfloat16
        torch.testing.assert_close(
            out.float(), swiglu_reference(projected), **BF16_TOLERANCE
        )

    @spawn_new_process_for_each_test
    @pytest.mark.usefixtures("triton_interpreter")
    def test_interpreted_kernel_matches_silu_gate_times_up(self):
        # Arrange: 40 columns leave part of the 64-wide tile masked.
        torch.manual_seed(0)
        projected = torch.randn(70, 80)

        # Act
        out = triton_kernels.swiglu(projected, None)

        # Assert
        torch.testing.assert_close(out, swiglu_reference(projected), **FP32_TOLERANCE)

    @pytest.mark.gpu
    @pytest.mark.parametrize(
        ("rows", "inter"),
        [
            pytest.param(4099, 512, id="partial_row_tile"),
            pytest.param(37, 600, id="partial_column_tile"),
        ],
    )
    def test_limit_matches_clamped_silu_gate_times_up(self, rows, inter):
        # Arrange
        torch.manual_seed(0)
        projected = clamped_inputs(rows, inter, "cuda", torch.bfloat16)

        # Act
        out = triton_kernels.swiglu(projected, SWIGLU_LIMIT)

        # Assert
        assert out.dtype == torch.bfloat16
        torch.testing.assert_close(
            out.float(), clamped_swiglu_reference(projected), **BF16_TOLERANCE
        )

    @spawn_new_process_for_each_test
    @pytest.mark.usefixtures("triton_interpreter")
    def test_interpreted_limit_matches_clamped_silu_gate_times_up(self):
        # Arrange
        torch.manual_seed(0)
        projected = clamped_inputs(70, 40, "cpu", torch.float32)

        # Act
        out = triton_kernels.swiglu(projected, SWIGLU_LIMIT)

        # Assert
        torch.testing.assert_close(
            out, clamped_swiglu_reference(projected), **FP32_TOLERANCE
        )

    def test_no_rows_returns_empty_output(self):
        # Act
        out = triton_kernels.swiglu(torch.empty(0, 1024), None)

        # Assert
        assert out.shape == (0, 512)


class TestSwigluBackward:
    @pytest.mark.gpu
    @pytest.mark.parametrize(
        ("rows", "inter"),
        [
            pytest.param(4099, 512, id="qwen36_expert_width_partial_row_tile"),
            pytest.param(37, 600, id="partial_column_tile"),
        ],
    )
    def test_matches_autograd_of_silu_gate_times_up(self, rows, inter):
        # Arrange
        torch.manual_seed(0)
        projected = torch.randn(rows, 2 * inter, device="cuda", dtype=torch.bfloat16)
        grad = torch.randn(rows, inter, device="cuda", dtype=torch.bfloat16)

        # Act
        grad_projected = triton_kernels.swiglu_backward(grad, projected, None)

        # Assert
        assert grad_projected.dtype == torch.bfloat16
        torch.testing.assert_close(
            grad_projected.float(),
            swiglu_grad_reference(grad, projected),
            **BF16_TOLERANCE,
        )

    @spawn_new_process_for_each_test
    @pytest.mark.usefixtures("triton_interpreter")
    def test_interpreted_kernel_matches_autograd_of_silu_gate_times_up(self):
        # Arrange
        torch.manual_seed(0)
        projected = torch.randn(70, 80)
        grad = torch.randn(70, 40)

        # Act
        grad_projected = triton_kernels.swiglu_backward(grad, projected, None)

        # Assert
        torch.testing.assert_close(
            grad_projected, swiglu_grad_reference(grad, projected), **FP32_TOLERANCE
        )

    @pytest.mark.gpu
    @pytest.mark.parametrize(
        ("rows", "inter"),
        [
            pytest.param(4099, 512, id="partial_row_tile"),
            pytest.param(37, 600, id="partial_column_tile"),
        ],
    )
    def test_limit_matches_autograd_of_clamped_silu_gate_times_up(self, rows, inter):
        # Arrange
        torch.manual_seed(0)
        projected = clamped_inputs(rows, inter, "cuda", torch.bfloat16)
        grad = torch.randn(rows, inter, device="cuda", dtype=torch.bfloat16)

        # Act
        grad_projected = triton_kernels.swiglu_backward(grad, projected, SWIGLU_LIMIT)

        # Assert
        assert grad_projected.dtype == torch.bfloat16
        torch.testing.assert_close(
            grad_projected.float(),
            clamped_swiglu_grad_reference(grad, projected),
            **BF16_TOLERANCE,
        )

    @spawn_new_process_for_each_test
    @pytest.mark.usefixtures("triton_interpreter")
    def test_interpreted_limit_matches_autograd_of_clamped_silu_gate_times_up(self):
        # Arrange
        torch.manual_seed(0)
        projected = clamped_inputs(70, 40, "cpu", torch.float32)
        grad = torch.randn(70, 40)

        # Act
        grad_projected = triton_kernels.swiglu_backward(grad, projected, SWIGLU_LIMIT)

        # Assert
        # Entries on ``±limit`` pass their gradient, as ``torch.clamp`` does.
        torch.testing.assert_close(
            grad_projected,
            clamped_swiglu_grad_reference(grad, projected),
            **FP32_TOLERANCE,
        )

    def test_no_rows_returns_empty_grad(self):
        # Act
        grad_projected = triton_kernels.swiglu_backward(
            torch.empty(0, 512), torch.empty(0, 1024), None
        )

        # Assert
        assert grad_projected.shape == (0, 1024)

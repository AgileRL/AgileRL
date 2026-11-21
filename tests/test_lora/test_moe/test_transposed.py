# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from agilerl.lora.moe import transposed as transposed_mod
from agilerl.lora.moe.grouped_gemm import group_offsets
from agilerl.lora.moe.transposed import expert_matmul


def per_expert_reference(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, counts: list[int]
) -> torch.Tensor:
    """``rows @ W[e] + b[e]`` one expert at a time."""
    return torch.cat(
        [
            rows @ weight[expert] + bias[expert]
            for expert, rows in enumerate(x.split(counts))
        ]
    )


class TestExpertMatmul:
    def test_matches_per_expert_outputs_and_grads(self) -> None:
        # Arrange: an empty expert in the middle; fp32 inner dims on 16 bytes.
        torch.manual_seed(0)
        counts = [2, 0, 4]
        x = torch.randn(6, 8, requires_grad=True)
        weight = torch.randn(3, 8, 4, requires_grad=True)
        bias = torch.randn(3, 4, requires_grad=True)
        grad = torch.randn(6, 4)
        ref_x, ref_weight, ref_bias = (
            tensor.detach().clone().requires_grad_(True) for tensor in (x, weight, bias)
        )
        counts_on_device = torch.tensor(counts)

        # Act
        out = expert_matmul(
            x, weight, bias, counts_on_device, group_offsets(counts_on_device, x.device)
        )
        out.backward(grad)
        reference = per_expert_reference(ref_x, ref_weight, ref_bias, counts)
        reference.backward(grad)

        # Assert: one fp32 GEMM per expert either way; tolerances cover summation order.
        assert torch.allclose(out, reference, atol=1e-6, rtol=1e-5)
        assert torch.allclose(x.grad, ref_x.grad, atol=1e-6, rtol=1e-5)
        assert torch.allclose(weight.grad, ref_weight.grad, atol=1e-6, rtol=1e-5)
        assert torch.allclose(bias.grad, ref_bias.grad, atol=1e-6, rtol=1e-5)

    def test_reads_dtensor_local_shard(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Arrange
        torch.manual_seed(0)
        weight = torch.randn(2, 4, 4)
        bias = torch.randn(2, 4)
        rows = torch.randn(4, 4)
        counts = torch.tensor([1, 3])
        offs = group_offsets(counts, rows.device)

        class Shard:
            def __init__(self, local: torch.Tensor) -> None:
                self.local = local

            def to_local(self) -> torch.Tensor:
                return self.local

        monkeypatch.setattr(transposed_mod, "DTensor", Shard)

        # Act
        sharded = expert_matmul(rows, Shard(weight), Shard(bias), counts, offs)

        # Assert
        assert torch.allclose(sharded, per_expert_reference(rows, weight, bias, [1, 3]))

    def test_unaligned_dims_raise_off_cuda(self) -> None:
        # Arrange: 3 fp32 output features are 12 bytes, off the 16-byte boundary.
        counts = torch.tensor([2, 2])
        rows = torch.randn(4, 4)

        # Act / Assert
        with pytest.raises(RuntimeError, match="have no kernel"):
            expert_matmul(
                rows,
                torch.randn(2, 4, 3),
                torch.randn(2, 3),
                counts,
                group_offsets(counts, rows.device),
            )

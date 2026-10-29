# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from peft import LoraConfig, inject_adapter_in_model
from torch import nn

from agilerl.lora.moe import grouped_gemm as moe_gemm
from agilerl.lora.moe.adapters import split_lora_delta

NUM_EXPERTS = 4
HIDDEN = 16
OUT = 8
COUNTS = torch.tensor([3, 0, 5, 4])


class SortedExperts(nn.Module):
    """Grouped linear over expert-sorted rows (GraniteMoe ``ParallelExperts`` convention)."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(NUM_EXPERTS, OUT, HIDDEN) * 0.1)

    def forward(self, inputs, expert_size):
        return moe_gemm.grouped_linear(inputs, self.weight, expert_size)


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.experts = SortedExperts()


def lora_block(rank):
    """bf16 sorted experts with one split-LoRA adapter on the stacked weight."""
    torch.manual_seed(0)
    config = LoraConfig(
        r=rank,
        lora_alpha=8,
        target_modules=[],
        target_parameters=["experts.weight"],
        lora_dropout=0.0,
        init_lora_weights=False,
    )
    model = inject_adapter_in_model(config, Block(), adapter_name="actor")
    return model.to(torch.bfloat16)


def run_delta(model, x):
    """Delta, input grad and LoRA grads of one forward and backward."""
    model.zero_grad(set_to_none=True)
    rows = x.clone().requires_grad_(True)
    delta = split_lora_delta(model.experts, rows, COUNTS, "actor")
    delta.float().square().sum().backward()
    grads = {
        name: param.grad.clone()
        for name, param in model.named_parameters()
        if param.grad is not None
    }
    return delta.detach(), rows.grad, grads


def raise_host_sync(*_args, **_kwargs):
    msg = "host sync"
    raise AssertionError(msg)


class TestSplitLoraDelta:
    @pytest.mark.parametrize("rank", [1, 2, 4, 8])
    def test_bf16_grouped_path_matches_the_unpadded_loop(self, monkeypatch, rank):
        # Arrange
        reference = lora_block(rank)
        model = lora_block(rank)
        torch.manual_seed(1)
        x = torch.randn(int(COUNTS.sum()), HIDDEN, dtype=torch.bfloat16)
        expected = run_delta(reference, x)
        expected_shapes = {
            name: tensor.shape for name, tensor in reference.state_dict().items()
        }

        # Act
        with monkeypatch.context() as patched:
            patched.setattr(moe_gemm, "_use_grouped_mm", lambda _x: True)
            # The per-expert loop reads group sizes back to the host.
            patched.setattr(torch.Tensor, "tolist", raise_host_sync)
            actual = run_delta(model, x)

        # Assert
        # bf16 keeps 8 mantissa bits; both paths sum 16-term products in a
        # different order, so allow a few bf16 ulps.
        tolerance = {"rtol": 2e-2, "atol": 2e-3}
        torch.testing.assert_close(actual[0], expected[0], **tolerance)
        torch.testing.assert_close(actual[1], expected[1], **tolerance)
        assert set(actual[2]) == set(expected[2])
        assert actual[2]
        for name, grad in expected[2].items():
            torch.testing.assert_close(actual[2][name], grad, **tolerance)
        shapes = {name: tensor.shape for name, tensor in model.state_dict().items()}
        assert shapes == expected_shapes

# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F
from peft import LoraConfig, inject_adapter_in_model
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode

from agilerl.lora.moe import grouped_gemm as moe_gemm
from agilerl.lora.moe import routed as moe_routed
from agilerl.lora.moe import (
    set_routed_experts_chunk_bytes,
    set_routed_experts_recompute,
    upgrade_moe_param_wrappers,
)

NUM_EXPERTS = 4
# Multiples of 4 keep fp32 rows 16-byte aligned, so the grouped GEMM path applies.
HIDDEN = 8
INTERMEDIATE = 8
RANK = 4
# Expert 1 gets no rows; experts 0, 2 and 3 get 10, 8 and 6.
TOP2_INDEX = torch.tensor(
    [
        [0, 2],
        [0, 3],
        [2, 0],
        [3, 0],
        [0, 2],
        [2, 3],
        [0, 3],
        [3, 2],
        [0, 2],
        [2, 0],
        [0, 3],
        [0, 2],
    ]
)
SYNC_OPS = {"bincount", "equal", "_local_scalar_dense", "nonzero"}


def relu_squared(t):
    return F.relu(t).square()


class GatedExperts(nn.Module):
    """Packed gated experts with a per-expert loop forward."""

    def __init__(self) -> None:
        super().__init__()
        self.num_experts = NUM_EXPERTS
        self.gate_up_proj = nn.Parameter(
            torch.randn(NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN) * 0.1
        )
        self.down_proj = nn.Parameter(
            torch.randn(NUM_EXPERTS, HIDDEN, INTERMEDIATE) * 0.1
        )
        self.act_fn = F.silu

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        for expert in range(self.num_experts):
            token_idx, top_k_pos = torch.where(top_k_index == expert)
            gate, up = F.linear(
                hidden_states[token_idx], self.gate_up_proj[expert]
            ).chunk(2, dim=-1)
            out = F.linear(self.act_fn(gate) * up, self.down_proj[expert])
            final.index_add_(
                0, token_idx, out * top_k_weights[token_idx, top_k_pos, None]
            )
        return final


class UngatedExperts(nn.Module):
    """Packed ungated relu² experts with a per-expert loop forward."""

    def __init__(self) -> None:
        super().__init__()
        self.num_experts = NUM_EXPERTS
        self.up_proj = nn.Parameter(
            torch.randn(NUM_EXPERTS, INTERMEDIATE, HIDDEN) * 0.1
        )
        self.down_proj = nn.Parameter(
            torch.randn(NUM_EXPERTS, HIDDEN, INTERMEDIATE) * 0.1
        )
        self.act_fn = relu_squared

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        for expert in range(self.num_experts):
            token_idx, top_k_pos = torch.where(top_k_index == expert)
            up = F.linear(hidden_states[token_idx], self.up_proj[expert])
            out = F.linear(self.act_fn(up), self.down_proj[expert])
            final.index_add_(
                0, token_idx, out * top_k_weights[token_idx, top_k_pos, None]
            )
        return final


class Block(nn.Module):
    def __init__(self, experts: nn.Module) -> None:
        super().__init__()
        self.experts = experts


EXPERTS = [
    pytest.param(GatedExperts, "experts.gate_up_proj", id="gated_silu"),
    pytest.param(UngatedExperts, "experts.up_proj", id="ungated_relu2"),
]


def lora_block(experts_cls, up_target):
    """Frozen-base experts with one trainable split-LoRA adapter on both projections."""
    torch.manual_seed(0)
    config = LoraConfig(
        r=RANK,
        lora_alpha=8,
        target_modules=[],
        target_parameters=[up_target, "experts.down_proj"],
        lora_dropout=0.0,
        init_lora_weights=False,
    )
    model = inject_adapter_in_model(config, Block(experts_cls()), adapter_name="actor")
    for name, param in model.named_parameters():
        param.requires_grad_("lora" in name)
    return model


def run_experts(model, hidden, top_k_weights):
    """Forward and backward; return output, input grad, router-weight grad and LoRA grads."""
    model.zero_grad(set_to_none=True)
    rows = hidden.clone().requires_grad_(True)
    weights = top_k_weights.clone().requires_grad_(True)
    out = model.experts(rows, TOP2_INDEX, weights)
    out.square().mean().backward()
    lora_grads = {
        name: param.grad.clone()
        for name, param in model.named_parameters()
        if param.grad is not None
    }
    return out.detach(), rows.grad, weights.grad, lora_grads


class OpRecorder(TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.ops: set[str] = set()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops.add(func.overloadpacket.__name__)
        return func(*args, **(kwargs or {}))


def raise_host_sync(*_args, **_kwargs):
    msg = "host sync"
    raise AssertionError(msg)


class TestRoutedExpertsLocalForwardParity:
    @pytest.mark.parametrize("use_grouped_mm", [False, True], ids=["loop", "gmm"])
    @pytest.mark.parametrize(
        "chunk_bytes",
        [
            pytest.param(1, id="one_row_chunks"),
            pytest.param(160, id="chunks_cut_experts"),
            pytest.param(1 << 30, id="one_chunk"),
        ],
    )
    @pytest.mark.parametrize("recompute", [False, True], ids=["graph", "recompute"])
    @pytest.mark.parametrize(("experts_cls", "up_target"), EXPERTS)
    def test_matches_per_expert_reference(
        self,
        monkeypatch,
        experts_cls,
        up_target,
        recompute,
        chunk_bytes,
        use_grouped_mm,
    ):
        # Arrange
        reference = lora_block(experts_cls, up_target)
        model = lora_block(experts_cls, up_target)
        upgrade_moe_param_wrappers(model)
        set_routed_experts_recompute(model, recompute)
        set_routed_experts_chunk_bytes(model, chunk_bytes)
        torch.manual_seed(1)
        hidden = torch.randn(TOP2_INDEX.shape[0], HIDDEN)
        top_k_weights = torch.rand(TOP2_INDEX.shape)
        if use_grouped_mm:
            monkeypatch.setattr(moe_gemm, "_use_grouped_mm", lambda _x: True)

        # Act
        expected = run_experts(reference, hidden, top_k_weights)
        actual = run_experts(model, hidden, top_k_weights)

        # Assert
        # fp32 with sums of at most 16 products split differently across
        # chunks and experts; only the accumulation order differs.
        tolerance = {"rtol": 1e-5, "atol": 1e-6}
        for got, want in zip(actual[:3], expected[:3], strict=True):
            torch.testing.assert_close(got, want, **tolerance)
        assert set(actual[3]) == set(expected[3])
        assert len(actual[3]) == 4
        for name, grad in expected[3].items():
            torch.testing.assert_close(actual[3][name], grad, **tolerance)


class TestRoutedExpertsLocalForwardChunks:
    @pytest.mark.parametrize(
        ("experts_cls", "up_target", "expected_sizes"),
        [
            # 32-byte rows; 160 bytes holds 5 rows.
            pytest.param(
                UngatedExperts, "experts.up_proj", [5] * 4 + [4], id="ungated"
            ),
            # 64-byte gate-up rows; 160 bytes holds 2 rows.
            pytest.param(GatedExperts, "experts.gate_up_proj", [2] * 12, id="gated"),
        ],
    )
    def test_chunk_bytes_sets_rows_per_chunk(
        self, monkeypatch, experts_cls, up_target, expected_sizes
    ):
        # Arrange
        model = lora_block(experts_cls, up_target)
        upgrade_moe_param_wrappers(model)
        set_routed_experts_chunk_bytes(model, 160)
        planned = []
        row_chunk_offsets = moe_routed.row_chunk_offsets

        def record(counts, total_rows, max_rows):
            sizes, offsets = row_chunk_offsets(counts, total_rows, max_rows)
            planned.append(sizes)
            return sizes, offsets

        monkeypatch.setattr(moe_routed, "row_chunk_offsets", record)

        # Act
        model.experts(
            torch.randn(TOP2_INDEX.shape[0], HIDDEN),
            TOP2_INDEX,
            torch.rand(TOP2_INDEX.shape),
        )

        # Assert
        assert planned == [expected_sizes]

    @pytest.mark.parametrize("recompute", [False, True], ids=["graph", "recompute"])
    @pytest.mark.parametrize(("experts_cls", "up_target"), EXPERTS)
    def test_grouped_mm_path_has_no_host_sync(
        self, monkeypatch, experts_cls, up_target, recompute
    ):
        # Arrange
        model = lora_block(experts_cls, up_target)
        upgrade_moe_param_wrappers(model)
        set_routed_experts_recompute(model, recompute)
        set_routed_experts_chunk_bytes(model, 160)
        monkeypatch.setattr(moe_gemm, "_use_grouped_mm", lambda _x: True)
        hidden = torch.randn(TOP2_INDEX.shape[0], HIDDEN, requires_grad=True)
        top_k_weights = torch.rand(TOP2_INDEX.shape, requires_grad=True)
        recorder = OpRecorder()

        # Act
        with monkeypatch.context() as patched, recorder:
            patched.setattr(torch.Tensor, "tolist", raise_host_sync)
            patched.setattr(torch.Tensor, "item", raise_host_sync)
            out = model.experts(hidden, TOP2_INDEX, top_k_weights)
            out.square().sum().backward()

        # Assert
        assert "_grouped_mm" in recorder.ops
        assert not recorder.ops & SYNC_OPS
        assert hidden.grad is not None
        assert top_k_weights.grad is not None

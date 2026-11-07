# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F
from peft import LoraConfig, inject_adapter_in_model
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode
from transformers import DeepseekV4Config, Glm5NextTextConfig, MiMoV2FlashConfig
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Experts
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextExperts
from transformers.models.mimo_v2_flash.modeling_mimo_v2_flash import (
    MiMoV2FlashExperts,
)

from agilerl.lora.moe import grouped_gemm as moe_gemm
from agilerl.lora.moe import (
    install_packed_expert_grouped_gemm,
    set_routed_experts_chunk_bytes,
    upgrade_moe_param_wrappers,
)
from agilerl.lora.moe import routed as moe_routed
from agilerl.lora.moe.wrappers import RoutedExpertsLoraWrapper

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
# Every token on expert 2 alone; the other experts get no rows.
ALL_TO_ONE_INDEX = torch.full((12, 1), 2)
# Three distinct experts per token.
TOP3_INDEX = torch.stack(
    [
        torch.arange(12) % NUM_EXPERTS,
        (torch.arange(12) + 1) % NUM_EXPERTS,
        (torch.arange(12) + 3) % NUM_EXPERTS,
    ],
    dim=-1,
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

SWIGLU_LIMIT = 10.0
# Unit-variance gate_up weights on 5x hidden rows give gate and up a standard
# deviation near 14, so many entries fall outside ``[-limit, limit]``.
HF_HIDDEN_SCALE = 5.0


def init_hf_experts(experts):
    """HF experts allocate empty weights; fill them from the current seed."""
    with torch.no_grad():
        experts.gate_up_proj.normal_()
        experts.down_proj.normal_(std=0.1)
    return experts


def deepseek_v4_experts():
    config = DeepseekV4Config(
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_local_experts=NUM_EXPERTS,
        swiglu_limit=SWIGLU_LIMIT,
    )
    return init_hf_experts(DeepseekV4Experts(config))


def glm5_next_experts():
    config = Glm5NextTextConfig(
        hidden_size=HIDDEN,
        moe_intermediate_size=INTERMEDIATE,
        num_local_experts=NUM_EXPERTS,
        swiglu_limit=SWIGLU_LIMIT,
    )
    return init_hf_experts(Glm5NextTextExperts(config))


def mimo_v2_flash_experts():
    config = MiMoV2FlashConfig(
        hidden_size=HIDDEN,
        moe_intermediate_size=INTERMEDIATE,
        num_local_experts=NUM_EXPERTS,
    )
    return init_hf_experts(MiMoV2FlashExperts(config))


CLAMPED_HF_EXPERTS = [
    pytest.param(deepseek_v4_experts, id="deepseek_v4"),
    pytest.param(glm5_next_experts, id="glm5_next"),
]


def assert_clamp_active(experts, hidden, top_k_index=TOP2_INDEX):
    """Some routed gate entries pass ``limit`` and some up entries pass ``±limit``."""
    projected = torch.einsum("th,eoh->teo", hidden, experts.gate_up_proj.detach())
    routed = projected[torch.arange(hidden.shape[0]).unsqueeze(-1), top_k_index]
    gate, up = routed.chunk(2, dim=-1)
    assert (gate > SWIGLU_LIMIT).any()
    assert (up.abs() > SWIGLU_LIMIT).any()


def lora_block(experts_cls, up_target, rank=RANK):
    """Frozen-base experts with one trainable split-LoRA adapter on both projections."""
    torch.manual_seed(0)
    config = LoraConfig(
        r=rank,
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


def force_recompute(model, enabled):
    """Pin every routed-experts LoRA wrapper to the recompute or full-graph path."""
    for module in model.modules():
        if isinstance(module, RoutedExpertsLoraWrapper):
            module.recompute = enabled


def run_experts(model, hidden, top_k_weights, top_k_index=TOP2_INDEX):
    """Forward and backward; return output, input grad, router-weight grad and LoRA grads."""
    model.zero_grad(set_to_none=True)
    rows = hidden.clone().requires_grad_(True)
    weights = top_k_weights.clone().requires_grad_(True)
    out = model.experts(rows, top_k_index, weights)
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


def use_backend(request, monkeypatch, backend):
    """Run expert GEMMs on the per-expert loop, ``torch.nn.functional.grouped_mm``, or reference expert kernels."""
    if backend == "gmm":
        monkeypatch.setattr(moe_gemm, "_use_grouped_mm", lambda _x: True)
    if backend == "kernels":
        return request.getfixturevalue("reference_expert_kernels")
    return None


class TestRoutedExpertsLocalForwardParity:
    @pytest.mark.parametrize("backend", ["loop", "gmm", "kernels"])
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
        request,
        monkeypatch,
        experts_cls,
        up_target,
        recompute,
        chunk_bytes,
        backend,
    ):
        # Arrange
        reference = lora_block(experts_cls, up_target)
        model = lora_block(experts_cls, up_target)
        upgrade_moe_param_wrappers(model)
        force_recompute(model, recompute)
        set_routed_experts_chunk_bytes(model, chunk_bytes)
        torch.manual_seed(1)
        hidden = torch.randn(TOP2_INDEX.shape[0], HIDDEN)
        top_k_weights = torch.rand(TOP2_INDEX.shape)
        use_backend(request, monkeypatch, backend)

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

    @pytest.mark.parametrize("backend", ["gmm", "kernels"])
    @pytest.mark.parametrize("recompute", [False, True], ids=["graph", "recompute"])
    @pytest.mark.parametrize(("experts_cls", "up_target"), EXPERTS)
    def test_grouped_path_has_no_host_sync(
        self, request, monkeypatch, experts_cls, up_target, recompute, backend
    ):
        # Arrange
        model = lora_block(experts_cls, up_target)
        upgrade_moe_param_wrappers(model)
        force_recompute(model, recompute)
        set_routed_experts_chunk_bytes(model, 160)
        use_backend(request, monkeypatch, backend)
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


class TestRoutedExpertsLoraRankAlignment:
    @pytest.mark.parametrize("rank", [1, 2, 4, 8])
    @pytest.mark.parametrize("recompute", [False, True], ids=["graph", "recompute"])
    @pytest.mark.parametrize(("experts_cls", "up_target"), EXPERTS)
    def test_bf16_grouped_path_matches_the_unpadded_loop(
        self, monkeypatch, experts_cls, up_target, recompute, rank
    ):
        # Arrange
        reference = lora_block(experts_cls, up_target, rank).to(torch.bfloat16)
        model = lora_block(experts_cls, up_target, rank).to(torch.bfloat16)
        for block in (reference, model):
            upgrade_moe_param_wrappers(block)
            force_recompute(block, recompute)
        torch.manual_seed(1)
        hidden = torch.randn(TOP2_INDEX.shape[0], HIDDEN, dtype=torch.bfloat16)
        top_k_weights = torch.rand(TOP2_INDEX.shape, dtype=torch.bfloat16)
        expected = run_experts(reference, hidden, top_k_weights)
        expected_shapes = {
            name: tensor.shape for name, tensor in reference.state_dict().items()
        }

        # Act
        with monkeypatch.context() as patched:
            patched.setattr(moe_gemm, "_use_grouped_mm", lambda _x: True)
            # The per-expert loop reads group sizes back to the host.
            patched.setattr(torch.Tensor, "tolist", raise_host_sync)
            actual = run_experts(model, hidden, top_k_weights)

        # Assert
        # bf16 keeps 8 mantissa bits; the two paths round sums of at most 16
        # products in a different order, so allow a few bf16 ulps.
        tolerance = {"rtol": 2e-2, "atol": 2e-3}
        for got, want in zip(actual[:3], expected[:3], strict=True):
            torch.testing.assert_close(got, want, **tolerance)
        assert set(actual[3]) == set(expected[3])
        assert len(actual[3]) == 4
        for name, grad in expected[3].items():
            torch.testing.assert_close(actual[3][name], grad, **tolerance)
        shapes = {name: tensor.shape for name, tensor in model.state_dict().items()}
        assert shapes == expected_shapes
        lora_a_rows = {shape[0] for name, shape in shapes.items() if "lora_A" in name}
        assert lora_a_rows == {NUM_EXPERTS * rank}


class TestRoutedExpertsLocalForwardExpertKernels:
    @pytest.mark.parametrize(
        "top_k_index",
        [
            pytest.param(TOP2_INDEX, id="top2_empty_expert"),
            pytest.param(ALL_TO_ONE_INDEX, id="all_to_one_expert"),
            pytest.param(TOP3_INDEX, id="top3"),
        ],
    )
    @pytest.mark.parametrize(
        "recompute",
        [pytest.param(False, id="graph"), pytest.param(True, id="recompute")],
    )
    @pytest.mark.parametrize(("experts_cls", "up_target"), EXPERTS)
    def test_matches_per_expert_reference(
        self,
        reference_expert_kernels,
        experts_cls,
        up_target,
        recompute,
        top_k_index,
    ):
        # Arrange
        reference = lora_block(experts_cls, up_target)
        model = lora_block(experts_cls, up_target)
        upgrade_moe_param_wrappers(model)
        force_recompute(model, recompute)
        set_routed_experts_chunk_bytes(model, 160)
        torch.manual_seed(1)
        hidden = torch.randn(top_k_index.shape[0], HIDDEN)
        top_k_weights = torch.rand(top_k_index.shape)

        # Act
        expected = run_experts(reference, hidden, top_k_weights, top_k_index)
        actual = run_experts(model, hidden, top_k_weights, top_k_index)

        # Assert
        # fp32 with sums of at most 36 products split differently across
        # chunks and experts; only the accumulation order differs.
        tolerance = {"rtol": 1e-5, "atol": 1e-6}
        for got, want in zip(actual[:3], expected[:3], strict=True):
            torch.testing.assert_close(got, want, **tolerance)
        assert set(actual[3]) == set(expected[3])
        for name, grad in expected[3].items():
            torch.testing.assert_close(actual[3][name], grad, **tolerance)
        # The graph path adds the low-rank product in the base GEMM and scales
        # rows in the scatter; the recompute path adds and scales in torch.
        # Both run a gated SiLU on the fused SwiGLU.
        expected_kernels = (
            {"matmul", "weight_grad", "scatter_add"}
            if recompute
            else {
                "matmul",
                "lora_matmul",
                "weight_grad",
                "scatter_add",
                "gather_scaled_rows",
            }
        )
        if experts_cls is GatedExperts:
            expected_kernels |= {"swiglu", "swiglu_backward"}
        assert reference_expert_kernels.used == expected_kernels


class TestRoutedExpertsLocalForwardHfGate:
    @pytest.mark.parametrize("backend", ["loop", "gmm", "kernels"])
    @pytest.mark.parametrize("recompute", [False, True], ids=["graph", "recompute"])
    @pytest.mark.parametrize("make_experts", CLAMPED_HF_EXPERTS)
    def test_clamped_lora_experts_match_the_hf_forward_and_backward(
        self, request, monkeypatch, make_experts, recompute, backend
    ):
        # Arrange
        reference = lora_block(make_experts, "experts.gate_up_proj")
        model = lora_block(make_experts, "experts.gate_up_proj")
        upgrade_moe_param_wrappers(model)
        force_recompute(model, recompute)
        set_routed_experts_chunk_bytes(model, 160)
        torch.manual_seed(1)
        hidden = torch.randn(TOP2_INDEX.shape[0], HIDDEN) * HF_HIDDEN_SCALE
        top_k_weights = torch.rand(TOP2_INDEX.shape)
        kernels = use_backend(request, monkeypatch, backend)

        # Act
        expected = run_experts(reference, hidden, top_k_weights)
        actual = run_experts(model, hidden, top_k_weights)

        # Assert
        assert_clamp_active(reference.experts.get_base_layer(), hidden)
        # fp32 sums of at most 16 products in a different order with terms near
        # 1e2 (ulp 7.6e-6); entries that cancel toward zero keep that absolute error.
        tolerance = {"rtol": 1e-5, "atol": 1e-4}
        for got, want in zip(actual[:3], expected[:3], strict=True):
            torch.testing.assert_close(got, want, **tolerance)
        assert set(actual[3]) == set(expected[3])
        assert len(actual[3]) == 4
        for name, grad in expected[3].items():
            torch.testing.assert_close(actual[3][name], grad, **tolerance)
        if kernels is not None:
            assert {"clamped_swiglu", "clamped_swiglu_backward"} <= kernels.used
            assert not kernels.used & {"swiglu", "swiglu_backward"}

    @pytest.mark.parametrize("backend", ["loop", "gmm", "kernels"])
    @pytest.mark.parametrize("make_experts", CLAMPED_HF_EXPERTS)
    def test_clamped_base_experts_match_the_hf_forward_and_backward(
        self, request, monkeypatch, make_experts, backend
    ):
        # Arrange
        torch.manual_seed(0)
        reference = make_experts()
        torch.manual_seed(0)
        patched = make_experts()
        install_packed_expert_grouped_gemm(patched)
        torch.manual_seed(1)
        hidden = torch.randn(TOP2_INDEX.shape[0], HIDDEN) * HF_HIDDEN_SCALE
        top_k_weights = torch.rand(TOP2_INDEX.shape)
        use_backend(request, monkeypatch, backend)

        def run(experts):
            experts.zero_grad(set_to_none=True)
            rows = hidden.clone().requires_grad_(True)
            weights = top_k_weights.clone().requires_grad_(True)
            out = experts(rows, TOP2_INDEX, weights)
            out.square().mean().backward()
            return (
                out.detach(),
                rows.grad,
                weights.grad,
                experts.gate_up_proj.grad,
                experts.down_proj.grad,
            )

        # Act
        expected = run(reference)
        actual = run(patched)

        # Assert
        assert_clamp_active(reference, hidden)
        # fp32 sums of at most 16 products in a different order.
        for got, want in zip(actual, expected, strict=True):
            torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("recompute", [False, True], ids=["graph", "recompute"])
    def test_default_gate_family_stays_on_the_fused_swiglu(
        self, reference_expert_kernels, recompute
    ):
        # Arrange
        reference = lora_block(mimo_v2_flash_experts, "experts.gate_up_proj")
        model = lora_block(mimo_v2_flash_experts, "experts.gate_up_proj")
        upgrade_moe_param_wrappers(model)
        force_recompute(model, recompute)
        set_routed_experts_chunk_bytes(model, 160)
        torch.manual_seed(1)
        hidden = torch.randn(TOP2_INDEX.shape[0], HIDDEN)
        top_k_weights = torch.rand(TOP2_INDEX.shape)

        # Act
        expected = run_experts(reference, hidden, top_k_weights)
        actual = run_experts(model, hidden, top_k_weights)

        # Assert
        # fp32 sums of at most 16 products in a different order, and the fused
        # SwiGLU backward's closed form against autograd's; a few ulps on ~1e1.
        tolerance = {"rtol": 1e-5, "atol": 1e-5}
        for got, want in zip(actual[:3], expected[:3], strict=True):
            torch.testing.assert_close(got, want, **tolerance)
        for name, grad in expected[3].items():
            torch.testing.assert_close(actual[3][name], grad, **tolerance)
        assert {"swiglu", "swiglu_backward"} <= reference_expert_kernels.used


class TestScatterScaledRows:
    def test_scales_rows_into_fp32_with_source_dtype_grads(
        self, reference_expert_kernels
    ):
        # Arrange
        base = torch.zeros(3, 2, requires_grad=True)
        source = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        weights = torch.tensor(
            [[0.5], [2.0], [1.0]], dtype=torch.bfloat16, requires_grad=True
        )
        index = torch.tensor([2, 0, 2])

        # Act
        out = moe_routed.ScatterScaledRows.apply(
            base.clone(), index, source, weights, reference_expert_kernels.kernels
        )
        out.backward(torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))

        # Assert
        assert out.dtype == torch.float32
        assert torch.equal(
            out.detach(), torch.tensor([[6.0, 8.0], [0.0, 0.0], [5.5, 7.0]])
        )
        assert torch.equal(
            base.grad, torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        )
        assert source.grad.dtype == torch.bfloat16
        assert torch.equal(
            source.grad.float(), torch.tensor([[2.5, 3.0], [2.0, 4.0], [5.0, 6.0]])
        )
        assert weights.grad.dtype == torch.bfloat16
        assert torch.equal(weights.grad.float(), torch.tensor([[17.0], [11.0], [61.0]]))

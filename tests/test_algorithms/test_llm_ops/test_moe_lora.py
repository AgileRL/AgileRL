# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

import copy
from itertools import pairwise
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn.functional as F
from peft import LoraConfig, inject_adapter_in_model
from peft.tuners.lora.layer import ParamWrapper
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    checkpoint_wrapper,
)
from torch.utils.checkpoint import checkpoint
from transformers.modeling_layers import GradientCheckpointingLayer

from agilerl.distributed import fsdp as dmod
from agilerl.distributed import full_shape_views
from agilerl.lora.fused import (
    ROUTING_STATE,
    patch_lora_for_fused_forward,
    set_fused_adapter_routing,
    unpatch_lora_for_fused_forward,
    unset_fused_adapter_routing,
)
from agilerl.lora.moe import (
    RoutedExpertsLoraWrapper,
    SortedExpertsLoraWrapper,
    TransposedExpertsLoraWrapper,
    install_packed_expert_grouped_gemm,
    materializes_expert_lora,
    moe_expert_target_parameters,
    set_routed_experts_chunk_bytes,
    set_routed_experts_recompute,
    transposed_experts_local_forward,
    upgrade_moe_param_wrappers,
)
from agilerl.lora.moe import adapters as moe_adapters
from agilerl.lora.moe import grouped_gemm as moe_gemm
from agilerl.lora.moe import layouts as moe_layouts
from agilerl.lora.moe import routed as moe_routed
from agilerl.lora.moe.recompute import LoraExpertsConfig, LoraExpertsFunction
from agilerl.utils.llm_utils import (
    expert_lora_vllm_key_map,
    filter_peft_state_dict_for_vllm_lora,
)

NUM_EXPERTS = 4
TOP_K = 2
HIDDEN = 8
INTERMEDIATE = 6


class _SortedExperts(nn.Module):
    """Grouped linear over expert-sorted rows (GraniteMoe ``ParallelExperts`` convention)."""

    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(NUM_EXPERTS, output_size, input_size) * 0.1
        )
        self.num_experts = NUM_EXPERTS

    def forward(self, inputs, expert_size):
        rows = inputs.split(expert_size, dim=0)
        return torch.cat(
            [F.linear(rows[e], self.weight[e]) for e in range(self.num_experts)]
        )


class _SortedTopKGate(nn.Module):
    """Top-k gate that returns ``(index_sorted_experts, batch_index, batch_gates, expert_size, logits)``.

    ``batch_index`` permutes tokens into expert-sorted order. JetMoE's ``router``
    uses this layout; any sibling ``router`` with the same tuple works.
    """

    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(HIDDEN, NUM_EXPERTS, bias=False)
        self.num_experts = NUM_EXPERTS
        self.top_k = TOP_K

    def forward(self, hidden_states):
        logits = self.layer(hidden_states)
        top_k_logits, top_k_indices = logits.topk(self.top_k, dim=1)
        gates = torch.softmax(top_k_logits, dim=1).type_as(hidden_states)
        flat_experts = top_k_indices.flatten()
        expert_size = (
            torch.bincount(flat_experts, minlength=self.num_experts).long().tolist()
        )
        index_sorted_experts = flat_experts.argsort(stable=True)
        batch_index = index_sorted_experts.div(self.top_k, rounding_mode="trunc")
        batch_gates = gates.flatten()[index_sorted_experts]
        return index_sorted_experts, batch_index, batch_gates, expert_size, logits


class _SortedMoeBlock(nn.Module):
    """MoE block: gate then grouped ``input_linear`` / ``output_linear`` (JetMoE layout)."""

    def __init__(self) -> None:
        super().__init__()
        self.router = _SortedTopKGate()
        self.input_linear = _SortedExperts(HIDDEN, 2 * INTERMEDIATE)
        self.output_linear = _SortedExperts(INTERMEDIATE, HIDDEN)

    def forward(self, hidden_states):
        _, batch_index, batch_gates, expert_size, _ = self.router(hidden_states)
        expert_inputs = hidden_states[batch_index]
        inner = self.input_linear(expert_inputs, expert_size)
        gate, up = inner.chunk(2, dim=-1)
        expert_outputs = self.output_linear(F.silu(gate) * up, expert_size)
        expert_outputs = expert_outputs * batch_gates.unsqueeze(-1)
        out = torch.zeros_like(hidden_states)
        return out.index_add(0, batch_index, expert_outputs)


class _RoutedExperts(nn.Module):
    """Self-routing packed experts (transformers-5 ``Qwen3MoeExperts`` convention)."""

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
        expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)
        for expert_idx in range(self.num_experts):
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current = hidden_states[token_idx]
            gate, up = F.linear(current, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current = F.linear(self.act_fn(gate) * up, self.down_proj[expert_idx])
            current = current * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, current)
        return final


class _RoutedMoeBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.router = nn.Linear(HIDDEN, NUM_EXPERTS, bias=False)
        self.experts = _RoutedExperts()

    def forward(self, hidden_states):
        logits = self.router(hidden_states)
        top_k_weights, top_k_index = torch.softmax(logits, dim=-1).topk(TOP_K, dim=-1)
        return self.experts(hidden_states, top_k_index, top_k_weights)


class _UngatedExperts(nn.Module):
    """Self-routing packed experts without a gate (NemotronH convention)."""

    def __init__(self) -> None:
        super().__init__()
        self.num_experts = NUM_EXPERTS
        self.up_proj = nn.Parameter(
            torch.randn(NUM_EXPERTS, INTERMEDIATE, HIDDEN) * 0.1
        )
        self.down_proj = nn.Parameter(
            torch.randn(NUM_EXPERTS, HIDDEN, INTERMEDIATE) * 0.1
        )
        self.act_fn = F.silu

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)
        for expert_idx in range(self.num_experts):
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current = F.linear(hidden_states[token_idx], self.up_proj[expert_idx])
            current = F.linear(self.act_fn(current), self.down_proj[expert_idx])
            current = current * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, current)
        return final


class _UngatedMoeBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.router = nn.Linear(HIDDEN, NUM_EXPERTS, bias=False)
        self.experts = _UngatedExperts()

    def forward(self, hidden_states):
        logits = self.router(hidden_states)
        top_k_weights, top_k_index = torch.softmax(logits, dim=-1).topk(TOP_K, dim=-1)
        return self.experts(hidden_states, top_k_index, top_k_weights)


class _TransposedExperts(nn.Module):
    """Packed experts with transposed ``[E, in, out]`` weights and biases.

    ``is_transposed=True`` keeps PEFT's LoRA delta in ``[E, in, out]``, the
    layout ``x @ W + b`` uses.
    """

    is_transposed = True

    def __init__(self) -> None:
        super().__init__()
        self.num_experts = NUM_EXPERTS
        self.gate_up_proj = nn.Parameter(
            torch.randn(NUM_EXPERTS, HIDDEN, 2 * INTERMEDIATE) * 0.1
        )
        self.gate_up_proj_bias = nn.Parameter(
            torch.zeros(NUM_EXPERTS, 2 * INTERMEDIATE)
        )
        self.down_proj = nn.Parameter(
            torch.randn(NUM_EXPERTS, INTERMEDIATE, HIDDEN) * 0.1
        )
        self.down_proj_bias = nn.Parameter(torch.zeros(NUM_EXPERTS, HIDDEN))
        self.alpha = 1.702
        self.limit = 7.0

    def forward(self, hidden_states, router_indices=None, routing_weights=None):
        final = torch.zeros_like(hidden_states)
        expert_mask = F.one_hot(router_indices, num_classes=self.num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)
        for expert_idx in range(self.num_experts):
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current = hidden_states[token_idx]
            gate_up = (
                current @ self.gate_up_proj[expert_idx]
                + self.gate_up_proj_bias[expert_idx]
            )
            gate, up = gate_up[..., ::2], gate_up[..., 1::2]
            gate = gate.clamp(max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)
            gated = (up + 1) * (gate * torch.sigmoid(gate * self.alpha))
            current = (
                gated @ self.down_proj[expert_idx] + self.down_proj_bias[expert_idx]
            )
            current = current * routing_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, current.to(hidden_states.dtype))
        return final


class _TransposedMoeBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.router = nn.Linear(HIDDEN, NUM_EXPERTS, bias=False)
        self.experts = _TransposedExperts()

    def forward(self, hidden_states):
        logits = self.router(hidden_states)
        router_top, router_indices = logits.topk(TOP_K, dim=-1)
        routing_weights = torch.softmax(router_top, dim=-1).type_as(hidden_states)
        return self.experts(hidden_states, router_indices, routing_weights)


class _FusedRoutedExperts(nn.Module):
    """Packed gated experts with silu fused into the GEMM (no ``act_fn``)."""

    def __init__(self) -> None:
        super().__init__()
        self.num_experts = NUM_EXPERTS
        self.gate_up_proj = nn.Parameter(
            torch.randn(NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN) * 0.1
        )
        self.down_proj = nn.Parameter(
            torch.randn(NUM_EXPERTS, HIDDEN, INTERMEDIATE) * 0.1
        )

    def forward(self, hidden_states, top_k_index, top_k_weights):
        return hidden_states


class _FusedRoutedMoeBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.experts = _FusedRoutedExperts()


def _lora_config(target_parameters, **overrides):
    config = {
        "r": 2,
        "lora_alpha": 4,
        "target_modules": [],
        "target_parameters": target_parameters,
        "lora_dropout": 0.0,
        "init_lora_weights": False,
    }
    config.update(overrides)
    return LoraConfig(**config)


def _build_pair(block_cls, target_parameters, **lora_overrides):
    """A (reference, upgraded) pair of adapter-injected blocks with identical weights."""
    torch.manual_seed(0)
    reference = inject_adapter_in_model(
        _lora_config(target_parameters, **lora_overrides),
        block_cls(),
        adapter_name="actor",
    )
    torch.manual_seed(0)
    upgraded = inject_adapter_in_model(
        _lora_config(target_parameters, **lora_overrides),
        block_cls(),
        adapter_name="actor",
    )
    assert upgrade_moe_param_wrappers(upgraded) > 0
    return reference, upgraded


def _sorted_pair():
    return _build_pair(_SortedMoeBlock, ["input_linear.weight", "output_linear.weight"])


def _routed_pair():
    return _build_pair(_RoutedMoeBlock, ["experts.gate_up_proj", "experts.down_proj"])


def _ungated_pair():
    return _build_pair(_UngatedMoeBlock, ["experts.up_proj", "experts.down_proj"])


def _transposed_pair():
    return _build_pair(
        _TransposedMoeBlock, ["experts.gate_up_proj", "experts.down_proj"]
    )


def _wrappers(model):
    return [m for m in model.modules() if isinstance(m, ParamWrapper)]


def _assert_grad_parity(reference, upgraded, atol):
    ref_grads = {
        name: p.grad for name, p in reference.named_parameters() if p.grad is not None
    }
    up_grads = {
        name: p.grad for name, p in upgraded.named_parameters() if p.grad is not None
    }
    assert set(ref_grads) == set(up_grads)
    assert any("lora" in name for name in ref_grads)
    for name, grad in ref_grads.items():
        assert torch.allclose(grad, up_grads[name], atol=atol), name


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(42)


@pytest.mark.parametrize(
    "pair_factory", [_sorted_pair, _routed_pair, _ungated_pair, _transposed_pair]
)
def test_split_lora_matches_peft_default(pair_factory):
    reference, upgraded = pair_factory()
    x = torch.randn(12, HIDDEN)

    ref_out = reference(x)
    up_out = upgraded(x)
    assert torch.allclose(ref_out, up_out, atol=1e-6)

    ref_out.square().mean().backward()
    up_out.square().mean().backward()
    _assert_grad_parity(reference, upgraded, atol=1e-5)


def test_upgrade_selects_wrapper_classes():
    _, sorted_block = _sorted_pair()
    assert {type(m) for m in _wrappers(sorted_block)} == {SortedExpertsLoraWrapper}

    _, routed_block = _routed_pair()
    # Only the outer wrapper of the nested chain is upgraded; the inner one is
    # bypassed by the outer's replacement forward.
    assert RoutedExpertsLoraWrapper in {type(m) for m in _wrappers(routed_block)}

    _, ungated_block = _ungated_pair()
    assert RoutedExpertsLoraWrapper in {type(m) for m in _wrappers(ungated_block)}

    _, transposed_block = _transposed_pair()
    assert TransposedExpertsLoraWrapper in {
        type(m) for m in _wrappers(transposed_block)
    }


class _OddExperts(nn.Module):
    """A stacked 3D weight used through neither supported calling convention."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(NUM_EXPERTS, 6, HIDDEN))

    def forward(self, x):
        return torch.einsum("th,eoh->teo", x, self.weight).mean(1)


def _odd_model():
    model = nn.Sequential()
    model.odd = _OddExperts()
    return model


def test_upgrade_is_idempotent_and_skips_unknown_conventions():
    model = inject_adapter_in_model(
        _lora_config(["odd.weight"]), _odd_model(), adapter_name="actor"
    )
    with pytest.warns(UserWarning, match="unrecognized module conventions"):
        assert upgrade_moe_param_wrappers(model) == 0

    _, upgraded = _sorted_pair()
    assert upgrade_moe_param_wrappers(upgraded) == 0


class TestMaterializesExpertLora:
    @pytest.mark.parametrize(
        "pair_factory", [_sorted_pair, _routed_pair, _ungated_pair, _transposed_pair]
    )
    def test_upgraded_wrappers_run_contracted(self, pair_factory):
        reference, upgraded = pair_factory()

        assert materializes_expert_lora(upgraded) is False
        assert materializes_expert_lora(reference) is True

    def test_unrecognized_convention_stays_materialized(self):
        model = inject_adapter_in_model(
            _lora_config(["odd.weight"]), _odd_model(), adapter_name="actor"
        )
        with pytest.warns(UserWarning, match="unrecognized module conventions"):
            upgrade_moe_param_wrappers(model)

        assert materializes_expert_lora(model) is True

    def test_dense_model_has_no_materialized_expert_lora(self):
        assert materializes_expert_lora(_odd_model()) is False


def test_disabled_adapters_match_base():
    torch.manual_seed(0)
    base = _SortedMoeBlock()
    upgraded = inject_adapter_in_model(
        _lora_config(["input_linear.weight", "output_linear.weight"]),
        copy.deepcopy(base),
        adapter_name="actor",
    )
    upgrade_moe_param_wrappers(upgraded)
    for module in _wrappers(upgraded):
        module.enable_adapters(False)
    x = torch.randn(10, HIDDEN)
    assert torch.allclose(upgraded(x), base(x), atol=1e-6)


def test_merged_adapters_match_split_forward():
    _, upgraded = _sorted_pair()
    x = torch.randn(10, HIDDEN)
    with torch.no_grad():
        split_out = upgraded(x)
        for module in _wrappers(upgraded):
            module.merge()
        merged_out = upgraded(x)
        for module in _wrappers(upgraded):
            module.unmerge()
        unmerged_out = upgraded(x)
    assert torch.allclose(split_out, merged_out, atol=1e-5)
    assert torch.allclose(split_out, unmerged_out, atol=1e-5)


def test_peft_attach_sizes_lora_from_global_shape_without_gathering():
    """Packed-expert LoRA ranks from DTensor global shape, never ``full_tensor()``."""
    block = _RoutedMoeBlock()
    experts = block.experts
    global_up = torch.Size([NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN])
    global_down = torch.Size([NUM_EXPERTS, HIDDEN, INTERMEDIATE])

    class FakeDTensor:
        def __init__(self, shape: torch.Size) -> None:
            self.shape = shape
            self.dtype = torch.float32
            self.device = torch.device("cpu")
            self.requires_grad = True

        def full_tensor(self):
            msg = "packed experts must not be gathered"
            raise AssertionError(msg)

    fake_up = FakeDTensor(global_up)
    fake_down = FakeDTensor(global_down)

    experts._parameters["gate_up_proj"] = fake_up
    experts._parameters["down_proj"] = fake_down

    with (
        patch.object(dmod, "DTensor", FakeDTensor),
        full_shape_views(experts, [fake_up, fake_down]),
    ):
        assert experts.gate_up_proj.shape == global_up
        assert experts.down_proj.shape == global_down

        model = inject_adapter_in_model(
            _lora_config(["experts.gate_up_proj", "experts.down_proj"]),
            block,
            adapter_name="actor",
        )
        assert upgrade_moe_param_wrappers(model) > 0

        wrappers = _wrappers(model)
        assert {wrapper.num_experts for wrapper in wrappers} == {NUM_EXPERTS}
        assert {wrapper.in_features for wrapper in wrappers} == {HIDDEN, INTERMEDIATE}

    assert experts._parameters["gate_up_proj"] is fake_up
    assert experts._parameters["down_proj"] is fake_down


@pytest.mark.parametrize(
    "pair_factory", [_sorted_pair, _routed_pair, _ungated_pair, _transposed_pair]
)
def test_fused_routing_uniform_and_base(pair_factory):
    _, upgraded = pair_factory()
    patch_lora_for_fused_forward(upgraded)
    x = torch.randn(12, HIDDEN)
    with torch.no_grad():
        plain = upgraded(x)

        set_fused_adapter_routing(upgraded, ["actor"] * 12)
        routed = upgraded(x)
        unset_fused_adapter_routing(upgraded)

        set_fused_adapter_routing(upgraded, ["__base__"] * 12)
        base_routed = upgraded(x)
        unset_fused_adapter_routing(upgraded)

        for module in _wrappers(upgraded):
            module.enable_adapters(False)
        base = upgraded(x)
    assert torch.allclose(routed, plain, atol=1e-6)
    assert torch.allclose(base_routed, base, atol=1e-6)
    unpatch_lora_for_fused_forward(upgraded)


def test_fused_mixed_sorted_actor_and_base_matches_uniform_slices():
    _, upgraded = _sorted_pair()
    patch_lora_for_fused_forward(upgraded)
    x = torch.randn(12, HIDDEN)
    with torch.no_grad():
        set_fused_adapter_routing(upgraded, ["actor"] * 12)
        all_actor = upgraded(x)
        set_fused_adapter_routing(upgraded, ["__base__"] * 12)
        all_base = upgraded(x)
        set_fused_adapter_routing(upgraded, ["actor"] * 6 + ["__base__"] * 6)
        mixed = upgraded(x)
    unset_fused_adapter_routing(upgraded)
    unpatch_lora_for_fused_forward(upgraded)

    assert torch.allclose(mixed[:6], all_actor[:6], atol=1e-5)
    assert torch.allclose(mixed[6:], all_base[6:], atol=1e-5)


def _expert_lora_model(block_cls, target_parameters, adapters, *, trainable):
    torch.manual_seed(0)
    model = block_cls()
    for name in adapters:
        model = inject_adapter_in_model(
            _lora_config(target_parameters, init_lora_weights=False),
            model,
            adapter_name=name,
        )
    assert upgrade_moe_param_wrappers(model) > 0
    for wrapper in _wrappers(model):
        wrapper.set_adapter(adapters[0])
    for name, param in model.named_parameters():
        if "lora" not in name:
            continue
        param.requires_grad_(any(tag in name for tag in trainable))
    return model


def _actor_reference_model(block_cls, target_parameters):
    return _expert_lora_model(
        block_cls,
        target_parameters,
        ("actor", "reference"),
        trainable=("actor",),
    )


def test_actor_reference_expert_lora_wrappers_and_freeze():
    model = _actor_reference_model(
        _UngatedMoeBlock, ["experts.up_proj", "experts.down_proj"]
    )
    wrappers = _wrappers(model)
    assert wrappers
    for wrapper in wrappers:
        assert set(wrapper.lora_A) == {"actor", "reference"}
        assert wrapper.lora_A["actor"].weight.requires_grad
        assert not wrapper.lora_A["reference"].weight.requires_grad

    x = torch.randn(12, HIDDEN)
    model.eval()
    with torch.no_grad():
        for wrapper in wrappers:
            wrapper.set_adapter("actor")
        actor_out = model(x)
        for wrapper in wrappers:
            wrapper.set_adapter("reference")
        reference_out = model(x)
        for wrapper in wrappers:
            wrapper.enable_adapters(False)
        base_out = model(x)
        for wrapper in wrappers:
            wrapper.enable_adapters(True)
            wrapper.set_adapter("actor")

    assert not torch.allclose(actor_out, reference_out, atol=1e-5)
    assert not torch.allclose(reference_out, base_out, atol=1e-5)

    before = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if "reference" in name
    }
    model.train()
    model(x).square().mean().backward()
    for name, param in model.named_parameters():
        if "actor" in name and param.requires_grad:
            assert param.grad is not None
            param.data.add_(param.grad, alpha=-0.1)
        if "reference" in name:
            assert param.grad is None or torch.count_nonzero(param.grad) == 0
    for name, snapshot in before.items():
        assert torch.equal(model.get_parameter(name), snapshot)


@pytest.mark.parametrize(
    ("block_cls", "targets"),
    [
        (_RoutedMoeBlock, ["experts.gate_up_proj", "experts.down_proj"]),
        (_UngatedMoeBlock, ["experts.up_proj", "experts.down_proj"]),
        (_SortedMoeBlock, ["input_linear.weight", "output_linear.weight"]),
        (_TransposedMoeBlock, ["experts.gate_up_proj", "experts.down_proj"]),
    ],
)
def test_fused_mixed_actor_reference_matches_uniform_slices(block_cls, targets):
    model = _actor_reference_model(block_cls, targets)
    x = torch.randn(12, HIDDEN)
    patch_lora_for_fused_forward(model)
    with torch.no_grad():
        set_fused_adapter_routing(model, ["actor"] * 12)
        all_actor = model(x)
        set_fused_adapter_routing(model, ["reference"] * 12)
        all_reference = model(x)
        set_fused_adapter_routing(model, ["actor"] * 6 + ["reference"] * 6)
        mixed = model(x)
    unset_fused_adapter_routing(model)
    unpatch_lora_for_fused_forward(model)

    assert torch.allclose(mixed[:6], all_actor[:6], atol=1e-5)
    assert torch.allclose(mixed[6:], all_reference[6:], atol=1e-5)


@pytest.mark.parametrize(
    ("block_cls", "targets"),
    [
        (_RoutedMoeBlock, ["experts.gate_up_proj", "experts.down_proj"]),
        (_UngatedMoeBlock, ["experts.up_proj", "experts.down_proj"]),
        (_SortedMoeBlock, ["input_linear.weight", "output_linear.weight"]),
        (_TransposedMoeBlock, ["experts.gate_up_proj", "experts.down_proj"]),
    ],
)
def test_fused_mixed_actor_critic_matches_uniform_slices(block_cls, targets):
    model = _expert_lora_model(
        block_cls,
        targets,
        ("actor", "critic"),
        trainable=("actor", "critic"),
    )
    x = torch.randn(12, HIDDEN)
    patch_lora_for_fused_forward(model)
    with torch.no_grad():
        set_fused_adapter_routing(model, ["actor"] * 12)
        all_actor = model(x)
        set_fused_adapter_routing(model, ["critic"] * 12)
        all_critic = model(x)
        set_fused_adapter_routing(model, ["actor"] * 6 + ["critic"] * 6)
        mixed = model(x)
    unset_fused_adapter_routing(model)
    unpatch_lora_for_fused_forward(model)

    assert torch.allclose(mixed[:6], all_actor[:6], atol=1e-5)
    assert torch.allclose(mixed[6:], all_critic[6:], atol=1e-5)


@pytest.mark.parametrize(
    ("block_cls", "targets"),
    [
        (_UngatedMoeBlock, ["experts.up_proj", "experts.down_proj"]),
        (_SortedMoeBlock, ["input_linear.weight", "output_linear.weight"]),
        (_TransposedMoeBlock, ["experts.gate_up_proj", "experts.down_proj"]),
    ],
)
def test_actor_critic_expert_lora_both_trainable(block_cls, targets):
    model = _expert_lora_model(
        block_cls,
        targets,
        ("actor", "critic"),
        trainable=("actor", "critic"),
    )
    x = torch.randn(12, HIDDEN)
    patch_lora_for_fused_forward(model)
    set_fused_adapter_routing(model, ["actor"] * 6 + ["critic"] * 6)
    model.train()
    model(x).square().mean().backward()
    unset_fused_adapter_routing(model)
    unpatch_lora_for_fused_forward(model)
    for name, param in model.named_parameters():
        if "lora" not in name:
            continue
        assert param.grad is not None
        assert param.requires_grad


def test_fused_mixed_routing_raises_on_fallback_param_wrapper():
    model = inject_adapter_in_model(
        _lora_config(["odd.weight"]), _odd_model(), adapter_name="actor"
    )
    with pytest.warns(UserWarning, match="unrecognized module conventions"):
        upgrade_moe_param_wrappers(model)
    patch_lora_for_fused_forward(model)
    x = torch.randn(12, HIDDEN)
    with torch.no_grad():
        set_fused_adapter_routing(model, ["actor"] * 12)
        uniform = model(x)
        unset_fused_adapter_routing(model)
        plain = model(x)
        set_fused_adapter_routing(model, ["actor"] * 6 + ["__base__"] * 6)
        with pytest.raises(RuntimeError, match="parameter-level LoRA"):
            model(x)
        unset_fused_adapter_routing(model)
    assert torch.allclose(uniform, plain, atol=1e-6)
    unpatch_lora_for_fused_forward(model)


def test_sorted_mixed_routing_requires_token_index() -> None:
    _, model = _sorted_pair()
    wrapper = next(
        module
        for module in _wrappers(model)
        if isinstance(module, SortedExpertsLoraWrapper)
    )
    x = torch.randn(8, HIDDEN)
    ROUTING_STATE[wrapper] = ["actor"] * 4 + ["__base__"] * 4

    with pytest.raises(RuntimeError, match="token_index"):
        wrapper(x, [2, 2, 2, 2])

    ROUTING_STATE.pop(wrapper, None)


def test_sorted_mixed_routing_rejects_mismatched_token_index() -> None:
    _, model = _sorted_pair()
    wrapper = next(
        module
        for module in _wrappers(model)
        if isinstance(module, SortedExpertsLoraWrapper)
    )
    x = torch.randn(8, HIDDEN)
    wrapper.token_index = torch.tensor([0, 1, 2])
    wrapper.n_tokens = 4
    ROUTING_STATE[wrapper] = ["actor"] * 4 + ["__base__"] * 4

    with pytest.raises(ValueError, match="does not match expert-sorted rows"):
        wrapper(x, [2, 2, 2, 2])

    ROUTING_STATE.pop(wrapper, None)


def test_sorted_mixed_routing_rejects_routing_that_does_not_divide_tokens() -> None:
    _, model = _sorted_pair()
    wrapper = next(
        module
        for module in _wrappers(model)
        if isinstance(module, SortedExpertsLoraWrapper)
    )
    x = torch.randn(8, HIDDEN)
    wrapper.token_index = torch.arange(8) % 4
    wrapper.n_tokens = 4
    ROUTING_STATE[wrapper] = ["actor", "__base__", "actor"]

    with pytest.raises(ValueError, match=r"routing covers 3 rows .* dimension is 4"):
        wrapper(x, [2, 2, 2, 2])

    ROUTING_STATE.pop(wrapper, None)


def test_sorted_gate_copies_batch_index_onto_wrappers() -> None:
    _, model = _sorted_pair()
    x = torch.randn(4, HIDDEN)

    model(x)

    wrappers = [
        module
        for module in _wrappers(model)
        if isinstance(module, SortedExpertsLoraWrapper)
    ]
    assert wrappers
    for wrapper in wrappers:
        assert wrapper.n_tokens == 4
        assert wrapper.token_index is not None
        assert wrapper.token_index.shape == (4 * TOP_K,)


def test_gate_token_index_hook_skips_non_tuple_output() -> None:
    _, model = _sorted_pair()
    hidden = torch.randn(4, HIDDEN)
    model.router.forward = lambda _hidden: torch.zeros(1)

    try:
        model.router(hidden)
    finally:
        del model.router.forward

    for wrapper in _wrappers(model):
        if isinstance(wrapper, SortedExpertsLoraWrapper):
            assert wrapper.token_index is None
            assert wrapper.n_tokens is None


def test_gate_token_index_hook_skips_non_tensor_batch_index() -> None:
    _, model = _sorted_pair()
    hidden = torch.randn(4, HIDDEN)
    model.router.forward = lambda _hidden: (torch.zeros(1), "not-a-tensor")

    try:
        model.router(hidden)
    finally:
        del model.router.forward

    for wrapper in _wrappers(model):
        if isinstance(wrapper, SortedExpertsLoraWrapper):
            assert wrapper.token_index is None
            assert wrapper.n_tokens is None


def test_fused_mixed_sorted_routing_uses_sequence_factor() -> None:
    model = _actor_reference_model(
        _SortedMoeBlock, ["input_linear.weight", "output_linear.weight"]
    )
    x = torch.randn(12, HIDDEN)
    patch_lora_for_fused_forward(model)
    with torch.no_grad():
        set_fused_adapter_routing(model, ["actor"] * 12)
        all_actor = model(x)
        set_fused_adapter_routing(model, ["reference"] * 12)
        all_reference = model(x)
        set_fused_adapter_routing(model, ["actor", "reference"])
        mixed = model(x)
    unset_fused_adapter_routing(model)
    unpatch_lora_for_fused_forward(model)

    assert torch.allclose(mixed[:6], all_actor[:6], atol=1e-5)
    assert torch.allclose(mixed[6:], all_reference[6:], atol=1e-5)


def test_moe_expert_target_parameters_detects_both_conventions():
    assert moe_expert_target_parameters(_SortedMoeBlock()) == [
        "input_linear.weight",
        "output_linear.weight",
    ]
    routed = nn.Sequential()
    routed.moe = _RoutedMoeBlock()
    assert moe_expert_target_parameters(routed) == [
        "moe.experts.down_proj",
        "moe.experts.gate_up_proj",
    ]
    ungated = nn.Sequential()
    ungated.mixer = _UngatedMoeBlock()
    assert moe_expert_target_parameters(ungated) == [
        "mixer.experts.down_proj",
        "mixer.experts.up_proj",
    ]
    fused = nn.Sequential()
    fused.moe = _FusedRoutedMoeBlock()
    assert moe_expert_target_parameters(fused) == [
        "moe.experts.down_proj",
        "moe.experts.gate_up_proj",
    ]
    transposed = nn.Sequential()
    transposed.mlp = _TransposedMoeBlock()
    assert moe_expert_target_parameters(transposed) == [
        "mlp.experts.down_proj",
        "mlp.experts.gate_up_proj",
    ]


def test_expert_lora_vllm_key_map_and_filter():
    _, sorted_block = _sorted_pair()
    key_map = expert_lora_vllm_key_map(sorted_block)
    assert key_map == {
        "input_linear": "experts.base_layer",
        "output_linear": "experts",
    }, key_map

    _, routed_block = _routed_pair()
    key_map = expert_lora_vllm_key_map(routed_block)
    # Whatever the nesting order, gate_up is assigned to <experts>.base_layer and
    # down on <experts> — the file format vLLM's fused-MoE loader parses.
    assert set(key_map.values()) == {"experts", "experts.base_layer"}
    gate_up_key = next(
        name
        for name, module in routed_block.named_modules()
        if isinstance(module, ParamWrapper) and module.parameter_name == "gate_up_proj"
    )
    assert key_map[gate_up_key] == "experts.base_layer"

    state = {
        f"{gate_up_key}.lora_A.weight": torch.zeros(2, 2),
        "other.lora_A.weight": torch.zeros(2, 2),
    }
    filtered = filter_peft_state_dict_for_vllm_lora(state, None, expert_key_map=key_map)
    assert list(filtered) == ["experts.base_layer.lora_A.weight"]

    _, ungated_block = _ungated_pair()
    key_map = expert_lora_vllm_key_map(ungated_block)
    up_key = next(
        name
        for name, module in ungated_block.named_modules()
        if isinstance(module, ParamWrapper) and module.parameter_name == "up_proj"
    )
    assert key_map[up_key] == "experts.base_layer"

    _, transposed_block = _transposed_pair()
    key_map = expert_lora_vllm_key_map(transposed_block)
    gate_up_key = next(
        name
        for name, module in transposed_block.named_modules()
        if isinstance(module, ParamWrapper) and module.parameter_name == "gate_up_proj"
    )
    assert key_map[gate_up_key] == "experts.base_layer"


def test_expert_lora_vllm_key_map_drops_checkpoint_wrapper_segment():
    # Arrange
    _, block = _routed_pair()
    unwrapped_map = expert_lora_vllm_key_map(block)
    model = nn.Module()
    model.layers = nn.ModuleList([checkpoint_wrapper(block)])

    # Act
    key_map = expert_lora_vllm_key_map(model)

    # Assert
    assert key_map == {
        f"layers.0.{key}": f"layers.0.{path}" for key, path in unwrapped_map.items()
    }
    assert set(key_map.values()) == {
        "layers.0.experts",
        "layers.0.experts.base_layer",
    }


def test_expert_lora_vllm_key_map_raises_on_unknown_parameter():
    class _Mystery(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mystery = nn.Parameter(torch.randn(NUM_EXPERTS, 6, HIDDEN))

        def forward(self, x):
            return x

    model = nn.Sequential()
    model.blk = _Mystery()
    model = inject_adapter_in_model(
        _lora_config(["blk.mystery"]), model, adapter_name="actor"
    )
    with pytest.raises(ValueError, match="No vLLM fused-MoE LoRA mapping"):
        expert_lora_vllm_key_map(model)


def _tiny_granite_hybrid():
    from transformers import GraniteMoeHybridConfig, GraniteMoeHybridForCausalLM

    config = GraniteMoeHybridConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=32,
        num_hidden_layers=2,
        layer_types=["mamba", "attention"],
        num_attention_heads=2,
        num_key_value_heads=1,
        num_local_experts=4,
        num_experts_per_tok=2,
        shared_intermediate_size=32,
        mamba_n_heads=4,
        mamba_d_head=16,
        mamba_n_groups=1,
        mamba_d_state=8,
        mamba_d_conv=2,
        mamba_expand=2,
        max_position_embeddings=64,
    )
    return GraniteMoeHybridForCausalLM(config).float()


def _tiny_qwen3_moe():
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    config = Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=32,
        moe_intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_experts=4,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        max_position_embeddings=64,
        head_dim=16,
    )
    return Qwen3MoeForCausalLM(config).float()


@pytest.mark.parametrize(
    ("build", "expected_targets"),
    [
        (
            # transformers >= 5.13 packs granite experts in the standard routed
            # convention; the sorted ParallelExperts convention stays covered by
            # the synthetic tests above.
            _tiny_granite_hybrid,
            [
                "block_sparse_moe.experts.down_proj",
                "block_sparse_moe.experts.gate_up_proj",
            ],
        ),
        (
            _tiny_qwen3_moe,
            ["mlp.experts.down_proj", "mlp.experts.gate_up_proj"],
        ),
    ],
    ids=["granite_hybrid", "qwen3_moe"],
)
def test_transformers_integration_parity(build, expected_targets):
    from peft import get_peft_model

    torch.manual_seed(0)
    base = build()
    assert moe_expert_target_parameters(base) == expected_targets

    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"],
        target_parameters=expected_targets,
        init_lora_weights=False,
        task_type="CAUSAL_LM",
    )
    reference = get_peft_model(copy.deepcopy(base), lora_config, adapter_name="actor")
    upgraded = get_peft_model(copy.deepcopy(base), lora_config, adapter_name="actor")
    upgraded.load_state_dict(reference.state_dict())
    assert upgrade_moe_param_wrappers(upgraded) > 0
    assert any(type(m) is RoutedExpertsLoraWrapper for m in upgraded.modules())

    ids = torch.randint(0, 64, (2, 10))
    ref_out = reference(input_ids=ids).logits
    up_out = upgraded(input_ids=ids).logits
    assert torch.allclose(ref_out, up_out, atol=5e-5)

    ref_out.square().mean().backward()
    up_out.square().mean().backward()
    _assert_grad_parity(reference, upgraded, atol=5e-4)

    key_map = expert_lora_vllm_key_map(upgraded)
    assert len(key_map) == 4  # two wrapped parameters in each of two MoE layers
    for vllm_key in key_map.values():
        assert vllm_key.endswith((".experts", ".experts.base_layer"))


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="grouped-GEMM path is CUDA-only"
)
def test_grouped_mm_fast_path_matches_loop_on_cuda():
    torch.manual_seed(0)
    reference = inject_adapter_in_model(
        _lora_config(["experts.gate_up_proj", "experts.down_proj"]),
        _RoutedMoeBlock(),
        adapter_name="actor",
    )
    torch.manual_seed(0)
    fast = inject_adapter_in_model(
        _lora_config(["experts.gate_up_proj", "experts.down_proj"]),
        _RoutedMoeBlock(),
        adapter_name="actor",
    )
    upgrade_moe_param_wrappers(reference)
    upgrade_moe_param_wrappers(fast)
    reference.cuda()
    fast.cuda()

    x = torch.randn(64, HIDDEN, device="cuda")
    fast_out = fast(x)
    if not moe_gemm._use_grouped_mm(x):
        pytest.skip("torch._grouped_mm unsupported on this GPU")
    with pytest.MonkeyPatch.context() as mp:
        # Force the loop path on the reference copy.
        mp.setattr(moe_gemm, "_use_grouped_mm", lambda _x: False)
        ref_out = reference(x)
    assert torch.allclose(ref_out, fast_out, atol=1e-5)

    ref_out.square().mean().backward()
    fast_out.square().mean().backward()
    _assert_grad_parity(reference, fast, atol=1e-4)


def test_install_packed_expert_grouped_gemm_skips_dense():
    assert install_packed_expert_grouped_gemm(nn.Linear(HIDDEN, HIDDEN)) == 0


def test_install_packed_expert_grouped_gemm_matches_routed_loop():
    torch.manual_seed(0)
    reference = _RoutedMoeBlock()
    patched = copy.deepcopy(reference)
    assert install_packed_expert_grouped_gemm(patched) == 1
    assert install_packed_expert_grouped_gemm(patched) == 0
    x = torch.randn(10, HIDDEN)
    assert torch.allclose(reference(x), patched(x), atol=1e-5)


def test_install_packed_expert_grouped_gemm_matches_sorted_loop():
    torch.manual_seed(0)
    reference = _SortedMoeBlock()
    patched = copy.deepcopy(reference)
    assert install_packed_expert_grouped_gemm(patched) == 2
    x = torch.randn(10, HIDDEN)
    assert torch.allclose(reference(x), patched(x), atol=1e-5)


def test_install_packed_expert_grouped_gemm_matches_ungated_loop():
    torch.manual_seed(0)
    reference = _UngatedMoeBlock()
    patched = copy.deepcopy(reference)
    assert install_packed_expert_grouped_gemm(patched) == 1
    x = torch.randn(10, HIDDEN)
    assert torch.allclose(reference(x), patched(x), atol=1e-5)


@pytest.mark.parametrize("pair_factory", [_sorted_pair, _routed_pair, _ungated_pair])
def test_install_packed_grouped_gemm_after_peft_upgrade_matches(pair_factory):
    reference, upgraded = pair_factory()
    assert install_packed_expert_grouped_gemm(upgraded) > 0
    x = torch.randn(12, HIDDEN)
    assert torch.allclose(reference(x), upgraded(x), atol=1e-6)

    reference.zero_grad()
    upgraded.zero_grad()
    reference(x).square().mean().backward()
    upgraded(x).square().mean().backward()
    _assert_grad_parity(reference, upgraded, atol=1e-5)


def test_grouped_linear_densifies_dtensor_weight(monkeypatch):
    class FakeDTensor:
        def __init__(self, local: torch.Tensor) -> None:
            self._local = local

        def to_local(self) -> torch.Tensor:
            return self._local

    local = torch.randn(NUM_EXPERTS, INTERMEDIATE, HIDDEN)
    monkeypatch.setattr(moe_gemm, "DTensor", FakeDTensor)
    x = torch.randn(8, HIDDEN)
    counts = [2, 2, 2, 2]
    out = moe_gemm.grouped_linear(x, FakeDTensor(local), counts)
    ref = moe_gemm.grouped_linear(x, local, counts)
    assert torch.allclose(out, ref)


def _mock_cuda_torch(monkeypatch, grouped_mm):
    """Route the grouped-mm probe's CUDA tensor factory calls to CPU tensors."""
    real_randn = torch.randn
    real_tensor = torch.tensor

    def cpu_randn(*shape, device=None, dtype=None, generator=None):
        return real_randn(*shape, dtype=dtype or torch.float32)

    def cpu_tensor(data, device=None, dtype=None):
        return real_tensor(data, dtype=dtype)

    monkeypatch.setattr(torch, "randn", cpu_randn)
    monkeypatch.setattr(torch, "tensor", cpu_tensor)
    monkeypatch.setattr(torch, "_grouped_mm", grouped_mm)
    monkeypatch.setattr(torch, "Generator", MagicMock())
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _index: (9, 0))


def test_grouped_mm_probe_false_without_op(monkeypatch):
    monkeypatch.delattr(torch, "_grouped_mm", raising=False)
    moe_gemm.grouped_mm_supported.cache_clear()

    assert moe_gemm.grouped_mm_supported(0, torch.float32) is False


def test_grouped_mm_probe_true_when_op_matches_reference(monkeypatch):
    def fake_grouped_mm(x, w_t, offs):
        bounds = [0, *offs.tolist()]
        return torch.cat(
            [x[start:end] @ w_t[i] for i, (start, end) in enumerate(pairwise(bounds))]
        )

    _mock_cuda_torch(monkeypatch, fake_grouped_mm)
    moe_gemm.grouped_mm_supported.cache_clear()

    assert moe_gemm.grouped_mm_supported(1, torch.bfloat16) is True


def test_grouped_mm_probe_false_when_op_mismatches(monkeypatch):
    _mock_cuda_torch(monkeypatch, lambda x, w_t, offs: torch.zeros(8, 4))
    moe_gemm.grouped_mm_supported.cache_clear()

    assert moe_gemm.grouped_mm_supported(2, torch.bfloat16) is False


@pytest.mark.parametrize(
    ("capability", "dtype"),
    [((8, 0), torch.bfloat16), ((12, 0), torch.bfloat16), ((9, 0), torch.float32)],
)
def test_grouped_mm_probe_false_without_fused_kernel(monkeypatch, capability, dtype):
    def fake_grouped_mm(x, w_t, offs):
        bounds = [0, *offs.tolist()]
        return torch.cat(
            [x[start:end] @ w_t[i] for i, (start, end) in enumerate(pairwise(bounds))]
        )

    _mock_cuda_torch(monkeypatch, fake_grouped_mm)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _index: capability)
    moe_gemm.grouped_mm_supported.cache_clear()

    assert moe_gemm.grouped_mm_supported(3, dtype) is False


def test_grouped_mm_probe_false_when_op_raises(monkeypatch):
    def boom(x, w_t, offs):
        msg = "no grouped mm"
        raise RuntimeError(msg)

    _mock_cuda_torch(monkeypatch, boom)
    moe_gemm.grouped_mm_supported.cache_clear()

    assert moe_gemm.grouped_mm_supported(3, torch.float32) is False


def loop_grouped_mm(x, w_t, offs):
    """Reference ``torch._grouped_mm``: one matmul per offset group."""
    bounds = [0, *offs.tolist()]
    return torch.cat(
        [x[start:end] @ w_t[i] for i, (start, end) in enumerate(pairwise(bounds))]
    )


class TestGroupedMmSupported:
    def test_first_probe_inside_checkpoint_backward_succeeds(self, monkeypatch):
        # Arrange
        torch.manual_seed(0)
        linear = nn.Linear(4, 4)
        hidden = torch.randn(2, 4, requires_grad=True)
        _mock_cuda_torch(monkeypatch, loop_grouped_mm)
        moe_gemm.grouped_mm_supported.cache_clear()
        probe_results = []

        def block(h: torch.Tensor) -> torch.Tensor:
            probe_results.append(moe_gemm.grouped_mm_supported(4, torch.bfloat16))
            return linear(h).relu()

        # Act
        out = checkpoint(block, hidden, use_reentrant=False)
        out.sum().backward()

        # Assert
        assert probe_results == [True, True]
        assert hidden.grad is not None

    def test_first_probe_under_no_grad_reports_supported(self, monkeypatch):
        # Arrange
        _mock_cuda_torch(monkeypatch, loop_grouped_mm)
        moe_gemm.grouped_mm_supported.cache_clear()

        # Act
        with torch.no_grad():
            supported = moe_gemm.grouped_mm_supported(5, torch.bfloat16)

        # Assert
        assert supported is True


def test_use_grouped_mm_consults_probe_for_cuda_tensor(monkeypatch):
    x = MagicMock()
    x.is_cuda = True
    x.device.index = 0
    x.dtype = torch.float32
    probe = MagicMock(return_value=True)
    monkeypatch.setattr(moe_gemm, "grouped_mm_supported", probe)

    assert moe_gemm._use_grouped_mm(x) is True
    probe.assert_called_once_with(0, torch.float32)


def test_use_grouped_mm_missing_index_uses_current_device(monkeypatch):
    x = MagicMock()
    x.is_cuda = True
    x.device.index = None
    x.dtype = torch.float16
    probe = MagicMock(return_value=False)
    monkeypatch.setattr(moe_gemm, "grouped_mm_supported", probe)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)

    assert moe_gemm._use_grouped_mm(x) is False
    probe.assert_called_once_with(3, torch.float16)


def test_counts_tensor_passes_tensors_through():
    counts = torch.tensor([2, 3])

    assert moe_gemm.counts_tensor(counts, torch.device("cpu")) is counts
    out = moe_gemm.counts_tensor([2, 3], torch.device("cpu"))
    assert torch.equal(out, torch.tensor([2, 3]))


def test_group_offsets_cumulates_counts():
    out = moe_gemm.group_offsets([2, 3], torch.device("cpu"))

    assert out.dtype == torch.int32
    assert torch.equal(out, torch.tensor([2, 5], dtype=torch.int32))


def test_routed_act_fn_falls_back_to_config_hidden_act():
    from types import SimpleNamespace

    experts = nn.Linear(4, 4)
    experts.config = SimpleNamespace(hidden_act="gelu")

    act_fn = moe_layouts.routed_experts_act_fn(experts)

    assert callable(act_fn)
    assert torch.allclose(act_fn(torch.tensor([0.0])), torch.tensor([0.0]), atol=1e-6)


def test_routed_act_fn_raises_without_activation():
    with pytest.raises(RuntimeError, match="activation function"):
        moe_layouts.routed_experts_act_fn(nn.Linear(4, 4))


def test_grouped_linear_noncontiguous_weight_skips_empty_experts():
    weight = torch.arange(3 * 4 * 5, dtype=torch.float32).reshape(3, 4, 5)
    padded = torch.zeros(3, 4, 8)
    padded[:, :, :5] = weight
    view = padded[:, :, :5]
    assert not view.is_contiguous()
    rows = torch.randn(4, 5)
    counts = [0, 3, 1]

    out = moe_gemm.grouped_linear(rows, view, counts)

    offset = 0
    parts = []
    for expert, count in enumerate(counts):
        if count == 0:
            continue
        parts.append(
            nn.functional.linear(rows[offset : offset + count], weight[expert])
        )
        offset += count
    assert torch.allclose(out, torch.cat(parts))


def test_grouped_linear_rejects_non_tensor_module_weight():
    module = nn.Module()
    module.weight = 123

    with pytest.raises(TypeError, match="must be a Tensor"):
        moe_gemm.grouped_linear(module, torch.ones(4, 4), [4])


def test_grouped_linear_rejects_non_tensor_input():
    with pytest.raises(TypeError, match="must be a Tensor"):
        moe_gemm.grouped_linear("nope", torch.ones(4, 4), [4])


def test_grouped_linear_fast_path_keeps_plain_transpose(monkeypatch):
    x = torch.randn(4, 8)
    weight = torch.randn(2, 4, 8)
    seen: dict[str, object] = {}

    def fake_grouped_mm(rows, operand, offs):
        seen["operand"] = operand
        seen["offs"] = offs
        return torch.zeros(4, 4)

    monkeypatch.setattr(moe_gemm, "_use_grouped_mm", lambda _x: True)
    monkeypatch.setattr(torch, "_grouped_mm", fake_grouped_mm)

    out = moe_gemm.grouped_linear(x, weight, [2, 2])

    assert out.shape == (4, 4)
    assert seen["operand"].data_ptr() == weight.data_ptr()
    assert seen["operand"].shape == (2, 8, 4)
    assert torch.equal(seen["offs"], torch.tensor([2, 4], dtype=torch.int32))


def test_grouped_linear_fast_path_copies_irregular_stride(monkeypatch):
    x = torch.randn(4, 8)
    base = torch.randn(2, 8, 16)
    weight = base[:, ::2, ::2]
    seen: dict[str, object] = {}

    def fake_grouped_mm(rows, operand, offs):
        seen["operand"] = operand
        return torch.zeros(4, 4)

    monkeypatch.setattr(moe_gemm, "_use_grouped_mm", lambda _x: True)
    monkeypatch.setattr(torch, "_grouped_mm", fake_grouped_mm)

    moe_gemm.grouped_linear(x, weight, [2, 2])

    operand = seen["operand"]
    assert operand.is_contiguous()
    assert operand.shape == (2, 8, 4)


def test_forward_param_names_empty_when_signature_missing():
    class Broken(nn.Module):
        forward = None

    assert moe_layouts._forward_param_names(Broken()) == []


def test_routed_projection_names_rejects_expert_bias():
    experts = _RoutedExperts()
    experts.down_proj_bias = nn.Parameter(torch.zeros(1))

    assert moe_layouts.routed_projection_names(experts) is None


def test_routed_projection_names_rejects_wrong_forward():
    class WrongForward(nn.Module):
        def __init__(self):
            super().__init__()
            self.down_proj = nn.Parameter(torch.randn(2, 4, 3))
            self.gate_up_proj = nn.Parameter(torch.randn(2, 6, 3))

        def forward(self, hidden_states):
            return hidden_states

    assert moe_layouts.routed_projection_names(WrongForward()) is None


def test_routed_projection_names_skips_biased_up_projection():
    experts = _RoutedExperts()
    experts.gate_up_proj_bias = nn.Parameter(torch.zeros(1))

    assert moe_layouts.routed_projection_names(experts) is None


def test_routed_projection_names_rejects_odd_gated_width():
    class OddGate(nn.Module):
        def __init__(self):
            super().__init__()
            self.down_proj = nn.Parameter(torch.randn(2, 4, 3))
            self.gate_up_proj = nn.Parameter(torch.randn(2, 3, 4))

        def forward(self, hidden_states, top_k_index, top_k_weights):
            return hidden_states

    assert moe_layouts.routed_projection_names(OddGate()) is None


def test_expert_counts_rejects_wrong_length():
    with pytest.raises(ValueError, match="per-expert counts"):
        moe_adapters.expert_counts([1, 2], 3)


def test_resolve_adapters_unmerges_when_disabled(monkeypatch):
    wrapper = MagicMock()
    wrapper.disable_adapters = True
    wrapper.merged = True
    monkeypatch.setattr(moe_adapters, "uniform_routed_adapter", lambda _w: None)

    assert moe_adapters.resolve_adapters(wrapper) == []
    wrapper.unmerge.assert_called_once_with()


def test_split_lora_delta_dtensor_fallback(monkeypatch):
    class FakeDTensor(nn.Parameter):
        pass

    experts, rank, total = 2, 2, 4
    x = torch.randn(total, 8)
    lora_a = MagicMock()
    lora_a.weight = FakeDTensor(torch.ones(experts, rank, 8))
    lora_a.return_value = (
        torch.arange(total * experts * rank).reshape(total, -1).float()
    )
    lora_b = MagicMock()
    lora_b.weight = FakeDTensor(torch.ones(6, experts * rank))
    lora_b.side_effect = lambda t: t @ torch.ones(t.shape[1], 6)
    wrapper = MagicMock()
    wrapper.lora_A = {"actor": lora_a}
    wrapper.lora_B = {"actor": lora_b}
    wrapper.scaling = {"actor": 2.0}
    wrapper.r = {"actor": rank}
    monkeypatch.setattr(moe_adapters, "DTensor", FakeDTensor)

    out = moe_adapters.split_lora_delta(
        wrapper, x, [2, 2], "actor", num_experts=experts
    )

    assert out.shape == (total, 6)
    lora_a.assert_called_once()
    lora_b.assert_called_once()


def test_split_lora_delta_stacked_layouts():
    torch.manual_seed(0)
    experts, rank, hidden, out_dim = 2, 2, 8, 6
    x = torch.randn(4, hidden)

    torch.manual_seed(1)
    a3 = torch.randn(experts, rank, hidden)
    b_exp_first = torch.randn(experts, out_dim, rank)
    b_exp_last = b_exp_first.permute(1, 2, 0).contiguous()

    def run(weight_b):
        lora_a = MagicMock()
        lora_a.weight = a3
        lora_b = MagicMock()
        lora_b.weight = weight_b
        wrapper = MagicMock()
        wrapper.lora_A = {"actor": lora_a}
        wrapper.lora_B = {"actor": lora_b}
        wrapper.scaling = {"actor": 1.0}
        wrapper.r = {"actor": rank}
        wrapper.num_experts = experts
        return moe_adapters.split_lora_delta(wrapper, x, [2, 2], "actor")

    grouped = run(b_exp_first)
    transposed = run(b_exp_last)
    assert grouped.shape == (4, out_dim)
    assert torch.allclose(grouped, transposed)


def test_split_lora_delta_chunked_accumulate_matches_full():
    torch.manual_seed(0)
    experts, rank, hidden, out_dim = 3, 2, 8, 6
    counts = [5, 0, 7]
    total = sum(counts)
    x = torch.randn(total, hidden, requires_grad=True)
    a3 = torch.randn(experts, rank, hidden, requires_grad=True)
    b3 = torch.randn(experts, out_dim, rank, requires_grad=True)

    def run(rows, weight_a, weight_b, destination, chunk_bytes):
        lora_a = MagicMock()
        lora_a.weight = weight_a
        lora_b = MagicMock()
        lora_b.weight = weight_b
        wrapper = MagicMock()
        wrapper.lora_A = {"actor": lora_a}
        wrapper.lora_B = {"actor": lora_b}
        wrapper.scaling = {"actor": 0.5}
        wrapper.r = {"actor": rank}
        wrapper.num_experts = experts
        return moe_adapters.split_lora_delta(
            wrapper,
            rows,
            counts,
            "actor",
            destination=destination,
            chunk_bytes=chunk_bytes,
        )

    full = run(x, a3, b3, None, moe_gemm.ROUTED_EXPERT_CHUNK_BYTES)
    rows = x.detach().clone().requires_grad_(True)
    weight_a = a3.detach().clone().requires_grad_(True)
    weight_b = b3.detach().clone().requires_grad_(True)
    # Non-leaf, same as the expert GEMM output the delta is added into.
    destination = torch.zeros(total, out_dim, requires_grad=True) + 0
    chunked = run(rows, weight_a, weight_b, destination, 1)

    assert chunked is destination
    assert torch.allclose(chunked, full)
    full.sum().backward()
    chunked.sum().backward()
    # Partial GEMMs sum in a different order than one GEMM.
    assert torch.allclose(rows.grad, x.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(weight_a.grad, a3.grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(weight_b.grad, b3.grad, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("experts_cls", [_UngatedExperts, _RoutedExperts])
def test_scatter_routed_chunks_match_module(experts_cls):
    torch.manual_seed(0)
    experts = experts_cls()
    hidden = torch.randn(5, HIDDEN, requires_grad=True)
    top_k_index = torch.randint(0, NUM_EXPERTS, (5, TOP_K))
    top_k_weights = torch.rand(5, TOP_K)
    ref = experts(hidden, top_k_index, top_k_weights)

    other = hidden.detach().clone().requires_grad_(True)
    out = moe_routed.routed_experts_local_forward(
        experts, other, top_k_index, top_k_weights, chunk_bytes=1
    )

    assert torch.allclose(out, ref, rtol=1e-5, atol=1e-6)
    out.sum().backward()
    assert other.grad is not None
    fine_hidden = other.grad.detach().clone()
    fine_params = {
        name: param.grad.detach().clone() for name, param in experts.named_parameters()
    }
    experts.zero_grad()

    coarse_hidden = hidden.detach().clone().requires_grad_(True)
    coarse = moe_routed.routed_experts_local_forward(
        experts,
        coarse_hidden,
        top_k_index,
        top_k_weights,
        chunk_bytes=768 * 1024 * 1024,
    )
    coarse.sum().backward()
    assert torch.allclose(coarse_hidden.grad, fine_hidden, rtol=1e-5, atol=1e-6)
    for name, param in experts.named_parameters():
        assert torch.allclose(param.grad, fine_params[name], rtol=1e-5, atol=1e-6)


def test_grouped_expert_rows_match_shuffled_rows():
    torch.manual_seed(1)
    experts = _UngatedExperts()
    expert_ids = torch.tensor([0, 0, 1, 2, 2, 3])
    hidden = torch.randn(expert_ids.shape[0], HIDDEN, requires_grad=True)
    weights = torch.rand(expert_ids.shape[0], 1)
    grouped = moe_routed.routed_experts_local_forward(
        experts,
        hidden,
        expert_ids.unsqueeze(-1),
        weights,
    )

    perm = torch.tensor([3, 0, 5, 1, 4, 2])
    shuffled_hidden = hidden.detach()[perm].clone().requires_grad_(True)
    shuffled = moe_routed.routed_experts_local_forward(
        experts,
        shuffled_hidden,
        expert_ids[perm].unsqueeze(-1),
        weights[perm],
    )
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(perm.shape[0])

    assert torch.allclose(grouped, shuffled[inverse], rtol=1e-5, atol=1e-6)
    grouped.sum().backward()
    shuffled.sum().backward()
    assert hidden.grad is not None
    assert torch.allclose(
        hidden.grad, shuffled_hidden.grad[inverse], rtol=1e-5, atol=1e-6
    )


def test_routed_local_forward_rejects_unsupported_layout():
    with pytest.raises(RuntimeError, match="supported packed layout"):
        moe_routed.routed_experts_local_forward(
            nn.Linear(4, 4),
            torch.randn(2, 4),
            torch.zeros(2, 1, dtype=torch.long),
            torch.ones(2, 1),
        )


def test_routed_local_forward_resolves_chain_adapters(monkeypatch):
    torch.manual_seed(0)
    experts = _RoutedExperts()
    hidden = torch.randn(4, HIDDEN)
    top_k_index = torch.randint(0, NUM_EXPERTS, (4, TOP_K))
    top_k_weights = torch.rand(4, TOP_K)
    chain = {"gate_up_proj": MagicMock()}
    monkeypatch.setattr(moe_routed, "resolve_adapters", lambda _w: [])

    out = moe_routed.routed_experts_local_forward(
        experts, hidden, top_k_index, top_k_weights, chain=chain
    )

    ref = moe_routed.routed_experts_local_forward(
        experts, hidden, top_k_index, top_k_weights
    )
    assert torch.allclose(out, ref)


def test_routed_wrapper_delegates_on_extra_args():
    _reference, upgraded = _routed_pair()
    wrapper = next(
        m for m in upgraded.modules() if isinstance(m, RoutedExpertsLoraWrapper)
    )
    hidden = torch.randn(4, HIDDEN)
    top_k_index = torch.zeros(4, TOP_K, dtype=torch.long)
    top_k_weights = torch.ones(4, TOP_K)

    with patch.object(
        ParamWrapper, "forward", return_value="peft-default"
    ) as mock_forward:
        out = wrapper(hidden, top_k_index, top_k_weights, "extra")

    mock_forward.assert_called_once()
    assert out == "peft-default"


def test_routed_wrapper_delegates_when_layout_unknown():
    _reference, upgraded = _routed_pair()
    wrapper = next(
        m for m in upgraded.modules() if isinstance(m, RoutedExpertsLoraWrapper)
    )
    experts = wrapper.get_base_layer()
    experts.gate_up_proj = nn.Parameter(torch.randn(2 * INTERMEDIATE, HIDDEN))
    hidden = torch.randn(4, HIDDEN)
    top_k_index = torch.zeros(4, TOP_K, dtype=torch.long)
    top_k_weights = torch.ones(4, TOP_K)

    with patch.object(
        ParamWrapper, "forward", return_value="peft-default"
    ) as mock_forward:
        with patch(
            "agilerl.lora.moe.wrappers.resolve_adapters",
            return_value=["actor"],
        ):
            out = wrapper(hidden, top_k_index, top_k_weights)

    mock_forward.assert_called_once()
    assert out == "peft-default"


class TestTransposedExpertsLocalForward:
    def test_rejects_other_layouts(self) -> None:
        with pytest.raises(RuntimeError, match="transposed packed-experts layout"):
            transposed_experts_local_forward(
                nn.Linear(4, 4),
                torch.randn(2, 4),
                torch.zeros(2, 1, dtype=torch.long),
                torch.ones(2, 1),
            )

    def test_resolves_adapters_from_the_chain(self) -> None:
        _reference, upgraded = _transposed_pair()
        wrapper = upgraded.experts
        hidden = torch.randn(4, HIDDEN)
        indices = torch.zeros(4, TOP_K, dtype=torch.long)
        weights = torch.full((4, TOP_K), 0.5)
        chain = moe_adapters.wrapper_chain(wrapper)

        direct = transposed_experts_local_forward(
            wrapper.get_base_layer(),
            hidden,
            indices,
            weights,
            chain=chain,
        )
        via_wrapper = wrapper(hidden, indices, weights)

        assert torch.allclose(direct, via_wrapper)


class TestTransposedExpertsLoraWrapperForward:
    def test_delegates_on_extra_args(self) -> None:
        _reference, upgraded = _transposed_pair()
        wrapper = upgraded.experts
        hidden = torch.randn(4, HIDDEN)
        indices = torch.zeros(4, TOP_K, dtype=torch.long)
        weights = torch.ones(4, TOP_K)

        with patch.object(ParamWrapper, "forward", return_value=hidden):
            out = wrapper(hidden, indices, weights, "extra")

        assert out is hidden

    def test_delegates_when_layout_unknown(self) -> None:
        _reference, upgraded = _transposed_pair()
        wrapper = upgraded.experts
        experts = wrapper.get_base_layer()
        experts.gate_up_proj = nn.Parameter(torch.zeros(NUM_EXPERTS, HIDDEN))
        hidden = torch.randn(4, HIDDEN)
        indices = torch.zeros(4, TOP_K, dtype=torch.long)
        weights = torch.ones(4, TOP_K)

        with patch.object(ParamWrapper, "forward", return_value=hidden):
            out = wrapper(hidden, indices, weights)

        assert out is hidden


ROUTED_LORA_BLOCKS = [
    pytest.param(
        _RoutedMoeBlock, ["experts.gate_up_proj", "experts.down_proj"], id="gated"
    ),
    pytest.param(
        _UngatedMoeBlock, ["experts.up_proj", "experts.down_proj"], id="ungated"
    ),
]
TOP2_INDEX = torch.tensor(
    [
        [0, 2],
        [1, 3],
        [2, 0],
        [3, 1],
        [1, 2],
        [0, 3],
        [2, 3],
        [1, 0],
        [3, 2],
        [0, 1],
        [2, 1],
        [3, 0],
    ]
)
# 5 gated or 7 ungated expert rows per chunk.
SMALL_CHUNK_BYTES = 240


class TestRoutedExpertsChunkedLora:
    @pytest.mark.parametrize("use_grouped_mm", [False, True], ids=["loop", "gmm"])
    @pytest.mark.parametrize(
        "top_k_index",
        [
            pytest.param(TOP2_INDEX, id="top2"),
            # 12 rows on expert 2 cross two chunk boundaries.
            pytest.param(torch.full((12, 1), 2), id="expert_spans_chunks"),
            pytest.param(torch.tensor([[0, 3]] * 12), id="zero_row_experts"),
        ],
    )
    @pytest.mark.parametrize(("block_cls", "targets"), ROUTED_LORA_BLOCKS)
    def test_multi_chunk_matches_peft_reference(
        self, monkeypatch, block_cls, targets, top_k_index, use_grouped_mm
    ):
        # Arrange
        reference, upgraded = _build_pair(block_cls, targets, r=4)
        torch.manual_seed(0)
        hidden = torch.randn(top_k_index.shape[0], HIDDEN)
        top_k_weights = torch.rand(top_k_index.shape)
        ref_hidden = hidden.clone().requires_grad_(True)
        up_hidden = hidden.clone().requires_grad_(True)
        set_routed_experts_chunk_bytes(upgraded, SMALL_CHUNK_BYTES)
        if use_grouped_mm:
            monkeypatch.setattr(moe_gemm, "_use_grouped_mm", lambda _x: True)

        # Act
        ref_out = reference.experts(ref_hidden, top_k_index, top_k_weights)
        up_out = upgraded.experts(up_hidden, top_k_index, top_k_weights)
        ref_out.square().mean().backward()
        up_out.square().mean().backward()

        # Assert
        assert torch.allclose(up_out, ref_out, rtol=1e-5, atol=1e-6)
        assert torch.allclose(up_hidden.grad, ref_hidden.grad, rtol=1e-5, atol=1e-6)
        _assert_grad_parity(reference, upgraded, atol=1e-6)

    @pytest.mark.parametrize(("block_cls", "targets"), ROUTED_LORA_BLOCKS)
    def test_mixed_routing_multi_chunk_matches_per_adapter_runs(
        self, block_cls, targets
    ):
        # Arrange
        model = _expert_lora_model(
            block_cls, targets, ("actor", "critic"), trainable=("actor", "critic")
        )
        patch_lora_for_fused_forward(model)
        x = torch.randn(12, HIDDEN)
        set_routed_experts_chunk_bytes(model, SMALL_CHUNK_BYTES)

        # Act
        set_fused_adapter_routing(model, ["actor"] * 6 + ["critic"] * 6)
        mixed = model(x)
        mixed.square().sum().backward()
        mixed_grads = {
            name: param.grad.clone()
            for name, param in model.named_parameters()
            if param.grad is not None
        }
        model.zero_grad()
        set_fused_adapter_routing(model, ["actor"] * 6)
        actor = model(x[:6])
        set_fused_adapter_routing(model, ["critic"] * 6)
        critic = model(x[6:])
        (actor.square().sum() + critic.square().sum()).backward()
        unset_fused_adapter_routing(model)
        unpatch_lora_for_fused_forward(model)

        # Assert
        assert torch.allclose(mixed, torch.cat([actor, critic]), rtol=1e-5, atol=1e-6)
        split_grads = {
            name: param.grad
            for name, param in model.named_parameters()
            if param.grad is not None
        }
        assert set(mixed_grads) == set(split_grads)
        assert any("critic" in name for name in mixed_grads)
        for name, grad in mixed_grads.items():
            assert torch.allclose(grad, split_grads[name], rtol=1e-5, atol=1e-6), name

    def test_partitioned_lora_multi_chunk_matches_peft_reference(self, monkeypatch):
        # Arrange
        class FakeDTensor(nn.Parameter):
            pass

        reference, upgraded = _build_pair(
            _RoutedMoeBlock, ["experts.gate_up_proj", "experts.down_proj"]
        )
        for wrapper in _wrappers(upgraded):
            for layer in (*wrapper.lora_A.values(), *wrapper.lora_B.values()):
                layer.weight = FakeDTensor(layer.weight.detach().clone())
        monkeypatch.setattr(moe_adapters, "DTensor", FakeDTensor)
        set_routed_experts_chunk_bytes(upgraded, SMALL_CHUNK_BYTES)
        hidden = torch.randn(12, HIDDEN)
        top_k_weights = torch.rand(12, TOP_K)

        # Act
        ref_out = reference.experts(hidden, TOP2_INDEX, top_k_weights)
        up_out = upgraded.experts(hidden, TOP2_INDEX, top_k_weights)

        # Assert
        assert torch.allclose(up_out, ref_out, rtol=1e-5, atol=1e-6)


class TestRoutedExpertsAlreadyGrouped:
    def test_explicit_flag_matches_detected_and_gathered_rows(self):
        # Arrange
        _reference, upgraded = _build_pair(
            _UngatedMoeBlock, ["experts.up_proj", "experts.down_proj"]
        )
        expert_ids = torch.tensor([[0], [0], [1], [3], [3], [3]])
        hidden = torch.randn(6, HIDDEN)
        top_k_weights = torch.rand(6, 1)
        outs = {}
        grads = {}

        # Act
        for flag in (None, True, False):
            rows = hidden.clone().requires_grad_(True)
            out = upgraded.experts(
                rows, expert_ids, top_k_weights, already_grouped=flag
            )
            out.square().sum().backward()
            outs[flag] = out.detach()
            grads[flag] = rows.grad

        # Assert
        for flag in (True, False):
            assert torch.equal(outs[flag], outs[None])
            assert torch.equal(grads[flag], grads[None])

    def test_rejects_more_than_one_expert_per_row(self):
        experts = _UngatedExperts()

        with pytest.raises(
            ValueError, match="already_grouped needs one expert per row"
        ):
            moe_routed.routed_experts_local_forward(
                experts,
                torch.randn(4, HIDDEN),
                torch.zeros(4, TOP_K, dtype=torch.long),
                torch.ones(4, TOP_K),
                already_grouped=True,
            )


RECOMPUTE_BLOCKS = [
    pytest.param(
        _RoutedMoeBlock,
        ["experts.gate_up_proj", "experts.down_proj"],
        F.silu,
        id="gated_silu",
    ),
    pytest.param(
        _UngatedMoeBlock,
        ["experts.up_proj", "experts.down_proj"],
        lambda t: F.relu(t).square(),
        id="ungated_relu2",
    ),
]
# Expert 1 gets no rows; experts 0, 2 and 3 get 10, 8 and 6.
UNEVEN_TOP2_INDEX = torch.tensor(
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


def _frozen_base_lora_block(block_cls, targets, act_fn):
    _reference, model = _build_pair(block_cls, targets, r=4)
    for name, param in model.named_parameters():
        param.requires_grad_("lora" in name)
    model.experts.get_base_layer().act_fn = act_fn
    return model


def _run_routed_experts(model, recompute, hidden, top_k_index, top_k_weights, **kw):
    """Forward and backward the experts; return output, input grads and LoRA grads."""
    set_routed_experts_recompute(model, recompute)
    model.zero_grad(set_to_none=True)
    rows = hidden.clone().requires_grad_(True)
    weights = top_k_weights.clone().requires_grad_(True)
    out = model.experts(rows, top_k_index, weights, **kw)
    out.square().mean().backward()
    lora_grads = {
        name: param.grad.clone()
        for name, param in model.named_parameters()
        if param.grad is not None
    }
    return out.detach(), rows.grad, weights.grad, lora_grads


def _assert_runs_match(lean, graph):
    lean_out, lean_rows, lean_weights, lean_loras = lean
    graph_out, graph_rows, graph_weights, graph_loras = graph
    assert torch.allclose(lean_out, graph_out, rtol=1e-5, atol=1e-6)
    assert torch.allclose(lean_rows, graph_rows, rtol=1e-5, atol=1e-6)
    assert torch.allclose(lean_weights, graph_weights, rtol=1e-5, atol=1e-6)
    assert set(lean_loras) == set(graph_loras)
    assert lean_loras
    assert all("lora" in name for name in lean_loras)
    for name, grad in graph_loras.items():
        assert torch.allclose(lean_loras[name], grad, rtol=1e-5, atol=1e-6), name


def _saved_activation_bytes(model, run):
    """Bytes of distinct non-parameter storages autograd saves while ``run`` executes."""
    param_storages = {p.untyped_storage().data_ptr() for p in model.parameters()}
    saved: dict[int, int] = {}

    def pack(tensor):
        storage = tensor.untyped_storage()
        if storage.data_ptr() not in param_storages:
            saved[storage.data_ptr()] = storage.nbytes()
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        run()
    return sum(saved.values())


class TestLoraExpertsFunction:
    @pytest.mark.parametrize("use_grouped_mm", [False, True], ids=["loop", "gmm"])
    @pytest.mark.parametrize(
        "top_k_index",
        [
            pytest.param(TOP2_INDEX, id="top2"),
            pytest.param(UNEVEN_TOP2_INDEX, id="uneven_with_empty_expert"),
            pytest.param(torch.full((12, 1), 2), id="expert_spans_chunks"),
        ],
    )
    @pytest.mark.parametrize(("block_cls", "targets", "act_fn"), RECOMPUTE_BLOCKS)
    def test_matches_graph_path(
        self, monkeypatch, block_cls, targets, act_fn, top_k_index, use_grouped_mm
    ):
        # Arrange
        model = _frozen_base_lora_block(block_cls, targets, act_fn)
        hidden = torch.randn(top_k_index.shape[0], HIDDEN)
        top_k_weights = torch.rand(top_k_index.shape)
        set_routed_experts_chunk_bytes(model, SMALL_CHUNK_BYTES)
        if use_grouped_mm:
            monkeypatch.setattr(moe_gemm, "_use_grouped_mm", lambda _x: True)

        # Act
        graph = _run_routed_experts(model, False, hidden, top_k_index, top_k_weights)
        lean = _run_routed_experts(model, True, hidden, top_k_index, top_k_weights)

        # Assert
        _assert_runs_match(lean, graph)

    @pytest.mark.parametrize(("block_cls", "targets", "act_fn"), RECOMPUTE_BLOCKS)
    def test_already_grouped_matches_graph_path(self, block_cls, targets, act_fn):
        # Arrange
        model = _frozen_base_lora_block(block_cls, targets, act_fn)
        expert_ids = torch.tensor([[0], [0], [2], [2], [2], [3], [3], [3], [3]])
        hidden = torch.randn(expert_ids.shape[0], HIDDEN)
        top_k_weights = torch.rand(expert_ids.shape[0], 1)
        set_routed_experts_chunk_bytes(model, SMALL_CHUNK_BYTES)

        # Act
        graph = _run_routed_experts(
            model, False, hidden, expert_ids, top_k_weights, already_grouped=True
        )
        lean = _run_routed_experts(
            model, True, hidden, expert_ids, top_k_weights, already_grouped=True
        )

        # Assert
        _assert_runs_match(lean, graph)

    @pytest.mark.parametrize(("block_cls", "targets", "act_fn"), RECOMPUTE_BLOCKS)
    def test_mixed_routing_matches_graph_path(self, block_cls, targets, act_fn):
        # Arrange
        model = _expert_lora_model(
            block_cls, targets, ("actor", "critic"), trainable=("actor", "critic")
        )
        for name, param in model.named_parameters():
            param.requires_grad_("lora" in name)
        model.experts.get_base_layer().act_fn = act_fn
        patch_lora_for_fused_forward(model)
        set_fused_adapter_routing(model, ["actor"] * 6 + ["critic"] * 6)
        x = torch.randn(12, HIDDEN)
        set_routed_experts_chunk_bytes(model, SMALL_CHUNK_BYTES)

        def run(recompute):
            set_routed_experts_recompute(model, recompute)
            model.zero_grad(set_to_none=True)
            rows = x.clone().requires_grad_(True)
            out = model(rows)
            out.square().sum().backward()
            grads = {
                name: param.grad.clone()
                for name, param in model.named_parameters()
                if param.grad is not None
            }
            return out.detach(), rows.grad, grads

        # Act
        graph_out, graph_rows, graph_grads = run(False)
        lean_out, lean_rows, lean_grads = run(True)
        unset_fused_adapter_routing(model)
        unpatch_lora_for_fused_forward(model)

        # Assert
        assert torch.allclose(lean_out, graph_out, rtol=1e-5, atol=1e-6)
        assert torch.allclose(lean_rows, graph_rows, rtol=1e-5, atol=1e-6)
        assert set(lean_grads) == set(graph_grads)
        assert any("critic" in name for name in lean_grads)
        for name, grad in graph_grads.items():
            assert torch.allclose(lean_grads[name], grad, rtol=1e-5, atol=1e-6), name

    def test_saves_rows_not_intermediate_activations(self):
        # Arrange
        torch.manual_seed(0)
        intermediate, n_tokens = 64, 256
        block = _UngatedMoeBlock()
        block.experts.up_proj = nn.Parameter(
            torch.randn(NUM_EXPERTS, intermediate, HIDDEN) * 0.1
        )
        block.experts.down_proj = nn.Parameter(
            torch.randn(NUM_EXPERTS, HIDDEN, intermediate) * 0.1
        )
        model = inject_adapter_in_model(
            _lora_config(["experts.up_proj", "experts.down_proj"]),
            block,
            adapter_name="actor",
        )
        upgrade_moe_param_wrappers(model)
        for name, param in model.named_parameters():
            param.requires_grad_("lora" in name)
        hidden = torch.randn(n_tokens, HIDDEN, requires_grad=True)
        top_k_index = torch.randint(0, NUM_EXPERTS, (n_tokens, TOP_K))
        top_k_weights = torch.rand(n_tokens, TOP_K, requires_grad=True)
        dispatched_row_bytes = n_tokens * TOP_K * HIDDEN * hidden.element_size()

        def run(recompute):
            set_routed_experts_recompute(model, recompute)
            return lambda: model.experts(hidden, top_k_index, top_k_weights)

        # Act
        graph_bytes = _saved_activation_bytes(model, run(False))
        lean_bytes = _saved_activation_bytes(model, run(True))

        # Assert
        # Lean saves the token rows plus int64 routing indices per expert row.
        assert lean_bytes < 2 * dispatched_row_bytes
        assert graph_bytes > 8 * dispatched_row_bytes
        assert lean_bytes * 5 < graph_bytes

    def test_trainable_base_weights_keep_graph_path(self):
        # Arrange
        _reference, model = _ungated_pair()
        for param in model.experts.get_base_layer().parameters():
            param.requires_grad_(True)
        hidden = torch.randn(12, HIDDEN)
        top_k_weights = torch.rand(12, TOP_K)

        # Act
        out = model.experts(hidden, TOP2_INDEX, top_k_weights)
        out.square().mean().backward()

        # Assert
        assert model.experts.get_base_layer().up_proj.grad is not None
        assert model.experts.get_base_layer().down_proj.grad is not None


class TestSetRoutedExpertsRecompute:
    def test_sets_every_routed_wrapper(self):
        # Arrange
        _reference, model = _routed_pair()
        wrappers = [
            m for m in model.modules() if isinstance(m, RoutedExpertsLoraWrapper)
        ]

        # Act
        set_routed_experts_recompute(model, False)

        # Assert
        assert wrappers
        assert all(wrapper.recompute is False for wrapper in wrappers)

    def test_auto_keeps_the_graph_only_inside_checkpoint_wrappers(self):
        # Arrange
        _reference, checkpointed = _routed_pair()
        _reference, plain = _routed_pair()
        model = nn.ModuleDict(
            {"checkpointed": checkpoint_wrapper(checkpointed), "plain": plain}
        )

        # Act
        set_routed_experts_recompute(model, None)

        # Assert
        assert checkpointed.experts.recompute is False
        assert plain.experts.recompute is True

    @pytest.mark.parametrize("gradient_checkpointing", [True, False])
    def test_auto_follows_the_hf_layer_checkpointing_flag(self, gradient_checkpointing):
        # Arrange
        _reference, block = _routed_pair()
        layer = _HfCheckpointLayer(block)
        layer.gradient_checkpointing = gradient_checkpointing

        # Act
        set_routed_experts_recompute(layer, None)

        # Assert
        assert block.experts.recompute is not gradient_checkpointing

    def test_explicit_true_recomputes_inside_checkpoint_wrappers(self):
        # Arrange
        _reference, block = _routed_pair()
        model = checkpoint_wrapper(block)

        # Act
        set_routed_experts_recompute(model, True)

        # Assert
        assert block.experts.recompute is True

    @pytest.mark.parametrize(("block_cls", "targets", "act_fn"), RECOMPUTE_BLOCKS)
    def test_auto_inside_checkpoint_matches_recompute(self, block_cls, targets, act_fn):
        # Arrange
        torch.manual_seed(0)
        block = _frozen_base_lora_block(block_cls, targets, act_fn)
        model = checkpoint_wrapper(
            block, checkpoint_impl=CheckpointImpl.NO_REENTRANT, preserve_rng_state=False
        )
        hidden = torch.randn(12, HIDDEN)
        set_routed_experts_chunk_bytes(model, SMALL_CHUNK_BYTES)

        def run(enabled):
            set_routed_experts_recompute(model, enabled)
            model.zero_grad(set_to_none=True)
            rows = hidden.clone().requires_grad_(True)
            out = model(rows)
            out.square().mean().backward()
            grads = {
                name: param.grad.clone()
                for name, param in model.named_parameters()
                if param.grad is not None
            }
            return out.detach(), rows.grad, grads, block.experts.recompute

        # Act
        auto_out, auto_rows, auto_grads, auto_recompute = run(None)
        lean_out, lean_rows, lean_grads, lean_recompute = run(True)

        # Assert
        assert (auto_recompute, lean_recompute) == (False, True)
        assert torch.allclose(auto_out, lean_out, rtol=1e-5, atol=1e-6)
        assert torch.allclose(auto_rows, lean_rows, rtol=1e-5, atol=1e-6)
        assert set(auto_grads) == set(lean_grads)
        assert auto_grads
        for name, grad in lean_grads.items():
            assert torch.allclose(auto_grads[name], grad, rtol=1e-5, atol=1e-6), name


class _HfCheckpointLayer(GradientCheckpointingLayer):
    """HF decoder-layer stand-in holding one MoE block."""

    def __init__(self, block: nn.Module) -> None:
        super().__init__()
        self.block = block

    def forward(self, hidden_states):
        return self.block(hidden_states)


class TestRoutedExpertChunkBudget:
    @pytest.mark.parametrize("recompute", [True, False], ids=["lean", "graph"])
    @pytest.mark.parametrize(("block_cls", "targets", "act_fn"), RECOMPUTE_BLOCKS)
    def test_default_budget_matches_one_row_chunks(
        self, block_cls, targets, act_fn, recompute
    ):
        # Arrange
        torch.manual_seed(0)
        model = _frozen_base_lora_block(block_cls, targets, act_fn)
        hidden = torch.randn(UNEVEN_TOP2_INDEX.shape[0], HIDDEN)
        top_k_weights = torch.rand(UNEVEN_TOP2_INDEX.shape)

        # Act
        default = _run_routed_experts(
            model, recompute, hidden, UNEVEN_TOP2_INDEX, top_k_weights
        )
        set_routed_experts_chunk_bytes(model, 1)
        one_row = _run_routed_experts(
            model, recompute, hidden, UNEVEN_TOP2_INDEX, top_k_weights
        )

        # Assert
        _assert_runs_match(default, one_row)

    def test_sets_sorted_wrapper_chunk(self):
        _, upgraded = _sorted_pair()

        set_routed_experts_chunk_bytes(upgraded, 1)

        sorted_wrappers = [
            m for m in _wrappers(upgraded) if isinstance(m, SortedExpertsLoraWrapper)
        ]
        assert sorted_wrappers
        assert {m.chunk_bytes for m in sorted_wrappers} == {1}


class TestScatterRows:
    def test_adds_source_rows_at_index(self):
        base = torch.zeros(3, 2)
        source = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        index = torch.tensor([2, 0, 2])

        out = moe_routed.ScatterRows.apply(base, index, source)

        assert torch.equal(out, torch.tensor([[3.0, 4.0], [0.0, 0.0], [6.0, 8.0]]))

    def test_gradcheck(self):
        torch.manual_seed(0)
        base = torch.randn(4, 3, dtype=torch.double, requires_grad=True)
        source = torch.randn(5, 3, dtype=torch.double, requires_grad=True)
        index = torch.tensor([2, 0, 2, 3, 0])

        assert torch.autograd.gradcheck(
            lambda b, s: moe_routed.ScatterRows.apply(b.clone(), index, s),
            (base, source),
        )


class TestGroupedGemmCoverage:
    def test_probe_returns_false_when_the_op_disagrees_on_values(self, monkeypatch):
        def wrong(x, w_t, offs):
            return torch.zeros_like(x)

        _mock_cuda_torch(monkeypatch, wrong)
        moe_gemm.grouped_mm_supported.cache_clear()

        assert moe_gemm.grouped_mm_supported(6, torch.float32) is False

    def test_operand_ready_rejects_a_matrix_that_is_not_3d(self):
        assert moe_gemm._grouped_mm_operand_ready(torch.ones(4, 4)) is False

    def test_operand_ready_rejects_a_misaligned_cuda_operand(self):
        mat = MagicMock()
        mat.dim.return_value = 3
        mat.element_size.return_value = 2
        mat.is_cuda = True
        mat.data_ptr.return_value = 1
        mat.stride.return_value = 1

        assert moe_gemm._grouped_mm_operand_ready(mat) is False

    def test_grouped_linear_returns_empty_when_every_expert_is_empty(self):
        x = torch.zeros(0, 5)
        weight = torch.randn(2, 4, 5)

        out = moe_gemm.grouped_linear(x, weight, [0, 0])

        assert out.shape == (0, 4)

    def test_grouped_matmul_uses_grouped_mm_when_the_operand_is_ready(
        self, monkeypatch
    ):
        x = torch.randn(4, 8)
        weight = torch.randn(2, 8, 4)
        offs = torch.tensor([2, 4], dtype=torch.int32)
        monkeypatch.setattr(moe_gemm, "_use_grouped_mm", lambda _x: True)
        monkeypatch.setattr(moe_gemm, "_grouped_mm_operand_ready", lambda _w: True)
        monkeypatch.setattr(
            torch, "_grouped_mm", lambda rows, w, offs: torch.ones(4, 4)
        )

        out = moe_gemm.grouped_matmul(x, weight, offs)

        assert torch.equal(out, torch.ones(4, 4))

    def test_grouped_matmul_returns_empty_when_every_expert_is_empty(self):
        x = torch.zeros(0, 4)
        weight = torch.randn(2, 4, 5)

        out = moe_gemm.grouped_matmul(
            x, weight, torch.tensor([0, 0], dtype=torch.int32)
        )

        assert out.shape == (0, 5)

    def test_iter_expert_row_chunks_stops_when_no_rows_are_taken(self):
        assert list(moe_gemm.iter_expert_row_chunks([2, 3], max_rows=0)) == []

    def test_add_grouped_linear_returns_when_x_is_empty(self):
        destination = torch.ones(0, 4)
        moe_gemm.add_grouped_linear(
            destination,
            torch.zeros(0, 5),
            torch.randn(1, 4, 5),
            [0],
            1.0,
            moe_gemm.ROUTED_EXPERT_CHUNK_BYTES,
        )

        assert destination.shape == (0, 4)

    def test_add_grouped_linear_casts_a_chunk_to_the_destination_dtype(self):
        destination = torch.zeros(2, 4, dtype=torch.float16)
        x = torch.ones(2, 5)
        weight = torch.ones(1, 4, 5)

        moe_gemm.add_grouped_linear(
            destination, x, weight, [2], 1.0, moe_gemm.ROUTED_EXPERT_CHUNK_BYTES
        )

        assert destination.dtype == torch.float16
        assert torch.allclose(destination.float(), torch.full((2, 4), 5.0))


def test_split_lora_delta_dtensor_fallback_adds_into_a_narrower_destination(
    monkeypatch,
):
    class FakeDTensor(nn.Parameter):
        pass

    experts, rank, total = 2, 2, 4
    x = torch.randn(total, 8)
    lora_a = MagicMock()
    lora_a.weight = FakeDTensor(torch.ones(experts, rank, 8))
    lora_a.return_value = (
        torch.arange(total * experts * rank).reshape(total, -1).float()
    )
    lora_b = MagicMock()
    lora_b.weight = FakeDTensor(torch.ones(6, experts * rank))
    lora_b.side_effect = lambda t: t @ torch.ones(t.shape[1], 6)
    wrapper = MagicMock()
    wrapper.lora_A = {"actor": lora_a}
    wrapper.lora_B = {"actor": lora_b}
    wrapper.scaling = {"actor": 2.0}
    wrapper.r = {"actor": rank}
    monkeypatch.setattr(moe_adapters, "DTensor", FakeDTensor)
    destination = torch.zeros(total, 6, dtype=torch.float16)

    out = moe_adapters.split_lora_delta(
        wrapper,
        x,
        [2, 2],
        "actor",
        num_experts=experts,
        destination=destination,
    )

    assert out is destination
    assert out.dtype == torch.float16


class TestSortedExpertsLoraWrapperDtype:
    def test_casts_a_mixed_routing_delta_to_the_result_dtype(self):
        _reference, upgraded = _sorted_pair()
        wrapper = upgraded.input_linear
        x = torch.randn(4, HIDDEN)
        expert_size = [1, 1, 1, 1]
        delta = torch.ones(4, 2 * INTERMEDIATE)
        ids = torch.tensor([0, 1, 0, 1])
        wrapper.token_index = torch.arange(4)
        wrapper.n_tokens = 4
        with (
            patch(
                "agilerl.lora.moe.wrappers.mixed_routing",
                return_value=["actor", "critic", "actor", "critic"],
            ),
            patch(
                "agilerl.lora.moe.wrappers.adapters_in_routing",
                return_value=["actor"],
            ),
            patch(
                "agilerl.lora.moe.wrappers.token_adapter_ids",
                return_value=(ids, {"actor": 0, "critic": 1}),
            ),
            patch.object(
                wrapper.get_base_layer(),
                "forward",
                return_value=torch.zeros(4, 2 * INTERMEDIATE, dtype=torch.float16),
            ),
            patch(
                "agilerl.lora.moe.wrappers.split_lora_delta",
                return_value=delta,
            ),
        ):
            out = wrapper(x, expert_size)

        assert out.dtype == torch.float16


class TestRoutedExpertDtypeAndDTensor:
    def test_add_expert_loras_casts_delta_to_the_destination_dtype(self):
        wrapper = MagicMock()
        wrapper.scaling = {"actor": 1.0}
        lora = moe_adapters.ExpertLora(
            wrapper=wrapper,
            adapter="actor",
            stacked=(torch.ones(1, 2, 4), torch.ones(1, 3, 2)),
            row_id=None,
        )
        destination = torch.zeros(2, 3, dtype=torch.float16)
        rows = torch.ones(2, 4)

        chunked = moe_routed.ChunkedLora(
            lora, moe_adapters.lora_operands(*lora.stacked, 1.0), None, None
        )

        moe_routed._add_expert_loras(
            destination, rows, [chunked], 0, torch.tensor([2], dtype=torch.int32)
        )

        assert destination.dtype == torch.float16
        assert destination.abs().sum() > 0

    def test_scatter_chunks_cast_down_rows_to_the_result_dtype(self, monkeypatch):
        hidden = torch.zeros(2, 4, dtype=torch.float16)
        rows = torch.ones(2, 4)
        up_weight = torch.ones(1, 6, 4)
        down_weight = torch.ones(1, 4, 6)
        routing = moe_routed.RoutedRows(
            counts=torch.tensor([2]),
            token_idx=torch.tensor([0, 1]),
            routed_weights=torch.ones(2, 1),
            row_ids=None,
        )

        def fake_matmul(x, weight, offs):
            return torch.ones(x.shape[0], weight.shape[2])

        monkeypatch.setattr(moe_routed, "grouped_matmul", fake_matmul)

        out = moe_routed._scatter_routed_expert_chunks(
            hidden,
            rows,
            up_weight,
            down_weight,
            act_fn=torch.nn.functional.silu,
            gated=False,
            routing=routing,
            up_loras=(),
            down_loras=(),
            chunk_bytes=moe_routed.ROUTED_EXPERT_CHUNK_BYTES,
        )

        assert out.dtype == hidden.dtype

    def test_recompute_unwraps_dtensor_weights(self, monkeypatch):
        class FakeDTensor(nn.Parameter):
            def to_local(self):
                return torch.ones(1, 4, 4)

        routing = moe_routed.RoutedRows(
            counts=torch.tensor([1]),
            token_idx=torch.tensor([0]),
            routed_weights=torch.ones(1, 1),
            row_ids=None,
        )
        captured = {}

        def fake_apply(*args, **_kwargs):
            captured["up"] = args[5]
            captured["down"] = args[6]
            return torch.zeros(1, 4)

        monkeypatch.setattr(moe_routed, "DTensor", FakeDTensor)
        monkeypatch.setattr(moe_routed.LoraExpertsFunction, "apply", fake_apply)

        out = moe_routed._recompute_routed_experts(
            torch.zeros(1, 4),
            FakeDTensor(torch.ones(1, 4, 4)),
            FakeDTensor(torch.ones(1, 4, 4)),
            act_fn=lambda t: t,
            gated=False,
            routing=routing,
            up_loras=(),
            down_loras=(),
            chunk_bytes=moe_routed.ROUTED_EXPERT_CHUNK_BYTES,
        )

        assert out.shape == (1, 4)
        assert not isinstance(captured["up"], FakeDTensor)
        assert not isinstance(captured["down"], FakeDTensor)


class TestLoraExpertsRecomputeGuards:
    def test_skips_a_chunk_when_nothing_requires_grad(self, monkeypatch):
        hidden = torch.ones(1, 4, requires_grad=True)
        token_idx = torch.tensor([0])
        weights = torch.ones(1, 1)
        offsets = torch.tensor([[1]], dtype=torch.int32)
        up_weight = torch.ones(1, 4, 4)
        down_weight = torch.ones(1, 4, 4)
        config = LoraExpertsConfig(
            act_fn=lambda t: t,
            gated=False,
            sizes=(1,),
            up_row_ids=(),
            down_row_ids=(),
        )
        monkeypatch.setattr(
            "agilerl.lora.moe.recompute.grouped_matmul",
            lambda *_a, **_k: torch.ones(1, 4),
        )

        out = LoraExpertsFunction.apply(
            hidden,
            token_idx,
            weights,
            offsets,
            None,
            up_weight,
            down_weight,
            config,
        )
        out.sum().backward()

        assert torch.equal(hidden.grad, torch.zeros_like(hidden))

    def test_skips_an_unused_leaf_grad(self, monkeypatch):
        hidden = torch.ones(1, 4, requires_grad=True)
        token_idx = torch.tensor([0])
        weights = torch.ones(1, 1)
        offsets = torch.tensor([[1]], dtype=torch.int32)
        up_weight = torch.ones(1, 4, 4)
        down_weight = torch.ones(1, 4, 4)
        config = LoraExpertsConfig(
            act_fn=lambda t: t,
            gated=False,
            sizes=(1,),
            up_row_ids=(),
            down_row_ids=(),
        )

        def _grad(outputs, inputs, grad_outputs, allow_unused=False):
            return tuple(None for _ in inputs)

        monkeypatch.setattr(
            "agilerl.lora.moe.recompute.grouped_matmul",
            lambda *_a, **_k: torch.ones(1, 4, requires_grad=True),
        )
        monkeypatch.setattr(torch.autograd, "grad", _grad)
        out = LoraExpertsFunction.apply(
            hidden,
            token_idx,
            weights,
            offsets,
            None,
            up_weight,
            down_weight,
            config,
        )
        out.sum().backward()

        assert torch.equal(hidden.grad, torch.zeros_like(hidden))


class TestLowRankDelta:
    @pytest.mark.parametrize("use_grouped_mm", [False, True], ids=["loop", "gmm"])
    def test_matches_per_expert_reference(self, monkeypatch, use_grouped_mm):
        # Arrange
        torch.manual_seed(0)
        counts = [3, 0, 2, 3]
        scaling = 0.3
        rows = torch.randn(sum(counts), HIDDEN, requires_grad=True)
        lora_a = torch.randn(NUM_EXPERTS, 4, HIDDEN, requires_grad=True)
        lora_b = torch.randn(NUM_EXPERTS, INTERMEDIATE, 4, requires_grad=True)
        offs = moe_gemm.group_offsets(counts, rows.device)
        if use_grouped_mm:
            monkeypatch.setattr(moe_gemm, "_use_grouped_mm", lambda _x: True)
        expected = torch.cat(
            [
                chunk @ lora_a[expert].T @ lora_b[expert].T * scaling
                for expert, chunk in enumerate(rows.split(counts))
            ]
        )
        expected_grads = torch.autograd.grad(
            expected.square().sum(), (rows, lora_a, lora_b)
        )

        # Act
        operands = moe_adapters.lora_operands(lora_a, lora_b, scaling)
        delta = moe_adapters.low_rank_delta(rows, *operands, offs)
        grads = torch.autograd.grad(delta.square().sum(), (rows, lora_a, lora_b))

        # Assert
        torch.testing.assert_close(delta, expected, rtol=1e-5, atol=1e-5)
        for grad, expected_grad in zip(grads, expected_grads, strict=True):
            torch.testing.assert_close(grad, expected_grad, rtol=1e-5, atol=1e-5)

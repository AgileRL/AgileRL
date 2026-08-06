# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

import copy

import pytest
import torch
import torch.nn.functional as F
from peft import LoraConfig, inject_adapter_in_model
from peft.tuners.lora.layer import ParamWrapper
from torch import nn

from agilerl.algorithms.core.llm_ops.fused_lora import (
    adapter_aligned_chunks,
    patch_lora_for_fused_forward,
    set_fused_adapter_routing,
    unpatch_lora_for_fused_forward,
    unset_fused_adapter_routing,
)
from agilerl.algorithms.core.llm_ops.moe_lora import (
    RoutedExpertsLoraWrapper,
    SortedExpertsLoraWrapper,
    moe_expert_target_parameters,
    upgrade_moe_param_wrappers,
)
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


class _SortedMoeBlock(nn.Module):
    """Routing parent for :class:`_SortedExperts`, mirroring GraniteMoeHybridMoE."""

    def __init__(self) -> None:
        super().__init__()
        self.router = nn.Linear(HIDDEN, NUM_EXPERTS, bias=False)
        self.input_linear = _SortedExperts(HIDDEN, 2 * INTERMEDIATE)
        self.output_linear = _SortedExperts(INTERMEDIATE, HIDDEN)

    def forward(self, hidden_states):
        logits = self.router(hidden_states)
        top_k_logits, top_k_indices = logits.topk(TOP_K, dim=1)
        gates = torch.softmax(top_k_logits, dim=1)
        flat_experts = top_k_indices.flatten()
        expert_size = (
            torch.bincount(flat_experts, minlength=NUM_EXPERTS).long().tolist()
        )
        order = flat_experts.argsort(stable=True)
        batch_index = order.div(TOP_K, rounding_mode="trunc")
        expert_inputs = hidden_states[batch_index]
        inner = self.input_linear(expert_inputs, expert_size)
        gate, up = inner.chunk(2, dim=-1)
        expert_outputs = self.output_linear(F.silu(gate) * up, expert_size)
        expert_outputs = expert_outputs * gates.flatten()[order, None]
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


def _build_pair(block_cls, target_parameters):
    """A (reference, upgraded) pair of adapter-injected blocks with identical weights."""
    torch.manual_seed(0)
    reference = inject_adapter_in_model(
        _lora_config(target_parameters), block_cls(), adapter_name="actor"
    )
    torch.manual_seed(0)
    upgraded = inject_adapter_in_model(
        _lora_config(target_parameters), block_cls(), adapter_name="actor"
    )
    assert upgrade_moe_param_wrappers(upgraded) > 0
    return reference, upgraded


def _sorted_pair():
    return _build_pair(_SortedMoeBlock, ["input_linear.weight", "output_linear.weight"])


def _routed_pair():
    return _build_pair(_RoutedMoeBlock, ["experts.gate_up_proj", "experts.down_proj"])


def _ungated_pair():
    return _build_pair(_UngatedMoeBlock, ["experts.up_proj", "experts.down_proj"])


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


@pytest.mark.parametrize("pair_factory", [_sorted_pair, _routed_pair, _ungated_pair])
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


def test_zero3_partitioned_adapters_use_module_call_path():
    reference, upgraded = _sorted_pair()
    for module in _wrappers(upgraded):
        for adapter in module.lora_A:
            module.lora_A[adapter].weight.ds_id = 1
            module.lora_B[adapter].weight.ds_id = 1
    x = torch.randn(10, HIDDEN)
    assert torch.allclose(reference(x), upgraded(x), atol=1e-5)


def test_zero3_partitioned_base_weights_raise_on_routed_convention():
    _, upgraded = _routed_pair()
    experts = _wrappers(upgraded)[0].get_base_layer()
    experts.gate_up_proj.ds_id = 1
    with pytest.raises(RuntimeError, match="ZeRO-3 partitioned expert weights"):
        upgraded(torch.randn(10, HIDDEN))


@pytest.mark.parametrize("pair_factory", [_sorted_pair, _routed_pair, _ungated_pair])
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


def test_fused_mixed_routing_raises():
    _, upgraded = _sorted_pair()
    patch_lora_for_fused_forward(upgraded)
    set_fused_adapter_routing(upgraded, ["actor"] * 6 + ["__base__"] * 6)
    with pytest.raises(RuntimeError, match="Fused multi-adapter routing"):
        upgraded(torch.randn(12, HIDDEN))
    unset_fused_adapter_routing(upgraded)
    unpatch_lora_for_fused_forward(upgraded)


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


def test_adapter_aligned_chunks():
    routing = ["ref"] * 5 + ["actor"] * 5
    assert adapter_aligned_chunks(routing, 4) == [(0, 4), (4, 5), (5, 9), (9, 10)]
    assert adapter_aligned_chunks(routing, 5) == [(0, 5), (5, 10)]
    assert adapter_aligned_chunks(["actor"] * 4, 8) == [(0, 4)]


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


def test_expert_lora_vllm_key_map_and_filter():
    _, sorted_block = _sorted_pair()
    key_map = expert_lora_vllm_key_map(sorted_block)
    assert key_map == {
        "input_linear": "experts.base_layer",
        "output_linear": "experts",
    }, key_map

    _, routed_block = _routed_pair()
    key_map = expert_lora_vllm_key_map(routed_block)
    # Whatever the nesting order, gate_up lands on <experts>.base_layer and
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
    from agilerl.algorithms.core.llm_ops import moe_lora

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
    if not moe_lora._use_grouped_mm(x):
        pytest.skip("torch._grouped_mm unsupported on this GPU")
    with pytest.MonkeyPatch.context() as mp:
        # Force the loop path on the reference copy.
        mp.setattr(moe_lora, "_use_grouped_mm", lambda _x: False)
        ref_out = reference(x)
    assert torch.allclose(ref_out, fast_out, atol=1e-5)

    ref_out.square().mean().backward()
    fast_out.square().mean().backward()
    _assert_grad_parity(reference, fast, atol=1e-4)


class _FakeDsStatus:
    def __init__(self, name: str) -> None:
        self.name = name


def _mark_ds(param, status=None):
    param.ds_id = 1
    if status is not None:
        param.ds_status = _FakeDsStatus(status)


def test_zero3_gathered_base_weights_run_split_forward():
    reference, upgraded = _routed_pair()
    experts = _wrappers(upgraded)[0].get_base_layer()
    # Leaf-gathered ZeRO-3 params report AVAILABLE; raw reads see full data.
    _mark_ds(experts.gate_up_proj, "AVAILABLE")
    _mark_ds(experts.down_proj, "AVAILABLE")
    x = torch.randn(10, HIDDEN)
    assert torch.allclose(reference(x), upgraded(x), atol=1e-5)


def test_zero3_gathered_adapters_use_fast_path():
    reference, upgraded = _sorted_pair()
    for module in _wrappers(upgraded):
        for adapter in module.lora_A:
            _mark_ds(module.lora_A[adapter].weight, "AVAILABLE")
            _mark_ds(module.lora_B[adapter].weight, "AVAILABLE")
    x = torch.randn(10, HIDDEN)
    assert torch.allclose(reference(x), upgraded(x), atol=1e-5)

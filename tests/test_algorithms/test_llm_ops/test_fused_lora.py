# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

import copy

import pytest
import torch
import torch.nn.functional as F
from peft import LoraConfig, inject_adapter_in_model
from torch import nn
from torch.profiler import ProfilerActivity, profile

from agilerl.algorithms.grpo import GRPO
from agilerl.lora.fused import (
    LORA_LAYER_CACHE,
    ROUTING_STATE,
    _is_routed_layer,
    _param_wrapper_routed_forward,
    get_cached_lora_layers,
    patch_lora_for_fused_forward,
    set_fused_adapter_routing,
    uniform_routed_adapter,
    unpatch_lora_for_fused_forward,
    unset_fused_adapter_routing,
)
from tests.test_algorithms.test_llms.llm_helpers import create_module


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(8, 6, bias=False)

    def forward(self, x):
        return self.proj(x)


class _TinyEmbedding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(10, 4)

    def forward(self, ids):
        return self.embed(ids)


class _TinyConv(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, 4, kernel_size=1, bias=False)

    def forward(self, x):
        return self.proj(x)


class _TinyFlatten(nn.Module):
    """Flattens (batch, seq, hidden) to (batch * seq, hidden) before its
    linear, like OPT's MLP and MoE experts.
    """

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(8, 6, bias=False)

    def forward(self, x):
        batch, seq, hidden = x.shape
        return self.proj(x.reshape(batch * seq, hidden)).reshape(batch, seq, -1)


def _lora_config(**overrides):
    config = {
        "r": 2,
        "lora_alpha": 4,
        "target_modules": ["proj"],
        # Random lora_B (not zeros) so every adapter produces a distinct delta.
        "init_lora_weights": False,
    }
    config.update(overrides)
    return LoraConfig(**config)


def _build_model(adapters=("actor", "critic"), module_cls=_Tiny, **config_overrides):
    torch.manual_seed(0)
    model = module_cls()
    for name in adapters:
        model = inject_adapter_in_model(
            _lora_config(**config_overrides), model, adapter_name=name
        )
    return model


def _enable_lora_grads(model):
    # set_adapter freezes inactive adapters; training keeps all routed
    # adapters trainable, so the gradient tests do too.
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad_(True)


class _ViewMatmul(torch.autograd.Function):
    """Matmul returning a view of its output, like bitsandbytes' quantized
    matmul: autograd forbids in-place modification of the result.
    """

    @staticmethod
    def forward(ctx, x, weight):
        ctx.save_for_backward(weight)
        out = x @ weight.t()
        return out.view(out.shape)

    @staticmethod
    def backward(ctx, grad_out):
        (weight,) = ctx.saved_tensors
        return grad_out @ weight, None


class _ViewLinear(nn.Module):
    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        self.weight = linear.weight

    def forward(self, x):
        return _ViewMatmul.apply(x, self.weight)


class TestRoutedForward:
    def test_rows_match_per_adapter_reference(self):
        model = _build_model()
        x = torch.randn(4, 8)
        model.proj.set_adapter("actor")
        ref_actor = model(x)
        model.proj.set_adapter("critic")
        ref_critic = model(x)
        assert not torch.allclose(ref_actor, ref_critic)

        patch_lora_for_fused_forward(model)
        set_fused_adapter_routing(model, ["actor"] * 2 + ["critic"] * 2)
        out = model(x)

        assert torch.allclose(out[:2], ref_actor[:2], atol=1e-6)
        assert torch.allclose(out[2:], ref_critic[2:], atol=1e-6)

    def test_base_rows_and_non_contiguous_runs(self):
        model = _build_model()
        x = torch.randn(4, 8)
        model.proj.set_adapter("actor")
        ref_actor = model(x)
        ref_base = model.proj.base_layer(x)

        patch_lora_for_fused_forward(model)
        set_fused_adapter_routing(model, ["actor", "__base__", "__base__", "actor"])
        out = model(x)

        assert torch.allclose(out[0], ref_actor[0], atol=1e-6)
        assert torch.allclose(out[1:3], ref_base[1:3], atol=1e-6)
        assert torch.allclose(out[3], ref_actor[3], atol=1e-6)

    def test_single_group_covers_whole_batch(self):
        model = _build_model()
        x = torch.randn(4, 8)
        model.proj.set_adapter("critic")
        ref_critic = model(x)
        ref_base = model.proj.base_layer(x)

        patch_lora_for_fused_forward(model)
        set_fused_adapter_routing(model, ["critic"] * 4)
        assert torch.allclose(model(x), ref_critic, atol=1e-6)
        set_fused_adapter_routing(model, ["__base__"] * 4)
        assert torch.allclose(model(x), ref_base, atol=1e-6)

    def test_adapter_missing_on_a_layer_leaves_rows_on_base_output(self):
        # "critic" only wraps a different layer, so on ``proj`` its rows get
        # the frozen base output — same as PEFT's mixed-batch behaviour.
        model = _build_model(adapters=("actor",))
        model.other = inject_adapter_in_model(
            _lora_config(), _Tiny(), adapter_name="critic"
        )
        x = torch.randn(2, 8)
        ref_base = model.proj.base_layer(x)

        patch_lora_for_fused_forward(model)
        set_fused_adapter_routing(model, ["actor", "critic"])
        out = model(x)
        assert torch.allclose(out[1], ref_base[1], atol=1e-6)

    def test_routing_and_batch_size_mismatch_raises(self):
        model = _build_model()
        patch_lora_for_fused_forward(model)
        set_fused_adapter_routing(model, ["actor", "actor", "critic"])
        with pytest.raises(ValueError, match="covers 3 rows"):
            model(torch.randn(4, 8))

    def test_equal_runs_split_image_rows_evenly(self):
        # Arrange: two samples with three images between them, repeated for
        # actor then critic, as a vision tower sees the doubled batch.
        model = _build_model()
        x = torch.randn(6, 8)
        model.proj.set_adapter("actor")
        ref_actor = model(x)
        model.proj.set_adapter("critic")
        ref_critic = model(x)
        patch_lora_for_fused_forward(model)
        set_fused_adapter_routing(model, ["actor"] * 2 + ["critic"] * 2)

        # Act
        out = model(x)

        # Assert
        assert torch.allclose(out[:3], ref_actor[:3], atol=1e-6)
        assert torch.allclose(out[3:], ref_critic[3:], atol=1e-6)

    def test_single_run_covers_any_number_of_image_rows(self):
        # Arrange
        model = _build_model()
        x = torch.randn(3, 8)
        model.proj.set_adapter("critic")
        ref_critic = model(x)
        patch_lora_for_fused_forward(model)
        set_fused_adapter_routing(model, ["critic"] * 2)

        # Act
        out = model(x)

        # Assert
        assert torch.allclose(out, ref_critic, atol=1e-6)

    def test_layer_that_flattens_batch_and_seq_scales_the_routing(self):
        model = _build_model(module_cls=_TinyFlatten)
        x = torch.randn(2, 3, 8)
        model.proj.set_adapter("actor")
        ref_actor = model(x)
        model.proj.set_adapter("critic")
        ref_critic = model(x)

        patch_lora_for_fused_forward(model)
        set_fused_adapter_routing(model, ["actor", "critic"])
        out = model(x)

        assert torch.allclose(out[0], ref_actor[0], atol=1e-6)
        assert torch.allclose(out[1], ref_critic[1], atol=1e-6)

    def test_conv_adapter_rows_match_per_adapter_reference(self):
        # Arrange
        model = _build_model(module_cls=_TinyConv)
        x = torch.randn(4, 3, 2, 2)
        model.proj.set_adapter("actor")
        ref_actor = model(x)
        model.proj.set_adapter("critic")
        ref_critic = model(x)
        patch_lora_for_fused_forward(model)
        set_fused_adapter_routing(model, ["actor"] * 2 + ["critic"] * 2)

        # Act
        out = model(x)

        # Assert
        assert not torch.allclose(ref_actor, ref_critic)
        assert torch.allclose(out[:2], ref_actor[:2], atol=1e-6)
        assert torch.allclose(out[2:], ref_critic[2:], atol=1e-6)

    def test_inactive_routing_runs_standard_forward(self):
        model = _build_model()
        x = torch.randn(4, 8)
        model.proj.set_adapter("actor")
        ref = model(x)

        patch_lora_for_fused_forward(model)
        assert torch.allclose(model(x), ref, atol=1e-6)

        set_fused_adapter_routing(model, ["critic"] * 4)
        unset_fused_adapter_routing(model)
        assert torch.allclose(model(x), ref, atol=1e-6)


class TestGradients:
    def test_fused_gradients_match_separate_per_adapter_passes(self):
        model = _build_model()
        x = torch.randn(4, 8)
        _enable_lora_grads(model)

        patch_lora_for_fused_forward(model)
        set_fused_adapter_routing(model, ["actor"] * 2 + ["critic"] * 2)
        model(x).sum().backward()
        layer = model.proj
        fused_grads = {
            name: layer.lora_A[name].weight.grad.clone() for name in ("actor", "critic")
        }
        model.zero_grad()
        unset_fused_adapter_routing(model)

        for name, rows in (("actor", x[:2]), ("critic", x[2:])):
            layer.set_adapter(name)
            _enable_lora_grads(model)
            model(rows).sum().backward()
            assert torch.allclose(
                layer.lora_A[name].weight.grad, fused_grads[name], atol=1e-6
            )
            model.zero_grad()

    def test_routing_survives_gradient_checkpoint_recompute(self):
        model = _build_model()
        x = torch.randn(4, 8, requires_grad=True)
        model.proj.set_adapter("actor")
        _enable_lora_grads(model)
        ref_actor = model(x)

        patch_lora_for_fused_forward(model)
        set_fused_adapter_routing(model, ["actor"] * 2 + ["critic"] * 2)
        out = torch.utils.checkpoint.checkpoint(model, x, use_reentrant=False)
        assert torch.allclose(out[:2], ref_actor[:2], atol=1e-6)
        out.sum().backward()
        assert model.proj.lora_A["critic"].weight.grad is not None

    def test_no_in_place_mutation_of_base_output(self):
        # bitsandbytes' quantized matmul returns a view that autograd forbids
        # editing in place; the fused forward must compose out of place.
        model = _build_model()
        model.proj.base_layer = _ViewLinear(model.proj.base_layer)
        x = torch.randn(4, 8, requires_grad=True)
        _enable_lora_grads(model)

        probe = model.proj.base_layer(x)
        with pytest.raises(RuntimeError, match="view"):
            probe += 1.0

        patch_lora_for_fused_forward(model)
        set_fused_adapter_routing(model, ["actor"] * 2 + ["critic"] * 2)
        model(x).sum().backward()


def _reference_lora(layer, adapter, x, lora_input=None):
    """PEFT's ``base(x) + lora_B(lora_A(dropout(x))) * scaling``, spelled out."""
    lora_input = x if lora_input is None else lora_input
    delta = layer.lora_B[adapter](layer.lora_A[adapter](lora_input))
    return layer.base_layer(x) + delta * layer.scaling[adapter]


def _lora_grads(layer, adapter, x):
    return (
        x.grad.clone(),
        layer.lora_A[adapter].weight.grad.clone(),
        layer.lora_B[adapter].weight.grad.clone(),
    )


def _single_adapter_layer(dtype=torch.float32, **config_overrides):
    # r=4, alpha=12: scaling 3, so a missing or doubled scale shows up.
    model = _build_model(adapters=("actor",), r=4, lora_alpha=12, **config_overrides)
    model.to(dtype)
    model.proj.base_layer.weight.requires_grad_(False)
    patch_lora_for_fused_forward(model)
    set_fused_adapter_routing(model, ["actor"] * 2)
    return model


class TestRoutedForwardLoraAccumulation:
    def test_fp32_output_and_grads_match_peft_formula(self):
        # Arrange
        model = _single_adapter_layer()
        layer = model.proj
        torch.manual_seed(1)
        x = torch.randn(2, 5, 8, requires_grad=True)
        x_ref = x.detach().clone().requires_grad_(True)

        # Act
        out = model(x)
        out.square().sum().backward()
        fused_grads = _lora_grads(layer, "actor", x)
        model.zero_grad()
        ref = _reference_lora(layer, "actor", x_ref)
        ref.square().sum().backward()
        ref_grads = _lora_grads(layer, "actor", x_ref)

        # Assert
        assert layer.scaling["actor"] == 3.0
        assert out.shape == (2, 5, 6)
        torch.testing.assert_close(out, ref, rtol=0, atol=1e-6)
        # A few fp32 ulps of summation order on gradients of magnitude ~100.
        for fused, expected in zip(fused_grads, ref_grads, strict=True):
            torch.testing.assert_close(fused, expected, rtol=1e-6, atol=1e-5)

    def test_forward_runs_one_addmm_without_scale_or_residual_kernels(self):
        # Arrange
        model = _single_adapter_layer()
        x = torch.randn(2, 5, 8)

        # Act
        with torch.no_grad(), profile(activities=[ProfilerActivity.CPU]) as prof:
            model(x)

        # Assert
        ops = {event.key: event.count for event in prof.key_averages()}
        assert ops.get("aten::addmm_") == 1
        assert "aten::mul" not in ops
        assert "aten::add" not in ops

    def test_single_run_backward_has_no_activation_sized_copies(self):
        # Arrange
        # Dense upstream gradient: ``.sum()`` would pass an expanded one that
        # mm backward must clone regardless of the LoRA path.
        model = _single_adapter_layer()
        x = torch.randn(2, 5, 8, requires_grad=True)
        grad_out = torch.randn(2, 5, 6)

        # Act
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            model(x).backward(grad_out)

        # Assert: only scalar copies; no slice zero-fill, gradient clone or cat.
        events = prof.key_averages(group_by_input_shape=True)
        ops = {event.key for event in events}
        assert not ops & {"aten::zero_", "aten::zeros", "aten::clone", "aten::cat"}
        copy_shapes = [e.input_shapes for e in events if e.key == "aten::copy_"]
        assert all(shape[0] == [] for shape in copy_shapes)

    def test_in_place_output_feeds_ops_that_save_it(self):
        # Arrange: tanh saves its output and mul saves both inputs, so a
        # stale version counter or a wrong value would surface in backward.
        model = _single_adapter_layer()
        layer = model.proj
        x = torch.randn(2, 5, 8, requires_grad=True)
        x_ref = x.detach().clone().requires_grad_(True)

        # Act
        out = model(x)
        (torch.tanh(out) * out).sum().backward()
        fused_grads = _lora_grads(layer, "actor", x)
        model.zero_grad()
        ref = _reference_lora(layer, "actor", x_ref)
        (torch.tanh(ref) * ref).sum().backward()
        ref_grads = _lora_grads(layer, "actor", x_ref)

        # Assert
        for fused, expected in zip(fused_grads, ref_grads, strict=True):
            torch.testing.assert_close(fused, expected, rtol=0, atol=1e-5)

    def test_in_place_with_frozen_input_trains_adapters(self):
        # Arrange: neither the input nor the base weight needs grad, so the
        # base output is a plain tensor the in-place GEMM attaches to the graph.
        model = _single_adapter_layer()
        layer = model.proj
        x = torch.randn(2, 5, 8)

        # Act
        model(x).sum().backward()
        a_grad = layer.lora_A["actor"].weight.grad.clone()
        b_grad = layer.lora_B["actor"].weight.grad.clone()
        model.zero_grad()
        _reference_lora(layer, "actor", x).sum().backward()

        # Assert
        torch.testing.assert_close(
            a_grad, layer.lora_A["actor"].weight.grad, rtol=0, atol=1e-5
        )
        torch.testing.assert_close(
            b_grad, layer.lora_B["actor"].weight.grad, rtol=0, atol=1e-5
        )

    def test_in_place_with_biased_base_under_checkpoint(self):
        # Arrange
        torch.manual_seed(0)
        model = inject_adapter_in_model(
            _lora_config(r=4, lora_alpha=12, target_modules=["0"]),
            nn.Sequential(nn.Linear(8, 6)),
            adapter_name="actor",
        )
        layer = model[0]
        x = torch.randn(2, 5, 8, requires_grad=True)
        x_ref = x.detach().clone().requires_grad_(True)
        patch_lora_for_fused_forward(model)
        set_fused_adapter_routing(model, ["actor"] * 2)

        # Act
        out = torch.utils.checkpoint.checkpoint(model, x, use_reentrant=False)
        out.square().sum().backward()
        fused_grads = _lora_grads(layer, "actor", x)
        model.zero_grad()
        ref = _reference_lora(layer, "actor", x_ref)
        ref.square().sum().backward()
        ref_grads = _lora_grads(layer, "actor", x_ref)

        # Assert
        torch.testing.assert_close(out, ref, rtol=0, atol=1e-6)
        for fused, expected in zip(fused_grads, ref_grads, strict=True):
            torch.testing.assert_close(fused, expected, rtol=0, atol=1e-5)

    def test_custom_base_single_run_stays_out_of_place(self):
        # Arrange: a view-returning base (like bitsandbytes) forbids in-place.
        model = _single_adapter_layer()
        layer = model.proj
        layer.base_layer = _ViewLinear(layer.base_layer)
        x = torch.randn(2, 5, 8, requires_grad=True)

        # Act
        out = model(x)
        out.sum().backward()

        # Assert
        torch.testing.assert_close(
            out, _reference_lora(layer, "actor", x), rtol=0, atol=1e-6
        )
        assert x.grad is not None

    def test_bf16_output_and_grads_match_fp32_reference(self):
        # Arrange: bf16 keeps 8 mantissa bits (ulp 2**-7 near 1); the fp32
        # reference uses the same bf16-rounded weights and inputs, so the gap
        # is a few output roundings: rtol 2e-2 allows ~2 ulps on |out| ~ 1-10.
        model = _single_adapter_layer(dtype=torch.bfloat16)
        layer = model.proj
        torch.manual_seed(1)
        x = torch.randn(2, 5, 8, dtype=torch.bfloat16, requires_grad=True)
        ref_model = copy.deepcopy(model).float()
        x_ref = x.detach().float().requires_grad_(True)

        # Act
        out = model(x)
        out.float().sum().backward()
        fused_grads = _lora_grads(layer, "actor", x)
        ref = _reference_lora(ref_model.proj, "actor", x_ref)
        ref.sum().backward()
        ref_grads = _lora_grads(ref_model.proj, "actor", x_ref)

        # Assert
        assert out.dtype == torch.bfloat16
        torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)
        for fused, expected in zip(fused_grads, ref_grads, strict=True):
            torch.testing.assert_close(fused.float(), expected, rtol=2e-2, atol=2e-2)

    def test_autocast_fuses_bf16_base_with_fp32_adapters(self):
        # Arrange: single-device training layout, fp32 adapters on a bf16 base
        # under bf16 autocast with PEFT's input cast disabled.
        model = _single_adapter_layer()
        base = model.proj.base_layer
        base.weight.data = base.weight.data.bfloat16().float()
        x = torch.randn(2, 5, 8, dtype=torch.bfloat16)
        with torch.no_grad():
            expected = _reference_lora(model.proj, "actor", x.float())
        base.to(torch.bfloat16)
        model.proj.cast_input_dtype_enabled = False

        # Act
        with (
            torch.no_grad(),
            torch.autocast("cpu", dtype=torch.bfloat16),
            profile(activities=[ProfilerActivity.CPU]) as prof,
        ):
            out = model(x)

        # Assert: same bf16 tolerance as the pure-bf16 case.
        ops = {event.key for event in prof.key_averages()}
        assert out.dtype == torch.bfloat16
        assert "aten::mul" not in ops
        torch.testing.assert_close(out.float(), expected, rtol=2e-2, atol=2e-2)

    def test_dropout_applies_to_lora_input_only(self):
        # Arrange
        model = _single_adapter_layer(lora_dropout=0.5)
        layer = model.proj
        x = torch.randn(2, 5, 8)

        # Act
        torch.manual_seed(3)
        out = model(x)
        torch.manual_seed(3)
        expected = _reference_lora(
            layer, "actor", x, lora_input=F.dropout(x, p=0.5, training=True)
        )

        # Assert
        torch.testing.assert_close(out, expected, rtol=0, atol=1e-6)
        assert not torch.allclose(out, _reference_lora(layer, "actor", x))

    def test_lora_bias_falls_back_to_unfused_formula(self):
        # Arrange
        model = _single_adapter_layer(lora_bias=True)
        layer = model.proj
        layer.lora_B["actor"].bias.data.normal_()
        x = torch.randn(2, 5, 8)

        # Act
        with torch.no_grad(), profile(activities=[ProfilerActivity.CPU]) as prof:
            out = model(x)

        # Assert
        ops = {event.key for event in prof.key_averages()}
        assert "aten::mul" in ops
        torch.testing.assert_close(
            out, _reference_lora(layer, "actor", x), rtol=0, atol=1e-6
        )

    def test_mixed_routing_accumulates_each_adapter_on_its_rows(self):
        # Arrange
        model = _build_model(r=4, lora_alpha=12)
        _enable_lora_grads(model)
        layer = model.proj
        x = torch.randn(4, 3, 8, requires_grad=True)
        patch_lora_for_fused_forward(model)
        set_fused_adapter_routing(model, ["actor"] * 2 + ["critic"] * 2)

        # Act
        out = model(x)
        out.sum().backward()
        critic_b_grad = layer.lora_B["critic"].weight.grad.clone()
        model.zero_grad()
        ref_critic = _reference_lora(layer, "critic", x[2:])
        ref_critic.sum().backward()

        # Assert
        torch.testing.assert_close(
            out[:2], _reference_lora(layer, "actor", x[:2]), rtol=0, atol=1e-6
        )
        torch.testing.assert_close(out[2:], ref_critic, rtol=0, atol=1e-6)
        torch.testing.assert_close(
            critic_b_grad, layer.lora_B["critic"].weight.grad, rtol=0, atol=1e-5
        )


class TestEmbeddingDelegation:
    def test_embedding_adapters_route_via_peft_mixed_forward(self):
        model = _build_model(module_cls=_TinyEmbedding, target_modules=["embed"])
        ids = torch.randint(0, 10, (2, 3))
        model.embed.set_adapter("actor")
        ref_actor = model(ids)
        model.embed.set_adapter("critic")
        ref_critic = model(ids)

        patch_lora_for_fused_forward(model)
        set_fused_adapter_routing(model, ["actor", "critic"])
        out = model(ids)

        assert torch.allclose(out[0], ref_actor[0], atol=1e-6)
        assert torch.allclose(out[1], ref_critic[1], atol=1e-6)


class TestSetFusedAdapterRoutingGuards:
    def test_set_raises_when_model_not_patched(self):
        model = _build_model()
        with pytest.raises(RuntimeError, match="patch_lora_for_fused_forward"):
            set_fused_adapter_routing(model, ["actor", "critic"])

    def test_set_noops_when_model_has_no_lora_layers(self):
        set_fused_adapter_routing(nn.Linear(2, 2), ["actor"])

    def test_unknown_adapter_name_raises(self):
        model = _build_model()
        patch_lora_for_fused_forward(model)
        with pytest.raises(ValueError, match="critc"):
            set_fused_adapter_routing(model, ["actor", "critc"])

    def test_empty_routing_raises(self):
        model = _build_model()
        patch_lora_for_fused_forward(model)
        with pytest.raises(ValueError, match="at least one row"):
            set_fused_adapter_routing(model, [])

    def test_merged_adapters_raise(self):
        model = _build_model(adapters=("actor",))
        patch_lora_for_fused_forward(model)
        model.proj.merge()
        with pytest.raises(RuntimeError, match="merged"):
            set_fused_adapter_routing(model, ["actor"])

    def test_dora_adapters_raise(self):
        model = _build_model(adapters=("actor",), use_dora=True, init_lora_weights=True)
        patch_lora_for_fused_forward(model)
        with pytest.raises(ValueError, match="DoRA"):
            set_fused_adapter_routing(model, ["actor"])

    def test_clear_on_unpatched_model_does_not_mask_the_patch_check(self):
        model = _build_model()
        unset_fused_adapter_routing(model)
        assert not _is_routed_layer(model.proj)
        with pytest.raises(RuntimeError, match="patch_lora_for_fused_forward"):
            set_fused_adapter_routing(model, ["actor"])


class TestPatchLifecycle:
    def test_repatch_is_idempotent(self):
        model = _build_model()
        patch_lora_for_fused_forward(model)
        routed = model.proj.__dict__["forward"]
        patch_lora_for_fused_forward(model)
        assert model.proj.__dict__["forward"] is routed

    def test_repatch_covers_layers_added_after_first_patch(self):
        model = _build_model(adapters=("actor",))
        patch_lora_for_fused_forward(model)
        model.late = inject_adapter_in_model(
            _lora_config(), _Tiny(), adapter_name="actor"
        )
        patch_lora_for_fused_forward(model)

        assert len(LORA_LAYER_CACHE[model]) == 2
        set_fused_adapter_routing(model, ["actor"])
        assert ROUTING_STATE[model.late.proj] == ["actor"]

    def test_unpatch_restores_original_forward_and_state(self):
        model = _build_model()
        x = torch.randn(2, 8)
        model.proj.set_adapter("actor")
        ref = model(x)

        patch_lora_for_fused_forward(model)
        set_fused_adapter_routing(model, ["critic", "critic"])
        unpatch_lora_for_fused_forward(model)

        assert "forward" not in model.proj.__dict__
        assert not _is_routed_layer(model.proj)
        assert model not in LORA_LAYER_CACHE
        assert torch.allclose(model(x), ref, atol=1e-6)
        with pytest.raises(RuntimeError, match="patch_lora_for_fused_forward"):
            set_fused_adapter_routing(model, ["actor"])

    def test_unpatch_on_never_patched_model_is_noop(self):
        model = _build_model()
        unpatch_lora_for_fused_forward(model)
        assert not _is_routed_layer(model.proj)

    def test_deepcopy_rebinds_routed_forward_to_the_copy(self):
        model = _build_model()
        patch_lora_for_fused_forward(model)
        clone = copy.deepcopy(model)

        assert clone.proj.__dict__["forward"].args[0] is clone.proj
        x = torch.randn(2, 8)
        set_fused_adapter_routing(clone, ["actor", "critic"])
        _ = clone(x)
        # The original stays unrouted.
        assert ROUTING_STATE.get(model.proj) is None


class TestLayerCache:
    def test_cache_is_stored_and_reused(self):
        model = _build_model()
        layers = get_cached_lora_layers(model)
        assert layers == [model.proj]
        assert LORA_LAYER_CACHE[model] is layers
        assert get_cached_lora_layers(model) is layers


class TestRoutedForwardAdapterComputeDtype:
    """fp32 adapters on a bf16 base, as DDP trains them (no autocast)."""

    @staticmethod
    def _bf16_base_fp32_adapters():
        model = _build_model(adapters=("actor",))
        for name, param in model.named_parameters():
            param.data = param.data.to(
                torch.float32 if "lora_" in name else torch.bfloat16
            )
        model.proj.base_layer.weight.requires_grad_(False)
        patch_lora_for_fused_forward(model)
        set_fused_adapter_routing(model, ["actor", "actor"])
        return model

    @pytest.mark.parametrize("cast_input", [True, False])
    def test_delta_runs_in_the_activation_dtype(self, cast_input):
        # Arrange
        model = self._bf16_base_fp32_adapters()
        for layer in get_cached_lora_layers(model):
            layer.cast_input_dtype_enabled = cast_input
        x = torch.randn(2, 8, dtype=torch.bfloat16, requires_grad=True)
        saved_dtypes = []

        def pack(tensor):
            saved_dtypes.append(tensor.dtype)
            return tensor

        # Act
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            out = model(x)

        # Assert: backward keeps no fp32 copy of the input or the adapters.
        assert out.dtype == torch.bfloat16
        assert saved_dtypes
        assert set(saved_dtypes) == {torch.bfloat16}

    def test_grads_stay_fp32_and_match_an_fp32_reference(self):
        # Arrange: the reference runs the same bf16-rounded weights and input
        # in fp32, so the gap is bf16 output rounding: rtol 2e-2 is ~2 ulps.
        model = self._bf16_base_fp32_adapters()
        layer = model.proj
        ref_layer = copy.deepcopy(layer).float()
        torch.manual_seed(1)
        x = torch.randn(2, 8, dtype=torch.bfloat16, requires_grad=True)
        x_ref = x.detach().float().requires_grad_(True)

        # Act
        out = model(x)
        out.float().sum().backward()
        ref = _reference_lora(ref_layer, "actor", x_ref)
        ref.sum().backward()

        # Assert
        grads = _lora_grads(layer, "actor", x)
        ref_grads = _lora_grads(ref_layer, "actor", x_ref)
        assert [grad.dtype for grad in grads] == [
            torch.bfloat16,
            torch.float32,
            torch.float32,
        ]
        torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)
        for grad, expected in zip(grads, ref_grads, strict=True):
            torch.testing.assert_close(grad.float(), expected, rtol=2e-2, atol=2e-2)


class TestParamWrapperRoutedForward:
    """Parameter-level LoRA wrappers under fused routing."""

    def test_uniform_routed_adapter_rejects_mixed_routing(self) -> None:
        model = _build_model(adapters=("actor",))
        layer = model.proj
        ROUTING_STATE[layer] = ["actor", "__base__"]

        with pytest.raises(RuntimeError, match="not supported on parameter-level"):
            uniform_routed_adapter(layer)

        ROUTING_STATE.pop(layer, None)

    def test_base_routing_disables_adapters_temporarily(self) -> None:
        model = _build_model(adapters=("actor",))
        layer = model.proj
        calls: list[bool] = []
        original_enable = layer.enable_adapters

        def _track(enabled: bool) -> None:
            calls.append(bool(enabled))
            return original_enable(enabled)

        layer.enable_adapters = _track  # type: ignore[method-assign]
        x = torch.randn(2, 8)
        layer.enable_adapters(False)
        expected = type(layer).forward(layer, x)
        layer.enable_adapters(True)
        calls.clear()
        ROUTING_STATE[layer] = ["__base__"]

        out = _param_wrapper_routed_forward(layer, type(layer).forward, x)

        assert torch.allclose(out, expected)
        assert calls == [False, True]
        ROUTING_STATE.pop(layer, None)

    def test_routed_adapter_applies_when_another_is_active(self) -> None:
        model = _build_model(adapters=("actor", "critic"))
        layer = model.proj
        x = torch.randn(2, 8)
        layer.set_adapter("actor")
        expected = type(layer).forward(layer, x)
        layer.set_adapter("critic")
        ROUTING_STATE[layer] = ["actor"]

        out = _param_wrapper_routed_forward(layer, type(layer).forward, x)

        assert torch.allclose(out, expected)
        assert list(layer.active_adapters) == ["critic"]
        ROUTING_STATE.pop(layer, None)


def _forward_ops(model, x):
    with torch.no_grad(), profile(activities=[ProfilerActivity.CPU]) as prof:
        out = model(x)
    return out, {event.key: event.count for event in prof.key_averages()}


def _unrouted_pair(**config_overrides):
    """A patched unrouted model and an unpatched PEFT copy with the same weights."""
    model = _build_model(adapters=("actor",), r=4, lora_alpha=12, **config_overrides)
    model.proj.base_layer.weight.requires_grad_(False)
    peft_model = copy.deepcopy(model)
    patch_lora_for_fused_forward(model)
    return model, peft_model


class TestUnroutedForward:
    def test_single_adapter_output_and_grads_match_peft(self):
        # Arrange
        model, peft_model = _unrouted_pair()
        torch.manual_seed(1)
        x = torch.randn(2, 5, 8, requires_grad=True)
        x_ref = x.detach().clone().requires_grad_(True)

        # Act
        out = model(x)
        out.square().sum().backward()
        ref = peft_model(x_ref)
        ref.square().sum().backward()

        # Assert: fp32, so only addmm's summation order separates the two.
        torch.testing.assert_close(out, ref, rtol=0, atol=1e-6)
        torch.testing.assert_close(
            _lora_grads(model.proj, "actor", x),
            _lora_grads(peft_model.proj, "actor", x_ref),
            rtol=1e-6,
            atol=1e-5,
        )

    def test_single_adapter_runs_one_addmm(self):
        model, _ = _unrouted_pair()

        _, ops = _forward_ops(model, torch.randn(2, 5, 8))

        assert ops.get("aten::addmm_") == 1
        assert "aten::mul" not in ops
        assert "aten::add" not in ops

    def test_eval_mode_dropout_runs_fused_and_matches_peft(self):
        # Arrange
        model, peft_model = _unrouted_pair(lora_dropout=0.1)
        model.eval()
        peft_model.eval()
        x = torch.randn(2, 5, 8)

        # Act
        out, ops = _forward_ops(model, x)

        # Assert: fp32, so only addmm's summation order separates the two.
        assert ops.get("aten::addmm_") == 1
        torch.testing.assert_close(out, peft_model(x), rtol=0, atol=1e-6)

    def test_training_mode_dropout_runs_peft(self):
        # Arrange
        model, _ = _unrouted_pair(lora_dropout=0.1)
        model.train()

        # Act
        _, ops = _forward_ops(model, torch.randn(2, 5, 8))

        # Assert
        assert "aten::addmm_" not in ops

    @pytest.mark.parametrize(
        ("config_overrides", "setup"),
        [
            ({"use_dora": True}, None),
            ({}, lambda layer: layer.enable_adapters(False)),
        ],
        ids=["dora", "adapters_disabled"],
    )
    def test_unsupported_layer_runs_peft(self, config_overrides, setup):
        # Arrange
        model, peft_model = _unrouted_pair(**config_overrides)
        for candidate in (model, peft_model):
            candidate.eval()
            if setup is not None:
                setup(candidate.proj)
        x = torch.randn(2, 5, 8)

        # Act
        out, ops = _forward_ops(model, x)

        # Assert
        assert "aten::addmm_" not in ops
        torch.testing.assert_close(out, peft_model(x), rtol=0, atol=0)

    def test_two_active_adapters_run_peft(self):
        # Arrange
        model = _build_model()
        model.proj.set_adapter(["actor", "critic"])
        peft_model = copy.deepcopy(model)
        patch_lora_for_fused_forward(model)
        x = torch.randn(2, 8)

        # Act
        out, ops = _forward_ops(model, x)

        # Assert
        assert "aten::addmm_" not in ops
        torch.testing.assert_close(out, peft_model(x), rtol=0, atol=0)


def _tiny_grpo() -> GRPO:
    torch.manual_seed(0)
    return GRPO(
        actor_network=create_module(
            input_size=6, max_tokens=4, vocab_size=64, device="cpu"
        ),
        pad_token_id=63,
        pad_token="<pad>",
        batch_size=2,
        group_size=2,
        max_output_tokens=4,
        max_model_len=12,
        wrap=False,
        gradient_checkpointing=False,
        device="cpu",
        lora_config=LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["linear_1"],
            task_type="CAUSAL_LM",
            lora_dropout=0.0,
            init_lora_weights=False,
        ),
    )


class TestUnroutedGRPOLogprobs:
    def test_logprobs_and_grads_match_peft(self):
        # Arrange
        grpo = _tiny_grpo()
        torch.manual_seed(1)
        ids = torch.randint(0, 63, (4, 6))
        trainable = {
            name: param
            for name, param in grpo.actor.named_parameters()
            if param.requires_grad
        }

        # Act
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            fused = grpo._get_logprobs(ids, batch_size=4)
        fused.sum().backward()
        fused_grads = {name: p.grad.clone() for name, p in trainable.items()}
        grpo.actor.zero_grad()
        unpatch_lora_for_fused_forward(grpo.actor)
        peft = grpo._get_logprobs(ids, batch_size=4)
        peft.sum().backward()

        # Assert: fp32, so only addmm's summation order separates the two.
        ops = {event.key for event in prof.key_averages()}
        assert "aten::addmm_" in ops
        assert fused_grads
        torch.testing.assert_close(fused, peft, rtol=0, atol=1e-6)
        for name, param in trainable.items():
            torch.testing.assert_close(fused_grads[name], param.grad, rtol=0, atol=1e-5)

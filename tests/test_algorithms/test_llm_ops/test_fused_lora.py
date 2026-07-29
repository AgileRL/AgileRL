# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

import copy

import pytest
import torch
from peft import LoraConfig, inject_adapter_in_model
from torch import nn

from agilerl.algorithms.core.llm_ops.fused_lora import (
    _LORA_LAYER_CACHE,
    _ROUTING_STATE,
    _is_routed_layer,
    get_cached_lora_layers,
    patch_lora_for_fused_forward,
    set_fused_adapter_routing,
    unpatch_lora_for_fused_forward,
    unset_fused_adapter_routing,
)


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
        set_fused_adapter_routing(model, ["actor"] * 3)
        with pytest.raises(ValueError, match="covers 3 rows"):
            model(torch.randn(4, 8))

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

        assert len(_LORA_LAYER_CACHE[model]) == 2
        set_fused_adapter_routing(model, ["actor"])
        assert _ROUTING_STATE[model.late.proj] == ["actor"]

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
        assert model not in _LORA_LAYER_CACHE
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
        assert _ROUTING_STATE.get(model.proj) is None


class TestLayerCache:
    def test_cache_is_stored_and_reused(self):
        model = _build_model()
        layers = get_cached_lora_layers(model)
        assert layers == [model.proj]
        assert _LORA_LAYER_CACHE[model] is layers
        assert get_cached_lora_layers(model) is layers


class TestFusedLoraInputCast:
    @staticmethod
    def _bf16_base_fp32_adapters():
        model = _build_model(adapters=("actor",))
        for name, param in model.named_parameters():
            param.data = param.data.to(
                torch.float32 if "lora_" in name else torch.bfloat16
            )
        patch_lora_for_fused_forward(model)
        set_fused_adapter_routing(model, ["actor", "actor"])
        return model

    def test_routed_forward_casts_inputs_by_default(self):
        model = self._bf16_base_fp32_adapters()
        assert model(torch.randn(2, 8, dtype=torch.bfloat16)).dtype == torch.bfloat16

    def test_routed_forward_honours_disabled_cast(self):
        model = self._bf16_base_fp32_adapters()
        for layer in get_cached_lora_layers(model):
            layer.cast_input_dtype_enabled = False

        with pytest.raises(RuntimeError):
            model(torch.randn(2, 8, dtype=torch.bfloat16))

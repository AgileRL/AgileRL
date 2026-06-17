from unittest.mock import patch

import pytest
import torch
from torch import nn

from agilerl.algorithms.core.llm_ops.fused_lora import (
    _fused_routing_pre_hook,
    _get_cached_lora_layers,
    clear_fused_adapter_routing,
    patch_lora_for_fused_forward,
    set_fused_adapter_routing,
    unpatch_lora_for_fused_forward,
)


class _DummyLoraLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_adapter_names = None

    def forward(self, x, adapter_names=None):  # noqa: ANN001
        self.last_adapter_names = adapter_names
        return x


class _DummyFusedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lora_a = _DummyLoraLayer()
        self.linear = nn.Linear(2, 2)
        self.lora_b = _DummyLoraLayer()


class _CacheRejectingFusedModel(_DummyFusedModel):
    def __setattr__(self, name, value):  # noqa: ANN001
        if name == "_fused_lora_layers":
            raise AttributeError("cache assignment not allowed")
        super().__setattr__(name, value)


class _BaseLayerLoraLayer(_DummyLoraLayer):
    """LoRA layer exposing a ``base_layer`` submodule, like real PEFT layers.

    ``base_layer`` returns its input unchanged, so the clone hook is the only
    thing that can produce a distinct output tensor.
    """

    def __init__(self) -> None:
        super().__init__()
        self.base_layer = nn.Identity()


class TestPatchLoraForFusedForward:
    def test_patch_lora_for_fused_forward_registers_hooks_and_cache(self):
        model = _DummyFusedModel()
        with patch(
            "agilerl.algorithms.core.llm_ops.fused_lora.LoraLayer", _DummyLoraLayer
        ):
            patch_lora_for_fused_forward(model)

        assert hasattr(model, "_fused_lora_layers")
        assert len(model._fused_lora_layers) == 2
        for layer in model._fused_lora_layers:
            assert layer._fused_adapter_routing is None
            assert len(layer._forward_pre_hooks) >= 1

    def test_patch_lora_for_fused_forward_noops_when_loralayer_none(self):
        model = _DummyFusedModel()
        with patch("agilerl.algorithms.core.llm_ops.fused_lora.LoraLayer", None):
            patch_lora_for_fused_forward(model)

        assert not hasattr(model, "_fused_lora_layers")
        assert not hasattr(model.lora_a, "_fused_adapter_routing")
        assert not hasattr(model.lora_b, "_fused_adapter_routing")

    def test_patch_lora_for_fused_forward_ignores_cache_assignment_errors(self):
        model = _CacheRejectingFusedModel()
        with patch(
            "agilerl.algorithms.core.llm_ops.fused_lora.LoraLayer", _DummyLoraLayer
        ):
            patch_lora_for_fused_forward(model)

        assert not hasattr(model, "_fused_lora_layers")
        for layer in (model.lora_a, model.lora_b):
            assert layer._fused_adapter_routing is None
            assert len(layer._forward_pre_hooks) >= 1


def test_set_and_clear_fused_adapter_routing_update_all_lora_layers():
    model = _DummyFusedModel()
    routing = ["actor", "critic"]
    with patch("agilerl.algorithms.core.llm_ops.fused_lora.LoraLayer", _DummyLoraLayer):
        patch_lora_for_fused_forward(model)
        set_fused_adapter_routing(model, routing)
        for layer in model._fused_lora_layers:
            assert layer._fused_adapter_routing == routing

        clear_fused_adapter_routing(model)
        for layer in model._fused_lora_layers:
            assert layer._fused_adapter_routing is None


class TestFusedRoutingPreHook:
    def test_fused_lora_hook_injects_adapter_names_into_forward_kwargs(self):
        model = _DummyFusedModel()
        routing = ["actor", "critic"]
        with patch(
            "agilerl.algorithms.core.llm_ops.fused_lora.LoraLayer", _DummyLoraLayer
        ):
            patch_lora_for_fused_forward(model)
            set_fused_adapter_routing(model, routing)
            _ = model.lora_a(torch.ones(1, 2))

        assert model.lora_a.last_adapter_names == routing

    def test_fused_routing_pre_hook_leaves_kwargs_unchanged_without_routing(self):
        module = _DummyLoraLayer()
        args = (torch.ones(1, 2),)
        kwargs = {"existing_kwarg": "present"}

        returned_args, returned_kwargs = _fused_routing_pre_hook(module, args, kwargs)

        assert returned_args == args
        assert returned_kwargs["existing_kwarg"] == "present"
        assert "adapter_names" not in returned_kwargs


class _AdapterAwareLoraLayer(_DummyLoraLayer):
    """Dummy layer exposing PEFT-style adapter containers for validation."""

    adapter_layer_names = ("lora_A", "lora_B")

    def __init__(self, adapters=("actor", "critic")) -> None:  # noqa: ANN001
        super().__init__()
        self.lora_A = nn.ModuleDict({name: nn.Identity() for name in adapters})
        self.lora_B = nn.ModuleDict({name: nn.Identity() for name in adapters})


class _AdapterAwareFusedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lora_a = _AdapterAwareLoraLayer()
        self.lora_b = _AdapterAwareLoraLayer()


class TestSetFusedAdapterRoutingGuards:
    def test_set_raises_when_model_not_patched(self):
        model = _DummyFusedModel()
        with (
            patch(
                "agilerl.algorithms.core.llm_ops.fused_lora.LoraLayer",
                _DummyLoraLayer,
            ),
            pytest.raises(RuntimeError, match="patch_lora_for_fused_forward"),
        ):
            set_fused_adapter_routing(model, ["actor", "critic"])

    def test_set_noops_when_model_has_no_lora_layers(self):
        model = nn.Linear(2, 2)
        with patch(
            "agilerl.algorithms.core.llm_ops.fused_lora.LoraLayer", _DummyLoraLayer
        ):
            set_fused_adapter_routing(model, ["actor"])

    def test_clear_on_unpatched_model_does_not_mask_the_patch_check(self):
        model = _DummyFusedModel()
        with patch(
            "agilerl.algorithms.core.llm_ops.fused_lora.LoraLayer", _DummyLoraLayer
        ):
            clear_fused_adapter_routing(model)
            assert not hasattr(model.lora_a, "_fused_adapter_routing")
            with pytest.raises(RuntimeError, match="patch_lora_for_fused_forward"):
                set_fused_adapter_routing(model, ["actor"])


class TestSetFusedAdapterRoutingValidation:
    def test_known_adapter_names_accepted(self):
        model = _AdapterAwareFusedModel()
        with patch(
            "agilerl.algorithms.core.llm_ops.fused_lora.LoraLayer", _DummyLoraLayer
        ):
            patch_lora_for_fused_forward(model)
            set_fused_adapter_routing(model, ["actor", "critic"])

        assert model.lora_a._fused_adapter_routing == ["actor", "critic"]

    def test_unknown_adapter_name_raises(self):
        model = _AdapterAwareFusedModel()
        with patch(
            "agilerl.algorithms.core.llm_ops.fused_lora.LoraLayer", _DummyLoraLayer
        ):
            patch_lora_for_fused_forward(model)
            with pytest.raises(ValueError, match="critc"):
                set_fused_adapter_routing(model, ["actor", "critc"])

    def test_base_adapter_name_always_allowed(self):
        model = _AdapterAwareFusedModel()
        with patch(
            "agilerl.algorithms.core.llm_ops.fused_lora.LoraLayer", _DummyLoraLayer
        ):
            patch_lora_for_fused_forward(model)
            set_fused_adapter_routing(model, ["__base__", "actor"])

        assert model.lora_a._fused_adapter_routing == ["__base__", "actor"]

    def test_adapter_names_unioned_across_layers(self):
        model = nn.Module()
        model.lora_a = _AdapterAwareLoraLayer(adapters=("actor",))
        model.lora_b = _AdapterAwareLoraLayer(adapters=("critic",))
        with patch(
            "agilerl.algorithms.core.llm_ops.fused_lora.LoraLayer", _DummyLoraLayer
        ):
            patch_lora_for_fused_forward(model)
            set_fused_adapter_routing(model, ["actor", "critic"])

        assert model.lora_b._fused_adapter_routing == ["actor", "critic"]

    def test_validation_skipped_when_no_adapter_containers(self):
        # Layers that don't expose PEFT's adapter containers (opaque test
        # doubles) keep the previous unvalidated behavior.
        model = _DummyFusedModel()
        with patch(
            "agilerl.algorithms.core.llm_ops.fused_lora.LoraLayer", _DummyLoraLayer
        ):
            patch_lora_for_fused_forward(model)
            set_fused_adapter_routing(model, ["anything-goes"])

        assert model.lora_a._fused_adapter_routing == ["anything-goes"]


class TestPatchIdempotency:
    def test_repatch_does_not_double_register_hooks(self):
        model = _DummyFusedModel()
        with patch(
            "agilerl.algorithms.core.llm_ops.fused_lora.LoraLayer", _DummyLoraLayer
        ):
            patch_lora_for_fused_forward(model)
            patch_lora_for_fused_forward(model)

        for layer in (model.lora_a, model.lora_b):
            assert len(layer._forward_pre_hooks) == 1

    def test_repatch_hooks_layers_added_after_first_patch(self):
        model = _DummyFusedModel()
        with patch(
            "agilerl.algorithms.core.llm_ops.fused_lora.LoraLayer", _DummyLoraLayer
        ):
            patch_lora_for_fused_forward(model)
            model.lora_c = _DummyLoraLayer()
            patch_lora_for_fused_forward(model)

            assert len(model._fused_lora_layers) == 3
            assert len(model.lora_c._forward_pre_hooks) == 1
            assert model.lora_c._fused_adapter_routing is None
            # Existing layers keep their single hook.
            assert len(model.lora_a._forward_pre_hooks) == 1

            set_fused_adapter_routing(model, ["actor"])
            assert model.lora_c._fused_adapter_routing == ["actor"]


class TestUnpatchLoraForFusedForward:
    def test_unpatch_removes_hooks_state_and_cache(self):
        model = _DummyFusedModel()
        with patch(
            "agilerl.algorithms.core.llm_ops.fused_lora.LoraLayer", _DummyLoraLayer
        ):
            patch_lora_for_fused_forward(model)
            set_fused_adapter_routing(model, ["actor", "critic"])
            unpatch_lora_for_fused_forward(model)

            for layer in (model.lora_a, model.lora_b):
                assert len(layer._forward_pre_hooks) == 0
                assert not hasattr(layer, "_fused_adapter_routing")
                assert not hasattr(layer, "_fused_routing_hook_handle")
            assert not hasattr(model, "_fused_lora_layers")

            # Forward no longer injects adapter_names.
            _ = model.lora_a(torch.ones(1, 2))
            assert model.lora_a.last_adapter_names is None

            # And routing cannot be silently re-applied without re-patching.
            with pytest.raises(RuntimeError, match="patch_lora_for_fused_forward"):
                set_fused_adapter_routing(model, ["actor"])

    def test_unpatch_then_repatch_restores_routing(self):
        model = _DummyFusedModel()
        with patch(
            "agilerl.algorithms.core.llm_ops.fused_lora.LoraLayer", _DummyLoraLayer
        ):
            patch_lora_for_fused_forward(model)
            unpatch_lora_for_fused_forward(model)
            patch_lora_for_fused_forward(model)
            set_fused_adapter_routing(model, ["actor"])
            _ = model.lora_a(torch.ones(1, 2))

        assert model.lora_a.last_adapter_names == ["actor"]
        assert len(model.lora_a._forward_pre_hooks) == 1

    def test_unpatch_on_never_patched_model_is_noop(self):
        model = _DummyFusedModel()
        with patch(
            "agilerl.algorithms.core.llm_ops.fused_lora.LoraLayer", _DummyLoraLayer
        ):
            unpatch_lora_for_fused_forward(model)

        assert not hasattr(model, "_fused_lora_layers")
        assert not hasattr(model.lora_a, "_fused_adapter_routing")


class TestBaseOutputCloneHook:
    """The base_layer forward hook clones the frozen base output only while
    fused routing is active (so PEFT's in-place LoRA accumulation can't mutate
    a bnb custom-Function output view)."""

    def test_clones_base_output_when_routing_active(self):
        model = nn.Module()
        model.lora_a = _BaseLayerLoraLayer()
        with patch(
            "agilerl.algorithms.core.llm_ops.fused_lora.LoraLayer", _DummyLoraLayer
        ):
            patch_lora_for_fused_forward(model)
            set_fused_adapter_routing(model, ["actor"])
            x = torch.ones(2, 2)
            out = model.lora_a.base_layer(x)

        # Identity returns its input; a distinct, equal tensor means the hook
        # cloned it.
        assert out is not x
        assert torch.equal(out, x)

    def test_noop_when_routing_inactive(self):
        model = nn.Module()
        model.lora_a = _BaseLayerLoraLayer()
        with patch(
            "agilerl.algorithms.core.llm_ops.fused_lora.LoraLayer", _DummyLoraLayer
        ):
            patch_lora_for_fused_forward(model)  # routing defaults to None
            x = torch.ones(2, 2)
            out = model.lora_a.base_layer(x)

        # No routing -> hook is a no-op -> Identity's input passes through.
        assert out is x

    def test_unpatch_removes_clone_handle(self):
        model = nn.Module()
        model.lora_a = _BaseLayerLoraLayer()
        with patch(
            "agilerl.algorithms.core.llm_ops.fused_lora.LoraLayer", _DummyLoraLayer
        ):
            patch_lora_for_fused_forward(model)
            unpatch_lora_for_fused_forward(model)
            assert not hasattr(model.lora_a, "_fused_base_clone_handle")
            assert len(model.lora_a.base_layer._forward_hooks) == 0


class TestGetCachedLoraLayers:
    def test_get_cached_lora_layers_returns_empty_when_loralayer_none(self):
        model = _DummyFusedModel()
        with patch("agilerl.algorithms.core.llm_ops.fused_lora.LoraLayer", None):
            layers = _get_cached_lora_layers(model)

        assert layers == []
        assert not hasattr(model, "_fused_lora_layers")

    def test_get_cached_lora_layers_ignores_cache_assignment_errors(self):
        model = _CacheRejectingFusedModel()
        with patch(
            "agilerl.algorithms.core.llm_ops.fused_lora.LoraLayer", _DummyLoraLayer
        ):
            layers = _get_cached_lora_layers(model)

        assert len(layers) == 2
        assert all(isinstance(layer, _DummyLoraLayer) for layer in layers)
        assert not hasattr(model, "_fused_lora_layers")

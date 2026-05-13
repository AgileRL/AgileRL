from unittest.mock import patch

import torch
from torch import nn

from agilerl.algorithms.core.fused_lora import (
    _fused_routing_pre_hook,
    _get_cached_lora_layers,
    clear_fused_adapter_routing,
    patch_lora_for_fused_forward,
    set_fused_adapter_routing,
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


class TestPatchLoraForFusedForward:
    def test_patch_lora_for_fused_forward_registers_hooks_and_cache(self):
        model = _DummyFusedModel()
        with patch("agilerl.algorithms.core.fused_lora.LoraLayer", _DummyLoraLayer):
            patch_lora_for_fused_forward(model)

        assert hasattr(model, "_fused_lora_layers")
        assert len(model._fused_lora_layers) == 2
        for layer in model._fused_lora_layers:
            assert layer._fused_adapter_routing is None
            assert len(layer._forward_pre_hooks) >= 1

    def test_patch_lora_for_fused_forward_noops_when_loralayer_none(self):
        model = _DummyFusedModel()
        with patch("agilerl.algorithms.core.fused_lora.LoraLayer", None):
            patch_lora_for_fused_forward(model)

        assert not hasattr(model, "_fused_lora_layers")
        assert not hasattr(model.lora_a, "_fused_adapter_routing")
        assert not hasattr(model.lora_b, "_fused_adapter_routing")

    def test_patch_lora_for_fused_forward_ignores_cache_assignment_errors(self):
        model = _CacheRejectingFusedModel()
        with patch("agilerl.algorithms.core.fused_lora.LoraLayer", _DummyLoraLayer):
            patch_lora_for_fused_forward(model)

        assert not hasattr(model, "_fused_lora_layers")
        for layer in (model.lora_a, model.lora_b):
            assert layer._fused_adapter_routing is None
            assert len(layer._forward_pre_hooks) >= 1


def test_set_and_clear_fused_adapter_routing_update_all_lora_layers():
    model = _DummyFusedModel()
    routing = ["actor", "critic"]
    with patch("agilerl.algorithms.core.fused_lora.LoraLayer", _DummyLoraLayer):
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
        with patch("agilerl.algorithms.core.fused_lora.LoraLayer", _DummyLoraLayer):
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


class TestGetCachedLoraLayers:
    def test_get_cached_lora_layers_returns_empty_when_loralayer_none(self):
        model = _DummyFusedModel()
        with patch("agilerl.algorithms.core.fused_lora.LoraLayer", None):
            layers = _get_cached_lora_layers(model)

        assert layers == []
        assert not hasattr(model, "_fused_lora_layers")

    def test_get_cached_lora_layers_ignores_cache_assignment_errors(self):
        model = _CacheRejectingFusedModel()
        with patch("agilerl.algorithms.core.fused_lora.LoraLayer", _DummyLoraLayer):
            layers = _get_cached_lora_layers(model)

        assert len(layers) == 2
        assert all(isinstance(layer, _DummyLoraLayer) for layer in layers)
        assert not hasattr(model, "_fused_lora_layers")

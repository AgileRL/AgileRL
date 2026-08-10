# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the colocated-vLLM helpers (GPU-free logic).

vLLM and the trainer each hold their own base; only LoRA adapters are synced
per rollout. These cover the deterministic, GPU-free helpers in
``vllm_colocate``:

* ``get_vllm_internal_model`` — reaching the live ``nn.Module`` inside an
  in-process engine across the known vLLM attribute layouts.
* ``patch_vllm_strip_multimodal_towers`` (and its ``_StrippedTower`` placeholder)
  — freeing the GPU memory of unused vision/audio towers.
* ``patch_vllm_lora_keep_resident`` — neutralizing ``reset_lora`` so the single
  persistent rollout-adapter slot is not zeroed between forwards.

All run on plain module trees with fakes, so no GPU is required.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from agilerl.algorithms.core.llm_ops.vllm_colocate import (
    _StrippedTower,
    get_vllm_internal_model,
    patch_vllm_3d_moe_lora_flag,
    patch_vllm_lora_keep_resident,
    patch_vllm_strip_multimodal_towers,
)

_PACKED = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}


def _quant_linear(
    out_shards: list[int], in_features: int, bias: bool = False
) -> nn.Module:
    """nn.Module with a fused, bnb-packed-style ``.weight`` Parameter.

    Each shard gets its own ``QuantState``-like object (only ``.shape`` is read);
    ``output_sizes`` mirrors vLLM's fused linear. Used to build a realistic fake
    internal model for the vLLM-internals access test.
    """
    m = nn.Module()
    w = nn.Parameter(torch.randn(sum(out_shards), in_features), requires_grad=False)
    offsets = [0]
    for s in out_shards:
        offsets.append(offsets[-1] + s)
    w.bnb_shard_offsets = offsets
    w.bnb_quant_state = {
        i: SimpleNamespace(shape=(out_shards[i], in_features))
        for i in range(len(out_shards))
    }
    m.weight = w
    m.output_sizes = list(out_shards)
    if bias:
        m.bias = nn.Parameter(torch.randn(sum(out_shards)), requires_grad=False)
    return m


def _plain(out_features: int, in_features: int | None = None) -> nn.Module:
    m = nn.Module()
    shape = (out_features,) if in_features is None else (out_features, in_features)
    m.weight = nn.Parameter(torch.randn(*shape), requires_grad=False)
    return m


def _decoder(
    vocab: int, hidden: int, n_layers: int, vocab_pad: int, bias: bool = False
) -> nn.Module:
    root = nn.Module()
    root.embed_tokens = _plain(vocab + vocab_pad, hidden)
    layers = []
    for _ in range(n_layers):
        layer = nn.Module()
        attn = nn.Module()
        attn.qkv_proj = _quant_linear(
            [hidden, hidden // 2, hidden // 2], hidden, bias=bias
        )
        attn.o_proj = _quant_linear([hidden], hidden)
        attn.q_norm = _plain(hidden // 2)
        attn.k_norm = _plain(hidden // 2)
        layer.self_attn = attn
        mlp = nn.Module()
        mlp.gate_up_proj = _quant_linear([hidden * 2, hidden * 2], hidden)
        mlp.down_proj = _quant_linear([hidden], hidden * 2)
        layer.mlp = mlp
        layer.input_layernorm = _plain(hidden)
        layer.post_attention_layernorm = _plain(hidden)
        # A raw (non-.weight) per-layer parameter, e.g. an AltUp/Laurel scale.
        layer.altup_scale = nn.Parameter(torch.randn(hidden), requires_grad=False)
        layers.append(layer)
    root.layers = nn.ModuleList(layers)
    root.norm = _plain(hidden)
    return root


class _FakeInternal(nn.Module):
    packed_modules_mapping = _PACKED

    def __init__(
        self,
        *,
        tie,
        multimodal,
        vocab=32,
        hidden=8,
        n_layers=1,
        vocab_pad=4,
        bias=False,
    ):
        super().__init__()
        decoder = _decoder(vocab, hidden, n_layers, vocab_pad, bias=bias)
        if multimodal:
            self.language_model = nn.Module()
            self.language_model.model = decoder
            if not tie:
                self.language_model.lm_head = _plain(vocab + vocab_pad, hidden)
            # A non-language tower.
            self.vision_tower = _plain(hidden, hidden)
        else:
            self.model = decoder
            if not tie:
                self.lm_head = _plain(vocab + vocab_pad, hidden)


def _wrap_llm(internal: nn.Module) -> SimpleNamespace:
    driver = SimpleNamespace(model_runner=SimpleNamespace(model=internal))
    engine = SimpleNamespace(model_executor=SimpleNamespace(driver_worker=driver))
    return SimpleNamespace(llm_engine=engine)


def _make_fake(*, tie=True, multimodal=False, vocab=32, vocab_pad=4, bias=False):
    internal = _FakeInternal(
        tie=tie, multimodal=multimodal, vocab=vocab, vocab_pad=vocab_pad, bias=bias
    )
    llm = _wrap_llm(internal)
    config = SimpleNamespace(
        vocab_size=vocab, num_hidden_layers=1, tie_word_embeddings=tie
    )
    return llm, config, internal


class TestGetVllmInternalModel:
    def test_direct_engine_path(self):
        llm, _config, internal = _make_fake()
        assert get_vllm_internal_model(llm) is internal

    def test_v1_engine_core_path(self):
        _llm, _config, internal = _make_fake()
        driver = SimpleNamespace(model_runner=SimpleNamespace(model=internal))
        inner_core = SimpleNamespace(
            model_executor=SimpleNamespace(driver_worker=driver)
        )
        engine = SimpleNamespace(engine_core=SimpleNamespace(engine_core=inner_core))
        llm = SimpleNamespace(llm_engine=engine)
        assert get_vllm_internal_model(llm) is internal

    def test_raises_when_unreachable(self):
        llm = SimpleNamespace(llm_engine=SimpleNamespace())
        with pytest.raises(RuntimeError, match="Could not locate"):
            get_vllm_internal_model(llm)


class TestStrippedTower:
    def test_falsy(self):
        # Falsy so ``if self.vision_tower:`` guards skip the stripped tower.
        assert bool(_StrippedTower("model.vision_tower")) is False

    def test_call_raises_with_path_and_remediation(self):
        tower = _StrippedTower("model.vision_tower")
        with pytest.raises(
            RuntimeError,
            match=r"Stripped multimodal tower 'model\.vision_tower' "
            r"\(called as a forward path\)",
        ) as exc:
            tower(torch.randn(1, 4))
        # The error tells the user which knob restores the towers.
        assert "strip_multimodal_towers=False" in str(exc.value)

    def test_getattr_raises_attribute_error_with_name(self):
        tower = _StrippedTower("audio_tower")
        assert tower._stripped_path == "audio_tower"  # the path itself is readable
        with pytest.raises(AttributeError, match="attribute 'forward' accessed"):
            _ = tower.forward


class TestStripMultimodalTowers:
    def _internal(self):
        # Plain-namespace holders: vLLM TP wrappers can hold towers as plain
        # attributes, and these keep the freed param counts real without
        # nn.Module registration. Registered-module holders are exercised by
        # test_module_registered_tower_is_stripped.
        inner = SimpleNamespace(vision_tower=_plain(4, 4))  # 16 params
        return SimpleNamespace(
            model=inner,
            multi_modal_projector=_plain(2, 2),  # 4 params
            embed_vision=None,  # explicit None slot: skipped, never stripped
        )

    def test_module_registered_tower_is_stripped(self):
        # The REAL vLLM shape: towers are registered child modules of an
        # nn.Module (e.g. Gemma3-MM's ``self.vision_tower =
        # SiglipVisionModel(...)``). ``nn.Module.__setattr__`` refuses to
        # replace a registered child with a non-Module, so the strip must drop
        # the ``_modules`` registration first and install the placeholder as a
        # plain instance attribute.
        internal = nn.Module()
        internal.vision_tower = _plain(4, 4)  # registered child module
        freed = patch_vllm_strip_multimodal_towers(_wrap_llm(internal))
        assert freed == {"vision_tower": 16}
        assert isinstance(internal.vision_tower, _StrippedTower)
        assert "vision_tower" not in internal._modules
        assert list(internal.parameters()) == []

    def test_strips_towers_and_reports_freed_params(self):
        internal = self._internal()
        freed = patch_vllm_strip_multimodal_towers(_wrap_llm(internal))
        assert freed == {"multi_modal_projector": 4, "model.vision_tower": 16}
        assert isinstance(internal.model.vision_tower, _StrippedTower)
        assert isinstance(internal.multi_modal_projector, _StrippedTower)
        # Any future forward through the stripped tower fails loudly...
        with pytest.raises(RuntimeError, match=r"model\.vision_tower"):
            internal.model.vision_tower(torch.randn(1, 4))
        # ...while falsy checks still short-circuit cleanly.
        assert not internal.model.vision_tower

    def test_second_call_is_idempotent(self):
        internal = self._internal()
        llm = _wrap_llm(internal)
        assert patch_vllm_strip_multimodal_towers(llm)
        assert patch_vllm_strip_multimodal_towers(llm) == {}

    def test_returns_empty_when_model_unreachable(self):
        llm = SimpleNamespace(llm_engine=SimpleNamespace())
        assert patch_vllm_strip_multimodal_towers(llm) == {}

    def test_unsizable_tower_reports_zero_params(self):
        internal = nn.Module()
        internal.vision_tower = SimpleNamespace()  # no .parameters(): count fails
        freed = patch_vllm_strip_multimodal_towers(_wrap_llm(internal))
        assert freed == {"vision_tower": 0}
        assert isinstance(internal.vision_tower, _StrippedTower)

    def test_custom_tower_attrs_strip_only_those(self):
        internal = SimpleNamespace(vision_tower=_plain(2, 2), spare_tower=_plain(2, 2))
        freed = patch_vllm_strip_multimodal_towers(
            _wrap_llm(internal), tower_attrs=["spare_tower"]
        )
        assert freed == {"spare_tower": 4}
        assert not isinstance(internal.vision_tower, _StrippedTower)

    def test_cuda_cache_emptied_only_when_available(self, monkeypatch):
        calls = []
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append(True))
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        assert patch_vllm_strip_multimodal_towers(_wrap_llm(self._internal()))
        assert calls == [True]
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert patch_vllm_strip_multimodal_towers(_wrap_llm(self._internal()))
        assert calls == [True]  # no CUDA: cache not touched


class _FakeVllmLoraLinear(nn.Module):
    """vLLM-style LoRA-wrapped layer: ``reset_lora`` zeroes the stacked slot."""

    def __init__(self):
        super().__init__()
        self.lora_b_stacked = (torch.ones(1, 1, 4, 2),)

    def reset_lora(self, index):
        self.lora_b_stacked[0][index] = 0


class TestLoraKeepResident:
    def _internal(self):
        internal = nn.Module()
        internal.q_proj = _FakeVllmLoraLinear()
        internal.v_proj = _FakeVllmLoraLinear()
        internal.no_slot = nn.Linear(2, 2)
        return internal

    def test_fake_reset_lora_zeroes_slot(self):
        # Control: the un-patched fake really clears the slot, so the
        # neutralization assertions below are not vacuous.
        layer = _FakeVllmLoraLinear()
        layer.reset_lora(0)
        assert torch.equal(layer.lora_b_stacked[0], torch.zeros(1, 1, 4, 2))

    def test_neutralizes_reset_lora_and_counts_layers(self):
        internal = self._internal()
        assert patch_vllm_lora_keep_resident(_wrap_llm(internal)) == 2
        # A no-LoRA batch calling reset_lora no longer wipes the adapter slot.
        internal.q_proj.reset_lora(0)
        assert torch.equal(internal.q_proj.lora_b_stacked[0], torch.ones(1, 1, 4, 2))
        assert internal.q_proj._agilerl_lora_resident is True

    def test_second_call_is_idempotent(self):
        internal = self._internal()
        llm = _wrap_llm(internal)
        assert patch_vllm_lora_keep_resident(llm) == 2
        assert patch_vllm_lora_keep_resident(llm) == 0

    def test_layers_without_lora_slot_left_alone(self):
        internal = self._internal()
        patch_vllm_lora_keep_resident(_wrap_llm(internal))
        assert not getattr(internal.no_slot, "_agilerl_lora_resident", False)

    def test_returns_zero_when_model_unreachable(self):
        llm = SimpleNamespace(llm_engine=SimpleNamespace())
        assert patch_vllm_lora_keep_resident(llm) == 0


class TestPatchVllm3dMoeLoraFlag:
    def test_returns_false_on_lookup_failure(self) -> None:
        assert (
            patch_vllm_3d_moe_lora_flag("/definitely/missing/agilerl-3d-moe-flag-model")
            is False
        )

    def test_sets_flag_and_is_idempotent(self, monkeypatch) -> None:
        import sys
        from types import ModuleType

        class _ModelCls:
            pass

        class _Entry:
            @staticmethod
            def load_model_cls():
                return _ModelCls

        transformers_mod = ModuleType("transformers")
        transformers_mod.AutoConfig = SimpleNamespace(
            from_pretrained=lambda *_a, **_k: SimpleNamespace(
                architectures=["FakeArch"]
            )
        )
        registry_mod = ModuleType("vllm.model_executor.models.registry")
        registry_mod.ModelRegistry = SimpleNamespace(
            models={"FakeArch": _Entry()},
        )
        # Minimal parent packages so ``from vllm... import`` resolves.
        for name in (
            "vllm",
            "vllm.model_executor",
            "vllm.model_executor.models",
        ):
            sys.modules.setdefault(name, ModuleType(name))
        monkeypatch.setitem(sys.modules, "transformers", transformers_mod)
        monkeypatch.setitem(
            sys.modules,
            "vllm.model_executor.models.registry",
            registry_mod,
        )

        assert patch_vllm_3d_moe_lora_flag("fake/model") is True
        assert _ModelCls.is_3d_moe_weight is True
        assert patch_vllm_3d_moe_lora_flag("fake/model") is False

"""Unit tests for the vLLM <-> HF zero-copy weight-sharing primitives.

These cover the deterministic, GPU-free logic: vLLM-internals access, the
generic bnb state-dict extraction (HF naming, fused-qkv/gate-up splitting via
packed_modules_mapping, zero-copy aliasing, vocab truncation, raw-parameter
passthrough, tied vs untied lm_head, language_model nesting) and the
storage-aliasing self-check.

The dense ``build_shared_hf_model`` graft is covered here too: the fake vLLM
weights live on CPU and the module's hard-coded CUDA device is redirected by a
fixture, so grafting onto a real (tiny) HF Llama skeleton runs without a GPU.
Only the quantized build path (real bnb ``Linear4bit`` + CUDA ``QuantState``)
needs the GPU box. The multimodal tower-strip and LoRA-keep-resident patches
operate on plain module trees and are covered with fakes as well.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from agilerl.algorithms.core.llm_ops.vllm_weight_sharing import (
    _bnb_linear_kwargs,
    _expandable_segments_enabled,
    _graft_param,
    _materialise_meta_buffers,
    _MISSING,
    _move_buffers_to_device,
    _navigate,
    _navigate_safe,
    _override_to,
    _plain_tensor,
    _quant_target,
    _resolve_dtype,
    _rewrite_buffers,
    _set_submodule,
    _StrippedTower,
    _target_exists,
    _truncate_vocab,
    _walk_tower_holders,
    assert_shared_storage,
    build_shared_hf_model,
    extract_vllm_bnb_state_dict,
    get_vllm_internal_model,
    patch_vllm_lora_keep_resident,
    patch_vllm_standby_sleep_mode,
    patch_vllm_strip_multimodal_towers,
    prepare_shared_base_for_kbit_training,
)

_PACKED = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}


def _quant_linear(
    out_shards: list[int], in_features: int, bias: bool = False
) -> nn.Module:
    """nn.Module with a fused, bnb-packed-style ``.weight`` Parameter.

    The real tensor would be packed uint8; for slicing/aliasing logic any
    contiguous tensor whose dim-0 carries the per-shard offsets works. Each
    shard gets its own ``QuantState``-like object (only ``.shape`` is read).
    ``output_sizes`` mirrors vLLM's fused linear (used to split a fused bias,
    which is dense even on a quantized base).
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
            # A non-language tower (must NOT be extracted).
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


class TestHelpers:
    def test_resolve_dtype_variants(self):
        assert _resolve_dtype(torch.bfloat16) is torch.bfloat16
        assert _resolve_dtype("torch.float16") is torch.float16
        assert _resolve_dtype("bfloat16") is torch.bfloat16

    def test_expandable_segments_detection(self, monkeypatch):
        monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        assert _expandable_segments_enabled() is True
        monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
        assert _expandable_segments_enabled() is False
        monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
        assert _expandable_segments_enabled() is False

    def test_override_to_is_noop(self):
        m = torch.nn.Linear(2, 2)
        assert _override_to(m, "cuda") is m

    def test_navigate_indexed(self):
        model = nn.Module()
        model.layers = nn.ModuleList([nn.Linear(2, 2), nn.Linear(3, 3)])
        assert _navigate(model, "layers.1").in_features == 3

    def test_set_submodule_nested_and_indexed(self):
        model = nn.Module()
        model.block = nn.Module()
        model.block.layers = nn.ModuleList([nn.Linear(2, 2)])
        new_leaf = nn.Linear(4, 4)
        _set_submodule(model, "block.layers.0", new_leaf)
        assert model.block.layers[0] is new_leaf
        replacement = nn.Linear(1, 1)
        _set_submodule(model, "block", replacement)
        assert model.block is replacement


class TestGraftHelpers:
    def test_bnb_linear_kwargs_maps_config(self):
        # Only the three Linear4bit/Params4bit knobs are read; the quant storage
        # dtype goes through _resolve_dtype (accepts a "torch.x" string).
        bnb_config = SimpleNamespace(
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_quant_storage="torch.uint8",
        )
        assert _bnb_linear_kwargs(bnb_config) == {
            "compress_statistics": True,
            "quant_type": "nf4",
            "quant_storage": torch.uint8,
        }

    def test_truncate_vocab(self):
        w = torch.randn(36, 4)
        t = _truncate_vocab(w, 32)
        assert t.shape == (32, 4)
        assert t.data_ptr() == w.data_ptr()  # a view, not a copy
        assert _truncate_vocab(w, 36) is w  # exact size: untouched
        assert _truncate_vocab(w, 0) is w  # falsy vocab_size: untouched

    def test_navigate_safe_returns_missing_sentinel(self):
        model = nn.Module()
        model.layers = nn.ModuleList([nn.Linear(2, 3)])
        assert _navigate_safe(model, "layers.0") is model.layers[0]
        assert _navigate_safe(model, "layers.1") is _MISSING  # IndexError
        assert _navigate_safe(model, "blocks.0") is _MISSING  # AttributeError

    def test_plain_tensor_strips_subclass_keeping_storage(self):
        t = torch.randn(2, 2)
        assert _plain_tensor(t) is t
        p = nn.Parameter(torch.randn(2, 2))  # a Tensor subclass
        view = _plain_tensor(p)
        assert type(view) is torch.Tensor
        assert view.data_ptr() == p.data_ptr()

    def test_graft_param_aliases_and_freezes(self):
        model = nn.Module()
        model.layers = nn.ModuleList([nn.Linear(2, 2)])
        src = torch.randn(2, 2)
        _graft_param(model, "layers.0.weight", src)
        weight = model.layers[0].weight
        assert isinstance(weight, nn.Parameter)
        assert not weight.requires_grad
        assert weight.data_ptr() == src.data_ptr()
        # Subclass (Parameter) sources are stripped before wrapping, still aliased.
        bias_src = nn.Parameter(torch.randn(2), requires_grad=False)
        _graft_param(model, "layers.0.bias", bias_src)
        assert model.layers[0].bias.data_ptr() == bias_src.data_ptr()
        assert not model.layers[0].bias.requires_grad

    def test_target_exists(self):
        model = nn.Module()
        model.block = nn.Module()
        model.block.weight = nn.Parameter(torch.randn(2), requires_grad=False)
        assert _target_exists(model, "block.weight") is True
        assert _target_exists(model, "block.missing") is False

    def test_quant_target_handles_wrappers_and_missing(self):
        model = nn.Module()
        model.plain = nn.Linear(2, 2)
        clip = nn.Module()
        clip.linear = nn.Linear(2, 2)  # Gemma4ClippableLinear-style wrapper
        model.clip = clip
        odd = nn.Module()
        odd.linear = nn.Module()  # a ``linear`` attr that is not nn.Linear
        model.odd = odd
        assert _quant_target(model, "plain") == "plain"
        assert _quant_target(model, "clip") == "clip.linear"
        assert _quant_target(model, "odd") == "odd"
        # HF omits the module (e.g. k/v on a KV-shared layer): no graft target.
        assert _quant_target(model, "missing") is None


class TestBufferRewriteHelpers:
    def test_rewrite_buffers_replaces_and_skips_none_slots(self):
        m = nn.Module()
        m.register_buffer("keep", torch.zeros(2))
        m.register_buffer("swap", torch.ones(3))
        m.register_buffer("empty_slot", None)
        seen = []

        def fn(buf):
            seen.append(buf.shape[0])
            return torch.full_like(buf, 7.0) if buf.shape[0] == 3 else None

        assert _rewrite_buffers(m, fn) == 1
        assert torch.equal(m.swap, torch.full((3,), 7.0))
        assert torch.equal(m.keep, torch.zeros(2))  # fn -> None: untouched
        assert m.empty_slot is None
        assert sorted(seen) == [2, 3]  # None slots never reach the callback

    def test_materialise_meta_buffers_replaces_only_meta(self):
        m = nn.Module()
        m.register_buffer(
            "tower_stat", torch.empty(2, 3, device="meta", dtype=torch.float16)
        )
        real = torch.ones(2)
        m.register_buffer("real_stat", real)
        assert _materialise_meta_buffers(m, torch.device("cpu")) == 1
        assert m.tower_stat.device.type == "cpu"
        assert m.tower_stat.shape == (2, 3)
        assert m.tower_stat.dtype == torch.float16
        assert m.real_stat is real

    def test_move_buffers_to_device(self):
        # "meta" is the only second device type guaranteed on a CUDA-less
        # runner, so use it as the destination; the move logic is device-agnostic.
        m = nn.Module()
        m.register_buffer("movable", torch.zeros(2))
        m.register_buffer("already_meta", torch.empty(1, device="meta"))
        _move_buffers_to_device(m, torch.device("meta"))
        assert m.movable.device.type == "meta"
        assert m.already_meta.device.type == "meta"
        # Already on the target device type: left as the very same object.
        m2 = nn.Module()
        keep = torch.zeros(2)
        m2.register_buffer("keep", keep)
        _move_buffers_to_device(m2, torch.device("cpu"))
        assert m2.keep is keep


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
    def test_repr_and_falsy(self):
        tower = _StrippedTower("model.vision_tower")
        assert repr(tower) == "_StrippedTower('model.vision_tower')"
        # Falsy so ``if self.vision_tower:`` guards skip the stripped tower.
        assert bool(tower) is False

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
        assert tower._stripped_path == "audio_tower"  # the slot itself is readable
        with pytest.raises(AttributeError, match="attribute 'forward' accessed"):
            _ = tower.forward

    def test_getattr_guard_when_path_slot_unset(self):
        # An instance whose slot was never set (e.g. mid-unpickle) must raise a
        # plain AttributeError instead of recursing through __getattr__.
        tower = _StrippedTower.__new__(_StrippedTower)
        with pytest.raises(AttributeError):
            _ = tower._stripped_path


class TestWalkTowerHolders:
    def test_yields_top_level_and_nested_holders(self):
        outer = nn.Module()
        inner = nn.Module()
        inner.audio_tower = _plain(2, 2)
        outer.model = inner
        outer.vision_tower = _plain(2, 2)
        outer.embed_vision = None  # explicit None slot: skipped
        found = [
            (holder is outer, attr, path)
            for holder, attr, path in _walk_tower_holders(outer)
        ]
        assert found == [
            (True, "vision_tower", "vision_tower"),
            (False, "audio_tower", "model.audio_tower"),
        ]

    def test_skips_already_stripped_towers(self):
        model = nn.Module()
        model.vision_tower = _StrippedTower("vision_tower")
        model.audio_tower = _plain(2, 2)
        found = [path for _holder, _attr, path in _walk_tower_holders(model)]
        assert found == ["audio_tower"]

    def test_custom_tower_attrs(self):
        model = nn.Module()
        model.vision_tower = _plain(2, 2)
        model.my_tower = _plain(2, 2)
        found = [path for _h, _a, path in _walk_tower_holders(model, ("my_tower",))]
        assert found == ["my_tower"]


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


class TestExtractVllmBnbStateDict:
    def test_keys_and_quant_states(self):
        llm, config, internal = _make_fake(tie=True)
        out = extract_vllm_bnb_state_dict(llm, config)

        lp = "model.layers.0"
        for proj in (
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ):
            assert f"{lp}.{proj}.weight" in out
            assert f"{lp}.{proj}.weight.quant_state" in out

        for name in (
            "model.embed_tokens.weight",
            "model.norm.weight",
            f"{lp}.input_layernorm.weight",
            f"{lp}.post_attention_layernorm.weight",
            f"{lp}.self_attn.q_norm.weight",
            f"{lp}.self_attn.k_norm.weight",
            f"{lp}.altup_scale",  # raw (non-.weight) parameter carried 1:1
        ):
            assert name in out

        # qkv split maps shard 0/1/2 -> q/k/v quant states.
        qkv_states = internal.model.layers[0].self_attn.qkv_proj.weight.bnb_quant_state
        assert out[f"{lp}.self_attn.q_proj.weight.quant_state"] is qkv_states[0]
        assert out[f"{lp}.self_attn.k_proj.weight.quant_state"] is qkv_states[1]
        assert out[f"{lp}.self_attn.v_proj.weight.quant_state"] is qkv_states[2]
        gu_states = internal.model.layers[0].mlp.gate_up_proj.weight.bnb_quant_state
        assert out[f"{lp}.mlp.gate_proj.weight.quant_state"] is gu_states[0]
        assert out[f"{lp}.mlp.up_proj.weight.quant_state"] is gu_states[1]

    def test_zero_copy_aliasing_of_shards(self):
        llm, config, internal = _make_fake(tie=True)
        out = extract_vllm_bnb_state_dict(llm, config)
        qkv = internal.model.layers[0].self_attn.qkv_proj.weight
        offsets = qkv.bnb_shard_offsets
        assert (
            out["model.layers.0.self_attn.q_proj.weight"].data_ptr() == qkv.data_ptr()
        )
        k_view = out["model.layers.0.self_attn.k_proj.weight"]
        assert k_view.shape[0] == offsets[2] - offsets[1]
        assert k_view.data_ptr() == qkv[offsets[1] : offsets[2]].data_ptr()

    def test_fused_bias_split_to_subnames(self):
        # Qwen2-style attention biases: the fused qkv_proj.bias splits into q/k/v
        # via output_sizes (the bias is dense, not in bnb_shard_offsets).
        # Regression for the silently-dropped-bias bug.
        llm, config, internal = _make_fake(tie=True, bias=True)
        out = extract_vllm_bnb_state_dict(llm, config)
        lp = "model.layers.0.self_attn"
        bias = internal.model.layers[0].self_attn.qkv_proj.bias
        q, k, _v = internal.model.layers[0].self_attn.qkv_proj.output_sizes
        assert out[f"{lp}.q_proj.bias"].data_ptr() == bias.data_ptr()
        assert out[f"{lp}.k_proj.bias"].data_ptr() == bias[q : q + k].data_ptr()
        assert out[f"{lp}.v_proj.bias"].data_ptr() == bias[q + k :].data_ptr()
        # The fused name must NOT be emitted (HF has no qkv_proj).
        assert f"{lp}.qkv_proj.bias" not in out

    def test_vocab_truncation_and_tied_lm_head_not_emitted(self):
        llm, config, _internal = _make_fake(tie=True, vocab=32, vocab_pad=4)
        out = extract_vllm_bnb_state_dict(llm, config)
        # embed_tokens padded 36 -> truncated to vocab_size 32.
        assert out["model.embed_tokens.weight"].shape[0] == 32
        # tied head is handled by tie_weights() at build time, not extracted.
        assert "lm_head.weight" not in out

    def test_untied_lm_head_extracted_and_truncated(self):
        llm, config, _internal = _make_fake(tie=False, vocab=32, vocab_pad=4)
        out = extract_vllm_bnb_state_dict(llm, config)
        assert out["lm_head.weight"].shape[0] == 32

    def test_multimodal_uses_language_model_prefix_and_skips_towers(self):
        llm, config, _internal = _make_fake(tie=True, multimodal=True)
        out = extract_vllm_bnb_state_dict(llm, config)
        assert "model.language_model.layers.0.self_attn.q_proj.weight" in out
        assert "model.language_model.embed_tokens.weight" in out
        # vision tower must not be extracted (text-only scope).
        assert not any("vision_tower" in k for k in out)

    def test_rejects_model_without_decoder(self):
        llm = _wrap_llm(nn.Module())
        config = SimpleNamespace(vocab_size=8, num_hidden_layers=1)
        with pytest.raises(RuntimeError, match="Could not locate the language-model"):
            extract_vllm_bnb_state_dict(llm, config)

    def test_vllm_lora_adapter_params_skipped(self):
        # vLLM LoRA-wraps each linear; the adapter slots must never be
        # extracted (the raw-param passthrough would otherwise carry them 1:1).
        llm, config, internal = _make_fake(tie=True)
        qkv = internal.model.layers[0].self_attn.qkv_proj
        qkv.lora_a = nn.Parameter(torch.zeros(2, 8), requires_grad=False)
        out = extract_vllm_bnb_state_dict(llm, config)
        assert not any("lora" in key.lower() for key in out)

    def test_quant_fused_without_mapping_entry_raises(self):
        # A multi-shard quantized weight whose leaf has no
        # packed_modules_mapping entry cannot be given HF names.
        llm, config, internal = _make_fake(tie=True)
        internal.model.layers[0].self_attn.weird_proj = _quant_linear([4, 4], 8)
        with pytest.raises(RuntimeError, match="Cannot name the 2 fused shards"):
            extract_vllm_bnb_state_dict(llm, config)


# --- Dense (unquantized) fakes: vLLM exposes a fused projection's per-shard
# sizes via ``output_sizes`` (and no ``bnb_quant_state``).


def _dense_linear(
    out_shards: list[int], in_features: int, bias: bool = False
) -> nn.Module:
    """nn.Module with a fused, dense ``.weight`` and vLLM-style ``output_sizes``.

    Dense counterpart of ``_quant_linear``: no ``bnb_quant_state``; the per-shard
    boundaries come from ``output_sizes`` (as vLLM's ``QKVParallelLinear`` /
    ``MergedColumnParallelLinear`` expose them). A fused ``bias`` splits the same
    way.
    """
    m = nn.Module()
    m.weight = nn.Parameter(
        torch.randn(sum(out_shards), in_features), requires_grad=False
    )
    m.output_sizes = list(out_shards)
    if bias:
        m.bias = nn.Parameter(torch.randn(sum(out_shards)), requires_grad=False)
    return m


def _dense_decoder(
    vocab: int, hidden: int, n_layers: int, vocab_pad: int, bias: bool = False
) -> nn.Module:
    root = nn.Module()
    root.embed_tokens = _plain(vocab + vocab_pad, hidden)
    layers = []
    for _ in range(n_layers):
        layer = nn.Module()
        attn = nn.Module()
        attn.qkv_proj = _dense_linear(
            [hidden, hidden // 2, hidden // 2], hidden, bias=bias
        )
        attn.o_proj = _plain(hidden, hidden)  # non-fused: carried 1:1
        layer.self_attn = attn
        mlp = nn.Module()
        mlp.gate_up_proj = _dense_linear([hidden * 2, hidden * 2], hidden)
        mlp.down_proj = _plain(hidden, hidden * 2)
        layer.mlp = mlp
        layer.input_layernorm = _plain(hidden)
        layer.post_attention_layernorm = _plain(hidden)
        layers.append(layer)
    root.layers = nn.ModuleList(layers)
    root.norm = _plain(hidden)
    return root


class _FakeDenseInternal(nn.Module):
    packed_modules_mapping = _PACKED

    def __init__(self, *, tie, vocab=32, hidden=8, n_layers=1, vocab_pad=4, bias=False):
        super().__init__()
        self.model = _dense_decoder(vocab, hidden, n_layers, vocab_pad, bias=bias)
        if not tie:
            self.lm_head = _plain(vocab + vocab_pad, hidden)


def _make_dense_fake(*, tie=True, vocab=32, vocab_pad=4, bias=False):
    internal = _FakeDenseInternal(tie=tie, vocab=vocab, vocab_pad=vocab_pad, bias=bias)
    llm = _wrap_llm(internal)
    config = SimpleNamespace(
        vocab_size=vocab, num_hidden_layers=1, tie_word_embeddings=tie
    )
    return llm, config, internal


class TestExtractDenseStateDict:
    def test_dense_fused_split_to_hf_subnames(self):
        llm, config, _internal = _make_dense_fake(tie=True)
        out = extract_vllm_bnb_state_dict(llm, config)
        lp = "model.layers.0"
        for proj in (
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ):
            assert f"{lp}.{proj}.weight" in out
            # Dense weights carry no quant_state.
            assert f"{lp}.{proj}.weight.quant_state" not in out
        for name in (
            "model.embed_tokens.weight",
            "model.norm.weight",
            f"{lp}.input_layernorm.weight",
            f"{lp}.post_attention_layernorm.weight",
        ):
            assert name in out

    def test_dense_zero_copy_aliasing_of_shards(self):
        llm, config, internal = _make_dense_fake(tie=True)
        out = extract_vllm_bnb_state_dict(llm, config)
        qkv = internal.model.layers[0].self_attn.qkv_proj.weight
        hidden = qkv.shape[1]
        q, k, v = hidden, hidden // 2, hidden // 2
        # q starts at row 0; k at q; v at q+k (split via output_sizes).
        assert (
            out["model.layers.0.self_attn.q_proj.weight"].data_ptr() == qkv.data_ptr()
        )
        k_view = out["model.layers.0.self_attn.k_proj.weight"]
        assert k_view.shape[0] == k
        assert k_view.data_ptr() == qkv[q : q + k].data_ptr()
        v_view = out["model.layers.0.self_attn.v_proj.weight"]
        assert v_view.shape[0] == v
        assert v_view.data_ptr() == qkv[q + k : q + k + v].data_ptr()

    def test_dense_fused_bias_split(self):
        # A dense fused qkv_proj.bias splits into q/k/v via output_sizes, aliased.
        llm, config, internal = _make_dense_fake(tie=True, bias=True)
        out = extract_vllm_bnb_state_dict(llm, config)
        lp = "model.layers.0.self_attn"
        bias = internal.model.layers[0].self_attn.qkv_proj.bias
        q, k, _v = internal.model.layers[0].self_attn.qkv_proj.output_sizes
        assert out[f"{lp}.q_proj.bias"].data_ptr() == bias.data_ptr()
        assert out[f"{lp}.k_proj.bias"].data_ptr() == bias[q : q + k].data_ptr()
        assert out[f"{lp}.v_proj.bias"].data_ptr() == bias[q + k :].data_ptr()
        assert f"{lp}.qkv_proj.bias" not in out

    def test_dense_non_fused_carried_1to1(self):
        llm, config, internal = _make_dense_fake(tie=True)
        out = extract_vllm_bnb_state_dict(llm, config)
        o = internal.model.layers[0].self_attn.o_proj.weight
        assert out["model.layers.0.self_attn.o_proj.weight"].data_ptr() == o.data_ptr()
        down = internal.model.layers[0].mlp.down_proj.weight
        assert out["model.layers.0.mlp.down_proj.weight"].data_ptr() == down.data_ptr()

    def test_dense_vocab_truncation_and_untied_lm_head(self):
        llm, config, _internal = _make_dense_fake(tie=False, vocab=32, vocab_pad=4)
        out = extract_vllm_bnb_state_dict(llm, config)
        assert out["model.embed_tokens.weight"].shape[0] == 32
        assert out["lm_head.weight"].shape[0] == 32

    def test_dense_fused_split_raises_without_output_sizes(self):
        llm, config, internal = _make_dense_fake(tie=True)
        # Drop output_sizes so the fused qkv weight cannot be named.
        del internal.model.layers[0].self_attn.qkv_proj.output_sizes
        with pytest.raises(RuntimeError, match="Cannot split the fused parameter"):
            extract_vllm_bnb_state_dict(llm, config)

    def test_dense_non_fused_bias_carried_1to1(self):
        # o_proj is not a packed module, so its bias passes through unsplit.
        llm, config, internal = _make_dense_fake(tie=True)
        attn = internal.model.layers[0].self_attn
        attn.o_proj = _dense_linear([8], 8, bias=True)
        out = extract_vllm_bnb_state_dict(llm, config)
        key = "model.layers.0.self_attn.o_proj.bias"
        assert out[key].data_ptr() == attn.o_proj.bias.data_ptr()
        assert out[key].shape == (8,)


# --- Dense ``build_shared_hf_model``: graft the shared fakes onto a real
# (tiny) HF Llama skeleton. The function targets ``torch.device("cuda", ...)``
# because in production the shared base lives in vLLM's GPU pool; the fixture
# below redirects the module-global ``torch`` to CPU, where the fake vLLM
# weights live, so the graft logic (which is device-agnostic) runs the same on
# a CUDA-less CI runner and on a GPU box. The quantized build path needs a real
# bnb ``Linear4bit`` + CUDA ``QuantState`` and is exercised on the GPU box.


class _CpuTorchProxy:
    """Stand-in for vllm_weight_sharing's module-global ``torch``.

    Redirects ``torch.device(...)`` to CPU and stubs the ``torch.cuda`` calls;
    everything else delegates to the real torch.
    """

    cuda = SimpleNamespace(
        current_device=lambda: 0,
        empty_cache=lambda: None,
        is_available=lambda: False,
    )

    @staticmethod
    def device(*args, **kwargs):
        return torch.device("cpu")

    def __getattr__(self, name):
        return getattr(torch, name)


@pytest.fixture
def cpu_shared_device(monkeypatch):
    import agilerl.algorithms.core.llm_ops.vllm_weight_sharing as vws

    monkeypatch.setattr(vws, "torch", _CpuTorchProxy())


def _tiny_llama_config(*, tie: bool):
    """Real HF config whose skeleton matches ``_FakeLlamaVllmInternal``.

    q out = 2 heads * head_dim 4 = 8; k/v out = 1 kv-head * 4 = 4;
    gate/up = intermediate 16; attention biases on; rotary ``inv_freq`` of 2.
    """
    from transformers import LlamaConfig

    return LlamaConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=16,
        tie_word_embeddings=tie,
        attention_bias=True,
    )


class _FakeLlamaVllmInternal(nn.Module):
    """Dense vLLM fake mirroring the tiny HF Llama of ``_tiny_llama_config``.

    Mirrors what vLLM exposes for a Llama-family base: fused qkv/gate_up with
    ``output_sizes``, attention biases, padded vocab, plus a loaded rotary
    ``inv_freq`` buffer the graft must copy over the skeleton's computed one
    (and buffers that must be ignored: a LoRA-slot name and one with no HF
    counterpart). ``vllm_only_extras`` adds params HF Llama does not have
    (q/k norms, a raw per-layer scale) to exercise the skip-with-warning path.
    """

    packed_modules_mapping = _PACKED

    def __init__(self, *, tie, vllm_only_extras=False, vocab=32, hidden=8, vocab_pad=4):
        super().__init__()
        root = nn.Module()
        root.embed_tokens = _plain(vocab + vocab_pad, hidden)
        layer = nn.Module()
        attn = nn.Module()
        attn.qkv_proj = _dense_linear(
            [hidden, hidden // 2, hidden // 2], hidden, bias=True
        )
        attn.o_proj = _dense_linear([hidden], hidden, bias=True)
        layer.self_attn = attn
        mlp = nn.Module()
        mlp.gate_up_proj = _dense_linear([hidden * 2, hidden * 2], hidden)
        mlp.down_proj = _plain(hidden, hidden * 2)
        layer.mlp = mlp
        layer.input_layernorm = _plain(hidden)
        layer.post_attention_layernorm = _plain(hidden)
        if vllm_only_extras:
            attn.q_norm = _plain(hidden // 2)
            attn.k_norm = _plain(hidden // 2)
            attn.q_norm.register_buffer("running_stat", torch.ones(1))
            layer.altup_scale = nn.Parameter(torch.randn(hidden), requires_grad=False)
        root.layers = nn.ModuleList([layer])
        root.norm = _plain(hidden)
        rotary = nn.Module()
        # A loaded (checkpoint) buffer: must overwrite HF's computed default.
        rotary.register_buffer("inv_freq", torch.tensor([3.0, 4.0]))
        # LoRA-slot buffers must never be grafted.
        rotary.register_buffer("lora_inv_freq", torch.zeros(2))
        root.rotary_emb = rotary
        # A buffer with no HF counterpart: silently skipped.
        root.register_buffer("vllm_only_stat", torch.ones(1))
        self.model = root
        if not tie:
            self.lm_head = _plain(vocab + vocab_pad, hidden)


def _build_dense_shared(*, tie=True, vllm_only_extras=False, **kwargs):
    internal = _FakeLlamaVllmInternal(tie=tie, vllm_only_extras=vllm_only_extras)
    llm = _wrap_llm(internal)
    model = build_shared_hf_model(
        llm,
        _tiny_llama_config(tie=tie),
        torch.float32,
        None,  # dense base: no BitsAndBytesConfig
        attn_implementation="eager",
        **kwargs,
    )
    return llm, internal, model


@pytest.mark.usefixtures("cpu_shared_device")
class TestBuildSharedHfModelDense:
    def test_share_towers_not_implemented(self):
        with pytest.raises(NotImplementedError, match="share_towers"):
            build_shared_hf_model(
                SimpleNamespace(),
                SimpleNamespace(),
                torch.float32,
                None,
                share_towers=True,
            )

    def test_dense_build_aliases_vllm_storage(self):
        llm, internal, model = _build_dense_shared(tie=True)
        assert type(model).__name__ == "LlamaForCausalLM"
        layer = model.model.layers[0]
        fake_attn = internal.model.layers[0].self_attn
        qkv_w, qkv_b = fake_attn.qkv_proj.weight, fake_attn.qkv_proj.bias
        q, k, _v = fake_attn.qkv_proj.output_sizes
        # Fused qkv split into the skeleton's q/k/v, weights AND biases aliased.
        assert layer.self_attn.q_proj.weight.data_ptr() == qkv_w.data_ptr()
        assert layer.self_attn.k_proj.weight.data_ptr() == qkv_w[q : q + k].data_ptr()
        assert layer.self_attn.v_proj.weight.data_ptr() == qkv_w[q + k :].data_ptr()
        assert layer.self_attn.k_proj.bias.data_ptr() == qkv_b[q : q + k].data_ptr()
        assert (
            layer.self_attn.o_proj.bias.data_ptr() == fake_attn.o_proj.bias.data_ptr()
        )
        gate_up = internal.model.layers[0].mlp.gate_up_proj.weight
        assert layer.mlp.gate_proj.weight.data_ptr() == gate_up.data_ptr()
        assert layer.mlp.up_proj.weight.data_ptr() == gate_up[16:].data_ptr()
        # Padded vocab rows dropped; the kept rows still alias vLLM's embedding.
        embed = model.model.embed_tokens.weight
        assert embed.shape == (32, 8)
        assert embed.data_ptr() == internal.model.embed_tokens.weight.data_ptr()
        # Tied head points at the grafted embedding.
        assert model.lm_head.weight.data_ptr() == embed.data_ptr()
        # The module's own self-check agrees.
        assert_shared_storage(llm, model)

    def test_dense_build_is_frozen_eval_with_noop_to(self):
        _llm, _internal, model = _build_dense_shared(tie=True)
        assert all(p.device.type != "meta" for p in model.parameters())
        assert all(not p.requires_grad for p in model.parameters())
        assert model.training is False
        # ``.to`` is frozen so accelerate can't re-cast (un-alias) the base.
        assert model.to("cuda") is model
        assert model.to(torch.float16) is model
        assert model.model.embed_tokens.weight.dtype == torch.float32

    def test_dense_build_grafts_loaded_vllm_buffers(self):
        _llm, internal, model = _build_dense_shared(tie=True)
        rotary = model.model.rotary_emb
        # vLLM's loaded value replaced the skeleton's computed default...
        assert torch.equal(rotary.inv_freq, torch.tensor([3.0, 4.0]))
        # ...by reference (detach), not by copy.
        assert (
            rotary.inv_freq.data_ptr() == internal.model.rotary_emb.inv_freq.data_ptr()
        )
        # LoRA-slot / vLLM-only buffers must not appear on the HF side.
        assert not hasattr(rotary, "lora_inv_freq")
        assert not hasattr(model.model, "vllm_only_stat")

    def test_dense_build_warns_and_skips_vllm_only_params(self):
        # q/k norms + the raw altup scale have no HF Llama target: skipped with
        # one summary warning rather than grafted onto a missing module.
        with pytest.warns(UserWarning, match="skipped 3 vLLM params with no HF"):
            _llm, _internal, model = _build_dense_shared(
                tie=True, vllm_only_extras=True
            )
        assert not hasattr(model.model.layers[0].self_attn, "q_norm")
        assert not hasattr(model.model.layers[0], "altup_scale")

    def test_dense_build_untied_lm_head_aliases_vllm(self):
        llm, internal, model = _build_dense_shared(tie=False)
        head = model.lm_head.weight
        assert head.shape == (32, 8)  # padded rows truncated
        assert head.data_ptr() == internal.lm_head.weight.data_ptr()
        assert head.data_ptr() != model.model.embed_tokens.weight.data_ptr()
        assert_shared_storage(llm, model)

    def test_dense_build_forward_runs_on_shared_weights(self):
        _llm, _internal, model = _build_dense_shared(tie=True)
        with torch.no_grad():
            out = model(input_ids=torch.tensor([[1, 2, 3]]))
        assert out.logits.shape == (1, 3, 32)
        assert torch.isfinite(out.logits).all()

    def test_dense_build_with_value_head(self):
        from agilerl.utils.ppo_value_head import AutoModelForCausalLMWithValueHead

        internal = _FakeLlamaVllmInternal(tie=True)
        llm = _wrap_llm(internal)
        wrapped = build_shared_hf_model(
            llm,
            _tiny_llama_config(tie=True),
            torch.float32,
            None,
            attn_implementation="eager",
            add_value_head=True,
        )
        assert isinstance(wrapped, AutoModelForCausalLMWithValueHead)
        # Trainer-only head: trainable, in the compute dtype, not vLLM-aliased.
        assert all(p.requires_grad for p in wrapped.v_head.parameters())
        assert wrapped.v_head.summary.weight.dtype == torch.float32
        # The wrapped base is still the shared one.
        assert_shared_storage(llm, wrapped.pretrained_model)

    def test_dense_build_raises_on_ungrafted_language_param(self):
        internal = _FakeLlamaVllmInternal(tie=True)
        del internal.model.norm  # the vLLM walk now misses model.norm.weight
        llm = _wrap_llm(internal)
        with pytest.raises(RuntimeError, match="language params un-grafted"):
            build_shared_hf_model(
                llm,
                _tiny_llama_config(tie=True),
                torch.float32,
                None,
                attn_implementation="eager",
            )


class _AliasLeaf(nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)


class _AliasModel(nn.Module):
    """Minimal HF-shaped model whose weights alias the extracted tensors."""

    def __init__(self, shared, config):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = _AliasLeaf(shared["model.embed_tokens.weight"])
        self.model.norm = _AliasLeaf(shared["model.norm.weight"])
        self.config = config


def _alias_tree_from_shared(shared, config):
    """Module tree whose names mirror the shared ``.weight`` keys, each leaf
    aliasing the extracted tensor (digit parts become submodules named "0")."""
    root = nn.Module()
    root.config = config
    for key, val in shared.items():
        if not key.endswith(".weight"):
            continue
        *parents, _leaf = key.split(".")
        cur = root
        for part in parents:
            nxt = getattr(cur, part, None)
            if not isinstance(nxt, nn.Module):
                nxt = nn.Module()
                setattr(cur, part, nxt)
            cur = nxt
        cur.weight = nn.Parameter(val, requires_grad=False)
    return root


class TestAssertSharedStorage:
    def test_passes_when_aliased(self):
        llm, config, _internal = _make_fake(tie=True)
        shared = extract_vllm_bnb_state_dict(llm, config)
        model = _AliasModel(shared, config)
        assert_shared_storage(llm, model)

    def test_raises_on_copy(self):
        llm, config, _internal = _make_fake(tie=True)
        shared = extract_vllm_bnb_state_dict(llm, config)
        model = _AliasModel(shared, config)
        model.model.norm.weight = nn.Parameter(
            shared["model.norm.weight"].clone(), requires_grad=False
        )
        with pytest.raises(RuntimeError, match="do not alias"):
            assert_shared_storage(llm, model)

    def test_skips_weightless_modules_and_checks_nested_linear_holder(self):
        llm, config, _internal = _make_dense_fake(tie=True)
        shared = extract_vllm_bnb_state_dict(llm, config)
        model = _alias_tree_from_shared(shared, config)
        # A name-matching module with no weight (and no .linear): skipped.
        model.model.embed_tokens = nn.Module()
        # A clipping-style wrapper: the real weight nested under ``.linear``.
        attn = getattr(model.model.layers, "0").self_attn
        wrapper = nn.Module()
        wrapper.linear = nn.Module()
        wrapper.linear.weight = nn.Parameter(
            shared["model.layers.0.self_attn.q_proj.weight"], requires_grad=False
        )
        attn.q_proj = wrapper
        assert_shared_storage(llm, model)

    def test_spot_check_stops_after_eight_aliased_params(self):
        llm, config, _internal = _make_dense_fake(tie=True)
        shared = extract_vllm_bnb_state_dict(llm, config)
        model = _alias_tree_from_shared(shared, config)
        # Walk order verifies 8 aliased params (embed + q/k/v/o + gate/up/down)
        # and stops: a copy further down is deliberately out of spot-check
        # reach, documenting that this is a spot check, not an exhaustive scan.
        model.model.norm.weight = nn.Parameter(
            shared["model.norm.weight"].clone(), requires_grad=False
        )
        assert_shared_storage(llm, model)

    def test_raises_when_no_shared_params_found(self):
        llm, config, _internal = _make_dense_fake(tie=True)
        model = nn.Module()
        model.config = config  # no module names match any shared key
        with pytest.raises(RuntimeError, match="found no shared parameters"):
            assert_shared_storage(llm, model)


class _GCBaseModel(nn.Module):
    """Minimal HF-ish base: input embeddings + recorded checkpointing calls.

    A plain ``nn.Module``, so it has NO ``enable_input_require_grads`` (that is
    a transformers ``PreTrainedModel`` method) — the reentrant prep must fall
    back to hooking the input embeddings.
    """

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(4, 2)
        self.gc_kwargs = None

    def get_input_embeddings(self):
        return self.embed

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.gc_kwargs = gradient_checkpointing_kwargs


class _GCModelWithEnable(_GCBaseModel):
    def __init__(self):
        super().__init__()
        self.enable_called = False

    def enable_input_require_grads(self):
        self.enable_called = True


class TestPrepareSharedBaseForKbit:
    def test_freezes_without_fp32_upcast(self):
        # The shared base must stay bf16 (aliased to vLLM) and frozen — the
        # stock peft kbit-prep would upcast it to fp32 (defeating sharing).
        model = nn.Linear(4, 4).to(torch.bfloat16)
        for p in model.parameters():
            p.requires_grad = True
        out = prepare_shared_base_for_kbit_training(
            model, use_gradient_checkpointing=False
        )
        assert out is model
        assert all(not p.requires_grad for p in model.parameters())
        assert all(p.dtype == torch.bfloat16 for p in model.parameters())

    def test_enables_gradient_checkpointing_no_reentrant_hook(self):
        from unittest.mock import MagicMock

        model = MagicMock()
        model.named_parameters.return_value = []
        prepare_shared_base_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        model.gradient_checkpointing_enable.assert_called_once()
        # use_reentrant=False -> the input-require-grads hack is skipped.
        model.enable_input_require_grads.assert_not_called()

    def test_reentrant_prefers_enable_input_require_grads(self):
        model = _GCModelWithEnable()
        # Defaults: checkpointing on, kwargs {} -> use_reentrant defaults True.
        out = prepare_shared_base_for_kbit_training(model)
        assert out is model
        assert model.enable_called is True
        assert model.gc_kwargs == {}
        assert not model.embed.weight.requires_grad

    def test_reentrant_fallback_hooks_input_embeddings(self):
        # Without ``enable_input_require_grads`` the forward hook must force
        # the (frozen) embedding output to require grad, so the reentrant
        # checkpointed graph can still backprop into the LoRA adapters.
        model = _GCBaseModel()
        prepare_shared_base_for_kbit_training(
            model, gradient_checkpointing_kwargs={"use_reentrant": True}
        )
        assert model.gc_kwargs == {"use_reentrant": True}
        assert not model.embed.weight.requires_grad
        out = model.embed(torch.tensor([0, 1]))
        assert out.requires_grad


class TestResolveAttnImplementation:
    def test_respects_explicit_choice(self):
        from agilerl.utils.llm_utils import resolve_attn_implementation

        assert resolve_attn_implementation("flex_attention") == "flex_attention"
        assert resolve_attn_implementation("eager") == "eager"

    def test_explicit_choice_wins_over_auto(self):
        from agilerl.utils.llm_utils import resolve_attn_implementation

        # Explicit values are returned unchanged, never auto-resolved.
        assert resolve_attn_implementation("flex_attention") == "flex_attention"
        assert resolve_attn_implementation("sdpa") == "sdpa"

    def test_auto_prefers_flash_when_available_else_sdpa(self, monkeypatch):
        import importlib.util

        from agilerl.utils.llm_utils import resolve_attn_implementation

        real = importlib.util.find_spec

        def fake_find_spec(name, *a, **k):
            if name == "flash_attn":
                return object()  # pretend installed
            return real(name, *a, **k)

        monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
        assert resolve_attn_implementation() == "flash_attention_2"
        assert resolve_attn_implementation("auto") == "flash_attention_2"

        def no_flash(name, *a, **k):
            if name == "flash_attn":
                return None
            return real(name, *a, **k)

        monkeypatch.setattr(importlib.util, "find_spec", no_flash)
        assert resolve_attn_implementation() == "sdpa"


class TestStandbyPatchSafeWithoutVllm:
    def test_noop_when_vllm_absent(self):
        # vLLM may be absent in this environment; the patch must no-op rather
        # than raise. (When vLLM is present it monkeypatches CuMemAllocator.)
        patch_vllm_standby_sleep_mode()


class TestVllmConfigWeightSharingDeprecated:
    """weight_sharing is deprecated/ignored in VLLMConfig.

    Colocated vLLM always shares (the LLM algorithm enforces it); VLLMConfig
    retains the field for API compatibility but no longer validates it.
    """

    def test_default_is_none(self):
        from agilerl.utils.algo_utils import VLLMConfig

        assert VLLMConfig().weight_sharing is None

    def test_explicit_values_accepted_without_raising(self):
        from agilerl.utils.algo_utils import VLLMConfig

        # No sleep_mode requirement, no quantization/reload validation: every
        # value constructs and round-trips unchanged.
        assert VLLMConfig(weight_sharing=True, sleep_mode=False).weight_sharing is True
        assert VLLMConfig(weight_sharing=False).weight_sharing is False
        assert (
            VLLMConfig(
                weight_sharing=True, quantization="bitsandbytes", sleep_mode=True
            ).weight_sharing
            is True
        )

    def test_no_reload_warning_emitted(self):
        # The old bnb "cannot reload in-place" warning was removed.
        import warnings as _w

        from agilerl.utils.algo_utils import VLLMConfig

        with _w.catch_warnings(record=True) as rec:
            _w.simplefilter("always")
            VLLMConfig(
                quantization="bitsandbytes", sleep_mode=True, weight_sharing=False
            )
        assert not any("reload" in str(x.message) for x in rec)

"""Unit tests for the vLLM <-> HF zero-copy weight-sharing primitives.

These cover the deterministic, GPU-free logic: vLLM-internals access, the
generic bnb state-dict extraction (HF naming, fused-qkv/gate-up splitting via
packed_modules_mapping, zero-copy aliasing, vocab truncation, raw-parameter
passthrough, tied vs untied lm_head, language_model nesting) and the
storage-aliasing self-check.

The full ``build_shared_hf_model`` graft needs CUDA + a real bitsandbytes
4-bit base, so it is exercised on the GPU box, not here.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from agilerl.algorithms.core.llm_ops.vllm_weight_sharing import (
    _expandable_segments_enabled,
    _navigate,
    _override_to,
    _resolve_dtype,
    _set_submodule,
    assert_shared_storage,
    extract_vllm_bnb_state_dict,
    get_vllm_internal_model,
    patch_vllm_standby_sleep_mode,
    prepare_shared_base_for_kbit_training,
)

_PACKED = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}


def _quant_linear(out_shards: list[int], in_features: int) -> nn.Module:
    """nn.Module with a fused, bnb-packed-style ``.weight`` Parameter.

    The real tensor would be packed uint8; for slicing/aliasing logic any
    contiguous tensor whose dim-0 carries the per-shard offsets works. Each
    shard gets its own ``QuantState``-like object (only ``.shape`` is read).
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
    return m


def _plain(out_features: int, in_features: int | None = None) -> nn.Module:
    m = nn.Module()
    shape = (out_features,) if in_features is None else (out_features, in_features)
    m.weight = nn.Parameter(torch.randn(*shape), requires_grad=False)
    return m


def _decoder(vocab: int, hidden: int, n_layers: int, vocab_pad: int) -> nn.Module:
    root = nn.Module()
    root.embed_tokens = _plain(vocab + vocab_pad, hidden)
    layers = []
    for _ in range(n_layers):
        layer = nn.Module()
        attn = nn.Module()
        attn.qkv_proj = _quant_linear([hidden, hidden // 2, hidden // 2], hidden)
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

    def __init__(self, *, tie, multimodal, vocab=32, hidden=8, n_layers=1, vocab_pad=4):
        super().__init__()
        decoder = _decoder(vocab, hidden, n_layers, vocab_pad)
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


def _make_fake(*, tie=True, multimodal=False, vocab=32, vocab_pad=4):
    internal = _FakeInternal(
        tie=tie, multimodal=multimodal, vocab=vocab, vocab_pad=vocab_pad
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


class TestResolveAttnImplementation:
    def test_respects_explicit_choice(self):
        from agilerl.utils.llm_utils import resolve_attn_implementation

        assert resolve_attn_implementation("flex_attention") == "flex_attention"
        assert resolve_attn_implementation("eager") == "eager"

    def test_env_var_overrides_auto_but_not_explicit(self, monkeypatch):
        from agilerl.utils.llm_utils import resolve_attn_implementation

        monkeypatch.setenv("AGILERL_ATTN_IMPLEMENTATION", "flex_attention")
        assert resolve_attn_implementation() == "flex_attention"
        assert resolve_attn_implementation("auto") == "flex_attention"
        # An explicit caller value still wins over the env var.
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


class TestVllmConfigBnbWithoutWeightSharingWarning:
    """bnb rollout + sleep without weight_sharing hits the broken bnb reload."""

    def _warns_reload(self, **kwargs) -> bool:
        from agilerl.utils.algo_utils import VLLMConfig

        with pytest.warns(UserWarning) as record:
            VLLMConfig(**kwargs)
        return any("reload" in str(w.message) for w in record)

    def test_bnb_sleep_without_sharing_warns(self):
        assert self._warns_reload(quantization="bitsandbytes", sleep_mode=True)

    def test_bnb_sleep_with_sharing_does_not_warn_reload(self):
        assert not self._warns_reload(
            quantization="bitsandbytes", sleep_mode=True, weight_sharing=True
        )

    def test_non_bnb_sleep_does_not_warn_reload(self):
        assert not self._warns_reload(quantization="awq", sleep_mode=True)

    def test_bnb_without_sleep_does_not_warn_reload(self):
        from agilerl.utils.algo_utils import VLLMConfig

        import warnings as _w

        with _w.catch_warnings(record=True) as rec:
            _w.simplefilter("always")
            VLLMConfig(quantization="bitsandbytes", sleep_mode=False)
        assert not any("reload" in str(x.message) for x in rec)

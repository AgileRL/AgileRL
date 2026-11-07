# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Decoder, layer-mix, MoE and tower fields read from a raw HF ``config.json``.

Pure functions of config dicts. No torch, no I/O.
"""

from __future__ import annotations

from typing import Any, Literal

from agilerl.arena.memory.resolve.attention import FAMILY_ATTN_IMPLEMENTATIONS

# Model types whose trainer RMSNorm keeps an fp32 copy of each norm input for
# backward under Liger too: Gemma 4's norm upcasts with ``hidden_states.float()``.
# Without Liger every model type does (:func:`liger_kernel_arch`).
FP32_RMS_NORM_MODEL_TYPES = frozenset({"gemma4_text"})

# Attention norms that upcast Q to fp32 inside the block (Qwen3/3.5 ``q_norm``,
# DeepSeek-V4 ``q_b_norm``).
QK_NORM_MODEL_TYPES = frozenset(
    {
        "qwen3",
        "qwen3_moe",
        "qwen3_5",
        "qwen3_5_text",
        "qwen3_5_moe",
        "qwen3_5_moe_text",
        "deepseek_v4",
    }
)

# Model types whose packed experts take the trainer's self-routing grouped-GEMM
# path, which runs experts in row chunks and writes straight into the layer output.
CHUNKED_ROUTED_EXPERT_MODEL_TYPES = frozenset(
    {"nemotron_h", "nemotron_h_omni", "qwen3_moe"}
)

# Model types whose packed experts take the self-routing LoRA wrapper, which
# recomputes the expert up-projection in backward outside a checkpointed block.
ROUTED_EXPERT_RECOMPUTE_MODEL_TYPES = frozenset(
    {"granitemoe", "granitemoehybrid", "nemotron_h", "nemotron_h_omni", "qwen3_moe"}
)

# Model types whose packed experts use gpt-oss's transposed layout
# (``x @ W + b``, clamped interleaved SwiGLU). AgileRL runs them on the
# grouped expert GEMMs and adds their split LoRA delta unchunked.
TRANSPOSED_EXPERT_MODEL_TYPES = frozenset({"gpt_oss"})

# Model types whose Mamba mixer runs HF's unfused path on devices of
# ``UNFUSED_MAMBA_SCAN_CAPABILITY``, where the fused combined kernel returns NaN.
UNFUSED_MAMBA_SCAN_MODEL_TYPES = frozenset({"nemotron_h", "nemotron_h_omni"})

# Model types whose q/k/v projections carry a bias their config does not flag.
QKV_BIAS_MODEL_TYPES = frozenset({"qwen2"})
# Model types whose every attention layer learns one sink logit per query head.
ATTENTION_SINK_MODEL_TYPES = frozenset({"deepseek_v4", "gpt_oss"})
# Families whose training TP plan splits a tied token embedding and head by column.
TIED_EMBEDDING_TENSOR_PARALLEL_MODEL_TYPES = frozenset(
    {"gemma4", "gemma4_text", "granitemoe", "granitemoehybrid"}
)
# Model types whose attention ``q_proj`` also projects a per-head output gate.
ATTENTION_OUTPUT_GATE_MODEL_TYPES = frozenset(
    {"qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"}
)
# MoE blocks that end on the shared expert plus an add, so a block recompute
# never reaches the shared expert's tensor-parallel reduce.
TRAILING_SHARED_EXPERT_MODEL_TYPES = frozenset({"nemotron_h", "nemotron_h_omni"})
# Dense and shared MLPs whose gate and up projections are one ``input_linear``.
FUSED_GATE_UP_MLP_MODEL_TYPES = frozenset({"granitemoehybrid"})

# Vision patches in the image vLLM encodes when profiling a model that keeps
# its towers, keyed by outer config ``model_type``. Qwen3.5/3.6: the
# preprocessor's 16,777,216-pixel longest edge over 16x16 patches. Families
# absent here have no measured profile and add no encoder term.
VISION_PROFILE_PATCHES_BY_MODEL_TYPE: dict[str, int] = {
    "qwen3_5": 65536,
    "qwen3_5_moe": 65536,
}

VisionResize = Literal["none", "merge_grid", "pooled_budget", "dynamic_tiles"]

# How each family's image processor sizes an image's patch grid, keyed by
# outer config ``model_type``. ``merge_grid``: Qwen2-VL ``smart_resize`` to
# multiples of patch x merge. ``pooled_budget``: Gemma4's aspect-preserving
# fit under soft tokens x pool^2 patches. ``dynamic_tiles``: Nemotron RADIO
# dynamic resolution between its min and max patches.
VISION_RESIZE_BY_MODEL_TYPE: dict[str, VisionResize] = {
    "qwen3_5": "merge_grid",
    "qwen3_5_moe": "merge_grid",
    "gemma4": "pooled_budget",
    "nemotron_h_omni": "dynamic_tiles",
}

# Families whose VisionFeatureCache keeps frozen tower outputs through a learn
# step. A LoRA-trained tower runs with autograd and bypasses the cache.
VISION_FEATURE_CACHE_MODEL_TYPES: frozenset[str] = frozenset(
    {"qwen3_5", "qwen3_5_moe", "gemma4"}
)

# Tower config model types whose encoder MLP has gate, up and down projections.
GATED_MLP_TOWER_MODEL_TYPES = frozenset({"gemma4_vision", "glm5_next_vision"})
# Tower config model types that keep one position table per image axis.
AXIS_POSITION_TOWER_MODEL_TYPES = frozenset({"gemma4_vision"})
# Conformer audio towers. Each block adds a second macaron feed-forward, a
# relative-position key projection and a light conv's h->2h and h->h linears.
CONFORMER_TOWER_MODEL_TYPES = frozenset({"gemma4_audio"})
# Outer config model types whose trainer and vLLM classes build the language
# model alone, so their vision and audio configs add no parameters.
TEXT_ONLY_MODEL_TYPES = frozenset({"mimo_v2"})

# Encoder config keys and the vLLM modalities each tower serves. vLLM skips
# building a tower once every one of its modalities is stripped.
TOWER_MODALITIES: dict[str, frozenset[str]] = {
    "vision_config": frozenset({"image", "video"}),
    "audio_config": frozenset({"audio"}),
}


def first_int(cfg: dict[str, Any], *keys: str, default: int = 0) -> int:
    """First non-empty integer among HF alias keys for the same field."""
    for key in keys:
        value = cfg.get(key)
        if value:
            return int(value)
    return default


def layer_type_list(text_cfg: dict[str, Any]) -> list[str]:
    """Per-layer type list from ``layer_types``, else ``layers_block_type``."""
    entries = text_cfg.get("layer_types") or text_cfg.get("layers_block_type") or []
    return [str(entry).lower() for entry in entries]


def is_recurrent_layer(entry: str) -> bool:
    """Mamba or linear-attention (gated delta net) layer: a recurrent state, no KV cache."""
    return "mamba" in entry or "linear" in entry


def sliding_layer_fraction(text_cfg: dict[str, Any]) -> float:
    """Fraction of layers using windowed attention.

    Only windowed layers cap KV growth. ``sliding_window_pattern`` is one
    full-attention layer every N. MiMo-V2's ``hybrid_layer_pattern`` marks
    each windowed layer 1 and each full-attention layer 0.
    """
    layer_types = text_cfg.get("layer_types")
    if layer_types:
        windowed = sum(1 for entry in layer_types if "sliding" in str(entry))
        return windowed / len(layer_types)
    hybrid = text_cfg.get("hybrid_layer_pattern")
    if hybrid:
        return sum(hybrid) / len(hybrid)
    pattern = text_cfg.get("sliding_window_pattern")
    if isinstance(pattern, int) and pattern > 1:
        return (pattern - 1) / pattern
    return 1.0


def layer_mix(text_cfg: dict[str, Any], n_layers: int) -> tuple[int | None, int]:
    """(attention layers, recurrent layers) for a hybrid state-space model.

    Reads ``layer_types`` / ``layers_block_type``, ``hybrid_override_pattern``,
    ``full_attention_interval``, or a parallel mixer (both counts = ``n_layers``).
    """
    entries = layer_type_list(text_cfg)
    if entries:
        recurrent = sum(1 for e in entries if is_recurrent_layer(e))
        if not recurrent:
            # An explicit layer list wins over leftover ``mamba_*`` keys.
            return None, 0
        attention = sum(
            1 for e in entries if "attention" in e and not is_recurrent_layer(e)
        )
        return attention, recurrent

    pattern = text_cfg.get("hybrid_override_pattern")
    if isinstance(pattern, str) and "M" in pattern:
        return pattern.count("*"), pattern.count("M")

    every = text_cfg.get("full_attention_interval")
    if isinstance(every, int) and every > 1:
        attention = n_layers // every
        return attention, n_layers - attention

    if (
        any(key.startswith("mamba_") for key in text_cfg)
        or "ssm_state_size" in text_cfg
    ):
        return n_layers, n_layers
    return None, 0


def ffn_layer_mix(text_cfg: dict[str, Any]) -> tuple[int | None, int | None]:
    """(dense-MLP layers, MoE layers) when FFN is a block of its own.

    Most stacks put an FFN in every block alongside the mixer, and both
    counts stay ``None`` (every layer). Block-exclusive layouts use one
    block kind per layer: Mamba, attention, MLP, or MoE, in
    ``layers_block_type`` ("moe" / "mlp") or as ``-`` / ``E`` in
    ``hybrid_override_pattern``. Mixer-only lists leave FFN counts unset.
    """
    entries = layer_type_list(text_cfg)
    if entries:
        moe = sum(1 for e in entries if e == "moe")
        mlp = sum(1 for e in entries if e in ("mlp", "-"))
        if moe or mlp:
            return mlp, moe

    pattern = text_cfg.get("hybrid_override_pattern")
    if (
        isinstance(pattern, str)
        and "M" in pattern
        and ("-" in pattern or "E" in pattern)
    ):
        return pattern.count("-"), pattern.count("E") or None
    return None, None


def decoder_fields(text_cfg: dict[str, Any]) -> dict[str, Any]:
    """Decoder width, depth, vocab, and attention-window fields."""
    n_heads = int(text_cfg["num_attention_heads"])
    hidden = int(text_cfg["hidden_size"])
    n_kv = first_int(text_cfg, "num_key_value_heads", default=n_heads)
    # MiMo-V2 names its sliding-window layers' KV heads apart from the
    # full-attention layers'.
    global_n_kv = None
    if text_cfg.get("swa_num_key_value_heads"):
        global_n_kv = n_kv
        n_kv = int(text_cfg["swa_num_key_value_heads"])
    head_dim = int(text_cfg.get("head_dim") or hidden // n_heads)
    if text_cfg.get("kv_lora_rank"):
        # MLA caches one latent every head shares; each query/key head is
        # its no-rope part plus the shared rope key.
        n_kv = 1
        head_dim = first_int(text_cfg, "qk_nope_head_dim") + first_int(
            text_cfg, "qk_rope_head_dim"
        )
    # relu-family activations are ungated (up+down); otherwise SwiGLU (gate+up+down).
    act = str(
        text_cfg.get("mlp_hidden_act") or text_cfg.get("hidden_act") or ""
    ).lower()
    gated = act not in {"relu", "relu2", "relu_squared", "squared_relu"}
    n_layers = text_cfg.get("num_hidden_layers") or len(layer_type_list(text_cfg))
    if not n_layers:
        msg = "Model config has neither num_hidden_layers nor a per-layer type list"
        raise KeyError(msg)
    return {
        "n_layers": int(n_layers),
        "hidden_size": hidden,
        # Expert-only stacks publish just the expert width.
        "intermediate_size": int(
            text_cfg.get("intermediate_size") or text_cfg["moe_intermediate_size"]
        ),
        "n_heads": n_heads,
        "n_kv_heads": n_kv,
        "global_n_kv_heads": global_n_kv,
        "head_dim": head_dim,
        "vocab_size": int(text_cfg["vocab_size"]),
        "tied_embeddings": bool(text_cfg.get("tie_word_embeddings", False)),
        "gated_mlp": gated,
        "fused_gate_up_mlp": text_cfg.get("model_type")
        in FUSED_GATE_UP_MLP_MODEL_TYPES,
        "fp32_norm_inputs": text_cfg.get("model_type") in FP32_RMS_NORM_MODEL_TYPES,
        "qk_norm": text_cfg.get("model_type") in QK_NORM_MODEL_TYPES,
        "chunked_routed_experts": text_cfg.get("model_type")
        in CHUNKED_ROUTED_EXPERT_MODEL_TYPES,
        "routed_expert_recompute": text_cfg.get("model_type")
        in ROUTED_EXPERT_RECOMPUTE_MODEL_TYPES,
        "transposed_experts": text_cfg.get("model_type")
        in TRANSPOSED_EXPERT_MODEL_TYPES,
        "unfused_mamba_scan_family": text_cfg.get("model_type")
        in UNFUSED_MAMBA_SCAN_MODEL_TYPES,
        "tensor_parallel_tied_embedding": bool(text_cfg.get("tie_word_embeddings"))
        and text_cfg.get("model_type") in TIED_EMBEDDING_TENSOR_PARALLEL_MODEL_TYPES,
        "attention_output_gate": text_cfg.get("model_type")
        in ATTENTION_OUTPUT_GATE_MODEL_TYPES,
        "attn_bias": bool(text_cfg.get("attention_bias") or text_cfg.get("qkv_bias"))
        or text_cfg.get("model_type") in QKV_BIAS_MODEL_TYPES,
        "flex_attention": FAMILY_ATTN_IMPLEMENTATIONS.get(
            text_cfg.get("model_type") or ""
        )
        == "flex_attention",
        "sliding_window": (
            text_cfg.get("sliding_window")
            if text_cfg.get("use_sliding_window", True)
            else None
        ),
        "sliding_window_layer_fraction": sliding_layer_fraction(text_cfg),
    }


def moe_fields(text_cfg: dict[str, Any]) -> dict[str, Any]:
    """Routed and shared expert fields. Dense models leave experts unset."""
    n_experts = first_int(
        text_cfg, "num_experts", "num_local_experts", "n_routed_experts"
    )
    shared_keys = [
        "moe_shared_expert_intermediate_size",
        "shared_expert_intermediate_size",
    ]
    expert_shared_inter = first_int(text_cfg, *shared_keys)
    # Granite's always-on MLP beside the routed experts; without routed
    # experts it is the dense MLP.
    if n_experts:
        shared_keys.append("shared_intermediate_size")
    shared_inter = first_int(text_cfg, *shared_keys)
    # Qwen-MoE names one shared expert by its width alone.
    n_shared = int(text_cfg.get("n_shared_experts") or (1 if shared_inter else 0))
    return {
        "n_experts": n_experts or None,
        "n_experts_per_tok": (
            int(text_cfg["num_experts_per_tok"])
            if text_cfg.get("num_experts_per_tok")
            else None
        ),
        "expert_intermediate_size": (
            int(text_cfg["moe_intermediate_size"])
            if text_cfg.get("moe_intermediate_size")
            else None
        ),
        "n_shared_experts": n_shared,
        "moe_latent_size": (
            int(text_cfg["moe_latent_size"])
            if text_cfg.get("moe_latent_size")
            else None
        ),
        "shared_expert_intermediate_size": shared_inter or None,
        "shared_expert_reduce_saved": bool(expert_shared_inter)
        and text_cfg.get("model_type") not in TRAILING_SHARED_EXPERT_MODEL_TYPES,
    }


def dense_ffn_layers(text_cfg: dict[str, Any], n_layers: int) -> int:
    """Layers of a MoE stack whose FFN is a dense MLP rather than routed experts.

    A ``"dense"`` entry in ``mlp_layer_types``, a 0 in a per-layer
    ``moe_layer_freq`` list (MiMo-V2), else DeepSeek's rule: layer ``i``
    routes experts when ``i >= first_k_dense_replace`` and
    ``i % moe_layer_freq == 0``.
    """
    if not moe_fields(text_cfg)["n_experts"]:
        return 0
    kinds = text_cfg.get("mlp_layer_types")
    if kinds:
        return sum(1 for kind in kinds if kind == "dense")
    freq = text_cfg.get("moe_layer_freq") or 1
    if isinstance(freq, list):
        return sum(1 for routed in freq[:n_layers] if not routed)
    first = int(text_cfg.get("first_k_dense_replace") or 0)
    return sum(1 for i in range(n_layers) if i < first or i % int(freq))


def attention_sink_layers(text_cfg: dict[str, Any], n_layers: int) -> int:
    """Attention layers with one learned sink logit per query head.

    MiMo-V2 flags sinks per layer kind; ``hybrid_layer_pattern`` marks each
    windowed layer 1.
    """
    if text_cfg.get("model_type") in ATTENTION_SINK_MODEL_TYPES:
        return n_layers
    windowed = sum((text_cfg.get("hybrid_layer_pattern") or [])[:n_layers])
    return (windowed if text_cfg.get("add_swa_attention_sink_bias") else 0) + (
        n_layers - windowed if text_cfg.get("add_full_attention_sink_bias") else 0
    )


def indexed_layers(text_cfg: dict[str, Any], n_layers: int, n_csa: int) -> int:
    """Attention layers running a sparse-attention (DSA) indexer."""
    if not text_cfg.get("index_n_heads"):
        return 0
    if n_csa:
        # DeepSeek-V4 runs the indexer inside each CSA compressor.
        return n_csa
    layers = layer_type_list(text_cfg) or ["attention"] * n_layers
    indexers = text_cfg.get("indexer_types") or ["full"] * n_layers
    return sum(
        1
        for layer, indexer in zip(layers[:n_layers], indexers[:n_layers], strict=True)
        if indexer == "full" and not is_recurrent_layer(layer)
    )


def encoder_params(cfg: dict[str, Any]) -> tuple[int, int]:
    """Rough (block, total) parameter counts of a transformer encoder from its config.

    Qwen vision configs name the layer, head and position counts ``depth``,
    ``num_heads`` and ``num_position_embeddings``.
    """
    h = int(cfg.get("hidden_size") or 0)
    layers = int(cfg.get("num_hidden_layers") or cfg.get("depth") or 0)
    if not h or not layers:
        return 0, 0
    heads = int(cfg.get("num_attention_heads") or cfg.get("num_heads") or 1)
    head_dim = int(cfg.get("head_dim") or h // heads)
    kv_heads = int(cfg.get("num_key_value_heads") or heads)
    intermediate = int(cfg.get("intermediate_size") or 4 * h)
    attn = h * heads * head_dim + 2 * h * kv_heads * head_dim + heads * head_dim * h
    model_type = cfg.get("model_type")
    mlp_matrices = 3 if model_type in GATED_MLP_TOWER_MODEL_TYPES else 2
    per_layer = attn + mlp_matrices * h * intermediate + 2 * h
    if model_type in CONFORMER_TOWER_MODEL_TYPES:
        per_layer += 2 * h * intermediate + 4 * h * h
    positions = int(
        cfg.get("position_embedding_size") or cfg.get("num_position_embeddings") or 0
    )
    if model_type in AXIS_POSITION_TOWER_MODEL_TYPES:
        positions *= 2
    blocks = layers * per_layer
    return blocks, blocks + positions * h


def tower_configs(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Vision and audio encoder configs the model builds from ``config``."""
    if config.get("model_type") in TEXT_ONLY_MODEL_TYPES:
        return []
    return [
        config[key] for key in TOWER_MODALITIES if isinstance(config.get(key), dict)
    ]


def _patch_merge(config: dict[str, Any], vision: dict[str, Any]) -> int:
    """Patches per side merged into one token; 1 when the config sets none.

    Qwen's spatial merge, Gemma4's pooling kernel, Nemotron's pixel shuffle.
    """
    merge = vision.get("spatial_merge_size") or vision.get("pooling_kernel_size")
    if not merge and config.get("downsample_ratio"):
        merge = round(1 / config["downsample_ratio"])
    return int(merge or 1)


def vision_geometry_fields(config: dict[str, Any]) -> dict[str, Any]:
    """Vision tower geometry; empty without a tower."""
    vision = config.get("vision_config")
    model_type = config.get("model_type", "")
    if not isinstance(vision, dict) or model_type in TEXT_ONLY_MODEL_TYPES:
        return {}
    merge = _patch_merge(config, vision)
    soft_tokens = int(config.get("vision_soft_tokens_per_image") or 0)
    return {
        "vision_hidden_size": int(
            config.get("vit_hidden_size") or vision.get("hidden_size") or 0
        ),
        "vision_patch_size": int(
            config.get("patch_size") or vision.get("patch_size") or 0
        ),
        "vision_intermediate_size": int(vision.get("intermediate_size") or 0),
        "vision_n_layers": int(
            vision.get("depth") or vision.get("num_hidden_layers") or 0
        ),
        "vision_in_channels": int(vision.get("in_channels") or 3),
        "vision_temporal_patch_size": int(vision.get("temporal_patch_size") or 1),
        "vision_out_hidden_size": int(vision.get("out_hidden_size") or 0),
        "vision_spatial_merge_size": merge,
        "vision_profile_patches": VISION_PROFILE_PATCHES_BY_MODEL_TYPE.get(
            model_type, 0
        ),
        "vision_resize": VISION_RESIZE_BY_MODEL_TYPE.get(model_type, "none"),
        "vision_min_patches": int(vision.get("min_num_patches") or 0),
        "vision_max_patches": int(
            vision.get("max_num_patches") or soft_tokens * merge**2
        ),
        "vision_feature_cache": model_type in VISION_FEATURE_CACHE_MODEL_TYPES,
    }

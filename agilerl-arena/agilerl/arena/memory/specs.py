# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Input specs for the GPU memory estimator.

Pydantic models for architecture, device, and settings. JSON-serializable
and torch-free.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from agilerl.arena.memory.resolve.lora import OMITTED_MAX_LORA_RANK
from agilerl.arena.memory.resolve.sequences import (
    COLOCATED_MAX_NUM_SEQS,
    OMITTED_GPU_MEMORY_UTILIZATION,
)
from agilerl.arena.models.fsdp import FSDPConfig

GiB = 1024**3
MiB = 1024**2

# Default Mamba-2 chunked-scan chunk length.
MAMBA_CHUNK_SIZE_DEFAULT = 256

# Model types whose trainer RMSNorm is the stock HF module, which keeps an
# fp32 copy of each norm input for backward. Gemma 4's norm upcasts with
# ``hidden_states.float()`` under Liger too.
FP32_RMS_NORM_MODEL_TYPES = frozenset({"granitemoe", "granitemoehybrid", "gemma4_text"})

# Model types with a norm before and after both the attention and the MLP.
SANDWICH_NORM_MODEL_TYPES = frozenset({"gemma4_text"})

# Model types Liger has no SwiGLU kernel for. The stock gated MLP keeps the
# activated gate for backward besides gate and up.
STOCK_GATED_MLP_MODEL_TYPES = frozenset({"granitemoehybrid"})

# Attention norms that upcast Q to fp32 inside the block (Qwen3/3.5 ``q_norm``).
QK_NORM_MODEL_TYPES = frozenset(
    {"qwen3", "qwen3_moe", "qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"}
)

# Model types whose packed experts take the trainer's self-routing grouped-GEMM
# path, which runs experts in row chunks and writes straight into the layer output.
CHUNKED_ROUTED_EXPERT_MODEL_TYPES = frozenset(
    {"nemotron_h", "nemotron_h_omni", "qwen3_moe"}
)

# Compute capability and model types where AgileRL trains Mamba layers with
# HF's eager ``torch_forward`` scan; the fused SSM kernels return NaN there.
EAGER_MAMBA_SCAN_CAPABILITY = (8, 9)
EAGER_MAMBA_SCAN_MODEL_TYPES = frozenset({"nemotron_h", "nemotron_h_omni"})

# Outer config model types whose vLLM class sets ``is_3d_moe_weight``, so the
# engine wraps experts in the 3D fused-MoE LoRA layout.
VLLM_3D_MOE_LORA_MODEL_TYPES = frozenset(
    {"gpt_oss", "interns1_pro", "qwen3_5_moe", "qwen3_omni_moe", "qwen3_vl_moe"}
)

# Outer config model types whose HF class lists its tower encoder blocks in
# ``_no_split_modules``, so FSDP shards each block as its own unit.
TOWER_BLOCK_UNIT_MODEL_TYPES = frozenset({"gemma4", "qwen3_5", "qwen3_5_moe"})

# Vision patches in the image vLLM encodes when profiling a model that keeps
# its towers, keyed by outer config ``model_type``. Qwen3.5/3.6: the
# preprocessor's 16,777,216-pixel longest edge over 16x16 patches. Families
# absent here have no measured profile and add no encoder term.
VISION_PROFILE_PATCHES_BY_MODEL_TYPE: dict[str, int] = {
    "qwen3_5": 65536,
    "qwen3_5_moe": 65536,
}

# CUDA context bytes keyed by torch device name: device memory in use after
# the first torch allocation.
CUDA_CONTEXT_BYTES_BY_DEVICE: dict[str, int] = {
    "NVIDIA A100-SXM4-40GB": 1268 * MiB,
    "NVIDIA A100-SXM4-80GB": 1268 * MiB,
    # Estimated from its driver reservation, SM count and resident threads.
    "NVIDIA H100 80GB HBM3": 1000 * MiB,
    "NVIDIA L4": 694 * MiB,
}
# Unlisted devices use the largest listed value.
CUDA_CONTEXT_BYTES_DEFAULT = 1268 * MiB

# NCCL communicator and buffer bytes on each rank of a multi-rank trainer,
# keyed by torch device name. Set by the interconnect; constant across rank
# count and sharding mode.
NCCL_BYTES_BY_DEVICE: dict[str, int] = {
    "NVIDIA A100-SXM4-40GB": 600 * MiB,
    "NVIDIA A100-SXM4-80GB": 600 * MiB,
    "NVIDIA L4": 91 * MiB,
}
NCCL_BYTES_DEFAULT = 600 * MiB

# Bytes per element.
DTYPE_BYTES: dict[str, float] = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
}

WeightDtype = Literal["fp32", "bf16", "fp16"]
Algorithm = Literal["grpo", "gspo", "cispo", "ppo", "reinforce", "dpo", "sft"]
LoraTargetScope = Literal["all-linear", "attention-only"]
PackedMoeDispatch = Literal["materialized", "contracted"]
# One layer of a block-exclusive stack: an attention, FFN (MLP or MoE) or Mamba layer.
BlockKind = Literal["attention", "ffn", "mamba"]
# One layer of a block-exclusive stack, with MLP and MoE told apart.
LayerKind = Literal["attention", "mamba", "mlp", "moe"]
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


def _encoder_params(cfg: dict[str, Any]) -> tuple[int, int]:
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
    per_layer = attn + 2 * h * intermediate + 2 * h
    positions = int(
        cfg.get("position_embedding_size") or cfg.get("num_position_embeddings") or 0
    )
    blocks = layers * per_layer
    return blocks, blocks + positions * h


def _tower_configs(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Vision and audio encoder configs present in ``config``."""
    return [
        config[key]
        for key in ("vision_config", "audio_config")
        if isinstance(config.get(key), dict)
    ]


def multimodal_tower_params(config: dict[str, Any]) -> int:
    """Lower-bound params treating vision/audio configs as transformer encoders.

    :param config: Hugging Face model config.
    :return: Tower parameter count.
    """
    return sum(_encoder_params(cfg)[1] for cfg in _tower_configs(config))


def multimodal_tower_block_params(config: dict[str, Any]) -> int:
    """Encoder-block share of :func:`multimodal_tower_params` that FSDP shards as units.

    :param config: Hugging Face model config.
    :return: Block parameter count; zero for models FSDP does not wrap per block.
    """
    if config.get("model_type") not in TOWER_BLOCK_UNIT_MODEL_TYPES:
        return 0
    return sum(_encoder_params(cfg)[0] for cfg in _tower_configs(config))


def _vision_fields(config: dict[str, Any]) -> dict[str, int]:
    """Vision tower geometry; empty without a tower."""
    vision = config.get("vision_config")
    if not isinstance(vision, dict):
        return {}
    model_type = config.get("model_type", "")
    # Patches per side merged into one token: Qwen's spatial merge, Gemma4's
    # pooling kernel, Nemotron's pixel shuffle.
    merge = vision.get("spatial_merge_size") or vision.get("pooling_kernel_size")
    if not merge and config.get("downsample_ratio"):
        merge = round(1 / config["downsample_ratio"])
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
        "vision_spatial_merge_size": int(merge or 1),
        "vision_profile_patches": VISION_PROFILE_PATCHES_BY_MODEL_TYPE.get(
            model_type, 0
        ),
        "vision_resize": VISION_RESIZE_BY_MODEL_TYPE.get(model_type, "none"),
        "vision_min_patches": int(vision.get("min_num_patches") or 0),
        "vision_max_patches": int(
            vision.get("max_num_patches") or soft_tokens * int(merge or 1) ** 2
        ),
        "vision_feature_cache": model_type in VISION_FEATURE_CACHE_MODEL_TYPES,
    }


def _sliding_layer_fraction(text_cfg: dict[str, Any]) -> float:
    """Fraction of layers using windowed attention.

    Only windowed layers cap KV growth. ``sliding_window_pattern`` is one
    full-attention layer every N.
    """
    layer_types = text_cfg.get("layer_types")
    if layer_types:
        windowed = sum(1 for entry in layer_types if "sliding" in str(entry))
        return windowed / len(layer_types)
    pattern = text_cfg.get("sliding_window_pattern")
    if isinstance(pattern, int) and pattern > 1:
        return (pattern - 1) / pattern
    return 1.0


def _first_int(cfg: dict[str, Any], *keys: str, default: int = 0) -> int:
    """First non-empty integer among HF alias keys for the same field."""
    for key in keys:
        value = cfg.get(key)
        if value:
            return int(value)
    return default


def _layer_type_list(text_cfg: dict[str, Any]) -> list[str]:
    """Per-layer type list from ``layer_types``, else ``layers_block_type``."""
    entries = text_cfg.get("layer_types") or text_cfg.get("layers_block_type") or []
    return [str(entry).lower() for entry in entries]


def _is_recurrent_layer(entry: str) -> bool:
    """Mamba or linear-attention (gated delta net) layer: a recurrent state, no KV cache."""
    return "mamba" in entry or "linear" in entry


def _layer_mix(text_cfg: dict[str, Any], n_layers: int) -> tuple[int | None, int]:
    """(attention layers, recurrent layers) for a hybrid state-space model.

    Reads ``layer_types`` / ``layers_block_type``, ``hybrid_override_pattern``,
    ``full_attention_interval``, or a parallel mixer (both counts = ``n_layers``).
    """
    entries = _layer_type_list(text_cfg)
    if entries:
        recurrent = sum(1 for e in entries if _is_recurrent_layer(e))
        if not recurrent:
            # An explicit layer list wins over leftover ``mamba_*`` keys.
            return None, 0
        attention = sum(
            1 for e in entries if "attention" in e and not _is_recurrent_layer(e)
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


def _ffn_layer_mix(text_cfg: dict[str, Any]) -> tuple[int | None, int | None]:
    """(dense-MLP layers, MoE layers) when FFN is a block of its own.

    Most stacks put an FFN in every block alongside the mixer, and both
    counts stay ``None`` (every layer). Block-exclusive layouts use one
    block kind per layer: Mamba, attention, MLP, or MoE, in
    ``layers_block_type`` ("moe" / "mlp") or as ``-`` / ``E`` in
    ``hybrid_override_pattern``. Mixer-only lists leave FFN counts unset.
    """
    entries = _layer_type_list(text_cfg)
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


def _layer_kind(entry: str) -> LayerKind:
    """One ``layers_block_type`` entry or ``hybrid_override_pattern`` character."""
    if entry in ("moe", "E"):
        return "moe"
    if entry in ("mlp", "-"):
        return "mlp"
    if entry == "*" or ("attention" in entry and not _is_recurrent_layer(entry)):
        return "attention"
    return "mamba"


def _layer_kinds(text_cfg: dict[str, Any]) -> tuple[LayerKind, ...]:
    """Each layer's kind, in order, for a block-exclusive stack; empty otherwise."""
    entries = _layer_type_list(text_cfg) or list(
        text_cfg.get("hybrid_override_pattern") or ""
    )
    if not {"moe", "mlp", "E", "-"} & set(entries):
        return ()
    return tuple(_layer_kind(entry) for entry in entries)


class MambaGeometry(TypedDict, total=False):
    """Normalised Mamba-2 / gated-delta-net state dimensions."""

    d_state: int
    d_conv: int
    n_groups: int
    n_heads: int
    d_head: int
    conv_dim: int
    chunk_size: int
    gated_delta: bool


def _mamba_state_geometry(text_cfg: dict[str, Any], hidden: int) -> MambaGeometry:
    """Mamba-2 / gated-delta-net state dimensions, normalised to one shape.

    vLLM stores two tensors per recurrent layer per slot: a causal-conv
    window of ``(conv_dim, d_conv - 1)`` and a recurrent state of
    ``(n_heads, d_head, d_state)``. The families differ only in ``conv_dim``.
    """
    if text_cfg.get("linear_key_head_dim"):  # gated-delta-net
        k_dim = int(text_cfg["linear_key_head_dim"])
        v_dim = int(text_cfg["linear_value_head_dim"])
        k_heads = int(text_cfg["linear_num_key_heads"])
        v_heads = int(text_cfg["linear_num_value_heads"])
        return {
            "d_state": k_dim,
            "d_conv": int(text_cfg.get("linear_conv_kernel_dim") or 0),
            "n_groups": 0,
            "n_heads": v_heads,
            "d_head": v_dim,
            "conv_dim": 2 * k_heads * k_dim + v_heads * v_dim,
            "chunk_size": int(text_cfg.get("chunk_size") or MAMBA_CHUNK_SIZE_DEFAULT),
            "gated_delta": True,
        }

    d_state = _first_int(text_cfg, "mamba_d_state", "ssm_state_size", "mamba_state_dim")
    n_groups = _first_int(text_cfg, "mamba_n_groups", "mamba_num_groups", "n_groups")
    n_heads = _first_int(text_cfg, "mamba_n_heads", "mamba_num_heads")
    d_head = _first_int(text_cfg, "mamba_d_head", "mamba_head_dim")
    d_inner = int(
        text_cfg.get("mamba_d_ssm")
        or n_heads * d_head
        or _first_int(text_cfg, "mamba_expand", "expand", default=2) * hidden
    )
    return {
        "d_state": d_state,
        "d_conv": _first_int(text_cfg, "mamba_d_conv", "conv_kernel"),
        "n_groups": n_groups,
        "n_heads": n_heads,
        "d_head": d_head,
        "conv_dim": d_inner + 2 * n_groups * d_state,
        "chunk_size": _first_int(
            text_cfg, "chunk_size", "mamba_chunk_size", default=MAMBA_CHUNK_SIZE_DEFAULT
        ),
    }


def _decoder_fields(text_cfg: dict[str, Any]) -> dict[str, Any]:
    """Decoder width, depth, vocab, and attention-window fields."""
    n_heads = int(text_cfg["num_attention_heads"])
    hidden = int(text_cfg["hidden_size"])
    n_kv = _first_int(text_cfg, "num_key_value_heads", default=n_heads)
    # relu-family activations are ungated (up+down); otherwise SwiGLU (gate+up+down).
    act = str(
        text_cfg.get("mlp_hidden_act") or text_cfg.get("hidden_act") or ""
    ).lower()
    gated = act not in {"relu", "relu2", "relu_squared", "squared_relu"}
    n_layers = text_cfg.get("num_hidden_layers") or len(_layer_type_list(text_cfg))
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
        "head_dim": int(text_cfg.get("head_dim") or hidden // n_heads),
        "vocab_size": int(text_cfg["vocab_size"]),
        "tied_embeddings": bool(text_cfg.get("tie_word_embeddings", False)),
        "gated_mlp": gated,
        "fp32_norm_inputs": text_cfg.get("model_type") in FP32_RMS_NORM_MODEL_TYPES,
        "stock_gated_mlp": text_cfg.get("model_type") in STOCK_GATED_MLP_MODEL_TYPES,
        "qk_norm": text_cfg.get("model_type") in QK_NORM_MODEL_TYPES,
        "chunked_routed_experts": text_cfg.get("model_type")
        in CHUNKED_ROUTED_EXPERT_MODEL_TYPES,
        "eager_mamba_scan_family": text_cfg.get("model_type")
        in EAGER_MAMBA_SCAN_MODEL_TYPES,
        "attn_bias": bool(text_cfg.get("attention_bias") or text_cfg.get("qkv_bias")),
        "sliding_window": (
            text_cfg.get("sliding_window")
            if text_cfg.get("use_sliding_window", True)
            else None
        ),
        "sliding_window_layer_fraction": _sliding_layer_fraction(text_cfg),
        "final_logit_softcap": text_cfg.get("final_logit_softcapping"),
    }


def _moe_fields(text_cfg: dict[str, Any]) -> dict[str, Any]:
    """Routed and shared expert fields. Dense models leave experts unset."""
    n_experts = _first_int(
        text_cfg, "num_experts", "num_local_experts", "n_routed_experts"
    )
    shared_inter = _first_int(
        text_cfg,
        "moe_shared_expert_intermediate_size",
        "shared_expert_intermediate_size",
    )
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
    }


def _layout_fields(
    text_cfg: dict[str, Any],
    n_layers: int,
    hidden: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Layer mix, mamba geometry, mixed-width fields, and multimodal towers.

    :param config: Outer HF config, for ``vision_config`` / ``audio_config``.
    """
    n_attention, n_mamba = _layer_mix(text_cfg, n_layers)
    n_mlp, n_moe = _ffn_layer_mix(text_cfg)
    geo = _mamba_state_geometry(text_cfg, hidden) if n_mamba else {}
    return {
        "n_attention_layers": n_attention,
        "n_mamba_layers": n_mamba,
        "n_mlp_layers": n_mlp,
        "n_moe_layers": n_moe,
        "layer_kinds": _layer_kinds(text_cfg),
        **{f"mamba_{key}": value for key, value in geo.items()},
        "global_head_dim": text_cfg.get("global_head_dim") or None,
        "n_kv_shared_layers": int(text_cfg.get("num_kv_shared_layers") or 0),
        "double_wide_mlp": bool(text_cfg.get("use_double_wide_mlp", False)),
        "per_layer_input_dim": int(text_cfg.get("hidden_size_per_layer_input") or 0),
        "norms_per_block": (
            4 if text_cfg.get("model_type") in SANDWICH_NORM_MODEL_TYPES else 2
        )
        + (1 if text_cfg.get("hidden_size_per_layer_input") else 0),
        "per_layer_input_vocab": int(text_cfg.get("vocab_size_per_layer_input") or 0),
        "multimodal_tower_params": multimodal_tower_params(config),
        "multimodal_tower_block_params": multimodal_tower_block_params(config),
        **_vision_fields(config),
    }


class ModelArch(BaseModel):
    """Dense (or MoE) decoder-only transformer geometry.

    Field names follow HF ``config.json`` semantics; use
    :meth:`from_hf_config` to build one from a raw config dict.
    """

    model_config = ConfigDict(frozen=True)

    n_layers: int
    hidden_size: int
    intermediate_size: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    vocab_size: int
    tied_embeddings: bool = False
    # Final-logit softcap of the head (Gemma family); the fused loss keeps
    # the full-logits path when one is set.
    final_logit_softcap: float | None = None
    # Caps KV growth per sequence on the windowed layers.
    sliding_window: int | None = None
    # Ignored when ``sliding_window`` is None.
    sliding_window_layer_fraction: float = 1.0
    attn_bias: bool = False
    # SwiGLU (gate + up + down). Ungated MLPs use two matrices (up + down).
    gated_mlp: bool = True
    # Trainer gated MLP saves the activated gate as its own tensor.
    stock_gated_mlp: bool = False
    # Trainer RMSNorm saves an fp32 copy of its input for backward.
    fp32_norm_inputs: bool = False
    # RMSNorms over the hidden state in one decoder block, including a
    # per-layer-input norm.
    norms_per_block: int = 2
    # Per-head Q/K norm upcasts Q to fp32 inside the block.
    qk_norm: bool = False
    # Trains Mamba layers with HF's eager scan on compute capability 8.9.
    eager_mamba_scan_family: bool = False

    # None for dense models. Weights/optimizer scale with the total expert
    # count; activations scale with the active (routed) count.
    n_experts: int | None = None
    n_experts_per_tok: int | None = None
    expert_intermediate_size: int | None = None
    n_shared_experts: int = 0
    # ``None`` falls back to ``expert_intermediate_size``.
    shared_expert_intermediate_size: int | None = None
    # Width routed experts read and write (Nemotron-H latent MoE). Each MoE
    # layer projects hidden -> latent before the experts and back after.
    # ``None`` runs experts at ``hidden_size``.
    moe_latent_size: int | None = None

    # Each layer is exactly one of Mamba, attention, MLP or MoE. ``None``
    # keeps the every-layer default.
    n_mlp_layers: int | None = None
    n_moe_layers: int | None = None
    # Layer kinds in order when the config lists them; empty when unknown.
    layer_kinds: tuple[LayerKind, ...] = ()
    # Packed experts run in row chunks on the self-routing grouped-GEMM path.
    chunked_routed_experts: bool = False
    # vLLM serves this model's expert LoRA in the 3D fused-MoE layout.
    vllm_3d_moe_lora: bool = False

    multimodal_tower_params: int = 0
    # Tower params inside encoder blocks FSDP shards as their own units.
    multimodal_tower_block_params: int = 0
    # Vision tower width per patch and the patch side. Zero without a vision
    # tower.
    vision_hidden_size: int = 0
    vision_patch_size: int = 0
    # Vision encoder MLP width, pixel channels, frames per temporal patch,
    # merged output width and side of the patch merge. Read by the engine's
    # encoder profile.
    vision_intermediate_size: int = 0
    vision_in_channels: int = 3
    vision_temporal_patch_size: int = 1
    vision_out_hidden_size: int = 0
    vision_spatial_merge_size: int = 1
    # Vision encoder blocks.
    vision_n_layers: int = 0
    # Patches vLLM encodes for its start-up profile image; 0 when unmeasured.
    vision_profile_patches: int = 0
    # The image processor's patch-grid rule, and its per-image patch bounds
    # under ``pooled_budget`` and ``dynamic_tiles``.
    vision_resize: VisionResize = "none"
    vision_min_patches: int = 0
    vision_max_patches: int = 0
    # VisionFeatureCache keeps this family's tower outputs through a learn step.
    vision_feature_cache: bool = False

    # Head dim on full-attention layers when it differs from the sliding
    # layers', which widens q/k/v on the full-attention layers.
    global_head_dim: int | None = None
    # Trailing layers that reuse an earlier layer's KV and add nothing to
    # the cache.
    n_kv_shared_layers: int = 0
    double_wide_mlp: bool = False
    # Looked up per token. Parameter block is
    # ``vocab_size_per_layer_input x n_layers x this``; live activation is
    # ``tokens x n_layers x this``.
    per_layer_input_dim: int = 0
    per_layer_input_vocab: int = 0

    # Attention layers, when fewer than ``n_layers``. None means all of them.
    n_attention_layers: int | None = None
    n_mamba_layers: int = 0
    mamba_d_state: int = 0
    mamba_d_conv: int = 0
    mamba_n_groups: int = 0
    mamba_n_heads: int = 0
    mamba_d_head: int = 0
    mamba_ssm_state_dtype: WeightDtype = "fp32"
    # Mamba-2: ``d_inner + 2 * n_groups * d_state``; gated-delta-net:
    # ``2 * k_heads * k_dim + v_heads * v_dim``.
    mamba_conv_dim: int = 0
    mamba_chunk_size: int = MAMBA_CHUNK_SIZE_DEFAULT
    # Recurrent layers are gated-delta-net rather than Mamba-2.
    mamba_gated_delta: bool = False

    @property
    def is_hybrid_ssm(self) -> bool:
        return self.n_mamba_layers > 0

    @property
    def attention_layers(self) -> int:
        """Layers that hold a KV cache."""
        if self.n_attention_layers is not None:
            return self.n_attention_layers
        return self.n_layers

    @property
    def mlp_width_factor(self) -> float:
        """Mean intermediate width relative to ``intermediate_size``."""
        if not self.double_wide_mlp or not self.n_kv_shared_layers:
            return 1.0
        wide = min(self.n_kv_shared_layers, self.n_layers)
        return (self.n_layers + wide) / self.n_layers

    @property
    def peak_mlp_width_factor(self) -> float:
        """Widest block's intermediate width relative to ``intermediate_size``.

        Per-block transients peak at the widest block. Parameter
        totals use :attr:`mlp_width_factor`.
        """
        if not self.double_wide_mlp or not self.n_kv_shared_layers:
            return 1.0
        return 2.0

    @property
    def mean_qkv_dim(self) -> int:
        """q+k+v width per token, averaged over layer types."""
        narrow = (self.n_heads + 2 * self.n_kv_heads) * self.head_dim
        if not self.global_head_dim or self.global_head_dim == self.head_dim:
            return narrow
        wide = (self.n_heads + 2 * self.n_kv_heads) * self.global_head_dim
        # ``sliding_window_layer_fraction`` is the share using the narrow head.
        f = self.sliding_window_layer_fraction if self.sliding_window else 0.0
        return int(f * narrow + (1.0 - f) * wide)

    @property
    def peak_qkv_dim(self) -> int:
        """q+k+v width of the widest block.

        Uses ``global_head_dim`` when it exceeds ``head_dim``. Per-block
        counterpart of :attr:`mean_qkv_dim`.
        """
        if not self.global_head_dim or self.global_head_dim == self.head_dim:
            return (self.n_heads + 2 * self.n_kv_heads) * self.head_dim
        return (self.n_heads + 2 * self.n_kv_heads) * self.global_head_dim

    @property
    def is_moe(self) -> bool:
        return self.n_experts is not None and self.n_experts > 1

    @property
    def expert_width(self) -> int:
        """Feature width of a routed expert's input and output rows."""
        return self.moe_latent_size or self.hidden_size

    @property
    def block_exclusive_layers(self) -> bool:
        """Whether each layer holds exactly one block kind."""
        return self.n_mlp_layers is not None or self.n_moe_layers is not None

    @property
    def sublayer_transients(self) -> bool:
        """Whether one block's activation peak sits in a single attention, FFN or recurrent sub-layer.

        True for block-exclusive stacks, and for gated-delta-net stacks, whose
        forward frees the mixer's tensors before the MLP beside it runs.
        """
        return self.block_exclusive_layers or self.mamba_gated_delta

    @property
    def moe_layers(self) -> int:
        """Layers carrying the routed experts (all of them by default)."""
        if self.n_moe_layers is not None:
            return self.n_moe_layers
        return self.n_layers

    @property
    def mlp_layers(self) -> int:
        """Layers carrying a dense MLP.

        Every layer by default; 0 on a MoE model unless a block-exclusive
        layout says otherwise (routed experts replace the dense FFN).
        """
        if self.n_mlp_layers is not None:
            return self.n_mlp_layers
        return 0 if self.is_moe else self.n_layers

    @classmethod
    def from_hf_config(cls, config: dict[str, Any]) -> Self:
        """Build a :class:`ModelArch` from a raw HF ``config.json`` dict.

        Unwraps a nested ``text_config`` or ``llm_config``. Tower params come from
        :func:`multimodal_tower_params`.
        """
        text_cfg = config.get("text_config") or config.get("llm_config") or config
        decoder = _decoder_fields(text_cfg)
        moe = _moe_fields(text_cfg)
        layout = _layout_fields(
            text_cfg, decoder["n_layers"], decoder["hidden_size"], config
        )
        return cls(
            **decoder,
            **moe,
            **layout,
            vllm_3d_moe_lora=config.get("model_type") in VLLM_3D_MOE_LORA_MODEL_TYPES,
        )


class WeightVariant(BaseModel):
    """One loadable form of a model's weights.

    Variants differ by tower stripping only.
    """

    model_config = ConfigDict(frozen=True)

    name: str = "base"
    stripped_multimodal: bool = False


class ModelSpec(BaseModel):
    """A model: geometry plus weight variants."""

    model_config = ConfigDict(frozen=True)

    model_id: str
    arch: ModelArch
    checkpoint_dtype: WeightDtype = "bf16"
    # Exact parameter count when known (e.g. from safetensors metadata).
    # ``None`` falls back to the analytic count from the geometry.
    n_params: int | None = None
    variants: tuple[WeightVariant, ...] = (WeightVariant(),)

    def variant(self, name: str) -> WeightVariant:
        for v in self.variants:
            if v.name == name:
                return v
        msg = (
            f"Model {self.model_id!r} has no weight variant {name!r}; "
            f"available: {[v.name for v in self.variants]}"
        )
        raise KeyError(msg)


class DeviceSpec(BaseModel):
    """Device capacity plus flags that pick which formula applies."""

    model_config = ConfigDict(frozen=True)

    total_bytes: int
    # Usable capacity override. ``None`` uses 95% of ``total_bytes`` for
    # training (allocator fragmentation) and the whole card for generation.
    # CUDA context is a demand-side component.
    available_bytes: int | None = None
    name: str | None = None
    # Bytes a bare CUDA context costs, before any tensor allocation.
    # ``None`` uses :data:`CUDA_CONTEXT_BYTES_DEFAULT`.
    cuda_context_bytes: int | None = None
    # CUDA compute capability ``(major, minor)``; ``None`` when unknown.
    compute_capability: tuple[int, int] | None = None

    @property
    def context_bytes(self) -> int:
        """This device's CUDA context."""
        if self.cuda_context_bytes is not None:
            return self.cuda_context_bytes
        if self.name is not None:
            return CUDA_CONTEXT_BYTES_BY_DEVICE.get(
                self.name, CUDA_CONTEXT_BYTES_DEFAULT
            )
        return CUDA_CONTEXT_BYTES_DEFAULT

    @property
    def nccl_bytes(self) -> int:
        """NCCL memory on each rank of a multi-rank trainer on this device."""
        if self.name is None:
            return NCCL_BYTES_DEFAULT
        return NCCL_BYTES_BY_DEVICE.get(self.name, NCCL_BYTES_DEFAULT)

    @property
    def usable_bytes(self) -> int:
        """Capacity the predicted peak is checked against.

        CUDA context is a demand-side component (NVML device-used). The 5%
        band is allocator fragmentation: a peak predicted at 100% of
        ``total_bytes`` exceeds usable capacity.
        """
        if self.available_bytes is not None:
            return self.available_bytes
        return int(self.total_bytes * 0.95)

    @property
    def generation_usable_bytes(self) -> int:
        """Capacity the generation peak is checked against.

        vLLM claims ``gpu_memory_utilization`` of the card once at startup, so
        the fragmentation band does not apply; the whole card is the limit.
        """
        if self.available_bytes is not None:
            return self.available_bytes
        return self.total_bytes


class TrainingSettings(BaseModel):
    """Training settings that affect GPU memory.

    Gradients scale with LoRA adapter parameters. Flat data parallel keeps
    AdamW state on the GPU. FSDP with ``optim_cpu_offload`` keeps it on CPU
    except during ``step()``. ``micro_batch_size`` is the rows of each
    gradient micro-batch.
    """

    model_config = ConfigDict(frozen=True)

    algorithm: Algorithm = "grpo"
    group_size: int = 8
    # Completion rows per ``learn`` call (``prompts x group_size``).
    # ``None`` assumes one prompt group.
    trajectories_per_update: int | None = None
    # Full context budget (prompt + completion), i.e. the worst-case
    # sequence length for activation and logprob tensors.
    max_model_len: int = 1024
    lora_rank: int = 16
    lora_target_scope: LoraTargetScope = "all-linear"
    lora_dropout: float = 0.0
    # Packed MoE expert matrices adapted via PEFT ``target_parameters``
    # (e.g. ``mixer.experts.up_proj``). Each targeted matrix adds a
    # per-expert rank decomposition on every MoE layer.
    lora_packed_target_matrices: int = 0
    # Packed-expert adapter path. ``"contracted"``: split LoRA (top-k gather,
    # per-expert ``F.linear`` + ``cat``, fp32 GEMMs), skipping the
    # effective-weight copy. ``"materialized"``: PEFT builds ``W_eff``.
    packed_moe_dispatch: PackedMoeDispatch = "materialized"
    # KL coefficient. At ``beta=0`` the estimate drops the reference row;
    # the warnings note that the fused pass still builds it.
    beta: float = 0.001
    # A frozen copy of the actor adapter used for reference logprobs.
    # ``False`` routes reference rows through the (immutable) base: one
    # adapter copy; the reference forward still runs.
    use_separate_reference_adapter: bool = True
    weight_dtype: WeightDtype = "bf16"
    gradient_checkpointing: bool = True
    # Save backward-saved activations to pinned host RAM.
    activation_offload: bool = False
    # Fused-logprob tile rows. ``None`` auto-tunes to a ~256 MiB fp32 logit
    # workspace, clamped to [128, 4096].
    chunk_rows: int | None = None
    # The loss runs Liger's fused kernel instead of the standard
    # fused-linear-logprob path.
    use_liger_loss: bool = True
    # Widest vLLM sampling support rollout records per token (``top_k``); 0
    # records none. The no-grad pass scores those tokens from their support's
    # lm_head rows.
    sampling_support_width: int = Field(default=0, ge=0)
    # The gradient pass scores supported tokens from their support rows too.
    # PPO and REINFORCE's Liger loss scores the full vocabulary.
    support_scored_loss: bool = False
    # Data-parallel trainer GPUs. The estimate is per GPU; this is the
    # shard group size.
    n_training_gpus: int = 1
    # None is flat data parallel: a full weight copy on each GPU.
    fsdp: FSDPConfig | None = None
    # Completion rows per gradient forward and backward.
    micro_batch_size: int = Field(default=1, ge=1)
    # PPO only: actor and critic rows share one gradient forward and backward.
    fuse_actor_critic_pass: bool = False
    # Images in each completion row, and the width and height the environment
    # renders them at.
    images_per_row: int = Field(default=0, ge=0)
    image_width: int = Field(default=0, ge=0)
    image_height: int = Field(default=0, ge=0)
    # LoRA adapts the vision tower, which then runs with autograd.
    vision_tower_lora: bool = False
    # Episodes train as segment rows, which lockstep ranks balance before learning.
    segmented_rows: bool = False
    # The main rank's async checkpoint snapshot also holds the Adam moments.
    checkpoint_optimizer: bool = False
    # Async rollout: the next rollout batch waits in host memory during learn.
    async_rollout: bool = False
    # Gradient forward packs into one row. The estimate still prices padded rows.
    sequence_packing: bool = False

    @property
    def trajectories(self) -> int:
        """Completion rows one learner GPU sees per update.

        ``None`` assumes a single prompt group. Under data parallelism the
        update shards across learner GPUs.
        """
        total = self.trajectories_per_update or self.group_size
        return max(-(-total // self.n_training_gpus), 1)

    @property
    def images_per_update(self) -> int:
        """Images one learner GPU holds per update."""
        return self.trajectories * self.images_per_row

    @property
    def shard_gpus(self) -> int:
        """GPUs one copy of the weights is sharded across.

        FSDP shards inside each ``shard_group_size`` group; the groups replicate.
        """
        if self.fsdp is None or self.fsdp.shard_group_size is None:
            return self.n_training_gpus
        return min(self.fsdp.shard_group_size, self.n_training_gpus)

    @property
    def uses_reference(self) -> bool:
        """Whether a reference policy is consulted.

        SFT has none. DPO always has one (beta is the preference temperature).
        Elsewhere the reference supplies the KL term, so ``beta=0`` drops it
        from the loss; the fused no-grad pass still builds the row.
        """
        if self.algorithm == "sft":
            return False
        if self.algorithm == "dpo":
            return True
        return self.beta != 0.0

    @property
    def uses_critic(self) -> bool:
        """PPO trains a critic: its own LoRA adapter plus a value head."""
        return self.algorithm == "ppo"

    @property
    def uses_generation_engine(self) -> bool:
        """Whether the run starts a vLLM engine.

        SFT and DPO train from a fixed dataset and start no engine.
        """
        return self.algorithm not in ("sft", "dpo")

    @property
    def has_nograd_pass(self) -> bool:
        """Whether a no-grad logprob pass exists as its own instant.

        SFT backpropagates its targets directly.
        """
        return self.algorithm != "sft"

    @property
    def uses_selective_logprobs(self) -> bool:
        """Whether loss and no-grad passes run Liger's selective kernel.

        PPO, REINFORCE, and the GRPO family run selective per-token logprobs;
        DPO and SFT run full-tile kernels either way.
        """
        return self.use_liger_loss and self.algorithm in (
            "grpo",
            "gspo",
            "cispo",
            "ppo",
            "reinforce",
        )

    @property
    def grad_graph_rows(self) -> int:
        """Autograd graphs live together at the loss backward.

        DPO keeps chosen and rejected graphs until one joint loss; others
        backpropagate one graph.
        """
        return 2 if self.algorithm == "dpo" else 1

    @property
    def grad_forward_rows(self) -> int:
        """Rows per gradient forward: the micro-batch, doubled when PPO fuses."""
        return self.micro_batch_size * (2 if self.fuse_actor_critic_pass else 1)

    @property
    def lora_casts_recompute_only(self) -> bool:
        """Whether PEFT fp32 input casts exist only in checkpoint recompute.

        Non-SFT forwards disable the casts; recompute after that context
        exits, so one block's casts are sized by the gradient row. SFT
        keeps casts on the original forward.
        """
        return self.algorithm != "sft"

    @property
    def n_trained_adapters(self) -> int:
        """Adapters carrying gradients and optimizer state.

        The reference adapter is frozen; PPO additionally trains a critic.
        """
        return 2 if self.uses_critic else 1

    @property
    def n_resident_adapters(self) -> int:
        """Adapter copies held on the device.

        The reference adapter stays resident whenever it exists: it is
        created at init, so beta=0 skips its forward only.
        """
        n = 1
        if self.use_separate_reference_adapter and self.algorithm != "sft":
            n += 1
        if self.uses_critic:
            n += 1
        return n

    @model_validator(mode="after")
    def _check_chunk_rows(self) -> Self:
        if self.chunk_rows is not None and self.chunk_rows < 1:
            msg = f"chunk_rows must be at least 1, got {self.chunk_rows}"
            raise ValueError(msg)
        if (
            self.fsdp is not None
            and self.fsdp.cpu_offload
            and self.fsdp.optim_cpu_offload
        ):
            msg = "FSDP cpu_offload and optim_cpu_offload are mutually exclusive"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _check_fuse_actor_critic_pass(self) -> Self:
        if self.fuse_actor_critic_pass and self.algorithm != "ppo":
            msg = (
                "fuse_actor_critic_pass=True requires algorithm='ppo', "
                f"got {self.algorithm!r}"
            )
            raise ValueError(msg)
        return self


class GenerationSettings(BaseModel):
    """vLLM settings (``VLLMConfig``) plus workload shape."""

    model_config = ConfigDict(frozen=True)

    gpu_memory_utilization: float = OMITTED_GPU_MEMORY_UTILIZATION
    max_num_seqs: int = COLOCATED_MAX_NUM_SEQS
    max_model_len: int = 1024
    # Worst-case prompt length. Prefill dominates generation memory; decode
    # adds one token per sequence per step, so the activation peak tracks
    # prompt tokens in flight. ``None`` assumes the whole context is prompt.
    max_prompt_len: int | None = None
    # ``None`` takes vLLM's async-engine default, capped at
    # ``max_num_seqs * max_model_len``.
    max_num_batched_tokens: int | None = None
    # ``limit_mm_per_prompt["image"]``. ``None`` takes vLLM's unset-modality limit.
    max_images_per_prompt: int | None = None
    # Pin the KV pool size. ``None`` lets vLLM derive it from
    # ``gpu_memory_utilization``.
    kv_cache_memory_bytes: int | None = None
    # ``True`` skips CUDA-graph capture, at some decode-throughput cost.
    # ``None``/``False`` keeps graphs on.
    enforce_eager: bool | None = None
    max_lora_rank: int = OMITTED_MAX_LORA_RANK
    max_loras: int = 1
    # Adapter ``target_parameters`` on packed experts. Any switches the
    # engine to vLLM's 3D fused-MoE LoRA layout, as AgileRL patches the
    # model's ``is_3d_moe_weight`` flag on.
    lora_packed_target_matrices: int = 0
    weight_dtype: WeightDtype = "bf16"
    # ``fp8`` stores the engine weights at one byte. KV and activations stay
    # at ``weight_dtype``.
    weight_quantization: Literal["none", "fp8"] = "none"
    # Ranks that share one engine copy. 1 is a single-GPU engine.
    tensor_parallel_size: int = Field(default=1, ge=1)
    # Name of the :class:`WeightVariant` the engine loads (vLLM may load a
    # different variant than the trainer, e.g. a tower-stripped export).
    weight_variant: str = "base"
    # Worst-case concurrent requests, ``prompts_in_flight * group_size``.
    # ``None`` assumes the schedule limit (``max_num_seqs``) is saturated.
    concurrent_requests: int | None = None

    @property
    def concurrency(self) -> int:
        if self.concurrent_requests is None:
            return self.max_num_seqs
        return min(self.concurrent_requests, self.max_num_seqs)

    @property
    def prompt_len(self) -> int:
        return min(self.max_prompt_len or self.max_model_len, self.max_model_len)

    @model_validator(mode="after")
    def _check_utilization(self) -> Self:
        if not 0.0 < self.gpu_memory_utilization <= 1.0:
            msg = (
                "gpu_memory_utilization must be in (0, 1], got "
                f"{self.gpu_memory_utilization}"
            )
            raise ValueError(msg)
        return self


class RunConfig(BaseModel):
    """A complete sizing question: model + devices + settings."""

    model_config = ConfigDict(frozen=True)

    model: ModelSpec
    train_device: DeviceSpec
    # The rollout engine lives on its own device; each phase bar sizes one
    # device with no cross-phase residuals.
    gen_device: DeviceSpec
    training: TrainingSettings = Field(default_factory=TrainingSettings)
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    # When True, include per-process job overhead in each phase bar.
    orchestrated: bool = False

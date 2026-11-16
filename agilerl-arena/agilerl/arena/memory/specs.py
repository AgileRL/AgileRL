# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Input specs for the GPU memory estimator.

Pydantic models for architecture, device, and settings. JSON-serializable
and torch-free.
"""

from __future__ import annotations

import re
from typing import Any, Literal, NamedTuple, TypedDict

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from agilerl.arena.memory.config_fields import (
    GATED_MLP_TOWER_MODEL_TYPES,
    VisionResize,
    attention_sink_layers,
    decoder_fields,
    dense_ffn_layers,
    encoder_params,
    ffn_layer_mix,
    first_int,
    indexed_layers,
    is_recurrent_layer,
    layer_mix,
    layer_type_list,
    moe_fields,
    tower_configs,
    vision_geometry_fields,
)
from agilerl.arena.memory.resolve.attention import resolve_attn_implementation
from agilerl.arena.memory.resolve.lora import OMITTED_MAX_LORA_RANK
from agilerl.arena.memory.resolve.sequences import (
    COLOCATED_MAX_NUM_SEQS,
    OMITTED_GPU_MEMORY_UTILIZATION,
)
from agilerl.arena.models.fsdp import FSDPConfig
from agilerl.arena.models.model_info import LORA_EXCLUDED_MODULES

GiB = 1024**3
MiB = 1024**2

# Default Mamba-2 chunked-scan chunk length.
MAMBA_CHUNK_SIZE_DEFAULT = 256

# Model types with a norm before and after both the attention and the MLP.
SANDWICH_NORM_MODEL_TYPES = frozenset({"gemma4_text"})

# Compute capability where AgileRL skips the fused combined Mamba kernel for
# ``UNFUSED_MAMBA_SCAN_MODEL_TYPES``; it returns NaN there.
UNFUSED_MAMBA_SCAN_CAPABILITY = (8, 9)

# Outer config model types whose vLLM class sets ``is_3d_moe_weight``, so the
# engine wraps experts in the 3D fused-MoE LoRA layout.
VLLM_3D_MOE_LORA_MODEL_TYPES = frozenset(
    {"gpt_oss", "interns1_pro", "qwen3_5_moe", "qwen3_omni_moe", "qwen3_vl_moe"}
)

# Outer config model types whose HF class lists its tower encoder blocks in
# ``_no_split_modules``, so FSDP shards each block as its own unit.
TOWER_BLOCK_UNIT_MODEL_TYPES = frozenset({"gemma4", "qwen3_5", "qwen3_5_moe"})
# Of those, the types whose vision patch embedder is a unit of its own too.
TOWER_EMBEDDER_UNIT_MODEL_TYPES = frozenset({"gemma4"})

# Families with a catalog training tensor-parallel plan. Under fsdp.tp > 1 the
# trainer leaves other families' decoder layers and activations unsharded; an
# untied lm_head is sharded for every family.
TENSOR_PARALLEL_MODEL_TYPES = frozenset(
    {
        "deepseek_v4",
        "gemma4",
        "gemma4_text",
        "glm5_next",
        "glm5_next_text",
        "gpt_oss",
        "granitemoe",
        "granitemoehybrid",
        "mimo_v2",
        "nemotron_h",
        "nemotron_h_omni",
        "qwen2",
        "qwen3",
        "qwen3_5",
        "qwen3_5_text",
        "qwen3_5_moe",
        "qwen3_5_moe_text",
    }
)
# Tower config model types whose family training TP plan splits each encoder
# block's fused qkv, output projection and MLP, and the patch merger's MLP.
TENSOR_PARALLEL_TOWER_MODEL_TYPES = frozenset(
    {"glm5_next_vision", "qwen3_5", "qwen3_5_moe"}
)
# Model types whose vLLM config keeps the SSM state cache in fp32 unless the
# checkpoint names a dtype. Other families keep it in the model dtype.
FP32_SSM_STATE_MODEL_TYPES = frozenset({"nemotron_h", "nemotron_h_omni"})
# Model types whose single KV projection is read as both key and value.
SHARED_KV_MODEL_TYPES = frozenset({"deepseek_v4"})
# Model types whose context-parallel attention keeps queries on the rank's
# shard and all-gathers the key side over the whole sequence. Other
# attention runs Ulysses: whole sequence, 1/cp of the heads.
CONTEXT_PARALLEL_KEY_GATHER_MODEL_TYPES = frozenset(
    {"deepseek_v4", "glm5_next", "glm5_next_text"}
)
# DeepSeek-V4 ``compress_ratios`` entries: 0 is a sliding-only layer, 4 a
# compressed sparse attention (CSA) layer, 128 a heavily compressed (HCA) layer.
CSA_COMPRESS_RATIO = 4
HCA_COMPRESS_RATIO = 128

# CUDA context bytes keyed by torch device name: device memory in use after
# the first torch allocation.
CUDA_CONTEXT_BYTES_BY_DEVICE: dict[str, int] = {
    "NVIDIA A100-SXM4-40GB": 1268 * MiB,
    "NVIDIA A100-SXM4-80GB": 1268 * MiB,
    # One-rank FSDP non-torch memory less the job and trainer library overhead.
    "NVIDIA H100 80GB HBM3": 1440 * MiB,
    "NVIDIA L4": 694 * MiB,
}
# Unlisted devices use the largest listed value.
CUDA_CONTEXT_BYTES_DEFAULT = 1268 * MiB

# NCCL bytes one rank holds for a 4-rank process group on one node, with NCCL
# cuMem off as the trainer runs FSDP, keyed by torch device name. L4 is
# scaled from its cuMem-on world group by the A100 cuMem-off ratio.
NCCL_GROUP_BYTES_BY_DEVICE: dict[str, int] = {
    "NVIDIA A100-SXM4-40GB": 820 * MiB,
    "NVIDIA A100-SXM4-80GB": 820 * MiB,
    "NVIDIA H100 80GB HBM3": 950 * MiB,
    "NVIDIA L4": 125 * MiB,
}
NCCL_GROUP_BYTES_DEFAULT = 820 * MiB

# Bytes per element.
DTYPE_BYTES: dict[str, float] = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
}

WeightDtype = Literal["fp32", "bf16", "fp16"]
TORCH_DTYPE_NAMES: dict[str, WeightDtype] = {
    "float32": "fp32",
    "bfloat16": "bf16",
    "float16": "fp16",
}
Algorithm = Literal["grpo", "gspo", "cispo", "ppo", "reinforce", "dpo", "sft"]
# Manifest algorithm names -> the estimator's algorithm identifiers. The
# estimator keys its reference/critic residency rules on these, so an
# unmapped LLM algorithm is a hard error.
ALGORITHM_NAMES: dict[str, Algorithm] = {
    "GRPO": "grpo",
    "GSPO": "gspo",
    "CISPO": "cispo",
    "OPD": "grpo",
    "LLMPPO": "ppo",
    "LLMREINFORCE": "reinforce",
    "DPO": "dpo",
    "SFT": "sft",
}
LoraTargetScope = Literal["all-linear", "attention-only"]
# LoRA factor a tensor-parallel plan splits: LoRA A of a row-parallel linear
# (``in``), LoRA B of a column-parallel one (``out``), or neither.
LoraSplit = Literal["none", "in", "out"]


class LoraLinear(NamedTuple):
    """``count`` linears of one shape that LoRA wraps.

    ``split`` names the factor training tensor parallel splits (``"in"``:
    LoRA A, ``"out"``: LoRA B). ``target`` is the module name a tower LoRA
    target matches; decoder linears leave it empty.
    """

    in_features: int
    out_features: int
    count: int
    split: LoraSplit = "none"
    target: str = ""


PackedMoeDispatch = Literal["materialized", "contracted"]
# One layer of a block-exclusive stack: an attention, FFN (MLP or MoE) or Mamba layer.
BlockKind = Literal["attention", "ffn", "mamba"]
# One layer of a block-exclusive stack, with MLP and MoE told apart.
LayerKind = Literal["attention", "mamba", "mlp", "moe"]


def multimodal_tower_params(config: dict[str, Any]) -> int:
    """Lower-bound params treating vision/audio configs as transformer encoders.

    :param config: Hugging Face model config.
    :return: Tower parameter count.
    """
    return sum(encoder_params(cfg)[1] for cfg in tower_configs(config))


def tensor_parallel_tower_params(config: dict[str, Any]) -> int:
    """Tower params the family's training TP plan splits: encoder qkv, proj and MLP, merger MLP.

    Output-projection biases stay whole. A ``projection_intermediate_size``
    merger is a bias-free SwiGLU; otherwise the merger maps the merged patch
    width to itself, then to ``out_hidden_size``.

    :param config: Hugging Face model config.
    :return: Split parameter count; zero for towers without a TP plan.
    """
    total = 0
    for cfg in tower_configs(config):
        if cfg.get("model_type") not in TENSOR_PARALLEL_TOWER_MODEL_TYPES:
            continue
        h = int(cfg["hidden_size"])
        inter = int(cfg["intermediate_size"])
        out = int(cfg["out_hidden_size"])
        qkv = 3 * h * h + (3 * h if cfg.get("attention_bias", True) else 0)
        inputs = 2 if cfg["model_type"] in GATED_MLP_TOWER_MODEL_TYPES else 1
        mlp = inputs * (h * inter + inter) + inter * h
        if cfg.get("projection_intermediate_size"):
            merger = 3 * out * int(cfg["projection_intermediate_size"])
        else:
            merged = h * int(cfg["spatial_merge_size"]) ** 2
            merger = merged * merged + merged + merged * out
        total += int(cfg["depth"]) * (qkv + h * h + mlp) + merger
    return total


def multimodal_tower_block_params(config: dict[str, Any]) -> int:
    """Encoder-block share of :func:`multimodal_tower_params` that FSDP shards as units.

    :param config: Hugging Face model config.
    :return: Block parameter count; zero for models FSDP does not wrap per block.
    """
    if config.get("model_type") not in TOWER_BLOCK_UNIT_MODEL_TYPES:
        return 0
    return sum(encoder_params(cfg)[0] for cfg in tower_configs(config))


def tower_block_units(config: dict[str, Any]) -> int:
    """FSDP units the trainer's vision tower holds, all ahead of the decoder layers.

    :param config: Hugging Face model config.
    :return: Vision blocks, plus the patch embedder where it is a unit; zero
        for models FSDP does not wrap per tower block.
    """
    model_type = config.get("model_type")
    if model_type not in TOWER_BLOCK_UNIT_MODEL_TYPES:
        return 0
    vision = config["vision_config"]
    blocks = first_int(vision, "depth", "num_hidden_layers")
    return blocks + (1 if model_type in TOWER_EMBEDDER_UNIT_MODEL_TYPES else 0)


def _vision_fields(config: dict[str, Any]) -> dict[str, Any]:
    """Vision tower geometry and tower LoRA linears; empty without a tower."""
    geometry = vision_geometry_fields(config)
    if not geometry:
        return {}
    return {**geometry, "tower_lora_linears": _tower_lora_linears(config)}


def _tower_lora_linears(config: dict[str, Any]) -> tuple[LoraLinear, ...]:
    """Tower and connector linears, by the target name tower LoRA matches.

    Split factors follow the family's training tensor-parallel plan: Qwen3.5
    splits the vision blocks and merger, Nemotron Omni the encoder MLP, and
    Gemma 4 keeps its towers whole. Empty for families without tower LoRA.
    """
    model_type = config.get("model_type")
    if model_type in {"qwen3_5", "qwen3_5_moe"}:
        return _qwen_vision_lora_linears(config["vision_config"])
    if model_type == "gemma4":
        text_hidden = int(config["text_config"]["hidden_size"])
        return _gemma4_vision_lora_linears(
            config["vision_config"], text_hidden
        ) + _gemma4_audio_lora_linears(config.get("audio_config"), text_hidden)
    if model_type == "nemotron_h_omni":
        return _radio_vision_lora_linears(config)
    return ()


def _qwen_vision_lora_linears(vision: dict[str, Any]) -> tuple[LoraLinear, ...]:
    """Qwen3.5 vision blocks and patch merger."""
    h = int(vision["hidden_size"])
    layers = int(vision["depth"])
    inter = int(vision["intermediate_size"])
    merged = h * int(vision["spatial_merge_size"]) ** 2
    return (
        LoraLinear(h, 3 * h, layers, "out", "qkv"),
        LoraLinear(h, inter, layers, "out", "linear_fc1"),
        LoraLinear(inter, h, layers, "in", "linear_fc2"),
        LoraLinear(merged, merged, 1, "out", "linear_fc1"),
        LoraLinear(merged, int(vision["out_hidden_size"]), 1, "in", "linear_fc2"),
    )


def _gemma4_vision_lora_linears(
    vision: dict[str, Any], text_hidden: int
) -> tuple[LoraLinear, ...]:
    """Gemma 4 vision encoder, patch embedder and vision embedder."""
    h = int(vision["hidden_size"])
    layers = int(vision["num_hidden_layers"])
    inter = int(vision["intermediate_size"])
    q = int(vision["num_attention_heads"]) * int(vision["head_dim"])
    kv = int(vision["num_key_value_heads"]) * int(vision["head_dim"])
    # Encoder linears sit inside clippable wrappers as ``linear``.
    return (
        LoraLinear(h, q, layers, target="linear"),
        LoraLinear(h, kv, 2 * layers, target="linear"),
        LoraLinear(q, h, layers, target="linear"),
        LoraLinear(h, inter, 2 * layers, target="linear"),
        LoraLinear(inter, h, layers, target="linear"),
        LoraLinear(3 * int(vision["patch_size"]) ** 2, h, 1, target="input_proj"),
        LoraLinear(h, text_hidden, 1, target="embedding_projection"),
    )


def _gemma4_audio_lora_linears(
    audio: dict[str, Any] | None, text_hidden: int
) -> tuple[LoraLinear, ...]:
    """Gemma 4 conformer audio tower and audio embedder; empty without one."""
    if not audio:
        return ()
    h = int(audio["hidden_size"])
    layers = int(audio["num_hidden_layers"])
    channels = audio["subsampling_conv_channels"]
    out = int(audio["output_proj_dims"])
    # Two feed-forwards (h -> 4h -> h), the light conv's start (h -> 2h) and
    # end, and attention q/k/v/post, all clippable ``linear`` wrappers.
    return (
        LoraLinear(h, 4 * h, 2 * layers, target="linear"),
        LoraLinear(4 * h, h, 2 * layers, target="linear"),
        LoraLinear(h, 2 * h, layers, target="linear"),
        LoraLinear(h, h, 5 * layers, target="linear"),
        LoraLinear(h, h, layers, target="relative_k_proj"),
        LoraLinear(
            int(channels[0]) // 4 * int(channels[1]), h, 1, target="input_proj_linear"
        ),
        LoraLinear(h, out, 1, target="output_proj"),
        LoraLinear(out, text_hidden, 1, target="embedding_projection"),
    )


def _radio_vision_lora_linears(config: dict[str, Any]) -> tuple[LoraLinear, ...]:
    """Nemotron Omni RADIO encoder, patch embedders and projector."""
    vision = config["vision_config"]
    h = int(vision["hidden_size"])
    layers = int(vision["num_hidden_layers"])
    inter = int(h * float(vision["mlp_ratio"]))
    patch = int(vision["num_channels"]) * int(vision["patch_size"]) ** 2
    merged = h * round(1 / config["downsample_ratio"]) ** 2
    projector = int(config["projector_hidden_size"])
    linears = [
        *(
            LoraLinear(h, h, layers, target=name)
            for name in ("q_proj", "k_proj", "v_proj", "o_proj")
        ),
        LoraLinear(h, inter, layers, "out", "fc1"),
        LoraLinear(inter, h, layers, "in", "fc2"),
        LoraLinear(patch, h, 1, target="patch_projection"),
        LoraLinear(merged, projector, 1, target="linear_1"),
        LoraLinear(
            projector, int(config["llm_config"]["hidden_size"]), 1, target="linear_2"
        ),
    ]
    frames = vision.get("video_temporal_patch_size")
    if frames:
        linears.append(
            LoraLinear(int(frames) * patch, h, 1, target="video_patch_projection")
        )
    return tuple(linears)


def _layer_kind(entry: str) -> LayerKind:
    """One ``layers_block_type`` entry or ``hybrid_override_pattern`` character."""
    if entry in ("moe", "E"):
        return "moe"
    if entry in ("mlp", "-"):
        return "mlp"
    if entry == "*" or ("attention" in entry and not is_recurrent_layer(entry)):
        return "attention"
    return "mamba"


def _layer_kinds(text_cfg: dict[str, Any]) -> tuple[LayerKind, ...]:
    """Each layer's kind, in order, for a block-exclusive stack; empty otherwise."""
    entries = layer_type_list(text_cfg) or list(
        text_cfg.get("hybrid_override_pattern") or ""
    )
    if not {"moe", "mlp", "E", "-"} & set(entries):
        return ()
    return tuple(_layer_kind(entry) for entry in entries)


def _layer_mixers(text_cfg: dict[str, Any]) -> tuple[BlockKind, ...]:
    """Each layer's mixer, in order, when every layer runs one beside its FFN; empty otherwise."""
    entries = layer_type_list(text_cfg)
    if _layer_kinds(text_cfg) or not any(is_recurrent_layer(e) for e in entries):
        return ()
    return tuple("mamba" if is_recurrent_layer(e) else "attention" for e in entries)


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
    kda: bool
    ssm_state_dtype: WeightDtype


def _mamba_state_geometry(text_cfg: dict[str, Any], hidden: int) -> MambaGeometry:
    """Mamba-2 / gated-delta-net state dimensions, normalised to one shape.

    vLLM stores two tensors per recurrent layer per slot: a causal-conv
    window of ``(conv_dim, d_conv - 1)`` and a recurrent state of
    ``(n_heads, d_head, d_state)``. The families differ only in ``conv_dim``.
    Kimi delta attention (KDA) is a gated delta net with equal q/k/v heads.
    """
    kda = text_cfg.get("linear_attn_config")
    if kda:
        heads = int(kda["num_heads"])
        dim = int(kda["head_dim"])
        return {
            "d_state": dim,
            "d_conv": int(kda["short_conv_kernel_size"]),
            "n_groups": 0,
            "n_heads": heads,
            "d_head": dim,
            "conv_dim": 3 * heads * dim,
            "chunk_size": int(text_cfg.get("chunk_size") or MAMBA_CHUNK_SIZE_DEFAULT),
            "gated_delta": True,
            "kda": True,
            "ssm_state_dtype": TORCH_DTYPE_NAMES[
                text_cfg.get("mamba_ssm_dtype") or "bfloat16"
            ],
        }
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
            "ssm_state_dtype": TORCH_DTYPE_NAMES[
                text_cfg.get("mamba_ssm_dtype") or "bfloat16"
            ],
        }

    d_state = first_int(text_cfg, "mamba_d_state", "ssm_state_size", "mamba_state_dim")
    n_groups = first_int(text_cfg, "mamba_n_groups", "mamba_num_groups", "n_groups")
    n_heads = first_int(text_cfg, "mamba_n_heads", "mamba_num_heads")
    d_head = first_int(text_cfg, "mamba_d_head", "mamba_head_dim")
    d_inner = int(
        text_cfg.get("mamba_d_ssm")
        or n_heads * d_head
        or first_int(text_cfg, "mamba_expand", "expand", default=2) * hidden
    )
    return {
        "d_state": d_state,
        "d_conv": first_int(text_cfg, "mamba_d_conv", "conv_kernel"),
        "n_groups": n_groups,
        "n_heads": n_heads,
        "d_head": d_head,
        "conv_dim": d_inner + 2 * n_groups * d_state,
        "chunk_size": first_int(
            text_cfg, "chunk_size", "mamba_chunk_size", default=MAMBA_CHUNK_SIZE_DEFAULT
        ),
        "ssm_state_dtype": TORCH_DTYPE_NAMES[
            text_cfg.get("mamba_ssm_cache_dtype")
            or (
                "float32"
                if text_cfg.get("model_type") in FP32_SSM_STATE_MODEL_TYPES
                else "bfloat16"
            )
        ],
    }


def has_routed_experts(config: dict[str, Any]) -> bool:
    """Whether a raw HF ``config.json`` describes an MoE model (:attr:`ModelArch.is_moe`).

    Unwraps a nested ``text_config`` or ``llm_config``.
    """
    text_cfg = config.get("text_config") or config.get("llm_config") or config
    n_experts = moe_fields(text_cfg)["n_experts"]
    return n_experts is not None and n_experts > 1


def _attention_fields(text_cfg: dict[str, Any], n_layers: int) -> dict[str, Any]:
    """Low-rank attention, sparse-attention indexer, KV compressor and hyper-connection fields."""
    ratios = [int(r) for r in (text_cfg.get("compress_ratios") or [])[:n_layers]]
    unknown = set(ratios) - {0, CSA_COMPRESS_RATIO, HCA_COMPRESS_RATIO}
    if unknown:
        msg = f"compress_ratios has unknown entries {sorted(unknown)}"
        raise ValueError(msg)
    n_csa = ratios.count(CSA_COMPRESS_RATIO)
    kv_lora_rank = int(text_cfg.get("kv_lora_rank") or 0)
    return {
        "q_lora_rank": int(text_cfg.get("q_lora_rank") or 0),
        "kv_lora_rank": kv_lora_rank,
        "qk_rope_head_dim": (
            int(text_cfg.get("qk_rope_head_dim") or 0) if kv_lora_rank else 0
        ),
        "v_head_dim": int(text_cfg.get("v_head_dim") or 0),
        "shared_kv": text_cfg.get("model_type") in SHARED_KV_MODEL_TYPES,
        "context_parallel_key_gather": text_cfg.get("model_type")
        in CONTEXT_PARALLEL_KEY_GATHER_MODEL_TYPES,
        "o_groups": int(text_cfg.get("o_groups") or 0),
        "o_lora_rank": int(text_cfg.get("o_lora_rank") or 0),
        "n_csa_layers": n_csa,
        "n_hca_layers": ratios.count(HCA_COMPRESS_RATIO),
        "n_indexed_layers": indexed_layers(text_cfg, n_layers, n_csa),
        "index_n_heads": int(text_cfg.get("index_n_heads") or 0),
        "index_head_dim": int(text_cfg.get("index_head_dim") or 0),
        "index_topk": int(text_cfg.get("index_topk") or 0),
        "index_kpool": int(text_cfg.get("index_kpool") or 0),
        "hc_mult": int(text_cfg.get("hc_mult") or 1),
        "attention_sink_layers": attention_sink_layers(text_cfg, n_layers),
    }


class CheckpointQuantization(BaseModel):
    """Bytes per parameter of a pre-quantized checkpoint's linears and routed experts, scales included."""

    model_config = ConfigDict(frozen=True)

    # Attention, dense MLP, shared-expert and recurrent-mixer linears.
    linear_bytes: float
    expert_bytes: float
    # Module names the checkpoint keeps at its dtype on attention layers and
    # on recurrent-mixer layers.
    unquantized_attention: frozenset[str] = frozenset()
    unquantized_mixer: frozenset[str] = frozenset()


def _checkpoint_quantization(
    config: dict[str, Any], text_cfg: dict[str, Any], n_layers: int
) -> CheckpointQuantization | None:
    """Stored bytes of a block-scaled FP8 checkpoint, whose routed experts may be MXFP4.

    :param config: Outer HF config.
    :param text_cfg: Decoder config.
    :param n_layers: Decoder layers; listed modules on later (MTP) layers are not served.
    :return: ``None`` without an FP8 ``quantization_config``.
    """
    quant = config.get("quantization_config") or text_cfg.get("quantization_config")
    if not quant or quant.get("quant_method") != "fp8":
        return None
    # Block-scaled FP8 keeps one scale per weight block: E8M0 under
    # ``scale_fmt: ue8m0``, else fp32. MXFP4 packs two values per byte with
    # one E8M0 scale per 32 values. vLLM keeps both as stored on SM 9.0+.
    scale_bytes = 1.0 if quant.get("scale_fmt") == "ue8m0" else 4.0
    block = quant.get("weight_block_size")
    linear = 1.0 + (scale_bytes / (block[0] * block[1]) if block else 0.0)
    fp4_experts = (
        text_cfg.get("expert_dtype") == "fp4" or quant.get("store_dtype") == "mxfp4"
    )
    layer_types = layer_type_list(text_cfg)
    attention: set[str] = set()
    mixer: set[str] = set()
    # vLLM reads ``ignored_layers`` first.
    for module in (
        quant.get("ignored_layers") or quant.get("modules_to_not_convert") or []
    ):
        match = re.search(r"(?:^|\.)layers\.(\d+)\.self_attn\.(\w+)$", module)
        if match is None or int(match.group(1)) >= n_layers:
            continue
        index, name = int(match.group(1)), match.group(2)
        if layer_types and is_recurrent_layer(layer_types[index]):
            mixer.add(name)
        else:
            attention.add(name)
    return CheckpointQuantization(
        linear_bytes=linear,
        expert_bytes=0.5 + 1.0 / 32 if fp4_experts else linear,
        unquantized_attention=frozenset(attention),
        unquantized_mixer=frozenset(mixer),
    )


def _layout_fields(
    text_cfg: dict[str, Any],
    n_layers: int,
    hidden: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Layer mix, mamba geometry, mixed-width fields, and multimodal towers.

    :param config: Outer HF config, for ``vision_config`` / ``audio_config``.
    """
    n_attention, n_mamba = layer_mix(text_cfg, n_layers)
    n_mlp, n_moe = ffn_layer_mix(text_cfg)
    n_kv_shared = int(text_cfg.get("num_kv_shared_layers") or 0)
    geo = _mamba_state_geometry(text_cfg, hidden) if n_mamba else {}
    return {
        "n_attention_layers": n_attention,
        "n_mamba_layers": n_mamba,
        "n_mlp_layers": n_mlp,
        "n_moe_layers": n_moe,
        "dense_ffn_layers": dense_ffn_layers(text_cfg, n_layers),
        "layer_kinds": _layer_kinds(text_cfg),
        "layer_mixers": _layer_mixers(text_cfg),
        **{f"mamba_{key}": value for key, value in geo.items()},
        "global_head_dim": text_cfg.get("global_head_dim") or None,
        "n_kv_shared_layers": n_kv_shared,
        "kv_shared_full_layers": sum(
            1
            for entry in layer_type_list(text_cfg)[n_layers - n_kv_shared :]
            if "sliding" not in entry
        ),
        "double_wide_mlp": bool(text_cfg.get("use_double_wide_mlp", False)),
        "per_layer_input_dim": int(text_cfg.get("hidden_size_per_layer_input") or 0),
        "norms_per_block": (
            4 if text_cfg.get("model_type") in SANDWICH_NORM_MODEL_TYPES else 2
        )
        + (1 if text_cfg.get("hidden_size_per_layer_input") else 0),
        "per_layer_input_vocab": int(text_cfg.get("vocab_size_per_layer_input") or 0),
        "multimodal_tower_params": multimodal_tower_params(config),
        "multimodal_tower_block_params": multimodal_tower_block_params(config),
        "tower_block_units": tower_block_units(config),
        "tensor_parallel_tower_params": tensor_parallel_tower_params(config),
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
    # Heads the KV cache stores; one under MLA, whose latent every head shares.
    n_kv_heads: int
    head_dim: int
    vocab_size: int
    tied_embeddings: bool = False
    # Caps KV growth per sequence on the windowed layers.
    sliding_window: int | None = None
    # Ignored when ``sliding_window`` is None.
    sliding_window_layer_fraction: float = 1.0
    attn_bias: bool = False
    # The trainer runs FlexAttention on the language model.
    flex_attention: bool = False
    # SwiGLU (gate + up + down). Ungated MLPs use two matrices (up + down).
    gated_mlp: bool = True
    # Dense and shared MLPs fuse gate and up into one linear (Granite ``input_linear``).
    fused_gate_up_mlp: bool = False
    # Trainer gated MLP saves the activated gate as its own tensor.
    stock_gated_mlp: bool = False
    # Trainer RMSNorm saves an fp32 copy of its input for backward.
    fp32_norm_inputs: bool = False
    # RMSNorms over the hidden state in one decoder block, including a
    # per-layer-input norm.
    norms_per_block: int = 2
    # Per-head Q/K norm upcasts Q to fp32 inside the block.
    qk_norm: bool = False
    # Trains Mamba layers on HF's unfused path on compute capability 8.9.
    unfused_mamba_scan_family: bool = False
    # Training tensor parallel shards this family's decoder layers.
    tensor_parallel_plan: bool = False
    # Training tensor parallel splits the tied token embedding by hidden column.
    tensor_parallel_tied_embedding: bool = False
    # ``q_proj`` doubles its query rows into a per-head output gate.
    attention_output_gate: bool = False
    # Trainer attention runs eager, which materialises every head's score matrix.
    eager_attention: bool = False

    # Low-rank query: hidden -> ``q_lora_rank`` -> heads. Zero projects
    # queries straight from hidden.
    q_lora_rank: int = 0
    # Multi-head latent attention (MLA): keys and values expand per head from
    # a ``kv_lora_rank`` latent, which the KV cache stores beside one shared
    # ``qk_rope_head_dim`` rope key.
    kv_lora_rank: int = 0
    qk_rope_head_dim: int = 0
    # Value head width; zero is ``head_dim``.
    v_head_dim: int = 0
    # One KV projection is read as both key and value.
    shared_kv: bool = False
    # Context-parallel attention gathers the key side over the whole
    # sequence for the rank's local queries.
    context_parallel_key_gather: bool = False
    # Grouped low-rank output: each of ``o_groups`` head groups projects to
    # ``o_lora_rank``, then one linear mixes the groups to hidden.
    o_groups: int = 0
    o_lora_rank: int = 0
    # DeepSeek-V4 compressed layers. Every layer keeps the sliding window;
    # CSA layers also cache one entry per CSA_COMPRESS_RATIO tokens from two
    # overlapping windows, HCA layers one per HCA_COMPRESS_RATIO tokens.
    n_csa_layers: int = 0
    n_hca_layers: int = 0
    # Attention layers running a sparse-attention (DSA) indexer, and its
    # heads, key width and kept entries per query. DeepSeek-V4's indexer
    # compresses its keys at the CSA rate.
    n_indexed_layers: int = 0
    index_n_heads: int = 0
    index_head_dim: int = 0
    index_topk: int = 0
    # Keys per pool a GLM DSA indexer scores together; zero scores no pools.
    index_kpool: int = 0
    # Parallel residual streams each block mixes in and out
    # (hyper-connections); 1 is a plain residual.
    hc_mult: int = 1
    # Attention layers that learn one sink logit per query head.
    attention_sink_layers: int = 0

    # None for dense models. Weights/optimizer scale with the total expert
    # count; activations scale with the active (routed) count.
    n_experts: int | None = None
    n_experts_per_tok: int | None = None
    expert_intermediate_size: int | None = None
    n_shared_experts: int = 0
    # ``None`` falls back to ``expert_intermediate_size``.
    shared_expert_intermediate_size: int | None = None
    # ``fsdp.save_moe_collectives`` saves the shared expert's tensor-parallel
    # reduce: the shared expert sits in the MoE module and the block recompute
    # reaches it (Qwen-MoE's sigmoid gate reads it). Granite's shared MLP sits
    # beside the MoE module; Nemotron-H's trails it.
    shared_expert_reduce_saved: bool = False
    # Width routed experts read and write (Nemotron-H latent MoE). Each MoE
    # layer projects hidden -> latent before the experts and back after.
    # ``None`` runs experts at ``hidden_size``.
    moe_latent_size: int | None = None

    # Each layer is exactly one of Mamba, attention, MLP or MoE. ``None``
    # keeps the every-layer default.
    n_mlp_layers: int | None = None
    n_moe_layers: int | None = None
    # MoE-stack layers whose FFN is a dense MLP at ``intermediate_size``
    # beside the mixer; the others route experts.
    dense_ffn_layers: int = 0
    # Layer kinds in order when the config lists them; empty when unknown.
    layer_kinds: tuple[LayerKind, ...] = ()
    # Each layer's mixer in order when every layer runs one beside its FFN
    # and the config lists them; empty otherwise.
    layer_mixers: tuple[BlockKind, ...] = ()
    # Packed experts run in row chunks on the self-routing grouped-GEMM path.
    chunked_routed_experts: bool = False
    # Packed experts use gpt-oss's transposed layout; see TRANSPOSED_EXPERT_MODEL_TYPES.
    transposed_experts: bool = False
    # Expert LoRA outside a checkpointed block saves only the routed rows'
    # inputs and indices and recomputes the up-projection in backward.
    routed_expert_recompute: bool = False
    # vLLM serves this model's expert LoRA in the 3D fused-MoE layout.
    vllm_3d_moe_lora: bool = False
    # Module names the trainer's adapter never wraps; the engine's LoRA slots
    # still cover them.
    lora_excluded_modules: frozenset[str] = frozenset()
    # Stored bytes of a pre-quantized checkpoint, which vLLM serves as stored.
    # The trainer dequantizes it at load.
    checkpoint_quantization: CheckpointQuantization | None = None

    multimodal_tower_params: int = 0
    # Tower params inside encoder blocks FSDP shards as their own units.
    multimodal_tower_block_params: int = 0
    # Tower FSDP units ahead of the decoder layers; the checkpoint stride
    # counts from the first of them.
    tower_block_units: int = 0
    # Tower params the training TP plan splits across ranks.
    tensor_parallel_tower_params: int = 0
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
    # Tower and connector linears ``lora_config.tower_target_modules`` can wrap.
    tower_lora_linears: tuple[LoraLinear, ...] = ()

    # Head dim on full-attention layers when it differs from the sliding
    # layers', which widens q/k/v on the full-attention layers.
    global_head_dim: int | None = None
    # KV heads on full-attention layers when they differ from the sliding
    # layers' ``n_kv_heads``.
    global_n_kv_heads: int | None = None
    # Trailing layers that reuse an earlier layer's KV and add nothing to
    # the cache. They have no k/v projections.
    n_kv_shared_layers: int = 0
    # Full-attention layers among the ``n_kv_shared_layers``.
    kv_shared_full_layers: int = 0
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
    # Gated-delta layers are KDA: separate q/k/v projections and low-rank
    # (hidden -> d_head -> heads) forget and output gates.
    mamba_kda: bool = False

    @property
    def is_hybrid_ssm(self) -> bool:
        return self.n_mamba_layers > 0

    @property
    def low_rank_attention(self) -> bool:
        """Whether attention projects through a query, KV latent or grouped output bottleneck."""
        return bool(self.q_lora_rank or self.kv_lora_rank or self.o_groups)

    @property
    def residual_width(self) -> int:
        """Width of the residual stream between blocks."""
        return self.hc_mult * self.hidden_size

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
    def mean_kv_heads(self) -> float:
        """KV heads per attention layer, averaged over sliding and full-attention layers."""
        if self.global_n_kv_heads is None:
            return self.n_kv_heads
        f = self.sliding_window_layer_fraction if self.sliding_window else 0.0
        return f * self.n_kv_heads + (1.0 - f) * self.global_n_kv_heads

    @property
    def mean_qkv_dim(self) -> int:
        """q+k+v width per token, averaged over layer types."""
        if self.kv_lora_rank:
            return self.peak_qkv_dim
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
        counterpart of :attr:`mean_qkv_dim`. MLA expands keys and values to
        every query head.
        """
        if self.kv_lora_rank:
            return self.n_heads * (
                2 * self.head_dim + (self.v_head_dim or self.head_dim)
            )
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

        True for block-exclusive stacks, and for stacks where each layer runs
        one mixer (attention or recurrent), whose forward frees the mixer's
        tensors before the FFN beside it runs.
        """
        one_mixer_per_layer = (
            self.n_mamba_layers > 0
            and self.n_attention_layers is not None
            and self.n_attention_layers + self.n_mamba_layers == self.n_layers
        )
        return self.block_exclusive_layers or one_mixer_per_layer

    @property
    def moe_layers(self) -> int:
        """Layers carrying the routed experts (all but the dense FFN layers by default)."""
        if self.n_moe_layers is not None:
            return self.n_moe_layers
        return self.n_layers - self.dense_ffn_layers

    @property
    def mlp_layers(self) -> int:
        """Layers carrying a dense MLP.

        Every layer by default; on a MoE model only its dense FFN layers
        unless a block-exclusive layout says otherwise (routed experts
        replace the dense FFN).
        """
        if self.n_mlp_layers is not None:
            return self.n_mlp_layers
        return self.dense_ffn_layers if self.is_moe else self.n_layers

    @classmethod
    def from_hf_config(cls, config: dict[str, Any]) -> Self:
        """Build a :class:`ModelArch` from a raw HF ``config.json`` dict.

        Unwraps a nested ``text_config`` or ``llm_config``. Tower params come from
        :func:`multimodal_tower_params`.
        """
        text_cfg = config.get("text_config") or config.get("llm_config") or config
        decoder = decoder_fields(text_cfg)
        moe = moe_fields(text_cfg)
        layout = _layout_fields(
            text_cfg, decoder["n_layers"], decoder["hidden_size"], config
        )
        return cls(
            **decoder,
            **moe,
            **layout,
            **_attention_fields(text_cfg, decoder["n_layers"]),
            tensor_parallel_plan=text_cfg.get("model_type")
            in TENSOR_PARALLEL_MODEL_TYPES,
            vllm_3d_moe_lora=config.get("model_type") in VLLM_3D_MOE_LORA_MODEL_TYPES,
            lora_excluded_modules=LORA_EXCLUDED_MODULES.get(
                config.get("model_type"), frozenset()
            ),
            checkpoint_quantization=_checkpoint_quantization(
                config, text_cfg, decoder["n_layers"]
            ),
            eager_attention=resolve_attn_implementation(
                config.get("model_type"), cuda=True
            )
            == "eager",
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
    # GPUs on this device's node. ``None`` puts every trainer rank on one node.
    gpus_per_node: int | None = Field(default=None, ge=1)
    # Dense bf16 tensor-core peak and HBM bandwidth; ``None`` when unknown.
    peak_bf16_flops: float | None = None
    memory_bytes_per_second: float | None = None
    # All-reduce bus bandwidth and small-message latency between GPUs of one
    # node; ``None`` when unknown.
    all_reduce_bytes_per_second: float | None = None
    all_reduce_latency_seconds: float | None = None
    # GPUs of one node talk over NVLink, not PCIe.
    nvlink: bool = False
    # Host RAM one GPU's process may hold, from
    # :func:`~agilerl.arena.memory.formulas.host_bytes_per_gpu`; ``None`` skips
    # the host check.
    host_bytes_per_gpu: int | None = Field(default=None, ge=0)

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
    def nccl_group_bytes(self) -> int:
        """NCCL memory one rank holds for a 4-rank process group on one node."""
        if self.name is None:
            return NCCL_GROUP_BYTES_DEFAULT
        return NCCL_GROUP_BYTES_BY_DEVICE.get(self.name, NCCL_GROUP_BYTES_DEFAULT)

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
    # the warnings note that the LLMPPO / LLMREINFORCE fused pass still builds it.
    beta: float = 0.001
    # A frozen copy of the actor adapter used for reference logprobs.
    # ``False`` routes reference rows through the (immutable) base: one
    # adapter copy; the reference forward still runs.
    use_separate_reference_adapter: bool = True
    weight_dtype: WeightDtype = "bf16"
    # ``True`` saves only block inputs and recomputes each block in backward.
    # ``False`` keeps every block's forward tensors until its backward.
    # ``None`` until resolve_training_memory picks it; the estimate needs it set.
    gradient_checkpointing: bool | None = True
    # Save backward-saved activations to pinned host RAM.
    activation_offload: bool = False
    # Widest vLLM sampling support rollout records per token (``top_k``); 0
    # records none. The gradient and no-grad passes score those tokens from
    # their support's lm_head rows.
    sampling_support_width: int = Field(default=0, ge=0)
    # Data-parallel trainer GPUs. The estimate is per GPU; this is the
    # shard group size.
    n_training_gpus: int = 1
    # None is flat data parallel: a full weight copy on each GPU.
    fsdp: FSDPConfig | None = None
    # MiB of one routed-expert row chunk under DDP; FSDP reads
    # ``fsdp.routed_expert_chunk_mib``. None is the 64 MiB default.
    routed_expert_chunk_mib: int | None = Field(default=None, ge=1)
    # Completion rows per gradient forward and backward.
    micro_batch_size: int = Field(default=1, ge=1)
    # Images in each completion row, and the width and height the environment
    # renders them at.
    images_per_row: int = Field(default=0, ge=0)
    image_width: int = Field(default=0, ge=0)
    image_height: int = Field(default=0, ge=0)
    # ``lora_config.tower_target_modules``: tower and connector module names
    # LoRA wraps. Any target runs the vision tower with autograd.
    tower_lora_targets: frozenset[str] = frozenset()
    # Episodes train as segment rows, which lockstep ranks balance before learning.
    segmented_rows: bool = False
    # The main rank's async checkpoint snapshot also holds the Adam moments.
    checkpoint_optimizer: bool = False
    # Async rollout: the next rollout batch waits in host memory during learn,
    # and the main rank holds the policy adapter's NIXL send buffer.
    async_rollout: bool = False
    # Async weight sync pushes merged weights: the main rank's NIXL send
    # buffer holds two rounds of merged weights, not the adapter.
    merged_lora_sync: bool = False
    # Engine ``tensor_parallel_size`` when weight sync runs over
    # ``nccl_reshard``, else None. With ``merged_lora_sync`` the refit holds
    # :class:`~agilerl.arena.memory.lora_formulas.NcclReshardRefitBytes`, not
    # a NIXL send buffer.
    nccl_reshard_engine_tensor_parallel: int | None = Field(default=None, ge=1)
    # Gradient forward packs into one row. A packed micro-batch of full-length
    # rows holds as many tokens as the padded one, so the estimate prices padded rows.
    sequence_packing: bool = False
    # Packed micro-batches hold at most this many tokens; the estimate prices
    # one packed row of this length. Requires ``sequence_packing``.
    micro_batch_tokens: int | None = Field(default=None, ge=1)

    @property
    def data_parallel_gpus(self) -> int:
        """Ranks that each hold a distinct batch shard.

        Tensor- and context-parallel ranks share one shard, so they do not
        divide it again.
        """
        if self.fsdp is None:
            return self.n_training_gpus
        return max(self.n_training_gpus // (self.fsdp.tp * self.fsdp.cp), 1)

    @property
    def trajectories(self) -> int:
        """Completion rows one learner GPU sees per update.

        ``None`` assumes a single prompt group. Under data parallelism the
        update shards across :attr:`data_parallel_gpus`.
        """
        total = self.trajectories_per_update or self.group_size
        return max(-(-total // self.data_parallel_gpus), 1)

    @property
    def images_per_update(self) -> int:
        """Images one learner GPU holds per update."""
        return self.trajectories * self.images_per_row

    @property
    def vision_tower_lora(self) -> bool:
        """LoRA adapts a tower, so the vision tower runs with autograd."""
        return bool(self.tower_lora_targets)

    @property
    def shard_group_gpus(self) -> int:
        """Ranks in one HSDP shard group; the groups replicate."""
        if self.fsdp is None or self.fsdp.shard_group_size is None:
            return self.n_training_gpus
        return min(self.fsdp.shard_group_size, self.n_training_gpus)

    @property
    def dense_fsdp_gpus(self) -> int:
        """GPUs one copy of the dense weights is sharded across.

        FSDP shards over ``dp_shard * cp``, the shard group divided by ``tp``;
        TP weights split ``tp`` ways first. HSDP groups replicate.
        """
        if self.fsdp is None:
            return self.n_training_gpus
        return self.shard_group_gpus // self.fsdp.tp

    @property
    def expert_shard_gpus(self) -> int:
        """GPUs packed expert weights are sharded across.

        Expert parallel times leftover FSDP covers the whole shard group, not
        the dense ``dp_shard * cp`` group.
        """
        if self.fsdp is None or self.fsdp.ep <= 1:
            return self.dense_fsdp_gpus
        return self.shard_group_gpus

    @property
    def uses_reference(self) -> bool:
        """Whether a reference policy is consulted.

        SFT has none. DPO always has one (beta is the preference temperature).
        Elsewhere the reference supplies the KL term, so ``beta=0`` drops it.
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

        SFT backpropagates its targets directly. REINFORCE takes old log-probs
        from the rollout, so it scores only for the KL penalty.
        """
        if self.algorithm == "reinforce":
            return self.uses_reference
        return self.algorithm != "sft"

    @property
    def grad_graph_rows(self) -> int:
        """Autograd graphs live together at the loss backward.

        DPO keeps chosen and rejected graphs until one joint loss; others
        backpropagate one graph.
        """
        return 2 if self.algorithm == "dpo" else 1

    @property
    def checkpoint_every_n_blocks(self) -> int:
        """Checkpoint stride. Flat data parallel checkpoints every block.

        :raises ValueError: When FSDP checkpointing is on and the stride is unset.
        """
        if self.fsdp is None or not self.gradient_checkpointing:
            return 1
        stride = self.fsdp.checkpoint_every_n_blocks
        if stride is None:
            msg = (
                "fsdp.checkpoint_every_n_blocks unset with gradient_checkpointing "
                "on; resolve_training_memory picks it before the estimate."
            )
            raise ValueError(msg)
        return stride

    @property
    def grad_forward_rows(self) -> int:
        """Rows per gradient forward: the micro-batch, doubled for PPO's fused actor and critic rows."""
        return self.micro_batch_size * (2 if self.algorithm == "ppo" else 1)

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
    def _check_fsdp_offload(self) -> Self:
        if (
            self.fsdp is not None
            and self.fsdp.cpu_offload
            and self.fsdp.optim_cpu_offload
        ):
            msg = "FSDP cpu_offload and optim_cpu_offload are mutually exclusive"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _check_micro_batch_tokens(self) -> Self:
        if self.micro_batch_tokens is None:
            return self
        if not self.sequence_packing:
            msg = "micro_batch_tokens requires sequence_packing=True"
            raise ValueError(msg)
        if self.micro_batch_tokens < self.max_model_len:
            msg = (
                f"micro_batch_tokens ({self.micro_batch_tokens}) is below the "
                f"longest row (max_model_len={self.max_model_len})"
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
    max_lora_rank: int = OMITTED_MAX_LORA_RANK
    # A teacher adapter sits in a second slot beside the policy's.
    served_loras: int = 1
    # Adapter ``target_parameters`` on packed experts. Any switches the
    # engine to vLLM's 3D fused-MoE LoRA layout, as AgileRL patches the
    # model's ``is_3d_moe_weight`` flag on.
    lora_packed_target_matrices: int = 0
    # Target scope of the trainer adapter a Ray rollout engine receives into
    # its NIXL buffer on every GPU. ``None`` when generation runs without
    # rollout engines.
    lora_sync_scope: LoraTargetScope | None = None
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
    # An async rollout episode can run more than one turn.
    multi_turn: bool = False
    # The rollout algorithm's ``min_p``.
    min_p: float = 0.0
    # Rollout requests carry a ``logit_bias`` masking the image token.
    logit_bias: bool = False
    # ``vllm_engine_args.speculative_config`` is set (``True``) or null
    # (``False``); ``None`` takes the rollout engine's multi-turn default.
    speculative_drafting: bool | None = None
    # A Ray rollout engine serves generation and ``vllm_engine_args`` does not
    # set ``lora_dual_stream: false``; it runs a dense model's LoRA layers on
    # two CUDA streams (``VLLM_LORA_ENABLE_DUAL_STREAM``).
    lora_dual_stream: bool = False
    # The engine serves merged weights with LoRA off: no adapter slots or LoRA
    # streams. With ``lora_sync_scope`` set, its NIXL buffer on every GPU holds
    # one round of merged weights.
    merged_lora_sync: bool = False
    # Trainer ranks one weight's rows are sharded across when weight sync runs
    # over ``nccl_reshard``, else None. With
    # ``merged_lora_sync`` each GPU holds one misc broadcast bucket; bulk
    # weights land in place.
    nccl_reshard_trainer_shards: int | None = Field(default=None, ge=1)
    # A distillation teacher scores prompt log-probs, a runtime buffer the
    # start-up profile does not size.
    scores_prompt_logprobs: bool = False
    # An unset ``gpu_memory_utilization`` on a scoring engine shrinks by that
    # buffer, so the engine peaks where a non-scoring one does.
    reserves_prompt_logprobs: bool = False

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

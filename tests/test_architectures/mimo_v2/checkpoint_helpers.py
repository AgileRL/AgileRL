# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tiny ``mimo_v2`` Hub checkpoints: FP8 / MXFP4 quantizers and their exact dequantized references."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from transformers.integrations.mxfp4 import convert_moe_packed_tensors

FP8_BLOCK = 128
MXFP4_BLOCK = 32
IMAGE_TOKEN_ID = 120
PATCH_SIZE = 2
TEMPORAL_PATCH_SIZE = 2
MERGE = 2
# Three channels, two frames, 2x2 pixels.
PATCH_DIM = 3 * TEMPORAL_PATCH_SIZE * PATCH_SIZE**2
LANGUAGE_PREFIX = "model.language_model."
VISION_PREFIX = "model.visual."
FP4_E2M1 = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ]
)


def tiny_hub_config() -> dict[str, Any]:
    """Two-layer ``mimo_v2`` ``config.json``: dense full-attention layer 0, MoE SWA layer 1."""
    return {
        "architectures": ["MiMoV2ForCausalLM"],
        "auto_map": {
            "AutoConfig": "configuration_mimo_v2.MiMoV2Config",
            "AutoModelForCausalLM": "modeling_mimo_v2.MiMoV2ForCausalLM",
        },
        "model_type": "mimo_v2",
        "add_full_attention_sink_bias": False,
        "add_swa_attention_sink_bias": True,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "attention_value_scale": 0.707,
        "bos_token_id": None,
        "dtype": "bfloat16",
        "eos_token_id": 126,
        "head_dim": 48,
        "hidden_act": "silu",
        "hidden_size": 64,
        "hybrid_layer_pattern": [0, 1],
        "initializer_range": 0.02,
        "intermediate_size": 96,
        "layernorm_epsilon": 1e-6,
        "max_position_embeddings": 256,
        "moe_intermediate_size": 32,
        "moe_layer_freq": [0, 1],
        "n_group": 1,
        "n_routed_experts": 8,
        "n_shared_experts": None,
        "norm_topk_prob": True,
        "num_attention_heads": 4,
        "num_experts_per_tok": 2,
        "num_hidden_layers": 2,
        "num_key_value_heads": 2,
        "num_nextn_predict_layers": 1,
        "pad_token_id": 127,
        "partial_rotary_factor": 0.334,
        "rope_theta": 10000000.0,
        "routed_scaling_factor": None,
        "scoring_func": "sigmoid",
        "sliding_window": 8,
        "swa_head_dim": 48,
        "swa_num_attention_heads": 4,
        "swa_num_key_value_heads": 4,
        "swa_rope_theta": 10000.0,
        "swa_v_head_dim": 32,
        "tie_word_embeddings": False,
        "topk_group": 1,
        "topk_method": "noaux_tc",
        "v_head_dim": 32,
        "vocab_size": 128,
        "image_token_id": IMAGE_TOKEN_ID,
        "processor_config": {
            "image_min_pixels": 4 * PATCH_SIZE**2,
            "image_max_pixels": 64 * PATCH_SIZE**2,
            "patch_size": PATCH_SIZE,
            "merge_size": MERGE,
            "temporal_patch_size": TEMPORAL_PATCH_SIZE,
        },
        "vision_config": {
            "depth": 3,
            "hidden_size": 32,
            "intermediate_size": 48,
            "hidden_act": "silu",
            "num_heads": 2,
            "num_key_value_heads": 1,
            "in_chans": 3,
            "patch_size": PATCH_SIZE,
            "temporal_patch_size": TEMPORAL_PATCH_SIZE,
            "spatial_merge_size": MERGE,
            "out_hidden_size": 64,
            "fullatt_block_indexes": [0],
            "vit_window_attn_types": [-1, 0, 1],
            "visual_token_window_size": 4,
            "use_sink": True,
            "num_query_groups": 1,
            "window_size": -1,
        },
        "quantization_config": {
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "ignored_layers": [
                "model.layers.0.self_attn.o_proj",
                "model.layers.1.self_attn.o_proj",
            ],
            "mxfp4_block_size": MXFP4_BLOCK,
            "quant_method": "fp8",
            "store_dtype": "mxfp4",
            "weight_block_size": [FP8_BLOCK, FP8_BLOCK],
        },
    }


def fp8_quantize(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FP8 e4m3 with one scale per 128x128 block, the grid padded to whole blocks.

    :return: FP8 weight, ``weight_scale_inv`` grid, and the exact fp32 dequantized weight.
    """
    rows, cols = weight.shape
    padded_rows = math.ceil(rows / FP8_BLOCK) * FP8_BLOCK
    padded_cols = math.ceil(cols / FP8_BLOCK) * FP8_BLOCK
    padded = torch.zeros(padded_rows, padded_cols)
    padded[:rows, :cols] = weight
    tiles = padded.view(
        padded_rows // FP8_BLOCK, FP8_BLOCK, padded_cols // FP8_BLOCK, FP8_BLOCK
    )
    scale = tiles.abs().amax(dim=(1, 3)) / 448.0
    quantized = (tiles / scale[:, None, :, None]).to(torch.float8_e4m3fn)
    dequantized = (quantized.float() * scale[:, None, :, None]).view(
        padded_rows, padded_cols
    )
    return (
        quantized.view(padded_rows, padded_cols)[:rows, :cols].contiguous(),
        scale,
        dequantized[:rows, :cols].contiguous(),
    )


def fp8_quantize_presharded(
    weight: torch.Tensor, chunks: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FP8-quantize each of ``chunks`` row chunks with its own padded scale grid."""
    parts = [fp8_quantize(chunk) for chunk in weight.chunk(chunks)]
    return tuple(torch.cat([part[index] for part in parts]) for index in range(3))


def mxfp4_quantize(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """OCP MXFP4: nearest E2M1 per value, one E8M0 exponent per 32 values, low nibble first.

    :return: Packed ``uint8`` ``[out, in/2]``, ``uint8`` scales ``[out, in/32]``, and
        the fp32 dequantized weight from transformers' GPT-OSS MXFP4 decoder.
    """
    rows, cols = weight.shape
    groups = weight.view(rows, cols // MXFP4_BLOCK, MXFP4_BLOCK)
    exponent = torch.floor(torch.log2(groups.abs().amax(-1))) - 2
    scaled = groups / torch.exp2(exponent)[..., None]
    index = (
        (scaled[..., None] - FP4_E2M1).abs().argmin(-1).view(rows, cols).to(torch.uint8)
    )
    packed = index[:, 0::2] | (index[:, 1::2] << 4)
    scale = (exponent + 127).to(torch.uint8)
    dequantized = convert_moe_packed_tensors(
        packed.view(1, rows, cols // MXFP4_BLOCK, MXFP4_BLOCK // 2),
        scale.view(1, rows, cols // MXFP4_BLOCK),
        dtype=torch.float32,
    ).transpose(1, 2)[0]
    return packed, scale, dequantized.contiguous()


def interleave_qkv(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, chunks: int
) -> torch.Tensor:
    """Fuse ``Q``, ``K``, ``V`` as ``[Q_0|K_0|V_0|...|Q_n|K_n|V_n]`` row chunks."""
    return torch.cat(
        [
            torch.cat(parts)
            for parts in zip(
                query.chunk(chunks), key.chunk(chunks), value.chunk(chunks), strict=True
            )
        ]
    )


def random_native_state(model: torch.nn.Module, seed: int) -> dict[str, torch.Tensor]:
    """bf16-representable random weights for every parameter and persistent buffer."""
    generator = torch.Generator().manual_seed(seed)
    state = {}
    for name, tensor in model.state_dict().items():
        if name.endswith("norm.weight"):
            values = 1 + 0.1 * torch.randn(tensor.shape, generator=generator)
        elif name.endswith("e_score_correction_bias"):
            values = 0.05 * torch.randn(tensor.shape, generator=generator)
        elif name.endswith("sinks"):
            values = torch.randn(tensor.shape, generator=generator)
        else:
            values = 0.08 * torch.randn(tensor.shape, generator=generator)
        state[name] = values.to(torch.bfloat16).float()
    return state


def hub_checkpoint(
    native: dict[str, torch.Tensor], config: Any
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Quantize a native vision-language state dict into ``mimo_v2`` Hub tensors.

    :param native: Native ``MiMoV2ForConditionalGeneration`` fp32 state dict.
    :param config: The native config, for layer types and head counts.
    :return: Hub tensors (with MTP and audio keys) and the native state dict
        holding each tensor's exact dequantized value.
    """
    hub = {
        "visual." + name.removeprefix(VISION_PREFIX): tensor.bfloat16()
        for name, tensor in native.items()
        if name.startswith(VISION_PREFIX)
    }
    native = {
        "model." + name.removeprefix(LANGUAGE_PREFIX)
        if name.startswith(LANGUAGE_PREFIX)
        else name: tensor
        for name, tensor in native.items()
        if not name.startswith(VISION_PREFIX)
    }
    text_reference = text_hub_checkpoint(native, config.text_config, hub)
    reference = {
        LANGUAGE_PREFIX + name.removeprefix("model.")
        if name.startswith("model.")
        else name: tensor
        for name, tensor in text_reference.items()
    }
    reference.update(
        {
            VISION_PREFIX + name.removeprefix("visual."): tensor.float()
            for name, tensor in hub.items()
            if name.startswith("visual.")
        }
    )
    return hub, reference


def text_hub_checkpoint(
    native: dict[str, torch.Tensor], config: Any, hub: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Add the quantized language-model Hub tensors of a text-only state dict to ``hub``.

    :param native: ``MiMoV2FlashForCausalLM`` fp32 state dict.
    :param config: The language-model config.
    :param hub: Hub tensors, extended in place.
    :return: ``native`` with each tensor's exact dequantized value.
    """
    reference = dict(native)
    chunks = config.num_key_value_heads
    for layer, layer_type in enumerate(config.layer_types):
        prefix = f"model.layers.{layer}"
        attention = f"{prefix}.self_attn"
        qkv = interleave_qkv(
            native[f"{attention}.q_proj.weight"],
            native[f"{attention}.k_proj.weight"],
            native[f"{attention}.v_proj.weight"],
            chunks,
        )
        quantized, scale, dequantized = fp8_quantize_presharded(qkv, chunks)
        hub[f"{attention}.qkv_proj.weight"] = quantized
        hub[f"{attention}.qkv_proj.weight_scale_inv"] = scale
        kv_heads = chunks * (2 if layer_type == "sliding_attention" else 1)
        sizes = [
            config.num_attention_heads // chunks * config.head_dim,
            kv_heads // chunks * config.head_dim,
            kv_heads // chunks * config.v_head_dim,
        ]
        pieces = [chunk.split(sizes) for chunk in dequantized.chunk(chunks)]
        for index, projection in enumerate(("q_proj", "k_proj", "v_proj")):
            reference[f"{attention}.{projection}.weight"] = torch.cat(
                [piece[index] for piece in pieces]
            )
        hub[f"{attention}.o_proj.weight"] = native[
            f"{attention}.o_proj.weight"
        ].bfloat16()
        if layer_type == "sliding_attention":
            hub[f"{attention}.attention_sink_bias"] = native[
                f"{attention}.sinks"
            ].bfloat16()
        if config.mlp_layer_types[layer] == "dense":
            for projection in ("gate_proj", "up_proj", "down_proj"):
                name = f"{prefix}.mlp.{projection}.weight"
                quantized, scale, dequantized = fp8_quantize(native[name])
                hub[name] = quantized
                hub[f"{name}_scale_inv"] = scale
                reference[name] = dequantized
        else:
            experts = f"{prefix}.mlp.experts"
            gate_up = native[f"{experts}.gate_up_proj"]
            down = native[f"{experts}.down_proj"]
            dequantized_gate_up, dequantized_down = [], []
            for expert in range(gate_up.shape[0]):
                gate, up = gate_up[expert].chunk(2)
                fused = []
                for projection, weight in (("gate_proj", gate), ("up_proj", up)):
                    packed, scale, dequantized = mxfp4_quantize(weight)
                    hub[f"{experts}.{expert}.{projection}.weight"] = packed
                    hub[f"{experts}.{expert}.{projection}.weight_scale"] = scale
                    fused.append(dequantized)
                dequantized_gate_up.append(torch.cat(fused))
                packed, scale, dequantized = mxfp4_quantize(down[expert])
                hub[f"{experts}.{expert}.down_proj.weight"] = packed
                hub[f"{experts}.{expert}.down_proj.weight_scale"] = scale
                dequantized_down.append(dequantized)
            reference[f"{experts}.gate_up_proj"] = torch.stack(dequantized_gate_up)
            reference[f"{experts}.down_proj"] = torch.stack(dequantized_down)
            for name in ("weight", "e_score_correction_bias"):
                hub[f"{prefix}.mlp.gate.{name}"] = native[
                    f"{prefix}.mlp.gate.{name}"
                ].bfloat16()
        for norm in ("input_layernorm", "post_attention_layernorm"):
            hub[f"{prefix}.{norm}.weight"] = native[
                f"{prefix}.{norm}.weight"
            ].bfloat16()
    for name in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"):
        hub[name] = native[name].bfloat16()
    hub["model.mtp.layers.0.input_layernorm.weight"] = torch.ones(config.hidden_size)
    hub["audio_encoder.layers.0.weight"] = torch.ones(4)
    hub["speech_embeddings.0.weight"] = torch.ones(4, 4)
    return reference


def write_hub_checkpoint(directory: Path, hub: dict[str, torch.Tensor]) -> None:
    """Write ``config.json`` and the tensors over two safetensors shards with an index."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(tiny_hub_config()))
    names = sorted(hub)
    shards = {
        "model-00001-of-00002.safetensors": names[: len(names) // 2],
        "model-00002-of-00002.safetensors": names[len(names) // 2 :],
    }
    weight_map = {}
    for filename, keys in shards.items():
        save_file(
            {key: hub[key].contiguous() for key in keys}, str(directory / filename)
        )
        weight_map.update(dict.fromkeys(keys, filename))
    index = {
        "metadata": {"save_format": "mxfp4", "tp_size": 2},
        "weight_map": weight_map,
    }
    (directory / "model.safetensors.index.json").write_text(json.dumps(index))

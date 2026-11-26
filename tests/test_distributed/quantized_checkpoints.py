# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tiny FP8 / FP4 DeepSeek-V4 and GLM-5.3 checkpoints in the Hub key layout, with exact dequantized references."""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
)

from agilerl.arena.models.model_info import SUPPORTED_MODEL_INFO

DSV4_HUB_ID = "deepseek-ai/DeepSeek-V4-Flash-0731"
GLM_HUB_ID = "zai-org/GLM-5.3-Flash"
FP8_BLOCK = 128
FP4_BLOCK = 32
FP8_E4M3_MAX = 448.0
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
INPUT_IDS = torch.tensor([[3, 17, 42, 5, 99, 64, 8, 21, 77, 1, 120, 33]])

# Native DeepSeek-V4 name -> Hub name, applied in order.
DSV4_HUB_RENAMES = (
    (r"^model\.", ""),
    (r"^embed_tokens\.weight$", "embed.weight"),
    (r"^lm_head\.weight$", "head.weight"),
    (r"^hc_head\.hc_(fn|base|scale)$", r"hc_head_\1"),
    (r"\.input_layernorm\.", ".attn_norm."),
    (r"\.post_attention_layernorm\.", ".ffn_norm."),
    (r"\.attn_hc\.(fn|base|scale)$", r".hc_attn_\1"),
    (r"\.ffn_hc\.(fn|base|scale)$", r".hc_ffn_\1"),
    (r"\.self_attn\.sinks$", ".self_attn.attn_sink"),
    (r"\.compressor\.indexer\.scorer\.weights_proj\.", ".indexer.weights_proj."),
    (r"\.compressor\.indexer\.q_b_proj\.", ".indexer.wq_b."),
    (r"\.compressor\.indexer\.", ".indexer.compressor."),
    (r"\.compressor\.kv_norm\.", ".compressor.norm."),
    (r"\.position_bias$", ".ape"),
    (r"\.q_a_proj\.", ".wq_a."),
    (r"\.q_b_proj\.", ".wq_b."),
    (r"\.kv_proj\.", ".wkv."),
    (r"\.compressor\.gate_proj\.", ".compressor.wgate."),
    (r"\.o_a_proj\.", ".wo_a."),
    (r"\.o_b_proj\.", ".wo_b."),
    (r"\.q_a_norm\.", ".q_norm."),
    (r"\.gate\.e_score_correction_bias$", ".gate.bias"),
    (r"\.shared_experts\.gate_proj\.", ".shared_experts.w1."),
    (r"\.shared_experts\.down_proj\.", ".shared_experts.w2."),
    (r"\.shared_experts\.up_proj\.", ".shared_experts.w3."),
    (r"\.self_attn\.", ".attn."),
    (r"\.mlp\.", ".ffn."),
)
GLM_HUB_RENAMES = (
    (r"\.attn_hc\.(fn|base|scale)$", r".hc_attn_\1"),
    (r"\.ffn_hc\.(fn|base|scale)$", r".hc_ffn_\1"),
    (r"\.self_attn\.forget_gate\.", ".self_attn."),
)

# Hub piece of a native tensor: (Hub key, index into the native tensor).
HubPiece = tuple[str, Any]


@dataclass(frozen=True)
class QuantizedCheckpoint:
    """A checkpoint directory and the native state its quantized weights dequantize to."""

    directory: Path
    reference: dict[str, torch.Tensor]
    auto_class: type
    config: Any


def fp8_block_quantize(
    weight: torch.Tensor, power_of_two_scale: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """E4M3 codes on 128x128 blocks, the stored scale grid, and the exact fp32 dequant."""
    rows, cols = weight.shape
    blocks = weight.float().reshape(
        rows // FP8_BLOCK, FP8_BLOCK, cols // FP8_BLOCK, FP8_BLOCK
    )
    scale = blocks.abs().amax(dim=(1, 3)) / FP8_E4M3_MAX
    if power_of_two_scale:
        scale = torch.exp2(torch.ceil(torch.log2(scale)))
    grid = scale[:, None, :, None]
    codes = (blocks / grid).to(torch.float8_e4m3fn)
    dequantized = (codes.float() * grid).reshape(rows, cols)
    stored_scale = scale.to(torch.float8_e8m0fnu) if power_of_two_scale else scale
    return codes.reshape(rows, cols), stored_scale, dequantized


def fp4_row_quantize(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """E2M1 codes on 1x32 blocks packed low nibble first, UE8M0 scales, and the exact fp32 dequant."""
    rows, cols = weight.shape
    blocks = weight.float().reshape(rows, cols // FP4_BLOCK, FP4_BLOCK)
    scale = torch.exp2(torch.ceil(torch.log2(blocks.abs().amax(-1) / 6.0)))
    scaled = blocks / scale[..., None]
    magnitude = (scaled.abs()[..., None] - FP4_E2M1[:8]).abs().argmin(-1)
    codes = (magnitude + 8 * (scaled < 0)).reshape(rows, cols)
    dequantized = FP4_E2M1[codes] * scale.repeat_interleave(FP4_BLOCK, dim=1)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8).view(torch.int8)
    return packed, scale.to(torch.float8_e8m0fnu), dequantized


def renamed(key: str, renames: tuple[tuple[str, str], ...]) -> str:
    for pattern, replacement in renames:
        key = re.sub(pattern, replacement, key)
    return key


def expert_pieces(
    key: str, tensor: torch.Tensor, gate: str, up: str, down: str
) -> list[HubPiece] | None:
    """Per-expert Hub weights that stack into a packed ``gate_up_proj`` / ``down_proj``."""
    if key.endswith("experts.gate_up_proj"):
        base = key.removesuffix("gate_up_proj")
        half = tensor.shape[1] // 2
        return [
            (f"{base}{expert}.{name}.weight", (expert, slice(start, start + half)))
            for expert in range(tensor.shape[0])
            for name, start in ((gate, 0), (up, half))
        ]
    if key.endswith("experts.down_proj"):
        base = key.removesuffix("down_proj")
        return [
            (f"{base}{expert}.{down}.weight", (expert,))
            for expert in range(tensor.shape[0])
        ]
    return None


def dsv4_hub_pieces(key: str, tensor: torch.Tensor) -> list[HubPiece]:
    pieces = expert_pieces(key, tensor, "w1", "w3", "w2") or [(key, (...,))]
    return [(renamed(hub_key, DSV4_HUB_RENAMES), index) for hub_key, index in pieces]


def glm_hub_pieces(key: str, tensor: torch.Tensor) -> list[HubPiece]:
    if key.endswith("self_attn.conv1d.weight"):
        third = tensor.shape[0] // 3
        return [
            (
                key.replace("conv1d", f"{name}_conv1d"),
                (slice(index * third, (index + 1) * third),),
            )
            for index, name in enumerate("qkv")
        ]
    pieces = expert_pieces(key, tensor, "gate_proj", "up_proj", "down_proj") or [
        (key, (...,))
    ]
    return [(renamed(hub_key, GLM_HUB_RENAMES), index) for hub_key, index in pieces]


def write_quantized_checkpoint(
    directory: Path,
    hub_config: dict[str, Any],
    auto_class: type,
    hub_pieces: Callable[[str, torch.Tensor], list[HubPiece]],
    fp8_weight: re.Pattern[str],
    fp4_weight: re.Pattern[str] | None,
    scale_name: str,
    power_of_two_scale: bool,
) -> QuantizedCheckpoint:
    """Save a random native model in the Hub layout, quantizing the matching Hub weights.

    :return: The directory and the native fp32 state the checkpoint dequantizes to.
    """
    config = AutoConfig.for_model(
        **{k: v for k, v in hub_config.items() if k != "quantization_config"}
    )
    config._attn_implementation = "eager"
    torch.manual_seed(0)
    model = auto_class.from_config(config, dtype=torch.float32)
    generator = torch.Generator().manual_seed(0)
    native = {}
    for key, tensor in model.state_dict().items():
        if tensor.is_floating_point():
            tensor = torch.randn(tensor.shape, generator=generator) * 0.05
        native[key] = tensor
    reference = {
        key: tensor.to(torch.bfloat16).float() if tensor.is_floating_point() else tensor
        for key, tensor in native.items()
    }
    stored: dict[str, torch.Tensor] = {}
    for key, tensor in native.items():
        for hub_key, index in hub_pieces(key, tensor):
            piece = tensor[index]
            if fp8_weight.search(hub_key):
                codes, scale, dequantized = fp8_block_quantize(
                    piece, power_of_two_scale
                )
            elif fp4_weight is not None and fp4_weight.search(hub_key):
                codes, scale, dequantized = fp4_row_quantize(piece)
            else:
                stored[hub_key] = (
                    piece.to(torch.bfloat16) if piece.is_floating_point() else piece
                )
                continue
            stored[hub_key] = codes
            stored[f"{hub_key.removesuffix('weight')}{scale_name}"] = scale
            reference[key][index] = dequantized
    directory.mkdir(parents=True, exist_ok=True)
    save_file(
        {key: tensor.contiguous() for key, tensor in stored.items()},
        str(directory / "model.safetensors"),
    )
    (directory / "config.json").write_text(json.dumps(hub_config), encoding="utf-8")
    return QuantizedCheckpoint(directory, reference, auto_class, config)


def tiny_dsv4_config() -> dict[str, Any]:
    """Three-layer DeepSeek-V4-Flash ``config.json``: sliding, compressed, and heavily compressed attention."""
    config = copy.deepcopy(SUPPORTED_MODEL_INFO[DSV4_HUB_ID].config)
    config.update(
        hidden_size=256,
        moe_intermediate_size=128,
        num_hidden_layers=3,
        num_hash_layers=1,
        num_attention_heads=4,
        head_dim=128,
        qk_rope_head_dim=64,
        q_lora_rank=128,
        o_groups=2,
        o_lora_rank=128,
        index_n_heads=2,
        index_head_dim=128,
        index_topk=4,
        vocab_size=256,
        n_routed_experts=4,
        num_experts_per_tok=2,
        sliding_window=8,
        compress_ratios=[0, 4, 128, 0],
        max_position_embeddings=4096,
    )
    # The draft (DSpark) head targets layers the tiny model does not have.
    for key in (
        "dspark_block_size",
        "dspark_noise_token_id",
        "dspark_target_layer_ids",
        "dspark_markov_rank",
    ):
        del config[key]
    return config


def tiny_glm_config() -> dict[str, Any]:
    """Two-layer GLM-5.3-Flash ``config.json``: dense KDA layer 0, MoE MLA layer 1, one vision block."""
    config = copy.deepcopy(SUPPORTED_MODEL_INFO[GLM_HUB_ID].config)
    text = config["text_config"]
    text.update(
        hidden_size=256,
        intermediate_size=256,
        moe_intermediate_size=128,
        num_hidden_layers=2,
        layer_types=["linear_attention", "deepseek_sparse_attention"],
        mlp_layer_types=["dense", "sparse"],
        indexer_types=["full", "full"],
        first_k_dense_replace=1,
        n_routed_experts=4,
        num_experts_per_tok=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        q_lora_rank=128,
        kv_lora_rank=128,
        qk_head_dim=128,
        qk_nope_head_dim=128,
        v_head_dim=128,
        index_n_heads=2,
        index_head_dim=128,
        index_topk=4,
        vocab_size=256,
        max_position_embeddings=4096,
        pad_token_id=0,
        eos_token_id=[1],
        num_nextn_predict_layers=0,
        linear_attn_config={
            **text["linear_attn_config"],
            "num_heads": 2,
            "head_dim": 128,
            "kda_layers": [0],
            "full_attn_layers": [1],
        },
    )
    config["vision_config"].update(
        depth=1,
        hidden_size=128,
        intermediate_size=256,
        num_heads=2,
        out_hidden_size=256,
        projection_intermediate_size=256,
    )
    config.update(
        image_token_id=250,
        video_token_id=251,
        image_start_token_id=252,
        image_end_token_id=253,
        video_start_token_id=254,
        video_end_token_id=255,
    )
    return config


def write_dsv4_checkpoint(directory: Path) -> QuantizedCheckpoint:
    """FP8 dense weights with UE8M0 ``.scale`` grids and FP4 experts, as DeepSeek-V4-Flash ships."""
    return write_quantized_checkpoint(
        directory,
        tiny_dsv4_config(),
        AutoModelForCausalLM,
        dsv4_hub_pieces,
        re.compile(
            r"\.attn\.(wq_a|wq_b|wkv|wo_a|wo_b|indexer\.wq_b)\.weight$"
            r"|\.shared_experts\.w[123]\.weight$"
        ),
        re.compile(r"\.experts\.\d+\.w[123]\.weight$"),
        "scale",
        power_of_two_scale=True,
    )


def write_glm_checkpoint(directory: Path) -> QuantizedCheckpoint:
    """FP8 MLP, expert and MLA weights with fp32 ``weight_scale_inv`` grids, as GLM-5.3-Flash ships."""
    return write_quantized_checkpoint(
        directory,
        tiny_glm_config(),
        AutoModelForImageTextToText,
        glm_hub_pieces,
        re.compile(
            r"language_model\.layers\.\d+\.mlp\."
            r"(experts\.\d+\.|shared_experts\.)?(gate|up|down)_proj\.weight$"
            r"|language_model\.layers\.\d+\.self_attn\."
            r"(q_a_proj|q_b_proj|kv_a_proj_with_mqa)\.weight$"
            r"|language_model\.layers\.1\.self_attn\.o_proj\.weight$"
        ),
        None,
        "weight_scale_inv",
        power_of_two_scale=False,
    )


def reference_model(checkpoint: QuantizedCheckpoint, loaded: nn.Module) -> nn.Module:
    """Native model holding the independently dequantized weights in ``loaded``'s parameter and buffer dtypes."""
    model = checkpoint.auto_class.from_config(checkpoint.config, dtype=torch.float32)
    model.load_state_dict(checkpoint.reference)
    # DeepSeek-V4 adds the router bias buffer to bf16 scores before top-k, so
    # its dtype picks the experts.
    dtypes = {
        name: tensor.dtype
        for name, tensor in (*loaded.named_parameters(), *loaded.named_buffers())
    }
    for name, tensor in (*model.named_parameters(), *model.named_buffers()):
        tensor.data = tensor.data.to(dtypes[name])
    return model.eval()

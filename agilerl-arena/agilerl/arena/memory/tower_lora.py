# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Tower and connector LoRA linears the memory estimator sizes per vision family."""

from __future__ import annotations

from typing import Any, Literal, NamedTuple

from agilerl.arena.memory.config_fields import tower_configs, vision_geometry_fields

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


def vision_fields(config: dict[str, Any]) -> dict[str, Any]:
    """Vision tower geometry and tower LoRA linears; empty without a tower."""
    geometry = vision_geometry_fields(config)
    if not geometry:
        return {}
    return {**geometry, "tower_lora_linears": _tower_lora_linears(config)}


def _tower_lora_linears(config: dict[str, Any]) -> tuple[LoraLinear, ...]:
    """Tower and connector linears, by the target name tower LoRA matches.

    Split factors follow the family's training tensor-parallel plan: Qwen3.5
    and GLM-5.3 split the vision blocks and merger, MiMo-V2 the vision blocks,
    Nemotron Omni the encoder MLP, and Gemma 4 keeps its towers whole. Empty for
    families without tower LoRA.
    """
    model_type = config.get("model_type")
    if model_type in {"qwen3_5", "qwen3_5_moe"}:
        return _qwen_vision_lora_linears(config["vision_config"])
    if model_type == "glm5_next":
        return _glm5_next_vision_lora_linears(config["vision_config"])
    if model_type == "mimo_v2":
        return _mimo_v2_vision_lora_linears(tower_configs(config)[0])
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


def _glm5_next_vision_lora_linears(vision: dict[str, Any]) -> tuple[LoraLinear, ...]:
    """GLM-5.3 vision blocks' qkv and SwiGLU, and the patch merger's SwiGLU."""
    h = int(vision["hidden_size"])
    layers = int(vision["depth"])
    inter = int(vision["intermediate_size"])
    out = int(vision["out_hidden_size"])
    context = int(vision["projection_intermediate_size"])
    return (
        LoraLinear(h, 3 * h, layers, "out", "qkv"),
        LoraLinear(h, inter, layers, "out", "gate_proj"),
        LoraLinear(h, inter, layers, "out", "up_proj"),
        LoraLinear(inter, h, layers, "in", "down_proj"),
        LoraLinear(out, context, 1, "out", "gate_proj"),
        LoraLinear(out, context, 1, "out", "up_proj"),
        LoraLinear(context, out, 1, "in", "down_proj"),
    )


def _mimo_v2_vision_lora_linears(vision: dict[str, Any]) -> tuple[LoraLinear, ...]:
    """MiMo-V2 vision blocks' grouped-query qkv and SwiGLU."""
    h = int(vision["hidden_size"])
    layers = int(vision["depth"])
    inter = int(vision["intermediate_size"])
    head_dim = int(vision["head_dim"])
    qkv = (int(vision["num_heads"]) + 2 * int(vision["num_key_value_heads"])) * head_dim
    return (
        LoraLinear(h, qkv, layers, "out", "qkv"),
        LoraLinear(h, inter, layers, "out", "gate_proj"),
        LoraLinear(h, inter, layers, "out", "up_proj"),
        LoraLinear(inter, h, layers, "in", "down_proj"),
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
    # Hub config.json names it llm_config; transformers' loaded config, text_config.
    text = config.get("text_config") or config["llm_config"]
    linears = [
        *(
            LoraLinear(h, h, layers, target=name)
            for name in ("q_proj", "k_proj", "v_proj", "o_proj")
        ),
        LoraLinear(h, inter, layers, "out", "fc1"),
        LoraLinear(inter, h, layers, "in", "fc2"),
        LoraLinear(patch, h, 1, target="patch_projection"),
        LoraLinear(merged, projector, 1, target="linear_1"),
        LoraLinear(projector, int(text["hidden_size"]), 1, target="linear_2"),
    ]
    frames = vision.get("video_temporal_patch_size")
    if frames:
        linears.append(
            LoraLinear(int(frames) * patch, h, 1, target="video_patch_projection")
        )
    return tuple(linears)

# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tiny Gemma 4 config shared by the Gemma 4 parallel tests."""

from __future__ import annotations

from transformers.models.gemma4.configuration_gemma4 import (
    Gemma4AudioConfig,
    Gemma4Config,
    Gemma4TextConfig,
    Gemma4VisionConfig,
)


def model_config() -> Gemma4Config:
    hidden = 64
    # The last two layers reuse the first two layers' K/V, as the checkpoint's do.
    text = Gemma4TextConfig(
        hidden_size=hidden,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        global_head_dim=32,
        sliding_window=4,
        layer_types=[
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
        ],
        num_kv_shared_layers=2,
        hidden_size_per_layer_input=16,
        vocab_size=32,
        vocab_size_per_layer_input=32,
        final_logit_softcapping=30.0,
    )
    vision = Gemma4VisionConfig(
        hidden_size=hidden,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        patch_size=2,
        position_embedding_size=64,
    )
    audio = Gemma4AudioConfig(
        hidden_size=hidden,
        num_hidden_layers=1,
        num_attention_heads=4,
        subsampling_conv_channels=[8, 4],
        output_proj_dims=hidden,
    )
    config = Gemma4Config(text_config=text, vision_config=vision, audio_config=audio)
    config._attn_implementation = "flex_attention"
    return config

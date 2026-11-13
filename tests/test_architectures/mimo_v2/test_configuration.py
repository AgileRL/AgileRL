# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for reading the ``mimo_v2`` Hub config as ``MiMoV2FlashConfig``."""

from __future__ import annotations

import json

import pytest
import torch
from transformers import AutoConfig, AutoModelForImageTextToText

from agilerl.architectures.mimo_v2.configuration import (
    MiMoV2Config,
    flash_config_fields,
)
from agilerl.architectures.mimo_v2.modeling import MiMoV2ForConditionalGeneration
from agilerl.arena.memory.specs import ModelArch
from tests.test_architectures.mimo_v2.checkpoint_helpers import tiny_hub_config


class TestFlashConfigFields:
    def test_maps_hub_fields_onto_native_fields(self):
        # Arrange
        raw = tiny_hub_config()

        # Act
        fields = flash_config_fields(raw)

        # Assert
        assert fields["rms_norm_eps"] == 1e-6
        assert fields["layer_types"] == ["full_attention", "sliding_attention"]
        assert fields["mlp_layer_types"] == ["dense", "sparse"]
        assert fields["rope_parameters"] == {
            "full_attention": {
                "rope_type": "default",
                "rope_theta": 10000000.0,
                "partial_rotary_factor": 0.334,
            },
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 0.334,
            },
        }
        assert fields["quantization_config"]["dequantize"] is True
        assert fields["quantization_config"]["weight_block_size"] == [128, 128]
        assert "vision_config" not in fields

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"swa_num_attention_heads": 8}, "share num_attention_heads"),
            ({"swa_num_key_value_heads": 2}, "twice as many KV heads"),
            ({"swa_v_head_dim": 48}, "share head dims"),
            ({"add_full_attention_sink_bias": True}, "sinks on SWA layers only"),
            ({"topk_method": "greedy"}, "noaux_tc only"),
            ({"n_shared_experts": 1}, "no shared experts"),
        ],
    )
    def test_rejects_layouts_the_native_class_cannot_build(self, overrides, match):
        raw = {**tiny_hub_config(), **overrides}

        with pytest.raises(ValueError, match=match):
            flash_config_fields(raw)


class TestMiMoV2ConfigFromPretrained:
    def test_auto_config_reads_hub_config_as_native_config(self, tmp_path):
        # Arrange
        (tmp_path / "config.json").write_text(json.dumps(tiny_hub_config()))

        # Act
        config = AutoConfig.from_pretrained(tmp_path)

        # Assert
        assert isinstance(config, MiMoV2Config)
        assert config.model_type == "mimo_v2"
        text_config = config.text_config
        assert text_config.model_type == "mimo_v2_text"
        assert text_config.num_local_experts == 8
        assert text_config.routed_scaling_factor == 1.0
        assert text_config.layer_types == ["full_attention", "sliding_attention"]
        assert config.quantization_config["dequantize"] is True
        assert config.vision_config.model_type == "mimo_v2_vision"
        assert config.vision_config.depth == 3
        assert config.image_token_id == 120

    def test_auto_model_builds_native_vision_language_model_from_config(self, tmp_path):
        # Arrange
        (tmp_path / "config.json").write_text(json.dumps(tiny_hub_config()))
        config = AutoConfig.from_pretrained(tmp_path)

        # Act
        with torch.device("meta"):
            model = AutoModelForImageTextToText.from_config(
                config, attn_implementation="eager"
            )

        # Assert
        assert type(model) is MiMoV2ForConditionalGeneration
        language_model = model.model.language_model
        assert type(language_model.layers[0].mlp.gate_proj) is torch.nn.Linear
        assert language_model.layers[1].self_attn.sinks.shape == (4,)
        assert len(model.model.visual.blocks) == 3

    def test_memory_estimator_reads_the_native_config(self, tmp_path):
        # Arrange
        (tmp_path / "config.json").write_text(json.dumps(tiny_hub_config()))
        config = AutoConfig.from_pretrained(tmp_path)

        # Act
        arch = ModelArch.from_hf_config(config.to_dict())

        # Assert
        assert (arch.vision_min_patches, arch.vision_max_patches) == (4, 64)

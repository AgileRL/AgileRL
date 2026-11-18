# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for reading the ``mimo_v2`` Hub config as ``MiMoV2FlashConfig``."""

from __future__ import annotations

import json

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.mimo_v2_flash.modeling_mimo_v2_flash import (
    MiMoV2FlashForCausalLM,
)

from agilerl.architectures.mimo_v2.config import MiMoV2Config, flash_config_fields
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
        assert config.num_local_experts == 8
        assert config.routed_scaling_factor == 1.0
        assert config.layer_types == ["full_attention", "sliding_attention"]
        assert config.quantization_config["dequantize"] is True

    def test_auto_model_builds_native_causal_lm_from_config(self, tmp_path):
        # Arrange
        (tmp_path / "config.json").write_text(json.dumps(tiny_hub_config()))
        config = AutoConfig.from_pretrained(tmp_path)

        # Act
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(
                config, attn_implementation="eager"
            )

        # Assert
        assert type(model) is MiMoV2FlashForCausalLM
        assert type(model.model.layers[0].mlp.gate_proj) is torch.nn.Linear
        assert model.model.layers[1].self_attn.sinks.shape == (4,)

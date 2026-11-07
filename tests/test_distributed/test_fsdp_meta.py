# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint quantization configs and the dequantize conversions the FSDP loader adds."""

from __future__ import annotations

import pytest
from transformers.utils.quantization_config import FineGrainedFP8Config, Mxfp4Config

from agilerl.distributed.fsdp_meta import (
    checkpoint_quantization_config,
    converter_checkpoint_routes,
    dequantizing_quantization_config,
    dequantizing_transform_groups,
)

FP8_CONFIG = {
    "quant_method": "fp8",
    "fmt": "e4m3",
    "scale_fmt": "ue8m0",
    "weight_block_size": [128, 128],
}


class TestCheckpointQuantizationConfig:
    def test_reads_top_level_config(self):
        assert (
            checkpoint_quantization_config(
                {"model_type": "deepseek_v4", "quantization_config": FP8_CONFIG}
            )
            == FP8_CONFIG
        )

    def test_reads_text_config(self):
        assert (
            checkpoint_quantization_config(
                {"text_config": {"quantization_config": FP8_CONFIG}}
            )
            == FP8_CONFIG
        )

    def test_unquantized_checkpoint_has_none(self):
        assert checkpoint_quantization_config({"model_type": "qwen3"}) is None


class TestDequantizingQuantizationConfig:
    def test_fp8_checkpoint_dequantizes_and_keeps_its_block_size(self):
        config = dequantizing_quantization_config(FP8_CONFIG)

        assert isinstance(config, FineGrainedFP8Config)
        assert config.dequantize is True
        assert config.weight_block_size == [128, 128]

    def test_mxfp4_checkpoint_dequantizes(self):
        config = dequantizing_quantization_config({"quant_method": "mxfp4"})

        assert isinstance(config, Mxfp4Config)
        assert config.dequantize is True

    @pytest.mark.parametrize(
        ("quantization_config", "name"),
        [
            (
                {"quant_method": "bitsandbytes", "load_in_4bit": True},
                "BitsAndBytesConfig",
            ),
            ({"quant_method": "awq", "bits": 4}, "AwqConfig"),
        ],
    )
    def test_other_quantization_raises(self, quantization_config, name):
        with pytest.raises(ValueError, match=f"Cannot train on {name} checkpoints"):
            dequantizing_quantization_config(quantization_config)


class TestDequantizingTransformGroups:
    def test_scale_routes_with_its_weight(self):
        # Arrange
        groups = dequantizing_transform_groups([("model.layers.0", [])], FP8_CONFIG)
        keys = ["model.layers.0.proj.weight", "model.layers.0.proj.scale"]

        # Act
        routes = converter_checkpoint_routes(keys, groups)

        # Assert
        assert [prefix for prefix, _ in groups] == ["model.layers.0", ""]
        assert {key for _, key in routes["model.layers.0.proj.weight"].sources} == set(
            keys
        )

    def test_unscaled_weight_reads_only_itself(self):
        groups = dequantizing_transform_groups([], FP8_CONFIG)

        routes = converter_checkpoint_routes(["norm.weight"], groups)

        assert [key for _, key in routes["norm.weight"].sources] == ["norm.weight"]

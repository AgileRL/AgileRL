# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""FSDP meta-device loading of tiny FP8 / FP4 DeepSeek-V4 and GLM-5.3 checkpoints."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from torch import nn
from transformers import AutoConfig, PretrainedConfig

from agilerl.distributed.fsdp import _load_sharded_weights_from_safetensors
from agilerl.distributed.fsdp_meta import init_rope_buffers
from tests.test_distributed.quantized_checkpoints import (
    INPUT_IDS,
    QuantizedCheckpoint,
    reference_model,
    write_dsv4_checkpoint,
    write_glm_checkpoint,
)


@pytest.fixture(
    scope="module",
    params=[write_dsv4_checkpoint, write_glm_checkpoint],
    ids=["deepseek_v4", "glm5_next"],
)
def quantized_checkpoint(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> QuantizedCheckpoint:
    write: Callable[[Path], QuantizedCheckpoint] = request.param
    return write(tmp_path_factory.mktemp("quantized"))


def meta_model(
    checkpoint: QuantizedCheckpoint,
    dtype: torch.dtype,
    config: PretrainedConfig | None = None,
) -> nn.Module:
    """Build the checkpoint's model on meta and allocate it empty on CPU."""
    if config is None:
        config = AutoConfig.from_pretrained(checkpoint.directory)
    with torch.device("meta"):
        model = checkpoint.auto_class.from_config(
            config, dtype=dtype, attn_implementation="eager"
        )
    model.to_empty(device="cpu")
    return model


class TestLoadShardedWeightsFromSafetensors:
    def test_dequantizes_every_weight_exactly(self, quantized_checkpoint):
        # Arrange
        model = meta_model(quantized_checkpoint, torch.float32)

        # Act
        _load_sharded_weights_from_safetensors(model)

        # Assert
        state = model.state_dict()
        assert state.keys() == quantized_checkpoint.reference.keys()
        for name, expected in quantized_checkpoint.reference.items():
            assert state[name].dtype == expected.dtype, name
            assert torch.equal(state[name], expected), name

    def test_bf16_logits_match_reference_model(self, quantized_checkpoint):
        # Arrange
        model = meta_model(quantized_checkpoint, torch.bfloat16)
        _load_sharded_weights_from_safetensors(model)
        init_rope_buffers(model)
        expected_model = reference_model(quantized_checkpoint, model)

        # Act
        with torch.no_grad():
            logits = model.eval()(input_ids=INPUT_IDS).logits
            expected = expected_model(input_ids=INPUT_IDS).logits

        # Assert
        # Same bf16 weights, buffers and CPU kernels on both sides.
        assert torch.equal(logits, expected)

    def test_fp8_bytes_without_quantization_config_raise(self, quantized_checkpoint):
        # Arrange
        config = AutoConfig.from_pretrained(quantized_checkpoint.directory)
        del config.quantization_config
        model = meta_model(quantized_checkpoint, torch.bfloat16, config)

        # Act / Assert
        with pytest.raises(
            RuntimeError, match="float8_e4m3fn with no dequantize route"
        ):
            _load_sharded_weights_from_safetensors(model)

    def test_unsupported_quantization_raises(self, quantized_checkpoint):
        # Arrange
        config = AutoConfig.from_pretrained(quantized_checkpoint.directory)
        config.quantization_config = {
            "quant_method": "bitsandbytes",
            "load_in_4bit": True,
        }
        model = meta_model(quantized_checkpoint, torch.bfloat16, config)

        # Act / Assert
        with pytest.raises(ValueError, match="Cannot train on BitsAndBytesConfig"):
            _load_sharded_weights_from_safetensors(model)

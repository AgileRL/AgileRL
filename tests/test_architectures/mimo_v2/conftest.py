# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import torch
from transformers.models.mimo_v2_flash.modeling_mimo_v2_flash import (
    MiMoV2FlashForCausalLM,
)

from agilerl.architectures import register_mimo_v2_native_loading
from agilerl.architectures.mimo_v2.config import MiMoV2Config
from tests.test_architectures.mimo_v2.checkpoint_helpers import (
    hub_checkpoint,
    random_native_state,
    tiny_hub_config,
    write_hub_checkpoint,
)

register_mimo_v2_native_loading()


@pytest.fixture(scope="module")
def native_config() -> MiMoV2Config:
    """Native config of the tiny checkpoint, on the eager attention backend."""
    config = MiMoV2Config.from_dict(tiny_hub_config())
    config._attn_implementation = "eager"
    return config


@pytest.fixture(scope="module")
def tiny_checkpoint(
    tmp_path_factory: pytest.TempPathFactory, native_config: MiMoV2Config
) -> tuple[Path, dict[str, torch.Tensor]]:
    """Quantized Hub checkpoint directory and the native state dict it dequantizes to."""
    native = random_native_state(MiMoV2FlashForCausalLM(native_config), seed=0)
    hub, reference = hub_checkpoint(native, native_config)
    directory = tmp_path_factory.mktemp("mimo_v2_hub")
    write_hub_checkpoint(directory, hub)
    return directory, reference

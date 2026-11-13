# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Model specs from HF configs, and the hashable training-settings key the planner caches on."""

import pytest

from agilerl.arena.memory.settings_key import TrainingSettingsKey
from agilerl.arena.memory.specs import (
    LoraLinear,
    ModelArch,
    TrainingSettings,
    multimodal_tower_block_params,
)
from agilerl.arena.models.fsdp import record_parallel_pins
from agilerl.arena.models.model_info import SUPPORTED_MODEL_INFO
from tests.test_memory.test_estimator import pinned_fsdp

SUPER_VL_ID = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16"


class TestModelArchFromHfConfig:
    def test_super_vl_loaded_config_sizes_the_projector_like_the_hub_config(self):
        # Arrange: transformers loads the Hub's llm_config as text_config.
        hub_config = SUPPORTED_MODEL_INFO[SUPER_VL_ID].config
        loaded_config = {
            key: value for key, value in hub_config.items() if key != "llm_config"
        }
        loaded_config["text_config"] = hub_config["llm_config"]

        # Act
        from_hub = ModelArch.from_hf_config(hub_config)
        from_loaded = ModelArch.from_hf_config(loaded_config)

        # Assert
        projector = next(
            linear
            for linear in from_loaded.tower_lora_linears
            if linear.target == "linear_2"
        )
        assert projector.out_features == hub_config["llm_config"]["hidden_size"]
        assert from_loaded.tower_lora_linears == from_hub.tower_lora_linears

    def test_mimo_v2_tower_lora_follows_the_tensor_parallel_splits(self):
        # Arrange
        config = SUPPORTED_MODEL_INFO["XiaomiMiMo/MiMo-V2.6-Flash-RL"].config

        # Act
        linears = ModelArch.from_hf_config(config).tower_lora_linears

        # Assert
        assert linears == (
            LoraLinear(1280, 3072, 28, "out", "qkv"),
            LoraLinear(1280, 4608, 28, "out", "gate_proj"),
            LoraLinear(1280, 4608, 28, "out", "up_proj"),
            LoraLinear(4608, 1280, 28, "in", "down_proj"),
        )

    def test_glm5_next_tower_lora_follows_the_tensor_parallel_splits(self):
        # Arrange
        config = SUPPORTED_MODEL_INFO["zai-org/GLM-5.3-Flash"].config

        # Act
        linears = ModelArch.from_hf_config(config).tower_lora_linears

        # Assert
        assert linears == (
            LoraLinear(1024, 3072, 24, "out", "qkv"),
            LoraLinear(1024, 4096, 24, "out", "gate_proj"),
            LoraLinear(1024, 4096, 24, "out", "up_proj"),
            LoraLinear(4096, 1024, 24, "in", "down_proj"),
            LoraLinear(4096, 10240, 1, "out", "gate_proj"),
            LoraLinear(4096, 10240, 1, "out", "up_proj"),
            LoraLinear(10240, 4096, 1, "in", "down_proj"),
        )

    def test_glm5_next_shards_tower_blocks_as_fsdp_units(self):
        config = SUPPORTED_MODEL_INFO["zai-org/GLM-5.3-Flash"].config

        assert multimodal_tower_block_params(config) > 0


class TestTrainingSettingsKeyOf:
    def test_equal_settings_share_a_key(self):
        # Arrange
        first = TrainingSettings(n_training_gpus=4, fsdp=pinned_fsdp())
        second = TrainingSettings(n_training_gpus=4, fsdp=pinned_fsdp())

        # Act
        keys = TrainingSettingsKey.of(first), TrainingSettingsKey.of(second)

        # Assert
        assert keys[0] == keys[1]
        assert hash(keys[0]) == hash(keys[1])

    @pytest.mark.parametrize(
        "other",
        [
            TrainingSettings(n_training_gpus=8, fsdp=pinned_fsdp()),
            TrainingSettings(n_training_gpus=4, fsdp=pinned_fsdp(tp=2)),
            TrainingSettings(n_training_gpus=4, fsdp=None),
            TrainingSettings(
                n_training_gpus=4, fsdp=record_parallel_pins(pinned_fsdp(), ["tp"])
            ),
        ],
        ids=["field", "fsdp-field", "no-fsdp", "parallel-pins"],
    )
    def test_any_differing_value_changes_the_key(self, other):
        # Arrange
        settings = TrainingSettings(n_training_gpus=4, fsdp=pinned_fsdp())

        # Act / Assert
        assert TrainingSettingsKey.of(settings) != TrainingSettingsKey.of(other)

    def test_key_keeps_the_values_it_was_made_from(self):
        # Arrange
        settings = TrainingSettings(n_training_gpus=4, fsdp=pinned_fsdp())
        before = TrainingSettingsKey.of(settings)

        # Act
        settings.fsdp.reshard_after_forward = False

        # Assert
        assert before != TrainingSettingsKey.of(settings)
        assert before == TrainingSettingsKey.of(
            TrainingSettings(n_training_gpus=4, fsdp=pinned_fsdp())
        )

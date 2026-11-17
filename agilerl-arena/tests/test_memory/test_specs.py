# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""The hashable training-settings key the planner caches on, and model architecture parsing."""

import pytest

from agilerl.arena.memory.settings_key import TrainingSettingsKey
from agilerl.arena.memory.specs import ModelArch, TrainingSettings
from agilerl.arena.models.fsdp import record_parallel_pins
from agilerl.arena.models.model_info import SUPPORTED_MODEL_INFO
from tests.test_memory.test_estimator import pinned_fsdp


class TestModelArchFromHfConfig:
    @pytest.mark.parametrize("hub_id", sorted(SUPPORTED_MODEL_INFO))
    def test_every_catalogue_checkpoint_config_parses(self, hub_id: str):
        # Act
        arch = ModelArch.from_hf_config(SUPPORTED_MODEL_INFO[hub_id].config)

        # Assert
        assert arch.n_layers > 0

    def test_super_vl_tower_linears_read_a_loaded_models_text_config(self):
        # Arrange: a loaded model's config names the language config text_config.
        model = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16"
        checkpoint = SUPPORTED_MODEL_INFO[model].config
        loaded = dict(checkpoint)
        loaded["text_config"] = loaded.pop("llm_config")

        # Act
        from_checkpoint = ModelArch.from_hf_config(checkpoint)
        from_loaded = ModelArch.from_hf_config(loaded)

        # Assert
        assert from_loaded.tower_lora_linears == from_checkpoint.tower_lora_linears
        assert from_loaded.tower_lora_linears

    def test_super_vl_projector_prefers_text_config_when_both_names_exist(self):
        # Arrange: both names, with different hidden sizes.
        model = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16"
        checkpoint = SUPPORTED_MODEL_INFO[model].config
        language = dict(checkpoint["llm_config"])
        both = {
            **checkpoint,
            "text_config": language,
            "llm_config": {
                **language,
                "hidden_size": int(language["hidden_size"]) + 64,
            },
        }

        # Act
        from_checkpoint = ModelArch.from_hf_config(checkpoint)
        from_both = ModelArch.from_hf_config(both)

        # Assert
        checkpoint_width = next(
            linear.out_features
            for linear in from_checkpoint.tower_lora_linears
            if linear.target == "linear_2"
        )
        both_width = next(
            linear.out_features
            for linear in from_both.tower_lora_linears
            if linear.target == "linear_2"
        )
        assert both_width == checkpoint_width
        assert both_width != both["llm_config"]["hidden_size"]


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

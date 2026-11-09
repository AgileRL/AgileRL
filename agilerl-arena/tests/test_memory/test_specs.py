# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""The hashable training-settings key the planner caches on."""

import pytest

from agilerl.arena.memory.specs import TrainingSettings, TrainingSettingsKey
from agilerl.arena.models.fsdp import record_parallel_pins
from tests.test_memory.test_estimator import pinned_fsdp


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

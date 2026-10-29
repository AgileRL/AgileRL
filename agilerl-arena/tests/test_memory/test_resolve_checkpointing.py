# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Gradient checkpointing is off only when the uncheckpointed step fits."""

from agilerl.arena.memory import formulas
from agilerl.arena.memory.estimator import estimate_training, training_breakdown
from agilerl.arena.memory.resolve.checkpointing import resolve_gradient_checkpointing
from agilerl.arena.memory.specs import DeviceSpec, GiB, ModelSpec, TrainingSettings
from tests.test_memory.test_formulas import QWEN_05B

MODEL = ModelSpec(model_id="Qwen/Qwen2.5-0.5B-Instruct", arch=QWEN_05B)
SETTINGS = TrainingSettings(micro_batch_size=4, max_model_len=4096)


def budget_device(usable_bytes: int) -> DeviceSpec:
    return DeviceSpec(total_bytes=80 * GiB, available_bytes=usable_bytes, name="budget")


def peak(gradient_checkpointing: bool) -> int:
    settings = SETTINGS.model_copy(
        update={"gradient_checkpointing": gradient_checkpointing}
    )
    return estimate_training(MODEL, budget_device(80 * GiB), settings).total_bytes


def buffered_budget(total_bytes: int) -> int:
    return int(total_bytes / (1 - formulas.MAX_UNDERPREDICTION)) + 1


class TestResolveGradientCheckpointing:
    def test_roomy_budget_turns_checkpointing_off(self):
        resolved = resolve_gradient_checkpointing(
            MODEL, budget_device(80 * GiB), SETTINGS, training_breakdown
        )

        assert resolved is False

    def test_uncheckpointed_peak_over_the_buffered_budget_keeps_checkpointing(self):
        # Arrange: the checkpointed step fits with buffer, the uncheckpointed one does not.
        usable = buffered_budget(peak(False)) - 1

        # Act
        resolved = resolve_gradient_checkpointing(
            MODEL, budget_device(usable), SETTINGS, training_breakdown
        )

        # Assert
        assert buffered_budget(peak(True)) < usable
        assert resolved is True

    def test_uncheckpointed_peak_inside_the_buffered_budget_turns_it_off(self):
        resolved = resolve_gradient_checkpointing(
            MODEL,
            budget_device(buffered_budget(peak(False))),
            SETTINGS,
            training_breakdown,
        )

        assert resolved is False

    def test_activation_offload_keeps_checkpointing(self):
        settings = SETTINGS.model_copy(update={"activation_offload": True})

        resolved = resolve_gradient_checkpointing(
            MODEL, budget_device(80 * GiB), settings, training_breakdown
        )

        assert resolved is True


class TestEstimateTrainingGradientCheckpointing:
    def test_uncheckpointed_step_keeps_more_activations(self):
        assert peak(False) > peak(True)

    def test_uncheckpointed_activations_grow_with_depth(self):
        # Arrange
        shallow = MODEL.model_copy(
            update={"arch": QWEN_05B.model_copy(update={"n_layers": 12})}
        )
        settings = SETTINGS.model_copy(update={"gradient_checkpointing": False})
        device = budget_device(80 * GiB)

        # Act
        deep_gap = (
            estimate_training(MODEL, device, settings).total_bytes
            - estimate_training(MODEL, device, SETTINGS).total_bytes
        )
        shallow_gap = (
            estimate_training(shallow, device, settings).total_bytes
            - estimate_training(shallow, device, SETTINGS).total_bytes
        )

        # Assert
        assert QWEN_05B.n_layers > 12
        assert deep_gap > shallow_gap > 0

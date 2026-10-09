# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Cheapest resource tier for a manifest."""

import json
from pathlib import Path

import pytest

from agilerl.arena.memory.estimator import MemoryComponent, PhaseBreakdown, RunEstimate
from agilerl.arena.memory.formulas import MAX_UNDERPREDICTION
from agilerl.arena.memory.manifest import run_config_from_manifest
from agilerl.arena.memory.resources import check_tiers, gpus_required
from agilerl.arena.models.manifest import TrainingManifest
from tests.test_memory.test_cli import MANIFEST, OVERSIZE, TINY_CONFIG


def tier(name, gpu_type, num_gpus, price):
    return {
        "name": name,
        "gpu_type": gpu_type,
        "num_gpus": num_gpus,
        "num_cpus": 8,
        "ram_gb": 64,
        "price_per_node_hour": price,
    }


def manifest_with(base, **training):
    return TrainingManifest.model_validate(
        {**base, "training": {**base["training"], **training}}
    )


def builder(manifest):
    model_config = json.loads(Path(TINY_CONFIG).read_text())
    return lambda device: run_config_from_manifest(manifest, device, model_config)


class TestGpusRequired:
    def test_counts_trainers_and_engines_per_member(self):
        manifest = manifest_with(
            MANIFEST,
            pop_size=2,
            training_gpus_per_agent=2,
            rollout_engines_per_agent=3,
            rollout_batch_size=4,
        )

        assert gpus_required(manifest.training) == 2 * (2 + 3)

    def test_auto_engines_need_one_engine_per_member(self):
        manifest = manifest_with(MANIFEST, rollout_engines_per_agent="auto")

        assert gpus_required(manifest.training) == 2


class TestCheckTiers:
    def test_orders_by_price_and_rules_out_unusable_tiers(self):
        # Arrange
        manifest = TrainingManifest.model_validate(MANIFEST)
        tiers = [
            tier("a100-2x", "NVIDIA A100-SXM4-80GB", 2, 9.0),
            tier("l4-2x", "NVIDIA L4", 2, 2.0),
            tier("l4-1x", "NVIDIA L4", 1, 1.0),
            tier("cpu", None, 0, 0.5),
            tier("tpu", "TPU v5e", 4, 1.5),
        ]

        # Act
        checks = check_tiers(tiers, manifest.training, builder(manifest))

        # Assert
        assert [c.tier["name"] for c in checks] == [
            "cpu",
            "l4-1x",
            "tpu",
            "l4-2x",
            "a100-2x",
        ]
        assert [c.reason for c in checks[:3]] == [
            "no GPU",
            "1 GPUs, the job needs 2",
            "unknown GPU 'TPU v5e'",
        ]
        assert [c.fits for c in checks[3:]] == [True, True]
        assert checks[3].config.train_device.name == "NVIDIA L4"
        assert checks[3].estimate.fits

    def test_a_tier_whose_gpu_is_too_small_is_over_budget(self):
        manifest = TrainingManifest.model_validate(OVERSIZE)
        tiers = [tier("t4-2x", "Tesla T4", 2, 1.0)]

        (check,) = check_tiers(tiers, manifest.training, builder(manifest))

        assert not check.fits
        assert check.reason is not None
        assert check.reason.endswith("over budget")
        assert check.estimate is not None
        assert not check.estimate.fits

    def test_a_point_estimate_inside_the_buffer_is_not_recommended(self, monkeypatch):
        # Arrange: the estimate is under the card and over the buffered line.
        usable = 10_000
        limit = int(usable * (1 - MAX_UNDERPREDICTION))
        predicted = limit + 1

        def fake_estimate(_config):
            phase = PhaseBreakdown(
                phase="training",
                components=(
                    MemoryComponent(key="weights", label="Weights", n_bytes=predicted),
                ),
                device_total_bytes=usable,
                device_usable_bytes=usable,
            )
            generation = PhaseBreakdown(
                phase="generation",
                components=(),
                device_total_bytes=usable,
                device_usable_bytes=usable,
            )
            return RunEstimate(training=phase, generation=generation)

        monkeypatch.setattr(
            "agilerl.arena.memory.resources.estimate_run", fake_estimate
        )
        manifest = TrainingManifest.model_validate(MANIFEST)

        # Act
        (check,) = check_tiers(
            [tier("l4-2x", "NVIDIA L4", 2, 2.0)], manifest.training, builder(manifest)
        )

        # Assert
        assert check.estimate.fits
        assert not check.fits
        assert check.reason == "training over budget"

    @pytest.mark.parametrize("tiers", [[], [tier("cpu", None, 0, 0.5)]])
    def test_no_fitting_tier_leaves_no_fit(self, tiers):
        manifest = TrainingManifest.model_validate(MANIFEST)

        checks = check_tiers(tiers, manifest.training, builder(manifest))

        assert not any(c.fits for c in checks)

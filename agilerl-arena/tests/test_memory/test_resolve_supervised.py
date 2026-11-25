# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""SFT and DPO step seconds."""

import copy

import pytest

from agilerl.arena.memory.gpus import lookup_gpu
from agilerl.arena.memory.resolve.split import learn_gpu_seconds
from agilerl.arena.memory.resolve.supervised import supervised_step_seconds
from agilerl.arena.memory.resources import check_tiers
from agilerl.arena.models.manifest import TrainingManifest
from tests.test_memory.test_cli import MANIFEST, SFT_MANIFEST
from tests.test_memory.test_resources import builder, tier, tiny_model

L4 = lookup_gpu("NVIDIA L4").device_spec()
H100 = lookup_gpu("NVIDIA H100 80GB HBM3").device_spec()
MODEL = tiny_model()


def supervised_manifest(name="SFT", trainers=1, **algorithm):
    doc = copy.deepcopy(SFT_MANIFEST)
    doc["algorithm"] = {"name": name, **algorithm}
    doc["training"]["training_gpus_per_agent"] = trainers
    return TrainingManifest.model_validate(doc)


def test_sft_step_is_a_forward_and_backward_over_full_rows():
    manifest = supervised_manifest(update_epochs=2)
    spec = manifest.algorithm

    seconds = supervised_step_seconds(manifest, MODEL, L4)

    assert seconds == learn_gpu_seconds(
        MODEL, L4, L4.peak_bf16_flops, spec.batch_size, spec.max_model_len, 4
    )


def test_dpo_trains_chosen_and_rejected_and_scores_the_reference():
    manifest = supervised_manifest("DPO", beta=0.1)
    spec = manifest.algorithm

    seconds = supervised_step_seconds(manifest, MODEL, L4)

    assert seconds == learn_gpu_seconds(
        MODEL, L4, L4.peak_bf16_flops, 2 * spec.batch_size, spec.max_model_len, 3
    )


def test_dpo_scores_the_reference_when_beta_is_zero():
    with_beta = supervised_manifest("DPO", beta=0.1)
    without = supervised_manifest("DPO", beta=0.0)

    assert supervised_step_seconds(without, MODEL, L4) == supervised_step_seconds(
        with_beta, MODEL, L4
    )


def test_more_trainers_share_the_step():
    one = supervised_step_seconds(supervised_manifest(trainers=1), MODEL, H100)
    four = supervised_step_seconds(supervised_manifest(trainers=4), MODEL, H100)

    assert four == pytest.approx(one / 4)


def test_a_faster_gpu_takes_a_shorter_step():
    manifest = supervised_manifest()

    assert supervised_step_seconds(manifest, MODEL, H100) < (
        supervised_step_seconds(manifest, MODEL, L4)
    )


def test_rollout_runs_are_not_timed_here():
    manifest = TrainingManifest.model_validate(MANIFEST)

    assert supervised_step_seconds(manifest, MODEL, L4) is None


def test_a_gpu_without_peak_flops_is_not_timed():
    device = L4.model_copy(update={"peak_bf16_flops": None})

    assert supervised_step_seconds(supervised_manifest(), MODEL, device) is None


def test_tier_checks_time_sft_runs():
    manifest = TrainingManifest.model_validate(SFT_MANIFEST)

    [check] = check_tiers([tier("l4", "NVIDIA L4", 1, 1.0)], manifest, MODEL, builder())

    assert check.fits
    assert check.step_seconds == pytest.approx(
        supervised_step_seconds(supervised_manifest(), MODEL, L4)
    )

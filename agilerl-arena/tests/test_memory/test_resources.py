# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Cheapest resource tier for a manifest."""

import copy
import json
from pathlib import Path

import pytest

from agilerl.arena.memory.estimator import MemoryComponent, PhaseBreakdown, RunEstimate
from agilerl.arena.memory.formulas import MAX_UNDERPREDICTION
from agilerl.arena.memory.manifest import lookup_gpu, run_config_from_manifest
from agilerl.arena.memory.resolve.batch import catalog_model
from agilerl.arena.memory.resolve.engine import resolve_engine_tensor_parallel
from agilerl.arena.memory.resolve.split import resolve_run_memory
from agilerl.arena.memory.resources import (
    check_tiers,
    engine_width,
    engines_per_member,
    gpus_required,
    searches_gpu_split,
    with_run_gpu_split,
)
from agilerl.arena.models.manifest import TrainingManifest
from tests.test_memory.test_cli import (
    MANIFEST,
    NO_TIER_FITS,
    OVERSIZE,
    SFT_MANIFEST,
    TINY_CONFIG,
)


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


def unset_width(base):
    doc = copy.deepcopy(base)
    del doc["algorithm"]["vllm_config"]["tensor_parallel_size"]
    return doc


def unset_split(base):
    doc = copy.deepcopy(base)
    doc["training"] = {
        key: value
        for key, value in doc["training"].items()
        if key != "rollout_engines_per_agent"
    }
    return doc


def with_model(base, model):
    doc = copy.deepcopy(base)
    doc["network"] = {**doc["network"], "pretrained_model_name_or_path": model}
    return doc


def moe_unset_layout():
    """Qwen3.6-35B-A3B at 32k tokens with the split and engine width omitted."""
    doc = unset_split(unset_width(with_model(MANIFEST, "Qwen/Qwen3.6-35B-A3B")))
    doc["network"]["max_context_length"] = 32768
    return doc


def builder():
    model_config = json.loads(Path(TINY_CONFIG).read_text())
    return lambda laid_out, device, width, engines: run_config_from_manifest(
        laid_out,
        device,
        model_config,
        tensor_parallel_size=width,
        rollout_engines=engines,
    )


class TestGpusRequired:
    def test_counts_trainers_and_engines_per_member(self):
        manifest = manifest_with(
            MANIFEST,
            pop_size=2,
            training_gpus_per_agent=2,
            rollout_engines_per_agent=3,
            rollout_batch_size=4,
        )

        assert gpus_required(manifest) == 2 * (2 + 3)

    def test_auto_engines_need_one_engine_per_member(self):
        manifest = manifest_with(MANIFEST, rollout_engines_per_agent="auto")

        assert gpus_required(manifest) == 2

    def test_tensor_parallel_engine_counts_a_gpu_per_rank(self):
        manifest = manifest_with(
            MANIFEST,
            pop_size=1,
            training_gpus_per_agent=2,
            rollout_engines_per_agent=2,
            rollout_batch_size=4,
        )

        assert gpus_required(manifest, tensor_parallel_size=4) == 2 + 2 * 4


class TestEngineWidth:
    def test_pinned_width_is_kept(self):
        doc = copy.deepcopy(MANIFEST)
        doc["algorithm"]["vllm_config"]["tensor_parallel_size"] = 2
        manifest = TrainingManifest.model_validate(doc)

        width = engine_width(manifest, 8, lookup_gpu("NVIDIA L4").device_spec())

        assert width == 2

    def test_unset_width_is_what_the_run_resolver_assigns(self):
        # Arrange
        manifest = TrainingManifest.model_validate(unset_width(MANIFEST))
        device = lookup_gpu("NVIDIA L4").device_spec()
        submitted = manifest.model_copy(deep=True)

        # Act
        width = engine_width(manifest, 4, device)

        # Assert
        assert width == resolve_engine_tensor_parallel(submitted, 4, device, 4)
        assert manifest.algorithm.vllm_config.tensor_parallel_size is None

    def test_unset_width_stays_on_one_node(self):
        # Arrange: 8 GPUs over 8 single-GPU nodes.
        manifest = TrainingManifest.model_validate(unset_width(MANIFEST))
        device = lookup_gpu("NVIDIA L4").device_spec()

        # Act
        width = engine_width(manifest, 8, device, gpus_per_node=1)

        # Assert
        assert width == 1

    def test_width_the_resolver_leaves_unset_starts_at_one(self):
        manifest = TrainingManifest.model_validate(unset_width(MANIFEST))

        width = engine_width(manifest, 1, lookup_gpu("NVIDIA L4").device_spec())

        assert width == 1

    def test_engineless_algorithm_has_width_one(self):
        manifest = TrainingManifest.model_validate(SFT_MANIFEST)

        width = engine_width(manifest, 4, lookup_gpu("NVIDIA L4").device_spec())

        assert width == 1


class TestEnginesPerMember:
    def test_auto_fills_the_surplus_at_the_width(self):
        manifest = manifest_with(MANIFEST, pop_size=2, rollout_engines_per_agent="auto")

        engines = engines_per_member(manifest, 8, tensor_parallel_size=2)

        assert engines == (8 - 2) // (2 * 2)

    def test_pinned_engines_are_kept(self):
        manifest = manifest_with(MANIFEST, rollout_engines_per_agent=2)

        assert engines_per_member(manifest, 8, tensor_parallel_size=1) == 2

    def test_too_few_gpus_raise(self):
        manifest = manifest_with(MANIFEST, rollout_engines_per_agent=2)

        with pytest.raises(ValueError, match="2 GPUs, the job needs 3"):
            engines_per_member(manifest, 2, tensor_parallel_size=1)


class TestSearchesGpuSplit:
    def test_async_run_with_neither_count_searches(self):
        manifest = TrainingManifest.model_validate(unset_split(MANIFEST))

        assert searches_gpu_split(manifest)

    @pytest.mark.parametrize(
        "pinned",
        [{"rollout_engines_per_agent": 2}, {"training_gpus_per_agent": 1}],
    )
    def test_either_count_pins_the_split(self, pinned):
        manifest = manifest_with(unset_split(MANIFEST), **pinned)

        assert not searches_gpu_split(manifest)

    def test_engineless_algorithm_does_not_search(self):
        manifest = TrainingManifest.model_validate(SFT_MANIFEST)

        assert not searches_gpu_split(manifest)


class TestWithRunGpuSplit:
    def test_writes_the_split_the_run_memory_search_picks(self):
        # Arrange
        manifest = TrainingManifest.model_validate(unset_split(MANIFEST))
        device = lookup_gpu("NVIDIA L4").device_spec()
        expected = resolve_run_memory(
            manifest, catalog_model(manifest.algorithm), device, 8
        ).split

        # Act
        laid_out = with_run_gpu_split(manifest, 8, device)

        # Assert
        assert expected is not None
        assert laid_out.training.training_gpus_per_agent == expected.trainers
        assert laid_out.training.rollout_engines_per_agent == expected.engines
        assert laid_out.training.rollout_batch_size == expected.rollout_batch_size
        assert manifest.training.rollout_batch_size is None
        assert "rollout_engines_per_agent" not in manifest.training.model_fields_set

    def test_omitted_width_is_picked_with_the_split(self):
        # Arrange: one H100 holds Qwen3.6-35B-A3B's weights with no KV room.
        manifest = TrainingManifest.model_validate(moe_unset_layout())
        device = lookup_gpu("H100-80").device_spec()

        # Act
        laid_out = with_run_gpu_split(manifest, 8, device)

        # Assert
        assert laid_out.algorithm.vllm_config.tensor_parallel_size == 2
        assert laid_out.training.training_gpus_per_agent == 2
        assert laid_out.training.rollout_engines_per_agent == 3
        assert laid_out.training.rollout_batch_size == 6
        assert manifest.algorithm.vllm_config.tensor_parallel_size is None

    def test_no_fitting_split_raises(self):
        # Arrange: Qwen3.8-27B LoRA training is over one L4.
        manifest = TrainingManifest.model_validate(
            unset_split(with_model(OVERSIZE, "Qwen/Qwen3.8-27B"))
        )
        device = lookup_gpu("NVIDIA L4").device_spec()

        # Act / Assert
        with pytest.raises(ValueError, match="No trainer / rollout-engine split of 2"):
            with_run_gpu_split(manifest, 2, device)

    def test_model_outside_the_catalogue_raises(self):
        # Arrange
        manifest = TrainingManifest.model_validate(
            unset_split(with_model(MANIFEST, "local/uncatalogued-model"))
        )
        device = lookup_gpu("NVIDIA L4").device_spec()

        # Act / Assert
        with pytest.raises(
            ValueError,
            match="'local/uncatalogued-model' is not in the memory estimator catalogue",
        ):
            with_run_gpu_split(manifest, 8, device)


class TestCheckTiers:
    def test_omitted_split_is_sized_at_the_split_the_run_picks(self):
        # Arrange
        manifest = TrainingManifest.model_validate(unset_split(MANIFEST))
        device = lookup_gpu("NVIDIA L4").device_spec()
        expected = with_run_gpu_split(manifest, 8, device)
        sized = []
        build = builder()

        def recording_build(laid_out, train_device, width, engines):
            sized.append((laid_out, engines))
            return build(laid_out, train_device, width, engines)

        # Act
        (check,) = check_tiers(
            [tier("l4-8x", "NVIDIA L4", 8, 8.0)], manifest, recording_build
        )

        # Assert
        ((laid_out, engines),) = sized
        assert check.fits
        assert (
            laid_out.training.training_gpus_per_agent,
            laid_out.training.rollout_batch_size,
            engines,
        ) == (
            expected.training.training_gpus_per_agent,
            expected.training.rollout_batch_size,
            expected.training.rollout_engines_per_agent,
        )

    def test_omitted_width_is_sized_at_the_width_the_split_picks(self):
        # Arrange
        manifest = TrainingManifest.model_validate(moe_unset_layout())
        widths = []
        build = builder()

        def recording_build(laid_out, train_device, width, engines):
            widths.append((width, engines))
            return build(laid_out, train_device, width, engines)

        # Act
        check_tiers(
            [tier("h100-8x", "NVIDIA H100 80GB HBM3", 8, 30.0)],
            manifest,
            recording_build,
        )

        # Assert
        assert widths == [(2, 3)]

    def test_no_fitting_split_keeps_the_search_reason(self):
        manifest = TrainingManifest.model_validate(
            unset_split(with_model(OVERSIZE, "Qwen/Qwen3.8-27B"))
        )

        (check,) = check_tiers(
            [tier("l4-2x", "NVIDIA L4", 2, 1.0)], manifest, builder()
        )

        assert not check.fits
        assert check.reason.startswith("No trainer / rollout-engine split of 2")
        assert check.estimate is not None

    @pytest.mark.parametrize(
        ("gpu_type", "label"), [("Tesla T4", "SM75"), ("Tesla V100", "SM70")]
    )
    def test_a_gpu_below_sm80_is_ruled_out_as_unsupported(self, gpu_type, label):
        manifest = TrainingManifest.model_validate(unset_split(MANIFEST))

        (check,) = check_tiers([tier("old-2x", gpu_type, 2, 1.0)], manifest, builder())

        assert not check.fits
        assert f"({label}) cannot run LLM training" in check.reason
        assert check.config is None
        assert check.estimate is None

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
        checks = check_tiers(tiers, manifest, builder())

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

    def test_tensor_parallel_engine_needs_a_gpu_per_rank(self):
        doc = copy.deepcopy(MANIFEST)
        doc["algorithm"]["vllm_config"]["tensor_parallel_size"] = 4
        manifest = TrainingManifest.model_validate(doc)
        tiers = [tier("l4-4x", "NVIDIA L4", 4, 4.0)]

        (check,) = check_tiers(tiers, manifest, builder())

        assert not check.fits
        assert check.reason == "4 GPUs, the job needs 5"

    def test_auto_engines_size_the_sequences_the_tier_splits(self):
        # Arrange: 8 prompts of group 4 over the four engines 5 GPUs leave.
        manifest = manifest_with(
            MANIFEST, rollout_engines_per_agent="auto", rollout_batch_size=8
        )
        tiers = [tier("l4-5x", "NVIDIA L4", 5, 5.0)]

        # Act
        (check,) = check_tiers(tiers, manifest, builder())

        # Assert
        assert check.config.generation.tensor_parallel_size == 1
        assert check.config.generation.max_num_seqs == 2 * 4

    def test_a_tier_whose_gpu_is_too_small_is_over_budget(self):
        manifest = TrainingManifest.model_validate(NO_TIER_FITS)
        tiers = [tier("l4-2x", "NVIDIA L4", 2, 1.0)]

        (check,) = check_tiers(tiers, manifest, builder())

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
            [tier("l4-2x", "NVIDIA L4", 2, 2.0)], manifest, builder()
        )

        # Assert
        assert check.estimate.fits
        assert not check.fits
        assert check.reason == "training over budget"

    @pytest.mark.parametrize("tiers", [[], [tier("cpu", None, 0, 0.5)]])
    def test_no_fitting_tier_leaves_no_fit(self, tiers):
        manifest = TrainingManifest.model_validate(MANIFEST)

        checks = check_tiers(tiers, manifest, builder())

        assert not any(c.fits for c in checks)

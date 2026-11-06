# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Cheapest resource tier for a manifest."""

import copy
import json
from pathlib import Path

import pytest

from agilerl.arena.memory import resources as resources_module
from agilerl.arena.memory.estimator import MemoryComponent, PhaseBreakdown, RunEstimate
from agilerl.arena.memory.formulas import MAX_UNDERPREDICTION
from agilerl.arena.memory.manifest import (
    device_spec_from_resource_class,
    lookup_gpu,
    run_config_from_manifest,
)
from agilerl.arena.memory.resolve.batch import catalog_model
from agilerl.arena.memory.resolve.engine import resolve_engine_tensor_parallel
from agilerl.arena.memory.resolve.parallel import ParallelLayout
from agilerl.arena.memory.resolve.split import SearchCache, resolve_run_memory
from agilerl.arena.memory.resources import (
    check_tiers,
    engine_width,
    engines_per_member,
    gpus_required,
    searches_gpu_split,
    split_manifest,
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


@pytest.fixture
def priced(monkeypatch):
    """``(tier name, node count)`` of every allocation the tier search prices."""
    calls = []
    real = resources_module._price

    def spy(tier, manifest, build_config, layout, num_nodes, cache):
        calls.append((tier["name"], num_nodes))
        return real(tier, manifest, build_config, layout, num_nodes, cache)

    monkeypatch.setattr(resources_module, "_price", spy)
    return calls


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


class TestSplitManifest:
    def test_writes_the_split_the_run_memory_search_picks(self):
        # Arrange
        manifest = TrainingManifest.model_validate(unset_split(MANIFEST))
        device = lookup_gpu("NVIDIA L4").device_spec()
        expected = resolve_run_memory(
            manifest, catalog_model(manifest.algorithm), device, 8
        ).split

        # Act
        laid_out, _step = split_manifest(manifest, 8, device)

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
        laid_out, _step = split_manifest(manifest, 8, device)

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
            split_manifest(manifest, 2, device)

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
            split_manifest(manifest, 8, device)


class TestCheckTiers:
    def test_omitted_split_is_sized_at_the_split_the_run_picks(self):
        # Arrange
        manifest = TrainingManifest.model_validate(unset_split(MANIFEST))
        device = lookup_gpu("NVIDIA L4").device_spec()
        expected, _step = split_manifest(manifest, 8, device)
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

    def test_tensor_parallel_engine_takes_a_second_node(self):
        # One 4-GPU node cannot hold a trainer and a width-4 engine. Two can.
        doc = copy.deepcopy(MANIFEST)
        doc["algorithm"]["vllm_config"]["tensor_parallel_size"] = 4
        manifest = TrainingManifest.model_validate(doc)
        tiers = [{**tier("l4-4x", "NVIDIA L4", 4, 4.0), "node_counts": [1, 2]}]

        (check,) = check_tiers(tiers, manifest, builder())

        assert check.fits
        assert check.num_nodes == 2
        assert check.engine_tensor_parallel == 4

    def test_a_tier_listing_node_counts_is_only_tried_at_those(self):
        doc = copy.deepcopy(MANIFEST)
        doc["algorithm"]["vllm_config"]["tensor_parallel_size"] = 4
        manifest = TrainingManifest.model_validate(doc)
        tiers = [{**tier("l4-4x", "NVIDIA L4", 4, 4.0), "node_counts": [1]}]

        (check,) = check_tiers(tiers, manifest, builder())

        assert not check.fits
        assert check.reason == "4 GPUs, the job needs 5"

    def test_a_tier_without_node_counts_is_its_num_gpus(self):
        doc = copy.deepcopy(MANIFEST)
        doc["algorithm"]["vllm_config"]["tensor_parallel_size"] = 4
        manifest = TrainingManifest.model_validate(doc)
        tiers = [{**tier("l4-4x", "NVIDIA L4", 8, 4.0), "gpus_per_node": 4}]

        (check,) = check_tiers(tiers, manifest, builder())

        assert check.fits
        assert check.num_nodes == 2

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

    def test_locks_the_least_parallel_layout_once_a_tier_fits(self, monkeypatch):
        # Arrange: a MoE, so expert parallel is legal; only A100 fits.
        manifest = TrainingManifest.model_validate(
            {
                **with_model(MANIFEST, "ibm-granite/granite-3.1-3b-a800m-instruct"),
                "training": {**MANIFEST["training"], "training_gpus_per_agent": 2},
            }
        )
        assert catalog_model(manifest.algorithm).arch.is_moe
        monkeypatch.setattr(
            "agilerl.arena.memory.resources.estimate_run",
            lambda config: flat_estimate(
                1_000, 10_000 if "A100" in config.train_device.name else 100
            ),
        )
        layouts = []

        def build(laid_out, device, width, engines):
            fsdp = laid_out.algorithm.fsdp
            layouts.append((fsdp.tp, fsdp.ep))
            return builder()(laid_out, device, width, engines)

        tiers = [
            tier("l4-1x", "NVIDIA L4", 1, 1.0),
            tier("l4-4x", "NVIDIA L4", 4, 2.0),
            tier("a100-4x", "NVIDIA A100-SXM4-80GB", 4, 8.0),
            tier("a100-8x", "NVIDIA A100-SXM4-80GB", 8, 16.0),
        ]

        # Act
        checks = check_tiers(tiers, manifest, build)

        # Assert
        by_name = {check.tier["name"]: check for check in checks}
        assert by_name["l4-1x"].reason == "1 GPUs, the job needs 3"
        assert by_name["l4-4x"].reason == "training over budget"
        assert by_name["a100-4x"].fits
        assert by_name["a100-4x"].num_nodes == 1
        assert by_name["a100-4x"].layout == ParallelLayout()
        assert by_name["a100-8x"].fits
        assert by_name["a100-8x"].layout == ParallelLayout()
        assert set(layouts) == {(1, 1)}

    def test_uses_more_nodes_before_more_parallelism(self, monkeypatch):
        # Arrange: 2 trainers and 2 engines need 4 GPUs; a 2-GPU node is short.
        manifest = TrainingManifest.model_validate(
            {
                **with_model(MANIFEST, "ibm-granite/granite-3.1-3b-a800m-instruct"),
                "training": {
                    **MANIFEST["training"],
                    "training_gpus_per_agent": 2,
                    "rollout_engines_per_agent": 2,
                },
            }
        )
        monkeypatch.setattr(
            "agilerl.arena.memory.resources.estimate_run",
            lambda _config: flat_estimate(1, 10_000),
        )
        tiers = [
            {**tier("l4-2x", "NVIDIA L4", 2, 1.0), "node_counts": [1, 2]},
            {**tier("a100-2x", "NVIDIA A100-SXM4-80GB", 2, 9.0), "node_counts": [1, 2]},
        ]

        # Act
        checks = check_tiers(tiers, manifest, builder())

        # Assert
        by_name = {check.tier["name"]: check for check in checks}
        assert by_name["l4-2x"].fits
        assert by_name["l4-2x"].num_nodes == 2
        assert by_name["l4-2x"].layout == ParallelLayout()
        assert by_name["a100-2x"].fits
        assert by_name["a100-2x"].num_nodes == 2

    def test_a_smaller_tier_than_the_lock_keeps_its_failure(self, monkeypatch):
        # Arrange: only the 8-GPU A100 node fits; the 4-GPU one is not re-tried.
        manifest = TrainingManifest.model_validate(MANIFEST)
        monkeypatch.setattr(
            "agilerl.arena.memory.resources.estimate_run",
            lambda config: flat_estimate(
                1_000, 10_000 if config.train_device.gpus_per_node == 8 else 100
            ),
        )
        tiers = [
            {**tier("a100-8x", "NVIDIA A100-SXM4-80GB", 8, 16.0), "node_counts": [1]},
            {**tier("a100-4x", "NVIDIA A100-SXM4-80GB", 4, 8.0), "node_counts": [1]},
        ]

        # Act
        checks = check_tiers(tiers, manifest, builder())

        # Assert
        assert [c.tier["name"] for c in checks] == ["a100-4x", "a100-8x"]
        assert checks[0].reason == "training over budget"
        assert checks[1].fits

    def test_a_low_host_miss_does_not_drop_a_roomier_tier(self, monkeypatch):
        # Arrange: 2 trainers and 2 engines need 4 GPUs, so one 1-GPU node is
        # short and the search prices the largest count next. The cheap
        # tier's host share is ram_gb / num_gpus and misses; the dearer
        # tier has more host RAM per GPU.
        manifest = TrainingManifest.model_validate(
            {
                **with_model(MANIFEST, "ibm-granite/granite-3.1-3b-a800m-instruct"),
                "training": {
                    **MANIFEST["training"],
                    "training_gpus_per_agent": 2,
                    "rollout_engines_per_agent": 2,
                },
            }
        )
        low = {
            **tier("l4-low", "NVIDIA L4", 2, 1.0),
            "gpus_per_node": 1,
            "ram_gb": 32,
            "node_counts": [1, 2, 4],
        }
        high = {
            **tier("l4-high", "NVIDIA L4", 1, 3.0),
            "gpus_per_node": 1,
            "ram_gb": 256,
            "node_counts": [1, 2, 4],
        }
        low_host = device_spec_from_resource_class(low).host_bytes_per_gpu

        def estimate(config):
            if config.train_device.host_bytes_per_gpu == low_host:
                return flat_estimate(10**15, 100)
            return flat_estimate(1, 10_000)

        monkeypatch.setattr(resources_module, "estimate_run", estimate)

        # Act
        checks = check_tiers([low, high], manifest, builder())

        # Assert
        by_name = {check.tier["name"]: check for check in checks}
        assert by_name["l4-high"].fits
        assert by_name["l4-high"].num_nodes == 4
        assert not by_name["l4-low"].fits

    def test_a_roomier_miss_still_skips_a_tighter_same_gpu_tier(
        self, monkeypatch, priced
    ):
        # Arrange
        manifest = manifest_with(
            MANIFEST, training_gpus_per_agent=4, rollout_engines_per_agent=1
        )
        monkeypatch.setattr(
            resources_module, "estimate_run", lambda _config: flat_estimate(10**15, 100)
        )
        rich = {
            **tier("l4-rich", "NVIDIA L4", 4, 1.0),
            "gpus_per_node": 2,
            "ram_gb": 256,
            "node_counts": [1, 2, 4],
        }
        poor = {
            **tier("l4-poor", "NVIDIA L4", 2, 2.0),
            "gpus_per_node": 2,
            "ram_gb": 32,
            "node_counts": [1, 2, 4],
        }

        # Act
        check_tiers([rich, poor], manifest, builder())

        # Assert
        assert ("l4-rich", 4) in priced
        assert ("l4-poor", 1) in priced
        assert all(num_nodes == 1 for name, num_nodes in priced if name == "l4-poor")

    def test_no_host_budget_covers_only_another_uncapped_tier(
        self, monkeypatch, priced
    ):
        # Arrange: omitted ram_gb skips the host check. A miss with no host
        # cap still skips another uncapped tier; a capped miss does not.
        manifest = manifest_with(
            MANIFEST, training_gpus_per_agent=4, rollout_engines_per_agent=1
        )
        monkeypatch.setattr(
            resources_module, "estimate_run", lambda _config: flat_estimate(10**15, 100)
        )

        def node_tier(name, price, ram_gb):
            row = {
                **tier(name, "NVIDIA L4", 4, price),
                "gpus_per_node": 2,
                "node_counts": [1, 2, 4],
            }
            if ram_gb is None:
                row.pop("ram_gb")
            else:
                row["ram_gb"] = ram_gb
            return row

        # Act
        check_tiers(
            [node_tier("open-a", 1.0, None), node_tier("open-b", 2.0, None)],
            manifest,
            builder(),
        )
        uncapped = list(priced)
        priced.clear()
        check_tiers(
            [node_tier("capped", 1.0, 64), node_tier("open", 2.0, None)],
            manifest,
            builder(),
        )

        # Assert
        assert ("open-a", 4) in uncapped
        assert ("open-b", 1) in uncapped
        assert all(num_nodes == 1 for name, num_nodes in uncapped if name == "open-b")
        assert ("open", 4) in priced

    def test_a_layout_that_fails_at_the_most_nodes_is_not_priced_at_fewer(
        self, monkeypatch, priced
    ):
        # Arrange
        manifest = TrainingManifest.model_validate(MANIFEST)
        monkeypatch.setattr(
            "agilerl.arena.memory.resources.estimate_run",
            lambda _config: flat_estimate(10**15, 100),
        )
        tiers = [{**tier("l4-4x", "NVIDIA L4", 4, 1.0), "node_counts": [1, 2, 4, 8]}]

        # Act
        check_tiers(tiers, manifest, builder())

        # Assert
        assert {num_nodes for _, num_nodes in priced} == {1, 8}

    def test_a_tier_skipped_after_a_same_gpu_miss_keeps_that_reason(self, monkeypatch):
        # Arrange: the cheap tier misses at 1 node and at its most, 8. The
        # dear tier offers no fewer than 4 nodes, so it is never priced.
        manifest = manifest_with(
            MANIFEST, training_gpus_per_agent=2, rollout_engines_per_agent=1
        )
        monkeypatch.setattr(
            resources_module, "estimate_run", lambda _config: flat_estimate(10**15, 100)
        )
        rich = {
            **tier("l4-rich", "NVIDIA L4", 2, 1.0),
            "ram_gb": 256,
            "node_counts": [1, 2, 8],
        }
        poor = {**tier("l4-poor", "NVIDIA L4", 2, 2.0), "node_counts": [4, 8]}

        # Act
        checks = check_tiers([rich, poor], manifest, builder())

        # Assert
        skipped = next(check for check in checks if check.tier["name"] == "l4-poor")
        assert skipped.reason == "training over budget"
        assert skipped.num_nodes == 8
        assert skipped.config is None

    def test_a_tier_the_lock_leaves_unpriced_says_why(self, monkeypatch):
        # Arrange: the cheaper A100 node fits first. The L4 has less memory;
        # the B200 only comes in pairs of nodes.
        manifest = TrainingManifest.model_validate(MANIFEST)
        monkeypatch.setattr(
            resources_module, "estimate_run", lambda _config: flat_estimate(1, 10_000)
        )
        tiers = [
            tier("a100-8x", "NVIDIA A100-SXM4-80GB", 8, 1.0),
            tier("l4-8x", "NVIDIA L4", 8, 2.0),
            {**tier("b200-8x", "NVIDIA B200", 8, 3.0), "node_counts": [2]},
        ]

        # Act
        checks = check_tiers(tiers, manifest, builder())

        # Assert
        by_name = {check.tier["name"]: check for check in checks}
        assert by_name["a100-8x"].fits
        assert by_name["l4-8x"].reason == (
            "fewer GPUs per node or less memory per GPU than a100-8x"
        )
        assert by_name["b200-8x"].reason == "no 1-node option, where a100-8x fits"
        assert by_name["b200-8x"].config is None

    def test_an_uncatalogued_model_is_priced_as_written(self, monkeypatch):
        # Arrange
        manifest = TrainingManifest.model_validate(
            with_model(MANIFEST, "local/uncatalogued-model")
        )
        monkeypatch.setattr(
            resources_module, "estimate_run", lambda _config: flat_estimate(1, 10_000)
        )

        # Act
        (check,) = check_tiers(
            [tier("l4-4x", "NVIDIA L4", 4, 1.0)], manifest, builder()
        )

        # Assert
        assert check.fits
        assert check.layout is None

    def test_a_pinned_shard_group_is_kept(self, monkeypatch):
        # Arrange: 2 trainers and 1 engine need 3 GPUs, so two 2-GPU nodes.
        doc = copy.deepcopy(MANIFEST)
        doc["algorithm"]["fsdp"] = {"shard_group_size": 2}
        manifest = manifest_with(
            doc, training_gpus_per_agent=2, rollout_engines_per_agent=1
        )
        monkeypatch.setattr(
            resources_module, "estimate_run", lambda _config: flat_estimate(1, 10_000)
        )
        tiers = [{**tier("l4-2x", "NVIDIA L4", 2, 1.0), "node_counts": [1, 2, 4]}]

        # Act
        (check,) = check_tiers(tiers, manifest, builder())

        # Assert
        assert check.fits
        assert check.num_nodes == 2
        assert check.layout == ParallelLayout(shard_group_size=2)

    def test_searched_layouts_stay_within_the_widest_node(self, monkeypatch):
        # Arrange: nothing fits, so every layout is priced; two 8-GPU nodes
        # give the split up to 15 trainers.
        manifest = TrainingManifest.model_validate(
            unset_split(unset_width(with_model(MANIFEST, "Qwen/Qwen3.6-35B-A3B")))
        )
        monkeypatch.setattr(
            resources_module, "estimate_run", lambda _config: flat_estimate(10**15, 100)
        )
        layouts = set()

        def build(laid_out, device, width, engines):
            fsdp = laid_out.algorithm.fsdp
            layouts.add((fsdp.tp, fsdp.ep))
            return builder()(laid_out, device, width, engines)

        tiers = [
            {
                **tier("h100-8x", "NVIDIA H100 80GB HBM3", 8, 1.0),
                "gpus_per_node": 8,
                "node_counts": [1, 2],
            }
        ]

        # Act
        check_tiers(tiers, manifest, build)

        # Assert
        assert (1, 8) in layouts
        assert max(tp * ep for tp, ep in layouts) == 8

    def test_unset_counts_split_the_gpus_at_the_least_parallel_layout(self):
        manifest = TrainingManifest.model_validate(unset_split(MANIFEST))
        assert searches_gpu_split(manifest)

        (check,) = check_tiers(
            [tier("l4-2x", "NVIDIA L4", 2, 2.0)], manifest, builder()
        )

        assert check.fits
        assert check.num_nodes == 1
        assert check.layout == ParallelLayout()
        assert check.trainers == 1
        assert check.engines == 1
        assert check.engine_tensor_parallel == 1

    def test_a_shared_search_cache_gives_the_same_verdicts(self):
        # Arrange
        manifest = TrainingManifest.model_validate(unset_split(MANIFEST))
        calls = [
            [{**tier("l4-2x", "NVIDIA L4", 2, 2.0), "node_counts": [count]}]
            for count in (1, 2)
        ]

        def verdicts(checks):
            return [
                (
                    c.fits,
                    c.reason,
                    c.num_nodes,
                    c.trainers,
                    c.engines,
                    c.engine_tensor_parallel,
                    c.step_seconds,
                )
                for c in checks
            ]

        # Act
        separate = [verdicts(check_tiers(t, manifest, builder())) for t in calls]
        cache = SearchCache()
        shared = [verdicts(check_tiers(t, manifest, builder(), cache)) for t in calls]

        # Assert
        assert cache.plans
        assert shared == separate

    def test_h100_80gb_is_priced_at_an_a100_80gb_lock(self, monkeypatch):
        # Arrange: H100 80GB shows a little less CUDA memory than A100 80GB.
        manifest = TrainingManifest.model_validate(MANIFEST)
        monkeypatch.setattr(
            resources_module, "estimate_run", lambda _config: flat_estimate(1, 10_000)
        )
        tiers = [
            tier("a100-8x", "NVIDIA A100-SXM4-80GB", 8, 16.0),
            tier("h100-8x", "NVIDIA H100 80GB HBM3", 8, 32.0),
        ]
        a100, h100 = (device_spec_from_resource_class(t) for t in tiers)
        assert h100.total_bytes < a100.total_bytes

        # Act
        checks = check_tiers(tiers, manifest, builder())

        # Assert
        assert [check.tier["name"] for check in checks] == ["a100-8x", "h100-8x"]
        assert checks[1].fits
        assert checks[1].config is not None
        assert checks[1].layout == checks[0].layout


def flat_estimate(peak: int, usable: int) -> RunEstimate:
    """A run whose training peak is ``peak`` bytes on a ``usable`` card, with an empty generation bar."""
    training = PhaseBreakdown(
        phase="training",
        components=(MemoryComponent(key="weights", label="Weights", n_bytes=peak),),
        device_total_bytes=usable,
        device_usable_bytes=usable,
    )
    generation = PhaseBreakdown(
        phase="generation",
        components=(),
        device_total_bytes=usable,
        device_usable_bytes=usable,
    )
    return RunEstimate(training=training, generation=generation)

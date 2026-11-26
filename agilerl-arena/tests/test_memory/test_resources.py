# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Cheapest resource tier for a manifest."""

import copy
import dataclasses
import json
from pathlib import Path

import pytest

from agilerl.arena.memory import resources as resources_module
from agilerl.arena.memory.estimator import MemoryComponent, PhaseBreakdown, RunEstimate
from agilerl.arena.memory.formulas import MAX_UNDERPREDICTION
from agilerl.arena.memory.manifest import (
    device_spec_from_resource_class,
    estimate_teacher_generation,
    lookup_gpu,
    run_config_from_manifest,
)
from agilerl.arena.memory.resolve.batch import catalog_model
from agilerl.arena.memory.resolve.engine import resolve_engine_tensor_parallel
from agilerl.arena.memory.resolve.fsdp import TrainingMemoryUnfitError
from agilerl.arena.memory.resolve.learn_costs import L4_LEARN_COSTS
from agilerl.arena.memory.resolve.lengths import ExpectedLengths
from agilerl.arena.memory.resolve.parallel import ParallelLayout
from agilerl.arena.memory.resolve.run import (
    SearchCache,
    SplitSearchError,
    resolve_run_memory,
)
from agilerl.arena.memory.resolve.split import RowTokens
from agilerl.arena.memory.resources import (
    FailureKind,
    NodeShardGroups,
    check_tiers,
    colocated_manifest,
    engine_width,
    engines_per_member,
    gpus_required,
    searches_gpu_split,
    split_manifest,
    teacher_sized_manifest,
    tier_sync_link,
)
from agilerl.arena.memory.solver import CannotSolve
from agilerl.arena.memory.specs import ModelArch, ModelSpec
from agilerl.arena.models.manifest import TrainingManifest
from agilerl.arena.models.model_info import SUPPORTED_MODEL_INFO
from tests.test_memory.test_cli import (
    MANIFEST,
    NO_TIER_FITS,
    OVERSIZE,
    SFT_MANIFEST,
    TINY_CONFIG,
)
from tests.test_memory.test_resolve_parallel import plan_for


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


def teacher_manifest(teacher="Qwen/Qwen3.8-27B", teacher_width=1, **training):
    """OPD on Qwen3.5-2B with a separate teacher model, one trainer and two engines."""
    return TrainingManifest.model_validate(
        {
            "algorithm": {
                "name": "OPD",
                "batch_size": 2,
                "vllm_config": {"tensor_parallel_size": 1},
                "teacher": {
                    "model": {
                        "pretrained_model_name_or_path": teacher,
                        "vllm_config": {"tensor_parallel_size": teacher_width},
                    }
                },
            },
            "environment": {
                "env_type": "rollout",
                "dataset": "openai/gsm8k",
                "reward_file_path": "reward.py",
                "prompt_template": {"user_0": "{question}"},
            },
            "network": {
                "pretrained_model_name_or_path": "Qwen/Qwen3.5-2B",
                "max_context_length": 512,
            },
            "training": {
                "max_steps": 100,
                "pop_size": 1,
                "training_gpus_per_agent": 1,
                "rollout_engines_per_agent": 2,
                "rollout_batch_size": 4,
                **training,
            },
        }
    )


def component(breakdown, key):
    return next(part for part in breakdown.components if part.key == key)


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


def without_expert_lora(doc):
    """Train no packed-expert LoRA, the layouts these MoE cases were sized for."""
    doc["network"]["lora_config"] = {"target_parameters": []}
    return doc


def moe_unset_layout():
    """Qwen3.6-35B-A3B at 32k tokens with the split and engine width omitted."""
    doc = unset_split(unset_width(with_model(MANIFEST, "Qwen/Qwen3.6-35B-A3B")))
    doc["network"]["max_context_length"] = 32768
    return without_expert_lora(doc)


def tiny_model():
    arch = ModelArch.from_hf_config(json.loads(Path(TINY_CONFIG).read_text()))
    return ModelSpec(model_id="tiny", arch=arch)


def builder():
    model_config = json.loads(Path(TINY_CONFIG).read_text())
    return lambda laid_out, device, width, engines, adapter_sync, nccl_reshard: (
        run_config_from_manifest(
            laid_out,
            device,
            model_config,
            tensor_parallel_size=width,
            rollout_engines=engines,
            adapter_sync=adapter_sync,
            nccl_reshard=nccl_reshard,
        )
    )


@pytest.fixture
def priced(monkeypatch):
    """``(tier name, node count)`` of every allocation the tier search prices."""
    calls = []
    real = resources_module._price

    def spy(
        tier,
        manifest,
        model,
        build_config,
        layout,
        num_nodes,
        cache,
        lengths,
        row_tokens,
    ):
        calls.append((tier["name"], num_nodes))
        return real(
            tier,
            manifest,
            model,
            build_config,
            layout,
            num_nodes,
            cache,
            lengths,
            row_tokens,
        )

    monkeypatch.setattr(resources_module, "_price", spy)
    return calls


@pytest.fixture
def built_layouts():
    """A builder that records ``(gpus per node, tp, ep, cp)`` for every allocation it sizes."""
    seen = set()

    def build(laid_out, device, width, engines, adapter_sync, nccl_reshard):
        fsdp = laid_out.algorithm.fsdp
        seen.add((device.gpus_per_node, fsdp.tp, fsdp.ep, fsdp.cp))
        return builder()(laid_out, device, width, engines, adapter_sync, nccl_reshard)

    return build, seen


@pytest.fixture
def built_split():
    """A builder that records ``(tp, ep, cp, trainers)`` for every allocation it sizes."""
    seen = set()

    def build(laid_out, device, width, engines, adapter_sync, nccl_reshard):
        fsdp = laid_out.algorithm.fsdp
        trainers = laid_out.training.training_gpus_per_agent
        seen.add((fsdp.tp, fsdp.ep, fsdp.cp, trainers))
        return builder()(laid_out, device, width, engines, adapter_sync, nccl_reshard)

    return build, seen


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

    def test_each_engine_counts_its_teacher_models_gpus(self):
        manifest = teacher_manifest(teacher_width=3)

        assert gpus_required(manifest, tensor_parallel_size=2) == 1 + 2 * (2 + 3)


class TestEnginesPerMemberWithTeacher:
    def test_auto_engines_leave_room_for_each_teacher(self):
        manifest = teacher_manifest(teacher_width=2, rollout_engines_per_agent="auto")

        assert engines_per_member(manifest, 7, tensor_parallel_size=1) == 2


class TestEstimateTeacherGeneration:
    def test_is_none_without_a_teacher_model(self):
        device = lookup_gpu("NVIDIA A100-SXM4-80GB").device_spec()

        estimate = estimate_teacher_generation(
            manifest_with(MANIFEST, rollout_engines_per_agent=1), device
        )

        assert estimate is None

    def test_a_larger_teacher_needs_more_of_its_gpu(self):
        # Arrange
        device = lookup_gpu("NVIDIA A100-SXM4-80GB").device_spec()
        small = teacher_manifest(teacher="Qwen/Qwen3.5-2B")
        large = teacher_manifest(teacher="Qwen/Qwen3.8-27B")

        # Act
        small_estimate = estimate_teacher_generation(small, device)
        large_estimate = estimate_teacher_generation(large, device)

        # Assert
        small_weights = component(small_estimate, "weights").n_bytes
        large_weights = component(large_estimate, "weights").n_bytes
        assert large_weights > 5 * small_weights

    def test_sharding_the_teacher_lowers_each_gpus_weights(self):
        # Arrange
        device = lookup_gpu("NVIDIA A100-SXM4-80GB").device_spec()

        # Act
        one = estimate_teacher_generation(
            teacher_manifest(teacher="Qwen/Qwen3.8-27B"), device
        )
        two = estimate_teacher_generation(
            teacher_manifest(teacher="Qwen/Qwen3.8-27B", teacher_width=2), device
        )

        # Assert
        assert component(two, "weights").n_bytes < component(one, "weights").n_bytes


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


class TestTierSyncLink:
    @pytest.mark.parametrize("gpus_per_node", [None, 8])
    def test_one_node_syncs_over_cuda_ipc(self, gpus_per_node):
        assert tier_sync_link(8, gpus_per_node) == "cuda_ipc"

    def test_allocation_across_nodes_has_no_known_link(self):
        assert tier_sync_link(16, 8) is None


class TestColocatedManifest:
    @pytest.mark.parametrize(("pop_size", "trainers"), [(1, 4), (2, 2), (8, 1)])
    def test_colocated_member_trains_on_every_gpu_it_owns(self, pop_size, trainers):
        manifest = manifest_with(SFT_MANIFEST, pop_size=pop_size)

        laid_out = colocated_manifest(manifest, 4)

        assert laid_out.training.training_gpus_per_agent == trainers
        assert manifest.training.training_gpus_per_agent == 1

    def test_authored_trainer_count_is_replaced(self):
        manifest = manifest_with(SFT_MANIFEST, training_gpus_per_agent=1)

        laid_out = colocated_manifest(manifest, 4)

        assert laid_out.training.training_gpus_per_agent == 4

    def test_async_manifest_is_unchanged(self):
        manifest = TrainingManifest.model_validate(MANIFEST)

        assert colocated_manifest(manifest, 8) is manifest


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
        laid_out, _plan = split_manifest(manifest, 8, device)

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
        laid_out, _plan = split_manifest(manifest, 8, device)

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


def unpinned_teacher_manifest(teacher):
    """Async OPD on Qwen3.5-2B from ``teacher``, with no GPU layout written."""
    return TrainingManifest.model_validate(
        {
            "algorithm": {
                "name": "OPD",
                "batch_size": 16,
                "max_output_tokens": 512,
                "teacher": {"model": {"pretrained_model_name_or_path": teacher}},
            },
            "environment": {
                "env_type": "rollout",
                "dataset": "openai/gsm8k",
                "reward_file_path": "reward.py",
                "prompt_template": {"user_0": "{question}"},
            },
            "network": {
                "pretrained_model_name_or_path": "Qwen/Qwen3.5-2B",
                "max_context_length": 2048,
            },
            "training": {"max_steps": 100, "pop_size": 1, "rollout_mode": "async"},
            "replay_buffer": {"kind": "llm"},
        }
    )


class TestTeacherSizedManifest:
    def test_an_unset_teacher_width_is_sized_on_a_copy(self):
        # Arrange
        manifest = unpinned_teacher_manifest("Qwen/Qwen3.8-27B")
        a100 = lookup_gpu("NVIDIA A100-SXM4-80GB").device_spec()

        # Act
        sized = teacher_sized_manifest(manifest, a100, 8, 8)

        # Assert
        assert sized.algorithm.teacher.model.vllm_config.tensor_parallel_size == 2
        assert manifest.algorithm.teacher.model.vllm_config.tensor_parallel_size is None

    def test_a_run_without_a_teacher_model_is_returned_as_is(self):
        manifest = manifest_with(MANIFEST, rollout_engines_per_agent=1)

        sized = teacher_sized_manifest(
            manifest, lookup_gpu("NVIDIA L4").device_spec(), 8, 8
        )

        assert sized is manifest

    def test_a_teacher_with_no_width_that_fits_is_returned_as_is(self):
        # Arrange: two GPUs leave no teacher width beside one trainer and one
        # student engine.
        manifest = unpinned_teacher_manifest("Qwen/Qwen3.8-27B")

        # Act
        sized = teacher_sized_manifest(
            manifest, lookup_gpu("NVIDIA L4").device_spec(), 2, 8
        )

        # Assert
        assert sized is manifest
        assert manifest.algorithm.teacher.model.vllm_config.tensor_parallel_size is None


class TestCheckTiers:
    def test_a_rollout_bundle_wider_than_a_node_does_not_fit(self):
        # Arrange: width 4 plus a width-2 teacher is 6 GPUs. Two 4-GPU nodes
        # have them, and one node cannot hold the bundle.
        manifest = teacher_manifest(
            "Qwen/Qwen3.5-9B", teacher_width=2, rollout_engines_per_agent=1
        )
        manifest.algorithm.vllm_config.tensor_parallel_size = 4
        tiers = [
            {
                **tier("a100-4x", "NVIDIA A100-SXM4-80GB", 4, 8.0),
                "gpus_per_node": 4,
                "node_counts": [2],
            }
        ]

        # Act
        (check,) = check_tiers(
            tiers, manifest, catalog_model(manifest.algorithm), builder()
        )

        # Assert
        assert not check.fits
        assert check.reason == "rollout bundle of 6 GPUs exceeds 4 GPUs on one node"

    def test_an_engine_wider_than_a_node_does_not_fit(self):
        # Arrange: a width-8 engine on 4-GPU nodes; no node can hold it.
        manifest = teacher_manifest(
            "Qwen/Qwen3.5-9B", teacher_width=2, rollout_engines_per_agent=1
        )
        manifest.algorithm.vllm_config.tensor_parallel_size = 8
        tiers = [
            {
                **tier("a100-4x", "NVIDIA A100-SXM4-80GB", 4, 8.0),
                "gpus_per_node": 4,
                "node_counts": [3],
            }
        ]

        # Act
        (check,) = check_tiers(
            tiers, manifest, catalog_model(manifest.algorithm), builder()
        )

        # Assert
        assert not check.fits
        assert (
            check.reason == "engine tensor_parallel_size 8 exceeds 4 GPUs on one node"
        )

    def test_a_teacher_model_without_a_layout_gets_a_split_that_fits(self):
        # Arrange: no student width, trainer or engine count, or teacher width.
        manifest = unpinned_teacher_manifest("Qwen/Qwen3.8-27B")

        # Act
        (check,) = check_tiers(
            [tier("a100-8x", "NVIDIA A100-SXM4-80GB", 8, 8.0)],
            manifest,
            catalog_model(manifest.algorithm),
            builder(),
        )

        # Assert
        assert check.fits, check.reason
        assert check.teacher_engine_tensor_parallel == 2
        per_engine = check.engine_tensor_parallel + check.teacher_engine_tensor_parallel
        assert check.trainers + check.engines * per_engine <= 8

    def test_a_run_without_a_teacher_model_reports_no_teacher_width(self):
        manifest = manifest_with(MANIFEST, rollout_engines_per_agent=1)

        (check,) = check_tiers(
            [tier("l4-8x", "NVIDIA L4", 8, 8.0)], manifest, tiny_model(), builder()
        )

        assert check.teacher_engine_tensor_parallel is None

    def test_a_teacher_too_big_for_its_gpu_rules_the_tier_out(self):
        # Arrange: a 27B teacher's bf16 weights alone exceed one L4.
        manifest = teacher_manifest(teacher="Qwen/Qwen3.8-27B", teacher_width=1)

        # Act
        (check,) = check_tiers(
            [tier("l4-8x", "NVIDIA L4", 8, 8.0)], manifest, tiny_model(), builder()
        )

        # Assert
        assert not check.fits
        assert check.reason == "teacher generation over budget"

    def test_colocated_run_is_sized_on_every_gpu_of_the_member(self):
        # Arrange
        manifest = TrainingManifest.model_validate(SFT_MANIFEST)
        sized = []
        build = builder()

        def recording_build(
            laid_out, train_device, width, engines, adapter_sync, nccl_reshard
        ):
            sized.append(laid_out.training.training_gpus_per_agent)
            return build(
                laid_out, train_device, width, engines, adapter_sync, nccl_reshard
            )

        # Act
        (check,) = check_tiers(
            [tier("l4-4x", "NVIDIA L4", 4, 4.0)],
            manifest,
            tiny_model(),
            recording_build,
        )

        # Assert
        assert check.fits
        assert check.trainers == 4
        assert set(sized) == {4}

    def test_omitted_split_is_sized_at_the_split_the_run_picks(self):
        # Arrange
        manifest = TrainingManifest.model_validate(unset_split(MANIFEST))
        device = lookup_gpu("NVIDIA L4").device_spec()
        expected, _plan = split_manifest(manifest, 8, device, sync_link="cuda_ipc")
        sized = []
        build = builder()

        def recording_build(
            laid_out, train_device, width, engines, adapter_sync, nccl_reshard
        ):
            sized.append((laid_out, engines))
            return build(
                laid_out, train_device, width, engines, adapter_sync, nccl_reshard
            )

        # Act
        (check,) = check_tiers(
            [tier("l4-8x", "NVIDIA L4", 8, 8.0)],
            manifest,
            tiny_model(),
            recording_build,
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

        def recording_build(
            laid_out, train_device, width, engines, adapter_sync, nccl_reshard
        ):
            widths.append((width, engines))
            return build(
                laid_out, train_device, width, engines, adapter_sync, nccl_reshard
            )

        # Act
        check_tiers(
            [tier("h100-8x", "NVIDIA H100 80GB HBM3", 8, 30.0)],
            manifest,
            tiny_model(),
            recording_build,
        )

        # Assert
        assert widths == [(2, 3)]

    def test_no_fitting_split_keeps_the_search_reason(self):
        manifest = TrainingManifest.model_validate(
            unset_split(with_model(OVERSIZE, "Qwen/Qwen3.8-27B"))
        )

        (check,) = check_tiers(
            [tier("l4-2x", "NVIDIA L4", 2, 1.0)], manifest, tiny_model(), builder()
        )

        assert not check.fits
        assert check.reason.startswith("No trainer / rollout-engine split of 2")
        assert check.estimate is not None
        assert check.failure == FailureKind.GPU_MEMORY
        assert check.host_limited is False

    @pytest.mark.parametrize(
        ("gpu_type", "label"), [("Tesla T4", "SM75"), ("Tesla V100", "SM70")]
    )
    def test_a_gpu_below_sm80_is_ruled_out_as_unsupported(self, gpu_type, label):
        manifest = TrainingManifest.model_validate(unset_split(MANIFEST))

        (check,) = check_tiers(
            [tier("old-2x", gpu_type, 2, 1.0)], manifest, tiny_model(), builder()
        )

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
        checks = check_tiers(tiers, manifest, tiny_model(), builder())

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

        (check,) = check_tiers(tiers, manifest, tiny_model(), builder())

        assert check.fits
        assert check.num_nodes == 2
        assert check.engine_tensor_parallel == 4

    def test_a_tier_listing_node_counts_is_only_tried_at_those(self):
        doc = copy.deepcopy(MANIFEST)
        doc["algorithm"]["vllm_config"]["tensor_parallel_size"] = 4
        manifest = TrainingManifest.model_validate(doc)
        tiers = [{**tier("l4-4x", "NVIDIA L4", 4, 4.0), "node_counts": [1]}]

        (check,) = check_tiers(tiers, manifest, tiny_model(), builder())

        assert not check.fits
        assert check.reason == "4 GPUs, the job needs 5"

    def test_a_tier_without_node_counts_is_its_num_gpus(self):
        doc = copy.deepcopy(MANIFEST)
        doc["algorithm"]["vllm_config"]["tensor_parallel_size"] = 4
        manifest = TrainingManifest.model_validate(doc)
        tiers = [{**tier("l4-4x", "NVIDIA L4", 8, 4.0), "gpus_per_node": 4}]

        (check,) = check_tiers(tiers, manifest, tiny_model(), builder())

        assert check.fits
        assert check.num_nodes == 2

    def test_num_gpus_over_part_of_a_node_is_the_allocation(self):
        # Arrange: 6 GPUs on 4-GPU nodes hold one trainer and five engines.
        manifest = manifest_with(MANIFEST, rollout_engines_per_agent=5)
        tiers = [{**tier("l4-6x", "NVIDIA L4", 6, 4.0), "gpus_per_node": 4}]

        # Act
        (check,) = check_tiers(tiers, manifest, tiny_model(), builder())

        # Assert
        assert check.fits
        assert check.num_nodes == 2
        assert check.engines == 5

    def test_num_gpus_under_one_node_is_the_allocation(self):
        # Arrange: 2 GPUs of a 4-GPU node cannot hold one trainer and two engines.
        manifest = manifest_with(MANIFEST, rollout_engines_per_agent=2)
        tiers = [{**tier("l4-2x", "NVIDIA L4", 2, 2.0), "gpus_per_node": 4}]

        # Act
        (check,) = check_tiers(tiers, manifest, tiny_model(), builder())

        # Assert
        assert not check.fits
        assert check.reason == "2 GPUs, the job needs 3"

    def test_auto_engines_size_the_sequences_the_tier_splits(self):
        # Arrange: 8 prompts of group 4 over the four engines 5 GPUs leave.
        manifest = manifest_with(
            MANIFEST, rollout_engines_per_agent="auto", rollout_batch_size=8
        )
        tiers = [tier("l4-5x", "NVIDIA L4", 5, 5.0)]

        # Act
        (check,) = check_tiers(tiers, manifest, tiny_model(), builder())

        # Assert
        assert check.config.generation.tensor_parallel_size == 1
        assert check.config.generation.max_num_seqs == 2 * 4

    def test_a_tier_whose_gpu_is_too_small_is_over_budget(self):
        manifest = TrainingManifest.model_validate(NO_TIER_FITS)
        tiers = [tier("l4-2x", "NVIDIA L4", 2, 1.0)]

        (check,) = check_tiers(tiers, manifest, tiny_model(), builder())

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
            [tier("l4-2x", "NVIDIA L4", 2, 2.0)], manifest, tiny_model(), builder()
        )

        # Assert
        assert check.estimate.fits
        assert not check.fits
        assert check.reason == "training over budget"

    @pytest.mark.parametrize("tiers", [[], [tier("cpu", None, 0, 0.5)]])
    def test_no_fitting_tier_leaves_no_fit(self, tiers):
        manifest = TrainingManifest.model_validate(MANIFEST)

        checks = check_tiers(tiers, manifest, tiny_model(), builder())

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

        def build(laid_out, device, width, engines, adapter_sync, nccl_reshard):
            fsdp = laid_out.algorithm.fsdp
            layouts.append((fsdp.tp, fsdp.ep))
            return builder()(
                laid_out, device, width, engines, adapter_sync, nccl_reshard
            )

        tiers = [
            tier("l4-1x", "NVIDIA L4", 1, 1.0),
            tier("l4-4x", "NVIDIA L4", 4, 2.0),
            tier("a100-4x", "NVIDIA A100-SXM4-80GB", 4, 8.0),
            tier("a100-8x", "NVIDIA A100-SXM4-80GB", 8, 16.0),
        ]

        # Act
        checks = check_tiers(tiers, manifest, tiny_model(), build)

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
        checks = check_tiers(tiers, manifest, tiny_model(), builder())

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
        checks = check_tiers(tiers, manifest, tiny_model(), builder())

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
        checks = check_tiers([low, high], manifest, tiny_model(), builder())

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
        check_tiers([rich, poor], manifest, tiny_model(), builder())

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
            tiny_model(),
            builder(),
        )
        uncapped = list(priced)
        priced.clear()
        check_tiers(
            [node_tier("capped", 1.0, 64), node_tier("open", 2.0, None)],
            manifest,
            tiny_model(),
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
        check_tiers(tiers, manifest, tiny_model(), builder())

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
        checks = check_tiers([rich, poor], manifest, tiny_model(), builder())

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
        checks = check_tiers(tiers, manifest, tiny_model(), builder())

        # Assert
        by_name = {check.tier["name"]: check for check in checks}
        assert by_name["a100-8x"].fits
        assert by_name["l4-8x"].reason == (
            "fewer GPUs per node or less memory per GPU than a100-8x"
        )
        assert by_name["b200-8x"].reason == "no 1-node option, where a100-8x fits"
        assert by_name["b200-8x"].config is None

    def test_an_uncatalogued_model_cannot_be_priced(self, monkeypatch):
        # Arrange
        manifest = TrainingManifest.model_validate(
            with_model(MANIFEST, "local/uncatalogued-model")
        )
        monkeypatch.setattr(
            resources_module, "estimate_run", lambda _config: flat_estimate(1, 10_000)
        )

        # Act / Assert
        with pytest.raises(
            ValueError,
            match="'local/uncatalogued-model' is not in the Arena model catalog",
        ):
            check_tiers(
                [tier("l4-4x", "NVIDIA L4", 4, 1.0)], manifest, tiny_model(), builder()
            )

    def test_an_uncatalogued_model_with_a_searched_split_cannot_be_priced(self):
        # Arrange
        manifest = TrainingManifest.model_validate(
            unset_split(with_model(MANIFEST, "acme/not-in-the-catalogue"))
        )

        # Act / Assert
        with pytest.raises(
            ValueError,
            match="'acme/not-in-the-catalogue' is not in the Arena model catalog",
        ):
            check_tiers(
                [tier("l4-2x", "NVIDIA L4", 2, 1.0)], manifest, tiny_model(), builder()
            )

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
        (check,) = check_tiers(tiers, manifest, tiny_model(), builder())

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

        def build(laid_out, device, width, engines, adapter_sync, nccl_reshard):
            fsdp = laid_out.algorithm.fsdp
            layouts.add((fsdp.tp, fsdp.ep, fsdp.cp))
            return builder()(
                laid_out, device, width, engines, adapter_sync, nccl_reshard
            )

        tiers = [
            {
                **tier("h100-8x", "NVIDIA H100 80GB HBM3", 8, 1.0),
                "gpus_per_node": 8,
                "node_counts": [1, 2],
            }
        ]

        # Act
        check_tiers(tiers, manifest, tiny_model(), build)

        # Assert
        assert (1, 8, 1) in layouts
        assert (1, 1, 8) in layouts
        assert max(ep for _, ep, _ in layouts) == 8
        assert max(tp * cp for tp, _, cp in layouts) == 8

    def test_a_tier_prices_only_searched_layouts_on_its_nodes(
        self, monkeypatch, built_layouts
    ):
        # Arrange: nothing fits, so every layout is priced. Two 8-GPU nodes
        # hold groups of 8; two 4-GPU nodes hold groups that divide 4, and
        # expert groups of both whole nodes.
        manifest = TrainingManifest.model_validate(
            unset_split(unset_width(with_model(MANIFEST, "Qwen/Qwen3.6-35B-A3B")))
        )
        monkeypatch.setattr(
            resources_module, "estimate_run", lambda _config: flat_estimate(10**15, 100)
        )
        build, seen = built_layouts
        tiers = [
            {
                **tier("h100-4x", "NVIDIA H100 80GB HBM3", 4, 1.0),
                "node_counts": [2],
            },
            {
                **tier("h100-8x", "NVIDIA H100 80GB HBM3", 8, 2.0),
                "node_counts": [2],
            },
        ]

        # Act
        check_tiers(tiers, manifest, tiny_model(), build)

        # Assert
        on_four = {(tp, ep, cp) for node, tp, ep, cp in seen if node == 4}
        on_eight = {(tp, ep, cp) for node, tp, ep, cp in seen if node == 8}
        assert {ep for _, ep, _ in on_four} == {1, 2, 4, 8}
        assert {tp * cp for tp, _, cp in on_four} == {1, 2, 4}
        assert (1, 8, 1) in on_eight
        assert (1, 1, 8) in on_eight

    def test_kept_trainers_over_two_nodes_price_no_straddling_group(
        self, monkeypatch, built_layouts
    ):
        # Arrange: 10 trainers on two 8-GPU nodes. ep=5 and ep=10 split the
        # 40 experts and the trainers, but a group of 5 or 10 crosses a node.
        manifest = manifest_with(
            with_model(MANIFEST, "ibm-granite/granite-3.1-3b-a800m-instruct"),
            training_gpus_per_agent=10,
            rollout_engines_per_agent=1,
        )
        monkeypatch.setattr(
            resources_module, "estimate_run", lambda _config: flat_estimate(10**15, 100)
        )
        build, seen = built_layouts
        tiers = [{**tier("l4-8x", "NVIDIA L4", 8, 1.0), "node_counts": [2]}]

        # Act
        (check,) = check_tiers(tiers, manifest, tiny_model(), build)

        # Assert
        assert not check.fits
        assert {ep for _, _, ep, _ in seen} == {1, 2}
        assert {tp * cp for _, tp, _, cp in seen} == {1, 2}

    @pytest.mark.parametrize(
        ("model", "trainers", "ep", "node_gpus", "num_nodes"),
        [
            ("Qwen/Qwen3.6-35B-A3B", 8, 8, 4, 4),
            ("ibm-granite/granite-3.1-3b-a800m-instruct", 10, 5, 8, 2),
        ],
        ids=["wider-than-node", "straddles-nodes"],
    )
    def test_a_pinned_layout_off_the_tier_nodes_is_priced_as_written(
        self, model, trainers, ep, node_gpus, num_nodes
    ):
        # Arrange: the batch splits over the trainers.
        doc = with_model(MANIFEST, model)
        doc["algorithm"]["fsdp"] = {"ep": ep}
        doc["algorithm"]["batch_size"] = trainers
        manifest = manifest_with(
            doc, training_gpus_per_agent=trainers, rollout_engines_per_agent=1
        )
        tiers = [
            {
                **tier("node-tier", "NVIDIA H100 80GB HBM3", node_gpus, 1.0),
                "node_counts": [1, 2, 4],
            }
        ]

        # Act
        (check,) = check_tiers(tiers, manifest, tiny_model(), builder())

        # Assert
        assert check.fits
        assert check.layout == ParallelLayout(ep=ep)
        assert check.num_nodes == num_nodes
        assert check.trainers == trainers

    def test_a_pinned_layout_off_the_nodes_is_priced_under_an_open_count(self):
        # Arrange: up to 15 trainers on two 8-GPU nodes; pinned ep=5 groups
        # straddle the nodes past 8 of them.
        doc = without_expert_lora(
            unset_split(
                with_model(MANIFEST, "ibm-granite/granite-3.1-3b-a800m-instruct")
            )
        )
        doc["algorithm"]["fsdp"] = {"ep": 5}
        doc["algorithm"]["batch_size"] = 120
        manifest = TrainingManifest.model_validate(doc)
        tiers = [
            {**tier("h100-8x", "NVIDIA H100 80GB HBM3", 8, 1.0), "node_counts": [2]}
        ]

        # Act
        (check,) = check_tiers(tiers, manifest, tiny_model(), builder())

        # Assert
        assert check.fits
        assert check.layout == ParallelLayout(ep=5)
        assert check.trainers == 10
        assert check.engines == 6

    def test_an_open_count_prices_searched_groups_on_one_node(
        self, monkeypatch, built_split
    ):
        # Arrange: nothing fits, so every layout is priced. Up to 15 trainers
        # on two 8-GPU nodes; ep=5 sits on one node at 5 trainers only.
        doc = without_expert_lora(
            unset_split(
                with_model(MANIFEST, "ibm-granite/granite-3.1-3b-a800m-instruct")
            )
        )
        doc["algorithm"]["batch_size"] = 40
        manifest = TrainingManifest.model_validate(doc)
        monkeypatch.setattr(
            resources_module, "estimate_run", lambda _config: flat_estimate(10**15, 100)
        )
        build, seen = built_split
        tiers = [
            {**tier("h100-8x", "NVIDIA H100 80GB HBM3", 8, 1.0), "node_counts": [2]}
        ]

        # Act
        check_tiers(tiers, manifest, tiny_model(), build)

        # Assert
        assert (1, 5, 1, 5) in seen
        assert all(
            8 % ep == 0 and 8 % (tp * cp) == 0
            for tp, ep, cp, trainers in seen
            if trainers > 8
        )

    def test_a_pinned_layout_the_trainers_cannot_split_is_priced_as_written(self):
        # Arrange: tp=2 does not divide 3 trainers.
        doc = copy.deepcopy(MANIFEST)
        doc["algorithm"]["fsdp"] = {"tp": 2}
        manifest = manifest_with(
            doc, training_gpus_per_agent=3, rollout_engines_per_agent=1
        )

        # Act
        (check,) = check_tiers(
            [tier("l4-4x", "NVIDIA L4", 4, 1.0)], manifest, tiny_model(), builder()
        )

        # Assert
        assert not check.fits
        assert check.reason == "No training layout can run on this model"
        assert check.layout is None

    def test_an_engineless_algorithm_fits_its_trainers_alone(self):
        manifest = TrainingManifest.model_validate(SFT_MANIFEST)

        (check,) = check_tiers(
            [tier("l4-1x", "NVIDIA L4", 1, 1.0)], manifest, tiny_model(), builder()
        )

        assert check.fits
        assert check.trainers == 1
        assert check.engines == 0

    def test_a_split_with_no_room_keeps_the_gpu_count_reason(self):
        # Arrange: one GPU holds no trainer and engine split, nor the
        # manifest's own trainer and engine.
        manifest = TrainingManifest.model_validate(unset_split(MANIFEST))

        # Act
        (check,) = check_tiers(
            [tier("l4-1x", "NVIDIA L4", 1, 1.0)], manifest, tiny_model(), builder()
        )

        # Assert
        assert not check.fits
        assert check.reason == "1 GPUs, the job needs 2"
        assert check.config is None
        assert check.failure == FailureKind.OTHER

    def test_unset_counts_split_the_gpus_at_the_least_parallel_layout(self):
        manifest = TrainingManifest.model_validate(unset_split(MANIFEST))
        assert searches_gpu_split(manifest)

        (check,) = check_tiers(
            [tier("l4-2x", "NVIDIA L4", 2, 2.0)], manifest, tiny_model(), builder()
        )

        assert check.fits
        assert check.num_nodes == 1
        assert check.layout == ParallelLayout()
        assert check.trainers == 1
        assert check.engines == 1
        assert check.engine_tensor_parallel == 1

    @pytest.mark.parametrize("batch_size", [2, 64])
    def test_searched_split_steps_within_its_generation_range(self, batch_size):
        # Arrange
        doc = unset_split(MANIFEST)
        doc["algorithm"]["batch_size"] = batch_size
        manifest = TrainingManifest.model_validate(doc)
        lengths = ExpectedLengths(
            prompt_tokens=100,
            generated_tokens_per_turn=250,
            turns=1,
            generated_tokens_per_turn_low=100,
            generated_tokens_per_turn_high=400,
        )

        # Act
        (check,) = check_tiers(
            [tier("l4-4x", "NVIDIA L4", 4, 4.0)],
            manifest,
            tiny_model(),
            builder(),
            expected_lengths=lengths,
        )

        # Assert
        assert check.fits
        assert check.step_seconds is not None
        assert check.step_seconds_low is not None
        assert check.step_seconds_high is not None
        assert check.step_seconds_low <= check.step_seconds <= check.step_seconds_high

    def test_default_lengths_give_one_step(self):
        # Arrange
        manifest = TrainingManifest.model_validate(unset_split(MANIFEST))

        # Act
        (check,) = check_tiers(
            [tier("l4-4x", "NVIDIA L4", 4, 4.0)], manifest, tiny_model(), builder()
        )

        # Assert
        assert check.step_seconds is not None
        assert check.step_seconds_low == check.step_seconds == check.step_seconds_high

    def test_multi_turn_split_steps_within_its_generation_range(self):
        # Arrange: a Sudoku-style env on Qwen2.5-0.5B.
        doc = unset_split(MANIFEST)
        doc["algorithm"]["batch_size"] = 16
        doc["algorithm"]["max_output_tokens"] = 256
        doc["environment"] = {
            "env_type": "rollout",
            "factory": "gem:make",
            "entrypoint": "game:Sudoku-v0-hard",
            "max_turns": 20,
            "max_reward": 1.0,
        }
        doc["network"] = {**doc["network"], "max_context_length": 4096}
        manifest = TrainingManifest.model_validate(doc)
        lengths = ExpectedLengths(
            prompt_tokens=600,
            generated_tokens_per_turn=150,
            turns=4,
            generated_tokens_per_turn_low=60,
            generated_tokens_per_turn_high=220,
        )

        # Act
        (check,) = check_tiers(
            [tier("l4-4x", "NVIDIA L4", 4, 4.0)],
            manifest,
            tiny_model(),
            builder(),
            expected_lengths=lengths,
        )

        # Assert
        assert check.fits
        assert check.step_seconds is not None
        assert check.step_seconds_low is not None
        assert check.step_seconds_high is not None
        assert check.step_seconds_low <= check.step_seconds <= check.step_seconds_high
        assert check.step_seconds_low < check.step_seconds_high

    def test_expected_lengths_price_one_step(self):
        # Arrange
        manifest = TrainingManifest.model_validate(unset_split(MANIFEST))
        tiers = [tier("l4-4x", "NVIDIA L4", 4, 4.0)]
        lengths = ExpectedLengths(
            prompt_tokens=200, generated_tokens_per_turn=250, turns=1
        )

        # Act
        (measured,) = check_tiers(
            tiers, manifest, tiny_model(), builder(), expected_lengths=lengths
        )
        (default,) = check_tiers(tiers, manifest, tiny_model(), builder())

        # Assert
        assert measured.step_seconds is not None
        assert measured.step_seconds_low == measured.step_seconds
        assert measured.step_seconds_high == measured.step_seconds
        assert measured.step_seconds != default.step_seconds

    def test_range_at_the_mean_reprices_the_searched_step(self):
        # Arrange
        manifest = TrainingManifest.model_validate(unset_split(MANIFEST))
        device = lookup_gpu("NVIDIA L4").device_spec()
        sync_link = tier_sync_link(4, device.gpus_per_node)
        lengths = ExpectedLengths(
            prompt_tokens=200, generated_tokens_per_turn=250, turns=1
        )
        laid_out, plan = split_manifest(
            manifest, 4, device, sync_link=sync_link, expected_lengths=lengths
        )
        at_mean = lengths.model_copy(
            update={
                "generated_tokens_per_turn_low": 250,
                "generated_tokens_per_turn_high": 250,
            }
        )

        # Act
        low, high = resources_module._step_range(
            laid_out,
            plan,
            catalog_model(laid_out.algorithm),
            device,
            sync_link,
            at_mean,
        )

        # Assert
        assert plan.split is not None
        assert low == pytest.approx(plan.split.step_seconds)
        assert high == pytest.approx(plan.split.step_seconds)

    def test_a_plan_without_a_split_has_no_step_range(self):
        # Arrange
        manifest = TrainingManifest.model_validate(unset_split(MANIFEST))
        device = lookup_gpu("NVIDIA L4").device_spec()
        sync_link = tier_sync_link(4, device.gpus_per_node)
        laid_out, plan = split_manifest(manifest, 4, device, sync_link=sync_link)

        # Act
        step_range = resources_module._step_range(
            laid_out,
            dataclasses.replace(plan, split=None),
            catalog_model(laid_out.algorithm),
            device,
            sync_link,
            None,
        )

        # Assert
        assert step_range is None

    def test_kept_counts_have_no_step_range(self):
        # Act
        (check,) = check_tiers(
            [tier("l4-2x", "NVIDIA L4", 2, 2.0)],
            TrainingManifest.model_validate(MANIFEST),
            tiny_model(),
            builder(),
        )

        # Assert
        assert check.step_seconds is None
        assert check.step_seconds_low is None
        assert check.step_seconds_high is None

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
        separate = [
            verdicts(check_tiers(t, manifest, tiny_model(), builder())) for t in calls
        ]
        cache = SearchCache()
        shared = [
            verdicts(check_tiers(t, manifest, tiny_model(), builder(), cache))
            for t in calls
        ]

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
        checks = check_tiers(tiers, manifest, tiny_model(), builder())

        # Assert
        assert [check.tier["name"] for check in checks] == ["a100-8x", "h100-8x"]
        assert checks[1].fits
        assert checks[1].config is not None
        assert checks[1].layout == checks[0].layout


def dataset_step(name, batch_size=8, context=512, num_gpus=1, row_tokens=None):
    doc = copy.deepcopy(SFT_MANIFEST)
    doc["algorithm"] = {"name": name, "batch_size": batch_size}
    doc["network"]["max_context_length"] = context
    manifest = TrainingManifest.model_validate(doc)
    (check,) = check_tiers(
        [tier("l4", "NVIDIA L4", num_gpus, 1.0)],
        manifest,
        tiny_model(),
        builder(),
        row_tokens=row_tokens,
    )
    assert check.fits
    return check.step_seconds


def dataset_token_work(name, **step):
    """``dataset_step`` less its host time, for a step that trains in one micro-batch.

    SFT's forward and backward are two model calls; DPO's chosen and rejected
    forwards and backwards and their reference forwards are six.
    """
    calls = {"SFT": 2, "DPO": 6}[name]
    return dataset_step(name, **step) - calls * L4_LEARN_COSTS.pass_overhead_seconds


class TestCheckTiersStepSeconds:
    @pytest.mark.parametrize("name", ["SFT", "DPO"])
    def test_sft_and_dpo_price_a_learn_step(self, name):
        # Act
        seconds = dataset_step(name)

        # Assert
        assert seconds is not None
        assert seconds > 0

    @pytest.mark.parametrize("name", ["SFT", "DPO"])
    def test_each_model_call_pays_the_host_overhead(self, name):
        # Act
        seconds = dataset_step(name)
        token_work = dataset_token_work(name)

        # Assert: the tiny model's step is mostly host time.
        assert 0 < token_work < seconds - token_work

    @pytest.mark.parametrize("name", ["SFT", "DPO"])
    def test_a_larger_batch_in_one_micro_batch_adds_only_token_work(self, name):
        # Act
        small = dataset_token_work(name, batch_size=8)
        large = dataset_token_work(name, batch_size=16)

        # Assert
        assert large == pytest.approx(2 * small, rel=1e-9)

    def test_a_longer_context_takes_at_least_proportionally_longer(self):
        # Act
        short = dataset_token_work("SFT", context=512)
        long = dataset_token_work("SFT", context=1024)

        # Assert: attention grows with the square of the row.
        assert long > 1.9 * short

    def test_more_gpus_shorten_the_step(self):
        # Act
        one = dataset_token_work("SFT", batch_size=16, num_gpus=1)
        four = dataset_token_work("SFT", batch_size=16, num_gpus=4)

        # Assert
        assert four == pytest.approx(one / 4, rel=1e-9)

    def test_dpo_costs_about_three_sft_steps(self):
        # Act
        sft = dataset_step("SFT")
        dpo = dataset_step("DPO")

        # Assert: two rows per pair, and a reference forward over both.
        assert dpo == pytest.approx(3 * sft, rel=0.1)

    @pytest.mark.parametrize("name", ["SFT", "DPO"])
    def test_short_dataset_rows_shorten_the_step(self, name):
        # Act
        full = dataset_step(name, context=512)
        short = dataset_step(name, context=512, row_tokens=RowTokens(128.0, 128.0**2))

        # Assert: token work shrinks; each micro-batch's fixed cost stays.
        assert full / 16 < short < full

    @pytest.mark.parametrize(
        ("name", "expected"),
        [("SFT", 0.47422456727567436), ("DPO", 1.4226737018270232)],
    )
    def test_unknown_row_lengths_keep_the_max_model_len_step(self, name, expected):
        # Act
        seconds = dataset_step(name, row_tokens=None)

        # Assert: the step priced at max_model_len rows.
        assert seconds == pytest.approx(expected, rel=1e-12)

    def test_a_searched_rollout_split_keeps_its_step(self):
        # Arrange
        manifest = TrainingManifest.model_validate(unset_split(MANIFEST))

        # Act
        (check,) = check_tiers(
            [tier("l4-2x", "NVIDIA L4", 2, 1.0)], manifest, tiny_model(), builder()
        )

        # Assert: the split's predicted step, unchanged by dataset pricing.
        assert check.step_seconds == pytest.approx(0.7876836424373541, rel=1e-12)

    def test_kept_rollout_counts_have_no_step(self):
        # Arrange
        manifest = TrainingManifest.model_validate(MANIFEST)

        # Act
        (check,) = check_tiers(
            [tier("l4-2x", "NVIDIA L4", 2, 1.0)], manifest, tiny_model(), builder()
        )

        # Assert
        assert check.fits
        assert check.step_seconds is None


def short_row_dataset_manifest(name: str, context: int = 4096) -> TrainingManifest:
    """Qwen3.5-4B LoRA SFT or DPO, eight rows a step."""
    columns = (
        {"prompt": "prompt", "chosen": "chosen", "rejected": "rejected"}
        if name == "DPO"
        else {"prompt": "prompt", "target": "chosen"}
    )
    return TrainingManifest.model_validate(
        {
            "algorithm": {"name": name, "batch_size": 8},
            "environment": {
                "dataset": "humans",
                "env_type": "dataset",
                "objective": "preference" if name == "DPO" else "sft",
                "columns": columns,
            },
            "network": {
                "pretrained_model_name_or_path": "Qwen/Qwen3.5-4B",
                "max_context_length": context,
                "lora_config": {"lora_r": 16, "lora_alpha": 32},
            },
            "training": {"max_steps": 60, "pop_size": 1},
        }
    )


class TestCheckTiersShortRowDatasetSteps:
    @pytest.mark.parametrize(
        ("name", "gpu_type", "gpus", "context", "mean", "rms", "measured"),
        [
            ("SFT", "NVIDIA A100-SXM4-80GB", 1, 4096, 285, 320, 1.81),
            ("DPO", "NVIDIA A100-SXM4-80GB", 1, 4096, 271, 315, 10.84),
            ("SFT", "NVIDIA L4", 1, 4096, 285, 320, 2.04),
            ("DPO", "NVIDIA L4", 1, 4096, 271, 315, 5.30),
            # Eight trainers, one row each.
            ("SFT", "NVIDIA H100 80GB HBM3", 8, 16384, 279, 323, 0.35),
            ("DPO", "NVIDIA H100 80GB HBM3", 8, 4096, 266, 326, 0.79),
            ("DPO", "NVIDIA H100 80GB HBM3", 8, 16384, 266, 326, 1.10),
            ("SFT", "NVIDIA H100 80GB HBM3", 8, 4096, 285, 320, 0.296),
        ],
    )
    def test_a_packed_short_row_step_lands_near_its_measured_time(
        self, name, gpu_type, gpus, context, mean, rms, measured
    ):
        # Arrange: about 300-token rows, where host time per model call
        # outweighs the token work.
        manifest = short_row_dataset_manifest(name, context)
        hub_id = "Qwen/Qwen3.5-4B"

        # Act
        (check,) = check_tiers(
            [{**tier("one", gpu_type, gpus, 1.0), "ram_gb": 160 * gpus}],
            manifest,
            catalog_model(manifest.algorithm),
            catalogue_builder(hub_id),
            row_tokens=RowTokens(mean, rms**2),
        )

        # Assert
        assert check.fits
        assert check.step_seconds == pytest.approx(measured, rel=0.15)


class TestTierCheckFailure:
    def test_a_plain_fit_has_no_failure_and_is_not_host_limited(self):
        manifest = TrainingManifest.model_validate(MANIFEST)

        (check,) = check_tiers(
            [tier("l4-2x", "NVIDIA L4", 2, 2.0)], manifest, tiny_model(), builder()
        )

        assert check.fits
        assert check.failure is None
        assert check.host_limited is False

    def test_teacher_generation_over_budget_is_gpu_memory(self):
        manifest = teacher_manifest(teacher="Qwen/Qwen3.8-27B", teacher_width=1)

        (check,) = check_tiers(
            [tier("l4-8x", "NVIDIA L4", 8, 8.0)], manifest, tiny_model(), builder()
        )

        assert not check.fits
        assert check.reason == "teacher generation over budget"
        assert check.failure == FailureKind.GPU_MEMORY
        assert check.host_limited is False

    def test_a_host_share_miss_is_host_memory(self):
        # Arrange: 8 GiB node RAM leaves a 0-byte host share on two GPUs.
        manifest = TrainingManifest.model_validate(MANIFEST)
        tight = {**tier("l4-2x", "NVIDIA L4", 2, 1.0), "ram_gb": 8}

        # Act
        (check,) = check_tiers([tight], manifest, tiny_model(), builder())

        # Assert
        assert not check.fits
        assert check.failure == FailureKind.HOST_MEMORY
        assert check.host_limited is True

    def test_host_short_of_the_layout_free_bytes_prices_one_layout(self, built_layouts):
        # Arrange: 60 GiB nodes leave 6.5 GiB per GPU. The trainer process and
        # the adapter snapshot fit that only without the buffer.
        build, seen = built_layouts
        manifest = TrainingManifest.model_validate(moe_unset_layout())
        tight = {
            **tier("h100-8x", "NVIDIA H100 80GB HBM3", 8, 1.0),
            "gpus_per_node": 8,
            "ram_gb": 60,
        }

        # Act
        (check,) = check_tiers(
            [tight], manifest, catalog_model(manifest.algorithm), build
        )

        # Assert
        assert not check.fits
        assert check.failure == FailureKind.HOST_MEMORY
        assert check.layout == ParallelLayout()
        assert seen == {(8, 1, 1, 1)}

    def test_no_gpu_is_other(self):
        manifest = TrainingManifest.model_validate(MANIFEST)

        (check,) = check_tiers(
            [tier("cpu", None, 0, 0.5)], manifest, tiny_model(), builder()
        )

        assert not check.fits
        assert check.reason == "no GPU"
        assert check.failure == FailureKind.OTHER
        assert check.host_limited is False

    def test_a_build_that_cannot_solve_is_gpu_memory(self):
        # Arrange
        manifest = TrainingManifest.model_validate(MANIFEST)

        message = "no micro batch fits"

        def cannot_solve(*_args):
            raise CannotSolve(message)

        # Act
        (check,) = check_tiers(
            [tier("l4-2x", "NVIDIA L4", 2, 1.0)], manifest, tiny_model(), cannot_solve
        )

        # Assert
        assert not check.fits
        assert check.reason == message
        assert check.config is None
        assert check.failure == FailureKind.GPU_MEMORY

    def test_an_unfit_build_keeps_its_own_failure_kind(self):
        # Arrange: the lightest plan's 2-byte host share is over its 1 byte.
        manifest = TrainingManifest.model_validate(MANIFEST)
        device = lookup_gpu("NVIDIA L4").device_spec()
        plan = plan_for(ParallelLayout(), 1, 100)
        breakdown = plan.breakdown.model_copy(
            update={
                "host": (
                    MemoryComponent(key="optimizer", label="Optimizer", n_bytes=2),
                ),
                "host_usable_bytes": 1,
            }
        )
        unfit = TrainingMemoryUnfitError(
            dataclasses.replace(plan, breakdown=breakdown), device
        )

        def nothing_fits(*_args):
            raise unfit

        # Act
        (check,) = check_tiers(
            [tier("l4-2x", "NVIDIA L4", 2, 1.0)], manifest, tiny_model(), nothing_fits
        )

        # Assert
        assert not check.fits
        assert check.reason == str(unfit)
        assert check.failure == FailureKind.HOST_MEMORY

    def test_a_host_reprice_that_fits_is_host_limited(self, monkeypatch):
        # Arrange: the GPU plan is treated as missing the host share so the
        # search re-prices; 64 GiB is enough for that later plan to fit.
        from agilerl.arena.memory.resolve.run import _host_share_fits as real_fits

        def miss_set_share(plan, host_bytes, require_buffer):
            if host_bytes is not None:
                return False
            return real_fits(plan, host_bytes, require_buffer)

        monkeypatch.setattr(
            "agilerl.arena.memory.resolve.run._host_share_fits", miss_set_share
        )
        manifest = TrainingManifest.model_validate(MANIFEST)

        # Act
        (check,) = check_tiers(
            [tier("l4-2x", "NVIDIA L4", 2, 2.0)], manifest, tiny_model(), builder()
        )

        # Assert
        assert check.fits
        assert check.failure is None
        assert check.host_limited is True

    def test_a_skipped_tier_copies_failure_and_host_limited(self, monkeypatch):
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
        checks = check_tiers([rich, poor], manifest, tiny_model(), builder())

        # Assert
        skipped = next(check for check in checks if check.tier["name"] == "l4-poor")
        assert skipped.reason == "training over budget"
        assert skipped.num_nodes == 8
        assert skipped.config is None
        assert skipped.failure == FailureKind.GPU_MEMORY
        assert skipped.host_limited is False

    def test_a_skipped_tier_copies_a_host_limited_miss(self):
        # Arrange: 8 GiB leaves a 0-byte host share. The poor tier is covered
        # by the rich miss at 8 nodes and is never priced.
        manifest = manifest_with(
            MANIFEST, training_gpus_per_agent=2, rollout_engines_per_agent=1
        )
        rich = {
            **tier("l4-rich", "NVIDIA L4", 2, 1.0),
            "ram_gb": 8,
            "node_counts": [1, 2, 8],
        }
        poor = {
            **tier("l4-poor", "NVIDIA L4", 2, 2.0),
            "ram_gb": 8,
            "node_counts": [4, 8],
        }

        # Act
        checks = check_tiers([rich, poor], manifest, tiny_model(), builder())

        # Assert
        skipped = next(check for check in checks if check.tier["name"] == "l4-poor")
        assert skipped.num_nodes == 8
        assert skipped.config is None
        assert skipped.failure == FailureKind.HOST_MEMORY
        assert skipped.host_limited is True

    def test_no_servable_engine_width_is_gpu_memory(self, monkeypatch):
        monkeypatch.setattr(
            "agilerl.arena.memory.resolve.run.servable_engine_widths",
            lambda *args, **kwargs: [],
        )
        manifest = TrainingManifest.model_validate(unset_split(unset_width(MANIFEST)))

        (check,) = check_tiers(
            [tier("l4-2x", "NVIDIA L4", 2, 1.0)], manifest, tiny_model(), builder()
        )

        assert not check.fits
        assert "a trainer and a rollout engine do not both fit" in check.reason
        assert check.failure == FailureKind.GPU_MEMORY

    def test_no_room_beside_an_engine_is_other(self):
        # One GPU cannot hold a trainer beside an engine. check_tiers then
        # sizes kept counts and reports a GPU-count miss; the split's kind is
        # other before that.
        manifest = TrainingManifest.model_validate(unset_split(MANIFEST))
        device = lookup_gpu("NVIDIA L4").device_spec()

        with pytest.raises(SplitSearchError) as raised:
            split_manifest(manifest, 1, device)

        assert raised.value.kind == FailureKind.OTHER


class TestSearchSteps:
    def test_hsdp_at_the_node_size_is_one_step(self):
        # Arrange: up to 15 trainers on two 8-GPU nodes.
        manifest = TrainingManifest.model_validate(unset_split(MANIFEST))
        tiers = [{**tier("l4-8x", "NVIDIA L4", 8, 1.0), "node_counts": [1, 2]}]

        # Act
        steps = resources_module._search_steps(manifest, tiers)

        # Assert
        assert steps == [
            ParallelLayout(),
            NodeShardGroups(ParallelLayout()),
            ParallelLayout(cp=2),
            ParallelLayout(tp=2),
        ]

    def test_an_expert_group_over_nodes_comes_first_with_node_shard_groups(self):
        # Arrange: Qwen3.6-35B-A3B SFT on 8 trainers, two 4-GPU nodes.
        doc = with_model(SFT_MANIFEST, "Qwen/Qwen3.6-35B-A3B")
        doc["training"]["training_gpus_per_agent"] = 8
        manifest = TrainingManifest.model_validate(doc)
        tiers = [{**tier("l4-4x", "NVIDIA L4", 4, 1.0), "node_counts": [1, 2]}]

        # Act
        steps = resources_module._search_steps(manifest, tiers)

        # Assert
        wide = steps.index(ParallelLayout(ep=8))
        assert steps[wide - 1] == NodeShardGroups(ParallelLayout(ep=8))
        assert NodeShardGroups(ParallelLayout(ep=4)) not in steps


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


DEEPSEEK = "deepseek-ai/DeepSeek-V4-Flash-0731"


def deepseek_kept_counts() -> TrainingManifest:
    """DeepSeek-V4-Flash GRPO with expert LoRA: 16 trainers and one engine at TP 4, layout unset."""
    doc = with_model(MANIFEST, DEEPSEEK)
    doc["algorithm"].update(batch_size=32, group_size=4, max_output_tokens=128)
    doc["algorithm"]["vllm_config"]["tensor_parallel_size"] = 4
    doc["network"].update(
        max_context_length=1024,
        lora_config={
            "lora_r": 16,
            "lora_alpha": 32,
            "target_parameters": ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
        },
    )
    doc["training"]["training_gpus_per_agent"] = 16
    return TrainingManifest.model_validate(doc)


def catalogue_builder(hub_id: str):
    """``build_config`` for a catalogued model, as the memory CLI builds it."""
    model_config = SUPPORTED_MODEL_INFO[hub_id].config
    n_params = SUPPORTED_MODEL_INFO[hub_id].num_params
    return lambda laid_out, device, width, engines, adapter_sync, nccl_reshard: (
        run_config_from_manifest(
            laid_out,
            device,
            model_config,
            n_params=n_params,
            tensor_parallel_size=width,
            rollout_engines=engines,
            adapter_sync=adapter_sync,
            nccl_reshard=nccl_reshard,
        )
    )


class TestCheckTiersCrossNodeExperts:
    def test_a_lock_gathering_experts_across_nodes_gives_way_to_whole_experts(self):
        # Arrange: 16 trainers on two of three 8-H100 nodes. EP 4 is the
        # first layout that fits and gathers expert pieces from the other
        # node; EP 16 holds every expert once, with a shard group per node.
        manifest = deepseek_kept_counts()
        tiers = [
            {
                **tier("h100-8x", "NVIDIA H100 80GB HBM3", 8, 30.0),
                "num_cpus": 200,
                "ram_gb": 1800,
                "node_counts": [3],
            }
        ]

        # Act
        (check,) = check_tiers(
            tiers,
            manifest,
            catalog_model(manifest.algorithm),
            catalogue_builder(DEEPSEEK),
        )

        # Assert
        assert check.fits
        assert check.num_nodes == 3
        assert check.layout == ParallelLayout(ep=16, shard_group_size=8)
        assert check.cross_node_expert_bytes > 0

    def test_mixed_node_sizes_try_expert_parallel_before_context_parallel(
        self, monkeypatch
    ):
        # Arrange: a 4-GPU tier beside an 8-GPU tier. EP 8 sits on the 8-GPU
        # node and would span the 4-GPU node; CP 2 also fits.
        manifest = deepseek_kept_counts()
        tiers = [node_tier("h100-4x", 4, 1.0), node_tier("h100-8x", 8, 2.0)]
        tried: list[tuple[str, ParallelLayout | None]] = []

        def fits(name: str, layout: ParallelLayout | None) -> bool:
            return name == "h100-8x" and layout == ParallelLayout(ep=8)

        monkeypatch.setattr(resources_module, "_price", record_price(tried, fits))

        # Act
        checks = check_tiers(
            tiers,
            manifest,
            catalog_model(manifest.algorithm),
            catalogue_builder(DEEPSEEK),
        )

        # Assert
        steps = resources_module._search_steps(manifest, tiers)
        assert steps.index(ParallelLayout(ep=8)) < steps.index(ParallelLayout(cp=2))
        chosen = next(check for check in checks if check.fits)
        assert chosen.tier["name"] == "h100-8x"
        assert chosen.layout == ParallelLayout(ep=8)
        assert ParallelLayout(cp=2) not in [layout for _, layout in tried]

    def test_a_smaller_node_keeps_context_parallel_ahead_of_a_spanning_expert_group(
        self, monkeypatch
    ):
        # Arrange: EP 8 spans the 4-GPU node. CP 2 fits there; EP 8 does not
        # fit on the 8-GPU node.
        manifest = deepseek_kept_counts()
        tiers = [node_tier("h100-4x", 4, 1.0), node_tier("h100-8x", 8, 2.0)]
        tried: list[tuple[str, ParallelLayout | None]] = []

        def fits(name: str, layout: ParallelLayout | None) -> bool:
            return name == "h100-4x" and layout == ParallelLayout(cp=2)

        monkeypatch.setattr(resources_module, "_price", record_price(tried, fits))

        # Act
        checks = check_tiers(
            tiers,
            manifest,
            catalog_model(manifest.algorithm),
            catalogue_builder(DEEPSEEK),
        )

        # Assert
        chosen = next(check for check in checks if check.fits)
        assert chosen.tier["name"] == "h100-4x"
        assert chosen.layout == ParallelLayout(cp=2)
        on_node = tried.index(("h100-4x", ParallelLayout(cp=2)))
        spanning = ("h100-4x", ParallelLayout(ep=8))
        if spanning in tried:
            assert tried.index(spanning) > on_node

    def test_a_spanning_expert_group_is_priced_after_on_node_layouts_miss(
        self, monkeypatch
    ):
        # Arrange: one 8-GPU node. EP 16 spans it. CP 2 misses, then EP 16 fits.
        manifest = deepseek_kept_counts()
        tiers = [node_tier("h100-8x", 8, 2.0)]
        tried: list[tuple[str, ParallelLayout | None]] = []

        def fits(name: str, layout: ParallelLayout | None) -> bool:
            return name == "h100-8x" and layout == ParallelLayout(ep=16)

        monkeypatch.setattr(resources_module, "_price", record_price(tried, fits))

        # Act
        (check,) = check_tiers(
            tiers,
            manifest,
            catalog_model(manifest.algorithm),
            catalogue_builder(DEEPSEEK),
        )

        # Assert
        assert check.layout == ParallelLayout(ep=16)
        assert tried.index(("h100-8x", ParallelLayout(cp=2))) < tried.index(
            ("h100-8x", ParallelLayout(ep=16))
        )


def node_tier(name: str, node_gpus: int, price: float) -> dict:
    """One H100 tier of ``node_gpus`` whose node counts hold 16 trainers."""
    return {
        **tier(name, "NVIDIA H100 80GB HBM3", node_gpus, price),
        "num_cpus": 200,
        "ram_gb": 1800,
        "node_counts": [16 // node_gpus],
    }


def record_price(tried, fits):
    """``_price`` that records ``(tier, layout)`` and fits where ``fits`` says so."""

    def price(
        tier,
        _manifest,
        _model,
        _build_config,
        layout,
        num_nodes,
        _cache,
        _lengths,
        _row_tokens,
    ):
        tried.append((tier["name"], layout))
        ok = fits(tier["name"], layout)
        return resources_module.TierCheck(
            tier,
            None if ok else "over budget",
            num_nodes=num_nodes,
            layout=layout if isinstance(layout, ParallelLayout) else None,
            failure=None if ok else FailureKind.GPU_MEMORY,
        )

    return price

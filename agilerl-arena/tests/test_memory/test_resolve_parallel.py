# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Least parallel training layout that fits."""

from __future__ import annotations

import copy

import pytest

from agilerl.arena.memory.breakdown import MemoryComponent, PhaseBreakdown
from agilerl.arena.memory.manifest import lookup_gpu, training_memory_from_manifest
from agilerl.arena.memory.resolve.batch import TrainerSplitError, catalog_model
from agilerl.arena.memory.resolve.fsdp import (
    TrainingMemoryPlan,
    TrainingMemoryUnfitError,
)
from agilerl.arena.memory.resolve.parallel import (
    ParallelLayout,
    first_fitting_plan,
    manifest_at_layout,
    parallel_layouts,
    trains_expert_parallel,
)
from agilerl.arena.memory.specs import DeviceSpec, ModelArch, TrainingSettings
from agilerl.arena.models.fsdp import FSDPConfig, record_parallel_pins
from agilerl.arena.models.manifest import TrainingManifest
from tests.test_memory.test_cli import MANIFEST
from tests.test_memory.test_estimator import MOE_MODEL
from tests.test_memory.test_formulas import NEMOTRON_H, QWEN_05B


def open_fsdp(**fields: object) -> FSDPConfig:
    """An FSDP config whose parallel fields the search may change."""
    return record_parallel_pins(FSDPConfig(**fields), fields)


def plan_for(layout: ParallelLayout, total: int, usable: int) -> TrainingMemoryPlan:
    """A plan whose peak is ``total`` bytes on a ``usable`` card."""
    fsdp = FSDPConfig(
        tp=layout.tp, ep=layout.ep, shard_group_size=layout.shard_group_size
    )
    settings = TrainingSettings(n_training_gpus=4, fsdp=fsdp)
    return TrainingMemoryPlan(
        gradient_checkpointing=False,
        checkpoint_every_n_blocks=1,
        micro_batch_size_per_gpu=1,
        micro_batch_tokens_per_gpu=None,
        fsdp=fsdp,
        routed_expert_chunk_mib=None,
        cost=1.0,
        tokens=1.0,
        settings=settings,
        breakdown=PhaseBreakdown(
            phase="training",
            components=(
                MemoryComponent(key="weights", label="Weights", n_bytes=total),
            ),
            device_total_bytes=usable,
            device_usable_bytes=usable,
        ),
    )


class TestParallelLayouts:
    def test_dense_model_stays_at_one_degree(self):
        layouts = parallel_layouts(QWEN_05B, 8, 8, open_fsdp(), False)

        assert layouts == [ParallelLayout()]

    def test_hsdp_is_second_when_trainers_span_two_nodes(self):
        layouts = parallel_layouts(QWEN_05B, 8, 4, open_fsdp(), False)

        assert layouts == [ParallelLayout(), ParallelLayout(shard_group_size=4)]

    def test_a_moe_tries_expert_parallel_then_tensor_within_it(self):
        arch = MOE_MODEL.arch.model_copy(update={"tensor_parallel_plan": True})

        layouts = parallel_layouts(arch, 4, 4, open_fsdp(), True)

        assert layouts == [
            ParallelLayout(),
            ParallelLayout(ep=2),
            ParallelLayout(ep=4),
            ParallelLayout(tp=2, ep=2),
        ]

    def test_a_dense_model_with_a_tensor_plan_tries_tensor_parallel(self):
        arch = QWEN_05B.model_copy(update={"tensor_parallel_plan": True})

        layouts = parallel_layouts(arch, 4, 4, open_fsdp(), False)

        # 14 query heads split over 2 ranks, not 4.
        assert layouts == [ParallelLayout(), ParallelLayout(tp=2)]

    @pytest.mark.parametrize(
        ("update", "tp_degrees"),
        [
            ({}, [1, 2, 4, 8]),
            ({"mamba_n_heads": 36}, [1, 2, 4]),
            ({"mamba_n_groups": 6}, [1, 2]),
            ({"tied_embeddings": False, "vocab_size": 2 * 65537}, [1, 2]),
        ],
        ids=["heads-and-kv-groups", "mamba-heads", "mamba-groups", "untied-vocab"],
    )
    def test_tensor_degrees_are_ones_the_trainer_can_split(self, update, tp_degrees):
        # Arrange: 40 query heads in groups of 5 split over 2, 4 and 8 ranks.
        arch = ModelArch.from_hf_config(NEMOTRON_H).model_copy(
            update={"tensor_parallel_plan": True, **update}
        )

        # Act
        layouts = parallel_layouts(arch, 8, 8, open_fsdp(), False, ranks_fixed=False)

        # Assert
        assert [layout.tp for layout in layouts] == tp_degrees

    def test_a_moe_the_trainer_cannot_shard_gets_neither_axis(self):
        arch = MOE_MODEL.arch.model_copy(update={"tensor_parallel_plan": True})

        layouts = parallel_layouts(arch, 4, 4, open_fsdp(), False)

        assert layouts == [ParallelLayout()]

    def test_a_pinned_tensor_degree_on_a_moe_is_kept(self):
        arch = MOE_MODEL.arch.model_copy(update={"tensor_parallel_plan": True})

        layouts = parallel_layouts(arch, 4, 4, open_fsdp(tp=2), True)

        assert layouts == [
            ParallelLayout(tp=2),
            ParallelLayout(tp=2, ep=2),
            ParallelLayout(tp=2, ep=4),
        ]

    def test_a_family_without_a_tensor_plan_never_gets_tensor_parallel(self):
        arch = MOE_MODEL.arch.model_copy(update={"tensor_parallel_plan": False})

        layouts = parallel_layouts(arch, 4, 4, open_fsdp(), True)

        assert arch.is_moe
        assert all(layout.tp == 1 for layout in layouts)
        assert ParallelLayout(ep=2) in layouts

    def test_degrees_stay_within_one_node(self):
        arch = MOE_MODEL.arch.model_copy(
            update={"tensor_parallel_plan": True, "n_experts": 16}
        )

        layouts = parallel_layouts(arch, 32, 4, open_fsdp(), True, ranks_fixed=False)

        assert max(layout.tp * layout.ep for layout in layouts) == 4

    def test_expert_parallel_degrees_divide_the_experts(self):
        arch = MOE_MODEL.arch.model_copy(update={"n_experts": 6})

        layouts = parallel_layouts(arch, 8, 8, open_fsdp(), True, ranks_fixed=False)

        assert sorted({layout.ep for layout in layouts}) == [1, 2, 3, 6]

    def test_an_explicit_degree_is_kept(self):
        layouts = parallel_layouts(MOE_MODEL.arch, 4, 4, open_fsdp(ep=2), True)

        assert layouts
        assert {layout.ep for layout in layouts} == {2}

    def test_a_pinned_shard_group_skips_degrees_it_cannot_hold(self):
        arch = MOE_MODEL.arch.model_copy(update={"tensor_parallel_plan": False})

        layouts = parallel_layouts(arch, 8, 8, open_fsdp(shard_group_size=2), True)

        assert layouts == [
            ParallelLayout(shard_group_size=2),
            ParallelLayout(ep=2, shard_group_size=2),
        ]

    def test_a_pinned_degree_that_does_not_divide_the_trainers_gives_none(self):
        layouts = parallel_layouts(MOE_MODEL.arch, 6, 8, open_fsdp(ep=4), True)

        assert layouts == []


class TestFirstFittingPlan:
    def test_stops_when_expert_parallel_is_what_makes_it_fit(self):
        # Arrange
        device = DeviceSpec(total_bytes=100, name="test")
        arch = MOE_MODEL.arch.model_copy(update={"tensor_parallel_plan": True})
        layouts = parallel_layouts(arch, 4, 4, open_fsdp(), True)
        tried: list[int] = []

        def price(layout: ParallelLayout) -> TrainingMemoryPlan:
            tried.append(layout.ep if layout.tp == 1 else layout.tp)
            if layout.ep == 1 and layout.tp == 1:
                raise TrainingMemoryUnfitError(plan_for(layout, 1_000, 100), device)
            return plan_for(layout, 1, 100)

        # Act
        plan = first_fitting_plan(layouts, price)

        # Assert
        assert plan.fsdp is not None
        assert plan.fsdp.ep == 2
        assert plan.fsdp.tp == 1
        assert tried == [1, 2]

    def test_raises_why_no_layout_splits_across_the_ranks(self):
        # Arrange
        def price(_layout: ParallelLayout) -> TrainingMemoryPlan:
            msg = "Batch size (6) must be divisible by the data-parallel size (4)."
            raise TrainerSplitError(msg)

        # Act / Assert
        with pytest.raises(TrainerSplitError, match=r"Batch size \(6\)"):
            first_fitting_plan([ParallelLayout(), ParallelLayout(ep=2)], price)


class TestTrainsExpertParallel:
    @pytest.mark.parametrize(
        ("hub_id", "expected"),
        [
            ("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16", True),
            ("openai/gpt-oss-20b", False),
            ("Qwen/Qwen2.5-0.5B-Instruct", False),
            ("local/uncatalogued-model", False),
        ],
    )
    def test_follows_the_catalogue(self, hub_id, expected):
        doc = copy.deepcopy(MANIFEST)
        doc["network"]["pretrained_model_name_or_path"] = hub_id
        manifest = TrainingManifest.model_validate(doc)

        assert trains_expert_parallel(manifest) is expected

    def test_an_algorithm_without_a_model_is_false(self):
        manifest = TrainingManifest.model_validate(
            {
                "algorithm": {"name": "DQN"},
                "environment": {"name": "LunarLander-v3", "num_envs": 8},
                "training": {"max_steps": 1000},
            }
        )

        assert trains_expert_parallel(manifest) is False


class TestManifestAtLayout:
    def test_pins_the_layout_so_the_search_tries_it_alone(self):
        manifest = TrainingManifest.model_validate(MANIFEST)
        assert manifest.algorithm.fsdp is None

        placed = manifest_at_layout(manifest, ParallelLayout(ep=2))

        fsdp = placed.algorithm.fsdp
        assert (fsdp.tp, fsdp.ep, fsdp.shard_group_size) == (1, 2, None)
        assert parallel_layouts(MOE_MODEL.arch, 4, 4, fsdp, True) == [
            ParallelLayout(ep=2)
        ]
        assert manifest.algorithm.fsdp is None

    def test_keeps_the_manifests_other_fsdp_fields(self):
        doc = copy.deepcopy(MANIFEST)
        doc["algorithm"]["fsdp"] = {"activation_offload": True}
        manifest = TrainingManifest.model_validate(doc)

        placed = manifest_at_layout(manifest, ParallelLayout(shard_group_size=4))

        assert placed.algorithm.fsdp.activation_offload is True
        assert placed.algorithm.fsdp.shard_group_size == 4

    def test_an_algorithm_without_a_training_layout_raises(self):
        manifest = TrainingManifest.model_validate(
            {
                "algorithm": {"name": "DQN"},
                "environment": {"name": "LunarLander-v3", "num_envs": 8},
                "training": {"max_steps": 1000},
            }
        )

        with pytest.raises(TypeError, match="DQN has no training layout"):
            manifest_at_layout(manifest, ParallelLayout())


class TestTrainingMemoryFromManifest:
    def test_a_model_that_fits_trains_ddp_at_degree_one(self):
        # Arrange
        doc = copy.deepcopy(MANIFEST)
        doc["training"]["training_gpus_per_agent"] = 2
        manifest = TrainingManifest.model_validate(doc)
        gpu = lookup_gpu("NVIDIA L4")
        assert gpu is not None
        model = catalog_model(manifest.algorithm)
        assert model is not None

        # Act
        plan = training_memory_from_manifest(manifest, model, gpu.device_spec())

        # Assert
        assert plan.fsdp is None
        assert plan.breakdown.fits_with_buffer

    def test_a_layout_that_cannot_divide_the_trainers_raises(self):
        doc = copy.deepcopy(MANIFEST)
        doc["algorithm"]["fsdp"] = {"shard_group_size": 4}
        doc["training"]["training_gpus_per_agent"] = 2
        manifest = TrainingManifest.model_validate(doc)
        gpu = lookup_gpu("NVIDIA L4")
        assert gpu is not None
        model = catalog_model(manifest.algorithm)
        assert model is not None

        with pytest.raises(TrainerSplitError, match="No training layout can run"):
            training_memory_from_manifest(manifest, model, gpu.device_spec())

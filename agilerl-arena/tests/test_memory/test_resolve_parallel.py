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
    layout_runs,
    manifest_at_layout,
    off_node_reason,
    parallel_layouts,
    trains_context_parallel,
    trains_expert_parallel,
)
from agilerl.arena.memory.specs import DeviceSpec, ModelArch, TrainingSettings
from agilerl.arena.models.fsdp import FSDPConfig, record_parallel_pins
from agilerl.arena.models.manifest import TrainingManifest
from agilerl.arena.models.model_info import SUPPORTED_MODEL_INFO
from tests.test_memory.test_cli import MANIFEST, SFT_MANIFEST
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
        moe_collectives_saveable=False,
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
        layouts = parallel_layouts(QWEN_05B, 8, 8, open_fsdp(), False, False)

        assert layouts == [ParallelLayout()]

    def test_hsdp_is_second_when_trainers_span_two_nodes(self):
        layouts = parallel_layouts(QWEN_05B, 8, 4, open_fsdp(), False, False)

        assert layouts == [ParallelLayout(), ParallelLayout(shard_group_size=4)]

    def test_a_moe_tries_expert_parallel_then_tensor_within_it(self):
        arch = MOE_MODEL.arch.model_copy(update={"tensor_parallel_plan": True})

        layouts = parallel_layouts(arch, 4, 4, open_fsdp(), True, False)

        assert layouts == [
            ParallelLayout(),
            ParallelLayout(ep=2),
            ParallelLayout(ep=4),
            ParallelLayout(tp=2, ep=2),
            ParallelLayout(tp=2, ep=4),
            ParallelLayout(tp=4, ep=4),
        ]

    def test_one_raised_axis_goes_expert_then_context_parallel(self):
        arch = MOE_MODEL.arch.model_copy(update={"tensor_parallel_plan": False})

        layouts = parallel_layouts(arch, 4, 4, open_fsdp(), True, True)

        assert layouts == [
            ParallelLayout(),
            ParallelLayout(ep=2),
            ParallelLayout(ep=4),
            ParallelLayout(cp=2),
            ParallelLayout(cp=4),
            ParallelLayout(ep=2, cp=2),
            ParallelLayout(ep=4, cp=2),
            ParallelLayout(ep=2, cp=4),
            ParallelLayout(ep=4, cp=4),
        ]

    def test_context_parallel_comes_before_tensor_parallel(self):
        arch = QWEN_05B.model_copy(update={"tensor_parallel_plan": True})

        layouts = parallel_layouts(arch, 4, 4, open_fsdp(), False, True)

        # 7 query heads per tensor rank do not split over 2 context ranks.
        assert layouts == [ParallelLayout(), ParallelLayout(cp=2), ParallelLayout(tp=2)]

    def test_a_run_the_trainer_cannot_split_gets_no_context_parallel(self):
        layouts = parallel_layouts(QWEN_05B, 4, 4, open_fsdp(), False, False)

        assert layouts == [ParallelLayout()]

    @pytest.mark.parametrize(
        ("update", "cp_degrees"),
        [
            ({}, [1, 2, 4, 8]),
            ({"mamba_n_heads": 36}, [1, 2, 4]),
            ({"mamba_n_groups": 6}, [1, 2]),
            ({"mamba_gated_delta": True, "mamba_n_heads": 36}, [1, 2, 4, 8]),
        ],
        ids=["heads-and-kv-heads", "mamba-heads", "mamba-groups", "gated-delta"],
    )
    def test_context_degrees_are_ones_the_trainer_can_split(self, update, cp_degrees):
        # Arrange: 40 query heads and 8 key-value heads split over 2, 4 and 8
        # ranks, not 5.
        arch = ModelArch.from_hf_config(NEMOTRON_H).model_copy(update=update)

        # Act
        layouts = parallel_layouts(
            arch, 8, 8, open_fsdp(), False, True, ranks_fixed=False
        )

        # Assert
        assert [layout.cp for layout in layouts] == cp_degrees

    def test_context_degrees_split_each_tensor_ranks_heads(self):
        # Arrange: 24 query heads split over 4 context ranks, but the 6 on one
        # of 4 tensor ranks do not.
        arch = ModelArch.from_hf_config(NEMOTRON_H).model_copy(
            update={"n_heads": 24, "tensor_parallel_plan": True}
        )

        # Act
        layouts = parallel_layouts(
            arch, 16, 16, open_fsdp(tp=4), False, True, ranks_fixed=False
        )

        # Assert
        assert layouts == [ParallelLayout(tp=4), ParallelLayout(tp=4, cp=2)]

    def test_a_dense_model_with_a_tensor_plan_tries_tensor_parallel(self):
        arch = QWEN_05B.model_copy(update={"tensor_parallel_plan": True})

        layouts = parallel_layouts(arch, 4, 4, open_fsdp(), False, False)

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
        layouts = parallel_layouts(
            arch, 8, 8, open_fsdp(), False, False, ranks_fixed=False
        )

        # Assert
        assert [layout.tp for layout in layouts] == tp_degrees

    def test_a_moe_the_trainer_cannot_shard_gets_neither_axis(self):
        arch = MOE_MODEL.arch.model_copy(update={"tensor_parallel_plan": True})

        layouts = parallel_layouts(arch, 4, 4, open_fsdp(), False, False)

        assert layouts == [ParallelLayout()]

    def test_a_pinned_tensor_degree_on_a_moe_is_kept(self):
        arch = MOE_MODEL.arch.model_copy(update={"tensor_parallel_plan": True})

        layouts = parallel_layouts(arch, 4, 4, open_fsdp(tp=2), True, False)

        assert layouts == [
            ParallelLayout(tp=2),
            ParallelLayout(tp=2, ep=2),
            ParallelLayout(tp=2, ep=4),
        ]

    def test_a_family_without_a_tensor_plan_never_gets_tensor_parallel(self):
        arch = MOE_MODEL.arch.model_copy(update={"tensor_parallel_plan": False})

        layouts = parallel_layouts(arch, 4, 4, open_fsdp(), True, False)

        assert arch.is_moe
        assert all(layout.tp == 1 for layout in layouts)
        assert ParallelLayout(ep=2) in layouts

    def test_expert_and_tensor_context_groups_each_stay_within_one_node(self):
        arch = MOE_MODEL.arch.model_copy(
            update={"tensor_parallel_plan": True, "n_experts": 16}
        )

        layouts = parallel_layouts(
            arch, 32, 4, open_fsdp(), True, True, ranks_fixed=False
        )

        assert max(layout.ep for layout in layouts) == 4
        assert max(layout.tp * layout.cp for layout in layouts) == 4
        assert ParallelLayout(tp=2, ep=4, cp=2) in layouts

    @pytest.mark.parametrize(
        ("hub_id", "n_gpus", "ep_degrees"),
        [
            ("ibm-granite/granite-3.1-3b-a800m-instruct", 10, [1, 2]),
            ("zai-org/GLM-5.3-Flash", 12, [1, 2, 4]),
        ],
        ids=["granite-10-trainers", "glm-12-trainers"],
    )
    def test_expert_groups_divide_the_node_when_trainers_span_nodes(
        self, hub_id, n_gpus, ep_degrees
    ):
        # Arrange: granite's 40 experts split over 5 ranks and GLM's 288 over
        # 3 and 6, but none of those divide an 8-GPU node.
        arch = ModelArch.from_hf_config(SUPPORTED_MODEL_INFO[hub_id].config)

        # Act
        layouts = parallel_layouts(arch, n_gpus, 8, open_fsdp(), True, False)

        # Assert
        assert sorted({layout.ep for layout in layouts}) == ep_degrees
        assert all(8 % (layout.tp * layout.cp) == 0 for layout in layouts)

    def test_an_open_count_keeps_groups_up_to_the_node(self):
        # Arrange: up to 12 trainers may fit 3 or 6 expert ranks on one node;
        # the split drops the counts where they straddle two.
        arch = ModelArch.from_hf_config(
            SUPPORTED_MODEL_INFO["zai-org/GLM-5.3-Flash"].config
        )

        # Act
        layouts = parallel_layouts(
            arch, 12, 8, open_fsdp(), True, False, ranks_fixed=False
        )

        # Assert
        assert sorted({layout.ep for layout in layouts}) == [1, 2, 3, 4, 6, 8]
        assert max(layout.tp * layout.cp for layout in layouts) <= 8

    def test_a_pinned_degree_that_crosses_nodes_is_kept(self):
        # Arrange: ep=5 is pinned; the searched tp=2 sits on one node.
        arch = ModelArch.from_hf_config(
            SUPPORTED_MODEL_INFO["ibm-granite/granite-3.1-3b-a800m-instruct"].config
        )

        # Act
        layouts = parallel_layouts(arch, 10, 8, open_fsdp(ep=5), True, False)

        # Assert
        assert layouts == [ParallelLayout(ep=5), ParallelLayout(tp=2, ep=5)]

    def test_a_set_cpu_offload_gets_no_expert_parallel(self):
        arch = MOE_MODEL.arch.model_copy(update={"tensor_parallel_plan": False})

        layouts = parallel_layouts(arch, 4, 4, open_fsdp(cpu_offload=True), True, True)

        assert layouts == [
            ParallelLayout(),
            ParallelLayout(cp=2),
            ParallelLayout(cp=4),
        ]

    def test_expert_parallel_degrees_divide_the_experts(self):
        arch = MOE_MODEL.arch.model_copy(update={"n_experts": 6})

        layouts = parallel_layouts(
            arch, 8, 8, open_fsdp(), True, False, ranks_fixed=False
        )

        assert sorted({layout.ep for layout in layouts}) == [1, 2, 3, 6]

    def test_an_explicit_degree_is_kept(self):
        layouts = parallel_layouts(MOE_MODEL.arch, 4, 4, open_fsdp(ep=2), True, False)

        assert layouts
        assert {layout.ep for layout in layouts} == {2}

    def test_a_pinned_shard_group_skips_degrees_it_cannot_hold(self):
        arch = MOE_MODEL.arch.model_copy(update={"tensor_parallel_plan": False})

        layouts = parallel_layouts(
            arch, 8, 8, open_fsdp(shard_group_size=2), True, False
        )

        assert layouts == [
            ParallelLayout(shard_group_size=2),
            ParallelLayout(ep=2, shard_group_size=2),
        ]

    def test_a_pinned_degree_that_does_not_divide_the_trainers_gives_none(self):
        layouts = parallel_layouts(MOE_MODEL.arch, 6, 8, open_fsdp(ep=4), True, False)

        assert layouts == []


class TestLayoutRuns:
    @pytest.mark.parametrize(
        ("n_gpus", "layout", "ranks_fixed", "expected"),
        [
            (8, ParallelLayout(tp=2, cp=2), True, True),
            (6, ParallelLayout(tp=2, cp=2), True, False),
            (8, ParallelLayout(tp=2, cp=2, shard_group_size=2), True, False),
            (3, ParallelLayout(tp=2, cp=2), False, False),
            (4, ParallelLayout(tp=2, cp=2), False, True),
        ],
        ids=[
            "fixed-fits",
            "fixed-indivisible",
            "shard-group",
            "open-over",
            "open-fits",
        ],
    )
    def test_a_tensor_context_group_divides_the_ranks(
        self, n_gpus, layout, ranks_fixed, expected
    ):
        assert layout_runs(n_gpus, layout, ranks_fixed) is expected

    @pytest.mark.parametrize(
        ("n_gpus", "layout", "expected"),
        [
            (6, ParallelLayout(tp=2, ep=3), True),
            (4, ParallelLayout(tp=2, ep=3), False),
            (8, ParallelLayout(ep=4, shard_group_size=8), True),
            (6, ParallelLayout(ep=4, shard_group_size=8), False),
        ],
        ids=["groups-fit", "groups-over", "shard-group-fits", "shard-group-over"],
    )
    def test_an_open_count_needs_one_count_every_group_divides(
        self, n_gpus, layout, expected
    ):
        # 2 tensor and 3 expert ranks need a multiple of 6 trainers.
        assert layout_runs(n_gpus, layout, False) is expected


class TestOffNodeReason:
    @pytest.mark.parametrize(
        ("n_gpus", "layout"),
        [
            (4, ParallelLayout(ep=4)),
            (16, ParallelLayout(ep=8)),
            (16, ParallelLayout(tp=2, cp=4)),
            (6, ParallelLayout(ep=3)),
        ],
        ids=["one-node", "ep-divides-node", "tp-cp-divides-node", "within-node"],
    )
    def test_groups_on_one_node_give_none(self, n_gpus, layout):
        assert off_node_reason(layout, ParallelLayout(), n_gpus, 8) is None

    @pytest.mark.parametrize(
        ("n_gpus", "node", "layout", "reason"),
        [
            (
                8,
                4,
                ParallelLayout(ep=8),
                "expert parallel group of 8 GPUs is wider than a 4-GPU node",
            ),
            (
                8,
                4,
                ParallelLayout(tp=2, cp=4),
                "tensor x context parallel group of 8 GPUs is wider than a 4-GPU node",
            ),
            (
                10,
                8,
                ParallelLayout(ep=5),
                "expert parallel groups of 5 GPUs straddle 8-GPU nodes",
            ),
            (
                12,
                8,
                ParallelLayout(tp=3),
                "tensor x context parallel groups of 3 GPUs straddle 8-GPU nodes",
            ),
        ],
        ids=["ep-wider", "tp-cp-wider", "ep-straddles", "tp-straddles"],
    )
    def test_a_group_off_one_node_says_why(self, n_gpus, node, layout, reason):
        assert off_node_reason(layout, ParallelLayout(), n_gpus, node) == reason

    @pytest.mark.parametrize(
        ("n_gpus", "layout"),
        [(10, ParallelLayout(ep=5)), (16, ParallelLayout(tp=16))],
        ids=["straddles", "wider"],
    )
    def test_a_group_at_the_pinned_degrees_runs_as_written(self, n_gpus, layout):
        assert off_node_reason(layout, layout, n_gpus, 8) is None

    def test_a_group_raised_over_a_pinned_degree_is_checked(self):
        # Arrange: tp=3 is pinned; the search raised cp to 2, a group of 6.
        layout = ParallelLayout(tp=3, cp=2)

        # Act
        reason = off_node_reason(layout, ParallelLayout(tp=3), 12, 8)

        # Assert
        assert (
            reason == "tensor x context parallel groups of 6 GPUs straddle 8-GPU nodes"
        )

    @pytest.mark.parametrize(
        ("layout", "reason"),
        [
            (ParallelLayout(ep=5), None),
            (
                ParallelLayout(ep=16),
                "expert parallel group of 16 GPUs is wider than a 8-GPU node",
            ),
        ],
        ids=["straddles-at-most", "wider"],
    )
    def test_an_open_count_checks_only_the_width(self, layout, reason):
        assert off_node_reason(layout, ParallelLayout(), 10, 8, False) == reason


class TestFirstFittingPlan:
    def test_stops_when_expert_parallel_is_what_makes_it_fit(self):
        # Arrange
        device = DeviceSpec(total_bytes=100, name="test")
        arch = MOE_MODEL.arch.model_copy(update={"tensor_parallel_plan": True})
        layouts = parallel_layouts(arch, 4, 4, open_fsdp(), True, False)
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
            ("openai/gpt-oss-20b", True),
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


ROLLOUT_ENV = {
    "env_type": "rollout",
    "name": "AgentWorldModel",
    "env_url": "http://env.invalid:8000",
    "max_turns": 1,
}


class TestTrainsContextParallel:
    @pytest.mark.parametrize(
        ("hub_id", "environment", "tower_modules", "expected"),
        [
            ("Qwen/Qwen2.5-0.5B-Instruct", None, None, True),
            ("Qwen/Qwen3.5-0.8B", None, None, True),
            ("Qwen/Qwen3.5-0.8B", {**ROLLOUT_ENV, "modality": "text"}, None, True),
            ("Qwen/Qwen3.5-0.8B", ROLLOUT_ENV, None, False),
            ("Qwen/Qwen3.5-0.8B", None, ["qkv"], False),
            ("local/uncatalogued-model", None, None, False),
        ],
        ids=[
            "text-model",
            "vl-on-a-dataset",
            "vl-on-a-text-env",
            "vl-on-an-env-of-unknown-modality",
            "vl-with-tower-lora",
            "uncatalogued",
        ],
    )
    def test_a_vision_language_model_needs_its_towers_stripped(
        self, hub_id, environment, tower_modules, expected
    ):
        # Arrange
        doc = copy.deepcopy(MANIFEST)
        doc["network"]["pretrained_model_name_or_path"] = hub_id
        if environment is not None:
            doc["environment"] = environment
        if tower_modules is not None:
            doc["network"]["lora_config"] = {
                "lora_r": 8,
                "lora_alpha": 16,
                "target_modules": ["q_proj"],
                "tower_target_modules": tower_modules,
            }
        manifest = TrainingManifest.model_validate(doc)

        # Act / Assert
        assert trains_context_parallel(manifest) is expected

    def test_an_algorithm_without_a_model_is_false(self):
        manifest = TrainingManifest.model_validate(
            {
                "algorithm": {"name": "DQN"},
                "environment": {"name": "LunarLander-v3", "num_envs": 8},
                "training": {"max_steps": 1000},
            }
        )

        assert trains_context_parallel(manifest) is False


class TestManifestAtLayout:
    def test_pins_the_layout_so_the_search_tries_it_alone(self):
        manifest = TrainingManifest.model_validate(MANIFEST)
        assert manifest.algorithm.fsdp is None

        placed = manifest_at_layout(manifest, ParallelLayout(ep=2))

        fsdp = placed.algorithm.fsdp
        assert (fsdp.tp, fsdp.ep, fsdp.cp, fsdp.shard_group_size) == (1, 2, 1, None)
        assert parallel_layouts(MOE_MODEL.arch, 4, 4, fsdp, True, True) == [
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

    def test_leaves_the_callers_manifest_unwritten(self):
        # Arrange
        doc = copy.deepcopy(MANIFEST)
        doc["algorithm"]["fsdp"] = {"activation_offload": True}
        manifest = TrainingManifest.model_validate(doc)
        before = manifest.model_dump()

        # Act
        placed = manifest_at_layout(manifest, ParallelLayout(tp=2, ep=2))

        # Assert
        assert manifest.model_dump() == before
        assert manifest.algorithm.fsdp.parallel_pins == frozenset()
        assert (placed.algorithm.fsdp.tp, placed.algorithm.fsdp.ep) == (2, 2)

    def test_result_fsdp_is_its_own_copy(self):
        # Arrange
        doc = copy.deepcopy(MANIFEST)
        doc["algorithm"]["fsdp"] = {"activation_offload": True}
        manifest = TrainingManifest.model_validate(doc)
        placed = manifest_at_layout(manifest, ParallelLayout(ep=2))
        again = manifest_at_layout(manifest, ParallelLayout(ep=2))

        # Act
        placed.algorithm.fsdp.cpu_offload = True
        placed.algorithm.fsdp.shard_group_size = 4

        # Assert
        assert manifest.algorithm.fsdp.cpu_offload is None
        assert manifest.algorithm.fsdp.shard_group_size is None
        assert again.algorithm.fsdp.cpu_offload is None
        assert again.algorithm.fsdp.shard_group_size is None

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

    def test_a_set_cpu_offload_keeps_the_search_off_expert_parallel(self):
        # Arrange: ep=1 fits only without the buffer, so the search goes on;
        # cpu_offload cannot run at ep > 1.
        doc = copy.deepcopy(SFT_MANIFEST)
        doc["algorithm"]["fsdp"] = {"cpu_offload": True}
        doc["network"] = {
            "pretrained_model_name_or_path": "Qwen/Qwen3.6-35B-A3B",
            "max_context_length": 65536,
        }
        doc["training"]["training_gpus_per_agent"] = 4
        manifest = TrainingManifest.model_validate(doc)
        gpu = lookup_gpu("NVIDIA L4")
        assert gpu is not None
        device = gpu.device_spec().model_copy(update={"gpus_per_node": 4})
        model = catalog_model(manifest.algorithm)
        assert model is not None

        # Act
        plan = training_memory_from_manifest(manifest, model, device)

        # Assert
        assert plan.fsdp is not None
        assert (plan.fsdp.tp, plan.fsdp.ep, plan.fsdp.cp) == (1, 1, 1)
        assert plan.fsdp.cpu_offload is True
        assert plan.breakdown.fits
        assert not plan.breakdown.fits_with_buffer

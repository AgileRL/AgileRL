# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Run memory: layout, trainer / engine split and weight-sync mode searched together."""

from __future__ import annotations

import json
import math

import pytest

from agilerl.arena.memory.breakdown import MemoryComponent, PhaseBreakdown
from agilerl.arena.memory.manifest import (
    nccl_reshard_layout_supported,
    training_memory_from_manifest,
    validated_run_manifest,
)
from agilerl.arena.memory.resolve.batch import catalog_model
from agilerl.arena.memory.resolve.fsdp import (
    TrainingMemoryPlan,
    TrainingMemoryUnfitError,
    learn_cost,
)
from agilerl.arena.memory.resolve.learn_costs import learn_costs
from agilerl.arena.memory.resolve.parallel import ParallelLayout, manifest_at_layout
from agilerl.arena.memory.resolve.run import (
    RunMemoryPlan,
    SearchCache,
    WeightSyncMode,
    _cached_training_plan,
    kept_split_memory,
    kept_weight_sync_mode,
    pick_weight_sync,
    resolve_run_memory,
    weight_sync_modes,
)
from agilerl.arena.memory.resolve.split import (
    GpuSplit,
    TrainerCost,
    choose_gpu_split,
    split_workload_from_manifest,
)
from agilerl.arena.memory.resolve.weight_sync import (
    SM90_NCCL_RESHARD_COSTS,
    SyncLink,
    nccl_reshard_sync_seconds,
)
from agilerl.arena.memory.specs import DeviceSpec, GiB, TrainingSettings
from agilerl.arena.models import model_info
from agilerl.arena.models.manifest import TrainingManifest
from agilerl.arena.models.model_info import supported_path
from tests.test_memory.test_cli import MANIFEST
from tests.test_memory.test_resolve_split import (
    _awm_manifest,
    _countdown_manifest,
    _device,
    _manifest,
    _sft_manifest,
    _split_on,
    _sudoku_manifest,
)
from tests.test_memory.test_resources import tiny_model, unset_split, with_model


def _oversize_countdown_manifest(batch_size: int) -> dict:
    """Countdown on Qwen3.8-27B LoRA at TP 1, CP 1; its training fits L4s at 6 trainers, not 1 to 5."""
    raw = _countdown_manifest(batch_size)
    raw["network"]["pretrained_model_name_or_path"] = "Qwen/Qwen3.8-27B"
    raw["algorithm"]["fsdp"] = {"tp": 1, "cp": 1}
    return raw


def _h100_nodes() -> DeviceSpec:
    return _device("H100-80").model_copy(update={"gpus_per_node": 8})


def _plan(
    raw: dict,
    gpus: int | None = 8,
    engine_width: int = 1,
    device: DeviceSpec | None = None,
) -> RunMemoryPlan:
    manifest = _manifest(raw, engine_width)
    return resolve_run_memory(
        manifest, catalog_model(manifest.algorithm), device or _device(), gpus
    )


def _split(raw: dict, **kwargs: object) -> GpuSplit:
    split = _plan(raw, **kwargs).split
    assert split is not None
    return split


def _training_plan(
    manifest: TrainingManifest, trainers: int, device: DeviceSpec
) -> TrainingMemoryPlan | None:
    trial = manifest.model_copy(deep=True)
    trial.training.training_gpus_per_agent = trainers
    try:
        return training_memory_from_manifest(
            trial, catalog_model(trial.algorithm), device
        )
    except TrainingMemoryUnfitError:
        return None


def _forced_split(
    raw: dict, trainers: int, gpus: int, engine_width: int = 1
) -> GpuSplit | None:
    """The fastest split of ``gpus`` that keeps exactly ``trainers`` trainers."""
    manifest = _manifest(raw, engine_width)
    device = _device()
    plan = _training_plan(manifest, trainers, device)
    if plan is None:
        return None
    return choose_gpu_split(
        catalog_model(manifest.algorithm),
        device,
        split_workload_from_manifest(manifest),
        gpus,
        engine_width,
        lambda count: TrainerCost.of(plan) if count == trainers else None,
    )


SEARCHED_SPLITS = [
    (_countdown_manifest(), 8, 1),
    (_countdown_manifest(batch_size=64), 8, 1),
    (_sudoku_manifest(), 8, 1),
    (_sudoku_manifest(), 8, 2),
    (_countdown_manifest(), 4, 1),
    (_countdown_manifest(batch_size=64), 4, 1),
    (_sudoku_manifest(), 4, 1),
]


SEARCHED_SPLIT_IDS = [
    "countdown-b60-8",
    "countdown-b64-8",
    "sudoku-8",
    "sudoku-tp2-8",
    "countdown-b60-4",
    "countdown-b64-4",
    "sudoku-4",
]


class TestResolveRunMemorySplit:
    @pytest.mark.parametrize(
        ("raw", "gpus", "engine_width"), SEARCHED_SPLITS, ids=SEARCHED_SPLIT_IDS
    )
    def test_split_fits_the_gpus_and_shards_the_batch_evenly(
        self, raw: dict, gpus: int, engine_width: int
    ) -> None:
        # Act
        plan = _plan(raw, gpus=gpus, engine_width=engine_width)

        # Assert
        split = plan.split
        assert split is not None
        batch_size = raw["algorithm"]["batch_size"]
        cycle = math.lcm(split.trainers, split.engines)
        assert split.trainers + split.engines * engine_width <= gpus
        assert batch_size % split.trainers == 0
        assert split.rollout_batch_size % cycle == 0
        assert cycle <= split.rollout_batch_size < batch_size + cycle
        assert plan.training.settings.n_training_gpus == split.trainers
        assert plan.training.breakdown.fits_with_buffer

    @pytest.mark.parametrize(
        ("raw", "gpus", "engine_width"), SEARCHED_SPLITS, ids=SEARCHED_SPLIT_IDS
    )
    def test_split_is_no_slower_than_any_other_trainer_count(
        self, raw: dict, gpus: int, engine_width: int
    ) -> None:
        # Arrange
        batch_size = raw["algorithm"]["batch_size"]
        dividing = [
            trainers
            for trainers in range(1, gpus - engine_width + 1)
            if batch_size % trainers == 0
        ]
        forced = [_forced_split(raw, n, gpus, engine_width) for n in dividing]
        others = [other for other in forced if other is not None]

        # Act
        split = _split(raw, gpus=gpus, engine_width=engine_width)

        # Assert
        assert len(others) > 1
        assert all(split.step_seconds <= other.step_seconds for other in others)
        assert split.step_seconds < max(other.step_seconds for other in others)

    def test_pinned_rollout_batch_size_limits_trainers_to_its_divisors(
        self,
    ) -> None:
        # Arrange
        raw = _countdown_manifest()
        raw["training"]["rollout_batch_size"] = 64

        # Act
        split = _split(raw)

        # Assert
        assert 64 % split.trainers == 0
        assert split.rollout_batch_size == 64

    def test_pinned_micro_batch_must_divide_each_rank_share(self) -> None:
        # Arrange: 12 groups of 8 over 4 ranks is 24 trajectories, not a
        # multiple of 16.
        raw = _sudoku_manifest()
        raw["algorithm"]["micro_batch_size_per_gpu"] = 16

        # Act
        split = _split(raw)

        # Assert
        assert (12 // split.trainers * 8) % 16 == 0

    def test_a_model_that_fits_keeps_degree_one(self) -> None:
        # Act
        plan = _plan(_countdown_manifest())

        # Assert: DDP, with no tensor, expert or context parallel group.
        assert plan.training.fsdp is None

    def test_a_pinned_shard_group_limits_trainers_to_its_multiples(self) -> None:
        # Arrange
        raw = _countdown_manifest()
        raw["algorithm"]["fsdp"] = {"shard_group_size": 2}

        # Act
        split = _split(raw)

        # Assert
        assert split.trainers % 2 == 0

    def test_two_gpus_give_one_trainer_one_engine(self) -> None:
        # Act
        split = _split(_sudoku_manifest(), gpus=2)

        # Assert
        assert (split.trainers, split.engines) == (1, 1)

    def test_a10g_gpus_get_a_split(self) -> None:
        # Act
        split = _split(_sudoku_manifest(), gpus=8, device=_device("A10G"))

        # Assert
        assert split.trainers + split.engines <= 8

    def test_no_room_for_a_trainer_beside_an_engine_raises(self) -> None:
        # Act / Assert
        with pytest.raises(
            ValueError,
            match=(
                r"^No trainer / rollout-engine split of 1 NVIDIA A100-SXM4-80GB "
                r"GPUs fits this run: a trainer and a rollout engine do not both "
                r"fit\. Use more or larger GPUs per agent \(a bigger resource tier "
                r"or a smaller population size\), or shorten the context length\.$"
            ),
        ):
            _plan(_sudoku_manifest(), gpus=1)

    def test_training_over_memory_at_every_trainer_count_raises(self) -> None:
        # Arrange: 9 GPUs with one TP4 engine leave at most 5 trainers.
        raw = _oversize_countdown_manifest(60)

        # Act / Assert
        with pytest.raises(
            ValueError,
            match=(
                r"^No trainer / rollout-engine split of 9 NVIDIA L4 GPUs fits this "
                r"run: training does not fit in memory at any trainer count\. "
                r"Use more or larger GPUs per agent \(a bigger resource tier or "
                r"a smaller population size\), or shorten the context length\.$"
            ),
        ):
            _plan(raw, gpus=9, engine_width=4, device=_device("L4"))

    @pytest.mark.parametrize(
        ("fsdp", "batch_size", "lora_r", "gpus"),
        [({"tp": 2}, 60, 32, 7), ({"shard_group_size": 4}, 64, 16, 9)],
        ids=["tensor-parallel", "shard-group"],
    )
    def test_training_over_memory_at_the_largest_fsdp_count_raises(
        self, fsdp: dict, batch_size: int, lora_r: int, gpus: int
    ) -> None:
        # Arrange: one TP4 engine leaves at most gpus - 4 trainers, an odd count
        # the FSDP width skips; the largest legal count shards and is over memory.
        # TP splits the adapters, so the TP case needs rank 32 to stay over.
        raw = _oversize_countdown_manifest(batch_size)
        raw["algorithm"]["fsdp"].update(fsdp)
        raw["network"]["lora_config"]["lora_r"] = lora_r

        # Act / Assert
        with pytest.raises(
            ValueError,
            match=(
                rf"^No trainer / rollout-engine split of {gpus} NVIDIA L4 GPUs fits "
                r"this run: training does not fit in memory at any trainer count\. "
                r"Use more or larger GPUs per agent \(a bigger resource tier or "
                r"a smaller population size\), or shorten the context length\.$"
            ),
        ):
            _plan(raw, gpus=gpus, engine_width=4, device=_device("L4"))

    def test_no_split_with_a_searched_engine_width_and_a_shared_cache_raises(
        self,
    ) -> None:
        # Arrange: the engine width is searched, so the cache also holds widths.
        raw = _oversize_countdown_manifest(60)
        raw["algorithm"]["vllm_config"] = {}
        manifest = validated_run_manifest(TrainingManifest.model_validate(raw))

        # Act / Assert
        with pytest.raises(
            ValueError,
            match=r"^No trainer / rollout-engine split of 9 NVIDIA L4 GPUs fits this run",
        ):
            resolve_run_memory(
                manifest,
                catalog_model(manifest.algorithm),
                _device("L4"),
                9,
                search_cache=SearchCache(),
            )

    def test_batch_no_trainer_count_that_fits_divides_raises(self) -> None:
        # Arrange: only 6 trainers fit, and 64 groups do not split over 6.
        raw = _oversize_countdown_manifest(64)

        # Act / Assert
        with pytest.raises(
            ValueError,
            match=(
                r"^No trainer / rollout-engine split of 10 NVIDIA L4 GPUs fits this "
                r"run: batch size 64 does not divide evenly across any trainer "
                r"count that fits in memory\. Use a batch size divisible by more "
                r"trainer counts, such as a power of two\.$"
            ),
        ):
            _plan(raw, gpus=10, engine_width=4, device=_device("L4"))

    def test_pinned_rollout_batch_no_trainer_count_that_fits_divides_raises(
        self,
    ) -> None:
        # Arrange: only 6 trainers fit; 60 groups split over 6, a pinned
        # 64-group cycle does not.
        raw = _oversize_countdown_manifest(60)
        raw["training"]["rollout_batch_size"] = 64

        # Act / Assert
        with pytest.raises(
            ValueError,
            match=(
                r"run: rollout batch size 64 does not divide evenly across any "
                r"trainer count that fits in memory\. Use a rollout batch size "
                r"divisible by more trainer counts, such as a power of two, or "
                r"leave it unset\.$"
            ),
        ):
            _plan(raw, gpus=10, engine_width=4, device=_device("L4"))

    @pytest.mark.parametrize(
        ("fsdp", "need"),
        [
            ({"tp": 8}, "FSDP tensor parallel needs"),
            ({"ep": 8}, "FSDP expert parallel needs"),
            ({"shard_group_size": 8}, "FSDP shard group needs"),
            (
                {"tp": 8, "shard_group_size": 8},
                "FSDP tensor parallel and shard group need",
            ),
            (
                {"tp": 8, "ep": 8, "shard_group_size": 8},
                "FSDP tensor parallel, expert parallel, and shard group need",
            ),
            ({"cp": 8}, "FSDP context parallel needs"),
            (
                {"tp": 2, "cp": 4, "ep": 8, "shard_group_size": 8},
                (
                    "FSDP tensor parallel, context parallel, expert parallel, "
                    "and shard group need"
                ),
            ),
        ],
        ids=[
            "tensor-parallel",
            "expert-parallel",
            "shard-group",
            "tensor-and-shard",
            "three",
            "context-parallel",
            "all-four",
        ],
    )
    def test_fsdp_width_wider_than_the_trainers_raises(
        self, fsdp: dict, need: str
    ) -> None:
        # Arrange: one engine leaves 7 of 8 GPUs, under an FSDP width of 8.
        raw = _sudoku_manifest()
        raw["algorithm"]["fsdp"] = fsdp

        # Act / Assert
        with pytest.raises(
            ValueError,
            match=(
                rf"run: {need} a multiple of 8 "
                r"trainer GPUs, and at most 7 fit beside a rollout engine\. Use "
                r"more GPUs per agent \(a bigger resource tier or a smaller "
                r"population size\)\.$"
            ),
        ):
            _plan(raw, gpus=8)

    @pytest.mark.parametrize(
        ("pins", "cause"),
        [
            (
                {"micro_batch_size_per_gpu": 7},
                (
                    r"run: micro_batch_size_per_gpu 7 does not divide the per-rank "
                    r"batch at any trainer count that shards the batch\. Use a "
                    r"micro_batch_size_per_gpu that divides those ranks, or leave it "
                    r"unset\.$"
                ),
            ),
            (
                {"mini_batch_size": 7},
                (
                    r"run: mini_batch_size 7 does not divide the per-rank batch at "
                    r"any trainer count that shards the batch\. Use a "
                    r"mini_batch_size that divides those ranks, or leave it unset\.$"
                ),
            ),
            (
                {"micro_batch_size_per_gpu": 7, "mini_batch_size": 7},
                (
                    r"run: micro_batch_size_per_gpu 7 and mini_batch_size 7 do not "
                    r"divide the per-rank batch at any trainer count that shards "
                    r"the batch\. Use values that divide those ranks, or leave them "
                    r"unset\.$"
                ),
            ),
        ],
        ids=["micro-batch", "mini-batch", "micro-and-mini"],
    )
    def test_per_rank_pin_no_trainer_count_that_shards_divides_raises(
        self, pins: dict, cause: str
    ) -> None:
        # Arrange: batch 12 shards 1, 2, 3, 4, and 6 trainers; 7 divides none
        # of those per-rank trajectory counts.
        raw = _sudoku_manifest()
        raw["algorithm"].update(pins)

        # Act / Assert
        with pytest.raises(ValueError, match=cause):
            _plan(raw, gpus=8)

    def test_no_split_reason_reads_the_least_parallel_layout(self) -> None:
        # Arrange: at TP 1 batch 7 shards over one trainer only, and its 56
        # trajectories do not split into micro-batches of 3. Read at TP 2, the
        # last layout tried, the cause would be the batch size.
        raw = _countdown_manifest(7)
        raw["algorithm"]["micro_batch_size_per_gpu"] = 3
        raw["training"]["rollout_batch_size"] = 7

        # Act / Assert
        with pytest.raises(
            ValueError,
            match=(
                r"^No trainer / rollout-engine split of 4 NVIDIA L4 GPUs fits "
                r"this run: micro_batch_size_per_gpu 3 does not divide the "
                r"per-rank batch at any trainer count that shards the batch\."
            ),
        ):
            _plan(raw, gpus=4, device=_device("L4"))

    def test_algorithm_without_rollouts_raises(self) -> None:
        # Arrange
        manifest = _sft_manifest()

        # Act / Assert
        with pytest.raises(TypeError, match="SFT does not generate rollouts"):
            resolve_run_memory(
                manifest, catalog_model(manifest.algorithm), _device(), 8
            )

    @pytest.mark.parametrize("gpus", [2, None], ids=["searched", "pinned"])
    def test_gpu_below_sm80_raises_unsupported(self, gpus: int | None) -> None:
        # Arrange
        raw = _sudoku_manifest()
        if gpus is None:
            raw["training"]["training_gpus_per_agent"] = 1
            raw["training"]["rollout_engines_per_agent"] = 1

        # Act / Assert
        with pytest.raises(
            ValueError, match=r"Tesla T4 \(SM75\) cannot run LLM training"
        ):
            _plan(raw, gpus=gpus, device=_device("T4"))

    def test_gpu_without_peak_figures_raises_a_catalogue_error(self) -> None:
        # Arrange
        device = DeviceSpec(total_bytes=80 * 1024**3, name="Exotic GPU")

        # Act / Assert
        with pytest.raises(
            ValueError, match="Exotic GPU has no peak bf16 or memory bandwidth"
        ):
            _plan(_sudoku_manifest(), gpus=8, device=device)

    def test_trainer_counts_nothing_fits_are_skipped(self) -> None:
        # Arrange: Qwen3.8-27B LoRA is over one L4 at some trainer counts.
        raw = _oversize_countdown_manifest(60)
        manifest = _manifest(raw, engine_width=4)
        unfit = [
            trainers
            for trainers in range(1, 14 - 4 + 1)
            if 60 % trainers == 0
            and _training_plan(manifest, trainers, _device("L4")) is None
        ]

        # Act
        split = _split(raw, gpus=14, engine_width=4, device=_device("L4"))

        # Assert
        assert unfit
        assert split.trainers not in unfit
        assert split.trainers + split.engines * 4 <= 14


class TestResolveRunMemory:
    @pytest.mark.parametrize(
        ("fsdp", "priced"),
        [
            pytest.param(None, [5], id="searched"),
            pytest.param({"ep": 5}, [5, 10], id="pinned"),
        ],
    )
    def test_only_a_pinned_expert_group_is_priced_across_nodes(
        self, fsdp: dict | None, priced: list[int]
    ) -> None:
        # Arrange: EP 5 on 8-GPU nodes; 10 trainers puts a group across two nodes.
        doc = unset_split(
            with_model(MANIFEST, "ibm-granite/granite-3.1-3b-a800m-instruct")
        )
        doc["algorithm"]["batch_size"] = 40
        if fsdp is not None:
            doc["algorithm"]["fsdp"] = fsdp
        manifest = TrainingManifest.model_validate(doc)
        device = _device("L4").model_copy(update={"gpus_per_node": 8})
        cache = SearchCache()

        # Act
        plan = resolve_run_memory(
            manifest,
            catalog_model(manifest.algorithm),
            device,
            16,
            None,
            True,
            cache,
            ParallelLayout(ep=5),
        )

        # Assert
        assert [sorted(plans) for plans in cache.plans.values()] == [priced]
        assert plan.split is not None
        assert (plan.split.trainers, plan.split.engines) == (5, 7)
        assert plan.training.fsdp is not None
        assert plan.training.fsdp.ep == 5

    @pytest.mark.parametrize(
        ("raw", "device_name"),
        [
            (_countdown_manifest(64), "A100-80"),
            (_countdown_manifest(60), "A100-80"),
            (_sudoku_manifest(), "A100-80"),
            (_countdown_manifest(64), "L4"),
            (_countdown_manifest(60), "L4"),
            (_sudoku_manifest(), "L4"),
        ],
        ids=["a-b64-a100", "a-b60-a100", "b-a100", "a-b64-l4", "a-b60-l4", "b-l4"],
    )
    def test_benchmark_plan_fits_and_beats_the_slowest_rungs(
        self, raw: dict, device_name: str
    ) -> None:
        # Arrange
        device = _device(device_name)

        # Act
        plan = _plan(raw, device=device)

        # Assert
        training = plan.training
        rows = training.settings.trajectories_per_update // (
            training.settings.n_training_gpus
        )
        slowest = learn_cost(
            catalog_model(_manifest(raw).algorithm).arch.n_layers,
            True,
            1,
            rows,
            training.tokens,
            "reshard",
            learn_costs(device),
        )
        assert training.breakdown.fits_with_buffer
        assert training.cost < slowest

    @pytest.mark.parametrize("device_name", ["A100-80", "L4"])
    def test_trains_ddp_when_a_ddp_plan_fits(self, device_name: str) -> None:
        # Arrange
        device = _device(device_name)
        ddp_raw = _countdown_manifest(64)
        del ddp_raw["algorithm"]["fsdp"]

        # Act
        plan = _plan(_countdown_manifest(64), device=device)

        # Assert
        trainers = plan.training.settings.n_training_gpus
        ddp = _training_plan(_manifest(ddp_raw), trainers, device)
        assert ddp is not None
        assert ddp.breakdown.fits_with_buffer
        assert plan.training.fsdp is None

    def test_shared_cache_keeps_each_host_budget(self) -> None:
        # Arrange: one GPU and node size, two host shares. Host 0 cannot
        # hold a trainer process; an unset share skips the host check.
        # Five GPUs leave at most four trainers, and 64 divides that count.
        manifest = _manifest(_countdown_manifest(64))
        model = catalog_model(manifest.algorithm)
        gpu = _device("A100-80").model_copy(update={"gpus_per_node": 8})
        tight = gpu.model_copy(update={"host_bytes_per_gpu": 0})
        roomy = gpu.model_copy(update={"host_bytes_per_gpu": None})
        cache = SearchCache()

        # Act / Assert: a tight host must not poison a roomy one, or the reverse.
        with pytest.raises(ValueError, match="training does not fit in memory"):
            resolve_run_memory(manifest, model, tight, 5, search_cache=cache)
        plan = resolve_run_memory(manifest, model, roomy, 5, search_cache=cache)

        assert plan.training.breakdown.fits_with_buffer
        with pytest.raises(ValueError, match="training does not fit in memory"):
            resolve_run_memory(manifest, model, tight, 5, search_cache=cache)

    def test_pinned_counts_skip_the_split_search(self) -> None:
        # Arrange
        raw = _countdown_manifest(64)
        raw["training"]["training_gpus_per_agent"] = 2
        raw["training"]["rollout_engines_per_agent"] = 2
        manifest = _manifest(raw)

        # Act
        plan = resolve_run_memory(
            manifest,
            catalog_model(manifest.algorithm),
            _device(),
            4,
            search_split=False,
        )

        # Assert
        assert plan.split is None
        assert plan.training.settings.n_training_gpus == 2
        assert manifest.training.training_gpus_per_agent == 2
        assert manifest.algorithm.micro_batch_size_per_gpu is None

    def test_pinned_counts_that_fit_no_layout_raise_the_plan_error(self) -> None:
        # Arrange: neither sync mode's trainers fit a 1 GiB GPU.
        raw = _countdown_manifest(64)
        raw["training"]["training_gpus_per_agent"] = 2
        raw["training"]["rollout_engines_per_agent"] = 2
        manifest = _manifest(raw)

        # Act / Assert
        with pytest.raises(TrainingMemoryUnfitError):
            resolve_run_memory(
                manifest,
                catalog_model(manifest.algorithm),
                _l4_with(1.0),
                4,
                search_split=False,
            )

    def test_drafting_moves_multi_turn_gpus_to_training(self) -> None:
        # Arrange
        plain_raw = _sudoku_manifest()
        plain_raw["algorithm"]["vllm_engine_args"] = {"speculative_config": None}

        # Act
        drafted = _split(_sudoku_manifest(), device=_device("H100-80"))
        plain = _split(plain_raw, device=_device("H100-80"))

        # Assert
        assert drafted.trainers > plain.trainers
        assert drafted.step_seconds < plain.step_seconds


def _plan_with_host(host_total: int, host_usable: int | None) -> TrainingMemoryPlan:
    """A trainer plan whose host components total ``host_total``."""
    return TrainingMemoryPlan(
        gradient_checkpointing=False,
        checkpoint_every_n_blocks=1,
        micro_batch_size_per_gpu=1,
        micro_batch_tokens_per_gpu=None,
        fsdp=None,
        routed_expert_chunk_mib=None,
        moe_collectives_saveable=False,
        cost=1.0,
        tokens=1.0,
        settings=TrainingSettings(),
        breakdown=PhaseBreakdown(
            phase="training",
            components=(MemoryComponent(key="weights", label="Weights", bytes=1),),
            device_total_bytes=100,
            device_usable_bytes=100,
            host=(MemoryComponent(key="process", label="Process", bytes=host_total),),
            host_usable_bytes=host_usable,
        ),
    )


class TestCachedTrainingPlan:
    def test_host_shares_share_the_plan_until_one_misses(self, monkeypatch) -> None:
        # Arrange: the GPU search stores 1000 host bytes. 10000 and an unset
        # share both fit that total; 100 and 50 do not.
        placed = manifest_at_layout(
            TrainingManifest.model_validate(MANIFEST), ParallelLayout()
        )
        model = tiny_model()
        mode = WeightSyncMode(False, "nixl")
        base = _device("L4")
        roomy = base.model_copy(update={"host_bytes_per_gpu": 10_000})
        unset = base.model_copy(update={"host_bytes_per_gpu": None})
        tight = base.model_copy(update={"host_bytes_per_gpu": 100})
        tighter = base.model_copy(update={"host_bytes_per_gpu": 50})
        cache = SearchCache()
        priced: list[int | None] = []

        def fake(manifest, model, device, adapter_sync, nccl_reshard):
            priced.append(device.host_bytes_per_gpu)
            if device.host_bytes_per_gpu is None:
                return _plan_with_host(1000, None)
            return _plan_with_host(
                device.host_bytes_per_gpu // 2, device.host_bytes_per_gpu
            )

        monkeypatch.setattr(
            "agilerl.arena.memory.resolve.run.training_memory_from_manifest", fake
        )

        def price(device):
            return _cached_training_plan(cache, placed, model, device, 1, mode, True)

        # Act
        first = price(roomy)
        second = price(unset)

        # Assert: one GPU search, priced with no host share.
        assert priced == [None]
        assert first is second
        assert first.breakdown.host_total_bytes == 1000
        assert len(cache.plans) == 1
        assert cache.host_misses == {}

        # Act: a share that misses prices later rungs; the same share does not.
        missed = price(tight)
        missed_again = price(tight)

        # Assert
        assert priced == [None, 100]
        assert missed is missed_again
        assert missed is not first
        assert missed.breakdown.host_usable_bytes == 100
        assert missed.breakdown.host_total_bytes == 50
        assert len(cache.host_misses) == 1

        # Act: a different share is priced again and does not reuse the miss.
        other = price(tighter)

        # Assert
        assert priced == [None, 100, 50]
        assert other is not missed
        assert other.breakdown.host_usable_bytes == 50
        assert len(cache.host_misses) == 2

    def test_a_share_inside_the_buffer_misses_only_when_required(
        self, monkeypatch
    ) -> None:
        # Arrange: host total equals the share, so it fits the point estimate
        # and misses the underprediction buffer.
        placed = manifest_at_layout(
            TrainingManifest.model_validate(MANIFEST), ParallelLayout()
        )
        model = tiny_model()
        mode = WeightSyncMode(False, "nixl")
        device = _device("L4").model_copy(update={"host_bytes_per_gpu": 1000})
        cache = SearchCache()
        priced: list[int | None] = []

        def fake(manifest, model, device, adapter_sync, nccl_reshard):
            priced.append(device.host_bytes_per_gpu)
            if device.host_bytes_per_gpu is None:
                return _plan_with_host(1000, None)
            return _plan_with_host(500, device.host_bytes_per_gpu)

        monkeypatch.setattr(
            "agilerl.arena.memory.resolve.run.training_memory_from_manifest", fake
        )

        # Act
        buffered = _cached_training_plan(cache, placed, model, device, 1, mode, True)
        bare = _cached_training_plan(cache, placed, model, device, 1, mode, False)

        # Assert
        assert priced == [None, 1000]
        assert buffered.breakdown.host_total_bytes == 500
        assert buffered.breakdown.host_usable_bytes == 1000
        assert bare is not buffered
        assert bare.breakdown.host_total_bytes == 1000
        assert bare.breakdown.host_usable_bytes is None
        assert len(cache.host_misses) == 1


class TestPickWeightSync:
    def test_merged_within_the_share_wins_over_cuda_ipc(self) -> None:
        # Arrange
        merged = _split_on(2, 10.5)
        adapter = _split_on(2, 10.0)

        # Act
        chosen = pick_weight_sync([merged, adapter], "cuda_ipc")

        # Assert
        assert chosen is merged

    @pytest.mark.parametrize("sync_link", ["rdma", "tcp"])
    def test_fastest_wins_over_other_links(self, sync_link: SyncLink) -> None:
        # Arrange
        merged = _split_on(2, 10.5)
        adapter = _split_on(2, 10.0)

        # Act
        chosen = pick_weight_sync([merged, adapter], sync_link)

        # Assert
        assert chosen is adapter

    def test_first_fitting_mode_wins_with_no_link(self) -> None:
        # Arrange
        merged = _split_on(2, 12.0)
        adapter = _split_on(2, 10.0)

        # Act
        chosen = pick_weight_sync([merged, adapter], None)

        # Assert
        assert chosen is merged

    def test_faster_merged_transport_beats_a_slower_one(self) -> None:
        # Arrange: nccl_reshard is listed after NIXL and is the faster merged step.
        nixl = _quoted(8.0, 8.0, 0.77, merged=True, transport="nixl")
        nccl = _quoted(8.0, 8.0, 0.31, merged=True, transport="nccl_reshard")
        adapter = _quoted(8.0, 12.0, 0.14, merged=False, transport="nixl")

        # Act
        chosen = pick_weight_sync([nixl, nccl, adapter], "cuda_ipc")

        # Assert
        assert chosen is nccl

    def test_fastest_merged_transport_wins_when_no_adapter_fits(self) -> None:
        # Arrange
        nixl = _quoted(8.0, 8.0, 0.77, merged=True, transport="nixl")
        nccl = _quoted(8.0, 8.0, 0.31, merged=True, transport="nccl_reshard")

        # Act
        chosen = pick_weight_sync([nixl, nccl], "cuda_ipc")

        # Assert
        assert chosen is nccl

    def test_adapter_wins_when_merged_sync_is_outside_the_share(self) -> None:
        # Arrange: merged adds 0.31 s on a 1 s learn-bound step, past the 10% share.
        merged = _quoted(1.0, 0.2, 0.31, merged=True, transport="nccl_reshard")
        adapter = _quoted(1.0, 0.4, 0.14, merged=False, transport="nixl")

        # Act
        chosen = pick_weight_sync([merged, adapter], "cuda_ipc")

        # Assert
        assert chosen is adapter


def _countdown_at(context: int, vllm_config: dict) -> dict:
    raw = _countdown_manifest(batch_size=64)
    raw["network"]["max_context_length"] = context
    raw["algorithm"]["vllm_config"] = vllm_config
    return raw


def _searched_split(raw: dict, device_name: str) -> GpuSplit:
    manifest = validated_run_manifest(TrainingManifest.model_validate(raw))
    plan = resolve_run_memory(
        manifest, catalog_model(manifest.algorithm), _device(device_name), 8
    )
    assert plan.split is not None
    return plan.split


class TestResolveRunMemoryEngineWidth:
    @pytest.mark.parametrize("device_name", ["A100-80", "H100-80"])
    @pytest.mark.parametrize("context", [1536, 4096])
    def test_single_turn_small_dense_model_stays_one_gpu(
        self, context: int, device_name: str
    ) -> None:
        # Act
        split = _searched_split(_countdown_at(context, {}), device_name)

        # Assert
        assert split.engine_width == 1
        assert (split.trainers, split.engines) == (4, 4)

    def test_multi_turn_moe_model_takes_two_tp2_engines(self) -> None:
        # Arrange: TP2 and TP4 steps are within ENGINE_WIDTH_TIE_SHARE; the
        # tie goes to TP2.
        raw = _awm_manifest()

        # Act
        split = _searched_split(raw, "H100-80")

        # Assert
        assert split.engine_width == 2
        assert (split.trainers, split.engines) == (4, 2)
        assert split.rollout_batch_size == 16
        assert split.max_num_batched_tokens == 32768


def _plan_over(
    raw: dict, device_name: str, sync_link: SyncLink | None, search_split: bool = True
) -> RunMemoryPlan:
    manifest = validated_run_manifest(TrainingManifest.model_validate(raw))
    return resolve_run_memory(
        manifest,
        catalog_model(manifest.algorithm),
        _device(device_name),
        8,
        sync_link,
        search_split,
    )


class TestWeightSyncModes:
    def test_engines_serving_merged_weights_price_both_modes(self) -> None:
        # Arrange
        manifest = _manifest(_sudoku_manifest())

        # Act
        modes = weight_sync_modes(manifest, _device("A100-80"), None)

        # Assert
        assert modes == (WeightSyncMode(True, "nixl"), WeightSyncMode(False, "nixl"))

    def test_quantized_engine_syncs_the_adapter(self) -> None:
        # Arrange
        raw = _sudoku_manifest()
        manifest = _manifest(raw)
        manifest.algorithm.vllm_config.quantization = "fp8"

        # Act
        modes = weight_sync_modes(manifest, _device("A100-80"), None)

        # Assert
        assert modes == (WeightSyncMode(False, "nixl"),)

    def test_mtp_drafting_engine_syncs_the_adapter(self) -> None:
        # Arrange
        raw = _sudoku_manifest()
        raw["algorithm"]["vllm_engine_args"] = {
            "speculative_config": {"method": "qwen3_5_mtp", "num_speculative_tokens": 1}
        }
        manifest = _manifest(raw)

        # Act
        modes = weight_sync_modes(manifest, _device("A100-80"), None)

        # Assert
        assert modes == (WeightSyncMode(False, "nixl"),)

    def test_algorithm_without_rollout_engines_syncs_nothing_merged(self) -> None:
        # Act
        modes = weight_sync_modes(_sft_manifest(), _device("A100-80"), None)

        # Assert
        assert modes == (WeightSyncMode(False, "nixl"),)

    def test_replaced_catalogue_entry_is_read_afresh(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        manifest = _manifest(_sudoku_manifest())
        hub_id = manifest.algorithm.pretrained_model_name_or_path
        unquantized = weight_sync_modes(manifest, _device("A100-80"), None)
        monkeypatch.setitem(
            model_info.SUPPORTED_MODEL_INFO,
            hub_id,
            model_info.SUPPORTED_MODEL_INFO[PREQUANTIZED_EXPERT_LORA_MODELS[0]],
        )

        # Act
        modes = weight_sync_modes(manifest, _device("A100-80"), None)

        # Assert
        assert unquantized == (
            WeightSyncMode(True, "nixl"),
            WeightSyncMode(False, "nixl"),
        )
        assert modes == (WeightSyncMode(False, "nixl"),)


def _l4_with(gib: float) -> DeviceSpec:
    """An L4 resized to ``gib`` GiB, so one sync mode's trainers fit and the other's do not."""
    return _device("L4").model_copy(update={"total_bytes": int(gib * 2**30)})


def _kept_4t4e(raw: dict) -> TrainingManifest:
    raw["training"]["training_gpus_per_agent"] = 4
    raw["training"]["rollout_engines_per_agent"] = 4
    return _manifest(raw)


def _kept_plan(
    manifest: TrainingManifest, device: DeviceSpec, sync_link: SyncLink | None
) -> RunMemoryPlan | None:
    return kept_split_memory(
        manifest, catalog_model(manifest.algorithm), device, 8, sync_link, SearchCache()
    )


def _kept_sync_over(
    manifest: TrainingManifest, device_name: str, sync_link: SyncLink | None
) -> bool:
    plan = _kept_plan(manifest, _device(device_name), sync_link)
    assert plan is not None
    return plan.merged_lora_sync


class TestKeptSplitMemory:
    def test_a_shared_cache_prices_each_model_on_its_own(self) -> None:
        # Arrange
        manifest = _kept_4t4e(_countdown_manifest(64))
        device = _device("H100-80")
        models = (tiny_model(), catalog_model(manifest.algorithm))
        fresh = [
            kept_split_memory(manifest, model, device, 8, "cuda_ipc", SearchCache())
            for model in models
        ]
        cache = SearchCache()

        # Act
        shared = [
            kept_split_memory(manifest, model, device, 8, "cuda_ipc", cache)
            for model in models
        ]

        # Assert
        assert shared == fresh
        assert fresh[0] != fresh[1]

    def test_generation_bound_counts_keep_merged_sync(self) -> None:
        # Arrange: sudoku at 4T4E on A100 is priced 62.8 s merged over RDMA,
        # 71.4 s with the adapter.
        manifest = _kept_4t4e(_sudoku_manifest())

        # Act
        merged = _kept_sync_over(manifest, "A100-80", "rdma")

        # Assert
        assert merged is True

    def test_learn_bound_counts_sync_the_adapter(self) -> None:
        # Arrange: countdown at 4T4E on H100 is learn-bound; a 3 s merged sync
        # over RDMA blocks the trainer and pauses the engines.
        manifest = _kept_4t4e(_countdown_manifest(64))

        # Act
        merged = _kept_sync_over(manifest, "H100-80", "rdma")

        # Assert
        assert merged is False

    def test_merged_is_kept_within_the_tie_share(self) -> None:
        # Arrange: countdown at 4T4E on H100 is priced 6% slower merged over CUDA IPC.
        manifest = _kept_4t4e(_countdown_manifest(64))

        # Act
        merged = _kept_sync_over(manifest, "H100-80", "cuda_ipc")

        # Assert
        assert merged is True

    def test_tcp_syncs_the_adapter(self) -> None:
        # Arrange: merged Qwen3-4B weights take over a minute per sync at 0.4 GB/s.
        manifest = _kept_4t4e(_sudoku_manifest())

        # Act
        merged = _kept_sync_over(manifest, "A100-80", "tcp")

        # Assert
        assert merged is False

    def test_unknown_link_syncs_merged_weights_unpriced(self) -> None:
        # Arrange
        manifest = _kept_4t4e(_countdown_manifest(64))

        # Act
        merged = _kept_sync_over(manifest, "H100-80", None)

        # Assert
        assert merged is True

    @pytest.mark.parametrize("sync_link", [None, "cuda_ipc"])
    def test_l4_syncs_the_adapter_when_the_nixl_merged_buffer_does_not_fit(
        self, sync_link: SyncLink | None
    ) -> None:
        # Arrange: countdown at 4 trainers and TP 1 needs 4.75 GiB syncing the
        # adapter. The NIXL merged send buffer does not fit a 5.25 GiB GPU, and
        # an L4 has no measured nccl_reshard rate.
        raw = _countdown_manifest(64)
        raw["algorithm"]["fsdp"] = {"tp": 1}
        manifest = _kept_4t4e(raw)

        # Act
        plan = _kept_plan(manifest, _l4_with(5.25), sync_link)

        # Assert
        assert plan is not None
        assert plan.split is None
        assert plan.merged_lora_sync is False
        assert plan.weight_sync_transport == "nixl"
        assert plan.training.settings.merged_lora_sync is False
        assert all(quote.sync_transport == "nixl" for quote in plan.quotes)

    @pytest.mark.parametrize(
        ("gib", "transport", "transports"),
        [
            (8.5, "nixl", {"nixl"}),
            (9.0, "nccl_reshard", {"nixl", "nccl_reshard"}),
        ],
    )
    def test_a100_takes_nccl_reshard_only_where_its_refit_groups_fit(
        self, gib: float, transport: str, transports: set[str]
    ) -> None:
        # Arrange: the same run on a small A100. nccl_reshard's two refit
        # NCCL groups hold more than the NIXL merged send buffer.
        raw = _countdown_manifest(64)
        raw["algorithm"]["fsdp"] = {"tp": 1}
        manifest = _kept_4t4e(raw)
        device = _device("A100-80").model_copy(update={"total_bytes": int(gib * GiB)})

        # Act
        plan = _kept_plan(manifest, device, "cuda_ipc")

        # Assert
        assert plan is not None
        assert plan.merged_lora_sync is True
        assert plan.weight_sync_transport == transport
        assert plan.training.settings.merged_lora_sync is True
        assert {quote.sync_transport for quote in plan.quotes} == transports

    def test_quantized_engine_syncs_the_adapter(self) -> None:
        # Arrange
        manifest = _kept_4t4e(_sudoku_manifest())
        manifest.algorithm.vllm_config.quantization = "fp8"

        # Act
        merged = _kept_sync_over(manifest, "A100-80", "cuda_ipc")

        # Assert
        assert merged is False

    def test_the_least_parallel_layout_with_a_buffered_fit_picks_the_mode(
        self,
    ) -> None:
        # Arrange: on a 5.25 GiB L4, 4 countdown trainers first fit with the
        # buffer at TP 2 syncing the adapter; merged sync fits there only
        # without it.
        manifest = _kept_4t4e(_countdown_manifest(64))

        # Act
        plan = _kept_plan(manifest, _l4_with(5.25), None)

        # Assert
        assert plan is not None
        assert plan.merged_lora_sync is False
        assert plan.training.fsdp is not None
        assert (plan.training.fsdp.tp, plan.training.fsdp.cp) == (2, 1)
        assert plan.training.breakdown.fits_with_buffer

    def test_no_fitting_mode_has_no_plan(self) -> None:
        # Arrange: neither mode's trainers fit a 1 GiB GPU at any layout.
        manifest = _kept_4t4e(_countdown_manifest(64))

        # Act
        plan = _kept_plan(manifest, _l4_with(1.0), "rdma")

        # Assert
        assert plan is None

    def test_counts_without_engine_gpus_take_the_first_fitting_mode(self) -> None:
        # Arrange: 8 trainers leave no GPU of 8 for an auto engine.
        raw = _countdown_manifest(64)
        raw["training"]["training_gpus_per_agent"] = 8
        manifest = _manifest(raw)

        # Act
        merged = _kept_sync_over(manifest, "H100-80", "rdma")

        # Assert
        assert merged is True

    def test_rollout_spec_without_engines_takes_the_first_fitting_mode(self) -> None:
        # Arrange: without a vLLM config the counts have no engine to price.
        manifest = _kept_4t4e(_countdown_manifest(64))
        manifest.algorithm.vllm_config = None

        # Act
        merged = _kept_sync_over(manifest, "H100-80", "rdma")

        # Assert
        assert merged is True

    def test_picks_the_mode_the_search_picks_at_the_same_counts(self) -> None:
        # Arrange: the search over RDMA syncs the adapter on countdown.
        searched = _plan_over(_countdown_manifest(64), "H100-80", "rdma").split
        assert searched is not None
        raw = _countdown_manifest(64)
        raw["training"]["training_gpus_per_agent"] = searched.trainers
        raw["training"]["rollout_engines_per_agent"] = searched.engines
        raw["training"]["rollout_batch_size"] = searched.rollout_batch_size
        manifest = _manifest(raw, searched.engine_width)

        # Act
        merged = _kept_sync_over(manifest, "H100-80", "rdma")

        # Assert
        assert merged is searched.merged_lora_sync


class TestKeptWeightSyncMode:
    def test_takes_the_mode_the_kept_plan_syncs(self) -> None:
        # Arrange: countdown at 4T4E on H100 is learn-bound over RDMA.
        manifest = _kept_4t4e(_countdown_manifest(64))

        # Act
        mode = kept_weight_sync_mode(
            manifest,
            catalog_model(manifest.algorithm),
            _device("H100-80"),
            8,
            "rdma",
            SearchCache(),
        )

        # Assert
        assert mode == WeightSyncMode(False, "nixl")

    def test_no_fitting_mode_takes_the_first_mode(self) -> None:
        # Arrange: neither mode's trainers fit a 1 GiB GPU.
        manifest = _kept_4t4e(_countdown_manifest(64))

        # Act
        mode = kept_weight_sync_mode(
            manifest,
            catalog_model(manifest.algorithm),
            _l4_with(1.0),
            8,
            "rdma",
            SearchCache(),
        )

        # Assert
        assert mode == WeightSyncMode(True, "nixl")


class TestResolveRunMemoryWeightSync:
    def test_nvlink_syncs_merged_weights_on_a_generation_bound_run(self) -> None:
        # Arrange: Sudoku on 8x A100 ran 29% faster per step with merged sync.
        # Its FSDP trainers gather every misc weight before an nccl_reshard
        # refit, so merged NIXL is the faster merged transport.
        raw = _sudoku_manifest()

        # Act
        plan = _plan_over(raw, "A100-80", "cuda_ipc")

        # Assert
        assert plan.merged_lora_sync is True
        assert plan.weight_sync_transport == "nixl"
        assert plan.split is not None
        assert plan.split.merged_lora_sync
        assert plan.training.fsdp is not None
        assert plan.split.sync_seconds < _quote(plan, True, "nccl_reshard").sync_seconds

    def test_tcp_syncs_the_adapter(self) -> None:
        # Act: merged Qwen3-4B weights take over a minute per sync at 0.4 GB/s.
        plan = _plan_over(_sudoku_manifest(), "A100-80", "tcp")

        # Assert
        assert plan.merged_lora_sync is False
        assert plan.split is not None
        assert not plan.split.merged_lora_sync
        assert plan.training.settings.merged_lora_sync is False

    def test_nvlink_syncs_merged_weights_on_a_trainer_bound_run(self) -> None:
        # Act: countdown on 8x H100. Merged generation is faster than the
        # adapter step, and the run was 1.6% faster merged.
        plan = _plan_over(_countdown_manifest(64), "H100-80", "cuda_ipc")

        # Assert
        assert plan.merged_lora_sync is True

    def test_rdma_countdown_keeps_merged_weights(self) -> None:
        # Act: merged generation on H100 countdown saves more than the
        # 100 Gb/s RDMA sync of the checkpoint costs.
        plan = _plan_over(_countdown_manifest(64), "H100-80", "rdma")

        # Assert
        assert plan.merged_lora_sync is True
        assert plan.weight_sync_transport == "nixl"
        assert plan.split is not None
        assert plan.split.merged_lora_sync

    def test_merged_is_kept_within_the_tie_share(self) -> None:
        # Act: AWM on 8x H100 is priced learn-bound and ran 7% faster merged.
        plan = _plan_over(_awm_manifest(), "H100-80", "cuda_ipc")

        # Assert
        assert plan.merged_lora_sync is True
        assert plan.split is not None
        assert (plan.split.trainers, plan.split.engines) == (4, 2)

    def test_mtp_drafting_syncs_the_adapter(self) -> None:
        # Arrange: merged sync does not update MTP draft weights.
        raw = _sudoku_manifest()
        raw["algorithm"]["vllm_engine_args"] = {
            "speculative_config": {"method": "mtp", "num_speculative_tokens": 1}
        }

        # Act
        plan = _plan_over(raw, "A100-80", "cuda_ipc")

        # Assert
        assert plan.merged_lora_sync is False
        assert plan.split is not None
        assert not plan.split.merged_lora_sync
        assert plan.training.settings.merged_lora_sync is False

    def test_unknown_link_syncs_merged_weights_unpriced(self) -> None:
        # Act
        plan = _plan_over(_sudoku_manifest(), "A100-80", None)

        # Assert
        assert plan.merged_lora_sync is True
        assert plan.split is not None
        assert plan.split.sync_seconds == 0.0

    @pytest.mark.parametrize(
        ("sync_link", "merged"),
        [("cuda_ipc", True), ("rdma", False), ("tcp", False)],
    )
    def test_kept_counts_price_the_sync_mode(
        self, sync_link: SyncLink, merged: bool
    ) -> None:
        # Arrange: countdown at 4T4E on H100 is learn-bound.
        raw = _countdown_manifest(64)
        raw["training"]["training_gpus_per_agent"] = 4
        raw["training"]["rollout_engines_per_agent"] = 4
        raw["algorithm"]["vllm_config"] = {"tensor_parallel_size": 1}

        # Act
        plan = _plan_over(raw, "H100-80", sync_link, search_split=False)

        # Assert
        assert plan.split is None
        assert plan.merged_lora_sync is merged
        assert plan.training.settings.merged_lora_sync is merged

    @pytest.mark.parametrize(
        ("gib", "merged"),
        [(12.0, True), (11.0, False)],
    )
    @pytest.mark.parametrize("sync_link", [None, "cuda_ipc"])
    def test_searched_split_syncs_the_adapter_only_when_merged_does_not_fit(
        self, sync_link: SyncLink | None, gib: float, merged: bool
    ) -> None:
        # Arrange: one countdown trainer needs 10.86 GiB syncing merged
        # weights and 10.42 GiB syncing the adapter.
        manifest = validated_run_manifest(
            TrainingManifest.model_validate(_countdown_manifest(64))
        )

        # Act
        plan = resolve_run_memory(
            manifest,
            catalog_model(manifest.algorithm),
            _l4_with(gib),
            2,
            sync_link,
        )

        # Assert
        assert plan.split is not None
        assert (plan.split.trainers, plan.split.engines) == (1, 1)
        assert plan.merged_lora_sync is merged
        assert plan.split.merged_lora_sync is merged
        assert plan.training.settings.merged_lora_sync is merged

    def test_one_trainer_gpu_syncs_the_adapter_when_merged_does_not_fit(
        self,
    ) -> None:
        # Arrange: a kept trainer on its own GPU, with no engine GPU to price.
        raw = _countdown_manifest(64)
        raw["training"]["training_gpus_per_agent"] = 1
        raw["training"]["rollout_engines_per_agent"] = 1
        manifest = _manifest(raw)

        # Act
        plan = resolve_run_memory(
            manifest,
            catalog_model(manifest.algorithm),
            _l4_with(11.0),
            1,
            None,
            False,
        )

        # Assert
        assert plan.merged_lora_sync is False
        assert plan.training.settings.merged_lora_sync is False
        assert plan.training.breakdown.fits


PREQUANTIZED_EXPERT_LORA_MODELS = [
    "deepseek-ai/DeepSeek-V4-Flash-0731",
    "zai-org/GLM-5.3-Flash",
    "XiaomiMiMo/MiMo-V2.6-Flash-RL",
]


def _prequantized_manifest(
    hub_id: str, vllm_config: dict | None = None
) -> TrainingManifest:
    """CISPO LoRA on a pre-quantized MoE checkpoint, experts sharded over 8 trainers.

    :param hub_id: Catalogued pre-quantized checkpoint.
    :param vllm_config: Engine config; empty when ``None``.
    :return: Validated run manifest.
    """
    raw = _countdown_at(4096, vllm_config or {})
    raw["network"]["pretrained_model_name_or_path"] = hub_id
    raw["algorithm"].update(name="CISPO", batch_size=32, fsdp={"ep": 8})
    return validated_run_manifest(TrainingManifest.model_validate(raw))


class TestResolveRunMemoryPrequantizedCheckpoint:
    @pytest.mark.parametrize("hub_id", PREQUANTIZED_EXPERT_LORA_MODELS)
    def test_three_h100_nodes_serve_one_tp8_engine(self, hub_id: str) -> None:
        # Arrange
        manifest = _prequantized_manifest(hub_id)

        # Act
        plan = resolve_run_memory(
            manifest, catalog_model(manifest.algorithm), _h100_nodes(), 24
        )

        # Assert
        assert plan.split is not None
        assert (plan.split.trainers, plan.split.engines) == (16, 1)
        assert plan.split.engine_width == 8
        assert plan.merged_lora_sync is False

    @pytest.mark.parametrize("hub_id", PREQUANTIZED_EXPERT_LORA_MODELS)
    def test_vllm_default_budget_leaves_no_room_for_the_received_adapter(
        self, hub_id: str
    ) -> None:
        # Arrange: a 0.9 budget plus the received adapter and its expert copy
        # outgrow one H100.
        manifest = _prequantized_manifest(
            hub_id, vllm_config={"gpu_memory_utilization": 0.9}
        )

        # Act / Assert
        with pytest.raises(
            ValueError, match="a trainer and a rollout engine do not both fit"
        ):
            resolve_run_memory(
                manifest, catalog_model(manifest.algorithm), _h100_nodes(), 24
            )

    @pytest.mark.parametrize(
        "hub_id", ["deepseek-ai/DeepSeek-V4-Flash-0731", "zai-org/GLM-5.3-Flash"]
    )
    def test_bf16_sized_engine_leaves_no_split(
        self, hub_id: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        bundled = json.loads(supported_path(hub_id).read_text(encoding="utf-8"))
        del bundled["config"]["quantization_config"]
        monkeypatch.setattr(
            model_info, "_bundled", lambda name: {hub_id: bundled}[name]
        )
        manifest = _prequantized_manifest(hub_id)

        # Act / Assert
        with pytest.raises(
            ValueError, match="a trainer and a rollout engine do not both fit"
        ):
            resolve_run_memory(
                manifest, catalog_model(manifest.algorithm), _h100_nodes(), 24
            )


def _quoted(
    learn: float,
    generation: float,
    sync: float,
    merged: bool,
    transport: str,
) -> GpuSplit:
    return GpuSplit(
        trainers=2,
        engines=1,
        engine_width=2,
        rollout_batch_size=4,
        learn_seconds=learn,
        generation_seconds=generation,
        sync_seconds=sync,
        merged_lora_sync=merged,
        sync_transport=transport,
    )


def _qwen35_9b_manifest(
    max_output_tokens: int,
    max_turns: int,
    batch_size: int,
    fsdp: dict[str, int] | None = None,
) -> TrainingManifest:
    """CISPO on Qwen3.5-9B, 2 trainers and 1 TP2 engine, the measured A100 layout."""
    raw = {
        "algorithm": {
            "name": "CISPO",
            "batch_size": batch_size,
            "group_size": 4,
            "max_output_tokens": max_output_tokens,
            "fsdp": fsdp or {},
            "vllm_config": {
                "tensor_parallel_size": 2,
                "gpu_memory_utilization": 0.85,
            },
        },
        "environment": {
            "env_type": "rollout",
            "factory": "gem:make",
            "entrypoint": "game:GuessTheNumber-v0",
            "max_turns": max_turns,
            "max_reward": 1.0,
        },
        "network": {
            "pretrained_model_name_or_path": "Qwen/Qwen3.5-9B",
            "max_context_length": 4096,
            "lora_config": {
                "lora_r": 16,
                "lora_alpha": 32,
                "target_modules": "all-linear",
            },
        },
        "training": {
            "rollout_mode": "async",
            "pop_size": 1,
            "training_gpus_per_agent": 2,
            "rollout_engines_per_agent": 1,
        },
        "replay_buffer": {"kind": "llm", "drop_uninformative_groups": False},
    }
    return validated_run_manifest(TrainingManifest.model_validate(raw))


def _quote(plan: RunMemoryPlan, merged: bool, transport: str) -> GpuSplit:
    return next(
        split
        for split in plan.quotes
        if split.merged_lora_sync is merged and split.sync_transport == transport
    )


class TestQwen35NineBWeightSync:
    def test_short_generation_keeps_the_adapter(self) -> None:
        # Arrange: one short turn at batch 1. The two trainers split it at
        # TP 2, which nccl_reshard cannot refit from, and merged NIXL sync
        # costs more than merged generation gains.
        manifest = _qwen35_9b_manifest(32, 1, 1)
        device = _device("A100-80").model_copy(update={"gpus_per_node": 4})

        # Act
        plan = resolve_run_memory(
            manifest, catalog_model(manifest.algorithm), device, 4, "cuda_ipc", False
        )

        # Assert
        nixl = _quote(plan, True, "nixl")
        adapter = _quote(plan, False, "nixl")
        assert plan.training.fsdp is not None
        assert plan.training.fsdp.tp == 2
        assert {split.sync_transport for split in plan.quotes} == {"nixl"}
        assert nixl.learn_seconds == pytest.approx(0.471, abs=0.01)
        assert nixl.generation_seconds == pytest.approx(0.232, abs=0.01)
        assert nixl.sync_seconds == pytest.approx(0.726, abs=0.01)
        assert adapter.learn_seconds == pytest.approx(0.471, abs=0.01)
        assert adapter.generation_seconds == pytest.approx(0.470, abs=0.01)
        assert adapter.sync_seconds == pytest.approx(0.359, abs=0.005)
        assert adapter.synced_step_seconds < nixl.synced_step_seconds
        assert plan.merged_lora_sync is False
        assert plan.weight_sync_transport == "nixl"

    def test_long_generation_takes_nccl_reshard(self) -> None:
        # Arrange: two long turns. Merged generation pays for the sync, and
        # nccl_reshard is the faster merged transport.
        manifest = _qwen35_9b_manifest(1024, 2, 4)
        device = _device("A100-80").model_copy(update={"gpus_per_node": 4})

        # Act
        plan = resolve_run_memory(
            manifest, catalog_model(manifest.algorithm), device, 4, "cuda_ipc", False
        )

        # Assert
        nccl = _quote(plan, True, "nccl_reshard")
        adapter = _quote(plan, False, "nixl")
        assert nccl.learn_seconds == pytest.approx(4.189, abs=0.02)
        assert nccl.generation_seconds == pytest.approx(11.028, abs=0.05)
        assert nccl.sync_seconds == pytest.approx(0.306, abs=0.005)
        assert adapter.generation_seconds == pytest.approx(17.045, abs=0.05)
        assert adapter.sync_seconds == pytest.approx(0.168, abs=0.005)
        assert nccl.synced_step_seconds < adapter.synced_step_seconds
        assert plan.merged_lora_sync is True
        assert plan.weight_sync_transport == "nccl_reshard"

    def test_b200_takes_nccl_reshard_at_the_sm90_rates(self) -> None:
        # Arrange: 2 DDP trainers and 1 TP2 engine. B200 has no measured row,
        # so the refit uses the SM90 rates.
        manifest = _qwen35_9b_manifest(1024, 2, 4)
        device = _device("B200").model_copy(update={"gpus_per_node": 4})
        model = catalog_model(manifest.algorithm)
        workload = split_workload_from_manifest(manifest)

        # Act
        plan = resolve_run_memory(manifest, model, device, 4, "cuda_ipc", False)

        # Assert
        nccl = _quote(plan, True, "nccl_reshard")
        assert plan.training.fsdp is None
        assert (nccl.trainers, nccl.engines, nccl.engine_width) == (2, 1, 2)
        shards = (
            workload.trainer_shard_group
            if workload.trainer_shard_group > 1
            else nccl.trainers
        )
        expected = nccl_reshard_sync_seconds(
            model,
            workload.lora_target_scope,
            nccl.engines,
            nccl.engine_width,
            shards,
            False,
            SM90_NCCL_RESHARD_COSTS,
        )
        assert nccl.sync_seconds == expected
        assert plan.merged_lora_sync is True
        assert plan.weight_sync_transport == "nccl_reshard"

    def test_h200_takes_nccl_reshard_at_the_h100_rates(self) -> None:
        # Arrange: the H200 trainers shard FSDP, so the refit gathers every
        # misc weight on top of the measured H100 transfers.
        manifest = _qwen35_9b_manifest(1024, 2, 4)
        device = _device("H200").model_copy(update={"gpus_per_node": 4})

        # Act
        plan = resolve_run_memory(
            manifest, catalog_model(manifest.algorithm), device, 4, "cuda_ipc", False
        )

        # Assert
        nccl = _quote(plan, True, "nccl_reshard")
        assert plan.training.fsdp is not None
        assert nccl.sync_seconds == pytest.approx(0.312, abs=0.001)
        assert _quote(plan, True, "nixl").sync_seconds == pytest.approx(
            0.725, abs=0.001
        )
        assert plan.merged_lora_sync is True
        assert plan.weight_sync_transport == "nccl_reshard"

    def test_measured_layout_prices_the_measured_step_ratio(self) -> None:
        # Arrange: the A100 run, 64-token turns over 2 turns at batch 4, took
        # 2.75 s per step merged over nccl_reshard and 4.12 s adapter (1.50x).
        manifest = _qwen35_9b_manifest(64, 2, 4)
        device = _device("A100-80").model_copy(update={"gpus_per_node": 4})

        # Act
        plan = resolve_run_memory(
            manifest, catalog_model(manifest.algorithm), device, 4, "cuda_ipc", False
        )

        # Assert
        nccl = _quote(plan, True, "nccl_reshard")
        adapter = _quote(plan, False, "nixl")
        assert adapter.synced_step_seconds / nccl.synced_step_seconds == pytest.approx(
            1.504, abs=0.005
        )

    @pytest.mark.parametrize("degree", ["tp", "cp"])
    def test_trainer_model_parallel_is_not_offered_nccl_reshard(
        self, degree: str
    ) -> None:
        # Arrange: nccl_reshard refits only from data-parallel trainer shards.
        manifest = _qwen35_9b_manifest(1024, 2, 4, {degree: 2})

        # Act
        modes = weight_sync_modes(manifest, _device("A100-80"), "cuda_ipc")

        # Assert
        assert modes == (WeightSyncMode(True, "nixl"), WeightSyncMode(False, "nixl"))

    def test_packed_expert_lora_is_not_offered_nccl_reshard(self) -> None:
        # Arrange: Qwen3.6-35B-A3B adapts its packed expert parameters.
        manifest = _manifest(_awm_manifest())

        # Act
        supported = nccl_reshard_layout_supported(manifest)
        modes = weight_sync_modes(manifest, _device("A100-80"), "cuda_ipc")

        # Assert
        assert manifest.algorithm.lora_config.target_parameters == [
            "mlp.experts.down_proj",
            "mlp.experts.gate_up_proj",
        ]
        assert supported is False
        assert {mode.transport for mode in modes} == {"nixl"}

    def test_moe_is_not_offered_nccl_reshard(self) -> None:
        # Arrange: the measured rate is a dense model. Experts stay on NIXL.
        manifest = _manifest(_awm_manifest())

        # Act
        modes = weight_sync_modes(manifest, _device("A100-80"), "cuda_ipc")

        # Assert
        assert all(mode.transport != "nccl_reshard" for mode in modes)

    def test_rdma_does_not_offer_the_one_node_nccl_price(self) -> None:
        # Arrange
        manifest = _qwen35_9b_manifest(1024, 2, 4)

        # Act
        modes = weight_sync_modes(manifest, _device("A100-80"), "rdma")

        # Assert
        assert modes == (WeightSyncMode(True, "nixl"), WeightSyncMode(False, "nixl"))

    def test_a100_dense_offers_nccl_reshard_first(self) -> None:
        # Arrange
        manifest = _qwen35_9b_manifest(1024, 2, 4)

        # Act
        modes = weight_sync_modes(manifest, _device("A100-80"), "cuda_ipc")

        # Assert
        assert modes == (
            WeightSyncMode(True, "nccl_reshard"),
            WeightSyncMode(True, "nixl"),
            WeightSyncMode(False, "nixl"),
        )

    @pytest.mark.parametrize("gpu", ["B200", "H200", "H100-80"])
    def test_other_nvlink_gpus_offer_nccl_reshard_first(self, gpu: str) -> None:
        # Arrange
        manifest = _qwen35_9b_manifest(1024, 2, 4)

        # Act
        modes = weight_sync_modes(manifest, _device(gpu), "cuda_ipc")

        # Assert
        assert modes == (
            WeightSyncMode(True, "nccl_reshard"),
            WeightSyncMode(True, "nixl"),
            WeightSyncMode(False, "nixl"),
        )

    def test_pcie_gpu_offers_nixl_only(self) -> None:
        # Arrange
        manifest = _qwen35_9b_manifest(1024, 2, 4)

        # Act
        modes = weight_sync_modes(manifest, _device("L4"), "cuda_ipc")

        # Assert
        assert modes == (WeightSyncMode(True, "nixl"), WeightSyncMode(False, "nixl"))


def _h100_guess_the_number_manifest(hub_id: str) -> TrainingManifest:
    """CISPO on GuessTheNumber, 4 trainers and 2 TP2 engines, the measured 8x H100 layout."""
    raw = {
        "algorithm": {
            "name": "CISPO",
            "batch_size": 8,
            "group_size": 4,
            "max_output_tokens": 512,
            "update_epochs": 1,
            "beta": 0.0,
            "fsdp": {},
            "vllm_config": {
                "tensor_parallel_size": 2,
                "gpu_memory_utilization": 0.85,
                "max_num_seqs": 32,
            },
        },
        "environment": {
            "env_type": "rollout",
            "factory": "gem:make",
            "entrypoint": "game:GuessTheNumber-v0-easy",
            "max_turns": 2,
            "max_reward": 1.0,
        },
        "network": {
            "pretrained_model_name_or_path": hub_id,
            "max_context_length": 2048,
        },
        "training": {
            "rollout_mode": "async",
            "pop_size": 1,
            "rollout_batch_size": 16,
            "training_gpus_per_agent": 4,
            "rollout_engines_per_agent": 2,
        },
        "replay_buffer": {"kind": "llm", "max_rollout_version_lag": 2},
    }
    return validated_run_manifest(TrainingManifest.model_validate(raw))


def _h100_plan(hub_id: str) -> RunMemoryPlan:
    manifest = _h100_guess_the_number_manifest(hub_id)
    device = _device("H100-80").model_copy(update={"gpus_per_node": 8})
    return resolve_run_memory(
        manifest, catalog_model(manifest.algorithm), device, 8, "cuda_ipc", False
    )


class TestH100GuessTheNumberWeightSync:
    @pytest.mark.parametrize(
        ("hub_id", "transport", "fsdp"),
        [
            # Wall per step over steps 5-30, nccl_reshard / merged NIXL /
            # adapter: 2.54 / 2.62 / 3.36 s.
            ("Qwen/Qwen3.5-2B", "nccl_reshard", False),
            # 4.13 / 4.39 / 5.58 s.
            ("Qwen/Qwen3.5-4B", "nccl_reshard", False),
            # 5.33 / 5.36 / 6.03 s: either merged transport is the fastest.
            ("Qwen/Qwen3.5-9B", "nccl_reshard", True),
            # 13.84 / 19.95 / 21.05 s. DDP misses the underprediction buffer
            # with these NCCL groups, so nccl_reshard trains FSDP.
            ("Qwen/Qwen3.8-27B", "nccl_reshard", True),
            # Merged NIXL / adapter: 10.63 / 12.70 s; MoE has no nccl_reshard.
            ("Qwen/Qwen3.6-35B-A3B", "nixl", True),
        ],
        ids=["2b", "4b", "9b", "27b", "35b-a3b"],
    )
    def test_takes_the_measured_fastest_mode(
        self, hub_id: str, transport: str, fsdp: bool
    ) -> None:
        # Act
        plan = _h100_plan(hub_id)

        # Assert
        assert plan.split is None
        assert plan.merged_lora_sync is True
        assert plan.weight_sync_transport == transport
        assert (plan.training.fsdp is not None) is fsdp

    @pytest.mark.parametrize(
        ("hub_id", "merged", "transport", "measured", "tolerance"),
        [
            ("Qwen/Qwen3.5-2B", True, "nccl_reshard", 0.111, 0.02),
            ("Qwen/Qwen3.5-2B", True, "nixl", 0.169, 0.12),
            ("Qwen/Qwen3.5-2B", False, "nixl", 0.109, 0.03),
            ("Qwen/Qwen3.5-4B", True, "nccl_reshard", 0.179, 0.02),
            ("Qwen/Qwen3.5-4B", True, "nixl", 0.320, 0.12),
            ("Qwen/Qwen3.5-4B", False, "nixl", 0.141, 0.03),
            ("Qwen/Qwen3.5-9B", True, "nccl_reshard", 0.353, 0.14),
            ("Qwen/Qwen3.5-9B", True, "nixl", 0.729, 0.12),
            ("Qwen/Qwen3.5-9B", False, "nixl", 0.294, 0.35),
            ("Qwen/Qwen3.8-27B", True, "nixl", 2.607, 0.12),
            ("Qwen/Qwen3.8-27B", False, "nixl", 0.328, 0.03),
            ("Qwen/Qwen3.6-35B-A3B", True, "nixl", 0.509, 0.12),
            ("Qwen/Qwen3.6-35B-A3B", False, "nixl", 0.512, 0.35),
        ],
    )
    def test_sync_quotes_match_the_measured_syncs(
        self,
        hub_id: str,
        merged: bool,
        transport: str,
        measured: float,
        tolerance: float,
    ) -> None:
        # Act: medians of weight_sync_s over steps 5-30, each at the layout the
        # run trained. FSDP trainers are priced with one gather rate, so their
        # adapter and nccl_reshard quotes are looser.
        quote = _quote(_h100_plan(hub_id), merged, transport)

        # Assert
        assert quote.sync_seconds == pytest.approx(measured, rel=tolerance)

    @pytest.mark.parametrize(
        ("hub_id", "expected", "rel", "abs_tol"),
        [
            ("Qwen/Qwen3.5-2B", 1.27, 0.17, None),
            ("Qwen/Qwen3.5-4B", 1.30, 0.17, None),
            ("Qwen/Qwen3.5-9B", 1.24, 0.17, None),
            # The two merged walls differ by more than the sync, so the
            # generation ratio that fits both is wider than the token rate.
            ("Qwen/Qwen3.8-27B", 1.633, None, 0.01),
            ("Qwen/Qwen3.6-35B-A3B", 1.23, 0.17, None),
        ],
    )
    def test_merged_engines_generate_at_the_measured_token_rate(
        self, hub_id: str, expected: float, rel: float | None, abs_tol: float | None
    ) -> None:
        # Arrange
        plan = _h100_plan(hub_id)

        # Act
        speedup = (
            _quote(plan, False, "nixl").generation_seconds
            / _quote(plan, True, "nixl").generation_seconds
        )

        # Assert
        if abs_tol is None:
            assert speedup == pytest.approx(expected, rel=rel)
        else:
            assert speedup == pytest.approx(expected, abs=abs_tol)

    @pytest.mark.parametrize(
        ("hub_id", "merged", "transport", "expected"),
        [
            ("Qwen/Qwen3.5-2B", True, "nccl_reshard", 2.768),
            ("Qwen/Qwen3.5-2B", True, "nixl", 2.824),
            ("Qwen/Qwen3.5-2B", False, "nixl", 3.053),
            ("Qwen/Qwen3.5-4B", True, "nccl_reshard", 4.055),
            ("Qwen/Qwen3.5-4B", True, "nixl", 4.235),
            ("Qwen/Qwen3.5-4B", False, "nixl", 4.777),
            ("Qwen/Qwen3.5-9B", True, "nccl_reshard", 5.290),
            ("Qwen/Qwen3.5-9B", True, "nixl", 5.617),
            ("Qwen/Qwen3.5-9B", False, "nixl", 6.887),
            ("Qwen/Qwen3.8-27B", True, "nccl_reshard", 15.630),
            ("Qwen/Qwen3.8-27B", True, "nixl", 17.071),
            ("Qwen/Qwen3.8-27B", False, "nixl", 23.917),
            ("Qwen/Qwen3.6-35B-A3B", True, "nixl", 12.143),
            ("Qwen/Qwen3.6-35B-A3B", False, "nixl", 13.975),
        ],
    )
    def test_step_quotes_match_the_measured_walls(
        self, hub_id: str, merged: bool, transport: str, expected: float
    ) -> None:
        # Act
        quote = _quote(_h100_plan(hub_id), merged, transport)

        # Assert
        assert quote.synced_step_seconds == pytest.approx(expected, abs=0.01)

    def test_2b_adapter_step_is_not_within_10_percent_of_nccl_reshard(self) -> None:
        # Act
        plan = _h100_plan("Qwen/Qwen3.5-2B")
        nccl = _quote(plan, True, "nccl_reshard")
        adapter = _quote(plan, False, "nixl")

        # Assert
        assert adapter.synced_step_seconds > 1.10 * nccl.synced_step_seconds

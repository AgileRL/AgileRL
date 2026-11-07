# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""An async member's trainer / engine split is chosen from the throughput model."""

from __future__ import annotations

import json
import math

import pytest

from agilerl.arena.memory.kv_cache import ROLLOUT_ENGINE_MAX_NUM_BATCHED_TOKENS
from agilerl.arena.memory.manifest import (
    GPU_CATALOGUE,
    GpuInfo,
    lookup_gpu,
    training_memory_from_manifest,
    unsupported_gpu_reason,
    validated_run_manifest,
)
from agilerl.arena.memory.resolve.batch import catalog_model
from agilerl.arena.memory.resolve.decode_costs import (
    MERGED_DECODE_SPEEDUP,
    MERGED_MOE_PREFILL_SPEEDUP,
)
from agilerl.arena.memory.resolve.fsdp import (
    TrainingMemoryPlan,
    TrainingMemoryUnfitError,
    learn_cost,
)
from agilerl.arena.memory.resolve.learn_costs import learn_costs
from agilerl.arena.memory.resolve.split import (
    IDLE_GPU_TIE_SHARE,
    MULTI_TURN_ROW_FILL,
    MULTI_TURN_TURN_DECODE_TOKENS,
    MULTI_TURN_TURN_SHARE,
    NGRAM_GENERATION_SPEEDUP,
    SINGLE_TURN_DECODE_SHARE,
    SINGLE_TURN_PROMPT_FILL,
    SINGLE_TURN_REASONING_DECODE_SHARE,
    TIED_DROP_GENERATION_FACTOR,
    GpuSplit,
    RunMemoryPlan,
    SplitWorkload,
    active_matmul_params,
    choose_engine_width,
    choose_gpu_split,
    engine_prefill_rate,
    engine_steps,
    fullest_split_within,
    kept_split_merged_lora_sync,
    merged_lora_sync_candidates,
    pick_weight_sync,
    resident_sequences,
    resolve_run_memory,
    split_workload_from_manifest,
    tensor_parallel_seconds,
    uncached_prefill_tokens,
)
from agilerl.arena.memory.resolve.weight_sync import SyncLink, merged_sync_seconds
from agilerl.arena.memory.specs import DeviceSpec, ModelArch, ModelSpec
from agilerl.arena.models import model_info
from agilerl.arena.models.manifest import TrainingManifest
from agilerl.arena.models.model_info import SUPPORTED_MODEL_INFO, supported_path


def _countdown_manifest(batch_size: int = 60) -> dict:
    """Single-turn GRPO on a dataset, shaped like speed benchmark A."""
    return {
        "algorithm": {
            "name": "GRPO",
            "batch_size": batch_size,
            "group_size": 8,
            "max_output_tokens": 1024,
            "fsdp": {},
        },
        "environment": {
            "env_type": "rollout",
            "dataset": "dataset.json",
            "rubric_file_path": "countdown_reward.py",
            "prompt_template": {
                "system_prompt": "Answer in <answer> </answer> tags.",
                "user_prompt": "Make {answer} from {question}.",
            },
            "max_reward": 2.0,
        },
        "network": {
            "pretrained_model_name_or_path": "Qwen/Qwen3-4B",
            "max_context_length": 1536,
            "lora_config": {"lora_r": 16, "lora_alpha": 32},
        },
        "training": {
            "rollout_mode": "async",
            "rollout_engines_per_agent": "auto",
            "pop_size": 1,
        },
        # The throughput fits trained every generated group.
        "replay_buffer": {"kind": "llm", "drop_uninformative_groups": False},
    }


def _oversize_countdown_manifest(batch_size: int) -> dict:
    """Countdown on Qwen3.8-27B LoRA; its training fits L4s at 6 trainers, not 1 to 5."""
    raw = _countdown_manifest(batch_size)
    raw["network"]["pretrained_model_name_or_path"] = "Qwen/Qwen3.8-27B"
    return raw


def _sudoku_manifest() -> dict:
    """Multi-turn CISPO on an env, shaped like speed benchmark B."""
    return {
        "algorithm": {
            "name": "CISPO",
            "batch_size": 12,
            "group_size": 8,
            "max_output_tokens": 1024,
            "fsdp": {},
        },
        "environment": {
            "env_type": "rollout",
            "factory": "gem:make",
            "entrypoint": "game:Sudoku-v0-hard",
            "max_turns": 50,
            "max_reward": 1.0,
        },
        "network": {
            "pretrained_model_name_or_path": "Qwen/Qwen3-4B",
            "max_context_length": 16384,
            "lora_config": {"lora_r": 16, "lora_alpha": 32},
        },
        "training": {
            "rollout_mode": "async",
            "rollout_engines_per_agent": "auto",
            "pop_size": 1,
        },
        # The throughput fits trained every generated group.
        "replay_buffer": {"kind": "llm", "drop_uninformative_groups": False},
    }


def _awm_manifest() -> dict:
    """Multi-turn CISPO LoRA on Qwen3.6-35B-A3B, shaped like the AWM training config."""
    return {
        "algorithm": {
            "name": "CISPO",
            "batch_size": 64,
            "group_size": 16,
            "mini_batch_size": 64,
            "max_output_tokens": 2048,
        },
        "environment": {
            "env_type": "rollout",
            "env_image": "env:latest",
            "modality": "text",
            "max_turns": 20,
        },
        "network": {
            "pretrained_model_name_or_path": "Qwen/Qwen3.6-35B-A3B",
            "max_context_length": 32768,
            "lora_config": {"lora_r": 8, "lora_alpha": 16},
        },
        "training": {
            "rollout_mode": "async",
            "rollout_engines_per_agent": "auto",
            "pop_size": 1,
        },
        # The throughput fits trained every generated group.
        "replay_buffer": {"kind": "llm", "drop_uninformative_groups": False},
    }


def _sft_manifest() -> TrainingManifest:
    return TrainingManifest.model_validate(
        {
            "algorithm": {"name": "SFT", "batch_size": 8},
            "environment": {"dataset": "openai/gsm8k"},
            "network": {
                "pretrained_model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct",
                "max_context_length": 512,
            },
            "training": {"max_steps": 100},
        }
    )


def _device(name: str = "A100-80") -> DeviceSpec:
    gpu = lookup_gpu(name)
    assert gpu is not None
    return gpu.device_spec()


def _manifest(raw: dict, engine_width: int = 1) -> TrainingManifest:
    raw["algorithm"]["vllm_config"] = {"tensor_parallel_size": engine_width}
    return validated_run_manifest(TrainingManifest.model_validate(raw))


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
        lambda count: plan.learn_cost_ratio if count == trainers else None,
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
        raw["algorithm"]["fsdp"] = fsdp
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
        ],
        ids=[
            "tensor-parallel",
            "expert-parallel",
            "shard-group",
            "tensor-and-shard",
            "all-three",
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
        raw = _countdown_manifest(60)
        raw["network"]["pretrained_model_name_or_path"] = "Qwen/Qwen3.8-27B"
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


def _workload(raw: dict, **updates: object) -> SplitWorkload:
    return split_workload_from_manifest(_manifest(raw)).model_copy(update=updates)


def _learn_bound_split(
    workload: SplitWorkload, trainers: int, device_name: str = "A100-80"
) -> GpuSplit:
    """The 8-GPU Qwen3-4B split with ``trainers`` trainers and every other GPU an engine.

    A learn step far slower than generation ties every engine count, and the
    tie goes to no idle GPUs.
    """
    split = choose_gpu_split(
        catalog_model(_manifest(_sudoku_manifest()).algorithm),
        _device(device_name),
        workload,
        8,
        1,
        lambda count: 1e9 if count == trainers else None,
    )
    assert split is not None
    assert split.engines == 8 - trainers
    return split


class TestChooseGpuSplit:
    def test_generation_covers_every_cycle_a_learn_step_needs(self) -> None:
        # Arrange: a pinned 64-group cycle; a 128-group learn step needs two.
        one_cycle = _workload(_countdown_manifest(64), rollout_batch_size=64)
        two_cycles = one_cycle.model_copy(update={"batch_size": 128})

        # Act
        single = _learn_bound_split(one_cycle, 4)
        double = _learn_bound_split(two_cycles, 4)

        # Assert
        assert single.rollout_batch_size == double.rollout_batch_size == 64
        assert double.generation_seconds == pytest.approx(
            2 * single.generation_seconds, rel=1e-12
        )

    def test_merged_sync_over_tcp_takes_fewer_engines(self) -> None:
        # Arrange: merged Qwen3-4B weights reach each engine at 0.4 GB/s.
        model = catalog_model(_manifest(_sudoku_manifest()).algorithm)
        unpriced = _workload(_sudoku_manifest(), merged_lora_sync=True)
        over_tcp = unpriced.model_copy(update={"sync_link": "tcp"})

        # Act
        by_step = choose_gpu_split(model, _device(), unpriced, 8, 1, lambda _: 1.0)
        by_synced_step = choose_gpu_split(
            model, _device(), over_tcp, 8, 1, lambda _: 1.0
        )

        # Assert: four engines generate faster but take twice the sync.
        assert by_step is not None
        assert by_synced_step is not None
        assert (by_step.trainers, by_step.engines) == (4, 4)
        assert (by_synced_step.trainers, by_synced_step.engines) == (6, 2)
        assert by_synced_step.step_seconds > by_step.step_seconds
        four_engine_sync = merged_sync_seconds(
            model, unpriced.lora_target_scope, 0, 4, 1, True, "tcp"
        )
        assert (
            by_synced_step.synced_step_seconds < by_step.step_seconds + four_engine_sync
        )

    def test_dropped_tied_groups_lengthen_generation_per_trained_step(self) -> None:
        # Arrange
        trains_every_group = _workload(_countdown_manifest(64), rollout_batch_size=64)
        drops_tied = trains_every_group.model_copy(
            update={"generated_per_trained": TIED_DROP_GENERATION_FACTOR}
        )

        # Act
        every_split = _learn_bound_split(trains_every_group, 4)
        drop_split = _learn_bound_split(drops_tied, 4)

        # Assert
        assert drop_split.learn_seconds == every_split.learn_seconds
        assert drop_split.generation_seconds == pytest.approx(
            TIED_DROP_GENERATION_FACTOR * every_split.generation_seconds, rel=1e-12
        )

    def test_cycle_rounded_past_the_batch_covers_part_of_the_next_step(self) -> None:
        # Arrange: three trainers and five engines round 12 groups up to 15.
        rounded_up = _workload(_sudoku_manifest())
        exact = rounded_up.model_copy(update={"batch_size": 15})

        # Act
        rounded_split = _learn_bound_split(rounded_up, 3)
        exact_split = _learn_bound_split(exact, 3)

        # Assert
        assert rounded_split.rollout_batch_size == exact_split.rollout_batch_size == 15
        assert rounded_split.generation_seconds == pytest.approx(
            12 / 15 * exact_split.generation_seconds, rel=1e-12
        )

    def test_fp8_weights_shorten_h100_generation(self) -> None:
        # Arrange
        bf16 = _workload(_countdown_manifest(64))
        fp8 = bf16.model_copy(update={"weight_quantization": "fp8"})

        # Act
        bf16_split = _learn_bound_split(bf16, 4, "H100-80")
        fp8_split = _learn_bound_split(fp8, 4, "H100-80")

        # Assert
        assert fp8_split.generation_seconds < 0.95 * bf16_split.generation_seconds

    def test_fp8_weights_lengthen_a100_generation(self) -> None:
        # Arrange: Marlin dequantizes fp8 in the GEMM below sm89.
        bf16 = _workload(_countdown_manifest(64))
        fp8 = bf16.model_copy(update={"weight_quantization": "fp8"})

        # Act
        bf16_split = _learn_bound_split(bf16, 4)
        fp8_split = _learn_bound_split(fp8, 4)

        # Assert
        assert fp8_split.generation_seconds > bf16_split.generation_seconds

    def test_unmeasured_gpu_prices_fp8_no_slower_than_bf16(self) -> None:
        # Arrange
        bf16 = _workload(_countdown_manifest(64))
        fp8 = bf16.model_copy(update={"weight_quantization": "fp8"})

        # Act
        bf16_split = _learn_bound_split(bf16, 4, "L4")
        fp8_split = _learn_bound_split(fp8, 4, "L4")

        # Assert
        assert fp8_split.generation_seconds <= bf16_split.generation_seconds

    def test_multi_turn_drafting_divides_decoding_by_the_ngram_speedup(
        self,
    ) -> None:
        # Arrange: Qwen3-4B drafts multi-turn episodes by default; decoding the
        # whole row leaves no prompt to prefill.
        drafted = _workload(_sudoku_manifest())
        drafted = drafted.model_copy(update={"decode_tokens": drafted.row_tokens})
        plain = drafted.model_copy(update={"speculative_drafting": False})

        # Act
        drafted_split = _learn_bound_split(drafted, 3, "H100-80")
        plain_split = _learn_bound_split(plain, 3, "H100-80")

        # Assert
        assert drafted_split.generation_seconds == pytest.approx(
            plain_split.generation_seconds / NGRAM_GENERATION_SPEEDUP, rel=1e-12
        )

    def test_multi_turn_min_p_generation_runs_without_drafting(self) -> None:
        # Arrange
        plain = _workload(_sudoku_manifest(), speculative_drafting=False)
        raw = _sudoku_manifest()
        raw["algorithm"]["min_p"] = 0.05
        min_p = _workload(raw)

        # Act
        plain_split = _learn_bound_split(plain, 3, "H100-80")
        min_p_split = _learn_bound_split(min_p, 3, "H100-80")

        # Assert
        assert min_p_split.generation_seconds == plain_split.generation_seconds

    def test_multi_turn_logit_bias_generation_runs_without_drafting(self) -> None:
        # Arrange
        plain = _workload(_sudoku_manifest(), speculative_drafting=False)
        biased = _workload(_sudoku_manifest(), logit_bias=True)

        # Act
        plain_split = _learn_bound_split(plain, 3, "H100-80")
        biased_split = _learn_bound_split(biased, 3, "H100-80")

        # Assert
        assert biased_split.generation_seconds == plain_split.generation_seconds

    def test_a_vision_model_workload_carries_logit_bias(self) -> None:
        # Arrange
        raw = _sudoku_manifest()
        raw["network"]["pretrained_model_name_or_path"] = "google/gemma-4-E4B-it"

        # Act
        workload = _workload(raw)

        # Assert
        assert workload.logit_bias is True
        assert _workload(_sudoku_manifest()).logit_bias is False

    def test_prompt_heavy_generation_runs_at_the_dense_prefill_rate(self) -> None:
        # Arrange: 8191 prompt tokens (8192 in whole KV blocks) and one decoded
        # per episode, 128 episodes per engine; its decode steps are under 0.5%
        # of the prefill.
        workload = _workload(
            _countdown_manifest(64),
            row_tokens=8192,
            decode_tokens=1,
            merged_lora_sync=False,
        )
        arch = _catalog_spec("Qwen/Qwen3-4B").arch

        # Act
        split = _learn_bound_split(workload, 4)

        # Assert
        rate = 0.4 * 312e12 / (2 * active_matmul_params(arch))
        assert split.generation_seconds == pytest.approx(128 * 8192 / rate, rel=5e-3)

    def test_merged_prompt_heavy_generation_runs_at_the_merged_dense_prefill_rate(
        self,
    ) -> None:
        # Arrange: the prompt-heavy cycle above on engines with LoRA off.
        workload = _workload(
            _countdown_manifest(64),
            row_tokens=8192,
            decode_tokens=1,
            merged_lora_sync=True,
        )
        arch = _catalog_spec("Qwen/Qwen3-4B").arch

        # Act
        split = _learn_bound_split(workload, 4)

        # Assert
        rate = MERGED_DECODE_SPEEDUP * 0.4 * 312e12 / (2 * active_matmul_params(arch))
        assert split.generation_seconds == pytest.approx(128 * 8192 / rate, rel=5e-3)

    def test_merged_engine_decodes_at_the_merged_speedup(self) -> None:
        # Arrange: every row token is decoded, so the cycle has no prompt to prefill.
        lora = _workload(_countdown_manifest(64), merged_lora_sync=False)
        lora = lora.model_copy(update={"decode_tokens": lora.row_tokens})
        merged = lora.model_copy(update={"merged_lora_sync": True})

        # Act
        lora_split = _learn_bound_split(lora, 4)
        merged_split = _learn_bound_split(merged, 4)

        # Assert
        assert merged_split.generation_seconds == pytest.approx(
            lora_split.generation_seconds / MERGED_DECODE_SPEEDUP, rel=1e-12
        )

    def test_engine_without_room_for_one_sequence_never_finishes(self) -> None:
        # Arrange: 5% of an A100-80 is under Qwen3-4B's bf16 weights.
        workload = _workload(_countdown_manifest(64), gpu_memory_utilization=0.05)

        # Act
        split = _learn_bound_split(workload, 4)

        # Assert
        assert split.generation_seconds == math.inf

    def test_trainer_counts_off_the_trainer_multiple_are_skipped(self) -> None:
        # Arrange
        model = catalog_model(_manifest(_countdown_manifest()).algorithm)
        any_count = _workload(_countdown_manifest(120), merged_lora_sync=False)
        even = any_count.model_copy(update={"trainer_multiple": 2})

        # Act
        any_split = choose_gpu_split(model, _device(), any_count, 8, 1, lambda _: 1.0)
        even_split = choose_gpu_split(model, _device(), even, 8, 1, lambda _: 1.0)

        # Assert
        assert any_split is not None
        assert even_split is not None
        assert any_split.trainers == 5
        assert even_split.trainers == 4

    def test_single_turn_generation_ignores_drafting(self) -> None:
        # Arrange
        plain = _workload(_countdown_manifest(64))
        drafted = plain.model_copy(update={"speculative_drafting": True})

        # Act
        plain_split = _learn_bound_split(plain, 4, "H100-80")
        drafted_split = _learn_bound_split(drafted, 4, "H100-80")

        # Assert
        assert drafted_split.generation_seconds == plain_split.generation_seconds

    def test_tensor_parallel_prefill_pays_each_steps_all_reduce(self) -> None:
        # Arrange: 8191 prompt tokens (8192 in whole KV blocks) and one decoded
        # per episode, 256 episodes on each of two TP2 engines; its decode
        # steps are under 0.5% of the prefill.
        workload = _workload(
            _countdown_manifest(64),
            row_tokens=8192,
            decode_tokens=1,
            merged_lora_sync=False,
        )
        model = _catalog_spec("Qwen/Qwen3-4B")
        device = _device()

        # Act
        split = choose_gpu_split(
            catalog_model(_manifest(_countdown_manifest(64)).algorithm),
            device,
            workload,
            8,
            2,
            lambda count: 1e9 if count == 4 else None,
        )

        # Assert
        assert split is not None
        assert split.engines == 2
        step_tokens = ROLLOUT_ENGINE_MAX_NUM_BATCHED_TOKENS
        compute = 2 * active_matmul_params(model.arch) / (0.4 * 312e12)
        all_reduce = tensor_parallel_seconds(model, device, 2, step_tokens)
        rate = 2 / (compute + 2 * all_reduce / step_tokens)
        assert split.generation_seconds == pytest.approx(256 * 8192 / rate, rel=5e-3)

    def test_multi_turn_cycle_holds_only_the_resident_episodes(self) -> None:
        # Arrange: two TP2 engines each hold 157 episodes at the final context,
        # 16 groups.
        workload = _awm_workload(2)

        # Act
        split = _awm_split(workload, 2)

        # Assert
        assert split.engines == 2
        assert split.rollout_batch_size == 16

    def test_multi_turn_cycle_keeps_a_group_per_trainer_and_engine(self) -> None:
        # Arrange: one resident sequence per engine holds no whole group.
        workload = _awm_workload(2).model_copy(update={"max_num_seqs": 1})

        # Act
        split = _awm_split(workload, 2)

        # Assert
        assert split.rollout_batch_size == math.lcm(4, 2)

    def test_single_turn_cycle_ignores_the_resident_episodes(self) -> None:
        # Arrange
        workload = _workload(_countdown_manifest(64), max_num_seqs=1)

        # Act
        split = _learn_bound_split(workload, 4)

        # Assert
        assert split.rollout_batch_size == 64

    def test_multi_turn_cycle_past_the_resident_episodes_keeps_its_prefix(
        self,
    ) -> None:
        # Arrange: 64 groups put 512 episodes on each TP2 engine, which keeps
        # 157 resident; 24 groups put 192. Neither cycle's KV pool has room
        # for a step above 8192.
        held = _awm_workload(2).model_copy(
            update={"rollout_batch_size": 24, "merged_lora_sync": False}
        )
        oversubscribed = held.model_copy(update={"rollout_batch_size": 64})

        # Act
        held_split = _awm_split(held, 2)
        oversubscribed_split = _awm_split(oversubscribed, 2)

        # Assert
        assert held_split.rollout_batch_size == 24
        assert oversubscribed_split.rollout_batch_size == 64
        assert held_split.generation_seconds == pytest.approx(481.36, rel=1e-3)
        assert oversubscribed_split.generation_seconds == pytest.approx(
            451.79, rel=1e-3
        )


def _awm_workload(engine_width: int) -> SplitWorkload:
    return split_workload_from_manifest(_manifest(_awm_manifest(), engine_width))


def _awm_model() -> ModelSpec:
    return catalog_model(_manifest(_awm_manifest()).algorithm)


def _awm_split(workload: SplitWorkload, engine_width: int) -> GpuSplit:
    """The 8-GPU H100 Qwen3.6-35B-A3B split with four learn-bound trainers."""
    split = choose_gpu_split(
        _awm_model(),
        _device("H100-80"),
        workload,
        8,
        engine_width,
        lambda count: 1e9 if count == 4 else None,
    )
    assert split is not None
    return split


class TestResidentSequences:
    @pytest.mark.parametrize(("engine_width", "expected"), [(2, 157), (4, 264)])
    def test_hybrid_engine_holds_its_pool_over_one_sequences_blocks(
        self, engine_width: int, expected: int
    ) -> None:
        # Arrange: Qwen3.6-35B-A3B multi-turn episodes at a 15074-token final
        # context in 85% of an H100 engine's pool.
        workload = _awm_workload(engine_width)

        # Act
        resident = resident_sequences(
            _awm_model(),
            _device("H100-80"),
            workload,
            engine_width,
            ROLLOUT_ENGINE_MAX_NUM_BATCHED_TOKENS,
        )

        # Assert
        assert resident == expected

    @pytest.mark.parametrize(
        ("engine_width", "expected"),
        [(2, (157, 152, 144, 127)), (4, (264, 259, 249, 229))],
    )
    def test_larger_steps_hold_fewer_sequences(
        self, engine_width: int, expected: tuple[int, ...]
    ) -> None:
        # Arrange
        workload = _awm_workload(engine_width)
        steps = (8192, 16384, 32768, 65536)

        # Act
        residents = tuple(
            resident_sequences(
                _awm_model(), _device("H100-80"), workload, engine_width, step
            )
            for step in steps
        )

        # Assert
        assert residents == expected

    def test_replicated_kv_heads_hold_no_more_sequences_per_rank(self) -> None:
        # Arrange: Qwen2.5-0.5B has 2 KV heads, so a TP4 rank holds one of them.
        raw = _countdown_manifest(64)
        raw["network"]["pretrained_model_name_or_path"] = "Qwen/Qwen2.5-0.5B-Instruct"
        workload = _workload(raw)
        model = catalog_model(_manifest(raw).algorithm)
        device = _device("H100-80")

        step = ROLLOUT_ENGINE_MAX_NUM_BATCHED_TOKENS

        # Act
        tp2 = resident_sequences(model, device, workload, 2, step)
        tp4 = resident_sequences(model, device, workload, 4, step)

        # Assert
        assert tp2 < tp4 < 1.01 * tp2

    def test_pinned_max_num_seqs_caps_the_fit(self) -> None:
        # Arrange
        workload = _awm_workload(4).model_copy(update={"max_num_seqs": 64})

        # Act
        resident = resident_sequences(
            _awm_model(),
            _device("H100-80"),
            workload,
            4,
            ROLLOUT_ENGINE_MAX_NUM_BATCHED_TOKENS,
        )

        # Assert
        assert resident == 64

    def test_engine_without_kv_room_holds_none(self) -> None:
        # Arrange: 5% of an H100 is under a TP4 rank's weights.
        workload = _awm_workload(4).model_copy(update={"gpu_memory_utilization": 0.05})

        # Act
        resident = resident_sequences(
            _awm_model(),
            _device("H100-80"),
            workload,
            4,
            ROLLOUT_ENGINE_MAX_NUM_BATCHED_TOKENS,
        )

        # Assert
        assert resident == 0


class TestEngineSteps:
    def test_moe_engine_may_raise_the_default_to_every_fitted_step(self) -> None:
        # Act
        steps = engine_steps(_awm_model(), _awm_workload(2))

        # Assert
        assert steps == (8192, 16384, 32768, 65536)

    def test_dense_engine_keeps_the_default(self) -> None:
        # Arrange
        raw = _sudoku_manifest()

        # Act
        steps = engine_steps(catalog_model(_manifest(raw).algorithm), _workload(raw))

        # Assert
        assert steps == (ROLLOUT_ENGINE_MAX_NUM_BATCHED_TOKENS,)

    def test_pinned_step_is_the_only_one(self) -> None:
        # Arrange
        workload = _awm_workload(2).model_copy(update={"max_num_batched_tokens": 4096})

        # Act
        steps = engine_steps(_awm_model(), workload)

        # Assert
        assert steps == (4096,)


class TestChooseGpuSplitEngineStep:
    @pytest.mark.parametrize(("engine_width", "expected"), [(2, 32768), (4, 16384)])
    def test_moe_engines_run_the_largest_step_that_holds_the_cycle(
        self, engine_width: int, expected: int
    ) -> None:
        # Arrange: 16 groups put 128 episodes on each TP2 engine, which holds
        # 144 at 32768 and 127 at 65536; 16 groups put 256 on one TP4 engine,
        # which holds 259 at 16384 and 249 at 32768.
        workload = _awm_workload(engine_width)

        # Act
        split = _awm_split(workload, engine_width)

        # Assert
        assert split.max_num_batched_tokens == expected

    def test_cycle_past_the_resident_episodes_keeps_the_default_step(self) -> None:
        # Arrange: 512 episodes on each TP2 engine, over 157 resident.
        workload = _awm_workload(2).model_copy(update={"rollout_batch_size": 64})

        # Act
        split = _awm_split(workload, 2)

        # Assert
        assert split.max_num_batched_tokens is None

    def test_turn_that_cannot_fill_a_raised_step_keeps_the_default(self) -> None:
        # Arrange: two single-episode groups per TP2 engine prefill 2 x 4160
        # tokens a turn, under a 16384 step.
        workload = _awm_workload(2).model_copy(
            update={"group_size": 1, "rollout_batch_size": 4, "per_rank_multiple": 1}
        )

        # Act
        split = _awm_split(workload, 2)

        # Assert
        assert split.max_num_batched_tokens is None

    def test_pinned_step_runs_as_set(self) -> None:
        # Arrange
        searched = _awm_workload(2)
        pinned = searched.model_copy(update={"max_num_batched_tokens": 32768})

        # Act
        searched_split = _awm_split(searched, 2)
        pinned_split = _awm_split(pinned, 2)

        # Assert
        assert searched_split.max_num_batched_tokens == 32768
        assert pinned_split.max_num_batched_tokens is None
        assert pinned_split.rollout_batch_size == searched_split.rollout_batch_size
        assert pinned_split.generation_seconds == searched_split.generation_seconds

    def test_dense_engines_keep_the_default_step(self) -> None:
        # Act
        split = _learn_bound_split(_workload(_sudoku_manifest()), 3, "H100-80")

        # Assert
        assert split.max_num_batched_tokens is None


class TestEnginePrefillRate:
    @pytest.mark.parametrize(
        ("width", "step", "efficiency"),
        [
            (2, 4096, 0.18),
            (2, 8192, 0.18),
            (2, 32768, 0.257),
            (2, 40000, 0.257),
            (4, 8192, 0.10),
            (4, 16384, 0.158),
            (1, 16384, 0.215),
            (8, 65536, 0.166),
        ],
        ids=[
            "tp2-below-fit",
            "tp2-8k",
            "tp2-32k",
            "tp2-above-fit",
            "tp4-8k",
            "tp4-16k",
            "tp1-takes-tp2",
            "tp8-takes-tp4",
        ],
    )
    def test_moe_prefill_runs_at_the_nearest_width_largest_fitted_step(
        self, width: int, step: int, efficiency: float
    ) -> None:
        # Arrange
        model = _awm_model()
        device = _device("H100-80")
        assert device.peak_bf16_flops is not None

        # Act
        rate = engine_prefill_rate(
            model, device, device.peak_bf16_flops, width, step, False
        )

        # Assert
        compute = (
            2 * active_matmul_params(model.arch) / (efficiency * device.peak_bf16_flops)
        )
        all_reduce = tensor_parallel_seconds(model, device, width, step)
        assert rate == pytest.approx(
            width / (compute + width * all_reduce / step), rel=1e-12
        )

    def test_merged_moe_engine_computes_at_the_merged_prefill_speedup(self) -> None:
        # Arrange
        model = _awm_model()
        device = _device("H100-80")
        assert device.peak_bf16_flops is not None

        # Act
        rate = engine_prefill_rate(model, device, device.peak_bf16_flops, 2, 8192, True)

        # Assert
        compute = (
            2
            * active_matmul_params(model.arch)
            / (0.18 * MERGED_MOE_PREFILL_SPEEDUP * device.peak_bf16_flops)
        )
        all_reduce = tensor_parallel_seconds(model, device, 2, 8192)
        assert rate == pytest.approx(2 / (compute + 2 * all_reduce / 8192), rel=1e-12)


class TestResolveRunMemory:
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


class TestSplitWorkloadFromManifest:
    def test_single_turn_reasoning_model_decodes_most_of_the_output_cap(
        self,
    ) -> None:
        # Act
        workload = split_workload_from_manifest(
            TrainingManifest.model_validate(_countdown_manifest())
        )

        # Assert: Qwen3-4B thinks; the prompt fills a fifth of the other 512.
        assert workload.turns == 1
        assert workload.decode_tokens == math.ceil(
            SINGLE_TURN_REASONING_DECODE_SHARE * 1024
        )
        assert workload.row_tokens == workload.decode_tokens + math.ceil(
            SINGLE_TURN_PROMPT_FILL * 512
        )
        assert workload.learn_passes == 3
        assert workload.rollout_batch_size is None

    @pytest.mark.parametrize(
        ("model", "chat_template_kwargs"),
        [
            ("ibm-granite/granite-3.1-3b-a800m-instruct", {}),
            ("Qwen/Qwen3-4B", {"enable_thinking": False}),
        ],
        ids=["non-reasoning-model", "thinking-off"],
    )
    def test_single_turn_without_reasoning_decodes_a_smaller_share(
        self, model: str, chat_template_kwargs: dict
    ) -> None:
        # Arrange
        raw = _countdown_manifest()
        raw["network"]["pretrained_model_name_or_path"] = model
        raw["environment"]["chat_template_kwargs"] = chat_template_kwargs

        # Act
        workload = split_workload_from_manifest(TrainingManifest.model_validate(raw))

        # Assert
        assert workload.decode_tokens == math.ceil(SINGLE_TURN_DECODE_SHARE * 1024)

    def test_multi_turn_spreads_the_row_over_several_turns(self) -> None:
        # Act
        workload = split_workload_from_manifest(
            TrainingManifest.model_validate(_sudoku_manifest())
        )

        # Assert
        assert 1 < workload.turns <= 50
        assert workload.decode_tokens < workload.row_tokens
        assert workload.per_rank_multiple == 1

    def test_multi_turn_runs_a_share_of_the_env_turns(self) -> None:
        # Arrange
        raw = _sudoku_manifest()
        raw["environment"]["max_turns"] = 20

        # Act
        workload = split_workload_from_manifest(TrainingManifest.model_validate(raw))

        # Assert
        assert workload.row_tokens == math.ceil(MULTI_TURN_ROW_FILL * 16384)
        assert workload.turns == math.ceil(MULTI_TURN_TURN_SHARE * 20)
        assert workload.decode_tokens == workload.turns * MULTI_TURN_TURN_DECODE_TOKENS

    def test_default_buffer_generates_more_groups_than_it_trains(self) -> None:
        # Arrange
        raw = _awm_manifest()
        raw["replay_buffer"] = {"kind": "llm"}

        # Act
        workload = split_workload_from_manifest(TrainingManifest.model_validate(raw))

        # Assert
        assert workload.generated_per_trained == TIED_DROP_GENERATION_FACTOR

    @pytest.mark.parametrize(
        ("group_size", "drop_uninformative_groups"),
        [(16, False), (1, True)],
        ids=["drop-off", "group-size-1"],
    )
    def test_trains_every_generated_group_when_nothing_is_dropped(
        self, group_size: int, drop_uninformative_groups: bool
    ) -> None:
        # Arrange
        raw = _awm_manifest()
        raw["algorithm"]["group_size"] = group_size
        raw["replay_buffer"] = {
            "kind": "llm",
            "drop_uninformative_groups": drop_uninformative_groups,
        }

        # Act
        workload = split_workload_from_manifest(TrainingManifest.model_validate(raw))

        # Assert
        assert workload.generated_per_trained == 1.0

    def test_algorithm_without_rollouts_raises(self) -> None:
        # Act / Assert
        with pytest.raises(TypeError, match="SFT does not generate rollouts"):
            split_workload_from_manifest(_sft_manifest())

    @pytest.mark.parametrize(
        ("quantization", "expected"), [(None, "none"), ("fp8", "fp8")]
    )
    def test_engine_weight_format_follows_the_vllm_quantization(
        self, quantization: str | None, expected: str
    ) -> None:
        # Arrange
        raw = _countdown_manifest()
        raw["algorithm"]["vllm_config"] = {"quantization": quantization}

        # Act
        workload = split_workload_from_manifest(TrainingManifest.model_validate(raw))

        # Assert
        assert workload.weight_quantization == expected

    @pytest.mark.parametrize(
        ("engine_args", "expected"),
        [
            ({}, None),
            ({"speculative_config": None}, False),
            ({"speculative_config": {"method": "ngram"}}, True),
        ],
    )
    def test_speculative_drafting_follows_the_engine_args(
        self, engine_args: dict, expected: bool | None
    ) -> None:
        # Arrange
        raw = _sudoku_manifest()
        raw["algorithm"]["vllm_engine_args"] = engine_args

        # Act
        workload = split_workload_from_manifest(TrainingManifest.model_validate(raw))

        # Assert
        assert workload.speculative_drafting is expected

    @pytest.mark.parametrize(
        ("vllm_config", "engine_args", "expected"),
        [
            ({}, {}, None),
            ({"max_num_batched_tokens": 16384}, {}, 16384),
            (
                {"max_num_batched_tokens": 16384},
                {"max_num_batched_tokens": 4096},
                4096,
            ),
            ({}, {"enable_chunked_prefill": False}, 16384),
            (
                {"max_num_batched_tokens": 32768},
                {"enable_chunked_prefill": False},
                32768,
            ),
        ],
        ids=[
            "unset",
            "vllm-config",
            "engine-args-win",
            "no-chunked-prefill",
            "no-chunked-prefill-pinned",
        ],
    )
    def test_pinned_step_follows_the_engine_config(
        self, vllm_config: dict, engine_args: dict, expected: int | None
    ) -> None:
        # Arrange: Sudoku runs a 16384-token context.
        raw = _sudoku_manifest()
        raw["algorithm"]["vllm_config"] = vllm_config
        raw["algorithm"]["vllm_engine_args"] = engine_args

        # Act
        workload = split_workload_from_manifest(TrainingManifest.model_validate(raw))

        # Assert
        assert workload.max_num_batched_tokens == expected

    def test_model_family_default_pins_the_step(self) -> None:
        # Arrange
        raw = _sudoku_manifest()
        raw["network"]["pretrained_model_name_or_path"] = (
            "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
        )

        # Act
        workload = split_workload_from_manifest(TrainingManifest.model_validate(raw))

        # Assert
        assert workload.max_num_batched_tokens == 8192


def _width_split(engine_width: int, step_seconds: float) -> GpuSplit:
    return GpuSplit(
        trainers=4,
        engines=4 // engine_width,
        engine_width=engine_width,
        rollout_batch_size=64,
        learn_seconds=1.0,
        generation_seconds=step_seconds,
    )


class TestChooseEngineWidth:
    def test_fastest_width_wins(self) -> None:
        # Arrange: TP2 is 1.2x faster than TP1.
        splits = [_width_split(1, 12.0), _width_split(2, 10.0)]

        # Act
        chosen = choose_engine_width(splits)

        # Assert
        assert chosen is not None
        assert chosen.engine_width == 2

    def test_fastest_of_several_widths_wins(self) -> None:
        # Arrange
        splits = [_width_split(1, 30.0), _width_split(2, 20.0), _width_split(4, 17.0)]

        # Act
        chosen = choose_engine_width(splits)

        # Assert
        assert chosen is not None
        assert chosen.engine_width == 4

    def test_width_within_ten_percent_of_the_fastest_ties_to_the_narrower(
        self,
    ) -> None:
        # Arrange: TP1 is 8% slower than TP4.
        splits = [_width_split(1, 10.8), _width_split(2, 12.0), _width_split(4, 10.0)]

        # Act
        chosen = choose_engine_width(splits)

        # Assert
        assert chosen is not None
        assert chosen.engine_width == 1

    def test_width_over_ten_percent_slower_loses(self) -> None:
        # Arrange: TP1 is 12% slower than TP2.
        splits = [_width_split(1, 11.2), _width_split(2, 10.0)]

        # Act
        chosen = choose_engine_width(splits)

        # Assert
        assert chosen is not None
        assert chosen.engine_width == 2

    def test_weight_sync_counts_toward_the_step(self) -> None:
        # Arrange: TP1 generates faster, but its merged sync to four engines
        # takes 4 s against TP2's 0.5 s.
        tp1 = _width_split(1, 10.0).model_copy(
            update={"sync_seconds": 4.0, "merged_lora_sync": True}
        )
        tp2 = _width_split(2, 12.0).model_copy(
            update={"sync_seconds": 0.5, "merged_lora_sync": True}
        )

        # Act
        chosen = choose_engine_width([tp1, tp2])

        # Assert
        assert chosen is not None
        assert chosen.engine_width == 2

    def test_tied_width_that_uses_every_gpu_beats_a_narrower_one(self) -> None:
        # Arrange: three TP1 engines leave a GPU idle; two TP2 engines use it.
        tp1 = _width_split(1, 10.0).model_copy(update={"engines": 3})
        tp2 = _width_split(2, 10.5)

        # Act
        chosen = choose_engine_width([tp1, tp2])

        # Assert
        assert chosen is not None
        assert chosen.engine_width == 2

    def test_no_splits_choose_nothing(self) -> None:
        # Act
        chosen = choose_engine_width([])

        # Assert
        assert chosen is None


def _split_on(engines: int, synced_step_seconds: float) -> GpuSplit:
    return GpuSplit(
        trainers=4,
        engines=engines,
        engine_width=2,
        rollout_batch_size=16,
        learn_seconds=synced_step_seconds,
        generation_seconds=1.0,
    )


class TestFullestSplitWithin:
    def test_split_using_every_gpu_wins_within_the_share(self) -> None:
        # Arrange: one TP2 engine leaves 2 of 8 GPUs idle for a 0.5% faster step.
        idle = _split_on(1, 820.7)
        full = _split_on(2, 824.8)

        # Act
        chosen = fullest_split_within([idle, full], IDLE_GPU_TIE_SHARE)

        # Assert
        assert chosen is full

    def test_idle_split_wins_past_the_share(self) -> None:
        # Arrange
        idle = _split_on(1, 800.0)
        full = _split_on(2, 800.0 * (1 + IDLE_GPU_TIE_SHARE) + 1.0)

        # Act
        chosen = fullest_split_within([full, idle], IDLE_GPU_TIE_SHARE)

        # Assert
        assert chosen is idle

    def test_earliest_split_wins_among_the_fullest(self) -> None:
        # Arrange
        first = _split_on(2, 10.1)
        second = _split_on(2, 10.0)

        # Act
        chosen = fullest_split_within([first, second], IDLE_GPU_TIE_SHARE)

        # Assert
        assert chosen is first


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


def _catalog_spec(hub_id: str) -> ModelSpec:
    return ModelSpec(
        model_id=hub_id,
        arch=ModelArch.from_hf_config(SUPPORTED_MODEL_INFO[hub_id].config),
    )


class TestTensorParallelSeconds:
    def test_prices_two_all_reduces_per_layer_at_three_times_the_ring(self) -> None:
        # Arrange
        model = _catalog_spec("Qwen/Qwen3-4B")
        device = _device("H100")

        # Act
        seconds = tensor_parallel_seconds(model, device, 2, 64)

        # Assert
        message_bytes = 64 * 2560 * 2
        expected = 3.0 * 2 * 36 * (10e-6 + message_bytes / 350e9)
        assert math.isclose(seconds, expected, rel_tol=1e-12)

    def test_one_gpu_engine_costs_nothing(self) -> None:
        # Act
        seconds = tensor_parallel_seconds(
            _catalog_spec("Qwen/Qwen3-4B"), _device("H100"), 1, 64
        )

        # Assert
        assert seconds == 0.0

    def test_grows_with_engine_width(self) -> None:
        # Arrange
        model = _catalog_spec("Qwen/Qwen3-4B")
        device = _device("H100")

        # Act
        seconds = [tensor_parallel_seconds(model, device, w, 64) for w in (2, 4, 8)]

        # Assert
        assert seconds[0] < seconds[1] < seconds[2]

    def test_grows_with_running_sequences(self) -> None:
        # Arrange
        model = _catalog_spec("Qwen/Qwen3-4B")
        device = _device("H100")

        # Act
        seconds = [tensor_parallel_seconds(model, device, 2, s) for s in (1, 64, 512)]

        # Assert
        assert seconds[0] < seconds[1] < seconds[2]

    def test_single_kind_layers_all_reduce_once_per_layer(self) -> None:
        # Arrange
        model = _catalog_spec("nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16")
        arch = model.arch

        # Act
        seconds = tensor_parallel_seconds(model, _device("H100"), 2, 1)

        # Assert
        expected = 3.0 * arch.n_layers * (10e-6 + arch.hidden_size * 2 / 350e9)
        assert math.isclose(seconds, expected, rel_tol=1e-12)

    def test_gpu_without_all_reduce_figures_raises(self) -> None:
        # Arrange
        device = DeviceSpec(name="Mystery GPU", total_bytes=80 * 1024**3)

        # Act / Assert
        with pytest.raises(ValueError, match="Mystery GPU has no all-reduce figures"):
            tensor_parallel_seconds(_catalog_spec("Qwen/Qwen3-4B"), device, 2, 64)


def _prefill_workload(row_tokens: int, decode_tokens: int, turns: int) -> SplitWorkload:
    return SplitWorkload(
        batch_size=64,
        group_size=16,
        row_tokens=row_tokens,
        decode_tokens=decode_tokens,
        turns=turns,
        learn_passes=3,
    )


class TestUncachedPrefillTokens:
    def test_single_turn_prefills_its_prompt_in_whole_blocks(self) -> None:
        # Arrange: 1434 prompt tokens in 16-token blocks.
        workload = _prefill_workload(2458, 1024, 1)

        # Act
        tokens = uncached_prefill_tokens(_catalog_spec("Qwen/Qwen3-4B"), workload, 1)

        # Assert
        assert tokens == 1440

    def test_each_turn_prefills_its_added_tokens_in_whole_blocks(self) -> None:
        # Arrange: 4915 prompt and env tokens over 10 turns, 491.5 a turn.
        workload = _prefill_workload(9831, 4916, 10)

        # Act
        tokens = uncached_prefill_tokens(_catalog_spec("Qwen/Qwen3-4B"), workload, 1)

        # Assert
        assert tokens == 10 * 496

    @pytest.mark.parametrize(
        ("engine_width", "expected"), [(2, 10 * 1040), (4, 10 * 1056)]
    )
    def test_hybrid_turns_round_up_to_the_aligned_block(
        self, engine_width: int, expected: int
    ) -> None:
        # Arrange: 983 tokens a turn; Qwen3.6-35B-A3B blocks are 1040 tokens
        # at TP2 and 528 at TP4.
        workload = _prefill_workload(19661, 9831, 10)

        # Act
        tokens = uncached_prefill_tokens(
            _catalog_spec("Qwen/Qwen3.6-35B-A3B"), workload, engine_width
        )

        # Assert
        assert tokens == expected

    def test_a_row_the_engine_decodes_whole_prefills_nothing(self) -> None:
        # Arrange
        workload = _prefill_workload(922, 922, 1)

        # Act
        tokens = uncached_prefill_tokens(_catalog_spec("Qwen/Qwen3-4B"), workload, 1)

        # Assert
        assert tokens == 0


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
        8,
        sync_link,
        search_split,
    )


class TestMergedLoraSyncCandidates:
    def test_engines_serving_merged_weights_price_both_modes(self) -> None:
        # Arrange
        manifest = _manifest(_sudoku_manifest())

        # Act
        modes = merged_lora_sync_candidates(manifest)

        # Assert
        assert modes == (True, False)

    def test_quantized_engine_syncs_the_adapter(self) -> None:
        # Arrange
        raw = _sudoku_manifest()
        manifest = _manifest(raw)
        manifest.algorithm.vllm_config.quantization = "fp8"

        # Act
        modes = merged_lora_sync_candidates(manifest)

        # Assert
        assert modes == (False,)

    def test_mtp_drafting_engine_syncs_the_adapter(self) -> None:
        # Arrange
        raw = _sudoku_manifest()
        raw["algorithm"]["vllm_engine_args"] = {
            "speculative_config": {"method": "qwen3_5_mtp", "num_speculative_tokens": 1}
        }
        manifest = _manifest(raw)

        # Act
        modes = merged_lora_sync_candidates(manifest)

        # Assert
        assert modes == (False,)


def _l4_with(gib: float) -> DeviceSpec:
    """An L4 resized to ``gib`` GiB, so one sync mode's trainers fit and the other's do not."""
    return _device("L4").model_copy(update={"total_bytes": int(gib * 2**30)})


def _kept_4t4e(raw: dict) -> TrainingManifest:
    raw["training"]["training_gpus_per_agent"] = 4
    raw["training"]["rollout_engines_per_agent"] = 4
    return _manifest(raw)


def _kept_sync_over(
    manifest: TrainingManifest, device_name: str, sync_link: SyncLink | None
) -> bool:
    return kept_split_merged_lora_sync(
        manifest,
        catalog_model(manifest.algorithm),
        _device(device_name),
        8,
        8,
        sync_link,
    )


class TestKeptSplitMergedLoraSync:
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
    def test_syncs_the_adapter_when_merged_trainers_do_not_fit(
        self, sync_link: SyncLink | None
    ) -> None:
        # Arrange: countdown at 4 trainers needs 4.75 GiB syncing the adapter;
        # the merged send buffer puts every plan over a 5.25 GiB GPU.
        manifest = _kept_4t4e(_countdown_manifest(64))

        # Act
        merged = kept_split_merged_lora_sync(
            manifest,
            catalog_model(manifest.algorithm),
            _l4_with(5.25),
            8,
            8,
            sync_link,
        )

        # Assert
        assert merged is False

    def test_quantized_engine_syncs_the_adapter(self) -> None:
        # Arrange
        manifest = _kept_4t4e(_sudoku_manifest())
        manifest.algorithm.vllm_config.quantization = "fp8"

        # Act
        merged = _kept_sync_over(manifest, "A100-80", "cuda_ipc")

        # Assert
        assert merged is False

    def test_no_fitting_mode_takes_the_first_candidate(self) -> None:
        # Arrange: neither mode's trainers fit a 1 GiB GPU.
        manifest = _kept_4t4e(_countdown_manifest(64))

        # Act
        merged = kept_split_merged_lora_sync(
            manifest,
            catalog_model(manifest.algorithm),
            _l4_with(1.0),
            8,
            8,
            "rdma",
        )

        # Assert
        assert merged is True

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


class TestGpuSplitSyncedStepSecondsProperty:
    def test_merged_sync_pauses_generation_too(self) -> None:
        # Arrange
        split = GpuSplit(
            trainers=4,
            engines=4,
            engine_width=1,
            rollout_batch_size=64,
            learn_seconds=10.0,
            generation_seconds=12.0,
            sync_seconds=2.0,
            merged_lora_sync=True,
        )

        # Act
        seconds = split.synced_step_seconds

        # Assert
        assert seconds == 14.0

    def test_adapter_sync_overlaps_generation(self) -> None:
        # Arrange
        split = GpuSplit(
            trainers=4,
            engines=4,
            engine_width=1,
            rollout_batch_size=64,
            learn_seconds=10.0,
            generation_seconds=12.0,
            sync_seconds=1.0,
        )

        # Act
        seconds = split.synced_step_seconds

        # Assert
        assert seconds == 12.0
        assert split.step_seconds == 12.0


class TestResolveRunMemoryWeightSync:
    def test_nvlink_syncs_merged_weights_on_a_generation_bound_run(self) -> None:
        # Act: Sudoku on 8x A100 ran 29% faster per step with merged sync.
        plan = _plan_over(_sudoku_manifest(), "A100-80", "cuda_ipc")

        # Assert
        assert plan.merged_lora_sync is True
        assert plan.split is not None
        assert plan.split.merged_lora_sync
        assert plan.split.sync_seconds > 1.0

    def test_tcp_syncs_the_adapter(self) -> None:
        # Act: merged Qwen3-4B weights take over a minute per sync at 0.4 GB/s.
        plan = _plan_over(_sudoku_manifest(), "A100-80", "tcp")

        # Assert
        assert plan.merged_lora_sync is False
        assert plan.split is not None
        assert not plan.split.merged_lora_sync
        assert plan.training.settings.merged_lora_sync is False

    def test_nvlink_syncs_merged_weights_on_a_trainer_bound_run(self) -> None:
        # Act: countdown on 8x H100 is priced 6.7% slower merged and ran 1.6% faster.
        plan = _plan_over(_countdown_manifest(64), "H100-80", "cuda_ipc")

        # Assert
        assert plan.merged_lora_sync is True

    def test_rdma_syncs_the_adapter_when_the_sync_outweighs_the_gain(self) -> None:
        # Act: 7.27 GB to each engine over one 100 Gb/s NIC costs more than
        # the generation merged engines save on countdown.
        plan = _plan_over(_countdown_manifest(64), "H100-80", "rdma")

        # Assert
        assert plan.merged_lora_sync is False
        assert plan.split is not None
        assert not plan.split.merged_lora_sync

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
            manifest, catalog_model(manifest.algorithm), _device("H100-80"), 24, 8
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
                manifest, catalog_model(manifest.algorithm), _device("H100-80"), 24, 8
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
                manifest, catalog_model(manifest.algorithm), _device("H100-80"), 24, 8
            )


class TestGpuCataloguePeakFigures:
    @pytest.mark.parametrize(
        "name", ["A100-40GB", "A100-80", "H100", "H200", "L4", "L40S", "B200"]
    )
    def test_training_gpus_carry_bf16_peak_and_bandwidth(self, name: str) -> None:
        # Act
        device = _device(name)

        # Assert
        assert device.peak_bf16_flops is not None
        assert device.peak_bf16_flops > 1e14
        assert device.memory_bytes_per_second is not None
        assert device.memory_bytes_per_second > 1e11

    @pytest.mark.parametrize(
        "gpu", [gpu for _, gpu in GPU_CATALOGUE], ids=lambda gpu: gpu.name
    )
    def test_every_gpu_that_can_train_has_split_figures(self, gpu: GpuInfo) -> None:
        # Arrange
        device = gpu.device_spec()

        # Act
        unsupported = unsupported_gpu_reason(device)

        # Assert
        has_figures = (
            device.peak_bf16_flops is not None
            and device.memory_bytes_per_second is not None
            and device.all_reduce_bytes_per_second is not None
            and device.all_reduce_latency_seconds is not None
        )
        assert has_figures != (unsupported is not None)

    def test_every_catalogue_entry_with_flops_has_bandwidth(self) -> None:
        # Assert
        for _, gpu in GPU_CATALOGUE:
            assert (gpu.bf16_tflops is None) == (gpu.memory_gb_per_second is None)

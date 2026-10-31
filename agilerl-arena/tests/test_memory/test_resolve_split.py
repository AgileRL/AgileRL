# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""An async member's trainer / engine split is chosen from the throughput model."""

from __future__ import annotations

import math

import pytest

from agilerl.arena.memory.manifest import (
    GPU_CATALOGUE,
    GpuInfo,
    lookup_gpu,
    training_memory_from_manifest,
    unsupported_gpu_reason,
    validated_run_manifest,
)
from agilerl.arena.memory.resolve.batch import catalog_model
from agilerl.arena.memory.resolve.fsdp import (
    TrainingMemoryPlan,
    TrainingMemoryUnfitError,
    learn_cost,
)
from agilerl.arena.memory.resolve.learn_costs import learn_costs
from agilerl.arena.memory.resolve.split import (
    NGRAM_GENERATION_SPEEDUP,
    GpuSplit,
    RunMemoryPlan,
    SplitWorkload,
    active_matmul_params,
    choose_engine_width,
    choose_gpu_split,
    resolve_run_memory,
    split_workload_from_manifest,
    tensor_parallel_seconds,
    uncached_prefill_tokens,
)
from agilerl.arena.memory.specs import DeviceSpec, ModelArch, ModelSpec
from agilerl.arena.models.manifest import TrainingManifest
from agilerl.arena.models.model_info import SUPPORTED_MODEL_INFO


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
        "replay_buffer": {"kind": "llm"},
    }


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
        "replay_buffer": {"kind": "llm"},
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
        assert batch_size <= split.rollout_batch_size < batch_size + cycle
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

    def test_no_split_raises(self) -> None:
        # Act / Assert
        with pytest.raises(ValueError, match="No trainer / rollout-engine split"):
            _plan(_sudoku_manifest(), gpus=1)

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
            for trainers in range(1, 10 - 4 + 1)
            if _training_plan(manifest, trainers, _device("L4")) is None
        ]

        # Act
        split = _split(raw, gpus=10, engine_width=4, device=_device("L4"))

        # Assert
        assert unfit
        assert split.trainers not in unfit
        assert split.trainers + split.engines * 4 <= 10


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

    def test_multi_turn_drafting_divides_generation_by_the_ngram_speedup(
        self,
    ) -> None:
        # Arrange: Qwen3-4B drafts multi-turn episodes by default.
        drafted = _workload(_sudoku_manifest())
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
        # Arrange: 8128 prompt tokens and 64 decoded per episode, 128 episodes
        # per engine.
        workload = _workload(_countdown_manifest(64), row_tokens=8192, decode_tokens=64)
        arch = _catalog_spec("Qwen/Qwen3-4B").arch

        # Act
        split = _learn_bound_split(workload, 4)

        # Assert
        rate = 0.4 * 312e12 / (2 * active_matmul_params(arch))
        assert split.generation_seconds == pytest.approx(128 * 8128 / rate, rel=1e-12)

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
        any_count = _workload(_countdown_manifest())
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
            manifest, catalog_model(manifest.algorithm), _device(), None
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
    def test_single_turn_decodes_the_output_cap_in_one_turn(self) -> None:
        # Act
        workload = split_workload_from_manifest(
            TrainingManifest.model_validate(_countdown_manifest())
        )

        # Assert
        assert workload.turns == 1
        assert workload.decode_tokens == min(1024, workload.row_tokens)
        assert workload.learn_passes == 3
        assert workload.rollout_batch_size is None

    def test_multi_turn_spreads_the_row_over_several_turns(self) -> None:
        # Act
        workload = split_workload_from_manifest(
            TrainingManifest.model_validate(_sudoku_manifest())
        )

        # Assert
        assert 1 < workload.turns <= 50
        assert workload.decode_tokens < workload.row_tokens
        assert workload.per_rank_multiple == 1

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

    def test_width_within_two_percent_of_the_fastest_ties_to_the_narrower(
        self,
    ) -> None:
        # Arrange: TP1 is 1.5% slower than TP4.
        splits = [_width_split(1, 10.15), _width_split(2, 11.0), _width_split(4, 10.0)]

        # Act
        chosen = choose_engine_width(splits)

        # Assert
        assert chosen is not None
        assert chosen.engine_width == 1

    def test_width_over_two_percent_slower_loses(self) -> None:
        # Arrange: TP1 is 2.5% slower than TP2.
        splits = [_width_split(1, 10.25), _width_split(2, 10.0)]

        # Act
        chosen = choose_engine_width(splits)

        # Assert
        assert chosen is not None
        assert chosen.engine_width == 2

    def test_no_splits_choose_nothing(self) -> None:
        # Act
        chosen = choose_engine_width([])

        # Assert
        assert chosen is None


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

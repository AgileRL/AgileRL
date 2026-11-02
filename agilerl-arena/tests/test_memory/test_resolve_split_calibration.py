# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""The GPU-split planner's learn and generation times match measured runs."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import pytest

from agilerl.arena.memory.manifest import (
    lookup_gpu,
    training_memory_from_manifest,
    validated_run_manifest,
)
from agilerl.arena.memory.resolve.batch import catalog_model
from agilerl.arena.memory.resolve.split import (
    GpuSplit,
    choose_gpu_split,
    split_workload_from_manifest,
)
from agilerl.arena.models.manifest import TrainingManifest

LEARN_TOLERANCE = 0.2
GENERATION_TOLERANCE = 0.15
# At the manifest's expected lengths, which cannot see an env's prompt, turn
# or completion lengths.
CONFIG_LEARN_TOLERANCE = 0.25
CONFIG_GENERATION_TOLERANCE = 0.45
# Runs computing at least this share of trained rows count as computing all.
FULL_COMPUTE_SHARE = 0.95


def _countdown(
    model: str = "Qwen/Qwen3-4B",
    batch_size: int = 64,
    max_output_tokens: int = 1024,
    max_context_length: int = 1536,
    lora_config: dict | None = None,
) -> dict:
    """Single-turn GRPO countdown, shaped like speed benchmark A."""
    return {
        "algorithm": {
            "name": "GRPO",
            "batch_size": batch_size,
            "group_size": 8,
            "max_output_tokens": max_output_tokens,
            "fsdp": {},
        },
        "environment": {
            "env_type": "rollout",
            "dataset": "dataset.json",
            "rubric_file_path": "countdown_reward.py",
            "prompt_template": {
                "system_prompt": "x",
                "user_prompt": "{question} {answer}",
            },
            "max_reward": 2.0,
        },
        "network": {
            "pretrained_model_name_or_path": model,
            "max_context_length": max_context_length,
            "lora_config": lora_config
            or {"lora_r": 16, "lora_alpha": 32, "target_modules": "all-linear"},
        },
        "training": {"rollout_mode": "async", "pop_size": 1},
        "replay_buffer": {"kind": "llm", "drop_uninformative_groups": False},
    }


def _granite_countdown(model: str, target_modules: list[str]) -> dict:
    """Single-turn GRPO countdown on a Granite MoE with expert LoRA."""
    return _countdown(
        model,
        32,
        512,
        1024,
        {
            "lora_r": 8,
            "lora_alpha": 32,
            "target_modules": target_modules,
            "target_parameters": [
                "block_sparse_moe.experts.gate_up_proj",
                "block_sparse_moe.experts.down_proj",
            ],
        },
    )


def _sudoku() -> dict:
    """Multi-turn CISPO on Sudoku-v0-hard, shaped like speed benchmark B."""
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
            "lora_config": {
                "lora_r": 16,
                "lora_alpha": 32,
                "target_modules": "all-linear",
            },
        },
        "training": {"rollout_mode": "async", "pop_size": 1},
        "replay_buffer": {"kind": "llm", "drop_uninformative_groups": False},
    }


def _awm(
    expert_lora: bool = False,
    quantization: str | None = None,
    drop_tied_groups: bool = False,
) -> dict:
    """Multi-turn CISPO LoRA on Qwen3.6-35B-A3B, shaped like the AWM training config.

    :param drop_tied_groups: Keep the buffer and env defaults: drop tied groups,
        replay stored completions and sample tasks adaptively.
    """
    lora_config: dict = {
        "lora_r": 8,
        "lora_alpha": 16,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "in_proj_qkv",
            "in_proj_a",
            "in_proj_b",
            "in_proj_z",
            "out_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    }
    if expert_lora:
        lora_config["target_parameters"] = [
            "mlp.experts.gate_up_proj",
            "mlp.experts.down_proj",
        ]
    algorithm: dict = {
        "name": "CISPO",
        "batch_size": 64,
        "group_size": 16,
        "mini_batch_size": 64,
        "beta": 0.0,
        "max_output_tokens": 2048,
    }
    if quantization is not None:
        algorithm["vllm_config"] = {"quantization": quantization}
    environment: dict = {
        "env_type": "rollout",
        "env_image": "env:latest",
        "modality": "text",
        "max_turns": 20,
        "max_reward": 1.0,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    replay_buffer: dict = {"kind": "llm"}
    if not drop_tied_groups:
        environment["adaptive_task_sampling"] = False
        replay_buffer["drop_uninformative_groups"] = False
        replay_buffer["group_replay_max_age"] = 0
    return {
        "algorithm": algorithm,
        "environment": environment,
        "network": {
            "pretrained_model_name_or_path": "Qwen/Qwen3.6-35B-A3B",
            "max_context_length": 32768,
            "lora_config": lora_config,
        },
        "training": {"rollout_mode": "async", "pop_size": 1},
        "replay_buffer": replay_buffer,
    }


@dataclass(frozen=True)
class CalibrationCase:
    """One measured run: its layout, mean lengths and per-step seconds."""

    name: str
    manifest: dict
    gpu: str
    trainers: int
    engines: int
    engine_width: int
    # ``None`` where the planner sized the cycle.
    rollout_batch_size: int | None
    # ``None`` where the engines ran vLLM's default step.
    max_num_batched_tokens: int | None
    # Mean learn seconds per step; ``None`` where only generation was measured.
    learn_seconds: float | None
    # Share of trained rows the learn step computed; zero-advantage rows skip.
    computed_share: float
    # Mean generation seconds per learn step; ``None`` for learn-only runs.
    generation_seconds: float | None
    # True where the trainers waited on the engines; otherwise the engines
    # waited on the trainers and the measured time is an upper bound.
    generation_bound: bool
    # Measured mean trained-row tokens, decoded tokens per row, and turns.
    row_tokens: int
    decode_tokens: int
    turns: int
    # False where KV thrash evicted the prefix cache (0-15% hits); the planner
    # prices later turns with their history cached.
    prefix_cache_held: bool
    # True where the learn efficiencies were fitted on this run.
    learn_fitted: bool
    source: str
    # Decoded tokens generated per trained token; above 1 where tied groups
    # dropped.
    generated_per_trained: float = 1.0
    # True where trainers balanced learn tokens across ranks, windows and
    # micro-batches; MoE learn ran slower before.
    balanced_learn: bool = False


GRANITE_31_ATTENTION = ["q_proj", "k_proj", "v_proj", "o_proj"]
GRANITE_4H_PROJECTIONS = [*GRANITE_31_ATTENTION, "in_proj", "out_proj"]

# Qwen3-4B and Granite runs are the speed benchmarks on 8x A100-80 and 8x
# H100-80. AWM runs are Qwen3.6-35B-A3B on 8x H100-80; only run 3 had KV
# admission.
CASES = [
    CalibrationCase(
        name="a100-countdown-4t4e",
        manifest=_countdown(),
        gpu="A100-80",
        trainers=4,
        engines=4,
        engine_width=1,
        rollout_batch_size=None,
        max_num_batched_tokens=None,
        learn_seconds=19.16,
        computed_share=1.0,
        generation_seconds=19.84,
        generation_bound=False,
        row_tokens=979,
        decode_tokens=872,
        turns=1,
        prefix_cache_held=True,
        learn_fitted=True,
        source="speedbench lora_a64_bf16_est",
    ),
    CalibrationCase(
        name="a100-sudoku-3t5e",
        manifest=_sudoku(),
        gpu="A100-80",
        trainers=3,
        engines=5,
        engine_width=1,
        rollout_batch_size=15,
        max_num_batched_tokens=None,
        learn_seconds=89.69,
        computed_share=1.0,
        generation_seconds=94.63,
        generation_bound=False,
        row_tokens=8861,
        decode_tokens=4668,
        turns=12,
        prefix_cache_held=True,
        learn_fitted=False,
        source="speedbench throttle_b3t5e",
    ),
    CalibrationCase(
        name="a100-sudoku-4t4e",
        manifest=_sudoku(),
        gpu="A100-80",
        trainers=4,
        engines=4,
        engine_width=1,
        rollout_batch_size=None,
        max_num_batched_tokens=None,
        learn_seconds=70.48,
        computed_share=1.0,
        generation_seconds=93.91,
        generation_bound=True,
        row_tokens=8867,
        decode_tokens=4713,
        turns=11,
        prefix_cache_held=True,
        learn_fitted=True,
        source="speedbench throttle_b4t4e",
    ),
    CalibrationCase(
        name="a100-granite31-2t",
        manifest=_granite_countdown(
            "ibm-granite/granite-3.1-3b-a800m-instruct", GRANITE_31_ATTENTION
        ),
        gpu="A100-80",
        trainers=2,
        engines=1,
        engine_width=1,
        rollout_batch_size=None,
        max_num_batched_tokens=None,
        learn_seconds=3.53,
        computed_share=1.0,
        generation_seconds=None,
        generation_bound=False,
        row_tokens=252,
        decode_tokens=142,
        turns=1,
        prefix_cache_held=True,
        learn_fitted=False,
        source="speedbench ddpchunk",
    ),
    CalibrationCase(
        name="h100-countdown-4t4e",
        manifest=_countdown(),
        gpu="H100-80",
        trainers=4,
        engines=4,
        engine_width=1,
        rollout_batch_size=None,
        max_num_batched_tokens=None,
        learn_seconds=9.48,
        computed_share=1.0,
        generation_seconds=9.85,
        generation_bound=False,
        row_tokens=987,
        decode_tokens=880,
        turns=1,
        prefix_cache_held=True,
        learn_fitted=True,
        source="h100bench sg_a64",
    ),
    CalibrationCase(
        name="h100-sudoku-3t5e",
        manifest=_sudoku(),
        gpu="H100-80",
        trainers=3,
        engines=5,
        engine_width=1,
        rollout_batch_size=None,
        max_num_batched_tokens=None,
        learn_seconds=41.77,
        computed_share=1.0,
        generation_seconds=48.25,
        generation_bound=False,
        row_tokens=8255,
        decode_tokens=4210,
        turns=11,
        prefix_cache_held=True,
        learn_fitted=False,
        source="h100bench qb01",
    ),
    CalibrationCase(
        name="h100-granite31-2t",
        manifest=_granite_countdown(
            "ibm-granite/granite-3.1-3b-a800m-instruct", GRANITE_31_ATTENTION
        ),
        gpu="H100-80",
        trainers=2,
        engines=1,
        engine_width=1,
        rollout_batch_size=None,
        max_num_batched_tokens=None,
        learn_seconds=2.07,
        computed_share=1.0,
        generation_seconds=None,
        generation_bound=False,
        row_tokens=259,
        decode_tokens=148,
        turns=1,
        prefix_cache_held=True,
        learn_fitted=False,
        source="h100bench qg01",
    ),
    CalibrationCase(
        name="h100-granite4h-2t",
        manifest=_granite_countdown(
            "ibm-granite/granite-4.0-h-tiny", GRANITE_4H_PROJECTIONS
        ),
        gpu="H100-80",
        trainers=2,
        engines=1,
        engine_width=1,
        rollout_batch_size=None,
        max_num_batched_tokens=None,
        learn_seconds=4.05,
        computed_share=1.0,
        generation_seconds=None,
        generation_bound=False,
        row_tokens=329,
        decode_tokens=227,
        turns=1,
        prefix_cache_held=True,
        learn_fitted=False,
        source="h100bench qh01",
    ),
    CalibrationCase(
        name="awm-tp4-rbs64-expert-lora",
        manifest=_awm(expert_lora=True),
        gpu="H100-80",
        trainers=4,
        engines=1,
        engine_width=4,
        rollout_batch_size=64,
        max_num_batched_tokens=8192,
        learn_seconds=615.99,
        computed_share=1.0,
        generation_seconds=889.3,
        generation_bound=True,
        row_tokens=12950,
        decode_tokens=988,
        turns=3,
        prefix_cache_held=False,
        learn_fitted=False,
        source="awm-bench arm A",
    ),
    CalibrationCase(
        name="awm-tp4-rbs64-fp8",
        manifest=_awm(expert_lora=True, quantization="fp8"),
        gpu="H100-80",
        trainers=4,
        engines=1,
        engine_width=4,
        rollout_batch_size=64,
        max_num_batched_tokens=8192,
        learn_seconds=None,
        computed_share=1.0,
        generation_seconds=1111.4,
        generation_bound=True,
        row_tokens=13084,
        decode_tokens=1062,
        turns=3,
        prefix_cache_held=False,
        learn_fitted=False,
        source="awm-bench arm B",
    ),
    CalibrationCase(
        name="awm-2tp2-rbs24",
        manifest=_awm(),
        gpu="H100-80",
        trainers=4,
        engines=2,
        engine_width=2,
        rollout_batch_size=24,
        max_num_batched_tokens=8192,
        learn_seconds=520.69,
        computed_share=1.0,
        generation_seconds=521.6,
        generation_bound=False,
        row_tokens=12873,
        decode_tokens=1039,
        turns=4,
        prefix_cache_held=False,
        learn_fitted=True,
        source="awm-bench arm D",
    ),
    CalibrationCase(
        name="awm-2tp2-rbs28-skip",
        manifest=_awm(),
        gpu="H100-80",
        trainers=4,
        engines=2,
        engine_width=2,
        rollout_batch_size=28,
        max_num_batched_tokens=8192,
        learn_seconds=444.37,
        computed_share=0.753,
        generation_seconds=496.0,
        generation_bound=True,
        row_tokens=13175,
        decode_tokens=1041,
        turns=4,
        prefix_cache_held=False,
        learn_fitted=False,
        source="awm-bench arm E",
    ),
    CalibrationCase(
        name="awm-run3-early",
        manifest=_awm(),
        gpu="H100-80",
        trainers=4,
        engines=2,
        engine_width=2,
        rollout_batch_size=28,
        max_num_batched_tokens=16384,
        learn_seconds=430.30,
        computed_share=0.592,
        generation_seconds=433.1,
        generation_bound=False,
        row_tokens=13222,
        decode_tokens=1069,
        turns=5,
        prefix_cache_held=True,
        learn_fitted=False,
        source="AWM run 3, steps 3-16",
    ),
    CalibrationCase(
        name="awm-run3-late",
        manifest=_awm(),
        gpu="H100-80",
        trainers=4,
        engines=2,
        engine_width=2,
        rollout_batch_size=28,
        max_num_batched_tokens=16384,
        learn_seconds=264.11,
        computed_share=0.333,
        generation_seconds=380.2,
        generation_bound=True,
        row_tokens=13826,
        decode_tokens=1097,
        turns=5,
        prefix_cache_held=True,
        learn_fitted=False,
        source="AWM run 3, steps 17-96",
    ),
    CalibrationCase(
        name="awm-drop-steps-3-4",
        manifest=_awm(drop_tied_groups=True),
        gpu="H100-80",
        trainers=4,
        engines=2,
        engine_width=2,
        rollout_batch_size=16,
        max_num_batched_tokens=16384,
        learn_seconds=452.5,
        computed_share=1.0,
        generation_seconds=467.5,
        generation_bound=True,
        row_tokens=12698,
        decode_tokens=895,
        turns=2,
        prefix_cache_held=True,
        learn_fitted=True,
        source="AWM job awm-final-20261101215249, steps 3-4",
        generated_per_trained=1.55,
        balanced_learn=True,
    ),
    CalibrationCase(
        name="awm-drop-steps-5-8",
        manifest=_awm(drop_tied_groups=True),
        gpu="H100-80",
        trainers=4,
        engines=2,
        engine_width=2,
        rollout_batch_size=16,
        max_num_batched_tokens=16384,
        learn_seconds=489.0,
        computed_share=0.98,
        generation_seconds=489.3,
        generation_bound=False,
        row_tokens=13490,
        decode_tokens=1212,
        turns=4,
        prefix_cache_held=True,
        learn_fitted=False,
        source="AWM job awm-final-20261101215249, steps 5-8",
        generated_per_trained=1.10,
        balanced_learn=True,
    ),
    CalibrationCase(
        name="awm-keep-steps-2-4",
        manifest=_awm(),
        gpu="H100-80",
        trainers=4,
        engines=2,
        engine_width=2,
        rollout_batch_size=16,
        max_num_batched_tokens=16384,
        learn_seconds=263.0,
        computed_share=0.427,
        generation_seconds=277.0,
        generation_bound=False,
        row_tokens=12514,
        decode_tokens=728,
        turns=2,
        prefix_cache_held=True,
        learn_fitted=False,
        source="AWM job awm-nodrop-20261101221427, steps 2-4",
        generated_per_trained=1.02,
        balanced_learn=True,
    ),
    CalibrationCase(
        name="awm-keep-steps-5-6",
        manifest=_awm(),
        gpu="H100-80",
        trainers=4,
        engines=2,
        engine_width=2,
        rollout_batch_size=16,
        max_num_batched_tokens=16384,
        learn_seconds=423.0,
        computed_share=0.735,
        generation_seconds=425.0,
        generation_bound=False,
        row_tokens=12914,
        decode_tokens=863,
        turns=3,
        prefix_cache_held=True,
        learn_fitted=False,
        source="AWM job awm-nodrop-20261101221427, steps 5-6",
        generated_per_trained=1.10,
        balanced_learn=True,
    ),
]


def _predicted_split(case: CalibrationCase, measured_lengths: bool) -> GpuSplit:
    """The planner's split at the case's layout, at measured or manifest lengths."""
    raw = copy.deepcopy(case.manifest)
    raw["algorithm"].setdefault("vllm_config", {})["tensor_parallel_size"] = (
        case.engine_width
    )
    raw["training"]["training_gpus_per_agent"] = case.trainers
    raw["training"]["rollout_engines_per_agent"] = case.engines
    if case.rollout_batch_size is not None:
        raw["training"]["rollout_batch_size"] = case.rollout_batch_size
    manifest = validated_run_manifest(TrainingManifest.model_validate(raw))
    gpu = lookup_gpu(case.gpu)
    assert gpu is not None
    device = gpu.device_spec()
    model = catalog_model(manifest.algorithm)
    ratio = training_memory_from_manifest(manifest, model, device).learn_cost_ratio
    update: dict = {"max_num_batched_tokens": case.max_num_batched_tokens}
    if measured_lengths:
        update |= {
            "row_tokens": case.row_tokens,
            "decode_tokens": case.decode_tokens,
            "turns": case.turns,
            "generated_per_trained": case.generated_per_trained,
        }
    workload = split_workload_from_manifest(manifest).model_copy(update=update)
    split = choose_gpu_split(
        model,
        device,
        workload,
        case.trainers + case.engines * case.engine_width,
        case.engine_width,
        lambda trainers: ratio if trainers == case.trainers else None,
    )
    assert split is not None
    assert (split.trainers, split.engines) == (case.trainers, case.engines)
    return split


def _unbalanced_moe(case: CalibrationCase) -> bool:
    """A MoE run measured before trainers balanced learn tokens."""
    manifest = TrainingManifest.model_validate(case.manifest)
    model = catalog_model(manifest.algorithm)
    assert model is not None
    return model.arch.is_moe and not case.balanced_learn


def _case_id(case: CalibrationCase) -> str:
    return f"{case.name}-{'fitted' if case.learn_fitted else 'held-out'}"


FULL_COMPUTE_CASES = [
    case
    for case in CASES
    if case.learn_seconds and case.computed_share >= FULL_COMPUTE_SHARE
]
MATCHED_LEARN_CASES = [case for case in FULL_COMPUTE_CASES if not _unbalanced_moe(case)]
ENGINE_BOUND_CASES = [
    case
    for case in CASES
    if case.generation_bound and case.generation_seconds and case.prefix_cache_held
]


class TestChooseGpuSplitCalibration:
    @pytest.mark.parametrize("case", MATCHED_LEARN_CASES, ids=_case_id)
    def test_learn_seconds_match_runs_that_computed_every_row(
        self, case: CalibrationCase
    ) -> None:
        # Act
        split = _predicted_split(case, True)

        # Assert
        assert split.learn_seconds == pytest.approx(
            case.learn_seconds, rel=LEARN_TOLERANCE
        )

    @pytest.mark.parametrize(
        "case",
        [case for case in FULL_COMPUTE_CASES if _unbalanced_moe(case)],
        ids=_case_id,
    )
    def test_learn_seconds_bound_moe_runs_with_unbalanced_learn_tokens(
        self, case: CalibrationCase
    ) -> None:
        # Arrange
        assert case.learn_seconds is not None

        # Act
        split = _predicted_split(case, True)

        # Assert
        assert split.learn_seconds <= (1 + LEARN_TOLERANCE) * case.learn_seconds

    @pytest.mark.parametrize(
        "case",
        [
            case
            for case in CASES
            if case.learn_seconds
            and case.balanced_learn
            and case.computed_share < FULL_COMPUTE_SHARE
        ],
        ids=_case_id,
    )
    def test_learn_seconds_bound_runs_that_skipped_zero_advantage_rows(
        self, case: CalibrationCase
    ) -> None:
        # Arrange
        assert case.learn_seconds is not None

        # Act
        split = _predicted_split(case, True)

        # Assert
        assert split.learn_seconds >= (1 - LEARN_TOLERANCE) * case.learn_seconds

    @pytest.mark.parametrize(
        "case",
        [
            case
            for case in CASES
            if case.generation_seconds is not None and case.prefix_cache_held
        ],
        ids=lambda case: case.name,
    )
    def test_generation_seconds_match_engine_bound_runs_and_bound_the_rest(
        self, case: CalibrationCase
    ) -> None:
        # Arrange
        assert case.generation_seconds is not None

        # Act
        split = _predicted_split(case, True)

        # Assert
        if case.generation_bound:
            assert split.generation_seconds == pytest.approx(
                case.generation_seconds, rel=GENERATION_TOLERANCE
            )
        else:
            assert split.generation_seconds <= (1 + GENERATION_TOLERANCE) * (
                case.generation_seconds
            )

    @pytest.mark.parametrize("case", MATCHED_LEARN_CASES, ids=_case_id)
    def test_learn_seconds_at_manifest_lengths_match_runs_that_computed_every_row(
        self, case: CalibrationCase
    ) -> None:
        # Act
        split = _predicted_split(case, False)

        # Assert
        assert split.learn_seconds == pytest.approx(
            case.learn_seconds, rel=CONFIG_LEARN_TOLERANCE
        )

    @pytest.mark.parametrize("case", ENGINE_BOUND_CASES, ids=lambda case: case.name)
    def test_generation_seconds_at_manifest_lengths_match_engine_bound_runs(
        self, case: CalibrationCase
    ) -> None:
        # Act
        split = _predicted_split(case, False)

        # Assert
        assert split.generation_seconds == pytest.approx(
            case.generation_seconds, rel=CONFIG_GENERATION_TOLERANCE
        )

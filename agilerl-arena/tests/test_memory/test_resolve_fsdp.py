# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""The training-memory search: DDP or FSDP, speed rungs by modeled cost, then memory-only last resorts."""

from __future__ import annotations

import logging
import math
from dataclasses import replace

import pytest

from agilerl.arena.memory import formulas
from agilerl.arena.memory.estimator import estimate_training
from agilerl.arena.memory.resolve.batch import ROW_FILL, micro_batch_row_candidates
from agilerl.arena.memory.resolve.fsdp import (
    RESIDENCIES,
    MemoryPins,
    TrainingMemoryPlan,
    TrainingMemoryUnfitError,
    gather_cost,
    learn_cost,
    resolve_training_memory,
    routed_chunks_per_pass,
)
from agilerl.arena.memory.resolve.learn_costs import SM80_LEARN_COSTS, SM90_LEARN_COSTS
from agilerl.arena.memory.specs import (
    DeviceSpec,
    GiB,
    MiB,
    ModelSpec,
    TrainingSettings,
)
from agilerl.arena.models.fsdp import FSDP_MEMORY_FIELDS, FSDPConfig
from tests.test_memory.test_estimator import (
    A100_80GB_CUDA_BYTES,
    GRANITE_MOE,
    H100_80GB_CUDA_BYTES,
    MOE_MODEL,
    SUPER_VL_MODEL,
    budget_device,
    catalog_spec,
    granite_grpo,
    super_vl_cispo,
    training_gpu,
)
from tests.test_memory.test_formulas import QWEN_05B

DENSE_MODEL = ModelSpec(model_id="Qwen/Qwen2.5-0.5B-Instruct", arch=QWEN_05B)


def moe_floor_settings(**fsdp: object) -> TrainingSettings:
    """Expert-LoRA EP 8 SFT at 16k tokens, one row, every block checkpointed."""
    return TrainingSettings(
        algorithm="sft",
        max_model_len=16384,
        lora_rank=8,
        lora_packed_target_matrices=2,
        packed_moe_dispatch="contracted",
        n_training_gpus=8,
        fsdp=FSDPConfig(ep=8, checkpoint_every_n_blocks=1, **fsdp),
    )


def moe_peak(chunk: int, optim_offload: bool, activation_offload: bool = False) -> int:
    """Resharded MoE floor peak at one chunk and optimizer placement, as the search prices it."""
    settings = moe_floor_settings(
        reshard_after_forward=True,
        activation_offload=activation_offload,
        routed_expert_chunk_mib=chunk,
        optim_cpu_offload=optim_offload,
        cpu_offload=False,
    ).model_copy(update={"activation_offload": activation_offload})
    return estimate_training(
        MOE_MODEL, budget_device(80 * GiB), settings, orchestrated=True
    ).total_bytes


def super_vl_vwa() -> TrainingSettings:
    """Super-VL VWA settings with every memory field left to the search."""
    settings = super_vl_cispo(29120, 16)
    unset = dict.fromkeys(FSDP_MEMORY_FIELDS)
    return settings.model_copy(
        update={"fsdp": replace(settings.fsdp, **unset, checkpoint_every_n_blocks=1)}
    )


def resolve_floor(settings: TrainingSettings, usable: int) -> TrainingMemoryPlan:
    """The plan for one checkpointed row, so only memory fields and residency move."""
    return resolve_training_memory(
        MOE_MODEL if settings.fsdp.ep > 1 else DENSE_MODEL,
        budget_device(usable),
        settings,
        MemoryPins(gradient_checkpointing=True, micro_batch_size_per_gpu=1),
        1,
    )


def dense_settings(n_training_gpus: int = 4, **fsdp: object) -> TrainingSettings:
    """Dense LoRA GRPO allowed to shard, rows padded to 4096 tokens."""
    return TrainingSettings(
        max_model_len=4096,
        micro_batch_size=1,
        n_training_gpus=n_training_gpus,
        fsdp=FSDPConfig(**fsdp),
    )


def dense_peak(settings: TrainingSettings, sharded: bool) -> int:
    """Dense floor peak, one checkpointed row, resharded FSDP or DDP, as the search prices it."""
    fsdp = replace(
        settings.fsdp,
        reshard_after_forward=True,
        routed_expert_chunk_mib=512,
        optim_cpu_offload=False,
        activation_offload=False,
        cpu_offload=False,
        checkpoint_every_n_blocks=1,
    )
    return estimate_training(
        DENSE_MODEL,
        budget_device(80 * GiB),
        settings.model_copy(
            update={
                "fsdp": fsdp if sharded else None,
                "gradient_checkpointing": True,
            }
        ),
        orchestrated=True,
    ).total_bytes


def buffered(peak: int) -> int:
    """Smallest budget that holds ``peak`` with the underprediction buffer."""
    return int(peak / (1 - formulas.MAX_UNDERPREDICTION)) + 1


class TestGatherCost:
    def test_residencies_cost_their_measured_gather_in_learn_tokens(self):
        # Act
        costs = {
            residency: gather_cost(residency, SM80_LEARN_COSTS)
            for residency in RESIDENCIES
        }

        # Assert
        reshard = SM80_LEARN_COSTS.reshard_gather_tokens
        assert costs == {"gathered": reshard / 2, "reshard": reshard}

    def test_ddp_gathers_nothing(self):
        assert gather_cost(None, SM80_LEARN_COSTS) == 0.0


class TestLearnCost:
    def test_more_checkpointed_blocks_cost_more(self):
        # Act
        costs = [
            learn_cost(24, on, stride, 2, 16 * 1024, "reshard", SM80_LEARN_COSTS)
            for on, stride in ((False, 1), (True, 4), (True, 2), (True, 1))
        ]

        # Assert
        assert costs == sorted(costs)
        assert len(set(costs)) == len(costs)

    def test_each_micro_batch_adds_its_overhead_and_gather(self):
        # Arrange
        tokens = 16 * 1024

        # Act
        one_pass = learn_cost(24, False, 1, 1, tokens, "gathered", SM80_LEARN_COSTS)
        three_passes = learn_cost(24, False, 1, 3, tokens, "gathered", SM80_LEARN_COSTS)

        # Assert
        per_pass = SM80_LEARN_COSTS.micro_batch_overhead_tokens + gather_cost(
            "gathered", SM80_LEARN_COSTS
        )
        assert one_pass == pytest.approx(tokens + per_pass)
        assert three_passes - one_pass == pytest.approx(2 * per_pass)

    def test_ddp_beats_gathered_at_the_same_micro_batch(self):
        # Act
        ddp = learn_cost(36, True, 2, 4, 64 * 1024, None, SM80_LEARN_COSTS)
        gathered = learn_cost(36, True, 2, 4, 64 * 1024, "gathered", SM80_LEARN_COSTS)
        resharded = learn_cost(36, True, 2, 4, 64 * 1024, "reshard", SM80_LEARN_COSTS)

        # Assert
        assert ddp < gathered < resharded

    def test_each_routed_chunk_adds_its_cost_per_micro_batch(self):
        # Act
        plain = learn_cost(32, False, 1, 2, 64 * 1024, None, SM80_LEARN_COSTS)
        chunked = learn_cost(32, False, 1, 2, 64 * 1024, None, SM80_LEARN_COSTS, 480)

        # Assert
        assert chunked - plain == pytest.approx(
            2 * 480 * SM80_LEARN_COSTS.routed_chunk_tokens
        )


class TestRoutedChunksPerPass:
    def test_counts_every_moe_layers_chunks_of_the_widest_row(self):
        # Arrange
        arch = catalog_spec(GRANITE_MOE).arch
        tokens = 39322
        inter = arch.expert_intermediate_size or arch.intermediate_size
        row_bytes = max(arch.expert_width, inter) * 2
        routed_bytes = tokens * arch.n_experts_per_tok * row_bytes

        # Act
        default = routed_chunks_per_pass(arch, True, tokens, 2.0, 64 * MiB)
        large = routed_chunks_per_pass(arch, True, tokens, 2.0, 512 * MiB)

        # Assert
        assert default == arch.moe_layers * math.ceil(routed_bytes / (64 * MiB))
        assert large == arch.moe_layers * math.ceil(routed_bytes / (512 * MiB))
        assert default > large

    def test_frozen_experts_and_dense_models_run_no_chunks(self):
        # Arrange
        arch = catalog_spec(GRANITE_MOE).arch

        # Act
        frozen = routed_chunks_per_pass(arch, False, 4096, 2.0, 64 * MiB)
        dense = routed_chunks_per_pass(QWEN_05B, True, 4096, 2.0, 64 * MiB)

        # Assert
        assert frozen == 0
        assert dense == 0


# Packed tokens per trainer of Countdown b64 (128 rows) and b60 (96 rows) at 1536,
# and of Sudoku b12 (96 rows) at 4096.
A64_TOKENS = 128 * 1536 * ROW_FILL
A60_TOKENS = 96 * 1536 * ROW_FILL
B_TOKENS = 96 * 4096 * ROW_FILL


class TestLearnCostMeasuredRanking:
    # Each pair is (faster, slower) as measured on 8x A100-80 for Qwen3-4B LoRA:
    # (gradient_checkpointing, checkpoint_every_n_blocks, passes, tokens, residency).
    # A frozen base whole on every rank gathers no base weights, so it prices as None.
    @pytest.mark.parametrize(
        ("faster", "slower"),
        [
            (
                (False, 1, math.ceil(A60_TOKENS / 14336), A60_TOKENS, None),
                (True, 4, math.ceil(A60_TOKENS / 19456), A60_TOKENS, None),
            ),
            (
                (False, 1, math.ceil(A60_TOKENS / 14336), A60_TOKENS, None),
                (False, 1, math.ceil(A60_TOKENS / 16384), A60_TOKENS, "reshard"),
            ),
            (
                (False, 1, math.ceil(A64_TOKENS / 14336), A64_TOKENS, None),
                (False, 1, math.ceil(A64_TOKENS / 15360), A64_TOKENS, "reshard"),
            ),
            (
                (False, 1, math.ceil(A64_TOKENS / 15360), A64_TOKENS, "reshard"),
                (False, 1, math.ceil(A64_TOKENS / 12288), A64_TOKENS, "reshard"),
            ),
            (
                (False, 1, 8, A64_TOKENS, None),
                (False, 1, 16, A64_TOKENS, None),
            ),
            (
                (False, 1, 16, A64_TOKENS, None),
                (False, 1, 16, A64_TOKENS, "reshard"),
            ),
            (
                (False, 1, 16, A64_TOKENS, "reshard"),
                (True, 1, 16, A64_TOKENS, "reshard"),
            ),
            (
                (False, 1, 16, A64_TOKENS, "reshard"),
                (True, 1, 1, A64_TOKENS, "reshard"),
            ),
            (
                (True, 4, math.ceil(B_TOKENS / 19456), B_TOKENS, None),
                (False, 1, math.ceil(B_TOKENS / 16384), B_TOKENS, "reshard"),
            ),
        ],
        ids=[
            "a60-off-14336-beats-every-4th-19456",
            "a60-whole-base-14336-beats-reshard-16384",
            "a64-whole-base-14336-beats-reshard-15360",
            "a64-15360-beats-12288",
            "a64-16-rows-beat-8-rows",
            "a64-whole-base-beats-reshard",
            "a64-off-beats-every-block",
            "a64-off-16-passes-beats-every-block-1-pass",
            "b-every-4th-whole-base-19456-beats-reshard-16384",
        ],
    )
    def test_faster_measured_run_costs_less(self, faster: tuple, slower: tuple) -> None:
        # Act
        faster_cost = learn_cost(36, *faster, SM80_LEARN_COSTS)
        slower_cost = learn_cost(36, *slower, SM80_LEARN_COSTS)

        # Assert
        assert faster_cost < slower_cost


class TestLearnCostH100MeasuredRanking:
    # Each pair is (faster, slower) as measured on 8x H100 80GB HBM3 for Qwen3-4B
    # LoRA Countdown b64 on four trainers, same fields as the A100 ranking.
    @pytest.mark.parametrize(
        ("faster", "slower"),
        [
            ((True, 1, 1, A64_TOKENS, None), (True, 2, 5, A64_TOKENS, None)),
            ((True, 2, 5, A64_TOKENS, None), (True, 4, 8, A64_TOKENS, None)),
            ((True, 1, 1, A64_TOKENS, None), (False, 1, 10, A64_TOKENS, None)),
            ((False, 1, 10, A64_TOKENS, None), (False, 1, 20, A64_TOKENS, None)),
            (
                (False, 1, 10, A64_TOKENS, "gathered"),
                (False, 1, 10, A64_TOKENS, "reshard"),
            ),
        ],
        ids=[
            "every-block-1-pass-beats-every-2nd-5-passes",
            "every-2nd-5-passes-beats-every-4th-8-passes",
            "every-block-1-pass-beats-off-10-passes",
            "off-10-passes-beats-off-20-passes",
            "gathered-beats-reshard",
        ],
    )
    def test_faster_measured_run_costs_less(self, faster: tuple, slower: tuple) -> None:
        # Act
        faster_cost = learn_cost(36, *faster, SM90_LEARN_COSTS)
        slower_cost = learn_cost(36, *slower, SM90_LEARN_COSTS)

        # Assert
        assert faster_cost < slower_cost

    def test_one_checkpointed_pass_beats_sixteen_uncheckpointed_unlike_a100(self):
        # Arrange
        one_pass = (True, 1, 1, A64_TOKENS, None)
        sixteen_passes = (False, 1, 16, A64_TOKENS, None)

        # Act
        h100 = [
            learn_cost(36, *run, SM90_LEARN_COSTS) for run in (one_pass, sixteen_passes)
        ]
        a100 = [
            learn_cost(36, *run, SM80_LEARN_COSTS) for run in (one_pass, sixteen_passes)
        ]

        # Assert
        assert h100[0] < h100[1]
        assert a100[0] > a100[1]


class TestResolveTrainingMemory:
    def test_ample_budget_picks_a_fitting_plan_cheaper_than_full_recompute(self):
        # Act
        plan = resolve_training_memory(
            DENSE_MODEL, budget_device(80 * GiB), dense_settings(), MemoryPins(), 16
        )

        # Assert
        one_row_full_recompute = learn_cost(
            QWEN_05B.n_layers, True, 1, 16, plan.tokens, None, SM80_LEARN_COSTS
        )
        assert plan.breakdown.fits_with_buffer
        assert 16 % plan.micro_batch_size_per_gpu == 0
        assert plan.cost < one_row_full_recompute

    def test_h100_runs_fewer_micro_batches_and_each_pick_is_cheapest_on_its_gpu(
        self,
    ):
        # Arrange
        a100, h100 = (
            DeviceSpec(
                total_bytes=80 * GiB,
                available_bytes=12 * GiB,
                name="budget",
                compute_capability=capability,
            )
            for capability in ((8, 0), (9, 0))
        )

        def pinned_to(plan: TrainingMemoryPlan) -> MemoryPins:
            return MemoryPins(
                gradient_checkpointing=plan.gradient_checkpointing,
                micro_batch_size_per_gpu=plan.micro_batch_size_per_gpu,
            )

        # Act
        a100_plan = resolve_training_memory(
            DENSE_MODEL, a100, dense_settings(), MemoryPins(), 64
        )
        h100_plan = resolve_training_memory(
            DENSE_MODEL, h100, dense_settings(), MemoryPins(), 64
        )
        h100_pick_on_a100 = resolve_training_memory(
            DENSE_MODEL, a100, dense_settings(), pinned_to(h100_plan), 64
        )
        a100_pick_on_h100 = resolve_training_memory(
            DENSE_MODEL, h100, dense_settings(), pinned_to(a100_plan), 64
        )

        # Assert
        for plan in (a100_plan, h100_plan):
            assert plan.breakdown.fits_with_buffer
            assert 64 % plan.micro_batch_size_per_gpu == 0
        assert h100_plan.micro_batch_size_per_gpu > a100_plan.micro_batch_size_per_gpu
        assert a100_plan.cost < h100_pick_on_a100.cost
        assert h100_plan.cost < a100_pick_on_h100.cost

    def test_picked_micro_batch_is_the_largest_that_fits_its_rungs(self):
        # Arrange
        device = budget_device(12 * GiB)

        # Act
        plan = resolve_training_memory(
            DENSE_MODEL, device, dense_settings(), MemoryPins(), 64
        )

        # Assert
        rows = micro_batch_row_candidates(64)
        larger = rows[rows.index(plan.micro_batch_size_per_gpu) + 1]
        over = estimate_training(
            DENSE_MODEL,
            device,
            plan.settings.model_copy(update={"micro_batch_size": larger}),
        )
        assert plan.breakdown.fits_with_buffer
        assert not over.fits_with_buffer

    def test_sharded_speed_rungs_come_before_any_last_resort(self):
        # Arrange: a set chunk keeps the run on FSDP; one checkpointed
        # resharded row fits with the buffer.
        settings = dense_settings(routed_expert_chunk_mib=512)
        usable = buffered(dense_peak(settings, sharded=True))

        # Act
        plan = resolve_training_memory(
            DENSE_MODEL, budget_device(usable), settings, MemoryPins(), 64
        )

        # Assert
        assert plan.fsdp is not None
        assert plan.breakdown.fits_with_buffer
        assert plan.fsdp.optim_cpu_offload is False
        assert plan.fsdp.activation_offload is False

    def test_pins_are_kept(self):
        # Arrange
        settings = dense_settings(checkpoint_every_n_blocks=3)
        pins = MemoryPins(gradient_checkpointing=True, micro_batch_size_per_gpu=2)

        # Act
        plan = resolve_training_memory(
            DENSE_MODEL, budget_device(80 * GiB), settings, pins, 16
        )

        # Assert
        assert plan.gradient_checkpointing is True
        assert plan.checkpoint_every_n_blocks == 3
        assert plan.micro_batch_size_per_gpu == 2

    def test_ddp_checkpoints_every_block_or_none(self):
        # Arrange
        settings = TrainingSettings(max_model_len=4096, micro_batch_size=1)

        # Act
        plan = resolve_training_memory(
            DENSE_MODEL,
            budget_device(buffered(dense_peak(dense_settings(1), sharded=False))),
            settings,
            MemoryPins(),
            64,
        )

        # Assert
        assert plan.fsdp is None
        assert plan.checkpoint_every_n_blocks == 1

    def test_packed_budget_steps_by_1024_tokens_above_the_longest_row(self):
        # Arrange
        settings = dense_settings().model_copy(
            update={"max_model_len": 1536, "micro_batch_tokens": 1536}
        )

        # Act
        plan = resolve_training_memory(
            DENSE_MODEL, budget_device(10 * GiB), settings, MemoryPins(), 128
        )

        # Assert
        budget = plan.micro_batch_tokens_per_gpu
        assert budget is not None
        assert budget in (1536, 128 * 1536) or budget % 1024 == 0
        assert plan.micro_batch_size_per_gpu * 1536 <= budget

    def test_pinned_token_budget_is_cut_to_the_mini_batch(self):
        # Arrange: 8 rows of 1536 tokens hold at most 12288 tokens.
        settings = dense_settings().model_copy(
            update={"max_model_len": 1536, "micro_batch_tokens": 1536}
        )
        pins = MemoryPins(micro_batch_tokens_per_gpu=65536)

        # Act
        plan = resolve_training_memory(
            DENSE_MODEL, budget_device(80 * GiB), settings, pins, 8
        )

        # Assert
        assert plan.micro_batch_tokens_per_gpu == 8 * 1536
        assert plan.settings.micro_batch_tokens == 8 * 1536

    def test_pinned_token_budget_inside_the_mini_batch_is_kept(self):
        settings = dense_settings().model_copy(
            update={"max_model_len": 1536, "micro_batch_tokens": 1536}
        )
        pins = MemoryPins(micro_batch_tokens_per_gpu=4096)

        plan = resolve_training_memory(
            DENSE_MODEL, budget_device(80 * GiB), settings, pins, 8
        )

        assert plan.micro_batch_tokens_per_gpu == 4096


class TestResolveTrainingMemoryLastResorts:
    def test_ample_budget_turns_on_none(self, caplog):
        # Act
        with caplog.at_level(logging.INFO):
            plan = resolve_floor(moe_floor_settings(), 80 * GiB)

        # Assert
        assert plan.fsdp.routed_expert_chunk_mib == 512
        assert plan.fsdp.optim_cpu_offload is False
        assert plan.fsdp.activation_offload is False
        assert "turning on" not in caplog.text

    def test_resharding_gives_way_before_any_last_resort(self):
        # Arrange: 512 MiB with the optimizer on the GPU fits only resharded.
        usable = buffered(moe_peak(512, False))

        # Act
        plan = resolve_floor(moe_floor_settings(), usable)

        # Assert
        assert plan.fsdp.reshard_after_forward is True
        assert plan.fsdp.routed_expert_chunk_mib == 512
        assert plan.fsdp.optim_cpu_offload is False
        assert plan.fsdp.activation_offload is False

    def test_unset_chunk_takes_the_largest_that_fits(self):
        # Arrange: 128 MiB fits with the optimizer on the GPU, 256 MiB does not.
        usable = (buffered(moe_peak(128, False)) + buffered(moe_peak(256, False))) // 2

        # Act
        plan = resolve_floor(moe_floor_settings(optim_cpu_offload=False), usable)

        # Assert
        assert plan.fsdp.routed_expert_chunk_mib == 128

    def test_gpu_optimizer_needs_the_underprediction_buffer(self):
        # Arrange
        settings = moe_floor_settings(routed_expert_chunk_mib=64)
        gpu_peak = moe_peak(64, False)

        # Act
        unbuffered = resolve_floor(settings, gpu_peak)
        roomy = resolve_floor(settings, buffered(gpu_peak))

        # Assert
        assert unbuffered.fsdp.optim_cpu_offload is True
        assert roomy.fsdp.optim_cpu_offload is False

    def test_last_resorts_turn_on_in_order_and_are_logged(self, caplog):
        # Arrange: only activation offload brings the floor under the GPU.
        usable = buffered(moe_peak(64, True, activation_offload=True))
        assert usable < buffered(moe_peak(64, True))

        # Act
        with caplog.at_level(logging.INFO):
            plan = resolve_floor(moe_floor_settings(), usable)

        # Assert
        turned_on = [
            record.args[0]
            for record in caplog.records
            if record.getMessage().startswith("Training memory: turning on")
        ]
        assert turned_on == [
            "fsdp.routed_expert_chunk_mib=256",
            "fsdp.routed_expert_chunk_mib=128",
            "fsdp.routed_expert_chunk_mib=64",
            "fsdp.optim_cpu_offload=True",
            "fsdp.activation_offload=True",
        ]
        assert plan.fsdp.routed_expert_chunk_mib == 64
        assert plan.fsdp.optim_cpu_offload is True
        assert plan.fsdp.activation_offload is True

    def test_activation_offload_is_priced_at_the_micro_batch_floor(self):
        # Arrange: one row fits without activation offload; four rows do not.
        usable = buffered(moe_peak(64, True)) + 1

        # Act
        plan = resolve_training_memory(
            MOE_MODEL,
            budget_device(usable),
            moe_floor_settings(),
            MemoryPins(gradient_checkpointing=True),
            4,
        )

        # Assert
        assert plan.micro_batch_size_per_gpu == 1
        assert plan.fsdp.activation_offload is False

    def test_lightest_plan_inside_the_buffer_is_kept(self):
        # Arrange: the lightest floor fits the card but not with the buffer.
        usable = moe_peak(64, True, activation_offload=True)

        # Act
        plan = resolve_floor(moe_floor_settings(), usable)

        # Assert
        assert plan.breakdown.fits
        assert not plan.breakdown.fits_with_buffer
        assert plan.fsdp.activation_offload is True

    def test_raises_when_nothing_fits(self):
        # Arrange
        usable = moe_peak(64, True, activation_offload=True) // 2

        # Act / Assert
        with pytest.raises(
            TrainingMemoryUnfitError, match="at the lightest settings"
        ) as raised:
            resolve_floor(moe_floor_settings(), usable)
        assert not raised.value.plan.breakdown.fits

    def test_full_cpu_offload_never_offloads_the_optimizer_again(self):
        # Act
        plan = resolve_floor(moe_floor_settings(cpu_offload=True), 80 * GiB)

        # Assert
        assert plan.fsdp.optim_cpu_offload is False

    def test_set_values_are_kept(self):
        # Arrange
        settings = moe_floor_settings(
            reshard_after_forward=True,
            activation_offload=True,
            routed_expert_chunk_mib=128,
            optim_cpu_offload=False,
        )

        # Act
        plan = resolve_floor(settings, 80 * GiB)

        # Assert
        assert plan.fsdp.reshard_after_forward is True
        assert plan.fsdp.activation_offload is True
        assert plan.fsdp.routed_expert_chunk_mib == 128
        assert plan.fsdp.optim_cpu_offload is False

    def test_same_inputs_pick_the_same_plan(self):
        # Arrange
        usable = (buffered(moe_peak(128, True)) + buffered(moe_peak(256, True))) // 2

        # Act
        plans = [resolve_floor(moe_floor_settings(), usable) for _ in range(3)]

        # Assert
        assert plans[0] == plans[1] == plans[2]


def fsdp_only(settings: TrainingSettings) -> TrainingSettings:
    """``settings`` with a set field that keeps the search on FSDP; a dense model ignores the chunk."""
    return settings.model_copy(
        update={"fsdp": replace(settings.fsdp, routed_expert_chunk_mib=512)}
    )


def ddp_only(settings: TrainingSettings) -> TrainingSettings:
    """``settings`` without FSDP, so the search prices DDP alone."""
    return settings.model_copy(update={"fsdp": None})


class TestResolveTrainingMemoryDataParallel:
    @pytest.mark.parametrize(
        "pins",
        [
            MemoryPins(),
            MemoryPins(gradient_checkpointing=True),
            MemoryPins(gradient_checkpointing=False),
        ],
        ids=["searched", "checkpointed", "uncheckpointed"],
    )
    @pytest.mark.parametrize("usable_gib", [12, 80])
    def test_pick_fits_and_costs_no_more_than_either_mode(self, pins, usable_gib):
        # Arrange
        device = budget_device(usable_gib * GiB)
        settings = dense_settings()

        # Act
        plan = resolve_training_memory(DENSE_MODEL, device, settings, pins, 16)
        ddp = resolve_training_memory(DENSE_MODEL, device, ddp_only(settings), pins, 16)
        sharded = resolve_training_memory(
            DENSE_MODEL, device, fsdp_only(settings), pins, 16
        )

        # Assert
        assert plan.breakdown.fits_with_buffer
        assert plan.cost <= min(ddp.cost, sharded.cost)
        assert (plan.fsdp is None) == (ddp.cost <= sharded.cost)

    @pytest.mark.parametrize("n_training_gpus", [1, 4])
    def test_trains_ddp_when_ddp_is_cheapest(self, n_training_gpus):
        # Arrange: without checkpointing both modes run the same micro-batch,
        # and only FSDP pays a frozen-base gather.
        settings = dense_settings(n_training_gpus)
        pins = MemoryPins(gradient_checkpointing=False, micro_batch_size_per_gpu=1)
        device = budget_device(80 * GiB)

        # Act
        plan = resolve_training_memory(DENSE_MODEL, device, settings, pins, 4)
        sharded = resolve_training_memory(
            DENSE_MODEL, device, fsdp_only(settings), pins, 4
        )

        # Assert
        assert plan.fsdp is None
        assert plan.settings.fsdp is None
        assert plan.routed_expert_chunk_mib is None
        assert plan.breakdown.fits_with_buffer
        assert plan.cost < sharded.cost
        assert plan.describe().startswith("DDP, ")

    def test_shards_when_ddp_needs_heavier_recompute_that_costs_more(self):
        # Arrange: one checkpointed 4096-token row. DDP checkpoints every
        # block; FSDP may checkpoint every 4th and pay a gather.
        settings = dense_settings()
        pins = MemoryPins(gradient_checkpointing=True, micro_batch_size_per_gpu=1)
        device = budget_device(80 * GiB)

        # Act
        plan = resolve_training_memory(DENSE_MODEL, device, settings, pins, 1)
        ddp = resolve_training_memory(DENSE_MODEL, device, ddp_only(settings), pins, 1)

        # Assert
        assert ddp.breakdown.fits_with_buffer
        assert ddp.checkpoint_every_n_blocks == 1
        assert plan.fsdp is not None
        assert plan.breakdown.fits_with_buffer
        assert plan.checkpoint_every_n_blocks > 1
        assert plan.cost < ddp.cost
        assert plan.describe().startswith("FSDP, ")

    def test_shards_when_no_ddp_plan_fits_with_the_buffer(self):
        # Arrange: DDP fits the card but not with the buffer.
        settings = dense_settings()
        usable = dense_peak(settings, sharded=False)

        # Act
        plan = resolve_floor(settings, usable)

        # Assert
        assert plan.fsdp is not None
        assert plan.breakdown.fits
        assert plan.describe().startswith("FSDP, ")

    def test_ddp_lightest_plan_inside_the_buffer_is_kept(self):
        # Arrange: the checkpointed one-row DDP plan fits the card but not
        # with the buffer.
        usable = dense_peak(dense_settings(), sharded=False)
        pins = MemoryPins(gradient_checkpointing=True, micro_batch_size_per_gpu=1)

        # Act
        plan = resolve_training_memory(
            DENSE_MODEL, budget_device(usable), ddp_only(dense_settings()), pins, 1
        )

        # Assert
        assert plan.fsdp is None
        assert plan.breakdown.fits
        assert not plan.breakdown.fits_with_buffer

    def test_ddp_raises_when_nothing_fits(self):
        # Arrange
        usable = dense_peak(dense_settings(), sharded=False) // 2
        pins = MemoryPins(gradient_checkpointing=True, micro_batch_size_per_gpu=1)

        # Act / Assert
        with pytest.raises(
            TrainingMemoryUnfitError, match="at the lightest settings"
        ) as raised:
            resolve_training_memory(
                DENSE_MODEL, budget_device(usable), ddp_only(dense_settings()), pins, 1
            )
        assert raised.value.plan.fsdp is None

    @pytest.mark.parametrize(
        "settings",
        [
            moe_floor_settings(),
            dense_settings(cpu_offload=True),
            dense_settings(reshard_after_forward=False),
        ],
        ids=["expert-parallel", "cpu-offload-set", "residency-set"],
    )
    def test_stays_fsdp_when_the_model_is_split_or_a_field_is_set(self, settings):
        # Act
        plan = resolve_floor(settings, 80 * GiB)

        # Assert
        assert plan.fsdp is not None
        assert plan.breakdown.fits_with_buffer

    def test_ddp_costs_less_than_fsdp_at_the_same_rungs(self):
        # Arrange
        pins = MemoryPins(gradient_checkpointing=False, micro_batch_size_per_gpu=1)
        device = budget_device(80 * GiB)

        # Act
        ddp = resolve_training_memory(DENSE_MODEL, device, dense_settings(), pins, 4)
        sharded = resolve_training_memory(
            DENSE_MODEL, device, dense_settings(reshard_after_forward=False), pins, 4
        )

        # Assert
        assert ddp.fsdp is None
        assert sharded.fsdp is not None
        assert ddp.cost < sharded.cost

    def test_trains_expert_lora_moe_ddp_on_the_largest_routed_chunk(self):
        # Arrange: measured on 2x A100-80, GC off: DDP at 512 MiB learns 10%
        # faster than at 64 MiB and matches FSDP at 512 MiB.
        settings = granite_grpo(GRANITE_MOE, False, 65536).model_copy(
            update={"fsdp": FSDPConfig()}
        )
        pins = MemoryPins(gradient_checkpointing=False)
        device = training_gpu("NVIDIA A100-SXM4-80GB", A100_80GB_CUDA_BYTES)
        model = catalog_spec(GRANITE_MOE)

        # Act
        plan = resolve_training_memory(model, device, settings, pins, 128)
        sharded = resolve_training_memory(model, device, fsdp_only(settings), pins, 128)

        # Assert
        assert plan.fsdp is None
        assert plan.routed_expert_chunk_mib == 512
        assert plan.settings.routed_expert_chunk_mib == 512
        assert "routed_expert_chunk_mib=512" in plan.describe()
        assert plan.breakdown.fits_with_buffer
        assert sharded.routed_expert_chunk_mib is None
        assert plan.cost < sharded.cost

    def test_ddp_turns_on_a_smaller_routed_chunk_when_nothing_fits(self):
        # Arrange: one checkpointed 64k-token row; the budget holds the
        # 256 MiB chunk with the buffer but not the 512 MiB one.
        settings = ddp_only(granite_grpo(GRANITE_MOE, True, 65536)).model_copy(
            update={"max_model_len": 65536}
        )
        pins = MemoryPins(gradient_checkpointing=True, micro_batch_size_per_gpu=1)
        model = catalog_spec(GRANITE_MOE)
        peaks = {
            mib: resolve_training_memory(
                model,
                budget_device(80 * GiB),
                settings.model_copy(update={"routed_expert_chunk_mib": mib}),
                pins,
                1,
            ).breakdown.total_bytes
            for mib in (256, 512)
        }
        usable = (buffered(peaks[256]) + buffered(peaks[512])) // 2

        # Act
        plan = resolve_training_memory(model, budget_device(usable), settings, pins, 1)

        # Assert
        assert peaks[256] < peaks[512]
        assert plan.fsdp is None
        assert plan.routed_expert_chunk_mib == 256
        assert plan.breakdown.fits_with_buffer

    def test_keeps_a_set_ddp_routed_chunk(self):
        # Arrange
        settings = ddp_only(granite_grpo(GRANITE_MOE, False, 65536)).model_copy(
            update={"routed_expert_chunk_mib": 128}
        )
        device = training_gpu("NVIDIA A100-SXM4-80GB", A100_80GB_CUDA_BYTES)

        # Act
        plan = resolve_training_memory(
            catalog_spec(GRANITE_MOE),
            device,
            settings,
            MemoryPins(gradient_checkpointing=False),
            128,
        )

        # Assert
        assert plan.routed_expert_chunk_mib == 128
        assert plan.settings.routed_expert_chunk_mib == 128

    def test_fsdp_pick_costs_no_more_than_resharding(self):
        # Act
        plan = resolve_floor(moe_floor_settings(), 80 * GiB)
        resharded = resolve_floor(
            moe_floor_settings(reshard_after_forward=True), 80 * GiB
        )

        # Assert
        assert plan.breakdown.fits_with_buffer
        assert plan.cost < resharded.cost


class TestResolveTrainingMemorySuperVlVwa:
    """VWA launch shape: 16 trainer GPUs, rows up to 24,000 + 5,120 tokens."""

    @pytest.mark.parametrize(
        ("name", "cuda_bytes"),
        [
            ("NVIDIA A100-SXM4-80GB", A100_80GB_CUDA_BYTES),
            ("NVIDIA H100 80GB HBM3", H100_80GB_CUDA_BYTES),
        ],
    )
    def test_offloads_activations_on_the_smallest_chunk(self, name, cuda_bytes):
        # Act
        plan = resolve_training_memory(
            SUPER_VL_MODEL,
            training_gpu(name, cuda_bytes),
            super_vl_vwa(),
            MemoryPins(gradient_checkpointing=True, micro_batch_size_per_gpu=1),
            1,
        )

        # Assert
        assert plan.fsdp.activation_offload is True
        assert plan.fsdp.routed_expert_chunk_mib == 64
        assert plan.fsdp.optim_cpu_offload is True

    def test_h100_pick_leaves_room_for_the_busiest_rank(self):
        # Arrange: EP routing spread A100 ranks 1.5 GiB over one learn step.
        rank_spread = int(1.5 * GiB)
        device = training_gpu("NVIDIA H100 80GB HBM3", H100_80GB_CUDA_BYTES)

        # Act
        plan = resolve_training_memory(
            SUPER_VL_MODEL,
            device,
            super_vl_vwa(),
            MemoryPins(gradient_checkpointing=True, micro_batch_size_per_gpu=1),
            1,
        )

        # Assert
        assert plan.breakdown.total_bytes + rank_spread < H100_80GB_CUDA_BYTES

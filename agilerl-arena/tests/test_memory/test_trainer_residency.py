# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Base-weight and allocator-slack terms on one training GPU."""

import pytest

from agilerl.arena.memory import formulas, fsdp_gather
from agilerl.arena.memory.specs import ModelArch, ModelSpec, TrainingSettings
from agilerl.arena.memory.trainer_residency import (
    fsdp_cpu_offload_host_bytes,
    fsdp_forward_gather_bytes,
    trainer_allocator_slack,
    trainer_base_bytes,
)
from agilerl.arena.models.fsdp import FSDPConfig
from agilerl.arena.models.model_info import SUPPORTED_MODEL_INFO

QWEN_05B = ModelArch(
    n_layers=24,
    hidden_size=896,
    intermediate_size=4864,
    n_heads=14,
    n_kv_heads=2,
    head_dim=64,
    vocab_size=151936,
    tied_embeddings=True,
    attn_bias=True,
)
TORCH_SIDE = 10 * 2**30


class TestTrainerBaseBytes:
    def test_ddp_base_holds_every_weight(self):
        # Arrange
        model = ModelSpec(model_id="qwen-0.5b", arch=QWEN_05B)
        counts = formulas.param_counts(QWEN_05B)
        settings = TrainingSettings(fsdp=None, n_training_gpus=4)

        # Act
        base = trainer_base_bytes(model, settings, counts, 4, TORCH_SIDE)

        # Assert: bf16 weights, no shard.
        assert base == counts.total * 2

    def test_sharded_base_holds_less_than_every_weight(self):
        model = ModelSpec(model_id="qwen-0.5b", arch=QWEN_05B)
        counts = formulas.param_counts(QWEN_05B)
        settings = TrainingSettings(
            fsdp=FSDPConfig(reshard_after_forward=True),
            n_training_gpus=4,
        )

        base = trainer_base_bytes(model, settings, counts, 4, TORCH_SIDE)

        assert base < counts.total * 2

    def test_shared_mamba_groups_keep_their_gathered_rows(self):
        # Arrange
        granite = ModelArch.from_hf_config(
            SUPPORTED_MODEL_INFO["ibm-granite/granite-4.0-h-tiny"].config
        )
        split = granite.model_copy(update={"mamba_n_groups": 2})
        counts = formulas.param_counts(granite)
        settings = TrainingSettings(fsdp=FSDPConfig(tp=2), n_training_gpus=4)

        # Act
        shared = trainer_base_bytes(
            ModelSpec(model_id="granite", arch=granite), settings, counts, 2, 0.0
        )
        unshared = trainer_base_bytes(
            ModelSpec(model_id="granite", arch=split), settings, counts, 2, 0.0
        )

        # Assert: bf16 B/C rows of one group, 2 x 128 state rows x (1536 + 4 + 1).
        assert shared - unshared == 36 * 256 * 1541 * 2

    def test_expert_lora_without_reshard_holds_one_expert_layer(self):
        # Arrange
        nemotron = ModelArch.from_hf_config(
            SUPPORTED_MODEL_INFO[
                "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
            ].config
        )
        model = ModelSpec(model_id="nemotron", arch=nemotron)
        counts = formulas.param_counts(nemotron)
        fsdp = FSDPConfig(ep=2, reshard_after_forward=False)
        base_only = TrainingSettings(fsdp=fsdp, n_training_gpus=4)
        expert_lora = base_only.model_copy(update={"lora_packed_target_matrices": 2})

        # Act
        held = trainer_base_bytes(model, base_only, counts, 4, 0.0)
        released = trainer_base_bytes(model, expert_lora, counts, 4, 0.0)

        # Assert: bf16, all but one MoE layer of this rank's half of the experts.
        layer = formulas.routed_expert_params_per_layer(nemotron) // 2
        assert held - released == (nemotron.moe_layers - 1) * layer * 2


class TestFsdpCpuOffloadHostBytes:
    def test_cpu_offload_matches_the_resident_shard(self):
        # Arrange
        model = ModelSpec(model_id="qwen-0.5b", arch=QWEN_05B)
        counts = formulas.param_counts(QWEN_05B)
        off = TrainingSettings(
            fsdp=FSDPConfig(reshard_after_forward=True, cpu_offload=True),
            n_training_gpus=4,
        )
        on = TrainingSettings(
            fsdp=FSDPConfig(reshard_after_forward=True, cpu_offload=False),
            n_training_gpus=4,
        )

        # Act
        host = fsdp_cpu_offload_host_bytes(model, off, counts, 4)
        gpu_delta = trainer_base_bytes(
            model, on, counts, 4, TORCH_SIDE
        ) - trainer_base_bytes(model, off, counts, 4, TORCH_SIDE)

        # Assert
        assert host == gpu_delta
        assert host > 0
        assert fsdp_cpu_offload_host_bytes(model, on, counts, 4) == 0.0

    def test_ddp_pins_nothing(self):
        model = ModelSpec(model_id="qwen-0.5b", arch=QWEN_05B)
        counts = formulas.param_counts(QWEN_05B)
        settings = TrainingSettings(fsdp=None, n_training_gpus=4)

        assert fsdp_cpu_offload_host_bytes(model, settings, counts, 4) == 0.0


class TestFsdpForwardGatherBytes:
    def test_resharded_forward_retains_one_bf16_block(self):
        counts = formulas.param_counts(QWEN_05B)
        settings = TrainingSettings(
            fsdp=FSDPConfig(reshard_after_forward=True, cpu_offload=False),
            n_training_gpus=4,
        )
        block = fsdp_gather.largest_block_params(counts, QWEN_05B)

        retained, staging = fsdp_forward_gather_bytes(QWEN_05B, settings, counts, 4)

        assert (retained, staging) == (block * 2, 0.0)

    def test_cpu_offload_stages_the_block_shard_on_the_gpu(self):
        counts = formulas.param_counts(QWEN_05B)
        settings = TrainingSettings(
            fsdp=FSDPConfig(reshard_after_forward=True, cpu_offload=True),
            n_training_gpus=4,
        )
        block = fsdp_gather.largest_block_params(counts, QWEN_05B)

        retained, staging = fsdp_forward_gather_bytes(QWEN_05B, settings, counts, 4)

        assert (retained, staging) == (block * 2, block * 2 / 4)

    def test_resident_frozen_tp_shards_leave_the_retained_output(self):
        # Arrange
        arch = QWEN_05B.model_copy(update={"tensor_parallel_plan": True})
        counts = formulas.param_counts(arch)
        settings = TrainingSettings(
            fsdp=FSDPConfig(tp=2, shard_frozen_tp=False, cpu_offload=True),
            n_training_gpus=4,
        )

        # Act
        retained, staging = fsdp_forward_gather_bytes(arch, settings, counts, 2)

        # Assert: only the 1829 per-layer norm elements gather, in bf16.
        assert settings.fsdp.frozen_tp_resident
        assert (retained, staging) == (1829 * 2, 1829 * 2 / 2)

    def test_ddp_and_one_gpu_gather_nothing(self):
        counts = formulas.param_counts(QWEN_05B)
        ddp = TrainingSettings(fsdp=None, n_training_gpus=4)
        single = TrainingSettings(fsdp=FSDPConfig(cpu_offload=True))

        assert fsdp_forward_gather_bytes(QWEN_05B, ddp, counts, 4) == (0.0, 0.0)
        assert fsdp_forward_gather_bytes(QWEN_05B, single, counts, 1) == (0.0, 0.0)


class TestTrainerAllocatorSlack:
    def test_fsdp_adds_the_all_gather_pool(self):
        # Arrange
        counts = formulas.param_counts(QWEN_05B)
        settings = TrainingSettings(fsdp=FSDPConfig(), n_training_gpus=4)
        fsdp = settings.fsdp
        pool = fsdp_gather.fsdp_gather_pool_params(
            counts,
            QWEN_05B,
            fsdp.ep,
            4,
        )

        # Act
        reserve, gather_pool = trainer_allocator_slack(
            QWEN_05B, settings, counts, 4, TORCH_SIDE
        )

        # Assert: bf16 weights, no fp32 norm temporaries.
        assert reserve == pytest.approx(
            TORCH_SIDE * formulas.ALLOCATOR_RESERVE_FRACTION_FSDP
        )
        assert gather_pool == pool * 2

    def test_ddp_has_no_all_gather_pool(self):
        counts = formulas.param_counts(QWEN_05B)
        settings = TrainingSettings(fsdp=None, n_training_gpus=4)

        reserve, gather_pool = trainer_allocator_slack(
            QWEN_05B, settings, counts, 4, TORCH_SIDE
        )

        assert reserve == pytest.approx(
            TORCH_SIDE * formulas.ALLOCATOR_RESERVE_FRACTION_DATA_PARALLEL
        )
        assert gather_pool == 0.0

    def test_single_gpu_has_no_all_gather_pool(self):
        counts = formulas.param_counts(QWEN_05B)

        reserve, gather_pool = trainer_allocator_slack(
            QWEN_05B, TrainingSettings(), counts, 1, TORCH_SIDE
        )

        assert reserve == pytest.approx(
            TORCH_SIDE * formulas.ALLOCATOR_RESERVE_FRACTION
        )
        assert gather_pool == 0.0

# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Base-weight and allocator-slack terms on one training GPU."""

import pytest

from agilerl.arena.memory import formulas, fsdp_gather
from agilerl.arena.memory.specs import ModelArch, ModelSpec, TrainingSettings
from agilerl.arena.memory.trainer_residency import (
    trainer_allocator_slack,
    trainer_base_bytes,
)
from agilerl.arena.models.fsdp import FSDPConfig

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

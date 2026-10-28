# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Base-weight and allocator-slack terms on one training GPU."""

import pytest

from agilerl.arena.memory import formulas
from agilerl.arena.memory.specs import ModelArch, TrainingSettings
from agilerl.arena.memory.trainer_residency import trainer_allocator_slack
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


class TestTrainerAllocatorSlack:
    def test_fsdp_adds_the_all_gather_pool(self):
        # Arrange
        counts = formulas.param_counts(QWEN_05B)
        settings = TrainingSettings(fsdp=FSDPConfig(), n_training_gpus=4)
        fsdp = settings.fsdp
        pool = formulas.fsdp_gather_pool_params(
            counts,
            QWEN_05B,
            fsdp.wrap_every_n_blocks,
            fsdp.prefetch_units,
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

    def test_single_gpu_has_no_all_gather_pool(self):
        counts = formulas.param_counts(QWEN_05B)

        reserve, gather_pool = trainer_allocator_slack(
            QWEN_05B, TrainingSettings(), counts, 1, TORCH_SIDE
        )

        assert reserve == pytest.approx(
            TORCH_SIDE * formulas.ALLOCATOR_RESERVE_FRACTION
        )
        assert gather_pool == 0.0

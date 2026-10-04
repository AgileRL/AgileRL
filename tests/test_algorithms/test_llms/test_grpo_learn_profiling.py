# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for ``GRPO.learn`` with ``profiling_config`` set."""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

pytest.importorskip("transformers", reason="LLM tests require transformers.")
pytest.importorskip("peft", reason="LLM tests require peft.")

from agilerl.algorithms.grpo import GRPO
from agilerl.arena.models.profiling import ProfilingConfig
from tests.test_algorithms.test_llms.llm_helpers import create_module


def cpu_grpo(**kwargs) -> GRPO:
    """Build a tiny unwrapped CPU GRPO."""
    defaults = {
        "actor_network": create_module(
            input_size=6, max_tokens=4, vocab_size=64, device="cpu"
        ),
        "pad_token_id": 63,
        "pad_token": "<pad>",
        "batch_size": 4,
        "group_size": 2,
        "max_output_tokens": 4,
        "max_model_len": 12,
        "wrap": False,
        "gradient_checkpointing": False,
        "device": "cpu",
        "use_liger_loss": False,
    }
    defaults.update(kwargs)
    return GRPO(**defaults)


def experiences(
    rewards: list[float], seq_len: int = 10, vocab_size: int = 64
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    """Random completions with full action masks, one per reward."""
    completion_ids = [
        torch.randint(0, vocab_size, (1, seq_len), dtype=torch.long) for _ in rewards
    ]
    action_masks = [torch.ones(1, seq_len - 1, dtype=torch.bool) for _ in rewards]
    return completion_ids, action_masks, torch.tensor(rewards, dtype=torch.float32)


class TestGRPOLearnProfiling:
    """``profiling_config`` traces one learn micro-batch and lets OOM propagate."""

    def test_learn_traces_only_the_configured_call(self, tmp_path: Path) -> None:
        # Arrange
        torch.manual_seed(0)
        grpo = cpu_grpo(
            micro_batch_size_per_gpu=2,
            profiling_config=ProfilingConfig(
                output_dir=str(tmp_path), torch_profile_step=2
            ),
        )
        batch = experiences([1.0, 0.0, -1.0, 2.0])

        # Act
        first = grpo.learn(batch)
        files_after_first = list(tmp_path.iterdir())
        second = grpo.learn(batch)

        # Assert
        assert files_after_first == []
        assert [path.name for path in tmp_path.iterdir()] == ["rank0_learn2_trace.json"]
        assert math.isfinite(first["loss"])
        assert math.isfinite(second["loss"])
        grpo.clean_up()

    def test_learn_reraises_out_of_memory(self, tmp_path: Path) -> None:
        # Arrange
        grpo = cpu_grpo(
            profiling_config=ProfilingConfig(
                output_dir=str(tmp_path), torch_profile_step=1
            ),
        )
        batch = experiences([1.0, -1.0, 0.0, 2.0])

        # Act / Assert
        with (
            patch.object(
                grpo, "_loss", side_effect=torch.OutOfMemoryError("expert GEMM")
            ),
            pytest.raises(torch.OutOfMemoryError, match="expert GEMM"),
        ):
            grpo.learn(batch)

        assert list(tmp_path.iterdir()) == []
        grpo.clean_up()

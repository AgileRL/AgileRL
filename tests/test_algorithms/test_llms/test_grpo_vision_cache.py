# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""GRPO learn runs a frozen vision tower once per vision row."""

from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Any

import pytest
import torch
from peft import LoraConfig

from agilerl.algorithms import GRPO
from tests.test_algorithms.test_llms.vision_helpers import (
    IMAGE_COUNTS,
    PAD_TOKEN_ID,
    TowerVisionModel,
    adapter_weights,
    learn_on_vision_episodes,
    tower,
)


def make_vision_grpo(**overrides: Any) -> GRPO:
    """Tiny fp32 GRPO on CPU with a KL reference pass over 4 episodes."""
    torch.manual_seed(0)
    kwargs: dict[str, Any] = {
        "actor_network": TowerVisionModel(),
        "pad_token_id": PAD_TOKEN_ID,
        "pad_token": "<pad>",
        "batch_size": 2,
        "group_size": 2,
        "beta": 0.01,
        "lr": 1e-2,
        "max_output_tokens": 4,
        "max_model_len": 12,
        "micro_batch_size_per_gpu": 2,
        "update_epochs": 1,
        "wrap": False,
        "gradient_checkpointing": False,
        "calc_position_embeddings": False,
        "device": "cpu",
        "use_liger_loss": False,
        "lora_config": LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["linear_1"],
            task_type="CAUSAL_LM",
            lora_dropout=0.0,
        ),
        **overrides,
    }
    return GRPO(**kwargs)


class TestGRPOLearnVisionCache:
    @pytest.mark.parametrize(
        ("segmented", "vision_rows"),
        # Segment rows add one filler row with one vision row per padded slot.
        [(True, sum(IMAGE_COUNTS) + 2), (False, sum(IMAGE_COUNTS))],
        ids=["segments", "episodes"],
    )
    def test_learn_runs_the_tower_once_per_vision_row(
        self, segmented: bool, vision_rows: int
    ) -> None:
        # Arrange
        agent = make_vision_grpo()

        # Act
        learn_on_vision_episodes(agent, segmented=segmented)

        # Assert
        assert sum(tower(agent).rows) == vision_rows

    @pytest.mark.parametrize("segmented", [True, False], ids=["segments", "episodes"])
    def test_learn_matches_the_uncached_tower(
        self, monkeypatch: pytest.MonkeyPatch, segmented: bool
    ) -> None:
        # Arrange
        cached = make_vision_grpo()
        uncached = make_vision_grpo()
        monkeypatch.setattr(uncached._vision_cache, "step", nullcontext)

        # Act
        cached_metrics = learn_on_vision_episodes(cached, segmented=segmented)
        uncached_metrics = learn_on_vision_episodes(uncached, segmented=segmented)

        # Assert: fp32 tower GEMMs over other batches may round differently in
        # the last bits.
        assert sum(tower(uncached).rows) > sum(tower(cached).rows)
        for key in ("loss", "kl"):
            assert math.isfinite(cached_metrics[key]), key
            assert math.isclose(
                cached_metrics[key], uncached_metrics[key], rel_tol=0.0, abs_tol=1e-6
            ), key
        cached_weights = adapter_weights(cached, "actor")
        uncached_weights = adapter_weights(uncached, "actor")
        assert cached_weights.keys() == uncached_weights.keys()
        for name, weight in cached_weights.items():
            assert torch.allclose(
                weight, uncached_weights[name], rtol=0.0, atol=1e-6
            ), name

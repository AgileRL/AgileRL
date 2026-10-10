# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for LLM REINFORCE learning from vision-language episodes.

Pure CPU on a tiny fp32 VL model; see :mod:`vision_helpers`.
"""

from __future__ import annotations

import math
import sys
from contextlib import nullcontext
from functools import partial
from typing import Any

import numpy as np
import pytest
import torch
import torch.distributed as dist

pytest.importorskip("transformers", reason="LLM tests require transformers.")
pytest.importorskip("peft", reason="LLM tests require peft.")

from peft import LoraConfig

from agilerl.algorithms.reinforce_llm import REINFORCE
from tests.test_algorithms.test_llms.segment_helpers import (
    pad_to_eight_rows,
    spawn_balanced_learn,
    use_fake_liger_policy_loss,
)
from tests.test_algorithms.test_llms.vision_helpers import (
    IMAGE_COUNTS,
    IMAGE_TOKEN_ID,
    PAD_TOKEN_ID,
    SEGMENTS,
    TowerVisionModel,
    VisionHiddenStatesModel,
    adapter_weights,
    expected_vision_rows,
    learn_on_vision_episodes,
    learn_rank_vision_episodes,
    routed_rows,
    tower,
    vision_config,
    vision_episodes,
    vision_model,
)


def make_vision_reinforce(**overrides: Any) -> REINFORCE:
    """Tiny fp32 REINFORCE on CPU with one optimizer step per learn over 4 episodes."""
    torch.manual_seed(0)
    kwargs: dict[str, Any] = {
        "actor_network": VisionHiddenStatesModel(vision_config()),
        "pad_token_id": PAD_TOKEN_ID,
        "pad_token": "<pad>",
        "batch_size": 4,
        "beta": 0.01,
        "lr": 1e-2,
        "max_output_tokens": 4,
        "max_model_len": 12,
        "micro_batch_size_per_gpu": 2,
        "mini_batch_size": 4,
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
    return REINFORCE(**kwargs)


class TestREINFORCELearnVision:
    @pytest.mark.parametrize("segmented", [True, False], ids=["segments", "episodes"])
    @pytest.mark.parametrize("liger", [False, True], ids=["standard", "liger"])
    def test_every_forward_gets_the_vision_rows_of_its_rows(
        self, monkeypatch: pytest.MonkeyPatch, segmented: bool, liger: bool
    ) -> None:
        # Arrange
        agent = make_vision_reinforce()
        if liger:
            use_fake_liger_policy_loss(
                agent, monkeypatch, "agilerl.algorithms.reinforce_llm"
            )
        model = vision_model(agent)
        expected = expected_vision_rows(segmented)

        # Act
        learn_on_vision_episodes(agent, segmented=segmented)

        # Assert
        routed = routed_rows(model.forwards, expected)
        for adapter in ("reference", "actor"):
            assert {tokens for name, tokens in routed if name == adapter} == set(
                expected
            ), adapter

    @pytest.mark.parametrize("segmented", [True, False], ids=["segments", "episodes"])
    def test_step_moves_the_actor_with_finite_loss(self, segmented: bool) -> None:
        # Arrange
        agent = make_vision_reinforce()
        before = adapter_weights(agent, "actor")

        # Act
        metrics = learn_on_vision_episodes(agent, segmented=segmented)

        # Assert
        for key in ("loss", "pg_loss", "kl", "entropy"):
            assert math.isfinite(metrics[key]), key
        after = adapter_weights(agent, "actor")
        assert before
        assert any(
            not torch.equal(after[name], weight) for name, weight in before.items()
        )

    @pytest.mark.parametrize("segmented", [True, False], ids=["segments", "episodes"])
    def test_images_change_log_probs(self, segmented: bool) -> None:
        # Arrange: ``entropy`` averages the actor's negative log-probs.
        agent = make_vision_reinforce()
        other = make_vision_reinforce()

        # Act
        metrics = learn_on_vision_episodes(agent, segmented=segmented)
        other_metrics = learn_on_vision_episodes(
            other, segmented=segmented, pixel_offset=1.0
        )

        # Assert
        assert metrics["entropy"] != pytest.approx(other_metrics["entropy"])

    def test_vision_filler_rows_train_with_one_vision_row(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange: 1-row micro-batches over 6 real rows, padded to 8 by a
        # simulated second rank. A filler row is the first segment's one
        # placeholder with that segment's vision row 0.
        agent = make_vision_reinforce(micro_batch_size_per_gpu=1)
        model = vision_model(agent)
        pad_to_eight_rows(monkeypatch)
        filler_tokens = (IMAGE_TOKEN_ID,)
        expected = {**expected_vision_rows(segmented=True), filler_tokens: [0]}

        # Act
        metrics = learn_on_vision_episodes(agent, segmented=True)

        # Assert
        assert metrics["train_rows_padded"] == pytest.approx(8.0)
        for key in ("loss", "pg_loss", "kl", "entropy"):
            assert math.isfinite(metrics[key]), key
        routed = routed_rows(model.forwards, expected)
        assert {adapter for adapter, tokens in routed if tokens == filler_tokens} == {
            "reference",
            "actor",
        }

    def test_vision_segments_need_the_image_token_id(self) -> None:
        # Arrange
        agent = make_vision_reinforce()
        token_ids, action_masks, rewards, turn_ids, pixel_values = vision_episodes()

        # Act / Assert
        with pytest.raises(ValueError, match="image_token_id is required"):
            agent.learn(
                (token_ids, action_masks, rewards),
                turn_ids=turn_ids,
                episode_segments=SEGMENTS,
                pixel_values=pixel_values,
                pixel_image_counts=IMAGE_COUNTS,
            )


def make_tower_vision_reinforce(**overrides: Any) -> REINFORCE:
    """:func:`make_vision_reinforce` on a :class:`TowerVisionModel` built from the same seed."""
    torch.manual_seed(0)
    return make_vision_reinforce(actor_network=TowerVisionModel(), **overrides)


class TestREINFORCELearnVisionCache:
    @pytest.mark.parametrize("segmented", [True, False], ids=["segments", "episodes"])
    def test_learn_runs_the_tower_once_per_vision_row(self, segmented: bool) -> None:
        # Arrange
        agent = make_tower_vision_reinforce()

        # Act
        learn_on_vision_episodes(agent, segmented=segmented)

        # Assert
        assert sum(tower(agent).rows) == sum(IMAGE_COUNTS)

    @pytest.mark.parametrize("segmented", [True, False], ids=["segments", "episodes"])
    def test_learn_matches_the_uncached_tower(
        self, monkeypatch: pytest.MonkeyPatch, segmented: bool
    ) -> None:
        # Arrange
        cached = make_tower_vision_reinforce()
        uncached = make_tower_vision_reinforce()
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
        for adapter in ("actor",):
            cached_weights = adapter_weights(cached, adapter)
            uncached_weights = adapter_weights(uncached, adapter)
            assert cached_weights.keys() == uncached_weights.keys()
            for name, weight in cached_weights.items():
                assert torch.allclose(
                    weight, uncached_weights[name], rtol=0.0, atol=1e-6
                ), name


@pytest.mark.skipif(
    sys.platform == "win32" or not dist.is_available(), reason="gloo unavailable"
)
class TestREINFORCELearnVisionCacheAcrossRanks:
    def test_balanced_rows_run_the_tower_once_per_vision_row(self) -> None:
        # Arrange: rank 0's 4 rows (5 vision rows) and rank 1's 2 rows (3 vision
        # rows, plus 2 filler vision rows in the no-grad forward) deal out as 3
        # and 3. Rank 0 receives a row with 2 vision rows, rank 1 two rows with 3.

        # Act
        cached = spawn_balanced_learn(
            make_tower_vision_reinforce,
            partial(learn_rank_vision_episodes, cached=True),
        )
        uncached = spawn_balanced_learn(
            make_tower_vision_reinforce,
            partial(learn_rank_vision_episodes, cached=False),
        )

        # Assert
        assert [tower_rows for (tower_rows, _), _ in cached] == [5 + 2, 5 + 3]
        for ((cached_rows, cached_losses), cached_steps), (
            (uncached_rows, uncached_losses),
            uncached_steps,
        ) in zip(cached, uncached, strict=True):
            assert uncached_rows > cached_rows
            for key, loss in cached_losses.items():
                assert math.isfinite(loss), key
                assert math.isclose(
                    loss, uncached_losses[key], rel_tol=0.0, abs_tol=1e-6
                ), key
            assert len(cached_steps) == len(uncached_steps) == 1
            for name, grad in cached_steps[0].items():
                np.testing.assert_allclose(
                    grad, uncached_steps[0][name], rtol=0.0, atol=1e-6, err_msg=name
                )

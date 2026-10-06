# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for LLM REINFORCE learning from vision-language episodes.

Pure CPU on a tiny fp32 VL model; see :mod:`vision_helpers`.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch

pytest.importorskip("transformers", reason="LLM tests require transformers.")
pytest.importorskip("peft", reason="LLM tests require peft.")

from peft import LoraConfig

from agilerl.algorithms.reinforce_llm import REINFORCE
from tests.test_algorithms.test_llms.segment_helpers import (
    pad_to_eight_rows,
    use_fake_liger_policy_loss,
)
from tests.test_algorithms.test_llms.vision_helpers import (
    IMAGE_COUNTS,
    IMAGE_TOKEN_ID,
    PAD_TOKEN_ID,
    SEGMENTS,
    VisionHiddenStatesModel,
    adapter_weights,
    expected_vision_rows,
    learn_on_vision_episodes,
    routed_rows,
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

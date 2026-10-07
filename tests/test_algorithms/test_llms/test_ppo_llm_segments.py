# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for LLM PPO training segmented episodes one segment per row.

Pure CPU on a tiny real model in fp32. The model scores each token from that
token alone, so a segment row's log-probs and values match its episode's.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch

pytest.importorskip("transformers", reason="LLM tests require transformers.")
pytest.importorskip("peft", reason="LLM tests require peft.")

from peft import LoraConfig

from agilerl.algorithms.ppo_llm import PPO as LLMPPO
from agilerl.utils.llm_utils import LEARN_PHASE_METRIC_NAMES
from tests.test_algorithms.test_llms.llm_helpers import create_value_head_module
from tests.test_algorithms.test_llms.segment_helpers import (
    EPISODE_SEGMENTS,
    MEAN_SEGMENT_ROW_ACTION_TOKENS,
    episode_sampling_logps,
    lora_weights,
    pad_to_eight_rows,
    record_step_gradients,
    segment_experiences,
    use_fake_liger_policy_loss,
)

PAD_TOKEN_ID = 63
VOCAB = 64


def _make_ppo(**overrides: Any) -> LLMPPO:
    """Tiny fp32 PPO on CPU with one optimizer step per learn over 4 episodes."""
    torch.manual_seed(0)
    kwargs: dict[str, Any] = {
        "actor_network": create_value_head_module(
            input_size=6, max_tokens=4, vocab_size=VOCAB, device="cpu"
        ),
        "pad_token_id": PAD_TOKEN_ID,
        "pad_token": "<pad>",
        "batch_size": 4,
        "lr_actor": 1e-2,
        "lr_critic": 1e-2,
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
            modules_to_save=["summary"],
        ),
        **overrides,
    }
    return LLMPPO(**kwargs)


def _experiences() -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    return segment_experiences(PAD_TOKEN_ID)


class TestPPOLearnEpisodeSegments:
    def test_trains_segment_rows_with_finite_loss(self) -> None:
        # Arrange
        agent = _make_ppo()
        before = lora_weights(agent)

        # Act
        metrics = agent.learn(_experiences(), episode_segments=EPISODE_SEGMENTS)

        # Assert
        for key in ("loss", "pg_loss", "vf_loss", "kl", "entropy"):
            assert math.isfinite(metrics[key]), key
        after = lora_weights(agent)
        assert any(not torch.equal(after[name], before[name]) for name in before)

    @pytest.mark.parametrize(
        "sampling_logps",
        [None, episode_sampling_logps(PAD_TOKEN_ID)],
        ids=["no_sampling_logps", "sampling_logps"],
    )
    @pytest.mark.parametrize("fuse", [False, True], ids=["split", "fused"])
    def test_one_micro_batch_matches_the_unsegmented_gradient(
        self,
        monkeypatch: pytest.MonkeyPatch,
        sampling_logps: list[torch.Tensor | None] | None,
        fuse: bool,
    ) -> None:
        # Arrange: every row fits one micro-batch, so returns, advantages and the
        # token-mean losses cover the same action tokens either way.
        overrides = {
            "batch_size": 8,
            "micro_batch_size_per_gpu": 8,
            "mini_batch_size": 8,
            "fuse_actor_critic_pass": fuse,
        }
        unsegmented = _make_ppo(**overrides)
        segmented = _make_ppo(**overrides)
        unsegmented_grads = record_step_gradients(unsegmented, monkeypatch)
        segmented_grads = record_step_gradients(segmented, monkeypatch)

        # Act
        unsegmented.learn(_experiences(), sampling_logps=sampling_logps)
        segmented.learn(
            _experiences(),
            sampling_logps=sampling_logps,
            episode_segments=EPISODE_SEGMENTS,
        )

        # Assert
        assert len(unsegmented_grads) == len(segmented_grads) == 1
        assert any(grad.abs().sum() > 0 for grad in unsegmented_grads[0].values())
        for name, grad in unsegmented_grads[0].items():
            # fp32 sums over differently shaped rows.
            assert torch.allclose(
                segmented_grads[0][name], grad, rtol=1e-5, atol=1e-7
            ), name

    @pytest.mark.parametrize("fuse", [False, True], ids=["split", "fused"])
    def test_filler_micro_batches_leave_the_gradient_unchanged(
        self, monkeypatch: pytest.MonkeyPatch, fuse: bool
    ) -> None:
        # Arrange: one optimizer step of 1-row micro-batches over 6 real rows,
        # once alone and once padded to 8 rows by a simulated second rank.
        overrides = {
            "micro_batch_size_per_gpu": 1,
            "mini_batch_size": 4,
            "fuse_actor_critic_pass": fuse,
        }
        alone = _make_ppo(**overrides)
        padded = _make_ppo(**overrides)
        alone_grads = record_step_gradients(alone, monkeypatch)
        padded_grads = record_step_gradients(padded, monkeypatch)

        # Act
        alone.learn(_experiences(), episode_segments=EPISODE_SEGMENTS)
        pad_to_eight_rows(monkeypatch)
        padded_metrics = padded.learn(_experiences(), episode_segments=EPISODE_SEGMENTS)

        # Assert
        assert padded_metrics["train_rows_padded"] == pytest.approx(8.0)
        assert padded_metrics["filler_token_frac"] == pytest.approx(0.25)
        assert len(alone_grads) == len(padded_grads) == 1
        for name, grad in alone_grads[0].items():
            # fp32 sums over a different micro-batch order.
            assert torch.allclose(padded_grads[0][name], grad, rtol=1e-5, atol=1e-7), (
                name
            )
        for key in ("loss", "pg_loss", "vf_loss", "kl", "entropy"):
            assert math.isfinite(padded_metrics[key]), key

    @pytest.mark.parametrize("fuse", [False, True], ids=["split", "fused"])
    def test_liger_metrics_leave_out_filler_micro_batches(
        self, monkeypatch: pytest.MonkeyPatch, fuse: bool
    ) -> None:
        # Arrange: the Liger kernel reports each micro-batch's action tokens as
        # its KL; 6 real 1-row micro-batches pad to 8 for a simulated second rank.
        agent = _make_ppo(
            micro_batch_size_per_gpu=1, mini_batch_size=4, fuse_actor_critic_pass=fuse
        )
        use_fake_liger_policy_loss(agent, monkeypatch, "agilerl.algorithms.ppo_llm")
        pad_to_eight_rows(monkeypatch)

        # Act
        metrics = agent.learn(_experiences(), episode_segments=EPISODE_SEGMENTS)

        # Assert
        assert metrics["train_rows_padded"] == pytest.approx(8.0)
        assert metrics["kl"] == pytest.approx(MEAN_SEGMENT_ROW_ACTION_TOKENS)

    def test_no_segments_matches_all_none_segments(self) -> None:
        # Arrange
        unset = _make_ppo()
        all_none = _make_ppo()

        # Act
        unset_metrics = unset.learn(_experiences(), episode_segments=None)
        all_none_metrics = all_none.learn(_experiences(), episode_segments=[None] * 4)

        # Assert
        timings = set(LEARN_PHASE_METRIC_NAMES)
        assert unset_metrics.keys() == all_none_metrics.keys()
        for key in unset_metrics.keys() - timings:
            assert all_none_metrics[key] == pytest.approx(
                unset_metrics[key], rel=0.0, abs=0.0, nan_ok=True
            ), key
        unset_weights = lora_weights(unset)
        for name, weight in lora_weights(all_none).items():
            assert torch.equal(weight, unset_weights[name]), name

    def test_segment_count_must_match_episode_count(self) -> None:
        agent = _make_ppo()

        with pytest.raises(
            ValueError, match="episode_segments has 3 entries for 4 episodes"
        ):
            agent.learn(_experiences(), episode_segments=EPISODE_SEGMENTS[:3])

    def test_trajectory_importance_sampling_is_rejected(self) -> None:
        agent = _make_ppo(importance_sampling_level="trajectory")

        with pytest.raises(
            ValueError, match="Set importance_sampling_level='token' or 'turn'"
        ):
            agent.learn(_experiences(), episode_segments=EPISODE_SEGMENTS)


class TestPPOLearnPhaseTimings:
    def test_reports_every_learn_phase_and_closes_the_timer(self) -> None:
        # Arrange
        agent = _make_ppo()

        # Act
        metrics = agent.learn(_experiences())

        # Assert
        assert set(LEARN_PHASE_METRIC_NAMES) <= set(metrics)
        assert all(metrics[key] >= 0.0 for key in LEARN_PHASE_METRIC_NAMES)
        assert metrics["learn_phase_forward_s"] > 0.0
        assert agent.shard_runtime.phase_timer.marks is None

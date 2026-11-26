# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for LLM REINFORCE training segmented episodes one segment per row.

Pure CPU on a tiny real model in fp32. The model scores each token from that
token alone, so a segment row's log-probs match its episode's.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Iterator
from functools import partial
from typing import Any

import pytest
import torch
import torch.distributed as dist

pytest.importorskip("transformers", reason="LLM tests require transformers.")
pytest.importorskip("peft", reason="LLM tests require peft.")

from peft import LoraConfig

from agilerl.algorithms import reinforce_llm
from agilerl.algorithms.reinforce_llm import REINFORCE
from agilerl.components.llm_rollout_data import EpisodeSegments
from agilerl.utils.llm_utils import LEARN_PHASE_METRIC_NAMES, masked_mean
from tests.rank_launch import run_ranks
from tests.test_algorithms.test_llms.llm_helpers import (
    create_module,
    optimizer_state,
    record_calls,
    scale_losses,
    score_every_token,
    trainable_weights,
)
from tests.test_algorithms.test_llms.segment_helpers import (
    EPISODE_SEGMENTS,
    balanced_learn_worker,
    episode_sampling_logps,
    episode_sampling_supports,
    learn_rank_episodes,
    learn_with_nan_loss_on_rank_zero,
    lora_weights,
    pad_to_eight_rows,
    rank_local_step_gradients,
    record_step_gradients,
    rows_from_other_ranks,
    segment_experiences,
)

PAD_TOKEN_ID = 63
VOCAB = 64


def _make_reinforce(**overrides: Any) -> REINFORCE:
    """Tiny fp32 REINFORCE on CPU with one optimizer step per learn over 4 episodes."""
    torch.manual_seed(0)
    kwargs: dict[str, Any] = {
        "actor_network": create_module(
            input_size=6, max_tokens=4, vocab_size=VOCAB, device="cpu"
        ),
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
        "calc_position_embeddings": False,
        "device": "cpu",
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


def _experiences() -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    return segment_experiences(PAD_TOKEN_ID)


class TestREINFORCELearnEpisodeSegments:
    def test_trains_segment_rows_with_finite_loss(self) -> None:
        # Arrange
        agent = _make_reinforce()
        before = lora_weights(agent)

        # Act
        metrics = agent.learn(
            _experiences(),
            sampling_logps=episode_sampling_logps(PAD_TOKEN_ID),
            episode_segments=EPISODE_SEGMENTS,
        )

        # Assert
        for key in ("loss", "pg_loss", "kl", "entropy"):
            assert math.isfinite(metrics[key]), key
        after = lora_weights(agent)
        assert any(not torch.equal(after[name], before[name]) for name in before)

    def test_one_micro_batch_matches_the_unsegmented_gradient(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange: every row fits one micro-batch, so the episode returns and the
        # token-mean loss cover the same action tokens either way.
        overrides = {
            "batch_size": 8,
            "micro_batch_size_per_gpu": 8,
            "mini_batch_size": 8,
        }
        unsegmented = _make_reinforce(**overrides)
        segmented = _make_reinforce(**overrides)
        unsegmented_grads = record_step_gradients(unsegmented, monkeypatch)
        segmented_grads = record_step_gradients(segmented, monkeypatch)

        # Act
        unsegmented.learn(
            _experiences(), sampling_logps=episode_sampling_logps(PAD_TOKEN_ID)
        )
        segmented.learn(
            _experiences(),
            sampling_logps=episode_sampling_logps(PAD_TOKEN_ID),
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

    def test_filler_micro_batches_leave_the_gradient_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange: one optimizer step of 1-row micro-batches over 6 real rows,
        # once alone and once padded to 8 rows by a simulated second rank.
        overrides = {"micro_batch_size_per_gpu": 1, "mini_batch_size": 4}
        alone = _make_reinforce(**overrides)
        padded = _make_reinforce(**overrides)
        alone_grads = record_step_gradients(alone, monkeypatch)
        padded_grads = record_step_gradients(padded, monkeypatch)

        # Act
        alone.learn(
            _experiences(),
            sampling_logps=episode_sampling_logps(PAD_TOKEN_ID),
            episode_segments=EPISODE_SEGMENTS,
        )
        pad_to_eight_rows(monkeypatch)
        padded_metrics = padded.learn(
            _experiences(),
            sampling_logps=episode_sampling_logps(PAD_TOKEN_ID),
            episode_segments=EPISODE_SEGMENTS,
        )

        # Assert
        assert padded_metrics["train_rows_padded"] == pytest.approx(8.0)
        assert padded_metrics["filler_token_frac"] == pytest.approx(0.25)
        assert len(alone_grads) == len(padded_grads) == 1
        for name, grad in alone_grads[0].items():
            # fp32 sums over a different micro-batch order.
            assert torch.allclose(padded_grads[0][name], grad, rtol=1e-5, atol=1e-7), (
                name
            )
        for key in ("loss", "pg_loss", "kl", "entropy"):
            assert math.isfinite(padded_metrics[key]), key

    def test_rows_from_other_ranks_train_to_the_same_gradient(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange: one optimizer step of 1-row micro-batches over 6 real rows.
        local = _make_reinforce(micro_batch_size_per_gpu=1)
        dealt = _make_reinforce(micro_batch_size_per_gpu=1)
        local_grads = record_step_gradients(local, monkeypatch)
        dealt_grads = record_step_gradients(dealt, monkeypatch)

        # Act
        local_metrics = local.learn(
            _experiences(),
            sampling_logps=episode_sampling_logps(PAD_TOKEN_ID),
            episode_segments=EPISODE_SEGMENTS,
        )
        monkeypatch.setattr(
            "agilerl.algorithms.core.base.balance_rows_across_ranks",
            rows_from_other_ranks,
        )
        dealt_metrics = dealt.learn(
            _experiences(),
            sampling_logps=episode_sampling_logps(PAD_TOKEN_ID),
            episode_segments=EPISODE_SEGMENTS,
        )

        # Assert
        assert len(local_grads) == len(dealt_grads) == 1
        assert any(grad.abs().sum() > 0 for grad in local_grads[0].values())
        for name, grad in local_grads[0].items():
            # fp32 sums over a different micro-batch order.
            assert torch.allclose(dealt_grads[0][name], grad, rtol=1e-5, atol=1e-7), (
                name
            )
        for key in ("loss", "pg_loss", "kl", "entropy"):
            assert dealt_metrics[key] == pytest.approx(
                local_metrics[key], rel=1e-5, abs=1e-7
            ), key

    def test_no_segments_matches_all_none_segments(self) -> None:
        # Arrange
        unset = _make_reinforce()
        all_none = _make_reinforce()

        # Act
        unset_metrics = unset.learn(
            _experiences(),
            sampling_logps=episode_sampling_logps(PAD_TOKEN_ID),
            episode_segments=None,
        )
        all_none_metrics = all_none.learn(
            _experiences(),
            sampling_logps=episode_sampling_logps(PAD_TOKEN_ID),
            episode_segments=[None] * 4,
        )

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
        agent = _make_reinforce()

        with pytest.raises(
            ValueError, match="episode_segments has 3 entries for 4 episodes"
        ):
            agent.learn(
                _experiences(),
                sampling_logps=episode_sampling_logps(PAD_TOKEN_ID),
                episode_segments=EPISODE_SEGMENTS[:3],
            )

    def test_trajectory_importance_sampling_is_rejected(self) -> None:
        agent = _make_reinforce(importance_sampling_level="trajectory")

        with pytest.raises(
            ValueError, match="Set importance_sampling_level='token' or 'turn'"
        ):
            agent.learn(
                _experiences(),
                sampling_logps=episode_sampling_logps(PAD_TOKEN_ID),
                episode_segments=EPISODE_SEGMENTS,
            )


def scale_surrogate_losses(
    _agent: REINFORCE, monkeypatch: pytest.MonkeyPatch, scales: Iterator[float]
) -> None:
    """Make each policy surrogate loss carry the next scale."""
    monkeypatch.setattr(
        reinforce_llm,
        "clipped_token_surrogate",
        scale_losses(reinforce_llm.clipped_token_surrogate, scales),
    )


class TestREINFORCELearnScoredTokenHead:
    def test_gradient_matches_the_full_head(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        full = _make_reinforce()
        scored = _make_reinforce()
        score_every_token(full, monkeypatch)
        full_grads = record_step_gradients(full, monkeypatch)
        scored_grads = record_step_gradients(scored, monkeypatch)

        # Act
        full.learn(
            _experiences(),
            sampling_logps=episode_sampling_logps(PAD_TOKEN_ID),
            episode_segments=EPISODE_SEGMENTS,
        )
        scored.learn(
            _experiences(),
            sampling_logps=episode_sampling_logps(PAD_TOKEN_ID),
            episode_segments=EPISODE_SEGMENTS,
        )

        # Assert
        assert len(full_grads) == len(scored_grads) == 1
        assert any(grad.abs().sum() > 0 for grad in full_grads[0].values())
        for name, grad in full_grads[0].items():
            # The head sees the scored rows as one flat row; fp32 GEMM
            # blocking follows the row count.
            torch.testing.assert_close(
                scored_grads[0][name], grad, rtol=1e-5, atol=1e-7, msg=name
            )


@pytest.mark.skipif(
    sys.platform == "win32" or not dist.is_available(), reason="gloo unavailable"
)
@pytest.mark.multi_rank
class TestREINFORCELearnBalancedAcrossRanks:
    def test_uneven_ranks_run_even_rows_with_their_local_gradients(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange: ReBN advantages stay on each rank's own episodes, so the
        # step averages rank 0's 4 local rows and rank 1's 2. Without a
        # reachable clip norm the step gradient is linear in the rows.
        make_agent = partial(_make_reinforce, max_grad_norm=1e30)
        rank0, rank1 = rank_local_step_gradients(
            make_agent, monkeypatch, sampling_ranks=(0, 1)
        )

        # Act
        reports = run_ranks(
            balanced_learn_worker,
            2,
            make_agent,
            partial(learn_rank_episodes, sampling_ranks=(0, 1)),
            backend=None,
        )

        # Assert: 4 and 2 rows are dealt out as 3 and 3.
        for padded_rows, steps in reports:
            assert padded_rows == pytest.approx(3.0)
            assert len(steps) == 1
            assert steps[0].keys() == rank0.keys()
            for name in rank0:
                # fp32 sums over a different micro-batch order and rank split.
                torch.testing.assert_close(
                    torch.from_numpy(steps[0][name]),
                    (4 * rank0[name] + 2 * rank1[name]) / 6,
                    rtol=1e-5,
                    atol=1e-5,
                    msg=lambda detail, name=name: f"{name}\n{detail}",
                )

    def test_a_non_finite_loss_on_one_rank_raises_on_every_rank(self) -> None:
        # Arrange: every loss of rank 0 is NaN, every loss of rank 1 finite.
        learn = partial(
            learn_with_nan_loss_on_rank_zero,
            scale_loss=scale_surrogate_losses,
            sampling_ranks=(0, 1),
        )

        # Act
        reports = run_ranks(
            balanced_learn_worker, 2, _make_reinforce, learn, backend=None
        )

        # Assert
        for rank, ((error, weights_held), steps) in enumerate(reports):
            assert f"rank={rank} local_finite={rank != 0}" in error
            assert weights_held
            assert steps == []


class TestREINFORCELearnMetrics:
    def test_metrics_are_means_of_the_micro_batch_losses(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange: one optimizer step of four 1-row micro-batches. The
        # fixture's sampling log-probs are far from the policy's, so no
        # off-policy mask keeps each loss its surrogate mean.
        agent = _make_reinforce(
            micro_batch_size_per_gpu=1, off_policy_token_mask_bounds=None
        )
        surrogate, calls = record_calls(reinforce_llm.clipped_token_surrogate)
        monkeypatch.setattr(reinforce_llm, "clipped_token_surrogate", surrogate)

        # Act
        metrics = agent.learn(
            _experiences(), sampling_logps=episode_sampling_logps(PAD_TOKEN_ID)
        )

        # Assert
        assert len(calls) == 4
        mean_loss = (
            sum(
                masked_mean(token_loss, args[2]).item()
                for args, _kwargs, (token_loss, _clipped) in calls
            )
            / 4
        )
        assert metrics["loss"] == pytest.approx(mean_loss, rel=1e-6)
        assert metrics["pg_loss"] == pytest.approx(mean_loss, rel=1e-6)


class TestREINFORCELearnNonFiniteLoss:
    def test_finite_losses_step(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Arrange
        agent = _make_reinforce(micro_batch_size_per_gpu=1)
        monkeypatch.setattr(
            reinforce_llm,
            "clipped_token_surrogate",
            scale_losses(reinforce_llm.clipped_token_surrogate, iter([1.0] * 4)),
        )
        before = trainable_weights(agent.actor)

        # Act
        agent.learn(_experiences(), sampling_logps=episode_sampling_logps(PAD_TOKEN_ID))

        # Assert
        after = trainable_weights(agent.actor)
        assert any(
            not torch.equal(new, old) for new, old in zip(after, before, strict=True)
        )

    @pytest.mark.parametrize(
        "window_scales",
        [[float("nan"), 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, float("nan")]],
        ids=["first_micro_batch", "last_micro_batch"],
    )
    def test_raises_before_the_step_when_a_micro_batch_loss_is_not_finite(
        self, monkeypatch: pytest.MonkeyPatch, window_scales: list[float]
    ) -> None:
        # Arrange: one optimizer step of four 1-row micro-batches per learn; a
        # finite learn first gives the optimizer state to keep.
        agent = _make_reinforce(micro_batch_size_per_gpu=1)
        scales = iter([1.0] * 4 + window_scales)
        monkeypatch.setattr(
            reinforce_llm,
            "clipped_token_surrogate",
            scale_losses(reinforce_llm.clipped_token_surrogate, scales),
        )
        agent.learn(_experiences(), sampling_logps=episode_sampling_logps(PAD_TOKEN_ID))
        before = trainable_weights(agent.actor)
        state_before = optimizer_state(agent.optimizer)

        # Act
        with pytest.raises(ValueError, match="Loss is not finite"):
            agent.learn(
                _experiences(), sampling_logps=episode_sampling_logps(PAD_TOKEN_ID)
            )

        # Assert: it raises at the window's last micro-batch, before the step.
        assert list(scales) == []
        for new, old in zip(trainable_weights(agent.actor), before, strict=True):
            assert torch.equal(new, old)
        state_after = optimizer_state(agent.optimizer)
        assert state_before
        assert len(state_after) == len(state_before)
        for new, old in zip(state_after, state_before, strict=True):
            assert torch.equal(new, old)

    def test_raises_when_the_micro_batch_left_pending_a_step_is_not_finite(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange: three micro-batches per step over four rows leave the last
        # micro-batch's gradients pending the next learn call.
        agent = _make_reinforce(micro_batch_size_per_gpu=1)
        agent.gradient_accumulation_steps = 3
        scales = iter([1.0, 1.0, 1.0, float("nan")])
        monkeypatch.setattr(
            reinforce_llm,
            "clipped_token_surrogate",
            scale_losses(reinforce_llm.clipped_token_surrogate, scales),
        )

        # Act / Assert
        with pytest.raises(ValueError, match="Loss is not finite"):
            agent.learn(
                _experiences(), sampling_logps=episode_sampling_logps(PAD_TOKEN_ID)
            )
        assert list(scales) == []


class TestREINFORCELearnPhaseTimings:
    def test_reports_every_learn_phase_and_closes_the_timer(self) -> None:
        # Arrange
        agent = _make_reinforce()

        # Act
        metrics = agent.learn(
            _experiences(), sampling_logps=episode_sampling_logps(PAD_TOKEN_ID)
        )

        # Assert
        assert set(LEARN_PHASE_METRIC_NAMES) <= set(metrics)
        assert all(metrics[key] >= 0.0 for key in LEARN_PHASE_METRIC_NAMES)
        assert metrics["learn_phase_forward_s"] > 0.0
        assert agent.shard_runtime.phase_timer.marks is None


class TestREINFORCELearnSamplingSupports:
    def test_forced_tokens_score_a_zero_log_prob(self) -> None:
        # Arrange: every action token is forced, so its log-prob over its
        # support is 0 whatever the policy.
        agent = _make_reinforce()
        supports = episode_sampling_supports(PAD_TOKEN_ID, width=1)
        sampling_logps = [torch.zeros(row.shape[0]) for row in supports]

        # Act
        metrics = agent.learn(
            _experiences(), sampling_logps=sampling_logps, sampling_supports=supports
        )

        # Assert
        assert metrics["vllm_is_delta_mean"] == pytest.approx(0.0, abs=1e-6)
        assert metrics["vllm_is_delta_max"] == pytest.approx(0.0, abs=1e-6)

    def test_segment_rows_match_the_unsegmented_gradient(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange: the fixture's sampling log-probs are far from the policy's,
        # so the off-policy mask would zero every token.
        overrides = {
            "batch_size": 8,
            "micro_batch_size_per_gpu": 8,
            "mini_batch_size": 8,
            "off_policy_token_mask_bounds": None,
        }
        unsegmented = _make_reinforce(**overrides)
        segmented = _make_reinforce(**overrides)
        unsegmented_grads = record_step_gradients(unsegmented, monkeypatch)
        segmented_grads = record_step_gradients(segmented, monkeypatch)
        learn_kwargs = {
            "sampling_logps": episode_sampling_logps(PAD_TOKEN_ID),
            "sampling_supports": episode_sampling_supports(PAD_TOKEN_ID),
        }

        # Act
        unsegmented.learn(_experiences(), **learn_kwargs)
        segmented.learn(
            _experiences(), episode_segments=EPISODE_SEGMENTS, **learn_kwargs
        )

        # Assert
        assert len(unsegmented_grads) == len(segmented_grads) == 1
        assert any(grad.abs().sum() > 0 for grad in unsegmented_grads[0].values())
        for name, grad in unsegmented_grads[0].items():
            # fp32 sums over differently shaped rows.
            assert torch.allclose(
                segmented_grads[0][name], grad, rtol=1e-5, atol=1e-7
            ), name


def _record_no_grad_forwards(
    agent: REINFORCE, monkeypatch: pytest.MonkeyPatch
) -> list[tuple[tuple[Any, ...], dict[str, Any], Any]]:
    recorded, calls = record_calls(agent._fused_forward_no_grad)
    monkeypatch.setattr(agent, "_fused_forward_no_grad", recorded)
    return calls


class TestREINFORCELearnNoGradRows:
    """The learn-start forward runs only for the KL reward penalty."""

    @pytest.mark.parametrize(
        "episode_segments", [None, EPISODE_SEGMENTS], ids=["episodes", "segments"]
    )
    def test_without_kl_skips_the_no_grad_forward(
        self,
        monkeypatch: pytest.MonkeyPatch,
        episode_segments: list[EpisodeSegments | None] | None,
    ) -> None:
        # Arrange
        agent = _make_reinforce(beta=0.0)
        calls = _record_no_grad_forwards(agent, monkeypatch)

        # Act
        metrics = agent.learn(
            _experiences(),
            sampling_logps=episode_sampling_logps(PAD_TOKEN_ID),
            episode_segments=episode_segments,
        )

        # Assert
        assert calls == []
        assert math.isfinite(metrics["loss"])
        assert metrics["vllm_is_rows_skipped"] == 0.0
        assert math.isfinite(metrics["vllm_is_delta_mean"])

    def test_kl_penalty_scores_reference_and_actor_on_every_row(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        agent = _make_reinforce(beta=0.01)
        calls = _record_no_grad_forwards(agent, monkeypatch)

        # Act
        metrics = agent.learn(
            _experiences(), sampling_logps=episode_sampling_logps(PAD_TOKEN_ID)
        )

        # Assert
        ((args, kwargs, _),) = calls
        assert args[0].shape[0] == 4
        assert kwargs["include_reference"] is True
        assert kwargs.get("include_actor", True) is True
        assert math.isfinite(metrics["loss"])

    def test_a_row_without_sampling_log_probs_raises(self) -> None:
        agent = _make_reinforce()
        logps = episode_sampling_logps(PAD_TOKEN_ID)

        with pytest.raises(
            ValueError,
            match=r"REINFORCE needs sampling_logps on every action token; row 2 has none, expected 4\.",
        ):
            agent.learn(
                _experiences(), sampling_logps=[logps[0], logps[1], None, logps[3]]
            )

    def test_a_short_row_raises(self) -> None:
        agent = _make_reinforce()
        logps = episode_sampling_logps(PAD_TOKEN_ID)

        with pytest.raises(ValueError, match=r"row 1 has 5, expected 6\."):
            agent.learn(
                _experiences(),
                sampling_logps=[logps[0], logps[1][:-1], logps[2], logps[3]],
            )

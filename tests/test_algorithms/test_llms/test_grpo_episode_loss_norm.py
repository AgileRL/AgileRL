# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for GRPO ``loss_norm="episode"``.

Each test reads the per-token gradient weight the update gives the policy
term, through a probe tensor standing in for the policy log-probs (standard
path) or hidden states (fused path). Pure CPU on a tiny real model in fp32.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pytest
import torch

pytest.importorskip("transformers", reason="LLM tests require transformers.")
pytest.importorskip("peft", reason="LLM tests require peft.")

from agilerl.algorithms import grpo as grpo_module
from agilerl.algorithms.grpo import GRPO
from agilerl.components.llm_rollout_data import EpisodeSegments
from tests.test_algorithms.test_llms.segment_helpers import (
    EPISODE_SEGMENTS,
    segment_experiences,
)
from tests.test_algorithms.test_llms.test_grpo_old_logprobs import (
    PAD_TOKEN_ID,
    _make_grpo,
)

KERNEL_NAME = "LigerFusedLinearGRPOFunction"
ADVANTAGE = 0.5
"""Magnitude of every episode's advantage under ``adv_norm="mean_only"``."""
EPISODE_ACTION_TOKENS = [6, 6, 4, 8]
"""Action tokens of each ``EPISODE_SEGMENTS`` episode, over all its segment rows."""


class _ProbeFusedKernel:
    """Fused-kernel stand-in whose per-token policy loss is ``-advantage * hidden[0]``.

    Reduces like liger: token-count loss types divide by ``num_items_in_batch``
    over the world size, ``grpo`` averages per-row means.
    """

    loss_types: ClassVar[list[str]] = []

    @classmethod
    def forward(
        cls,
        ctx,
        _input,
        weight,
        selected_token_ids,
        attention_mask,
        advantages,
        bias=None,
        ref_per_token_logps=None,
        old_per_token_logps=None,
        ref_input=None,
        ref_weight=None,
        ref_bias=None,
        beta=0.04,
        epsilon_low=0.2,
        epsilon_high=0.2,
        loss_type="dapo",
        max_completion_length=None,
        importance_sampling_level="token",
        sapo_temperature_pos=1.0,
        sapo_temperature_neg=1.05,
        temperature=1.0,
        compiled=True,
        use_ref_model=True,
        chunk_size=1,
        vllm_is_ratio=None,
        delta=None,
        use_bias_correction_kl=False,
        vespo_k_pos=2.0,
        vespo_lambda_pos=3.0,
        vespo_k_neg=3.0,
        vespo_lambda_neg=2.0,
        num_items_in_batch=None,
    ):
        """Linear per-token loss in the probe under the requested reduction."""
        cls.loss_types.append(loss_type)
        mask = attention_mask.to(_input.dtype)
        masked = -advantages.unsqueeze(1) * _input[..., 0] * mask
        if loss_type in {"dapo", "cispo"}:
            count = (
                torch.as_tensor(float(num_items_in_batch))
                if num_items_in_batch is not None
                else mask.sum()
            )
            loss = masked.sum() / torch.clamp(count, min=1.0)
        else:
            loss = (masked.sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
        return loss, (torch.zeros(()),)

    @classmethod
    def apply(cls, *args):
        """Invoke ``forward`` the way ``torch.autograd.Function.apply`` does."""
        return cls.forward(None, *args)


def _probe_policy_weights(
    agent: GRPO,
    monkeypatch: pytest.MonkeyPatch,
    episode_segments: list[EpisodeSegments | None] | None,
    *,
    fused: bool,
) -> tuple[torch.Tensor, np.ndarray, torch.Tensor]:
    """Per-token policy-term weights one ``learn`` gives each training row.

    :return: ``(R, T-1)`` weights, ``(R,)`` source episode per row, and the
        ``(R, T-1)`` action mask of the training rows.
    """
    state: dict[str, Any] = {}
    original_loss = agent._loss

    def tracking_loss(minibatch_idxs, token_ids, action_mask, *args, **kwargs):
        if "probe" not in state:
            rows, width = int(token_ids.shape[0]), int(token_ids.shape[1])
            hidden_dim = int(agent._get_lm_head().weight.shape[1])
            shape = (rows, width, hidden_dim) if fused else (rows, width - 1)
            state["probe"] = torch.zeros(shape, requires_grad=True)
            state["mask"] = action_mask
        state["rows"] = torch.as_tensor(minibatch_idxs)
        return original_loss(minibatch_idxs, token_ids, action_mask, *args, **kwargs)

    def zero_learn_start_log_probs(_token_ids, action_masks, *_args, **_kwargs):
        zeros = torch.zeros(action_masks.shape)
        return zeros, zeros.clone()

    def accumulate(loss, accumulation_steps=None):
        steps = accumulation_steps or agent.gradient_accumulation_steps
        (loss / steps).backward()

    split_rows = agent._segment_rows

    def recording_segment_rows(*args, **kwargs):
        rows, steps = split_rows(*args, **kwargs)
        state["row_episodes"] = rows.row_episodes
        return rows, steps

    monkeypatch.setattr(agent, "_loss", tracking_loss)
    monkeypatch.setattr(agent, "_learn_start_log_probs", zero_learn_start_log_probs)
    monkeypatch.setattr(agent, "_backward_pass", accumulate)
    monkeypatch.setattr(agent, "_segment_rows", recording_segment_rows)
    if fused:
        agent.use_liger_loss = True
        monkeypatch.setattr(grpo_module, "HAS_LIGER_KERNEL", True)
        monkeypatch.setattr(_ProbeFusedKernel, "loss_types", [])
        monkeypatch.setattr(grpo_module, KERNEL_NAME, _ProbeFusedKernel, raising=False)
        monkeypatch.setattr(
            agent,
            "_actor_hidden_states",
            lambda _ids, _pixel_values=None: state["probe"][state["rows"]],
        )
    else:
        monkeypatch.setattr(
            agent,
            "_get_logprobs",
            lambda *_args, **_kwargs: state["probe"][state["rows"]],
        )

    agent.learn(segment_experiences(PAD_TOKEN_ID), episode_segments=episode_segments)

    grad = state["probe"].grad
    assert grad is not None
    per_token = grad[..., : state["mask"].shape[1], 0] if fused else grad
    row_episodes = state.get("row_episodes", np.arange(per_token.shape[0]))
    return per_token.abs() / ADVANTAGE, row_episodes, state["mask"]


def _episode_totals(weights: torch.Tensor, row_episodes: np.ndarray) -> list[float]:
    """Summed weight of each real episode's tokens."""
    return [
        float(weights[torch.as_tensor(row_episodes == episode)].sum())
        for episode in range(int(row_episodes.max()) + 1)
    ]


def _make_episode_grpo(**overrides: Any) -> GRPO:
    """Tiny GRPO whose four episodes share one optimizer step of two micro-batches."""
    return _make_grpo(**{"adv_norm": "mean_only", "loss_type": "cispo", **overrides})


class TestGRPOEpisodeLossNormLearn:
    """``learn`` gives each episode of a window the same total policy weight."""

    @pytest.mark.parametrize("fused", [False, True], ids=["standard", "liger"])
    @pytest.mark.parametrize("loss_type", ["cispo", "grpo"])
    def test_segment_rows_across_micro_batches_weigh_each_episode_equally(
        self, monkeypatch: pytest.MonkeyPatch, fused: bool, loss_type: str
    ) -> None:
        # Arrange: 4 episodes of 6, 6, 4 and 8 action tokens split into 6
        # segment rows, one window of three 2-row micro-batches.
        agent = _make_episode_grpo(loss_norm="episode", loss_type=loss_type)

        # Act
        weights, row_episodes, mask = _probe_policy_weights(
            agent, monkeypatch, EPISODE_SEGMENTS, fused=fused
        )

        # Assert
        assert len(row_episodes) == 6
        assert _episode_totals(weights, row_episodes) == pytest.approx([0.25] * 4)
        for episode, tokens in enumerate(EPISODE_ACTION_TOKENS):
            rows = torch.as_tensor(row_episodes == episode)
            episode_weights = weights[rows][mask[rows].bool()]
            assert torch.allclose(
                episode_weights, torch.full_like(episode_weights, 1 / (4 * tokens))
            )
        assert torch.all(weights[~mask.bool()] == 0.0)
        if fused:
            expected_type = "cispo" if loss_type == "cispo" else "dapo"
            assert set(_ProbeFusedKernel.loss_types) == {expected_type}

    @pytest.mark.parametrize("fused", [False, True], ids=["standard", "liger"])
    def test_unsegmented_rows_across_micro_batches_weigh_each_episode_equally(
        self, monkeypatch: pytest.MonkeyPatch, fused: bool
    ) -> None:
        # Arrange: the same 4 episodes as one row each, two 2-row micro-batches.
        agent = _make_episode_grpo(loss_norm="episode")

        # Act
        weights, row_episodes, _mask = _probe_policy_weights(
            agent, monkeypatch, None, fused=fused
        )

        # Assert
        assert _episode_totals(weights, row_episodes) == pytest.approx([0.25] * 4)

    @pytest.mark.parametrize("fused", [False, True], ids=["standard", "liger"])
    def test_accumulation_window_keeps_weighing_episodes_by_length(
        self, monkeypatch: pytest.MonkeyPatch, fused: bool
    ) -> None:
        # Arrange
        agent = _make_episode_grpo(loss_norm="accumulation_window")

        # Act
        weights, row_episodes, _mask = _probe_policy_weights(
            agent, monkeypatch, EPISODE_SEGMENTS, fused=fused
        )

        # Assert: every action token of the window weighs 1 / 24.
        total_tokens = sum(EPISODE_ACTION_TOKENS)
        assert _episode_totals(weights, row_episodes) == pytest.approx(
            [tokens / total_tokens for tokens in EPISODE_ACTION_TOKENS]
        )

    def test_episode_mode_records_the_window_token_share(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        agent = _make_episode_grpo(loss_norm="episode")

        # Act
        _probe_policy_weights(agent, monkeypatch, EPISODE_SEGMENTS, fused=False)

        # Assert
        assert agent._window_action_tokens == pytest.approx(
            float(sum(EPISODE_ACTION_TOKENS))
        )


class TestGRPORowEpisodeActionTokens:
    def test_segment_rows_share_their_episode_total(self) -> None:
        # Arrange: episode 0 spans rows 0 and 1, episode 1 is row 2, row 3 is filler.
        mask = torch.zeros(4, 6, dtype=torch.bool)
        mask[0, :2] = True
        mask[1, :3] = True
        mask[2, :4] = True

        # Act
        tokens = GRPO._row_episode_action_tokens(mask, np.array([0, 0, 1, -1]))

        # Assert
        assert tokens.tolist() == [5.0, 5.0, 4.0, 1.0]


class TestGRPOEpisodeBalancedAdvantages:
    @staticmethod
    def _window_weights(
        agent: GRPO,
        mask: torch.Tensor,
        row_episodes: np.ndarray,
        window_idxs: np.ndarray,
    ) -> torch.Tensor:
        """Per-token policy weights of one window's accumulated update."""
        advantages = torch.ones(mask.shape[0], 1)
        episode_tokens = GRPO._row_episode_action_tokens(mask, row_episodes)
        scaled = agent._episode_balanced_advantages(
            advantages, mask, window_idxs, episode_tokens
        )
        per_token = torch.zeros(mask.shape, requires_grad=True)
        shares = agent._reduce_masked_loss(per_token * scaled, mask)
        (shares.mean() / agent.gradient_accumulation_steps).backward()
        assert per_token.grad is not None
        return per_token.grad

    def test_an_episode_straddling_windows_counts_by_its_token_fraction(self) -> None:
        # Arrange: episode 0 (4 tokens) is whole; episode 1 has 3 of its 6
        # tokens in this window and the rest (row 2) in another.
        agent = _make_episode_grpo(loss_norm="episode")
        mask = torch.zeros(3, 8, dtype=torch.bool)
        mask[0, :4] = True
        mask[1, :3] = True
        mask[2, :3] = True

        # Act
        weights = self._window_weights(
            agent, mask, np.array([0, 1, 1]), np.array([0, 1])
        )

        # Assert: 1.5 episodes in the window; each token weighs 1 / (1.5 * 4)
        # and 1 / (1.5 * 6).
        assert weights[0, :4].tolist() == pytest.approx([1 / 6] * 4)
        assert weights[1, :3].tolist() == pytest.approx([1 / 9] * 3)
        assert float(weights.sum()) == pytest.approx(1.0)
        assert torch.all(weights[2] == 0.0)

    def test_episodes_on_other_ranks_share_the_window(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange: a simulated second rank holds one 10-token episode.
        agent = _make_episode_grpo(loss_norm="episode")
        mask = torch.zeros(2, 12, dtype=torch.bool)
        mask[0, :2] = True
        mask[1, :10] = True

        def add_other_rank(totals: torch.Tensor, op: Any = None) -> None:
            totals += torch.tensor([10.0, 1.0])

        monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
        monkeypatch.setattr(torch.distributed, "all_reduce", add_other_rank)
        monkeypatch.setattr(grpo_module, "_liger_normalizer_world_size", lambda: 2)

        # Act
        weights = self._window_weights(agent, mask, np.array([0, 1]), np.array([0, 1]))

        # Assert: 3 episodes across ranks; the rank's gradient is averaged
        # with the other rank's, so this rank carries 2 of 3 episodes at 2x.
        assert float(weights[0].sum()) / 2 == pytest.approx(1 / 3)
        assert float(weights[1].sum()) / 2 == pytest.approx(1 / 3)
        assert agent._window_action_tokens == pytest.approx(11.0)

    def test_kl_term_stays_the_window_token_mean(self) -> None:
        # Arrange: zero advantage leaves only the KL term.
        agent = _make_episode_grpo(loss_norm="episode", loss_type="grpo", beta=0.1)
        mask = torch.zeros(2, 6, dtype=torch.bool)
        mask[0, :2] = True
        mask[1, :4] = True
        episode_tokens = GRPO._row_episode_action_tokens(mask, np.array([0, 1]))
        scaled = agent._episode_balanced_advantages(
            torch.zeros(2, 1), mask, np.array([0, 1]), episode_tokens
        )
        log_probs = torch.zeros(2, 6)
        reference = torch.full((2, 6), 0.3)

        # Act
        loss, _kl, _clipfrac = agent._compute_policy_loss(
            mask, log_probs, log_probs, reference, scaled, None, "token", "grpo"
        )

        # Assert
        token_kl = float(torch.exp(torch.tensor(0.3)) - 0.3 - 1)
        steps = agent.gradient_accumulation_steps
        assert float(loss) / steps == pytest.approx(0.1 * token_kl, rel=1e-5)

    def test_window_without_action_tokens_keeps_advantages_finite(self) -> None:
        agent = _make_episode_grpo(loss_norm="episode")
        mask = torch.zeros(2, 4, dtype=torch.bool)
        episode_tokens = GRPO._row_episode_action_tokens(mask, np.array([0, -1]))

        scaled = agent._episode_balanced_advantages(
            torch.tensor([[0.7], [0.0]]), mask, np.array([0, 1]), episode_tokens
        )

        assert scaled.flatten().tolist() == pytest.approx([0.7, 0.0])


class TestGRPOEpisodeLossNormConfig:
    def test_episode_mode_is_accepted(self) -> None:
        agent = _make_episode_grpo(loss_norm="episode")

        assert agent.loss_norm == "episode"

    def test_unknown_mode_lists_episode(self) -> None:
        with pytest.raises(ValueError, match="'episode'"):
            _make_episode_grpo(loss_norm="per_episode")

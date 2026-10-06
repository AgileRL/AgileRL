# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for GRPO training segmented episodes one segment per row.

Covers the cross-rank window loss normalizer and ``learn`` with
``episode_segments``. Pure CPU on a tiny real model in fp32.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
import torch

pytest.importorskip("transformers", reason="LLM tests require transformers.")
pytest.importorskip("peft", reason="LLM tests require peft.")

from agilerl.algorithms.grpo import GRPO, _liger_global_token_count
from agilerl.utils.segment_rows import split_episode_segments
from agilerl.utils.vision_rows import VisionRows
from tests.test_algorithms.test_llms.segment_helpers import (
    EPISODE_SEGMENTS,
    NUM_SEGMENT_ROWS,
    SEQ_LEN,
    episode_batch,
    lora_weights,
    pad_to_eight_rows,
    record_step_gradients,
    segment_experiences,
)
from tests.test_algorithms.test_llms.test_grpo_old_logprobs import (
    PAD_TOKEN_ID,
    _make_grpo,
)


def _experiences() -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    return segment_experiences(PAD_TOKEN_ID)


def _rollout_sampling_logps(agent: GRPO) -> list[torch.Tensor]:
    """Per-episode flat log-probs of the learn-start policy on the segment rows."""
    ids, mask = episode_batch(PAD_TOKEN_ID)
    rows = split_episode_segments(ids, mask, EPISODE_SEGMENTS, PAD_TOKEN_ID)
    _, row_log_probs, _ = agent._fused_forward_no_grad(
        rows.token_ids, NUM_SEGMENT_ROWS, include_reference=False
    )
    per_row = [
        row_log_probs[row][rows.action_masks[row]] for row in range(NUM_SEGMENT_ROWS)
    ]
    return [
        torch.cat([per_row[row] for row in np.flatnonzero(rows.row_episodes == ep)])
        for ep in range(4)
    ]


class TestLigerGlobalTokenCount:
    def test_sums_the_count_over_the_process_group(
        self, gloo_process_group: None
    ) -> None:
        mask = torch.tensor([[True, False, True], [True, True, False]])

        assert _liger_global_token_count(mask) == 4.0


class TestGRPOSegmentRows:
    def test_liger_token_loss_pads_rows_to_the_widest_rank(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange: a simulated second rank's segment rows are 14 tokens wide.
        agent = _make_grpo()
        agent.use_liger_loss = True
        monkeypatch.setattr("agilerl.algorithms.core.base.get_world_size", lambda: 2)
        monkeypatch.setattr(
            "agilerl.algorithms.core.base.allreduce_minmax_int",
            lambda value: (value, 14) if value == SEQ_LEN else (value, value),
        )
        ids, mask = episode_batch(PAD_TOKEN_ID)

        # Act
        rows, _accumulation_steps = agent._segment_rows(
            ids, mask, EPISODE_SEGMENTS, np.arange(4)
        )

        # Assert
        assert tuple(rows.token_ids.shape) == (NUM_SEGMENT_ROWS, 14)
        assert bool((rows.token_ids[:, SEQ_LEN:] == PAD_TOKEN_ID).all())


class TestGRPOReduceMaskedLoss:
    def test_segment_window_is_the_per_token_mean_across_ranks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange: a 2-micro-batch window on each of 2 simulated ranks; rank 0
        # has 3 action tokens, rank 1 has 5.
        agent = _make_grpo()
        agent._segment_accumulation_steps = 2
        rank_losses = [
            torch.tensor([[1.0, 2.0, 0.0], [3.0, 0.0, 0.0]]),
            torch.tensor([[4.0, 5.0, 6.0], [7.0, 8.0, 0.0]]),
        ]
        rank_masks = [
            torch.tensor([[True, True, False], [True, False, False]]),
            torch.tensor([[True, True, True], [True, True, False]]),
        ]
        global_tokens = float(sum(int(mask.sum()) for mask in rank_masks))
        monkeypatch.setattr(
            "agilerl.algorithms.grpo._liger_global_token_count",
            lambda _mask: global_tokens,
        )
        monkeypatch.setattr(
            "agilerl.algorithms.grpo._liger_normalizer_world_size", lambda: 2
        )

        # Act: each backward divides by the accumulation steps; ranks average.
        rank_gradients = []
        for loss, mask in zip(rank_losses, rank_masks, strict=True):
            agent._record_global_window_action_tokens(mask, np.arange(2))
            rank_gradients.append(
                sum(
                    agent._reduce_masked_loss(loss[row : row + 1], mask[row : row + 1])
                    .mean()
                    .item()
                    / 2
                    for row in range(2)
                )
            )
        averaged = sum(rank_gradients) / 2

        # Assert
        token_sum = sum(
            float((loss * mask).sum())
            for loss, mask in zip(rank_losses, rank_masks, strict=True)
        )
        assert averaged == pytest.approx(token_sum / global_tokens, rel=1e-6)

    def test_all_padding_segment_window_has_a_finite_zero_loss(self) -> None:
        # Arrange: no rank has an action token in the window.
        agent = _make_grpo()
        agent._segment_accumulation_steps = 2
        loss = torch.tensor([[1.0, float("nan")], [2.0, 3.0]])
        mask = torch.zeros(2, 2, dtype=torch.bool)
        agent._record_global_window_action_tokens(mask, np.arange(2))

        # Act
        shares = agent._reduce_masked_loss(loss, mask)

        # Assert
        assert shares.tolist() == [0.0, 0.0]

    def test_without_a_window_rows_use_their_own_tokens(self) -> None:
        agent = _make_grpo()
        loss = torch.tensor([[1.0, 2.0, 0.0, 0.0], [3.0, 0.0, 5.0, 0.0]])
        mask = torch.tensor([[True, True, False, False], [True, False, True, False]])

        shares = agent._reduce_masked_loss(loss, mask)

        assert shares.tolist() == pytest.approx([1.5, 4.0])


class TestGRPOLearnEpisodeSegments:
    def test_trains_segment_rows_with_finite_loss(self) -> None:
        # Arrange: 6 segment rows, one optimizer step per 2-row micro-batch.
        torch.manual_seed(0)
        agent = _make_grpo(mini_batch_size=2)
        before = lora_weights(agent)

        # Act
        metrics = agent.learn(_experiences(), episode_segments=EPISODE_SEGMENTS)

        # Assert
        assert math.isfinite(metrics["loss"])
        assert math.isfinite(metrics["kl_old"])
        assert math.isfinite(metrics["entropy"])
        after = lora_weights(agent)
        assert any(not torch.equal(after[name], before[name]) for name in before)

    def test_padding_rows_count_as_rollout_rows(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange: another rank has 8 rows, so this rank pads its 6 with 2.
        torch.manual_seed(0)
        agent = _make_grpo(old_logprobs_source="rollout")
        sampling_logps = _rollout_sampling_logps(agent)
        pad_to_eight_rows(monkeypatch)

        # Act
        metrics = agent.learn(
            _experiences(),
            sampling_logps=sampling_logps,
            episode_segments=EPISODE_SEGMENTS,
        )

        # Assert
        assert metrics["old_logprobs_trainer_rows"] == 0.0
        assert math.isfinite(metrics["loss"])

    def test_rows_fill_whole_optimizer_steps(self) -> None:
        # Arrange: 6 rows, 2-row micro-batches, 2 micro-batches per step.
        torch.manual_seed(0)
        agent = _make_grpo()

        # Act
        agent.learn(_experiences(), episode_segments=EPISODE_SEGMENTS)

        # Assert: no micro-batch waits for the next learn's optimizer step.
        assert agent.shard_runtime.micro_batches_until_step(2) == 2

    @pytest.mark.parametrize(
        ("other_rank_rows", "micro_batches"), [(13, 16), (2, 8), (6, 8)]
    )
    def test_runs_one_optimizer_step_per_episode_on_uneven_ranks(
        self,
        monkeypatch: pytest.MonkeyPatch,
        other_rank_rows: int,
        micro_batches: int,
    ) -> None:
        # Arrange: 4 episodes split into 6 rows; a simulated second rank has
        # ``other_rank_rows`` rows. One-row micro-batches, one per step unsegmented.
        torch.manual_seed(0)
        agent = _make_grpo(micro_batch_size_per_gpu=1, mini_batch_size=1)
        for module in ("agilerl.algorithms.core.base", "agilerl.algorithms.grpo"):
            monkeypatch.setattr(f"{module}.get_world_size", lambda: 2)
            monkeypatch.setattr(
                f"{module}.allreduce_minmax_int",
                lambda value: (
                    (min(value, other_rank_rows), max(value, other_rank_rows))
                    if value == NUM_SEGMENT_ROWS
                    else (value, value)
                ),
            )
        steps = []
        losses = []
        optimizer_step = agent.optimizer.step
        objective_loss = agent._objective_loss

        def counting_step(*args: Any, **kwargs: Any) -> None:
            steps.append(len(losses))
            optimizer_step(*args, **kwargs)

        def recording_loss(*args: Any, **kwargs: Any) -> Any:
            result = objective_loss(*args, **kwargs)
            losses.append(result[0].item())
            return result

        monkeypatch.setattr(agent.optimizer, "step", counting_step)
        monkeypatch.setattr(agent, "_objective_loss", recording_loss)

        # Act
        metrics = agent.learn(_experiences(), episode_segments=EPISODE_SEGMENTS)

        # Assert
        window = micro_batches // 4
        assert steps == [window, 2 * window, 3 * window, 4 * window]
        assert len(losses) == micro_batches
        assert all(math.isfinite(loss) for loss in losses)
        assert math.isfinite(metrics["loss"])

    @pytest.mark.parametrize("loss_norm", ["micro_batch", "accumulation_window"])
    def test_padding_only_micro_batches_keep_metrics_finite(
        self, monkeypatch: pytest.MonkeyPatch, loss_norm: str
    ) -> None:
        # Arrange: another rank has 8 rows, so 2 of this rank's 1-row
        # micro-batches hold only a padding row. A KL coefficient makes KL measured.
        torch.manual_seed(0)
        agent = _make_grpo(
            micro_batch_size_per_gpu=1,
            mini_batch_size=1,
            loss_norm=loss_norm,
            beta=0.1,
        )
        pad_to_eight_rows(monkeypatch)

        # Act
        metrics = agent.learn(_experiences(), episode_segments=EPISODE_SEGMENTS)

        # Assert
        for key in ("loss", "kl", "clipfrac", "entropy", "kl_old"):
            assert math.isfinite(metrics[key]), key

    def test_padding_frac_counts_real_rows_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange: real rows hold 36 of 60 positions; 2 padding rows follow.
        torch.manual_seed(0)
        agent = _make_grpo(micro_batch_size_per_gpu=1, mini_batch_size=1)
        pad_to_eight_rows(monkeypatch)

        # Act
        metrics = agent.learn(_experiences(), episode_segments=EPISODE_SEGMENTS)

        # Assert
        assert metrics["padding_frac_before_packing"] == pytest.approx(0.4)

    def test_reports_padded_rows_and_filler_share(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange: 6 real rows pad to 8; the unpacked forward runs whole rows.
        torch.manual_seed(0)
        agent = _make_grpo(micro_batch_size_per_gpu=1, mini_batch_size=1)
        pad_to_eight_rows(monkeypatch)

        # Act
        metrics = agent.learn(_experiences(), episode_segments=EPISODE_SEGMENTS)

        # Assert
        assert metrics["train_rows_padded"] == pytest.approx(8.0)
        assert metrics["filler_token_frac"] == pytest.approx(0.25)

    def test_filler_rows_leave_the_gradient_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange: one optimizer step of 1-row micro-batches over 6 real rows,
        # once alone and once padded to 8 rows by a simulated second rank.
        alone = _make_grpo(micro_batch_size_per_gpu=1)
        padded = _make_grpo(micro_batch_size_per_gpu=1)
        alone_grads = record_step_gradients(alone, monkeypatch)
        padded_grads = record_step_gradients(padded, monkeypatch)

        # Act
        alone.learn(_experiences(), episode_segments=EPISODE_SEGMENTS)
        pad_to_eight_rows(monkeypatch)
        padded_metrics = padded.learn(_experiences(), episode_segments=EPISODE_SEGMENTS)

        # Assert
        assert padded_metrics["train_rows_padded"] == pytest.approx(8.0)
        assert len(alone_grads) == len(padded_grads) == 1
        assert any(grad.abs().sum() > 0 for grad in alone_grads[0].values())
        for name, grad in alone_grads[0].items():
            # fp32 sums over a different micro-batch order.
            assert torch.allclose(padded_grads[0][name], grad, rtol=1e-5, atol=1e-7), (
                name
            )

    def test_segment_count_must_match_episode_count(self) -> None:
        agent = _make_grpo()

        with pytest.raises(
            ValueError, match="episode_segments has 3 entries for 4 episodes"
        ):
            agent.learn(_experiences(), episode_segments=EPISODE_SEGMENTS[:3])

    def test_no_segments_matches_all_none_segments(self) -> None:
        # Arrange
        unset = _make_grpo()
        all_none = _make_grpo()

        # Act
        torch.manual_seed(1)
        unset_metrics = unset.learn(_experiences(), episode_segments=None)
        torch.manual_seed(1)
        all_none_metrics = all_none.learn(_experiences(), episode_segments=[None] * 4)

        # Assert
        timings = {key for key in unset_metrics if key.startswith("learn_phase_")}
        assert unset_metrics.keys() == all_none_metrics.keys()
        for key in unset_metrics.keys() - timings:
            assert all_none_metrics[key] == pytest.approx(
                unset_metrics[key], rel=0.0, abs=0.0, nan_ok=True
            ), key
        unset_weights = lora_weights(unset)
        for name, weight in lora_weights(all_none).items():
            assert torch.equal(weight, unset_weights[name]), name

    def test_trajectory_importance_sampling_is_rejected(self) -> None:
        agent = _make_grpo(importance_sampling_level="trajectory")

        with pytest.raises(
            ValueError, match="Set importance_sampling_level='token' or 'turn'"
        ):
            agent.learn(_experiences(), episode_segments=EPISODE_SEGMENTS)

    @pytest.mark.parametrize(
        ("loss_type", "loss_norm"),
        [
            ("cispo", "micro_batch"),
            ("grpo", "micro_batch"),
            ("grpo", "accumulation_window"),
        ],
    )
    def test_token_normalized_liger_losses_are_accepted(
        self, loss_type: str, loss_norm: str
    ) -> None:
        # Arrange
        agent = _make_grpo(loss_type=loss_type, loss_norm=loss_norm)
        agent.use_liger_loss = True

        # Act
        result = agent._check_segments_supported(agent.importance_sampling_level)

        # Assert
        assert result is None


class TestGRPOLoss:
    @staticmethod
    def _minibatch_pixel_values(
        vision_rows: VisionRows | None, monkeypatch: pytest.MonkeyPatch
    ) -> torch.Tensor | None:
        """Pixel values ``_loss`` hands the objective for rows ``[2, 0]`` of 3."""
        agent = _make_grpo()
        per_token = torch.zeros(3, 3)
        seen: dict[str, torch.Tensor | None] = {}

        def objective(batch_ids: torch.Tensor, *args: Any) -> tuple[torch.Tensor, ...]:
            seen["pixel_values"] = args[-1]
            zero = torch.tensor(0.0)
            return zero, zero, zero, per_token[: batch_ids.shape[0]]

        monkeypatch.setattr(agent, "_objective_loss", objective)
        agent._loss(
            np.array([2, 0]),
            torch.zeros(3, 4, dtype=torch.long),
            per_token.bool(),
            torch.zeros(3, 1),
            per_token,
            per_token,
            vision_rows=vision_rows,
        )
        return seen["pixel_values"]

    def test_minibatch_gets_its_own_vision_rows(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange: rows 0, 1, 2 hold 2, 1 and 3 vision rows.
        pixel_values = torch.arange(6.0).reshape(6, 1)
        vision_rows = VisionRows(pixel_values, image_counts=[2, 1, 3])

        # Act
        selected = self._minibatch_pixel_values(vision_rows, monkeypatch)

        # Assert
        assert selected is not None
        assert torch.equal(selected, pixel_values[[3, 4, 5, 0, 1]])

    def test_text_minibatch_has_no_vision_rows(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert self._minibatch_pixel_values(None, monkeypatch) is None

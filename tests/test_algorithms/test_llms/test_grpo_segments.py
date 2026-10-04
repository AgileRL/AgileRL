# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for GRPO training segmented episodes one segment per row.

Covers splitting episodes into segment rows, padding rows for equal cross-rank
micro-batch counts, the cross-rank window loss normalizer, and ``learn`` with
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

from agilerl.algorithms.grpo import (
    GRPO,
    SegmentRows,
    pad_segment_rows,
    segment_window_layout,
    split_episode_segments,
)
from agilerl.components.llm_rollout_data import EpisodeSegments
from tests.test_algorithms.test_llms.test_grpo_old_logprobs import (
    PAD_TOKEN_ID,
    _make_grpo,
)

PAD = 9
SEQ_LEN = 10
EPISODE_SEGMENTS = [
    EpisodeSegments(token_lengths=torch.tensor([4, 6])),
    None,
    EpisodeSegments(token_lengths=torch.tensor([5, 3])),
    None,
]
REAL_LENGTHS = [10, 8, 8, 10]
NUM_SEGMENT_ROWS = 6
SHORT_MIDDLE_SEGMENTS = [
    EpisodeSegments(
        token_lengths=torch.tensor([4, 2]), pixel_rows=torch.tensor([2, 1])
    ),
    None,
]


def _small_batch() -> dict[str, Any]:
    """Two ``T = 6`` episodes: episode 0 splits ``[3, 3]``, episode 1 is unsegmented."""
    return {
        "token_ids": torch.tensor([[1, 2, 3, 4, 5, 6], [7, 8, 1, 2, PAD, PAD]]),
        "action_masks": torch.tensor(
            [[False, True, False, False, True], [False, True, True, True, False]]
        ),
        "turn_ids": torch.tensor([[-1, 0, -1, -1, 1], [-1, 0, 0, 1, -1]]),
        "sampling_logps": [
            torch.tensor([-0.1, -0.2]),
            torch.tensor([-0.3, -0.4, -0.5]),
        ],
        "pixel_values": torch.arange(10.0).reshape(5, 2),
        "pixel_image_counts": [3, 2],
        "episode_segments": [
            EpisodeSegments(
                token_lengths=torch.tensor([3, 3]), pixel_rows=torch.tensor([1, 2])
            ),
            None,
        ],
    }


def _split_small_batch(advantages: torch.Tensor, **overrides: Any) -> SegmentRows:
    batch = {**_small_batch(), **overrides}
    return split_episode_segments(
        batch["token_ids"],
        batch["action_masks"],
        advantages,
        batch["episode_segments"],
        PAD,
        turn_ids=batch["turn_ids"],
        sampling_logps=batch["sampling_logps"],
        pixel_values=batch["pixel_values"],
        pixel_image_counts=batch["pixel_image_counts"],
    )


def _episode_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """Four ``T = 10`` episodes laid out per ``EPISODE_SEGMENTS``.

    Each segment's first token is prompt and the position predicting the next
    segment's first token is not an action.
    """
    generator = torch.Generator().manual_seed(3)
    ids = torch.randint(0, PAD_TOKEN_ID, (4, SEQ_LEN), generator=generator)
    mask = torch.zeros(4, SEQ_LEN - 1, dtype=torch.bool)
    for episode, segments in enumerate(EPISODE_SEGMENTS):
        lengths = (
            [REAL_LENGTHS[episode]]
            if segments is None
            else segments.token_lengths.tolist()
        )
        start = 0
        for length in lengths:
            mask[episode, start + 1 : start + length - 1] = True
            start += length
        ids[episode, start:] = PAD_TOKEN_ID
    return ids, mask


def _experiences() -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    ids, mask = _episode_batch()
    return list(ids.split(1)), list(mask.split(1)), torch.tensor([1.0, 0.0, 0.0, 1.0])


def _rollout_sampling_logps(agent: GRPO) -> list[torch.Tensor]:
    """Per-episode flat log-probs of the learn-start policy on the segment rows."""
    ids, mask = _episode_batch()
    rows = split_episode_segments(
        ids, mask, torch.zeros(4, 1), EPISODE_SEGMENTS, PAD_TOKEN_ID
    )
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


def _pad_to_eight_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run as one of two ranks whose other rank has 8 segment rows."""
    monkeypatch.setattr("agilerl.algorithms.grpo.get_world_size", lambda: 2)
    monkeypatch.setattr(
        "agilerl.algorithms.grpo.allreduce_minmax_int",
        lambda value: (value, 8) if value == NUM_SEGMENT_ROWS else (value, value),
    )


def _lora_weights(agent: GRPO) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().clone()
        for name, param in agent.actor.named_parameters()
        if "lora_" in name
    }


class TestSplitEpisodeSegments:
    def test_segment_rows_are_the_episode_slices(self) -> None:
        # Act
        rows = _split_small_batch(torch.tensor([[0.5], [-1.0]]))

        # Assert
        assert torch.equal(
            rows.token_ids,
            torch.tensor(
                [
                    [1, 2, 3, PAD, PAD, PAD],
                    [4, 5, 6, PAD, PAD, PAD],
                    [7, 8, 1, 2, PAD, PAD],
                ]
            ),
        )
        assert torch.equal(
            rows.action_masks,
            torch.tensor(
                [
                    [False, True, False, False, False],
                    [False, True, False, False, False],
                    [False, True, True, True, False],
                ]
            ),
        )
        assert rows.turn_ids is not None
        assert torch.equal(
            rows.turn_ids,
            torch.tensor([[-1, 0, -1, -1, -1], [-1, 1, -1, -1, -1], [-1, 0, 0, 1, -1]]),
        )
        assert rows.row_episodes.tolist() == [0, 0, 1]

    def test_episode_advantages_repeat_per_row(self) -> None:
        rows = _split_small_batch(torch.tensor([[0.5], [-1.0]]))

        assert torch.equal(rows.advantages, torch.tensor([[0.5], [0.5], [-1.0]]))

    def test_per_token_advantages_slice_like_the_mask(self) -> None:
        # Arrange
        advantages = torch.tensor(
            [[0.0, 0.5, 0.0, 0.0, 0.7], [0.0, -1.0, -1.0, -2.0, 0.0]]
        )

        # Act
        rows = _split_small_batch(advantages)

        # Assert
        assert torch.equal(
            rows.advantages,
            torch.tensor(
                [
                    [0.0, 0.5, 0.0, 0.0, 0.0],
                    [0.0, 0.7, 0.0, 0.0, 0.0],
                    [0.0, -1.0, -1.0, -2.0, 0.0],
                ]
            ),
        )

    def test_sampling_logps_split_by_row_action_counts(self) -> None:
        rows = _split_small_batch(torch.zeros(2, 1))

        assert rows.sampling_logps is not None
        assert [logps.tolist() for logps in rows.sampling_logps] == [
            pytest.approx([-0.1]),
            pytest.approx([-0.2]),
            pytest.approx([-0.3, -0.4, -0.5]),
        ]

    def test_short_sampling_logps_leave_segment_rows_without_logps(self) -> None:
        # Arrange: episode 0 has 2 action tokens but 1 log-prob.
        sampling_logps = [torch.tensor([-0.1]), torch.tensor([-0.3, -0.4])]

        # Act
        rows = _split_small_batch(torch.zeros(2, 1), sampling_logps=sampling_logps)

        # Assert
        assert rows.sampling_logps is not None
        assert rows.sampling_logps[0] is None
        assert rows.sampling_logps[1] is None
        assert torch.equal(rows.sampling_logps[2], torch.tensor([-0.3, -0.4]))

    def test_pixel_rows_become_per_row_image_counts(self) -> None:
        rows = _split_small_batch(torch.zeros(2, 1))

        assert rows.pixel_image_counts == [1, 2, 2]
        assert rows.pixel_values is not None
        assert torch.equal(rows.pixel_values, torch.arange(10.0).reshape(5, 2))

    def test_all_segmented_rows_shrink_to_the_longest_segment(self) -> None:
        # Arrange
        segments = [
            EpisodeSegments(token_lengths=torch.tensor([3, 3])),
            EpisodeSegments(token_lengths=torch.tensor([2, 4])),
        ]

        # Act
        rows = _split_small_batch(
            torch.zeros(2, 1),
            episode_segments=segments,
            pixel_values=None,
            pixel_image_counts=None,
        )

        # Assert
        assert torch.equal(
            rows.token_ids,
            torch.tensor(
                [[1, 2, 3, PAD], [4, 5, 6, PAD], [7, 8, PAD, PAD], [1, 2, PAD, PAD]]
            ),
        )
        assert rows.action_masks.shape == (4, 3)
        assert rows.row_episodes.tolist() == [0, 0, 1, 1]

    def test_unsegmented_episodes_pass_through(self) -> None:
        # Arrange
        batch = _small_batch()
        advantages = torch.tensor([[0.5], [-1.0]])

        # Act
        rows = _split_small_batch(advantages, episode_segments=[None, None])

        # Assert
        assert torch.equal(rows.token_ids, batch["token_ids"])
        assert torch.equal(rows.action_masks, batch["action_masks"])
        assert torch.equal(rows.turn_ids, batch["turn_ids"])
        assert torch.equal(rows.advantages, advantages)
        assert rows.pixel_image_counts == [3, 2]

    def test_vision_rows_need_image_counts(self) -> None:
        with pytest.raises(ValueError, match="pixel_image_counts is required"):
            _split_small_batch(torch.zeros(2, 1), pixel_image_counts=None)

    def test_vision_batch_needs_segment_pixel_rows(self) -> None:
        segments = [EpisodeSegments(token_lengths=torch.tensor([3, 3])), None]

        with pytest.raises(
            ValueError, match="Episode 0 has segments without pixel_rows"
        ):
            _split_small_batch(torch.zeros(2, 1), episode_segments=segments)


class TestSegmentRowsTrainingRows:
    def test_maps_kept_episodes_to_their_rows(self) -> None:
        rows = _split_small_batch(torch.zeros(2, 1))

        assert rows.training_rows(np.array([1])).tolist() == [2]
        assert rows.training_rows(np.array([0, 1])).tolist() == [0, 1, 2]

    def test_padding_rows_always_train(self) -> None:
        rows = pad_segment_rows(_split_small_batch(torch.zeros(2, 1)), 5, 6, PAD)

        assert rows.training_rows(np.array([1])).tolist() == [2, 3, 4]


class TestSegmentWindowLayout:
    @pytest.mark.parametrize(
        ("num_rows", "optimizer_steps", "micro_batch_size", "expected"),
        [
            (13, 4, 1, (16, 4)),
            (16, 4, 1, (16, 4)),
            (4, 4, 1, (4, 1)),
            (6, 1, 2, (6, 3)),
            (9, 2, 2, (12, 3)),
        ],
    )
    def test_pads_rows_to_whole_windows_of_each_optimizer_step(
        self,
        num_rows: int,
        optimizer_steps: int,
        micro_batch_size: int,
        expected: tuple[int, int],
    ) -> None:
        assert (
            segment_window_layout(num_rows, optimizer_steps, micro_batch_size)
            == expected
        )

    def test_no_rows_need_no_padding(self) -> None:
        assert segment_window_layout(0, 0, 2) == (0, 1)


class TestPadSegmentRows:
    def test_padding_rows_carry_no_actions_or_advantage(self) -> None:
        # Arrange
        rows = _split_small_batch(torch.tensor([[0.5], [-1.0]]))

        # Act
        padded = pad_segment_rows(rows, num_rows=5, width=8, pad_token_id=PAD)

        # Assert
        assert torch.equal(
            padded.token_ids[3:], torch.tensor([[1, 2, 3, PAD, PAD, PAD, PAD, PAD]] * 2)
        )
        assert padded.action_masks.shape == (5, 7)
        assert not padded.action_masks[3:].any()
        assert torch.equal(
            padded.advantages, torch.tensor([[0.5], [0.5], [-1.0], [0.0], [0.0]])
        )
        assert padded.turn_ids is not None
        assert torch.equal(padded.turn_ids[3:], torch.full((2, 7), -1))
        assert padded.row_episodes.tolist() == [0, 0, 1, -1, -1]
        assert padded.sampling_logps is not None
        assert [logps.numel() for logps in padded.sampling_logps[3:]] == [0, 0]

    def test_padding_rows_copy_the_shortest_row(self) -> None:
        # Arrange: rows of 4, 2 and 4 real tokens.
        rows = _split_small_batch(
            torch.zeros(2, 1), episode_segments=SHORT_MIDDLE_SEGMENTS
        )

        # Act
        padded = pad_segment_rows(rows, num_rows=5, width=6, pad_token_id=PAD)

        # Assert
        assert torch.equal(
            padded.token_ids[3:], torch.tensor([[5, 6, PAD, PAD, PAD, PAD]] * 2)
        )
        assert not padded.action_masks[3:].any()

    def test_padding_rows_repeat_the_shortest_row_vision_block(self) -> None:
        # Arrange: the shortest row owns vision row 2.
        rows = _split_small_batch(
            torch.zeros(2, 1), episode_segments=SHORT_MIDDLE_SEGMENTS
        )
        pixel_values = torch.arange(10.0).reshape(5, 2)

        # Act
        padded = pad_segment_rows(rows, num_rows=5, width=6, pad_token_id=PAD)

        # Assert
        assert padded.pixel_image_counts == [2, 1, 2, 1, 1]
        assert padded.pixel_values is not None
        assert torch.equal(
            padded.pixel_values,
            torch.cat([pixel_values, pixel_values[2:3], pixel_values[2:3]]),
        )

    def test_target_width_pads_real_rows(self) -> None:
        # Arrange
        advantages = torch.tensor(
            [[0.0, 0.5, 0.0, 0.0, 0.7], [0.0, -1.0, -1.0, -2.0, 0.0]]
        )
        rows = _split_small_batch(advantages)

        # Act
        padded = pad_segment_rows(rows, num_rows=4, width=8, pad_token_id=PAD)

        # Assert
        assert torch.equal(
            padded.token_ids[:3, 6:], torch.full((3, 2), PAD, dtype=torch.long)
        )
        assert not padded.action_masks[:, 5:].any()
        assert padded.advantages.shape == (4, 7)
        assert torch.equal(padded.advantages[:3, :5], rows.advantages)
        assert torch.equal(padded.advantages[:, 5:], torch.zeros(4, 2))
        assert torch.equal(padded.advantages[3], torch.zeros(7))

    def test_missing_sampling_logps_stay_missing(self) -> None:
        rows = _split_small_batch(torch.zeros(2, 1), sampling_logps=None)

        padded = pad_segment_rows(rows, num_rows=4, width=6, pad_token_id=PAD)

        assert padded.sampling_logps is None


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
        before = _lora_weights(agent)

        # Act
        metrics = agent.learn(_experiences(), episode_segments=EPISODE_SEGMENTS)

        # Assert
        assert math.isfinite(metrics["loss"])
        assert math.isfinite(metrics["kl_old"])
        assert math.isfinite(metrics["entropy"])
        after = _lora_weights(agent)
        assert any(not torch.equal(after[name], before[name]) for name in before)

    def test_padding_rows_count_as_rollout_rows(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange: another rank has 8 rows, so this rank pads its 6 with 2.
        torch.manual_seed(0)
        agent = _make_grpo(old_logprobs_source="rollout")
        sampling_logps = _rollout_sampling_logps(agent)
        _pad_to_eight_rows(monkeypatch)

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
        monkeypatch.setattr("agilerl.algorithms.grpo.get_world_size", lambda: 2)
        monkeypatch.setattr(
            "agilerl.algorithms.grpo.allreduce_minmax_int",
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
        _pad_to_eight_rows(monkeypatch)

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
        _pad_to_eight_rows(monkeypatch)

        # Act
        metrics = agent.learn(_experiences(), episode_segments=EPISODE_SEGMENTS)

        # Assert
        assert metrics["padding_frac_before_packing"] == pytest.approx(0.4)

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
        unset_weights = _lora_weights(unset)
        for name, weight in _lora_weights(all_none).items():
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
        result = agent._check_segments_supported()

        # Assert
        assert result is None

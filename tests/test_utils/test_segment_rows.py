# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``agilerl.utils.segment_rows``."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from agilerl.components.llm_rollout_data import EpisodeSegments
from agilerl.utils.segment_rows import (
    TEXT_FILLER_TOKENS,
    SegmentRows,
    append_vision_tails,
    filler_stand_in,
    filler_token_frac,
    pad_row_values,
    pad_segment_rows,
    segment_window_layout,
    split_episode_segments,
)

PAD = 9
IMG = 8
SHORT_MIDDLE_SEGMENTS = [
    EpisodeSegments(
        token_lengths=torch.tensor([4, 2]), pixel_rows=torch.tensor([2, 1])
    ),
    None,
]


def small_batch() -> dict[str, Any]:
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


def split_small_batch(**overrides: Any) -> SegmentRows:
    batch = {**small_batch(), **overrides}
    return split_episode_segments(
        batch["token_ids"],
        batch["action_masks"],
        batch["episode_segments"],
        PAD,
        turn_ids=batch["turn_ids"],
        sampling_logps=batch["sampling_logps"],
        pixel_values=batch["pixel_values"],
        pixel_image_counts=batch["pixel_image_counts"],
    )


def split_small_text_batch(**overrides: Any) -> SegmentRows:
    return split_small_batch(pixel_values=None, pixel_image_counts=None, **overrides)


def vision_rows() -> SegmentRows:
    """Two rows whose image placeholders take two ids per vision row.

    Row 0 has 7 real tokens and 2 vision rows; its first vision row's ids end at
    position 3. Row 1 has 5 real tokens and 1 vision row ending at position 5.
    """
    return SegmentRows(
        token_ids=torch.tensor(
            [[1, IMG, IMG, 2, IMG, IMG, 3], [4, 5, 6, IMG, IMG, PAD, PAD]]
        ),
        action_masks=torch.tensor(
            [[False, False, True, False, False, True]] * 2, dtype=torch.bool
        ),
        row_episodes=np.array([0, 1], dtype=np.intp),
        row_starts=np.array([0, 0], dtype=np.intp),
        row_ends=np.array([7, 7], dtype=np.intp),
        pixel_values=torch.arange(6.0).reshape(3, 2),
        pixel_image_counts=[2, 1],
    )


class TestSplitEpisodeSegments:
    def test_segment_rows_are_the_episode_slices(self) -> None:
        # Act
        rows = split_small_batch()

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
        assert rows.row_starts.tolist() == [0, 3, 0]
        assert rows.row_ends.tolist() == [3, 6, 6]

    def test_sampling_logps_split_by_row_action_counts(self) -> None:
        rows = split_small_batch()

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
        rows = split_small_batch(sampling_logps=sampling_logps)

        # Assert
        assert rows.sampling_logps is not None
        assert rows.sampling_logps[0] is None
        assert rows.sampling_logps[1] is None
        assert torch.equal(rows.sampling_logps[2], torch.tensor([-0.3, -0.4]))

    def test_pixel_rows_become_per_row_image_counts(self) -> None:
        rows = split_small_batch()

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
        rows = split_small_text_batch(episode_segments=segments)

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
        batch = small_batch()

        # Act
        rows = split_small_batch(episode_segments=[None, None])

        # Assert
        assert torch.equal(rows.token_ids, batch["token_ids"])
        assert torch.equal(rows.action_masks, batch["action_masks"])
        assert torch.equal(rows.turn_ids, batch["turn_ids"])
        assert rows.pixel_image_counts == [3, 2]

    def test_vision_rows_need_image_counts(self) -> None:
        with pytest.raises(ValueError, match="pixel_image_counts is required"):
            split_small_batch(pixel_image_counts=None)

    def test_vision_batch_needs_segment_pixel_rows(self) -> None:
        segments = [EpisodeSegments(token_lengths=torch.tensor([3, 3])), None]

        with pytest.raises(
            ValueError, match="Episode 0 has segments without pixel_rows"
        ):
            split_small_batch(episode_segments=segments)


class TestSegmentRowsTrainingRows:
    def test_maps_kept_episodes_to_their_rows(self) -> None:
        rows = split_small_batch()

        assert rows.training_rows(np.array([1])).tolist() == [2]
        assert rows.training_rows(np.array([0, 1])).tolist() == [0, 1, 2]

    def test_padding_rows_always_train(self) -> None:
        rows = pad_segment_rows(split_small_text_batch(), 5, 6, PAD)

        assert rows.training_rows(np.array([1])).tolist() == [2, 3, 4]


class TestSegmentRowsSplitFrame:
    def test_rows_take_their_episode_positions(self) -> None:
        # Arrange
        frame = torch.arange(10.0).reshape(2, 5)

        # Act
        rows = split_small_text_batch().split_frame(frame, -1.0)

        # Assert
        assert torch.equal(
            rows,
            torch.tensor(
                [
                    [0.0, 1.0, -1.0, -1.0, -1.0],
                    [3.0, 4.0, -1.0, -1.0, -1.0],
                    [5.0, 6.0, 7.0, 8.0, 9.0],
                ]
            ),
        )

    def test_filler_rows_and_widened_positions_get_the_pad_value(self) -> None:
        # Arrange
        rows = pad_segment_rows(split_small_text_batch(), 4, 7, PAD)

        # Act
        split = rows.split_frame(torch.arange(10.0).reshape(2, 5), -1.0)

        # Assert
        assert split.shape == (4, 6)
        assert torch.equal(split[2], torch.tensor([5.0, 6.0, 7.0, 8.0, 9.0, -1.0]))
        assert torch.equal(split[3], torch.full((6,), -1.0))


class TestSegmentRowsMergeFrame:
    def test_round_trips_every_position_a_row_covers(self) -> None:
        # Arrange
        rows = pad_segment_rows(split_small_text_batch(), 4, 6, PAD)
        frame = torch.arange(10.0).reshape(2, 5)

        # Act
        merged = rows.merge_frame(rows.split_frame(frame, 0.0), 2, 5, -1.0)

        # Assert: position 2 predicts segment 1's first token and has no row.
        assert torch.equal(
            merged,
            torch.tensor([[0.0, 1.0, -1.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0, 9.0]]),
        )

    def test_filler_rows_do_not_write_to_the_frame(self) -> None:
        # Arrange
        rows = pad_segment_rows(split_small_text_batch(), 4, 6, PAD)
        values = torch.zeros(4, 5)
        values[3] = 100.0

        # Act
        merged = rows.merge_frame(values, 2, 5, -1.0)

        # Assert
        assert merged.max().item() == 0.0


class TestSegmentRowsSplitAdvantages:
    def test_episode_advantages_repeat_per_row(self) -> None:
        rows = split_small_batch()

        advantages = rows.split_advantages(torch.tensor([[0.5], [-1.0]]))

        assert torch.equal(advantages, torch.tensor([[0.5], [0.5], [-1.0]]))

    def test_per_token_advantages_slice_like_the_mask(self) -> None:
        # Arrange
        advantages = torch.tensor(
            [[0.0, 0.5, 0.0, 0.0, 0.7], [0.0, -1.0, -1.0, -2.0, 0.0]]
        )

        # Act
        rows = split_small_batch().split_advantages(advantages)

        # Assert
        assert torch.equal(
            rows,
            torch.tensor(
                [
                    [0.0, 0.5, 0.0, 0.0, 0.0],
                    [0.0, 0.7, 0.0, 0.0, 0.0],
                    [0.0, -1.0, -1.0, -2.0, 0.0],
                ]
            ),
        )

    def test_filler_rows_get_zero_advantage(self) -> None:
        rows = pad_segment_rows(split_small_text_batch(), 5, 8, PAD)

        advantages = rows.split_advantages(torch.tensor([[0.5], [-1.0]]))

        assert torch.equal(
            advantages, torch.tensor([[0.5], [0.5], [-1.0], [0.0], [0.0]])
        )

    def test_widened_rows_pad_per_token_advantages_with_zero(self) -> None:
        # Arrange
        advantages = torch.tensor(
            [[0.0, 0.5, 0.0, 0.0, 0.7], [0.0, -1.0, -1.0, -2.0, 0.0]]
        )
        rows = pad_segment_rows(split_small_text_batch(), 4, 8, PAD)

        # Act
        split = rows.split_advantages(advantages)

        # Assert
        assert split.shape == (4, 7)
        assert torch.equal(split[2, :5], advantages[1])
        assert torch.equal(split[:, 5:], torch.zeros(4, 2))
        assert torch.equal(split[3], torch.zeros(7))


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
    def test_padding_rows_carry_no_actions_or_episode(self) -> None:
        # Arrange
        rows = split_small_text_batch()

        # Act
        padded = pad_segment_rows(rows, num_rows=5, width=8, pad_token_id=PAD)

        # Assert
        assert torch.equal(
            padded.token_ids[3:], torch.tensor([[1, 2, 3, PAD, PAD, PAD, PAD, PAD]] * 2)
        )
        assert padded.action_masks.shape == (5, 7)
        assert not padded.action_masks[3:].any()
        assert padded.turn_ids is not None
        assert torch.equal(padded.turn_ids[3:], torch.full((2, 7), -1))
        assert padded.row_episodes.tolist() == [0, 0, 1, -1, -1]
        assert padded.row_starts.tolist() == [0, 3, 0, 0, 0]
        assert padded.row_ends.tolist() == [3, 6, 6, 0, 0]
        assert padded.sampling_logps is not None
        assert [logps.numel() for logps in padded.sampling_logps[3:]] == [0, 0]

    def test_padding_rows_copy_the_shortest_row(self) -> None:
        # Arrange: rows of 4, 2 and 4 real tokens.
        rows = split_small_text_batch(episode_segments=SHORT_MIDDLE_SEGMENTS)

        # Act
        padded = pad_segment_rows(rows, num_rows=5, width=6, pad_token_id=PAD)

        # Assert
        assert torch.equal(
            padded.token_ids[3:], torch.tensor([[5, 6, PAD, PAD, PAD, PAD]] * 2)
        )
        assert not padded.action_masks[3:].any()

    def test_text_padding_rows_keep_the_first_tokens_of_the_shortest_row(
        self,
    ) -> None:
        # Arrange: rows of 30 and 24 real tokens.
        token_ids = torch.full((2, 30), PAD)
        token_ids[0] = 1
        token_ids[1, :24] = torch.arange(24).remainder(PAD - 1)
        rows = SegmentRows(
            token_ids=token_ids,
            action_masks=torch.ones(2, 29, dtype=torch.bool),
            row_episodes=np.array([0, 1], dtype=np.intp),
            row_starts=np.array([0, 0], dtype=np.intp),
            row_ends=np.array([30, 30], dtype=np.intp),
        )

        # Act
        padded = pad_segment_rows(rows, num_rows=3, width=30, pad_token_id=PAD)

        # Assert
        filler = padded.token_ids[2]
        assert torch.equal(
            filler[:TEXT_FILLER_TOKENS], token_ids[1, :TEXT_FILLER_TOKENS]
        )
        assert torch.equal(
            filler[TEXT_FILLER_TOKENS:],
            torch.full((30 - TEXT_FILLER_TOKENS,), PAD),
        )
        assert not padded.action_masks[2].any()

    def test_vision_padding_rows_end_after_one_vision_row(self) -> None:
        # Arrange: row 0 is longer but its first vision row ends sooner.
        rows = vision_rows()

        # Act
        padded = pad_segment_rows(
            rows, num_rows=4, width=8, pad_token_id=PAD, image_token_id=IMG
        )

        # Assert
        assert torch.equal(
            padded.token_ids[2:],
            torch.tensor([[1, IMG, IMG, PAD, PAD, PAD, PAD, PAD]] * 2),
        )
        assert not padded.action_masks[2:].any()
        assert padded.row_episodes.tolist() == [0, 1, -1, -1]

    def test_vision_padding_rows_carry_one_vision_row_each(self) -> None:
        # Arrange
        rows = vision_rows()
        pixel_values = torch.arange(6.0).reshape(3, 2)

        # Act
        padded = pad_segment_rows(
            rows, num_rows=4, width=8, pad_token_id=PAD, image_token_id=IMG
        )

        # Assert
        assert padded.pixel_image_counts == [2, 1, 1, 1]
        assert padded.pixel_values is not None
        assert torch.equal(
            padded.pixel_values,
            torch.cat([pixel_values, pixel_values[0:1], pixel_values[0:1]]),
        )

    def test_every_row_keeps_two_placeholder_ids_per_vision_row(self) -> None:
        padded = pad_segment_rows(
            vision_rows(), num_rows=5, width=8, pad_token_id=PAD, image_token_id=IMG
        )

        assert padded.pixel_image_counts is not None
        assert (padded.token_ids == IMG).sum(-1).tolist() == [
            2 * count for count in padded.pixel_image_counts
        ]

    def test_vision_rows_need_the_image_token_id(self) -> None:
        with pytest.raises(ValueError, match="image_token_id is required"):
            pad_segment_rows(vision_rows(), num_rows=4, width=8, pad_token_id=PAD)

    def test_placeholders_must_split_evenly_over_vision_rows(self) -> None:
        # Arrange: row 0 has 5 placeholder ids for 2 vision rows.
        rows = vision_rows()
        rows.token_ids[0, 3] = IMG

        # Act / Assert
        with pytest.raises(
            ValueError,
            match=r"Row 0 has 5 image placeholder ids \(id 8\) for 2 vision rows",
        ):
            pad_segment_rows(
                rows, num_rows=4, width=8, pad_token_id=PAD, image_token_id=IMG
            )

    def test_target_width_pads_real_rows(self) -> None:
        # Arrange
        rows = split_small_text_batch()

        # Act
        padded = pad_segment_rows(rows, num_rows=4, width=8, pad_token_id=PAD)

        # Assert
        assert torch.equal(
            padded.token_ids[:3, 6:], torch.full((3, 2), PAD, dtype=torch.long)
        )
        assert not padded.action_masks[:, 5:].any()

    def test_missing_sampling_logps_stay_missing(self) -> None:
        rows = split_small_text_batch(sampling_logps=None)

        padded = pad_segment_rows(rows, num_rows=4, width=6, pad_token_id=PAD)

        assert padded.sampling_logps is None


class TestPadRowValues:
    def test_per_token_values_widen_and_gain_filler_rows(self) -> None:
        # Arrange: three split rows of five action positions, padded to 5 x 7.
        rows = split_small_text_batch()
        padded = pad_segment_rows(rows, num_rows=5, width=8, pad_token_id=PAD)
        values = torch.arange(15.0).reshape(3, 5)

        # Act
        result = pad_row_values(values, padded, pad_value=-1.0)

        # Assert
        assert result.shape == (5, 7)
        assert torch.equal(result[:3, :5], values)
        assert torch.equal(result[:3, 5:], torch.full((3, 2), -1.0))
        assert torch.equal(result[3:], torch.full((2, 7), -1.0))

    @pytest.mark.parametrize(
        "values",
        [torch.tensor([[0.5], [1.5], [2.5]]), torch.tensor([0.5, 1.5, 2.5])],
        ids=["column", "flat"],
    )
    def test_per_row_values_only_gain_filler_rows(self, values: torch.Tensor) -> None:
        # Arrange
        padded = pad_segment_rows(
            split_small_text_batch(), num_rows=5, width=8, pad_token_id=PAD
        )

        # Act
        result = pad_row_values(values, padded)

        # Assert
        assert result.shape == (5, *values.shape[1:])
        assert result.flatten().tolist() == [0.5, 1.5, 2.5, 0.0, 0.0]

    def test_keeps_the_value_dtype(self) -> None:
        padded = pad_segment_rows(
            split_small_text_batch(), num_rows=4, width=6, pad_token_id=PAD
        )

        result = pad_row_values(torch.tensor([[3], [4], [5]]), padded, pad_value=-1)

        assert result.dtype == torch.long
        assert result.flatten().tolist() == [3, 4, 5, -1]


class TestAppendVisionTails:
    @staticmethod
    def rows_with_a_text_row() -> SegmentRows:
        """Rows with 2, 0 and 1 vision rows; row 1's last real token is the pad id."""
        return SegmentRows(
            token_ids=torch.tensor(
                [
                    [1, IMG, IMG, 2, IMG, IMG, 3],
                    [4, 5, 6, 7, PAD, PAD, PAD],
                    [8, IMG, IMG, 2, 3, PAD, PAD],
                ]
            ),
            action_masks=torch.tensor(
                [
                    [False, False, True, False, False, True],
                    [False, True, True, True, False, False],
                    [False, False, True, True, False, False],
                ]
            ),
            row_episodes=np.array([0, 1, 2], dtype=np.intp),
            row_starts=np.array([0, 0, 0], dtype=np.intp),
            row_ends=np.array([7, 5, 5], dtype=np.intp),
            turn_ids=torch.tensor(
                [
                    [-1, -1, 0, -1, -1, 1],
                    [-1, 0, 0, 0, -1, -1],
                    [-1, -1, 0, 0, -1, -1],
                ]
            ),
            pixel_values=torch.arange(6.0).reshape(3, 2),
            pixel_image_counts=[2, 0, 1],
        )

    def test_text_row_gets_one_vision_row_after_its_real_tokens(self) -> None:
        # Arrange
        rows = self.rows_with_a_text_row()

        # Act
        tailed = append_vision_tails(rows, PAD, IMG)

        # Assert: the tail is row 0's prefix up to its first vision row.
        assert tailed.token_ids.tolist() == [
            [1, IMG, IMG, 2, IMG, IMG, 3, PAD],
            [4, 5, 6, 7, PAD, 1, IMG, IMG],
            [8, IMG, IMG, 2, 3, PAD, PAD, PAD],
        ]
        assert tailed.pixel_image_counts == [2, 1, 1]
        assert tailed.pixel_values is not None
        assert torch.equal(
            tailed.pixel_values,
            torch.tensor([[0.0, 1.0], [2.0, 3.0], [0.0, 1.0], [4.0, 5.0]]),
        )

    def test_tail_positions_hold_no_actions(self) -> None:
        # Arrange
        rows = self.rows_with_a_text_row()

        # Act
        tailed = append_vision_tails(rows, PAD, IMG)

        # Assert
        assert tailed.turn_ids is not None
        assert torch.equal(tailed.action_masks[:, :6], rows.action_masks)
        assert not tailed.action_masks[:, 6].any()
        assert tailed.turn_ids[:, 6].tolist() == [-1, -1, -1]

    def test_rows_that_all_have_vision_rows_are_unchanged(self) -> None:
        rows = vision_rows()

        assert append_vision_tails(rows, PAD, IMG) is rows


class TestFillerTokenFrac:
    def test_packed_forward_counts_real_tokens(self) -> None:
        # Arrange: real rows of 4 and 3 tokens, a filler row of 1 token.
        token_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, PAD], [1, PAD, PAD, PAD]])

        # Act
        frac = filler_token_frac(
            token_ids, np.array([False, False, True]), PAD, packed=True
        )

        # Assert
        assert frac == pytest.approx(1 / 8)

    def test_padded_forward_counts_whole_rows(self) -> None:
        token_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, PAD], [1, PAD, PAD, PAD]])

        frac = filler_token_frac(
            token_ids, np.array([False, False, True]), PAD, packed=False
        )

        assert frac == pytest.approx(1 / 3)

    def test_no_rows_have_no_filler(self) -> None:
        frac = filler_token_frac(
            torch.zeros(0, 4, dtype=torch.long),
            np.zeros(0, dtype=bool),
            PAD,
            packed=True,
        )

        assert frac == 0.0


class TestFillerStandIn:
    def test_marks_the_first_position_of_every_row_as_turn_zero(self) -> None:
        # Arrange
        action_masks = torch.zeros(2, 4, dtype=torch.bool)
        turn_ids = torch.full((2, 4), -1)

        # Act
        mask, turns = filler_stand_in(action_masks, turn_ids)

        # Assert
        assert torch.equal(
            mask, torch.tensor([[True, False, False, False]] * 2, dtype=torch.bool)
        )
        assert torch.equal(turns, torch.tensor([[0, -1, -1, -1]] * 2))

    def test_keeps_the_mask_dtype(self) -> None:
        mask, _ = filler_stand_in(torch.zeros(1, 3), torch.full((1, 3), -1))

        assert mask.dtype == torch.float32
        assert mask.tolist() == [[1.0, 0.0, 0.0]]

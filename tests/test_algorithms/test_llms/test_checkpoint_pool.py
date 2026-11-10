# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Highest-coverage scheduling and the locality filters for a checkpoint pool."""

from __future__ import annotations

import pytest
import torch

from agilerl.algorithms.checkpoint_pool import (
    CheckpointCoverage,
    HistoricalTeacherPool,
    bump_priorities,
    empirical_success_rate,
    kept_rows,
    locality_token_mask,
    select_highest_coverage,
)

EARLY = CheckpointCoverage(10, "checkpoint_10", frozenset({"a", "c"}))
LATE = CheckpointCoverage(20, "checkpoint_20", frozenset({"b"}))


class TestEmpiricalSuccessRate:
    def test_counts_strictly_positive_returns(self) -> None:
        assert empirical_success_rate([1.0, 0.0, -1.0, 0.5]) == pytest.approx(0.5)

    def test_an_empty_sample_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one rollout"):
            empirical_success_rate([])


class TestSelectHighestCoverage:
    def test_picks_the_checkpoint_covering_the_most_misses(self) -> None:
        index = select_highest_coverage(
            [EARLY, LATE],
            student_solved={"a"},
            weights={"a": 1.0, "b": 1.0, "c": 1.0},
        )

        assert index == 0

    def test_breaks_ties_toward_the_earlier_checkpoint(self) -> None:
        same = CheckpointCoverage(30, "checkpoint_30", frozenset({"c"}))

        index = select_highest_coverage(
            [same, EARLY],
            student_solved=set(),
            weights={"c": 1.0},
        )

        assert index == 1

    def test_stops_when_no_candidate_covers_a_miss(self) -> None:
        index = select_highest_coverage(
            [EARLY, LATE],
            student_solved={"a", "b", "c"},
            weights={"a": 1.0, "b": 1.0, "c": 1.0},
        )

        assert index is None

    def test_a_higher_priority_miss_beats_two_base_misses(self) -> None:
        index = select_highest_coverage(
            [EARLY, LATE],
            student_solved=set(),
            weights={"a": 1.0, "b": 5.0, "c": 1.0},
        )

        assert index == 1


class TestBumpPriorities:
    def test_solved_tasks_drop_and_uncovered_tasks_rise(self) -> None:
        updated = bump_priorities(
            {"a": 1.0, "b": 2.0, "c": 1.0, "d": 2.5},
            ["a", "b", "c", "d"],
            student_solved={"a"},
            teacher_solved={"b"},
            increment=1.0,
            cap=3.0,
        )

        assert updated == {"a": 0.0, "b": 1.0, "c": 2.0, "d": 3.0}


class TestHistoricalTeacherPool:
    def test_activates_the_earlier_checkpoint_on_a_tie(self) -> None:
        pool = HistoricalTeacherPool(
            [LATE, EARLY],
            val_ids=["a", "b", "c"],
            student_solved={"a"},
            held_out_score=0.4,
        )

        assert pool.active is not None
        assert pool.active.step == 10
        assert pool.miss_coverage == 2
        assert pool.student_checkpoint.path == "checkpoint_10"

    def test_reschedule_raises_priority_and_can_switch(self) -> None:
        pool = HistoricalTeacherPool(
            [EARLY, LATE],
            val_ids=["a", "b", "c"],
            student_solved={"a"},
            held_out_score=0.4,
            cap=3.0,
        )

        # Each candidate covers one miss at weight 1; the earlier one wins.
        assert pool.active is not None
        assert pool.active.step == 10

        chosen = pool.reschedule({"a"})

        # "b" was not covered by the teacher that just taught, so it outranks "c".
        assert chosen is not None
        assert chosen.step == 20
        assert pool.weights["b"] == 2.0

    def test_reschedule_stops_when_the_student_covers_the_pool(self) -> None:
        pool = HistoricalTeacherPool(
            [EARLY, LATE],
            val_ids=["a", "b", "c"],
            student_solved=set(),
            held_out_score=0.4,
        )

        chosen = pool.reschedule({"a", "b", "c"})

        assert chosen is None
        assert pool.active is None

    def test_one_checkpoint_raises(self) -> None:
        with pytest.raises(ValueError, match="at least two checkpoints"):
            HistoricalTeacherPool(
                [EARLY],
                val_ids=["a"],
                student_solved=set(),
                held_out_score=0.0,
            )

    def test_a_missing_validation_set_raises(self) -> None:
        with pytest.raises(ValueError, match="validation set"):
            HistoricalTeacherPool(
                [EARLY, LATE],
                val_ids=[],
                student_solved=set(),
                held_out_score=0.0,
            )

    def test_empty_coverage_at_the_start_raises(self) -> None:
        with pytest.raises(ValueError, match="covers no validation task"):
            HistoricalTeacherPool(
                [EARLY, LATE],
                val_ids=["a", "b", "c"],
                student_solved={"a", "b", "c"},
                held_out_score=1.0,
            )

    def test_student_checkpoint_breaks_ties_toward_the_later_step(self) -> None:
        tied = CheckpointCoverage(40, "checkpoint_40", frozenset({"a", "c"}))
        pool = HistoricalTeacherPool(
            [EARLY, tied, LATE],
            val_ids=["a", "b", "c"],
            student_solved=set(),
            held_out_score=0.2,
        )

        assert pool.student_checkpoint.step == 40


class TestKeptRows:
    def test_keeps_a_failure_only_when_the_teacher_rate_is_higher(self) -> None:
        kept = kept_rows(
            rewards=[0.0, 1.0, 0.0, 0.0],
            teacher_rates=[1.0, 1.0, 0.0, 0.0],
            group_size=2,
        )

        assert kept == [True, False, False, False]

    def test_equal_rates_including_both_zero_are_dropped(self) -> None:
        kept = kept_rows(
            rewards=[0.0, 0.0, 1.0, 1.0],
            teacher_rates=[0.0, 0.0, 1.0, 1.0],
            group_size=2,
        )

        assert kept == [False, False, False, False]

    def test_averages_per_rollout_teacher_rewards(self) -> None:
        kept = kept_rows(
            rewards=[0.0, 0.0],
            teacher_rates=[1.0, 0.0],
            group_size=2,
        )

        assert kept == [True, True]

    def test_a_partial_group_raises(self) -> None:
        with pytest.raises(ValueError, match="not divisible"):
            kept_rows([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], group_size=2)


class TestLocalityTokenMask:
    def test_a_dropped_row_trains_no_tokens(self) -> None:
        mask = torch.tensor([False, True, False, True])

        dropped = locality_token_mask(mask, kept=False)

        assert not dropped.any()

    def test_tool_return_positions_stay_out_of_a_kept_row(self) -> None:
        # False positions are tool returns and padding.
        mask = torch.tensor([False, True, False, True])

        kept = locality_token_mask(mask, kept=True)

        assert kept.tolist() == [False, True, False, True]

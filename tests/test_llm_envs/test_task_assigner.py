# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for adaptive row sampling in :class:`~agilerl.llm_envs.task_assigner.TaskAssigner`."""

from __future__ import annotations

import json
from collections import Counter

import pytest

from agilerl.llm_envs.task_assigner import TASK_OUTCOME_DECAY, TaskAssigner


def feed(assigner: TaskAssigner, row: int, *, informative: bool, times: int) -> None:
    """Record ``times`` identical outcomes for ``row``."""
    for _ in range(times):
        assigner.record_outcome(row, informative=informative)


class TestTaskAssignerAdaptiveSampling:
    def test_draws_the_informative_row_far_more_often_than_the_tied_row(self) -> None:
        # Arrange
        assigner = TaskAssigner(2, seed=0, adaptive=True)
        feed(assigner, 0, informative=False, times=20)
        feed(assigner, 1, informative=True, times=20)
        observed = (1 - TASK_OUTCOME_DECAY**20) / (1 - TASK_OUTCOME_DECAY)
        tied_weight = 1 / (observed + 2)
        informative_weight = (observed + 1) / (observed + 2)
        tied_share = tied_weight / (tied_weight + informative_weight)

        # Act
        counts = Counter(assigner.next_row() for _ in range(4000))

        # Assert
        stats = {row.row: row for row in assigner.row_stats()}
        assert stats[0].weight == pytest.approx(tied_weight)
        assert stats[1].weight == pytest.approx(informative_weight)
        assert stats[0].observed == pytest.approx(observed)
        assert stats[1].informative == pytest.approx(observed)
        assert counts[1] > 8 * counts[0]
        assert counts[0] / 4000 == pytest.approx(tied_share, abs=0.02)
        assert tied_share > 0.07

    def test_an_always_tied_row_keeps_a_weight_above_seven_percent(self) -> None:
        assigner = TaskAssigner(1, seed=0, adaptive=True)

        feed(assigner, 0, informative=False, times=500)

        [stats] = assigner.row_stats()
        assert 0.07 < stats.weight < 0.072

    def test_unseen_rows_start_at_even_odds(self) -> None:
        assigner = TaskAssigner(3, seed=0, adaptive=True)

        assert [row.weight for row in assigner.row_stats()] == [0.5, 0.5, 0.5]

    def test_a_tied_row_recovers_once_it_turns_informative(self) -> None:
        # Arrange
        assigner = TaskAssigner(1, seed=0, adaptive=True)
        feed(assigner, 0, informative=False, times=20)
        [before] = assigner.row_stats()

        # Act
        feed(assigner, 0, informative=True, times=16)

        # Assert
        [after] = assigner.row_stats()
        assert before.weight < 0.09
        assert after.weight > 0.7

    def test_the_same_seed_and_outcomes_draw_the_same_rows(self) -> None:
        def draws(seed: int) -> list[int]:
            assigner = TaskAssigner(6, seed=seed, adaptive=True)
            feed(assigner, 2, informative=True, times=5)
            feed(assigner, 4, informative=False, times=5)
            return [assigner.next_row() for _ in range(50)]

        assert draws(3) == draws(3)
        assert draws(3) != draws(4)

    def test_draws_stay_in_this_ranks_shard(self) -> None:
        assigner = TaskAssigner(10, seed=0, rank=1, world_size=2, adaptive=True)
        feed(assigner, 7, informative=True, times=10)

        rows = {assigner.next_row() for _ in range(200)}

        assert rows == {5, 6, 7, 8, 9}
        assert [row.row for row in assigner.row_stats()] == [5, 6, 7, 8, 9]

    def test_counts_shard_sized_draws_as_epochs(self) -> None:
        assigner = TaskAssigner(4, seed=0, adaptive=True)

        for _ in range(9):
            assigner.next_row()

        assert assigner.num_epochs == 2

    def test_rejects_an_outcome_for_a_row_outside_the_shard(self) -> None:
        assigner = TaskAssigner(10, seed=0, rank=0, world_size=2, adaptive=True)

        with pytest.raises(ValueError, match=r"row 5 is outside this shard \[0, 5\)"):
            assigner.record_outcome(5, informative=True)


class TestTaskAssignerEpochCycle:
    def test_outcomes_leave_the_epoch_cycle_unchanged_when_not_adaptive(self) -> None:
        # Arrange
        reference = TaskAssigner(5, seed=0)
        assigner = TaskAssigner(5, seed=0)
        feed(assigner, 0, informative=False, times=10)
        feed(assigner, 3, informative=True, times=10)

        # Act
        rows = [assigner.next_row() for _ in range(15)]

        # Assert
        assert rows == [reference.next_row() for _ in range(15)]
        assert sorted(rows[:5]) == list(range(5))
        assert sorted(rows[5:10]) == list(range(5))
        assert assigner.num_epochs == 2


class TestTaskAssignerStateDict:
    def test_lists_only_rows_that_finished_a_group(self) -> None:
        # Arrange
        assigner = TaskAssigner(4, seed=0, adaptive=True)
        assigner.record_outcome(1, informative=True)
        feed(assigner, 3, informative=False, times=2)

        # Act
        state = assigner.state_dict()

        # Assert
        assert state == [
            {"row": 1, "informative": 1.0, "observed": 1.0},
            {"row": 3, "informative": 0.0, "observed": TASK_OUTCOME_DECAY + 1.0},
        ]

    def test_uses_global_row_indices_on_a_later_shard(self) -> None:
        assigner = TaskAssigner(10, seed=0, rank=1, world_size=2, adaptive=True)
        assigner.record_outcome(7, informative=True)

        assert assigner.state_dict() == [
            {"row": 7, "informative": 1.0, "observed": 1.0}
        ]

    def test_is_empty_before_any_outcome(self) -> None:
        assert TaskAssigner(3, seed=0, adaptive=True).state_dict() == []


class TestTaskAssignerLoadStateDict:
    def test_json_round_trip_restores_every_row_weight(self) -> None:
        # Arrange
        saved = TaskAssigner(6, seed=0, adaptive=True)
        feed(saved, 0, informative=False, times=20)
        feed(saved, 2, informative=True, times=5)
        feed(saved, 4, informative=True, times=3)
        feed(saved, 4, informative=False, times=4)
        restored = TaskAssigner(6, seed=0, adaptive=True)

        # Act
        restored.load_state_dict(json.loads(json.dumps(saved.state_dict())))

        # Assert
        assert restored.row_stats() == saved.row_stats()

    def test_restored_counts_steer_the_draws_like_the_original(self) -> None:
        # Arrange
        saved = TaskAssigner(6, seed=5, adaptive=True)
        feed(saved, 1, informative=True, times=10)
        feed(saved, 3, informative=False, times=10)
        restored = TaskAssigner(6, seed=5, adaptive=True)
        restored.load_state_dict(saved.state_dict())

        # Act
        restored_rows = [restored.next_row() for _ in range(50)]

        # Assert
        assert restored_rows == [saved.next_row() for _ in range(50)]

    def test_keeps_only_this_shards_rows_from_a_merged_state(self) -> None:
        # Arrange
        merged = [
            {"row": 1, "informative": 2.0, "observed": 3.0},
            {"row": 6, "informative": 0.5, "observed": 4.0},
        ]
        assigner = TaskAssigner(10, seed=0, rank=1, world_size=2, adaptive=True)

        # Act
        assigner.load_state_dict(merged)

        # Assert
        stats = {row.row: row for row in assigner.row_stats()}
        assert stats[6].informative == 0.5
        assert stats[6].observed == 4.0
        assert stats[6].weight == pytest.approx(1.5 / 6.0)
        assert assigner.state_dict() == [
            {"row": 6, "informative": 0.5, "observed": 4.0}
        ]

    def test_resets_rows_missing_from_the_state_to_unseen(self) -> None:
        # Arrange
        assigner = TaskAssigner(3, seed=0, adaptive=True)
        feed(assigner, 0, informative=True, times=4)
        feed(assigner, 2, informative=False, times=4)

        # Act
        assigner.load_state_dict([{"row": 2, "informative": 1.0, "observed": 1.0}])

        # Assert
        assert [row.weight for row in assigner.row_stats()] == [0.5, 0.5, 2 / 3]

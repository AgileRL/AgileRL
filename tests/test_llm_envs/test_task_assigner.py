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
        assigner.record_outcome(row, informative=informative, success=None)


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

        assert rows == {1, 3, 5, 7, 9}
        assert [row.row for row in assigner.row_stats()] == [1, 3, 5, 7, 9]

    def test_counts_shard_sized_draws_as_epochs(self) -> None:
        assigner = TaskAssigner(4, seed=0, adaptive=True)

        for _ in range(9):
            assigner.next_row()

        assert assigner.num_epochs == 2

    def test_rejects_an_outcome_for_a_row_outside_the_shard(self) -> None:
        assigner = TaskAssigner(10, seed=0, rank=0, world_size=2, adaptive=True)

        with pytest.raises(
            ValueError,
            match=r"row 5 is outside this shard: rank 0 of 2 owns rows 0, 2, \.\.\., 8\.",
        ):
            assigner.record_outcome(5, informative=True, success=None)

    def test_rejects_an_outcome_for_a_row_past_the_shard(self) -> None:
        assigner = TaskAssigner(5, seed=0, rank=0, world_size=2, adaptive=True)

        with pytest.raises(ValueError, match="row 4 is outside this shard"):
            assigner.record_outcome(4, informative=True, success=None)


class TestTaskAssignerSharding:
    @pytest.mark.parametrize("world_size", [2, 4])
    def test_every_rank_gets_the_site_mix_of_a_site_sorted_dataset(
        self, world_size: int
    ) -> None:
        # Arrange
        sites = ["shopping"] * 8 + ["classifieds"] * 4
        ranks = [
            TaskAssigner(len(sites), seed=0, rank=rank, world_size=world_size)
            for rank in range(world_size)
        ]
        shard_size = len(sites) // world_size

        # Act
        mixes = [
            Counter(sites[assigner.next_row()] for _ in range(shard_size))
            for assigner in ranks
        ]

        # Assert
        expected = Counter(shopping=8 // world_size, classifieds=4 // world_size)
        assert mixes == [expected] * world_size

    def test_ranks_draw_disjoint_rows_that_cover_the_dataset(self) -> None:
        ranks = [TaskAssigner(12, seed=0, rank=rank, world_size=3) for rank in range(3)]

        epochs = [[assigner.next_row() for _ in range(4)] for assigner in ranks]

        assert [sorted(rows) for rows in epochs] == [
            [0, 3, 6, 9],
            [1, 4, 7, 10],
            [2, 5, 8, 11],
        ]

    def test_the_same_seed_draws_the_same_rows_on_a_rank(self) -> None:
        def draws(seed: int) -> list[int]:
            assigner = TaskAssigner(20, seed=seed, rank=1, world_size=2)
            return [assigner.next_row() for _ in range(30)]

        assert draws(3) == draws(3)
        assert draws(3) != draws(4)

    def test_a_merged_state_restores_every_ranks_rows(self) -> None:
        # Arrange
        saved = [
            TaskAssigner(10, seed=0, rank=rank, world_size=2, adaptive=True)
            for rank in range(2)
        ]
        feed(saved[0], 4, informative=True, times=3)
        feed(saved[1], 3, informative=False, times=5)
        merged = saved[0].state_dict() + saved[1].state_dict()
        restored = [
            TaskAssigner(10, seed=0, rank=rank, world_size=2, adaptive=True)
            for rank in range(2)
        ]

        # Act
        for assigner in restored:
            assigner.load_state_dict(merged)

        # Assert
        assert [a.row_stats() for a in restored] == [a.row_stats() for a in saved]


def decayed(times: int) -> float:
    """Decayed count after ``times`` outcomes of one row."""
    return (1 - TASK_OUTCOME_DECAY**times) / (1 - TASK_OUTCOME_DECAY)


class TestTaskAssignerFamilyPrior:
    def test_an_unseen_row_inherits_its_familys_pooled_rate(self) -> None:
        # Arrange
        assigner = TaskAssigner(
            5, seed=0, adaptive=True, families=["a", "a", "a", "b", "b"]
        )

        # Act
        feed(assigner, 0, informative=True, times=6)
        feed(assigner, 1, informative=True, times=2)
        feed(assigner, 1, informative=False, times=1)
        feed(assigner, 3, informative=False, times=8)

        # Assert
        weights = [row.weight for row in assigner.row_stats()]
        family_a_informative = decayed(6) + TASK_OUTCOME_DECAY * decayed(2)
        family_a = (family_a_informative + 1) / (decayed(6) + decayed(3) + 2)
        family_b = 1 / (decayed(8) + 2)
        assert weights[2] == pytest.approx(family_a)
        assert weights[4] == pytest.approx(family_b)
        assert weights[2] > 0.75
        assert weights[4] < 0.13

    def test_a_row_with_groups_moves_from_its_family_rate_to_its_own(self) -> None:
        # Arrange
        assigner = TaskAssigner(
            4, seed=0, adaptive=True, families=["a"] * 4, family_prior_strength=2.0
        )
        feed(assigner, 0, informative=True, times=10)
        feed(assigner, 1, informative=True, times=10)
        weights = []

        # Act
        for _ in range(3):
            feed(assigner, 3, informative=False, times=4)
            weights.append(assigner.row_stats()[3].weight)

        # Assert
        observed = decayed(12)
        family_rate = (2 * decayed(10) + 1) / (2 * decayed(10) + observed + 2)
        assert weights[0] > weights[1] > weights[2]
        assert weights[2] == pytest.approx(2 * family_rate / (observed + 2))
        assert weights[2] < 0.15

    def test_a_row_tied_twice_drops_further_below_its_family_than_today(self) -> None:
        # Arrange
        assigner = TaskAssigner(
            3, seed=0, adaptive=True, families=["a"] * 3, family_prior_strength=1.0
        )
        today = TaskAssigner(3, seed=0, adaptive=True)
        for sampler in (assigner, today):
            feed(sampler, 0, informative=True, times=1)
            feed(sampler, 0, informative=False, times=1)

        # Act
        for sampler in (assigner, today):
            feed(sampler, 1, informative=False, times=2)

        # Assert
        [_, tied_twice, unseen] = assigner.row_stats()
        [_, tied_twice_today, unseen_today] = today.row_stats()
        assert tied_twice.weight / unseen.weight == pytest.approx(1 / (decayed(2) + 1))
        assert tied_twice_today.weight / unseen_today.weight == pytest.approx(
            1 / (decayed(2) + 2) / 0.5
        )
        assert tied_twice.weight / unseen.weight < 0.35
        assert tied_twice_today.weight / unseen_today.weight > 0.5

    def test_no_family_keys_give_todays_weights_and_draws(self) -> None:
        # Arrange
        today = TaskAssigner(6, seed=2, adaptive=True)
        keyless = TaskAssigner(6, seed=2, adaptive=True, families=[None] * 6)
        for sampler in (today, keyless):
            feed(sampler, 1, informative=True, times=5)
            feed(sampler, 4, informative=False, times=7)

        # Act
        draws = [keyless.next_row() for _ in range(60)]

        # Assert
        assert keyless.row_stats() == today.row_stats()
        assert draws == [today.next_row() for _ in range(60)]

    def test_a_row_without_a_key_keeps_todays_weight_beside_a_family(self) -> None:
        # Arrange
        assigner = TaskAssigner(3, seed=0, adaptive=True, families=[None, "a", "a"])

        # Act
        feed(assigner, 0, informative=False, times=3)
        feed(assigner, 1, informative=True, times=3)

        # Assert
        [ungrouped, _, unseen] = assigner.row_stats()
        assert ungrouped.weight == pytest.approx(1 / (decayed(3) + 2))
        assert unseen.weight == pytest.approx((decayed(3) + 1) / (decayed(3) + 2))

    def test_reads_a_later_shards_keys_by_global_row(self) -> None:
        # Arrange
        assigner = TaskAssigner(
            4,
            seed=0,
            rank=1,
            world_size=2,
            adaptive=True,
            families=["b", "a", "b", "a"],
        )

        # Act
        feed(assigner, 1, informative=False, times=6)

        # Assert
        [_, unseen] = assigner.row_stats()
        assert unseen.weight == pytest.approx(1 / (decayed(6) + 2))

    def test_rejects_a_key_list_that_does_not_cover_the_dataset(self) -> None:
        with pytest.raises(ValueError, match="families has 2 keys for a 3-row dataset"):
            TaskAssigner(3, families=["a", "b"])

    def test_rejects_a_prior_strength_that_is_not_positive(self) -> None:
        with pytest.raises(
            ValueError, match=r"family_prior_strength must be > 0, got 0\.0"
        ):
            TaskAssigner(2, families=["a", "a"], family_prior_strength=0.0)


class TestTaskAssignerRecordOutcome:
    def test_counts_each_group_success_class_per_row(self) -> None:
        # Arrange
        assigner = TaskAssigner(3, seed=0)
        outcomes = [
            (0, False, "tied_failure"),
            (0, False, "tied_failure"),
            (0, True, "mixed"),
            (2, False, "tied_success"),
            (2, True, None),
        ]

        # Act
        for row, informative, success in outcomes:
            assigner.record_outcome(row, informative=informative, success=success)

        # Assert
        counts = [
            (row.tied_failure, row.mixed, row.tied_success)
            for row in assigner.row_stats()
        ]
        assert counts == [(2, 1, 0), (0, 0, 0), (0, 0, 1)]

    def test_success_counts_do_not_decay(self) -> None:
        # Arrange
        assigner = TaskAssigner(1, seed=0, adaptive=True)

        # Act
        for _ in range(40):
            assigner.record_outcome(0, informative=False, success="tied_failure")

        # Assert
        [stats] = assigner.row_stats()
        assert stats.tied_failure == 40
        assert stats.observed < 40

    def test_counts_are_kept_per_shard_row(self) -> None:
        # Arrange
        assigner = TaskAssigner(10, seed=0, rank=1, world_size=2)

        # Act
        assigner.record_outcome(7, informative=True, success="mixed")

        # Assert
        stats = {row.row: row for row in assigner.row_stats()}
        assert stats[7].mixed == 1
        assert sum(row.mixed for row in stats.values()) == 1


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
        assigner.record_outcome(1, informative=True, success=None)
        feed(assigner, 3, informative=False, times=2)

        # Act
        state = assigner.state_dict()

        # Assert
        assert state == [
            {
                "row": 1,
                "informative": 1.0,
                "observed": 1.0,
                "tied_failure": 0,
                "mixed": 0,
                "tied_success": 0,
            },
            {
                "row": 3,
                "informative": 0.0,
                "observed": TASK_OUTCOME_DECAY + 1.0,
                "tied_failure": 0,
                "mixed": 0,
                "tied_success": 0,
            },
        ]

    def test_lists_each_rows_success_counts(self) -> None:
        # Arrange
        assigner = TaskAssigner(2, seed=0, adaptive=True)
        for success in ("tied_failure", "tied_failure", "mixed", "tied_success"):
            assigner.record_outcome(1, informative=success == "mixed", success=success)

        # Act
        [state] = assigner.state_dict()

        # Assert
        assert (state["tied_failure"], state["mixed"], state["tied_success"]) == (
            2,
            1,
            1,
        )

    def test_uses_global_row_indices_on_a_later_shard(self) -> None:
        assigner = TaskAssigner(10, seed=0, rank=1, world_size=2, adaptive=True)
        assigner.record_outcome(7, informative=True, success=None)

        assert [outcome["row"] for outcome in assigner.state_dict()] == [7]

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

    def test_json_round_trip_restores_every_row_success_count(self) -> None:
        # Arrange
        saved = TaskAssigner(3, seed=0, adaptive=True)
        saved.record_outcome(0, informative=False, success="tied_failure")
        saved.record_outcome(2, informative=True, success="mixed")
        saved.record_outcome(2, informative=False, success="tied_success")
        restored = TaskAssigner(3, seed=0, adaptive=True)
        restored.record_outcome(1, informative=False, success="tied_failure")

        # Act
        restored.load_state_dict(json.loads(json.dumps(saved.state_dict())))

        # Assert
        assert [
            (row.tied_failure, row.mixed, row.tied_success)
            for row in restored.row_stats()
        ] == [(1, 0, 0), (0, 0, 0), (0, 1, 1)]

    def test_a_state_without_success_counts_loads_them_as_zero(self) -> None:
        # Arrange
        assigner = TaskAssigner(2, seed=0, adaptive=True)
        assigner.record_outcome(1, informative=False, success="tied_failure")

        # Act
        assigner.load_state_dict([{"row": 1, "informative": 1.0, "observed": 2.0}])

        # Assert
        [_unseen, row] = assigner.row_stats()
        assert (row.observed, row.tied_failure, row.mixed, row.tied_success) == (
            2.0,
            0,
            0,
            0,
        )

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

    def test_json_round_trip_restores_family_weights_and_draws(self) -> None:
        # Arrange
        families = ["a", "a", "a", "b", "b", None]
        saved = TaskAssigner(6, seed=1, adaptive=True, families=families)
        feed(saved, 0, informative=True, times=4)
        feed(saved, 3, informative=False, times=5)
        feed(saved, 5, informative=True, times=2)
        restored = TaskAssigner(6, seed=1, adaptive=True, families=families)

        # Act
        restored.load_state_dict(json.loads(json.dumps(saved.state_dict())))

        # Assert
        assert restored.row_stats() == saved.row_stats()
        assert [restored.next_row() for _ in range(50)] == [
            saved.next_row() for _ in range(50)
        ]

    def test_a_state_saved_without_families_loads_into_a_family_prior(self) -> None:
        # Arrange
        old_state = [
            {"row": 0, "informative": 3.0, "observed": 3.0},
            {"row": 2, "informative": 0.0, "observed": 2.0},
        ]
        assigner = TaskAssigner(
            4,
            seed=0,
            adaptive=True,
            families=["a", "a", "b", "b"],
            family_prior_strength=1.0,
        )

        # Act
        assigner.load_state_dict(json.loads(json.dumps(old_state)))

        # Assert
        assert [row.weight for row in assigner.row_stats()] == pytest.approx(
            [(3 + 0.8) / 4, 0.8, 0.25 / 3, 0.25]
        )

    def test_keeps_only_this_shards_rows_from_a_merged_state(self) -> None:
        # Arrange
        merged = [
            {"row": 2, "informative": 2.0, "observed": 3.0},
            {"row": 7, "informative": 0.5, "observed": 4.0},
        ]
        assigner = TaskAssigner(10, seed=0, rank=1, world_size=2, adaptive=True)

        # Act
        assigner.load_state_dict(merged)

        # Assert
        stats = {row.row: row for row in assigner.row_stats()}
        assert stats[7].informative == 0.5
        assert stats[7].observed == 4.0
        assert stats[7].weight == pytest.approx(1.5 / 6.0)
        assert [outcome["row"] for outcome in assigner.state_dict()] == [7]

    def test_resets_rows_missing_from_the_state_to_unseen(self) -> None:
        # Arrange
        assigner = TaskAssigner(3, seed=0, adaptive=True)
        feed(assigner, 0, informative=True, times=4)
        feed(assigner, 2, informative=False, times=4)

        # Act
        assigner.load_state_dict([{"row": 2, "informative": 1.0, "observed": 1.0}])

        # Assert
        assert [row.weight for row in assigner.row_stats()] == [0.5, 0.5, 2 / 3]

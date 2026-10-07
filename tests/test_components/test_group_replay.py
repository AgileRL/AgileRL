# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for :mod:`agilerl.components.group_replay`."""

import pytest

from agilerl.components.group_replay import GroupReplay, GroupReplayStore


class TestGroupReplayStoreInit:
    def test_starts_empty(self) -> None:
        store = GroupReplayStore(max_age=3)

        assert len(store) == 0
        assert store.replay_for("task", tied_return=0.0, current_version=0) is None

    def test_negative_max_age_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_age must be >= 0"):
            GroupReplayStore(max_age=-1)

    def test_zero_returns_per_task_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_returns_per_task must be >= 1"):
            GroupReplayStore(max_age=1, max_returns_per_task=0)


class TestGroupReplayStoreRecord:
    def test_keeps_one_trajectory_per_return(self) -> None:
        # Arrange
        store = GroupReplayStore(max_age=3)

        # Act
        store.record(7, "fail", 0.0, rollout_version=4)
        store.record(7, "win", 1.0, rollout_version=4)
        store.record(7, "fail again", 0.0, rollout_version=5)

        # Assert
        assert len(store) == 2
        assert store.replay_for(7, tied_return=1.0, current_version=5) == GroupReplay(
            "fail again", 0.0, age=0
        )

    def test_an_older_trajectory_does_not_replace_a_newer_one(self) -> None:
        store = GroupReplayStore(max_age=3)
        store.record(7, "newer", 1.0, rollout_version=5)

        store.record(7, "older", 1.0, rollout_version=4)

        assert store.replay_for(7, 0.0, current_version=5) == GroupReplay(
            "newer", 1.0, 0
        )

    def test_a_new_return_past_the_bound_evicts_the_oldest(self) -> None:
        # Arrange
        store = GroupReplayStore(max_age=10, max_returns_per_task=2)
        store.record("t", "a", 0.2, rollout_version=1)
        store.record("t", "b", 0.8, rollout_version=2)

        # Act
        store.record("t", "c", 0.5, rollout_version=3)

        # Assert
        assert len(store) == 2
        assert store.replay_for("t", 0.0, current_version=3) == GroupReplay("b", 0.8, 1)
        assert store.replay_for("t", 0.8, current_version=3) == GroupReplay("c", 0.5, 0)

    def test_the_bound_is_per_task(self) -> None:
        store = GroupReplayStore(max_age=10, max_returns_per_task=1)

        store.record("a", "a0", 0.0, rollout_version=1)
        store.record("b", "b0", 0.0, rollout_version=1)

        assert len(store) == 2


class TestGroupReplayStoreReplayFor:
    def test_picks_the_highest_return_other_than_the_tied_one(self) -> None:
        # Arrange
        store = GroupReplayStore(max_age=3)
        for name, episode_return in (("low", 0.2), ("mid", 0.5), ("top", 1.0)):
            store.record("t", name, episode_return, rollout_version=2)

        # Act
        tied_at_mid = store.replay_for("t", 0.5, current_version=2)
        tied_at_top = store.replay_for("t", 1.0, current_version=2)

        # Assert
        assert tied_at_mid == GroupReplay("top", 1.0, 0)
        assert tied_at_top == GroupReplay("mid", 0.5, 0)

    def test_an_entry_at_max_age_is_replayed(self) -> None:
        store = GroupReplayStore(max_age=2)
        store.record("t", "won", 1.0, rollout_version=3)

        assert store.replay_for("t", 0.0, current_version=5) == GroupReplay(
            "won", 1.0, 2
        )

    def test_a_stale_entry_falls_back_to_a_fresh_lower_return(self) -> None:
        store = GroupReplayStore(max_age=2)
        store.record("t", "old win", 1.0, rollout_version=1)
        store.record("t", "partial", 0.5, rollout_version=4)

        assert store.replay_for("t", 0.0, current_version=4) == GroupReplay(
            "partial", 0.5, 0
        )

    def test_only_the_tied_return_stored_gives_no_replay(self) -> None:
        store = GroupReplayStore(max_age=2)
        store.record("t", "fail", 0.0, rollout_version=3)

        assert store.replay_for("t", 0.0, current_version=3) is None


class TestGroupReplayStoreDropStale:
    def test_drops_only_entries_past_max_age(self) -> None:
        # Arrange
        store = GroupReplayStore(max_age=2)
        store.record("old", "a", 1.0, rollout_version=1)
        store.record("edge", "b", 1.0, rollout_version=3)
        store.record("new", "c", 1.0, rollout_version=5)

        # Act
        store.drop_stale(current_version=5)

        # Assert
        assert len(store) == 2
        assert store.replay_for("old", 0.0, current_version=5) is None
        assert store.replay_for("edge", 0.0, current_version=5) == GroupReplay(
            "b", 1.0, 2
        )
        assert store.replay_for("new", 0.0, current_version=5) == GroupReplay(
            "c", 1.0, 0
        )

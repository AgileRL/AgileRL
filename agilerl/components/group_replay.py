# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Keep each task's latest trajectory per episode return to replay into groups whose returns tie."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from typing import Generic, TypeVar

__all__ = ["GroupReplay", "GroupReplayStore", "StoredTrajectory"]

TrajectoryT = TypeVar("TrajectoryT")


@dataclass(frozen=True)
class StoredTrajectory(Generic[TrajectoryT]):
    """A trajectory and the policy version that sampled it.

    :param trajectory: The trajectory, kept exactly as it was sampled.
    :param rollout_version: Policy version the trajectory was sampled under.
    """

    trajectory: TrajectoryT
    rollout_version: int


@dataclass(frozen=True)
class GroupReplay(Generic[TrajectoryT]):
    """A stored trajectory chosen for a tied group, with its return and age.

    :param trajectory: The stored trajectory.
    :param episode_return: The trajectory's episode return.
    :param age: Policy versions between its sampling and the current policy.
    """

    trajectory: TrajectoryT
    episode_return: float
    age: int


class GroupReplayStore(Generic[TrajectoryT]):
    """Each task's most recent trajectory per distinct episode return.

    A task keeps at most ``max_returns_per_task`` returns; recording a new one
    past that evicts the task's oldest entry. Entries older than ``max_age``
    policy versions are never replayed. The store lives in memory only and
    refills from new rollouts after a restart.

    :param max_age: Policy versions a stored trajectory stays eligible for replay.
    :param max_returns_per_task: Distinct returns kept per task.
    """

    def __init__(self, max_age: int, max_returns_per_task: int = 4) -> None:
        """Start an empty store."""
        if max_age < 0:
            msg = f"max_age must be >= 0, got {max_age}."
            raise ValueError(msg)
        if max_returns_per_task < 1:
            msg = f"max_returns_per_task must be >= 1, got {max_returns_per_task}."
            raise ValueError(msg)
        self.max_age = int(max_age)
        self.max_returns_per_task = int(max_returns_per_task)
        self._entries: dict[Hashable, dict[float, StoredTrajectory[TrajectoryT]]] = {}

    def __len__(self) -> int:
        """Return the number of stored trajectories across all tasks."""
        return sum(len(by_return) for by_return in self._entries.values())

    def record(
        self,
        task: Hashable,
        trajectory: TrajectoryT,
        episode_return: float,
        rollout_version: int,
    ) -> None:
        """Make ``trajectory`` the task's entry for ``episode_return`` unless a newer one is stored.

        :param task: Key of the task the trajectory ran on.
        :param trajectory: The trajectory.
        :param episode_return: Its episode return.
        :param rollout_version: Policy version the trajectory was sampled under.
        """
        by_return = self._entries.setdefault(task, {})
        stored = by_return.get(episode_return)
        if stored is not None and rollout_version < stored.rollout_version:
            return
        if stored is None and len(by_return) >= self.max_returns_per_task:
            oldest = min(by_return, key=lambda value: by_return[value].rollout_version)
            del by_return[oldest]
        by_return[episode_return] = StoredTrajectory(trajectory, int(rollout_version))

    def replay_for(
        self,
        task: Hashable,
        tied_return: float,
        current_version: int,
    ) -> GroupReplay[TrajectoryT] | None:
        """Return the task's highest-return entry other than ``tied_return`` within ``max_age``.

        :param task: Key of the task a group tied on.
        :param tied_return: The return every member of the group got.
        :param current_version: Version of the policy being trained.
        :return: The chosen trajectory with its return and age, or ``None``.
        """
        candidates = [
            (episode_return, stored)
            for episode_return, stored in self._entries.get(task, {}).items()
            if episode_return != tied_return
            and int(current_version) - stored.rollout_version <= self.max_age
        ]
        if not candidates:
            return None
        episode_return, stored = max(candidates, key=lambda item: item[0])
        return GroupReplay(
            stored.trajectory,
            episode_return,
            int(current_version) - stored.rollout_version,
        )

    def drop_stale(self, current_version: int) -> None:
        """Delete every entry older than ``max_age`` policy versions.

        :param current_version: Version of the policy being trained.
        """
        oldest = int(current_version) - self.max_age
        kept = {
            task: {
                episode_return: stored
                for episode_return, stored in by_return.items()
                if stored.rollout_version >= oldest
            }
            for task, by_return in self._entries.items()
        }
        self._entries = {
            task: by_return for task, by_return in kept.items() if by_return
        }

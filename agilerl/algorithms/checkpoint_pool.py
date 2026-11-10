# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Schedule frozen GRPO checkpoints and drop distillation rows they should not teach."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CheckpointCoverage:
    """One saved checkpoint and the validation tasks it solved."""

    step: int
    path: str
    solved: frozenset[str]


def select_highest_coverage(
    candidates: Sequence[CheckpointCoverage],
    student_solved: Collection[str],
    weights: Mapping[str, float],
) -> int | None:
    """Index of the checkpoint covering the most weighted student misses.

    Ties break toward the earlier checkpoint, then the earlier index. ``None``
    when every candidate's weighted coverage is zero.

    :param candidates: Checkpoints in any order.
    :type candidates: Sequence[CheckpointCoverage]
    :param student_solved: Validation tasks the student currently solves.
    :type student_solved: Collection[str]
    :param weights: Priority of each validation task.
    :type weights: Mapping[str, float]
    :return: The chosen index, or ``None`` when coverage is empty.
    :rtype: int | None
    """
    student = set(student_solved)
    best_index: int | None = None
    best_key: tuple[float, int, int] | None = None
    for index, candidate in enumerate(candidates):
        coverage = sum(
            weights.get(task, 1.0) for task in candidate.solved if task not in student
        )
        if coverage <= 0.0:
            continue
        key = (coverage, -candidate.step, -index)
        if best_key is None or key > best_key:
            best_key = key
            best_index = index
    return best_index


def bump_priorities(
    weights: Mapping[str, float],
    tasks: Sequence[str],
    student_solved: Collection[str],
    teacher_solved: Collection[str],
    increment: float,
    cap: float,
) -> dict[str, float]:
    """Next task priorities after one scheduling step.

    Solved tasks go to zero. Unsolved tasks the current teacher covers return
    to one. Every other task increases by ``increment``, capped at ``cap``.

    :param weights: Priorities before the update.
    :type weights: Mapping[str, float]
    :param tasks: Validation task ids the pool tracks.
    :type tasks: Sequence[str]
    :param student_solved: Tasks the student solves after the interval.
    :type student_solved: Collection[str]
    :param teacher_solved: Tasks the teacher that just taught solves.
    :type teacher_solved: Collection[str]
    :param increment: Priority added to a task neither side solved.
    :type increment: float
    :param cap: Highest priority a task can reach.
    :type cap: float
    :return: The updated priorities.
    :rtype: dict[str, float]
    """
    student = set(student_solved)
    teacher = set(teacher_solved)
    updated: dict[str, float] = {}
    for task in tasks:
        if task in student:
            updated[task] = 0.0
        elif task in teacher:
            updated[task] = 1.0
        else:
            updated[task] = min(weights.get(task, 1.0) + increment, cap)
    return updated


def best_student_checkpoint(
    checkpoints: Sequence[CheckpointCoverage],
) -> CheckpointCoverage:
    """Checkpoint with the most solved tasks; ties go to the later step.

    :param checkpoints: The pool.
    :type checkpoints: Sequence[CheckpointCoverage]
    :return: The checkpoint the student should start from.
    :rtype: CheckpointCoverage
    :raises ValueError: If ``checkpoints`` is empty.
    """
    if not checkpoints:
        msg = "checkpoint pool needs at least two checkpoints, got 0."
        raise ValueError(msg)
    return max(checkpoints, key=lambda item: (len(item.solved), item.step))


class HistoricalTeacherPool:
    """One active checkpoint, chosen by weighted coverage of the student's misses."""

    def __init__(
        self,
        checkpoints: Sequence[CheckpointCoverage],
        val_ids: Sequence[str],
        student_solved: Collection[str],
        held_out_score: float,
        increment: float = 1.0,
        cap: float = 3.0,
        schedule_interval: int = 10,
    ) -> None:
        """Validate the pool and activate the first teacher.

        :param checkpoints: Saved checkpoints of one run.
        :type checkpoints: Sequence[CheckpointCoverage]
        :param val_ids: Validation task ids. Empty is an error.
        :type val_ids: Sequence[str]
        :param student_solved: Tasks the student solves at the start.
        :type student_solved: Collection[str]
        :param held_out_score: The student's held-out score, logged as a metric.
        :type held_out_score: float
        :param increment: Priority added when neither side solves a task.
        :type increment: float
        :param cap: Highest priority a task can reach.
        :type cap: float
        :param schedule_interval: Learn steps between teacher choices.
        :type schedule_interval: int
        :raises ValueError: If the pool has fewer than two checkpoints, the
            validation ids are empty, or no checkpoint covers a student miss.
        """
        self.checkpoints = list(checkpoints)
        self.tasks = list(dict.fromkeys(val_ids))
        if len(self.checkpoints) < 2:
            msg = (
                "checkpoint pool needs at least two checkpoints, "
                f"got {len(self.checkpoints)}."
            )
            raise ValueError(msg)
        if not self.tasks:
            msg = "checkpoint pool needs a validation set."
            raise ValueError(msg)
        self.increment = increment
        self.cap = cap
        self.schedule_interval = schedule_interval
        self.held_out_score = held_out_score
        self.student_solved = frozenset(student_solved)
        self.weights = dict.fromkeys(self.tasks, 1.0)
        self._active: int | None = None
        if self.activate() is None:
            msg = "checkpoint pool covers no validation task the student misses."
            raise ValueError(msg)

    @property
    def active(self) -> CheckpointCoverage | None:
        """The checkpoint scoring rollouts, or ``None`` after coverage runs out."""
        if self._active is None:
            return None
        return self.checkpoints[self._active]

    @property
    def student_checkpoint(self) -> CheckpointCoverage:
        """Best validation checkpoint, which the student run should resume."""
        return best_student_checkpoint(self.checkpoints)

    @property
    def miss_coverage(self) -> int:
        """Validation tasks some checkpoint solves and the student does not."""
        union: set[str] = set()
        for item in self.checkpoints:
            union.update(item.solved)
        return len(union - set(self.student_solved))

    def activate(self) -> CheckpointCoverage | None:
        """Choose the teacher for the next interval.

        :return: The chosen checkpoint, or ``None`` when coverage is empty.
        :rtype: CheckpointCoverage | None
        """
        index = select_highest_coverage(
            self.checkpoints, self.student_solved, self.weights
        )
        self._active = index
        if index is None:
            return None
        return self.checkpoints[index]

    def reschedule(self, student_solved: Collection[str]) -> CheckpointCoverage | None:
        """Update priorities from the teacher that just taught, then choose again.

        :param student_solved: Tasks the student solves after the interval.
        :type student_solved: Collection[str]
        :return: The next checkpoint, or ``None`` when coverage is empty.
        :rtype: CheckpointCoverage | None
        """
        if self._active is None:
            return None
        teacher = self.checkpoints[self._active]
        self.student_solved = frozenset(student_solved)
        self.weights = bump_priorities(
            self.weights,
            self.tasks,
            self.student_solved,
            teacher.solved,
            self.increment,
            self.cap,
        )
        return self.activate()


def empirical_success_rate(rewards: Sequence[float]) -> float:
    """Share of rollouts whose return is strictly positive.

    :param rewards: One return per rollout of one problem.
    :type rewards: Sequence[float]
    :return: The success rate in ``[0, 1]``.
    :rtype: float
    :raises ValueError: If ``rewards`` is empty.
    """
    if not rewards:
        msg = "A success rate needs at least one rollout."
        raise ValueError(msg)
    return sum(reward > 0.0 for reward in rewards) / len(rewards)


def kept_rows(
    rewards: Sequence[float],
    teacher_rates: Sequence[float],
    group_size: int,
) -> list[bool]:
    """Rows whose problem the teacher solves more often, and which the student failed.

    Rewards and teacher rates are grouped by ``group_size`` (one problem). A
    problem is kept when the teacher's mean rate is strictly higher. Inside a
    kept problem a row is kept when its reward is not strictly positive.

    :param rewards: One outcome reward per student rollout.
    :type rewards: Sequence[float]
    :param teacher_rates: One teacher outcome per row. A repeated problem rate
        and per-rollout 0/1 rewards both average to the problem's rate.
    :type teacher_rates: Sequence[float]
    :param group_size: Rollouts of one problem.
    :type group_size: int
    :return: ``True`` for each row the distillation loss should train.
    :rtype: list[bool]
    :raises ValueError: If the rows do not form complete groups.
    """
    if group_size < 1:
        msg = f"group_size must be >= 1, got {group_size}."
        raise ValueError(msg)
    if len(rewards) != len(teacher_rates) or not rewards:
        msg = (
            "locality filter needs a reward and a teacher rate on every row; "
            f"got {len(rewards)} rewards and {len(teacher_rates)} rates."
        )
        raise ValueError(msg)
    if len(rewards) % group_size != 0:
        msg = (
            f"{len(rewards)} rollout rows are not divisible by group_size {group_size}."
        )
        raise ValueError(msg)
    kept = [False] * len(rewards)
    for start in range(0, len(rewards), group_size):
        stop = start + group_size
        student_rate = sum(rewards[start:stop]) / group_size
        teacher_rate = sum(teacher_rates[start:stop]) / group_size
        if teacher_rate <= student_rate:
            continue
        for offset, reward in enumerate(rewards[start:stop]):
            kept[start + offset] = reward <= 0.0
    return kept


def locality_token_mask(action_mask: torch.Tensor, kept: bool) -> torch.Tensor:
    """Action mask of one row after the failed-rollout filter.

    A dropped row contributes no tokens. Positions that were already false
    (tool returns, observations, padding) stay false.

    :param action_mask: The row's action mask.
    :type action_mask: torch.Tensor
    :param kept: Whether this failed rollout is distilled.
    :type kept: bool
    :return: The mask the loss should see.
    :rtype: torch.Tensor
    """
    if kept:
        return action_mask
    return torch.zeros_like(action_mask)

# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Assign each episode a ``(seed, row_index)`` task; a GRPO group shares one."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict

import torch
from typing_extensions import NotRequired

__all__ = ["GroupSuccess", "TaskAssigner", "TaskRowOutcome", "TaskRowStats"]

# A row's outcome counts halve after this many newer outcomes of that row.
TASK_OUTCOME_HALF_LIFE = 8
# Decayed counts stay below 1 / (1 - decay) ~= 12, so a row's weight stays above ~0.07.
TASK_OUTCOME_DECAY = 0.5 ** (1 / TASK_OUTCOME_HALF_LIFE)

# Whether a group's members all reached the success threshold, none did, or some did.
GroupSuccess = Literal["tied_failure", "mixed", "tied_success"]
GROUP_SUCCESS_KINDS: tuple[GroupSuccess, ...] = (
    "tied_failure",
    "mixed",
    "tied_success",
)


@dataclass(frozen=True)
class TaskRowStats:
    """Decayed group outcomes of one dataset row and the sampling weight they give it.

    :param row: Dataset row index.
    :param informative: Decayed count of groups whose rewards differed.
    :param observed: Decayed count of finished groups.
    :param weight: ``(informative + 1) / (observed + 2)``; ``0.5`` for an unseen row.
    :param tied_failure: Groups where no member reached the success threshold.
    :param mixed: Groups where some members reached the success threshold.
    :param tied_success: Groups where every member reached the success threshold.
    """

    row: int
    informative: float
    observed: float
    weight: float
    tied_failure: int
    mixed: int
    tied_success: int


class TaskRowOutcome(TypedDict):
    """Group outcomes of one dataset row, in the JSON-able form a checkpoint stores.

    The success counts may be absent; they load as zero.

    :param row: Dataset row index.
    :param informative: Decayed count of groups whose rewards differed.
    :param observed: Decayed count of finished groups.
    :param tied_failure: Groups where no member reached the success threshold.
    :param mixed: Groups where some members reached the success threshold.
    :param tied_success: Groups where every member reached the success threshold.
    """

    row: int
    informative: float
    observed: float
    tied_failure: NotRequired[int]
    mixed: NotRequired[int]
    tied_success: NotRequired[int]


def _mix_seed(value: int) -> int:
    """Spread a task seed via splitmix64, truncated to a seed every env accepts.

    The result is masked to 31 bits because an env seeding through numpy rejects
    anything at or above ``2**32``, and that is where a seed most often ends up.
    Truncation costs nothing here: the seeds only have to be far apart, and
    2**31 of them is far more than any run draws.
    """
    z = (value + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
    return (z ^ (z >> 31)) & ((1 << 31) - 1)


class TaskAssigner:
    """Assign each episode a ``(seed, row_index)`` task; a GRPO group shares one.

    Rank ``r`` owns rows ``r, r + world_size, r + 2 * world_size, ...``, so a
    dataset ordered by site or template gives every rank the same mix. Each
    shard has the same size and is reshuffled each epoch; a procedural env has
    no rows and is seeded instead.
    With ``adaptive``, each row is instead drawn from the shard with probability
    proportional to its :class:`TaskRowStats` weight, fed by :meth:`record_outcome`.

    :param dataset_size: Rows in the env's dataset; ``0`` for a procedural env.
    :param seed: Seed for the per-epoch shuffle or weighted draw (``None`` -> a fixed default).
    :param rank: This process's shard index in ``[0, world_size)``.
    :param world_size: Number of data-parallel shards (``1`` = no sharding).
    :param adaptive: Draw rows by recent informative-group rate instead of the epoch cycle.
    """

    def __init__(
        self,
        dataset_size: int,
        *,
        seed: int | None = None,
        rank: int = 0,
        world_size: int = 1,
        adaptive: bool = False,
    ) -> None:
        """Build an assigner over this rank's shard with a seeded per-epoch shuffle."""
        if world_size < 1:
            msg = f"world_size must be >= 1, got {world_size}."
            raise ValueError(msg)
        if not 0 <= rank < world_size:
            msg = f"rank must be in [0, {world_size}), got {rank}."
            raise ValueError(msg)
        self.dataset_size = int(dataset_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self._shard_size = self.dataset_size // self.world_size
        if self.dataset_size > 0 and self._shard_size == 0:
            msg = (
                f"rank {rank} of {world_size} gets an empty shard of a "
                f"{dataset_size}-row dataset; reduce world_size."
            )
            raise ValueError(msg)
        self.adaptive = adaptive
        #: Completed full passes over this rank's shard (shard-sized draws when adaptive).
        self.num_epochs = 0
        self._generator = torch.Generator().manual_seed(
            seed if seed is not None else 42
        )
        self._epoch_order: list[int] = []
        self._pos = 0
        self._draws = 0
        self._informative = torch.zeros(self._shard_size, dtype=torch.float64)
        self._observed = torch.zeros(self._shard_size, dtype=torch.float64)
        # Undecayed group counts per row, one column per GROUP_SUCCESS_KINDS entry.
        self._success_counts = torch.zeros(
            (self._shard_size, len(GROUP_SUCCESS_KINDS)), dtype=torch.long
        )

    def _row_weights(self) -> torch.Tensor:
        """Smoothed informative rate of each shard row, in shard order."""
        return (self._informative + 1.0) / (self._observed + 2.0)

    def _shard_row(self, index: int) -> int:
        """Dataset row at position ``index`` of this rank's shard."""
        return self.rank + index * self.world_size

    def _shard_index(self, row: int) -> int | None:
        """Position of dataset ``row`` in this rank's shard; ``None`` when another rank owns it."""
        index, remainder = divmod(int(row) - self.rank, self.world_size)
        if remainder != 0 or not 0 <= index < self._shard_size:
            return None
        return index

    def next_row(self) -> int:
        """Next row from the shard: the epoch-reshuffled stream, or a weighted draw when adaptive."""
        if self.adaptive:
            index = int(
                torch.multinomial(self._row_weights(), 1, generator=self._generator)
            )
            self._draws += 1
            self.num_epochs = self._draws // self._shard_size
            return self._shard_row(index)
        if self._pos >= len(self._epoch_order):  # epoch boundary (and first call)
            if self._epoch_order:
                self.num_epochs += 1
            self._epoch_order = [
                self._shard_row(index)
                for index in torch.randperm(
                    self._shard_size, generator=self._generator
                ).tolist()
            ]
            self._pos = 0
        row = self._epoch_order[self._pos]
        self._pos += 1
        return row

    def record_outcome(
        self,
        row: int,
        informative: bool,
        success: GroupSuccess | None,
    ) -> None:
        """Fold one finished group's outcome on ``row`` into that row's counts.

        :param row: Dataset row the group ran on; must be in this rank's shard.
        :param informative: Whether the group's rewards differed across members.
        :param success: How many members reached the success threshold; ``None``
            when the env has no threshold.
        :raises ValueError: If ``row`` is outside this rank's shard.
        """
        index = self._shard_index(row)
        if index is None:
            msg = (
                f"row {row} is outside this shard: rank {self.rank} of "
                f"{self.world_size} owns rows {self.rank}, "
                f"{self._shard_row(1)}, ..., {self._shard_row(self._shard_size - 1)}."
            )
            raise ValueError(msg)
        self._informative[index] *= TASK_OUTCOME_DECAY
        self._informative[index] += float(informative)
        self._observed[index] *= TASK_OUTCOME_DECAY
        self._observed[index] += 1.0
        if success is not None:
            self._success_counts[index, GROUP_SUCCESS_KINDS.index(success)] += 1

    def row_stats(self) -> list[TaskRowStats]:
        """Outcomes and sampling weight of every row in this rank's shard."""
        weights = self._row_weights().tolist()
        counts = self._success_counts.tolist()
        return [
            TaskRowStats(
                row=self._shard_row(index),
                informative=float(self._informative[index]),
                observed=float(self._observed[index]),
                weight=float(weights[index]),
                tied_failure=counts[index][0],
                mixed=counts[index][1],
                tied_success=counts[index][2],
            )
            for index in range(self._shard_size)
        ]

    def state_dict(self) -> list[TaskRowOutcome]:
        """Outcome counts of every row in this rank's shard that has finished a group."""
        counts = self._success_counts.tolist()
        return [
            TaskRowOutcome(
                row=self._shard_row(index),
                informative=float(self._informative[index]),
                observed=float(self._observed[index]),
                tied_failure=counts[index][0],
                mixed=counts[index][1],
                tied_success=counts[index][2],
            )
            for index in torch.nonzero(self._observed).flatten().tolist()
        ]

    def load_state_dict(self, state: list[TaskRowOutcome]) -> None:
        """Replace this shard's outcome counts with ``state``; rows not in ``state`` reset to unseen.

        Rows outside this rank's shard are skipped, so ``state`` may hold every
        rank's rows at once.

        :param state: Outcomes from :meth:`state_dict`, from one or more shards.
        """
        self._informative.zero_()
        self._observed.zero_()
        self._success_counts.zero_()
        for outcome in state:
            index = self._shard_index(outcome["row"])
            if index is not None:
                self._informative[index] = float(outcome["informative"])
                self._observed[index] = float(outcome["observed"])
                self._success_counts[index] = torch.tensor(
                    [outcome.get(kind, 0) for kind in GROUP_SUCCESS_KINDS]
                )

    def next_task(
        self,
        base_seed: int | None,
        seed_offset: int,
    ) -> tuple[int | None, int | None]:
        """One group's ``(seed, row_index)``: the mixed seed and the next shard row.

        :param base_seed: Run base seed; ``None`` leaves the env unseeded.
        :param seed_offset: Group offset added to ``base_seed`` before mixing.
        :return: ``(seed, row_index)``; ``row_index`` is ``None`` for a procedural env.
        """
        seed = (
            None if base_seed is None else _mix_seed(int(base_seed) + int(seed_offset))
        )
        row = self.next_row() if self.dataset_size > 0 else None
        return seed, row

    def assign(
        self,
        batch_size: int,
        group_size: int,
        *,
        base_seed: int | None = None,
        seed_offset: int = 0,
    ) -> list[tuple[int | None, int | None]]:
        """Tasks for one batch: ``batch_size * group_size`` ``(seed, row_index)`` pairs.

        Item ``i``'s seed is ``base_seed + seed_offset + i`` spread through
        :func:`_mix_seed`, so consecutive windows are far apart in an env's
        seed space rather than walking it linearly — an env whose seed-to-task
        mapping has short-period structure would otherwise recycle tasks on a
        fixed cycle. The pair is repeated ``group_size`` times so the whole
        group shares one task. Callers must keep group seeds unique across
        batches (advance ``base_seed`` or ``seed_offset``).
        """
        out: list[tuple[int | None, int | None]] = []
        for item in range(batch_size):
            out.extend(
                [self.next_task(base_seed, int(seed_offset) + item)] * group_size
            )
        return out

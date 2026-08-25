# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Assign each episode a ``(seed, row_index)`` task; a GRPO group shares one."""

from __future__ import annotations

import torch

__all__ = ["TaskAssigner"]


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

    Dataset rows are reshuffled each epoch and split into one equal shard per
    data-parallel rank; a procedural env has no rows and is seeded instead.

    :param dataset_size: Rows in the env's dataset; ``0`` for a procedural env.
    :param seed: Seed for the per-epoch shuffle (``None`` -> a fixed default).
    :param rank: This process's shard index in ``[0, world_size)``.
    :param world_size: Number of data-parallel shards (``1`` = no sharding).
    """

    def __init__(
        self,
        dataset_size: int,
        *,
        seed: int | None = None,
        rank: int = 0,
        world_size: int = 1,
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
        self._shard_start = self.rank * self._shard_size
        if self.dataset_size > 0 and self._shard_size == 0:
            msg = (
                f"rank {rank} of {world_size} gets an empty shard of a "
                f"{dataset_size}-row dataset; reduce world_size."
            )
            raise ValueError(msg)
        #: Completed full passes over this rank's shard of the dataset rows.
        self.num_epochs = 0
        self._generator = torch.Generator().manual_seed(
            seed if seed is not None else 42
        )
        self._epoch_order: list[int] = []
        self._pos = 0

    def next_row(self) -> int:
        """Next row from the epoch-reshuffled shard stream (bumps :attr:`num_epochs`)."""
        if self._pos >= len(self._epoch_order):  # epoch boundary (and first call)
            if self._epoch_order:
                self.num_epochs += 1
            self._epoch_order = (
                torch.randperm(self._shard_size, generator=self._generator)
                + self._shard_start
            ).tolist()
            self._pos = 0
        row = self._epoch_order[self._pos]
        self._pos += 1
        return row

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
            seed = (
                None
                if base_seed is None
                else _mix_seed(int(base_seed) + int(seed_offset) + item)
            )
            row = self.next_row() if self.dataset_size > 0 else None
            out.extend([(seed, row)] * group_size)
        return out

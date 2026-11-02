# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Micro-batches of packed LLM rows sized by a per-GPU token budget."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def pack_rows_by_tokens(
    rows: npt.NDArray[np.intp], row_tokens: npt.NDArray[np.int64], budget: int
) -> list[npt.NDArray[np.intp]]:
    """Group ``rows`` in order into micro-batches of at most ``budget`` tokens.

    A row longer than ``budget`` forms a micro-batch alone.

    :param rows: Row indices in training order.
    :param row_tokens: Real tokens per row, indexed by row.
    :param budget: Most tokens per micro-batch.
    :return: Micro-batches of row indices, covering ``rows`` in order.
    """
    micro_batches: list[npt.NDArray[np.intp]] = []
    start = 0
    used = 0
    for position, row in enumerate(rows):
        tokens = int(row_tokens[row])
        if position > start and used + tokens > budget:
            micro_batches.append(rows[start:position])
            start = position
            used = 0
        used += tokens
    if len(rows) > start:
        micro_batches.append(rows[start:])
    return micro_batches


def deal_rows_by_tokens(
    rows: npt.NDArray[np.intp], row_tokens: npt.NDArray[np.int64], sizes: list[int]
) -> list[npt.NDArray[np.intp]]:
    """Deal ``rows`` into bins of at most ``sizes`` rows that hold about the same tokens.

    Rows go out most tokens first, each to the bin with the fewest tokens that
    still has room; ties go to the earlier row and the earlier bin.

    :param rows: Row indices to deal; ``sum(sizes)`` must hold them.
    :param row_tokens: Real tokens per row, indexed by row.
    :param sizes: Most rows per bin.
    :return: One bin of row indices per entry of ``sizes``.
    """
    loads = np.zeros(len(sizes), dtype=np.int64)
    room = np.asarray(sizes, dtype=np.int64)
    bins: list[list[int]] = [[] for _ in sizes]
    for row in rows[np.argsort(-row_tokens[rows], kind="stable")]:
        target = int(np.argmin(np.where(room > 0, loads, np.iinfo(np.int64).max)))
        bins[target].append(int(row))
        loads[target] += row_tokens[row]
        room[target] -= 1
    return [np.asarray(rows_in_bin, dtype=np.intp) for rows_in_bin in bins]


def fits_budget(
    micro_batches: list[npt.NDArray[np.intp]],
    row_tokens: npt.NDArray[np.int64],
    budget: int,
) -> bool:
    """Whether every micro-batch holds a row and at most ``budget`` tokens, or one row.

    :param micro_batches: Micro-batches of row indices.
    :param row_tokens: Real tokens per row, indexed by row.
    :param budget: Most tokens per micro-batch of more than one row.
    :return: Whether ``micro_batches`` fit.
    """
    return all(
        len(rows) == 1 or (len(rows) > 1 and row_tokens[rows].sum() <= budget)
        for rows in micro_batches
    )


def pack_rows_evenly(
    rows: npt.NDArray[np.intp], row_tokens: npt.NDArray[np.int64], budget: int
) -> list[npt.NDArray[np.intp]]:
    """Fewest micro-batches of at most ``budget`` tokens that :func:`deal_rows_by_tokens` fills.

    A row longer than ``budget`` forms a micro-batch alone.

    :param rows: Row indices to pack.
    :param row_tokens: Real tokens per row, indexed by row.
    :param budget: Most tokens per micro-batch.
    :return: Micro-batches of row indices.
    """
    count = -(-int(row_tokens[rows].sum()) // budget)
    while True:
        micro_batches = deal_rows_by_tokens(rows, row_tokens, [len(rows)] * count)
        if fits_budget(micro_batches, row_tokens, budget):
            return micro_batches
        count += 1


def fill_micro_batches(
    micro_batches: list[npt.NDArray[np.intp]],
    spare_rows: npt.NDArray[np.intp],
    row_tokens: npt.NDArray[np.int64],
    count: int,
    budget: int,
) -> list[npt.NDArray[np.intp]]:
    """Spread ``micro_batches`` over ``count``, first adding the shortest ``spare_rows`` they lack.

    Spare rows join, only as many as it takes to hold one row per micro-batch.
    The rows are dealt evenly over ``count`` micro-batches when those fit
    ``budget``, else the largest micro-batches split (:func:`split_micro_batches`).
    Ranks run micro-batches in lockstep, so dealt ones run most tokens first
    and line up with the largest of every other rank.

    :param micro_batches: Micro-batches of row indices that fit ``budget``.
    :param spare_rows: Rows outside ``micro_batches`` that may run.
    :param row_tokens: Real tokens per row, indexed by row.
    :param count: Micro-batches to reach, at least ``len(micro_batches)``.
    :param budget: Most tokens per micro-batch of more than one row.
    :return: ``count`` micro-batches, most tokens first when dealt.
    :raises ValueError: If even the spare rows cannot fill ``count`` micro-batches.
    """
    missing = count - sum(map(len, micro_batches))
    if missing > 0:
        order = np.argsort(row_tokens[spare_rows], kind="stable")[:missing]
        micro_batches = [*micro_batches, *(spare_rows[i : i + 1] for i in order)]
    rows = np.concatenate([np.zeros(0, dtype=np.intp), *micro_batches])
    dealt = deal_rows_by_tokens(rows, row_tokens, [len(rows)] * count)
    if fits_budget(dealt, row_tokens, budget):
        return sorted(dealt, key=lambda rows: -row_tokens[rows].sum())
    return split_micro_batches(micro_batches, row_tokens, count)


def balance_windows(
    windows: list[npt.NDArray[np.intp]],
    trained: npt.NDArray[np.bool_],
    row_tokens: npt.NDArray[np.int64],
) -> list[npt.NDArray[np.intp]]:
    """Deal the ``trained`` rows of ``windows`` over them so each trains about the same tokens.

    Each window keeps its size. Rows outside ``trained`` fill the slots left
    over, in their order.

    :param windows: Row indices of each optimizer window.
    :param trained: ``(R,)`` rows the windows train, indexed by row.
    :param row_tokens: Real tokens per row, indexed by row.
    :return: Windows of the same sizes over the same rows.
    """
    rows = np.concatenate(windows)
    sizes = [len(window) for window in windows]
    dealt = deal_rows_by_tokens(rows[trained[rows]], row_tokens, sizes)
    spare = rows[~trained[rows]]
    balanced = []
    for size, window in zip(sizes, dealt, strict=True):
        take = size - len(window)
        balanced.append(np.concatenate([window, spare[:take]]))
        spare = spare[take:]
    return balanced


def split_micro_batches(
    micro_batches: list[npt.NDArray[np.intp]],
    row_tokens: npt.NDArray[np.int64],
    count: int,
) -> list[npt.NDArray[np.intp]]:
    """Split the largest micro-batches until there are ``count``.

    Each split cuts the multi-row micro-batch with the most tokens where its
    halves are closest in tokens, so no micro-batch grows.

    :param micro_batches: Micro-batches of row indices in training order.
    :param row_tokens: Real tokens per row, indexed by row.
    :param count: Micro-batches to reach, at least ``len(micro_batches)``.
    :return: ``count`` micro-batches covering the same rows in the same order.
    :raises ValueError: If the rows cannot fill ``count`` micro-batches.
    """
    micro_batches = list(micro_batches)
    totals = [int(row_tokens[rows].sum()) for rows in micro_batches]
    while len(micro_batches) < count:
        splittable = [i for i, rows in enumerate(micro_batches) if len(rows) > 1]
        if not splittable:
            msg = (
                f"Cannot split {sum(map(len, micro_batches))} rows into "
                f"{count} micro-batches."
            )
            raise ValueError(msg)
        target = max(splittable, key=totals.__getitem__)
        rows = micro_batches[target]
        prefix = np.cumsum(row_tokens[rows])[:-1]
        cut = int(np.argmin(np.abs(prefix - totals[target] / 2))) + 1
        micro_batches[target : target + 1] = [rows[:cut], rows[cut:]]
        totals[target : target + 1] = [
            int(prefix[cut - 1]),
            totals[target] - int(prefix[cut - 1]),
        ]
    return micro_batches

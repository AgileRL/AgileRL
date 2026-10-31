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

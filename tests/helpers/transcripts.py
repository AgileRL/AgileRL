# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Helpers for reading the assistant mask of an encoded chat transcript."""

from itertools import groupby


def assistant_runs(mask: list[bool]) -> list[tuple[int, int]]:
    """``(start, end)`` of each run of trained tokens."""
    runs: list[tuple[int, int]] = []
    start = 0
    for trained, group in groupby(mask):
        end = start + len(list(group))
        if trained:
            runs.append((start, end))
        start = end
    return runs

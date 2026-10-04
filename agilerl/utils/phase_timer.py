# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Seconds spent in each named phase of one training step."""

from __future__ import annotations

import time
from itertools import pairwise

import torch


class PhaseTimer:
    """Charge the time between consecutive marks to named phases.

    On CUDA each mark records an event on the current stream, and
    :meth:`stop` reads every duration after one synchronize. Elsewhere marks
    read :func:`time.perf_counter`. Marks outside ``start``/``stop`` do nothing.
    """

    def __init__(self) -> None:
        self.marks: list[tuple[str, torch.cuda.Event | float]] | None = None
        self.use_cuda = False

    def start(self, device: torch.device | str) -> None:
        """Open a timing window on ``device``.

        :param device: Device whose stream the CUDA events are recorded on.
        :type device: torch.device | str
        """
        self.use_cuda = torch.device(device).type == "cuda"
        self.marks = [("", self._now())]

    def mark(self, phase: str) -> None:
        """Charge the time since the previous mark to ``phase``.

        :param phase: Name of the phase that just ended.
        :type phase: str
        """
        if self.marks is not None:
            self.marks.append((phase, self._now()))

    def stop(self) -> dict[str, float]:
        """Close the window.

        :return: Seconds per phase, summed over repeated marks.
        :rtype: dict[str, float]
        """
        marks = self.marks or []
        self.marks = None
        if self.use_cuda:
            torch.cuda.synchronize()
        seconds: dict[str, float] = {}
        for (_, begin), (phase, end) in pairwise(marks):
            elapsed = begin.elapsed_time(end) / 1000.0 if self.use_cuda else end - begin
            seconds[phase] = seconds.get(phase, 0.0) + elapsed
        return seconds

    def _now(self) -> torch.cuda.Event | float:
        if not self.use_cuda:
            return time.perf_counter()
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

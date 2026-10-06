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
        self.marks: list[str] | None = None
        self.events: list[torch.cuda.Event] = []
        self.times: list[float] = []
        self.use_cuda = False

    def start(self, device: torch.device | str) -> None:
        """Open a timing window on ``device``.

        :param device: Device whose stream the CUDA events are recorded on.
        :type device: torch.device | str
        """
        self.use_cuda = torch.device(device).type == "cuda"
        self.marks = [""]
        self.events = []
        self.times = []
        self._record()

    def mark(self, phase: str) -> None:
        """Charge the time since the previous mark to ``phase``.

        :param phase: Name of the phase that just ended.
        :type phase: str
        """
        if self.marks is not None:
            self.marks.append(phase)
            self._record()

    def stop(self) -> dict[str, float]:
        """Close the window.

        :return: Seconds per phase, summed over repeated marks.
        :rtype: dict[str, float]
        """
        phases = (self.marks or [])[1:]
        self.marks = None
        if self.use_cuda:
            torch.cuda.synchronize()
            elapsed = [
                begin.elapsed_time(end) / 1000.0 for begin, end in pairwise(self.events)
            ]
        else:
            elapsed = [end - begin for begin, end in pairwise(self.times)]
        self.events = []
        self.times = []
        seconds: dict[str, float] = {}
        for phase, duration in zip(phases, elapsed, strict=True):
            seconds[phase] = seconds.get(phase, 0.0) + duration
        return seconds

    def _record(self) -> None:
        if not self.use_cuda:
            self.times.append(time.perf_counter())
            return
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self.events.append(event)

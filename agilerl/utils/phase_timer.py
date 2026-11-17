# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Seconds spent in each named phase of one training step."""

from __future__ import annotations

import logging
import time
from itertools import pairwise

import torch

logger = logging.getLogger(__name__)


class PhaseTimer:
    """Charge the time between consecutive marks to named phases.

    On CUDA each mark records an event on the current stream, and
    :meth:`stop` reads every duration after one synchronize. Elsewhere marks
    read :func:`time.perf_counter`. Marks outside ``start``/``stop`` do nothing.
    Progress goes to the debug log: the first mark of each phase in a window,
    then at most one mark every 30 s of host time.
    """

    def __init__(self) -> None:
        self.marks: list[str] | None = None
        self.events: list[torch.cuda.Event] = []
        self.times: list[float] = []
        self.use_cuda = False
        self.host_start = 0.0
        self.host_logged = 0.0
        self.logged_phases: set[str] = set()

    def start(self, device: torch.device | str) -> None:
        """Open a timing window on ``device``.

        :param device: Device whose stream the CUDA events are recorded on.
        :type device: torch.device | str
        """
        self.use_cuda = torch.device(device).type == "cuda"
        self.marks = [""]
        self.events = []
        self.times = []
        self.logged_phases = set()
        self.host_start = self._record()
        self.host_logged = self.host_start
        logger.debug("Phase timer: window opened on %s", device)

    def mark(self, phase: str) -> None:
        """Charge the time since the previous mark to ``phase``.

        :param phase: Name of the phase that just ended.
        :type phase: str
        """
        if self.marks is None:
            return
        self.marks.append(phase)
        now = self._record()
        log_interval_s = 30.0
        if phase in self.logged_phases and now - self.host_logged < log_interval_s:
            return
        self.logged_phases.add(phase)
        self.host_logged = now
        # On CUDA the host only queues the mark, so it can run ahead of the GPU.
        logger.debug(
            "Phase timer: %s ended %.1f s (host) into the window",
            phase,
            now - self.host_start,
        )

    def stop(self) -> dict[str, float]:
        """Close the window.

        :return: Seconds per phase, summed over repeated marks.
        :rtype: dict[str, float]
        """
        if self.marks is None:
            return {}
        phases = self.marks[1:]
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
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Phase timer: window closed after %.1f s (host); %s",
                time.perf_counter() - self.host_start,
                ", ".join(
                    f"{phase}={duration:.2f}s" for phase, duration in seconds.items()
                ),
            )
        return seconds

    def _record(self) -> float:
        now = time.perf_counter()
        if self.use_cuda:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self.events.append(event)
        else:
            self.times.append(now)
        return now

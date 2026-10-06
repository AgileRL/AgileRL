# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :class:`agilerl.utils.phase_timer.PhaseTimer` on CPU."""

from __future__ import annotations

import pytest

from agilerl.utils.phase_timer import PhaseTimer


def _drive_clock(monkeypatch: pytest.MonkeyPatch, ticks: list[float]) -> None:
    clock = iter(ticks)
    monkeypatch.setattr(
        "agilerl.utils.phase_timer.time.perf_counter",
        lambda: next(clock),
    )


class TestPhaseTimerStop:
    def test_charges_each_gap_to_the_mark_that_ends_it(self, monkeypatch):
        # Arrange
        _drive_clock(monkeypatch, [10.0, 12.0, 12.5])
        timer = PhaseTimer()

        # Act
        timer.start("cpu")
        timer.mark("slow")
        timer.mark("fast")
        seconds = timer.stop()

        # Assert
        assert seconds == {"slow": 2.0, "fast": 0.5}

    def test_sums_repeated_phases(self, monkeypatch):
        # Arrange
        _drive_clock(monkeypatch, [0.0, 1.0, 1.25, 2.25, 2.5])
        timer = PhaseTimer()

        # Act
        timer.start("cpu")
        for _ in range(2):
            timer.mark("step")
            timer.mark("other")
        seconds = timer.stop()

        # Assert
        assert seconds == {"step": 2.0, "other": 0.5}

    def test_marks_outside_a_window_are_ignored(self):
        # Arrange
        timer = PhaseTimer()

        # Act
        timer.mark("before")
        timer.start("cpu")
        timer.mark("inside")
        inside = timer.stop()
        timer.mark("after")
        after = timer.stop()

        # Assert
        assert set(inside) == {"inside"}
        assert after == {}

    def test_start_discards_an_unclosed_window(self):
        # Arrange
        timer = PhaseTimer()

        # Act
        timer.start("cpu")
        timer.mark("abandoned")
        timer.start("cpu")
        timer.mark("kept")
        seconds = timer.stop()

        # Assert
        assert set(seconds) == {"kept"}
        assert seconds["kept"] == pytest.approx(0.0, abs=0.01)

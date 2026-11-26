# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :class:`agilerl.utils.phase_timer.PhaseTimer` on CPU."""

from __future__ import annotations

import logging

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
        _drive_clock(monkeypatch, [10.0, 12.0, 12.5, 13.0])
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
        _drive_clock(monkeypatch, [0.0, 1.0, 1.25, 2.25, 2.5, 3.0])
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


class TestPhaseTimerDebugLog:
    def test_logs_open_first_mark_of_each_phase_and_close(self, monkeypatch, caplog):
        # Arrange
        _drive_clock(monkeypatch, [0.0, 1.0, 2.5, 3.0, 4.0])
        caplog.set_level(logging.DEBUG, logger="agilerl.utils.phase_timer")
        timer = PhaseTimer()

        # Act
        timer.start("cpu")
        timer.mark("forward")
        timer.mark("backward")
        timer.mark("forward")
        timer.stop()

        # Assert
        assert caplog.messages == [
            "Phase timer: window opened on cpu",
            "Phase timer: forward ended 1.0 s (host) into the window",
            "Phase timer: backward ended 2.5 s (host) into the window",
            (
                "Phase timer: window closed after 4.0 s (host); "
                "forward=1.50s, backward=1.50s"
            ),
        ]

    def test_logs_a_repeated_phase_again_after_30_s(self, monkeypatch, caplog):
        # Arrange
        _drive_clock(monkeypatch, [0.0, 1.0, 20.0, 31.0, 40.0, 41.0])
        caplog.set_level(logging.DEBUG, logger="agilerl.utils.phase_timer")
        timer = PhaseTimer()

        # Act
        timer.start("cpu")
        for _ in range(4):
            timer.mark("forward")
        seconds = timer.stop()

        # Assert
        assert caplog.messages[1:-1] == [
            "Phase timer: forward ended 1.0 s (host) into the window",
            "Phase timer: forward ended 31.0 s (host) into the window",
        ]
        assert seconds == {"forward": 40.0}

    def test_a_new_window_logs_each_phase_again(self, monkeypatch, caplog):
        # Arrange
        _drive_clock(monkeypatch, [0.0, 1.0, 2.0, 10.0, 11.0, 12.0])
        caplog.set_level(logging.DEBUG, logger="agilerl.utils.phase_timer")
        timer = PhaseTimer()

        # Act
        timer.start("cpu")
        timer.mark("forward")
        timer.stop()
        timer.start("cpu")
        timer.mark("forward")
        timer.stop()

        # Assert
        first_mark = "Phase timer: forward ended 1.0 s (host) into the window"
        assert caplog.messages.count(first_mark) == 2

    def test_stop_outside_a_window_logs_nothing(self, caplog):
        # Arrange
        caplog.set_level(logging.DEBUG, logger="agilerl.utils.phase_timer")
        timer = PhaseTimer()

        # Act
        seconds = timer.stop()

        # Assert
        assert seconds == {}
        assert caplog.messages == []

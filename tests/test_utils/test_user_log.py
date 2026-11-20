# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``agilerl.utils.user_log``."""

from __future__ import annotations

import logging

import pytest

from agilerl.utils.user_log import USER_LOGGER_NAME, format_duration, log_for_user


class TestLogForUser:
    def test_logs_on_the_user_logger_with_event_and_fields(self, caplog):
        with caplog.at_level(logging.INFO, logger=USER_LOGGER_NAME):
            log_for_user("step_progress", "Step 3/20", step=3, max_steps=20)

        (record,) = caplog.records
        assert record.name == "agilerl.user"
        assert record.levelno == logging.INFO
        assert record.getMessage() == "Step 3/20"
        assert record.user_event == "step_progress"
        assert record.user_fields == {"step": 3, "max_steps": 20}

    def test_record_names_the_calling_code(self, caplog):
        with caplog.at_level(logging.INFO, logger=USER_LOGGER_NAME):
            log_for_user("eval", "Eval at step 1")

        assert caplog.records[0].filename == "test_user_log.py"

    def test_warning_level_is_kept(self, caplog):
        with caplog.at_level(logging.INFO, logger=USER_LOGGER_NAME):
            log_for_user("reward_error", "2 episodes failed", logging.WARNING)

        assert [r.levelno for r in caplog.records] == [logging.WARNING]


class TestFormatDuration:
    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (0.4, "0s"),
            (12.0, "12s"),
            (220.0, "3m 40s"),
            (3600.0, "1h 0m"),
            (6480.4, "1h 48m"),
        ],
    )
    def test_renders_the_two_largest_units(self, seconds, expected):
        assert format_duration(seconds) == expected

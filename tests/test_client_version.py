# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Client version header and Arena's upgrade signals."""

from __future__ import annotations

import json

import httpx
import pytest

from agilerl.client_version import (
    CLIENT_UPDATE_HEADER,
    SNOWFLAKE_UPGRADE_COMMAND,
    announce_update,
    print_update_notice,
    upgrade_required_message,
)
from agilerl.models import __version__

UPGRADE_DESCRIPTION = (
    "agilerl 0.9.0 is older than 1.0.0, the oldest version this Arena "
    'supports. Update with: pip install -U "agilerl>=1.0.0"'
)
UPGRADE_BODY = {"error": "client_upgrade_required", "description": UPGRADE_DESCRIPTION}


@pytest.fixture(autouse=True)
def fresh_process() -> None:
    announce_update.cache_clear()


class TestPrintUpdateNotice:
    def test_a_newer_release_prints_one_line_per_process(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Arrange
        response = httpx.Response(200, headers={CLIENT_UPDATE_HEADER: "9.9.9"})

        # Act
        print_update_notice(response)
        print_update_notice(response)

        # Assert
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == (
            f"agilerl 9.9.9 is available (you have {__version__}). "
            "Run: pip install -U agilerl\n"
        )

    def test_the_snowflake_command_is_shown_when_given(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        response = httpx.Response(200, headers={CLIENT_UPDATE_HEADER: "9.9.9"})

        print_update_notice(response, SNOWFLAKE_UPGRADE_COMMAND)

        assert capsys.readouterr().err.endswith(
            'Run: pip install -U "agilerl[snowflake]"\n'
        )

    @pytest.mark.parametrize("headers", [{}, {CLIENT_UPDATE_HEADER: " "}])
    def test_no_newer_release_prints_nothing(
        self, capsys: pytest.CaptureFixture[str], headers: dict[str, str]
    ) -> None:
        print_update_notice(httpx.Response(200, headers=headers))

        assert capsys.readouterr().err == ""


class TestUpgradeRequiredMessage:
    def test_reads_the_description_of_the_error_envelope(self) -> None:
        assert upgrade_required_message(json.dumps(UPGRADE_BODY)) == UPGRADE_DESCRIPTION

    def test_a_plain_text_body_is_the_message(self) -> None:
        assert upgrade_required_message("Upgrade agilerl.") == ("Upgrade agilerl.")

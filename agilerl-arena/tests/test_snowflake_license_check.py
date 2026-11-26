# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Licence check the driver-check job runs inside Snowflake."""

from __future__ import annotations

import runpy
import warnings

import httpx
import pytest

from agilerl.arena.snowflake import license as license_mod
from agilerl.arena.snowflake.license import MISSING_CREDENTIALS
from agilerl.arena.snowflake.license_check import main

ENABLED_BODY = {
    "ok": True,
    "data": {"schemaVersion": 1, "features": {"snowflakeTraining": True}},
}
REAL_CLIENT = httpx.Client


def answer_with(
    monkeypatch: pytest.MonkeyPatch, response: httpx.Response | Exception
) -> list[str]:
    """Route the licence check's HTTP client to one scripted answer."""
    urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(
        license_mod.httpx,
        "Client",
        lambda: REAL_CLIENT(transport=httpx.MockTransport(handler)),
    )
    return urls


@pytest.fixture
def head_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARENA_ORG_KEY", "arena_org_key")
    monkeypatch.setenv("ARENA_EXTERNAL_USER_ID", "user-1")
    monkeypatch.setenv("ARENA_BASE_URL", "https://arena.internal")
    monkeypatch.setenv("SNOWFLAKE_ACCOUNT", "ORG-ACCOUNT")


class TestMain:
    @pytest.mark.usefixtures("head_env")
    def test_allowed_training_prints_ok(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        urls = answer_with(monkeypatch, httpx.Response(200, json=ENABLED_BODY))

        code = main()

        assert code == 0
        assert capsys.readouterr().out == "arena-license-check: ok\n"
        assert urls == ["https://arena.internal/api/cli/v1/capabilities"]

    @pytest.mark.usefixtures("head_env")
    def test_a_placeholder_page_prints_the_licence_error(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        answer_with(monkeypatch, httpx.Response(200, html="<html>Parked</html>"))

        code = main()

        assert code == 1
        assert capsys.readouterr().out == (
            "arena-license-check: https://arena.internal answered HTTP 200 with "
            "text/html, which is not the Arena API; the Arena URL must be "
            "reachable from the public internet.\n"
        )

    @pytest.mark.usefixtures("head_env")
    def test_an_unreachable_host_prints_the_cause(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        answer_with(monkeypatch, httpx.ConnectError("Name or service not known"))

        code = main()

        assert code == 1
        assert capsys.readouterr().out == (
            "arena-license-check: Could not reach Arena at https://arena.internal: "
            "Name or service not known.\n"
        )

    def test_missing_secrets_print_the_credentials_error(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.delenv("ARENA_ORG_KEY", raising=False)
        monkeypatch.delenv("ARENA_EXTERNAL_USER_ID", raising=False)
        urls = answer_with(monkeypatch, httpx.Response(200, json=ENABLED_BODY))

        code = main()

        assert code == 1
        assert (
            capsys.readouterr().out == f"arena-license-check: {MISSING_CREDENTIALS}\n"
        )
        assert urls == []

    @pytest.mark.usefixtures("head_env")
    def test_module_entrypoint_exits_with_the_result_code(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        answer_with(monkeypatch, httpx.Response(200, json=ENABLED_BODY))

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*found in sys.modules.*",
                category=RuntimeWarning,
            )
            with pytest.raises(SystemExit) as raised:
                runpy.run_module(
                    "agilerl.arena.snowflake.license_check", run_name="__main__"
                )

        assert raised.value.code == 0
        assert capsys.readouterr().out == "arena-license-check: ok\n"

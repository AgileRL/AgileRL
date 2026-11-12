# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Arena licence check and Snowflake usage posts from the head."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from pathlib import Path

import httpx
import pytest
from typing_extensions import Self

from agilerl.arena.client_version import (
    CLIENT_UPDATE_HEADER,
    CLIENT_VERSION_HEADER,
    announce_update,
)
from agilerl.arena.models import __version__
from agilerl.arena.snowflake import license as license_mod
from agilerl.arena.snowflake.license import (
    AUTH_REJECTED,
    COMPLETE_FAILED,
    HEARTBEAT_FAILED,
    MISSING_CREDENTIALS,
    OPEN_FAILED,
    SNOWFLAKE_DENIED,
    RunRenewalRefusedError,
    SnowflakeLicenseError,
    arena_api_base_url,
    check_snowflake_license,
    complete_snowflake_run,
    fetch_capabilities,
    heartbeat_snowflake_run,
    open_head_run,
    retry_after_seconds,
)

CONTRACT = json.loads(
    (Path(__file__).parent / "fixtures" / "snowflake_usage_contract.json").read_text(
        encoding="utf-8"
    )
)
RUN_ID = "arena-20261109163457-abc123"
BASE_URL = "https://arena.test"
ENV = {
    "ARENA_ORG_KEY": "org-key",
    "ARENA_EXTERNAL_USER_ID": "user-1",
    "SNOWFLAKE_ACCOUNT": "ACME-PROD",
    "ARENA_BASE_URL": BASE_URL,
}
# Two GPU_NV_M nodes of four GPUs; job.json "gpus" is the run total.
JOB = {"run_id": RUN_ID, "nodes": 2, "gpus": 8, "family": "GPU_NV_M"}
STARTED_AT = datetime(2026, 11, 11, 18, 0, tzinfo=timezone.utc)
TOKEN_BODY = {"ok": True, "data": {"runToken": "v1.k1.payload.sig"}}
ENABLED_BODY = {
    "ok": True,
    "data": {"schemaVersion": 1, "features": {"snowflakeTraining": True}},
}
OUTDATED = (
    "agilerl-arena 0.9.0 is no longer supported; the minimum is 1.0.0. "
    'Run: pip install -U "agilerl-arena[snowflake]"'
)
OUTDATED_BODY = {"error": "client_upgrade_required", "description": OUTDATED}
HEAD_OUTDATED = (
    "agilerl-arena 0.9.0 is no longer supported; the minimum is 1.0.0. "
    'Run: pip install -U "agilerl-arena[snowflake]", then run arena snowflake '
    "setup to update the trainer image."
)
NOT_ARENA = (
    "which is not the Arena API; the Arena URL must be reachable from the "
    "public internet."
)


def reply(
    status: int, body: object | None = None, headers: dict[str, str] | None = None
) -> httpx.Response:
    return httpx.Response(status, json=body or {"ok": status < 400}, headers=headers)


def flag_off() -> httpx.Response:
    return httpx.Response(
        403,
        json={
            "error": "forbidden",
            "description": "Snowflake training is not enabled for this organization",
        },
    )


class Arena:
    """Stands in for ``httpx.Client``; answers each request with the next scripted reply."""

    def __init__(self, replies: list[httpx.Response | Exception]) -> None:
        self.replies = replies
        self.calls: list[dict[str, object]] = []

    def __call__(self) -> Arena:
        return self

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    def get(self, url: str, headers: dict[str, str], timeout: float) -> httpx.Response:
        return self._answer("GET", url, headers, None, timeout)

    def post(
        self, url: str, headers: dict[str, str], json: object, timeout: float
    ) -> httpx.Response:
        return self._answer("POST", url, headers, json, timeout)

    def _answer(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        body: object,
        timeout: float,
    ) -> httpx.Response:
        self.calls.append(
            {
                "method": method,
                "url": url,
                "headers": headers,
                "json": body,
                "timeout": timeout,
            }
        )
        answer = self.replies.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


@pytest.fixture
def arena(monkeypatch: pytest.MonkeyPatch) -> Arena:
    fake = Arena([])
    monkeypatch.setattr(license_mod.httpx, "Client", fake)
    return fake


def contract_url(name: str) -> str:
    return BASE_URL + CONTRACT[name]["path"].replace("{runId}", RUN_ID)


class TestUsageContract:
    def test_open_sends_the_contract_keys_and_returns_the_token(
        self, arena: Arena
    ) -> None:
        # Arrange
        arena.replies = [reply(201, TOKEN_BODY)]

        # Act
        token = open_head_run(JOB, ENV, STARTED_AT, lambda seconds: None)

        # Assert
        assert token == "v1.k1.payload.sig"
        call = arena.calls[0]
        assert call["method"] == CONTRACT["open"]["method"]
        assert call["url"] == contract_url("open")
        assert set(call["json"]) == set(CONTRACT["open"]["request"])
        assert call["json"] == {
            "runId": RUN_ID,
            "snowflakeAccount": "ACME-PROD",
            "gpuCount": 8,
            "gpuFamily": "GPU_NV_M",
            "startedAt": "2026-11-11T18:00:00+00:00",
        }

    def test_heartbeat_sends_an_empty_body_and_returns_the_new_token(
        self, arena: Arena
    ) -> None:
        arena.replies = [reply(200, TOKEN_BODY)]

        token = heartbeat_snowflake_run(RUN_ID, ENV, lambda seconds: None)

        assert token == "v1.k1.payload.sig"
        call = arena.calls[0]
        assert call["method"] == CONTRACT["heartbeat"]["method"]
        assert call["url"] == contract_url("heartbeat")
        assert call["json"] == {}
        assert CONTRACT["heartbeat"]["request"] == []

    def test_complete_sends_the_contract_keys(self, arena: Arena) -> None:
        # Arrange
        arena.replies = [reply(200)]
        before = datetime.now(timezone.utc)

        # Act
        complete_snowflake_run(RUN_ID, ENV, lambda seconds: None)

        # Assert
        call = arena.calls[0]
        assert call["method"] == CONTRACT["complete"]["method"]
        assert call["url"] == contract_url("complete")
        body = call["json"]
        assert set(body) == set(CONTRACT["complete"]["request"]) == {"endedAt"}
        assert datetime.fromisoformat(body["endedAt"]) >= before
        assert call["timeout"] == 30.0

    def test_complete_with_no_budget_makes_one_short_attempt(
        self, arena: Arena
    ) -> None:
        arena.replies = [reply(503), reply(200)]
        sleeps: list[float] = []

        with pytest.raises(SnowflakeLicenseError, match=re.escape(COMPLETE_FAILED)):
            complete_snowflake_run(RUN_ID, ENV, sleeps.append, 0.0, 10.0)

        assert sleeps == []
        assert [call["timeout"] for call in arena.calls] == [10.0]

    def test_complete_retries_reuse_one_ended_at(self, arena: Arena) -> None:
        arena.replies = [reply(503), reply(200)]

        complete_snowflake_run(RUN_ID, ENV, lambda seconds: None)

        assert arena.calls[0]["json"] == arena.calls[1]["json"]

    def test_every_post_carries_the_org_key_and_user_headers(
        self, arena: Arena
    ) -> None:
        arena.replies = [reply(201, TOKEN_BODY), reply(200, TOKEN_BODY), reply(200)]

        open_head_run(JOB, ENV, STARTED_AT, lambda seconds: None)
        heartbeat_snowflake_run(RUN_ID, ENV, lambda seconds: None)
        complete_snowflake_run(RUN_ID, ENV, lambda seconds: None)

        for call in arena.calls:
            assert set(call["headers"]) == set(CONTRACT["headers"])
            assert call["headers"]["Authorization"] == "Bearer org-key"
            assert call["headers"]["X-External-User-Id"] == "user-1"


class TestPostRetries:
    def test_transport_error_is_retried(self, arena: Arena) -> None:
        arena.replies = [httpx.ConnectError("refused"), reply(201, TOKEN_BODY)]
        sleeps: list[float] = []

        token = open_head_run(JOB, ENV, STARTED_AT, sleeps.append)

        assert token == "v1.k1.payload.sig"
        assert sleeps == [2.0]

    def test_server_errors_back_off_exponentially(self, arena: Arena) -> None:
        arena.replies = [reply(503), reply(502), reply(200)]
        sleeps: list[float] = []

        complete_snowflake_run(RUN_ID, ENV, sleeps.append)

        assert sleeps == [2.0, 4.0]
        assert len(arena.calls) == 3

    def test_429_waits_for_retry_after_seconds(self, arena: Arena) -> None:
        arena.replies = [
            reply(429, headers={"Retry-After": "7"}),
            reply(200, TOKEN_BODY),
        ]
        sleeps: list[float] = []

        heartbeat_snowflake_run(RUN_ID, ENV, sleeps.append)

        assert sleeps == [7.0]

    def test_gives_up_before_five_minutes_of_waiting(self, arena: Arena) -> None:
        # Arrange
        arena.replies = [reply(503) for _ in range(20)]
        sleeps: list[float] = []

        # Act
        with pytest.raises(SnowflakeLicenseError, match=re.escape(COMPLETE_FAILED)):
            complete_snowflake_run(RUN_ID, ENV, sleeps.append)

        # Assert
        assert sleeps == [2.0, 4.0, 8.0, 16.0, 32.0, 60.0, 60.0, 60.0]
        assert len(arena.calls) == 9

    def test_transport_failure_past_the_budget_names_the_post(
        self, arena: Arena
    ) -> None:
        arena.replies = [httpx.ConnectError("refused") for _ in range(20)]

        with pytest.raises(SnowflakeLicenseError, match=re.escape(HEARTBEAT_FAILED)):
            heartbeat_snowflake_run(RUN_ID, ENV, lambda seconds: None)

    @pytest.mark.parametrize(
        ("status", "message"),
        [
            (400, f"{COMPLETE_FAILED} Arena answered 400."),
            (401, AUTH_REJECTED),
            (403, AUTH_REJECTED),
            (409, f"{COMPLETE_FAILED} Arena answered 409."),
        ],
    )
    def test_client_errors_are_not_retried(
        self, arena: Arena, status: int, message: str
    ) -> None:
        arena.replies = [reply(status)]
        sleeps: list[float] = []

        with pytest.raises(SnowflakeLicenseError, match=re.escape(message)):
            complete_snowflake_run(RUN_ID, ENV, sleeps.append)

        assert sleeps == []
        assert len(arena.calls) == 1


class TestOpenHeadRun:
    def test_rejected_credentials_name_the_org_key(self, arena: Arena) -> None:
        arena.replies = [reply(401)]

        with pytest.raises(SnowflakeLicenseError, match=re.escape(AUTH_REJECTED)):
            open_head_run(JOB, ENV, STARTED_AT, lambda seconds: None)

    def test_response_without_a_run_token_fails(self, arena: Arena) -> None:
        arena.replies = [reply(201, {"ok": True, "data": {}})]

        with pytest.raises(SnowflakeLicenseError, match="no runToken") as caught:
            open_head_run(JOB, ENV, STARTED_AT, lambda seconds: None)

        assert str(caught.value).startswith(OPEN_FAILED)

    def test_missing_credentials_send_nothing(self, arena: Arena) -> None:
        env = {**ENV, "ARENA_ORG_KEY": " "}

        with pytest.raises(SnowflakeLicenseError, match=re.escape(MISSING_CREDENTIALS)):
            open_head_run(JOB, env, STARTED_AT, lambda seconds: None)

        assert arena.calls == []

    def test_missing_snowflake_account_sends_nothing(self, arena: Arena) -> None:
        env = {**ENV, "SNOWFLAKE_ACCOUNT": ""}

        with pytest.raises(
            SnowflakeLicenseError, match="SNOWFLAKE_ACCOUNT is required"
        ):
            open_head_run(JOB, env, STARTED_AT, lambda seconds: None)

        assert arena.calls == []

    def test_flag_off_asks_agilerl_to_enable_snowflake(self, arena: Arena) -> None:
        arena.replies = [flag_off()]

        with pytest.raises(SnowflakeLicenseError) as caught:
            open_head_run(JOB, ENV, STARTED_AT, lambda seconds: None)

        assert str(caught.value) == SNOWFLAKE_DENIED
        assert len(arena.calls) == 1

    def test_auth_layer_403_names_the_org_key(self, arena: Arena) -> None:
        arena.replies = [httpx.Response(403, text="Forbidden")]

        with pytest.raises(SnowflakeLicenseError) as caught:
            open_head_run(JOB, ENV, STARTED_AT, lambda seconds: None)

        assert str(caught.value) == AUTH_REJECTED

    def test_non_json_success_body_fails(self, arena: Arena) -> None:
        arena.replies = [httpx.Response(200, text="<html>proxy</html>")]

        with pytest.raises(SnowflakeLicenseError, match="not JSON") as caught:
            open_head_run(JOB, ENV, STARTED_AT, lambda seconds: None)

        assert str(caught.value).startswith(OPEN_FAILED)

    def test_upgrade_required_fails_without_a_retry(self, arena: Arena) -> None:
        arena.replies = [reply(426, OUTDATED_BODY)]

        with pytest.raises(SnowflakeLicenseError) as caught:
            open_head_run(JOB, ENV, STARTED_AT, lambda seconds: None)

        assert str(caught.value) == HEAD_OUTDATED
        assert len(arena.calls) == 1


class TestHeartbeatSnowflakeRun:
    @pytest.mark.parametrize(
        ("answer", "status"),
        [
            (flag_off, 403),
            (lambda: reply(409, {"error": "conflict"}), 409),
        ],
    )
    def test_flag_off_or_run_ended_refuses_renewal(
        self, arena: Arena, answer: Callable[[], httpx.Response], status: int
    ) -> None:
        arena.replies = [answer()]

        with pytest.raises(RunRenewalRefusedError, match=f"answered {status}"):
            heartbeat_snowflake_run(RUN_ID, ENV, lambda seconds: None)

        assert len(arena.calls) == 1

    def test_auth_layer_403_is_not_a_refused_renewal(self, arena: Arena) -> None:
        arena.replies = [httpx.Response(403, text="Forbidden")]

        with pytest.raises(SnowflakeLicenseError) as caught:
            heartbeat_snowflake_run(RUN_ID, ENV, lambda seconds: None)

        assert not isinstance(caught.value, RunRenewalRefusedError)
        assert str(caught.value) == AUTH_REJECTED

    def test_non_json_success_body_fails(self, arena: Arena) -> None:
        arena.replies = [httpx.Response(200, text="ok")]

        with pytest.raises(SnowflakeLicenseError, match="not JSON") as caught:
            heartbeat_snowflake_run(RUN_ID, ENV, lambda seconds: None)

        assert str(caught.value).startswith(HEARTBEAT_FAILED)

    def test_rejected_org_key_is_not_a_refused_renewal(self, arena: Arena) -> None:
        arena.replies = [reply(401)]

        with pytest.raises(SnowflakeLicenseError) as caught:
            heartbeat_snowflake_run(RUN_ID, ENV, lambda seconds: None)

        assert not isinstance(caught.value, RunRenewalRefusedError)
        assert str(caught.value) == AUTH_REJECTED

    def test_unknown_run_fails_without_a_retry(self, arena: Arena) -> None:
        arena.replies = [reply(404)]

        with pytest.raises(SnowflakeLicenseError, match="answered 404"):
            heartbeat_snowflake_run(RUN_ID, ENV, lambda seconds: None)

        assert len(arena.calls) == 1


class TestCheckSnowflakeLicense:
    def test_enabled_flag_passes(self, arena: Arena) -> None:
        arena.replies = [reply(200, ENABLED_BODY)]

        check_snowflake_license(ENV)

        assert arena.calls[0]["method"] == "GET"
        assert arena.calls[0]["url"] == f"{BASE_URL}/api/cli/v1/capabilities"
        assert arena.calls[0]["headers"]["Authorization"] == "Bearer org-key"

    def test_disabled_flag_asks_agilerl_to_enable_it(self, arena: Arena) -> None:
        body = {
            "ok": True,
            "data": {"schemaVersion": 1, "features": {"snowflakeTraining": False}},
        }
        arena.replies = [reply(200, body)]

        with pytest.raises(SnowflakeLicenseError) as caught:
            check_snowflake_license(ENV)

        assert str(caught.value) == SNOWFLAKE_DENIED
        assert "Ask AgileRL to enable Snowflake training" in SNOWFLAKE_DENIED

    @pytest.mark.parametrize("status", [401, 403])
    def test_rejected_credentials_name_the_org_key(
        self, arena: Arena, status: int
    ) -> None:
        arena.replies = [reply(status)]

        with pytest.raises(SnowflakeLicenseError, match=re.escape(AUTH_REJECTED)):
            check_snowflake_license(ENV)

    def test_features_that_are_not_an_object_deny(self, arena: Arena) -> None:
        body = {
            "ok": True,
            "data": {"schemaVersion": 1, "features": ["snowflakeTraining"]},
        }
        arena.replies = [reply(200, body)]

        with pytest.raises(SnowflakeLicenseError) as caught:
            check_snowflake_license(ENV)

        assert str(caught.value) == SNOWFLAKE_DENIED

    @pytest.mark.parametrize(
        ("answer", "message"),
        [
            (reply(503), "Arena at https://arena.test returned HTTP 503."),
            (
                httpx.Response(200, html="<html>placeholder</html>"),
                f"https://arena.test answered HTTP 200 with text/html, {NOT_ARENA}",
            ),
            (
                reply(200, {"ok": True, "data": ["snowflakeTraining"]}),
                f"https://arena.test answered HTTP 200 with application/json, {NOT_ARENA}",
            ),
            (
                reply(
                    200,
                    {
                        "ok": True,
                        "data": {
                            "schemaVersion": 2,
                            "features": {"snowflakeTraining": True},
                        },
                    },
                ),
                f"https://arena.test answered HTTP 200 with application/json, {NOT_ARENA}",
            ),
            (
                httpx.ConnectError("[Errno -2] Name or service not known"),
                (
                    "Could not reach Arena at https://arena.test: "
                    "[Errno -2] Name or service not known."
                ),
            ),
        ],
        ids=["server-error", "html", "data-not-object", "other-schema", "dns"],
    )
    def test_unreadable_capabilities_name_the_arena_url(
        self, arena: Arena, answer: httpx.Response | Exception, message: str
    ) -> None:
        arena.replies = [answer]

        with pytest.raises(SnowflakeLicenseError) as caught:
            check_snowflake_license(ENV)

        assert str(caught.value) == message
        assert len(arena.calls) == 1

    def test_sends_the_client_version(self, arena: Arena) -> None:
        arena.replies = [reply(200, ENABLED_BODY)]

        check_snowflake_license(ENV)

        assert arena.calls[0]["headers"][CLIENT_VERSION_HEADER] == __version__

    def test_upgrade_required_fails_the_run_start(self, arena: Arena) -> None:
        arena.replies = [reply(426, OUTDATED_BODY)]

        with pytest.raises(SnowflakeLicenseError) as caught:
            check_snowflake_license(ENV)

        assert str(caught.value) == HEAD_OUTDATED

    def test_the_head_prints_no_update_notice(
        self, arena: Arena, capsys: pytest.CaptureFixture[str]
    ) -> None:
        announce_update.cache_clear()
        arena.replies = [
            reply(200, ENABLED_BODY, headers={CLIENT_UPDATE_HEADER: "9.9.9"})
        ]

        check_snowflake_license(ENV)

        assert capsys.readouterr().err == ""


class TestFetchCapabilities:
    def test_returns_the_capabilities_data(self, arena: Arena) -> None:
        arena.replies = [reply(200, ENABLED_BODY)]

        data = fetch_capabilities(BASE_URL, {"Authorization": "Bearer org-key"})

        assert data == ENABLED_BODY["data"]

    @pytest.mark.parametrize(
        ("answer", "message"),
        [
            (
                httpx.Response(
                    200,
                    html="<html>Site coming soon</html>",
                    headers={"content-type": "text/html; charset=utf-8"},
                ),
                f"https://arena.test answered HTTP 200 with text/html, {NOT_ARENA}",
            ),
            (
                httpx.ConnectError("[Errno 8] nodename nor servname provided"),
                (
                    "Could not reach Arena at https://arena.test: "
                    "[Errno 8] nodename nor servname provided."
                ),
            ),
            (
                httpx.ConnectTimeout("timed out"),
                "Could not reach Arena at https://arena.test: timed out.",
            ),
            (reply(500), "Arena at https://arena.test returned HTTP 500."),
            (reply(401), AUTH_REJECTED),
        ],
        ids=["html-200", "connection-error", "timeout", "server-error", "rejected"],
    )
    def test_a_failed_request_says_what_went_wrong(
        self, arena: Arena, answer: httpx.Response | Exception, message: str
    ) -> None:
        arena.replies = [answer]

        with pytest.raises(SnowflakeLicenseError) as caught:
            fetch_capabilities(f"{BASE_URL}/", {"Authorization": "Bearer org-key"})

        assert str(caught.value) == message

    def test_upgrade_required_shows_the_server_description_as_is(
        self, arena: Arena
    ) -> None:
        arena.replies = [reply(426, OUTDATED_BODY)]

        with pytest.raises(SnowflakeLicenseError) as caught:
            fetch_capabilities(BASE_URL, {"Authorization": "Bearer org-key"})

        assert str(caught.value) == OUTDATED

    def test_an_update_header_prints_one_notice(
        self, arena: Arena, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Arrange
        announce_update.cache_clear()
        update = {CLIENT_UPDATE_HEADER: "9.9.9"}
        arena.replies = [
            reply(200, ENABLED_BODY, headers=update),
            reply(200, ENABLED_BODY, headers=update),
        ]

        # Act
        fetch_capabilities(BASE_URL, {})
        data = fetch_capabilities(BASE_URL, {})

        # Assert
        assert data == ENABLED_BODY["data"]
        assert capsys.readouterr().err == (
            f"agilerl-arena 9.9.9 is available (you have {__version__}). "
            'Run: pip install -U "agilerl-arena[snowflake]"\n'
        )


class TestArenaApiBaseUrl:
    def test_a_bare_host_is_read_as_https(self) -> None:
        assert arena_api_base_url("arena.test") == "https://arena.test"

    def test_scheme_port_and_path_are_dropped(self) -> None:
        assert arena_api_base_url("http://Arena.Test:8080/api") == "https://arena.test"

    @pytest.mark.parametrize("value", ["https://*.agilerl.com", "https://", "a b.com"])
    def test_a_value_that_is_not_one_host_is_rejected(self, value: str) -> None:
        with pytest.raises(ValueError, match="ARENA_BASE_URL must name one host"):
            arena_api_base_url(value)


class TestRetryAfterSeconds:
    def test_integer_seconds(self) -> None:
        assert retry_after_seconds(reply(429, headers={"Retry-After": " 12 "})) == 12.0

    def test_http_date(self) -> None:
        when = datetime.now(timezone.utc) + timedelta(seconds=120)
        response = reply(
            503, headers={"Retry-After": format_datetime(when, usegmt=True)}
        )

        assert 118.0 <= retry_after_seconds(response) <= 120.0

    def test_past_date_waits_zero(self) -> None:
        response = reply(503, headers={"Retry-After": "Wed, 21 Oct 2015 07:28:00 GMT"})

        assert retry_after_seconds(response) == 0.0

    def test_absent_header(self) -> None:
        assert retry_after_seconds(reply(503)) is None

    @pytest.mark.parametrize("value", ["soon", "Wed, 21 Oct 2015 07:28:00 -0000", "-5"])
    def test_unreadable_header_is_logged_and_ignored(
        self, value: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        response = reply(503, headers={"Retry-After": value})

        assert retry_after_seconds(response) is None
        assert "unreadable Retry-After" in caplog.text

    def test_unreadable_header_falls_back_to_backoff(self, arena: Arena) -> None:
        arena.replies = [
            reply(503, headers={"Retry-After": "soon"}),
            reply(200, TOKEN_BODY),
        ]
        sleeps: list[float] = []

        heartbeat_snowflake_run(RUN_ID, ENV, sleeps.append)

        assert sleeps == [2.0]

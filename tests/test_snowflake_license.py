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

from agilerl.client_version import (
    CLIENT_UPDATE_HEADER,
    CLIENT_VERSION_HEADER,
    announce_update,
)
from agilerl.models import __version__
from agilerl.snowflake import license as license_mod
from agilerl.snowflake.license import (
    AUTH_REJECTED,
    CHANNEL_ENV,
    CLAIM_START_FAILED,
    COMPLETE_FAILED,
    DEPLOYMENT_CLOSE_FAILED,
    DEPLOYMENT_HEARTBEAT_FAILED,
    DEPLOYMENT_OPEN_FAILED,
    HEARTBEAT_FAILED,
    MISSING_CREDENTIALS,
    NATIVE_APP_CHANNEL,
    OPEN_FAILED,
    PURCHASED_ENV,
    REGION_ENV,
    ROTATE_FAILED,
    SNOWFLAKE_DENIED,
    TRIAL_ENDED,
    USAGE_FAILED,
    AppKey,
    AuthRejectedError,
    ClaimAccount,
    ClaimClosedError,
    ClaimLinkedError,
    ClaimWaitingError,
    DeploymentClosedError,
    DeploymentRefusedError,
    RunRenewalRefusedError,
    SnowflakeClaim,
    SnowflakeDeployment,
    SnowflakeLicenseError,
    SnowflakeUpgradeRequiredError,
    SnowflakeUsage,
    TrainingDisabledError,
    UnknownUsageError,
    arena_api_base_url,
    check_snowflake_license,
    close_snowflake_deployment,
    collect_snowflake_claim_key,
    complete_snowflake_run,
    fetch_capabilities,
    heartbeat_snowflake_deployment,
    heartbeat_snowflake_run,
    open_head_run,
    open_snowflake_deployment,
    partner_headers,
    prove_snowflake_claim,
    retry_after_seconds,
    rotate_snowflake_app_key,
    snowflake_priced_families,
    snowflake_usage,
    start_snowflake_claim,
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
    "agilerl 0.9.0 is older than 1.0.0, the oldest version this Arena "
    'supports. Update with: pip install -U "agilerl[snowflake]>=1.0.0"'
)
OUTDATED_BODY = {"error": "client_upgrade_required", "description": OUTDATED}
HEAD_OUTDATED = (
    "agilerl 0.9.0 is older than 1.0.0, the oldest version this Arena "
    'supports. Update with: pip install -U "agilerl[snowflake]>=1.0.0", '
    "then upgrade the Arena app: ALTER APPLICATION <app> UPGRADE."
)
NOT_ARENA = (
    "which is not the Arena API; the Arena URL must be reachable from the "
    "public internet."
)
DEPLOYMENT_ID = "QWEN_CHECKPOINT_2-20261116T130000Z"
DEPLOYMENT = SnowflakeDeployment(
    deployment_id=DEPLOYMENT_ID,
    snowflake_account="AB12345",
    snowflake_region="AWS_EU_WEST_2",
    gpu_count=1,
    gpu_family="GPU_NV_S",
    started_at=datetime(2026, 11, 16, 13, 0, tzinfo=timezone.utc),
    model="qwen",
    version="checkpoint_2",
    run_id=RUN_ID,
    purchased=True,
)
INACTIVE = "The Snowflake entitlement is not active."
HEADERS = partner_headers("org-key", "user-1")
OBSERVED_AT = datetime(2026, 11, 16, 15, 0, tzinfo=timezone.utc)


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

    def test_open_sends_the_channel_and_region_only_with_their_variables(
        self, arena: Arena
    ) -> None:
        # Arrange
        arena.replies = [reply(201, TOKEN_BODY), reply(201, TOKEN_BODY)]
        app_env = {
            **ENV,
            CHANNEL_ENV: NATIVE_APP_CHANNEL,
            REGION_ENV: "AWS_EU_WEST_2",
            PURCHASED_ENV: "false",
        }

        # Act
        open_head_run(JOB, app_env, STARTED_AT, lambda seconds: None)
        open_head_run(JOB, ENV, STARTED_AT, lambda seconds: None)

        # Assert
        app_body, cli_body = (call["json"] for call in arena.calls)
        assert app_body["channel"] == "native_app"
        assert app_body["snowflakeRegion"] == "AWS_EU_WEST_2"
        assert app_body["purchased"] is False
        assert set(app_body) == set(CONTRACT["open"]["request"]) | set(
            CONTRACT["open"]["optional_request"]
        )
        assert set(cli_body) == set(CONTRACT["open"]["request"])

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


class TestSnowflakeUsage:
    def test_run_usage_gets_the_contract_path_and_parses_the_hours(
        self, arena: Arena
    ) -> None:
        # Arrange
        arena.replies = [reply(200, CONTRACT["usage"]["examples"]["closed"])]
        headers = partner_headers("org-key", "user-1")

        # Act
        usage = snowflake_usage("run", RUN_ID, BASE_URL, headers, lambda s: None)

        # Assert
        assert usage == SnowflakeUsage(
            "closed", CONTRACT["usage"]["examples"]["closed"]["data"]["billableHours"]
        )
        call = arena.calls[0]
        assert call["method"] == CONTRACT["usage"]["method"]
        path = CONTRACT["usage"]["paths"]["run"].replace("{runId}", RUN_ID)
        assert call["url"] == BASE_URL + path
        assert set(call["headers"]) == set(CONTRACT["headers"])
        assert call["headers"]["Authorization"] == "Bearer org-key"

    def test_deployment_usage_gets_the_deployment_path(self, arena: Arena) -> None:
        arena.replies = [reply(200, CONTRACT["usage"]["examples"]["open"])]
        headers = partner_headers("org-key", "user-1")

        usage = snowflake_usage(
            "deployment", "dep-1", BASE_URL, headers, lambda s: None
        )

        assert usage.state == "open"
        path = CONTRACT["usage"]["paths"]["deployment"]
        assert arena.calls[0]["url"] == BASE_URL + path.replace(
            "{deploymentId}", "dep-1"
        )

    def test_server_errors_are_retried(self, arena: Arena) -> None:
        arena.replies = [reply(503), reply(200, CONTRACT["usage"]["examples"]["open"])]
        sleeps: list[float] = []

        usage = snowflake_usage("run", RUN_ID, BASE_URL, {}, sleeps.append)

        assert usage.state == "open"
        assert sleeps == [2.0]

    def test_404_is_an_unknown_item(self, arena: Arena) -> None:
        arena.replies = [reply(404)]

        with pytest.raises(UnknownUsageError, match=f"no run {RUN_ID}"):
            snowflake_usage("run", RUN_ID, BASE_URL, {}, lambda s: None)

    def test_rejected_credentials_raise(self, arena: Arena) -> None:
        arena.replies = [reply(401)]

        with pytest.raises(SnowflakeLicenseError, match=re.escape(AUTH_REJECTED)):
            snowflake_usage("run", RUN_ID, BASE_URL, {}, lambda s: None)

    def test_other_client_errors_name_the_status_without_a_retry(
        self, arena: Arena
    ) -> None:
        arena.replies = [reply(400)]

        with pytest.raises(SnowflakeLicenseError) as caught:
            snowflake_usage("run", RUN_ID, BASE_URL, {}, lambda s: None)

        assert str(caught.value) == f"{USAGE_FAILED} Arena answered 400."
        assert len(arena.calls) == 1

    def test_a_body_that_is_not_json_raises(self, arena: Arena) -> None:
        arena.replies = [httpx.Response(200, text="<html>placeholder</html>")]

        with pytest.raises(SnowflakeLicenseError) as caught:
            snowflake_usage("run", RUN_ID, BASE_URL, {}, lambda s: None)

        assert str(caught.value) == f"{USAGE_FAILED} The response is not JSON."

    @pytest.mark.parametrize(
        "body",
        [
            {"data": {"state": "open"}},
            {"data": {"state": "running", "billableHours": []}},
            {"data": {"state": "open", "billableHours": {}}},
        ],
    )
    def test_a_body_without_state_and_hours_raises(
        self, arena: Arena, body: dict[str, object]
    ) -> None:
        arena.replies = [reply(200, body)]

        with pytest.raises(SnowflakeLicenseError, match=re.escape(USAGE_FAILED)):
            snowflake_usage("run", RUN_ID, BASE_URL, {}, lambda s: None)

    def test_an_unknown_kind_raises_before_any_request(self, arena: Arena) -> None:
        with pytest.raises(ValueError, match="Unknown usage kind job"):
            snowflake_usage("job", RUN_ID, BASE_URL, {}, lambda s: None)

        assert arena.calls == []


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

    def test_retry_after_never_undercuts_the_backoff(self, arena: Arena) -> None:
        arena.replies = [
            reply(429, headers={"Retry-After": "0"}),
            reply(429, headers={"Retry-After": "1"}),
            reply(503, headers={"Retry-After": "1"}),
            reply(200, TOKEN_BODY),
        ]
        sleeps: list[float] = []

        heartbeat_snowflake_run(RUN_ID, ENV, sleeps.append)

        assert sleeps == [2.0, 4.0, 8.0]

    def test_retry_after_is_capped(self, arena: Arena) -> None:
        arena.replies = [
            reply(429, headers={"Retry-After": "600"}),
            reply(200, TOKEN_BODY),
        ]
        sleeps: list[float] = []

        heartbeat_snowflake_run(RUN_ID, ENV, sleeps.append)

        assert sleeps == [60.0]

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


class TestOpenHeadRunRefusals:
    @pytest.mark.parametrize("status", [400, 409, 422])
    def test_refusal_includes_arenas_description(
        self, arena: Arena, status: int
    ) -> None:
        arena.replies = [reply(status, {"error": "conflict", "description": INACTIVE})]

        with pytest.raises(SnowflakeLicenseError) as caught:
            open_head_run(JOB, ENV, STARTED_AT, lambda seconds: None)

        assert str(caught.value) == f"{OPEN_FAILED} Arena answered {status}: {INACTIVE}"

    def test_refusal_without_a_description_names_the_status(self, arena: Arena) -> None:
        arena.replies = [httpx.Response(409, text="Conflict")]

        with pytest.raises(SnowflakeLicenseError) as caught:
            open_head_run(JOB, ENV, STARTED_AT, lambda seconds: None)

        assert str(caught.value) == f"{OPEN_FAILED} Arena answered 409."


class TestOpenSnowflakeDeployment:
    def test_posts_the_contract_body_with_the_partner_headers(
        self, arena: Arena
    ) -> None:
        # Arrange
        arena.replies = [
            reply(201, {"deploymentId": DEPLOYMENT_ID, "billableHours": []})
        ]

        # Act
        open_snowflake_deployment(DEPLOYMENT, BASE_URL, HEADERS, lambda s: None)

        # Assert
        [call] = arena.calls
        assert call["method"] == "POST"
        assert call["url"] == f"{BASE_URL}/api/cli/v1/snowflake/deployments"
        assert call["json"] == {
            "deploymentId": DEPLOYMENT_ID,
            "snowflakeAccount": "AB12345",
            "snowflakeRegion": "AWS_EU_WEST_2",
            "gpuCount": 1,
            "gpuFamily": "GPU_NV_S",
            "startedAt": "2026-11-16T13:00:00+00:00",
            "channel": "native_app",
            "model": "qwen",
            "version": "checkpoint_2",
            "purchased": True,
            "runId": RUN_ID,
        }
        assert call["headers"]["Authorization"] == "Bearer org-key"
        assert call["headers"]["X-External-User-Id"] == "user-1"

    def test_leaves_out_run_id_when_there_is_none(self, arena: Arena) -> None:
        arena.replies = [reply(200)]
        deployment = SnowflakeDeployment(
            DEPLOYMENT_ID,
            "AB12345",
            "AWS_EU_WEST_2",
            2,
            "GPU_NV_M",
            OBSERVED_AT,
            "qwen",
            "v1",
            None,
            True,
        )

        open_snowflake_deployment(deployment, BASE_URL, HEADERS, lambda s: None)

        assert "runId" not in arena.calls[0]["json"]

    def test_server_errors_are_retried(self, arena: Arena) -> None:
        arena.replies = [reply(503), httpx.ConnectError("refused"), reply(201)]
        sleeps: list[float] = []

        open_snowflake_deployment(DEPLOYMENT, BASE_URL, HEADERS, sleeps.append)

        assert sleeps == [2.0, 4.0]

    @pytest.mark.parametrize("status", [400, 404, 409, 422])
    def test_refusal_includes_arenas_description_without_a_retry(
        self, arena: Arena, status: int
    ) -> None:
        arena.replies = [reply(status, {"error": "conflict", "description": INACTIVE})]

        with pytest.raises(DeploymentRefusedError) as caught:
            open_snowflake_deployment(DEPLOYMENT, BASE_URL, HEADERS, lambda s: None)

        assert str(caught.value) == (
            f"{DEPLOYMENT_OPEN_FAILED} Arena answered {status}: {INACTIVE}"
        )
        assert len(arena.calls) == 1

    @pytest.mark.parametrize(
        ("answer", "message", "refused"),
        [
            (flag_off, SNOWFLAKE_DENIED, True),
            (lambda: httpx.Response(403, text="Forbidden"), AUTH_REJECTED, True),
            (lambda: httpx.Response(401, text="Unauthorized"), AUTH_REJECTED, False),
        ],
    )
    def test_rejections_name_the_cause_and_only_403_is_a_refusal(
        self,
        arena: Arena,
        answer: Callable[[], httpx.Response],
        message: str,
        refused: bool,
    ) -> None:
        arena.replies = [answer()]

        with pytest.raises(SnowflakeLicenseError) as caught:
            open_snowflake_deployment(DEPLOYMENT, BASE_URL, HEADERS, lambda s: None)

        assert str(caught.value) == message
        assert isinstance(caught.value, DeploymentRefusedError) is refused

    def test_exhausted_retries_are_not_a_refusal(self, arena: Arena) -> None:
        arena.replies = [reply(503) for _ in range(20)]

        with pytest.raises(SnowflakeLicenseError) as caught:
            open_snowflake_deployment(DEPLOYMENT, BASE_URL, HEADERS, lambda s: None)

        assert not isinstance(caught.value, DeploymentRefusedError)

    def test_other_client_errors_are_not_a_refusal(self, arena: Arena) -> None:
        arena.replies = [reply(410, {"error": "gone", "description": INACTIVE})]

        with pytest.raises(SnowflakeLicenseError) as caught:
            open_snowflake_deployment(DEPLOYMENT, BASE_URL, HEADERS, lambda s: None)

        assert type(caught.value) is SnowflakeLicenseError
        assert str(caught.value) == (
            f"{DEPLOYMENT_OPEN_FAILED} Arena answered 410: {INACTIVE}"
        )
        assert len(arena.calls) == 1


class TestHeartbeatSnowflakeDeployment:
    def test_posts_observed_at_to_the_deployment_path(self, arena: Arena) -> None:
        arena.replies = [reply(200, {"billableHours": []})]

        heartbeat_snowflake_deployment(
            DEPLOYMENT_ID, OBSERVED_AT, BASE_URL, HEADERS, lambda s: None
        )

        [call] = arena.calls
        assert call["url"] == (
            f"{BASE_URL}/api/cli/v1/snowflake/deployments/{DEPLOYMENT_ID}/heartbeat"
        )
        assert call["json"] == {"observedAt": "2026-11-16T15:00:00+00:00"}

    @pytest.mark.parametrize(
        ("answer", "error", "message"),
        [
            (flag_off, TrainingDisabledError, re.escape(SNOWFLAKE_DENIED)),
            (
                lambda: reply(409, {"error": "conflict"}),
                DeploymentClosedError,
                "closed",
            ),
            (lambda: reply(404), UnknownUsageError, f"no deployment {DEPLOYMENT_ID}"),
            (
                lambda: httpx.Response(401, text="Unauthorized"),
                AuthRejectedError,
                re.escape(AUTH_REJECTED),
            ),
        ],
    )
    def test_refusals_raise_their_own_error(
        self,
        arena: Arena,
        answer: Callable[[], httpx.Response],
        error: type[Exception],
        message: str,
    ) -> None:
        arena.replies = [answer()]

        with pytest.raises(error, match=message) as caught:
            heartbeat_snowflake_deployment(
                DEPLOYMENT_ID, OBSERVED_AT, BASE_URL, HEADERS, lambda s: None
            )

        assert type(caught.value) is error
        assert len(arena.calls) == 1

    def test_exhausted_retries_name_the_heartbeat(self, arena: Arena) -> None:
        arena.replies = [reply(503), reply(503)]

        with pytest.raises(
            SnowflakeLicenseError, match=re.escape(DEPLOYMENT_HEARTBEAT_FAILED)
        ):
            heartbeat_snowflake_deployment(
                DEPLOYMENT_ID, OBSERVED_AT, BASE_URL, HEADERS, lambda s: None, 2.0
            )

        assert len(arena.calls) == 2

    def test_other_client_errors_name_the_heartbeat(self, arena: Arena) -> None:
        arena.replies = [reply(400, {"error": "bad_request", "description": INACTIVE})]

        with pytest.raises(SnowflakeLicenseError) as caught:
            heartbeat_snowflake_deployment(
                DEPLOYMENT_ID, OBSERVED_AT, BASE_URL, HEADERS, lambda s: None
            )

        assert type(caught.value) is SnowflakeLicenseError
        assert str(caught.value) == (
            f"{DEPLOYMENT_HEARTBEAT_FAILED} Arena answered 400: {INACTIVE}"
        )
        assert len(arena.calls) == 1


class TestCloseSnowflakeDeployment:
    def test_posts_ended_at_and_a_repeat_succeeds(self, arena: Arena) -> None:
        arena.replies = [reply(200), reply(200)]

        for _ in range(2):
            close_snowflake_deployment(
                DEPLOYMENT_ID, OBSERVED_AT, BASE_URL, HEADERS, lambda s: None
            )

        assert [call["url"] for call in arena.calls] == [
            f"{BASE_URL}/api/cli/v1/snowflake/deployments/{DEPLOYMENT_ID}/close"
        ] * 2
        assert arena.calls[0]["json"] == {"endedAt": "2026-11-16T15:00:00+00:00"}

    def test_404_is_an_unknown_deployment(self, arena: Arena) -> None:
        arena.replies = [reply(404)]

        with pytest.raises(UnknownUsageError, match=f"no deployment {DEPLOYMENT_ID}"):
            close_snowflake_deployment(
                DEPLOYMENT_ID, OBSERVED_AT, BASE_URL, HEADERS, lambda s: None
            )

    def test_other_refusals_name_the_close(self, arena: Arena) -> None:
        arena.replies = [
            reply(400, {"error": "bad_request", "description": "Bad time."})
        ]

        with pytest.raises(SnowflakeLicenseError) as caught:
            close_snowflake_deployment(
                DEPLOYMENT_ID, OBSERVED_AT, BASE_URL, HEADERS, lambda s: None
            )

        assert (
            str(caught.value)
            == f"{DEPLOYMENT_CLOSE_FAILED} Arena answered 400: Bad time."
        )


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
                reply(200, {"ok": False, "data": ENABLED_BODY["data"]}),
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
        ids=[
            "server-error",
            "html",
            "data-not-object",
            "not-ok",
            "other-schema",
            "dns",
        ],
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
        assert isinstance(caught.value, SnowflakeUpgradeRequiredError)

    def test_another_426_is_not_an_upgrade(self, arena: Arena) -> None:
        arena.replies = [reply(426, {"error": "other", "description": "Refused."})]

        with pytest.raises(SnowflakeLicenseError) as caught:
            fetch_capabilities(BASE_URL, {"Authorization": "Bearer org-key"})

        assert str(caught.value) == "Refused."
        assert not isinstance(caught.value, SnowflakeUpgradeRequiredError)

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
            f"agilerl 9.9.9 is available (you have {__version__}). "
            'Run: pip install -U "agilerl[snowflake]"\n'
        )


class TestSnowflakePricedFamilies:
    def test_returns_the_listed_families_in_upper_case(self) -> None:
        caps = {"features": {"snowflakePricedGpuFamilies": ["GPU_NV_S", "gpu_nv_m"]}}

        assert snowflake_priced_families(caps) == frozenset({"GPU_NV_S", "GPU_NV_M"})

    def test_an_empty_list_means_nothing_is_priced(self) -> None:
        caps = {"features": {"snowflakePricedGpuFamilies": []}}

        assert snowflake_priced_families(caps) == frozenset()

    @pytest.mark.parametrize(
        "caps",
        [
            {},
            {"features": "on"},
            {"features": {"snowflakeTraining": True}},
            {"features": {"snowflakePricedGpuFamilies": "GPU_NV_S"}},
        ],
    )
    def test_is_none_when_arena_does_not_list_them(
        self, caps: dict[str, object]
    ) -> None:
        assert snowflake_priced_families(caps) is None


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


ACCOUNT = ClaimAccount(
    region="AWS_EU_WEST_2",
    account="OZ12967",
    account_name="ARENA_CUSTOMER_TEST",
    organization_name="QURHFZE",
    application_name="AGILERL_ARENA",
    purchased=False,
)
CLAIM_BODY = {
    "ok": True,
    "data": {
        "claimId": "claim-1",
        "claimSecret": "claim-s3cr3t",
        "expiresAt": "2026-11-20T20:00:00+00:00",
    },
}
KEY_BODY = {
    "ok": True,
    "data": {
        "orgKey": "arena_org_issued-key",
        "externalUserId": "snowflake-app",
        "organizationName": "Acme ML (Snowflake)",
        "trialEndsAt": "2026-11-27T18:00:00+00:00",
    },
}


def refused(status: int, error: str) -> httpx.Response:
    return reply(status, {"error": error, "description": f"Arena says {error}."})


class TestStartSnowflakeClaim:
    def test_posts_the_install_identity_without_auth_and_returns_the_claim(
        self, arena: Arena
    ) -> None:
        # Arrange
        arena.replies = [reply(201, CLAIM_BODY)]

        # Act
        claim = start_snowflake_claim(ACCOUNT, BASE_URL, lambda s: None)

        # Assert
        assert claim == SnowflakeClaim(
            "claim-1",
            "claim-s3cr3t",
            datetime(2026, 11, 20, 20, 0, tzinfo=timezone.utc),
        )
        [call] = arena.calls
        assert call["url"] == f"{BASE_URL}/api/cli/v1/snowflake/claims"
        assert "Authorization" not in call["headers"]
        assert call["headers"]["X-Arena-Client-Version"] == __version__
        assert call["json"] == {
            "snowflakeRegion": "AWS_EU_WEST_2",
            "snowflakeAccount": "OZ12967",
            "accountName": "ARENA_CUSTOMER_TEST",
            "organizationName": "QURHFZE",
            "applicationName": "AGILERL_ARENA",
            "purchased": False,
            "appVersion": __version__,
        }

    def test_a_refusal_shows_arenas_description(self, arena: Arena) -> None:
        arena.replies = [refused(403, "snowflake_trial_ended")]

        with pytest.raises(
            SnowflakeLicenseError, match="snowflake_trial_ended"
        ) as caught:
            start_snowflake_claim(ACCOUNT, BASE_URL, lambda s: None)

        assert type(caught.value) is SnowflakeLicenseError

    def test_a_rate_limit_is_retried_for_ten_seconds_only(self, arena: Arena) -> None:
        arena.replies = [refused(429, "too_many_claims") for _ in range(5)]
        sleeps: list[float] = []

        with pytest.raises(SnowflakeLicenseError, match=re.escape(CLAIM_START_FAILED)):
            start_snowflake_claim(ACCOUNT, BASE_URL, sleeps.append)

        assert sleeps == [2.0, 4.0]

    def test_a_reply_without_the_claim_secret_is_refused(self, arena: Arena) -> None:
        body = {"ok": True, "data": {"claimId": "claim-1", "expiresAt": "x"}}
        arena.replies = [reply(201, body)]

        with pytest.raises(SnowflakeLicenseError, match="has no claimSecret"):
            start_snowflake_claim(ACCOUNT, BASE_URL, lambda s: None)

    def test_a_reply_with_an_unreadable_expiry_is_refused(self, arena: Arena) -> None:
        body = {
            "ok": True,
            "data": {"claimId": "claim-1", "claimSecret": "s", "expiresAt": "soon"},
        }
        arena.replies = [reply(201, body)]

        with pytest.raises(SnowflakeLicenseError, match="no RFC 3339 expiresAt"):
            start_snowflake_claim(ACCOUNT, BASE_URL, lambda s: None)


class TestProveSnowflakeClaim:
    def test_posts_the_secret_and_code_and_returns_the_signup_link(
        self, arena: Arena
    ) -> None:
        link = "https://arena.test/marketplace/signup#token=t"
        arena.replies = [reply(200, {"ok": True, "data": {"signupUrl": link}})]

        url = prove_snowflake_claim(
            "claim-1", "claim-s3cr3t", "code-7f3c", BASE_URL, lambda s: None
        )

        assert url == link
        [call] = arena.calls
        assert call["url"] == f"{BASE_URL}/api/cli/v1/snowflake/claims/claim-1/prove"
        assert call["json"] == {"claimSecret": "claim-s3cr3t", "code": "code-7f3c"}
        assert "claim-s3cr3t" not in call["url"]

    @pytest.mark.parametrize(
        ("answer", "error"),
        [
            (refused(409, "install_not_visible_yet"), ClaimWaitingError),
            (refused(403, "claim_rejected"), ClaimClosedError),
            (refused(410, "claim_expired"), ClaimClosedError),
            (refused(400, "bad_request"), SnowflakeLicenseError),
        ],
    )
    def test_refusals_raise_their_own_error(
        self, arena: Arena, answer: httpx.Response, error: type[Exception]
    ) -> None:
        arena.replies = [answer]

        with pytest.raises(error, match="Arena says") as caught:
            prove_snowflake_claim("claim-1", "s", "c", BASE_URL, lambda s: None)

        assert type(caught.value) is error
        assert "claim-s3cr3t" not in str(caught.value)


class TestClaimAnswersAsBuilt:
    def test_a_linked_claim_on_prove_raises_claim_linked(self, arena: Arena) -> None:
        arena.replies = [refused(409, "claim_linked")]

        with pytest.raises(ClaimLinkedError, match="claim_linked"):
            prove_snowflake_claim("claim-1", "s", "c", BASE_URL, lambda s: None)

    def test_an_unavailable_snowflake_is_retried_then_refused(
        self, arena: Arena
    ) -> None:
        arena.replies = [refused(502, "snowflake_unavailable") for _ in range(4)]
        sleeps: list[float] = []

        with pytest.raises(SnowflakeLicenseError, match=re.escape(CLAIM_START_FAILED)):
            start_snowflake_claim(ACCOUNT, BASE_URL, sleeps.append)

        assert sleeps == [2.0, 4.0]

    def test_a_plain_text_rate_limit_waits_for_retry_after(self, arena: Arena) -> None:
        limited = httpx.Response(
            429, text="Too Many Requests", headers={"Retry-After": "3"}
        )
        arena.replies = [limited, reply(201, CLAIM_BODY)]
        sleeps: list[float] = []

        claim = start_snowflake_claim(ACCOUNT, BASE_URL, sleeps.append)

        assert claim.claim_id == "claim-1"
        assert sleeps == [3.0]


class TestCollectSnowflakeClaimKey:
    def test_returns_the_key_arena_issued(self, arena: Arena) -> None:
        arena.replies = [reply(200, KEY_BODY)]

        key = collect_snowflake_claim_key(
            "claim-1", "claim-s3cr3t", BASE_URL, lambda s: None
        )

        assert key == AppKey(
            "arena_org_issued-key",
            "snowflake-app",
            "Acme ML (Snowflake)",
            "2026-11-27T18:00:00+00:00",
        )
        [call] = arena.calls
        assert call["url"] == f"{BASE_URL}/api/cli/v1/snowflake/claims/claim-1/key"
        assert call["json"] == {"claimSecret": "claim-s3cr3t"}

    def test_an_unlinked_claim_waits(self, arena: Arena) -> None:
        arena.replies = [refused(409, "claim_not_linked")]

        with pytest.raises(ClaimWaitingError, match="claim_not_linked"):
            collect_snowflake_claim_key("claim-1", "s", BASE_URL, lambda s: None)

    def test_a_collected_claim_is_closed(self, arena: Arena) -> None:
        arena.replies = [refused(410, "claim_collected")]

        with pytest.raises(ClaimClosedError, match="claim_collected"):
            collect_snowflake_claim_key("claim-1", "s", BASE_URL, lambda s: None)

    def test_a_key_without_trial_or_name_reads_as_none(self, arena: Arena) -> None:
        body = {"ok": True, "data": {"orgKey": "k", "externalUserId": "snowflake-app"}}
        arena.replies = [reply(200, body)]

        key = collect_snowflake_claim_key("claim-1", "s", BASE_URL, lambda s: None)

        assert key == AppKey("k", "snowflake-app", None, None)


class TestRotateSnowflakeAppKey:
    def test_posts_with_the_current_key_and_returns_the_new_one(
        self, arena: Arena
    ) -> None:
        arena.replies = [reply(200, KEY_BODY)]

        key = rotate_snowflake_app_key(BASE_URL, HEADERS, lambda s: None)

        assert key.org_key == "arena_org_issued-key"
        [call] = arena.calls
        assert call["url"] == f"{BASE_URL}/api/cli/v1/snowflake/app-keys/rotate"
        assert call["headers"]["Authorization"] == "Bearer org-key"
        assert call["json"] == {}

    def test_a_refused_key_raises_auth_rejected(self, arena: Arena) -> None:
        arena.replies = [httpx.Response(403, text="Forbidden")]

        with pytest.raises(AuthRejectedError, match=re.escape(AUTH_REJECTED)):
            rotate_snowflake_app_key(BASE_URL, HEADERS, lambda s: None)

    def test_a_refused_rotation_raises_rotate_failed(self, arena: Arena) -> None:
        arena.replies = [refused(400, "bad_request")]

        with pytest.raises(SnowflakeLicenseError, match=re.escape(ROTATE_FAILED)):
            rotate_snowflake_app_key(BASE_URL, HEADERS, lambda s: None)

    def test_a_reply_that_is_not_json_is_refused(self, arena: Arena) -> None:
        arena.replies = [httpx.Response(200, text="<html>")]

        with pytest.raises(SnowflakeLicenseError, match="not JSON") as caught:
            rotate_snowflake_app_key(BASE_URL, HEADERS, lambda s: None)

        assert str(caught.value).startswith(ROTATE_FAILED)

    def test_a_reply_without_data_is_refused(self, arena: Arena) -> None:
        arena.replies = [reply(200, {"ok": True})]

        with pytest.raises(SnowflakeLicenseError, match="has no data"):
            rotate_snowflake_app_key(BASE_URL, HEADERS, lambda s: None)


def trial_ended() -> httpx.Response:
    return refused(403, "snowflake_trial_ended")


class TestPlatformReviewAnswers:
    def test_a_start_arena_cannot_see_yet_waits_without_retrying(
        self, arena: Arena
    ) -> None:
        arena.replies = [refused(409, "install_not_visible_yet")]

        with pytest.raises(ClaimWaitingError, match="install_not_visible_yet"):
            start_snowflake_claim(ACCOUNT, BASE_URL, lambda s: None)

        assert len(arena.calls) == 1

    def test_a_start_for_an_install_that_lost_access_is_refused(
        self, arena: Arena
    ) -> None:
        arena.replies = [refused(403, "claim_rejected")]

        with pytest.raises(SnowflakeLicenseError, match="claim_rejected") as caught:
            start_snowflake_claim(ACCOUNT, BASE_URL, lambda s: None)

        assert type(caught.value) is SnowflakeLicenseError

    def test_too_many_claims_on_prove_waits_and_is_not_retried(
        self, arena: Arena
    ) -> None:
        arena.replies = [refused(429, "too_many_claims"), reply(200)]

        with pytest.raises(ClaimWaitingError, match="too_many_claims") as caught:
            prove_snowflake_claim("claim-1", "s", "c", BASE_URL, lambda s: None)

        assert type(caught.value) is ClaimWaitingError
        assert len(arena.calls) == 1

    def test_a_run_heartbeat_after_the_trial_stops_renewal(self, arena: Arena) -> None:
        arena.replies = [trial_ended()]

        with pytest.raises(RunRenewalRefusedError, match=re.escape(TRIAL_ENDED)):
            heartbeat_snowflake_run(RUN_ID, ENV, lambda seconds: None)

    def test_a_repeat_run_open_after_the_trial_names_the_trial(
        self, arena: Arena
    ) -> None:
        arena.replies = [trial_ended()]

        with pytest.raises(
            SnowflakeLicenseError, match=re.escape(TRIAL_ENDED)
        ) as caught:
            open_head_run(JOB, ENV, STARTED_AT, lambda seconds: None)

        assert type(caught.value) is SnowflakeLicenseError

    def test_a_deployment_heartbeat_after_the_trial_is_a_training_refusal(
        self, arena: Arena
    ) -> None:
        arena.replies = [trial_ended()]

        with pytest.raises(TrainingDisabledError, match=re.escape(TRIAL_ENDED)):
            heartbeat_snowflake_deployment(
                DEPLOYMENT_ID, OBSERVED_AT, BASE_URL, HEADERS, lambda s: None
            )

    def test_a_deployment_open_after_the_trial_is_refused(self, arena: Arena) -> None:
        arena.replies = [trial_ended()]

        with pytest.raises(DeploymentRefusedError, match=re.escape(TRIAL_ENDED)):
            open_snowflake_deployment(DEPLOYMENT, BASE_URL, HEADERS, lambda s: None)

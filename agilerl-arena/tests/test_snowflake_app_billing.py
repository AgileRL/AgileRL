# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Native App billing: metering deployments and reconciling usage from the app."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from snowflake.snowpark.exceptions import SnowparkSQLException
from snowflake_app.app_code.arena_app import (
    DeploymentCalls,
    meter_deployments,
    reconcile_usage,
)

from agilerl.arena.snowflake import license as license_mod
from agilerl.arena.snowflake.billing import reconcile
from agilerl.arena.snowflake.license import (
    SNOWFLAKE_DENIED,
    DeploymentClosedError,
    DeploymentRefusedError,
    SnowflakeDeployment,
    SnowflakeLicenseError,
    TrainingDisabledError,
    UnknownUsageError,
)
from tests.test_snowflake_billing import (
    HOUR_13_MS,
    HOUR_MS,
    NOW_MS,
    RUN_ID,
    Result,
    hour_at,
    unknown,
)
from tests.test_snowflake_billing import (
    BillingSession as UsageBillingSession,
)
from tests.test_snowflake_license import Arena, reply


class BillingSession(UsageBillingSession):
    """Refuses a service-role grant once that service is gone."""

    def sql(self, statement: str) -> Result:
        """Run ``statement``, refusing a grant for a service that is gone.

        :param statement: SQL to run.
        :return: The statement result.
        """
        if statement.startswith("GRANT SERVICE ROLE app_state."):
            service = statement.removeprefix("GRANT SERVICE ROLE app_state.").split(
                "!", 1
            )[0]
            if service not in self.services:
                msg = f"Service '{service}' does not exist or not authorized."
                raise SnowparkSQLException(msg)
        return super().sql(statement)


NOW = datetime(2026, 11, 16, 15, 0, tzinfo=timezone.utc)
SECRETS = {
    "org_key": "arena_org_key-1",
    "external_user_id": "user-1",
    "base_url": "https://arena.test",
}
DEPLOYMENTS = "https://arena.test/api/cli/v1/snowflake/deployments"


class DeploymentArena:
    """Records deployment opens, heartbeats and closes, raising the error scripted for an id."""

    def __init__(
        self,
        heartbeat_errors: dict[str, Exception] | None = None,
        open_error: Exception | None = None,
        close_errors: dict[str, Exception] | None = None,
    ) -> None:
        self.heartbeat_errors = heartbeat_errors or {}
        self.open_error = open_error
        self.close_errors = close_errors or {}
        self.opens: list[SnowflakeDeployment] = []
        self.heartbeats: list[tuple[str, datetime]] = []
        self.closes: list[tuple[str, datetime]] = []

    def calls(self) -> DeploymentCalls:
        return DeploymentCalls(self.open, self.heartbeat, self.close)

    def open(self, deployment: SnowflakeDeployment) -> None:
        self.opens.append(deployment)
        if self.open_error is not None:
            raise self.open_error

    def heartbeat(self, deployment_id: str, now: datetime) -> None:
        self.heartbeats.append((deployment_id, now))
        if deployment_id in self.heartbeat_errors:
            raise self.heartbeat_errors[deployment_id]

    def close(self, deployment_id: str, now: datetime) -> None:
        self.closes.append((deployment_id, now))
        if deployment_id in self.close_errors:
            raise self.close_errors[deployment_id]


def serving_session(services: dict[str, str]) -> BillingSession:
    """Session with an open deployment ``dep-<service>`` and a served model per service."""
    session = BillingSession()
    for n, service in enumerate(services):
        session.add_item(
            f"dep-{service.lower()}", HOUR_13_MS + n, "deployment", service_name=service
        )
        session.served.append(served_row(service))
    session.services = {
        name: status for name, status in services.items() if status != "GONE"
    }
    return session


def served_row(service: str) -> dict[str, object]:
    return {
        "service_name": service,
        "model": "qwen",
        "version": "checkpoint_2",
        "run_id": RUN_ID,
        "gpu_family": "GPU_NV_S",
        "gpu_count": 1,
    }


def item(session: BillingSession, usage_id: str) -> dict[str, object]:
    [row] = [row for row in session.items if row["usage_id"] == usage_id]
    return row


class TestBillingSession:
    def test_grant_of_a_missing_service_raises(self) -> None:
        # Arrange
        session = BillingSession()
        session.services = {"LIVE": "RUNNING"}
        statement = (
            "GRANT SERVICE ROLE app_state.GONE!ALL_ENDPOINTS_USAGE "
            "TO APPLICATION ROLE app_user"
        )

        # Act
        with pytest.raises(SnowparkSQLException, match="GONE"):
            session.sql(statement)

        # Assert
        assert session.grants == []
        assert session.services == {"LIVE": "RUNNING"}


class TestMeterDeployments:
    def test_heartbeats_running_services_and_closes_the_rest(self) -> None:
        # Arrange
        session = serving_session(
            {"A": "RUNNING", "B": "SUSPENDED", "C": "GONE", "D": "FAILED"}
        )
        session.add_item(RUN_ID, HOUR_13_MS)
        arena = DeploymentArena()

        # Act
        summary = meter_deployments(session, arena.calls(), NOW)

        # Assert
        assert summary == (
            "deployments: 4, heartbeat: 1, closed: 3, opened: 0, dropped: 0, "
            "missing: 0, failed: 0"
        )
        assert arena.heartbeats == [("dep-a", NOW)]
        assert arena.closes == [("dep-b", NOW), ("dep-c", NOW), ("dep-d", NOW)]
        assert item(session, "dep-c")["closed_at"] == NOW_MS
        assert item(session, "dep-a")["closed_at"] is None
        assert arena.opens == []
        assert not any(s.startswith("DROP") for s in session.statements)

    def test_suspend_closes_the_deployment_and_resume_opens_a_new_one(self) -> None:
        # Arrange
        session = serving_session({"A": "RUNNING"})
        arena = DeploymentArena()
        later = NOW.replace(hour=16)
        meter_deployments(session, arena.calls(), NOW)
        session.services["A"] = "SUSPENDED"

        # Act
        suspended = meter_deployments(session, arena.calls(), NOW)
        session.services["A"] = "RUNNING"
        resumed = meter_deployments(session, arena.calls(), later)

        # Assert
        assert suspended.startswith(
            "deployments: 1, heartbeat: 0, closed: 1, opened: 0"
        )
        assert resumed.startswith("deployments: 0, heartbeat: 0, closed: 0, opened: 1")
        assert arena.closes == [("dep-a", NOW)]
        assert item(session, "dep-a")["closed_at"] == NOW_MS
        assert [row["service_name"] for row in session.served] == ["A"]
        [deployment] = arena.opens
        assert (deployment.deployment_id, deployment.started_at) == (
            "A-20261116T160000Z",
            later,
        )
        assert item(session, "A-20261116T160000Z")["closed_at"] is None

    @pytest.mark.parametrize("status", ["SUSPENDED", "SUSPENDING"])
    def test_a_suspended_service_with_no_open_deployment_is_left(
        self, status: str
    ) -> None:
        # Arrange
        session = BillingSession()
        session.served.append(served_row("A"))
        session.services = {"A": status}
        arena = DeploymentArena()

        # Act
        summary = meter_deployments(session, arena.calls(), NOW)

        # Assert
        assert summary == (
            "deployments: 0, heartbeat: 0, closed: 0, opened: 0, dropped: 0, "
            "missing: 0, failed: 0"
        )
        assert (arena.opens, arena.heartbeats, arena.closes) == ([], [], [])
        assert session.items == []
        assert session.services == {"A": status}

    def test_closed_and_settled_items_are_not_metered(self) -> None:
        # Arrange
        session = serving_session({"A": "GONE"})
        item(session, "dep-a")["closed_at"] = NOW_MS - HOUR_MS
        settled = session.add_item(
            "dep-old", HOUR_13_MS, "deployment", service_name="A"
        )
        settled["settled_at"] = NOW_MS
        arena = DeploymentArena()

        # Act
        summary = meter_deployments(session, arena.calls(), NOW)

        # Assert
        assert summary.startswith("deployments: 0, heartbeat: 0, closed: 0")
        assert arena.closes == []

    def test_training_off_drops_the_service_and_closes_the_deployment(self) -> None:
        # Arrange
        session = serving_session({"A": "RUNNING", "B": "RUNNING"})
        arena = DeploymentArena({"dep-a": TrainingDisabledError(SNOWFLAKE_DENIED)})

        # Act
        summary = meter_deployments(session, arena.calls(), NOW)

        # Assert
        assert "heartbeat: 1, closed: 0, opened: 0, dropped: 1" in summary
        assert session.services == {"B": "RUNNING"}
        assert arena.closes == [("dep-a", NOW)]
        assert arena.opens == []
        dropped = item(session, "dep-a")
        assert (dropped["closed_at"], dropped["last_error"]) == (
            NOW_MS,
            SNOWFLAKE_DENIED,
        )
        assert dropped["last_error_at"] == NOW_MS
        assert session.grants == [
            (
                "GRANT SERVICE ROLE app_state.B!ALL_ENDPOINTS_USAGE "
                "TO APPLICATION ROLE app_user"
            )
        ]

    def test_training_off_leaves_the_row_open_when_the_close_fails(self) -> None:
        # Arrange
        session = serving_session({"A": "RUNNING", "B": "RUNNING"})
        arena = DeploymentArena(
            {"dep-a": TrainingDisabledError(SNOWFLAKE_DENIED)},
            close_errors={"dep-a": SnowflakeLicenseError("Arena is down")},
        )
        later = NOW.replace(hour=16)

        # Act
        failed = meter_deployments(session, arena.calls(), NOW)

        # Assert
        assert failed.endswith("missing: 0, failed: 1")
        assert "A" not in session.services
        assert item(session, "dep-a")["closed_at"] is None
        assert arena.heartbeats == [("dep-a", NOW)]
        assert arena.closes == [("dep-a", NOW)]

        # Act
        arena.close_errors.clear()
        retried = meter_deployments(session, arena.calls(), later)

        # Assert
        assert "closed: 1" in retried
        assert item(session, "dep-a")["closed_at"] == NOW_MS + HOUR_MS
        assert arena.closes == [("dep-a", NOW), ("dep-a", later)]

    def test_training_off_marks_closed_when_arena_has_no_deployment(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Arrange
        session = serving_session({"A": "RUNNING", "B": "RUNNING"})
        arena = DeploymentArena(
            {"dep-a": TrainingDisabledError(SNOWFLAKE_DENIED)},
            close_errors={"dep-a": UnknownUsageError("no deployment")},
        )

        # Act
        summary = meter_deployments(session, arena.calls(), NOW)

        # Assert
        assert "heartbeat: 1, closed: 0, opened: 0, dropped: 1" in summary
        dropped = item(session, "dep-a")
        assert (dropped["closed_at"], dropped["last_error"]) == (
            NOW_MS,
            SNOWFLAKE_DENIED,
        )
        assert "Arena has no deployment dep-a to close." in caplog.text
        assert session.grants == [
            (
                "GRANT SERVICE ROLE app_state.B!ALL_ENDPOINTS_USAGE "
                "TO APPLICATION ROLE app_user"
            )
        ]

    def test_a_deployment_arena_closed_is_reopened_with_the_service_kept(
        self,
    ) -> None:
        # Arrange
        session = serving_session({"A": "RUNNING"})
        arena = DeploymentArena({"dep-a": DeploymentClosedError("Arena closed it.")})

        # Act
        summary = meter_deployments(session, arena.calls(), NOW)

        # Assert
        assert "heartbeat: 0, closed: 0, opened: 1, dropped: 0" in summary
        assert session.services == {"A": "RUNNING"}
        old = item(session, "dep-a")
        assert (old["closed_at"], old["settled_at"]) == (NOW_MS, None)
        assert old["last_error"] == "Arena closed it."
        [deployment] = arena.opens
        assert deployment.deployment_id == "A-20261116T150000Z"
        assert (deployment.model, deployment.version, deployment.run_id) == (
            "qwen",
            "checkpoint_2",
            RUN_ID,
        )
        assert (deployment.gpu_family, deployment.gpu_count) == ("GPU_NV_S", 1)
        assert (deployment.snowflake_account, deployment.started_at) == ("AB12345", NOW)
        new = item(session, "A-20261116T150000Z")
        assert (new["kind"], new["service_name"], new["closed_at"]) == (
            "deployment",
            "A",
            None,
        )
        assert arena.closes == []

    def test_a_refused_reopen_drops_the_service_and_records_why(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Arrange
        session = serving_session({"A": "RUNNING"})
        refusal = "Could not record the Snowflake deployment with Arena. Arena answered 409: Inactive."
        arena = DeploymentArena(
            {"dep-a": DeploymentClosedError("Arena closed it.")},
            DeploymentRefusedError(refusal),
        )

        # Act
        summary = meter_deployments(session, arena.calls(), NOW)

        # Assert
        assert "opened: 0, dropped: 1" in summary
        assert session.services == {}
        new = item(session, "A-20261116T150000Z")
        assert (new["closed_at"], new["last_error"]) == (NOW_MS, refusal)
        assert new["last_error_at"] == NOW_MS
        assert f"Deployment A-20261116T150000Z: {refusal}" in caplog.text

    def test_a_live_service_with_no_deployment_gets_one(self) -> None:
        # Arrange
        session = BillingSession()
        session.served.append(served_row("ORPHAN"))
        session.services = {"ORPHAN": "RUNNING"}
        arena = DeploymentArena()

        # Act
        summary = meter_deployments(session, arena.calls(), NOW)

        # Assert
        assert summary == (
            "deployments: 0, heartbeat: 0, closed: 0, opened: 1, dropped: 0, "
            "missing: 0, failed: 0"
        )
        assert [d.deployment_id for d in arena.opens] == ["ORPHAN-20261116T150000Z"]
        assert item(session, "ORPHAN-20261116T150000Z")["service_name"] == "ORPHAN"

    def test_app_user_can_call_every_live_served_service(self) -> None:
        # Arrange
        session = serving_session({"A": "RUNNING", "B": "SUSPENDED"})
        session.served.append(served_row("ORPHAN"))
        session.services["ORPHAN"] = "PENDING"
        arena = DeploymentArena()

        # Act
        meter_deployments(session, arena.calls(), NOW)

        # Assert
        assert session.grants == [
            f"GRANT SERVICE ROLE app_state.{service}!ALL_ENDPOINTS_USAGE "
            "TO APPLICATION ROLE app_user"
            for service in ("A", "ORPHAN")
        ]

    def test_a_recorded_serve_whose_service_never_started_is_left_pending(
        self,
    ) -> None:
        # Arrange
        session = BillingSession()
        session.served.append(served_row("PENDING"))
        arena = DeploymentArena()

        # Act
        summary = meter_deployments(session, arena.calls(), NOW)
        billed = reconcile(session, unknown, NOW_MS)

        # Assert
        assert "opened: 0" in summary
        assert arena.opens == []
        assert session.items == []
        assert billed.startswith("items: 0,")

    def test_unknown_deployment_is_logged_and_left(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Arrange
        session = serving_session({"A": "RUNNING", "B": "RUNNING"})
        item(session, "dep-a")["first_seen_at"] = NOW_MS
        arena = DeploymentArena({"dep-a": UnknownUsageError("no deployment")})

        # Act
        summary = meter_deployments(session, arena.calls(), NOW)

        # Assert
        assert summary.endswith(
            "heartbeat: 1, closed: 0, opened: 0, dropped: 0, missing: 1, failed: 0"
        )
        assert session.services == {"A": "RUNNING", "B": "RUNNING"}
        assert item(session, "dep-a")["closed_at"] is None
        assert arena.opens == []
        assert "Arena has no deployment dep-a of service A. Leaving it." in caplog.text

    def test_a_never_seen_404_opens_the_live_service(self) -> None:
        # Arrange
        session = serving_session({"A": "RUNNING", "B": "RUNNING"})
        arena = DeploymentArena({"dep-a": UnknownUsageError("no deployment")})

        # Act
        summary = meter_deployments(session, arena.calls(), NOW)

        # Assert
        assert summary.endswith(
            "heartbeat: 1, closed: 0, opened: 1, dropped: 0, missing: 1, failed: 0"
        )
        assert [deployment.deployment_id for deployment in arena.opens] == [
            "A-20261116T150000Z"
        ]
        assert item(session, "dep-a")["closed_at"] is None
        assert item(session, "A-20261116T150000Z")["service_name"] == "A"

    def test_first_arena_failure_stops_metering_and_opening(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Arrange
        session = serving_session({"A": "RUNNING", "B": "RUNNING"})
        session.served.append(served_row("ORPHAN"))
        session.services["ORPHAN"] = "RUNNING"
        arena = DeploymentArena({"dep-a": SnowflakeLicenseError("Arena is down")})

        # Act
        summary = meter_deployments(session, arena.calls(), NOW)

        # Assert
        assert summary.endswith("missing: 0, failed: 1")
        assert [deployment for deployment, _ in arena.heartbeats] == ["dep-a"]
        assert arena.opens == []
        assert "meters no more deployments: Arena is down" in caplog.text

    def test_nothing_served_lists_no_services(self) -> None:
        session = BillingSession()
        session.add_item(RUN_ID, HOUR_13_MS)
        arena = DeploymentArena()

        summary = meter_deployments(session, arena.calls(), NOW)

        assert summary.startswith("deployments: 0, heartbeat: 0")
        assert not any(s.startswith("SHOW") for s in session.statements)


@pytest.fixture
def arena(monkeypatch: pytest.MonkeyPatch) -> Arena:
    """Arena behind the app's saved secrets."""
    monkeypatch.setitem(
        sys.modules,
        "_snowflake",
        SimpleNamespace(get_generic_secret_string=lambda name: SECRETS[name]),
    )
    fake = Arena([])
    monkeypatch.setattr(license_mod.httpx, "Client", fake)
    return fake


def open_usage(cents: int) -> httpx.Response:
    return reply(
        200,
        {"ok": True, "data": {"state": "open", "billableHours": [hour_at(0, cents)]}},
    )


class TestArenaAppLogging:
    def test_arena_records_reach_the_root_logger_snowflake_captures(self) -> None:
        # Arrange
        # The autouse conftest fixture rewires agilerl.arena, so import in a fresh interpreter.
        arena_root = Path(__file__).resolve().parents[1]
        script = (
            "import logging\n"
            "import snowflake_app.app_code.arena_app\n"
            "from agilerl.arena.snowflake import billing\n"
            "seen = []\n"
            "class Capture(logging.Handler):\n"
            "    def emit(self, record):\n"
            "        seen.append((record.name, record.levelname, record.getMessage()))\n"
            "logging.getLogger().addHandler(Capture())\n"
            "billing.logger.error('Could not bill run r')\n"
            "print(seen)\n"
        )

        # Act
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=arena_root,
            env={**os.environ, "PYTHONPATH": str(arena_root)},
            capture_output=True,
            text=True,
            check=True,
        )

        # Assert
        assert result.stdout.strip() == (
            "[('agilerl.arena.snowflake.billing', 'ERROR', 'Could not bill run r')]"
        )
        assert "Could not bill" not in result.stderr


class TestReconcileUsage:
    def test_heartbeats_the_service_then_bills_the_deployment_hours(
        self, arena: Arena
    ) -> None:
        # Arrange
        session = serving_session({"A": "RUNNING"})
        arena.replies = [reply(200, {"billableHours": []}), open_usage(957)]

        # Act
        summary = reconcile_usage(session)

        # Assert
        assert summary.startswith(
            "deployments: 1, heartbeat: 1, closed: 0, opened: 0, dropped: 0, "
            "missing: 0, failed: 0; items: 1, settled: 0, events: 1"
        )
        assert [call["url"] for call in arena.calls] == [
            f"{DEPLOYMENTS}/dep-a/heartbeat",
            f"{DEPLOYMENTS}/dep-a/usage",
        ]
        [event] = session.events
        assert (event["subclass"], event["base_charge"]) == ("deployment", 9.57)

    def test_a_non_listing_install_is_refused_and_logged_for_the_event_table(
        self, arena: Arena, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Arrange
        session = serving_session({"A": "RUNNING"})
        session.statuses = [
            SnowparkSQLException(
                "090898 (0A000): Application instance is not installed from listing"
            )
        ]
        arena.replies = [reply(200, {"billableHours": []}), open_usage(957)]

        # Act
        summary = reconcile_usage(session)

        # Assert
        assert "items: 1, settled: 0, events: 0, failed: 1" in summary
        assert "unconfirmed: 0" in summary
        assert session.billed == []
        assert "not installed from listing" in str(item(session, "dep-a")["last_error"])
        [record] = [
            r for r in caplog.records if r.name == "agilerl.arena.snowflake.billing"
        ]
        assert (record.levelname, record.getMessage()) == (
            "ERROR",
            "Could not bill deployment dep-a",
        )

    def test_a_reopened_deployment_still_bills_the_closed_ones_hours(
        self, arena: Arena
    ) -> None:
        # Arrange
        session = serving_session({"A": "RUNNING"})
        arena.replies = [
            reply(409, {"error": "conflict"}),
            reply(201, {"billableHours": []}),
            reply(
                200,
                {
                    "ok": True,
                    "data": {"state": "closed", "billableHours": [hour_at(0, 10)]},
                },
            ),
            reply(200, {"ok": True, "data": {"state": "open", "billableHours": []}}),
        ]

        # Act
        summary = reconcile_usage(session)

        # Assert
        assert "opened: 1" in summary
        assert "items: 2, settled: 1, events: 1" in summary
        assert arena.calls[1]["url"] == DEPLOYMENTS
        assert item(session, "dep-a")["settled_at"] is not None
        assert json.loads(session.events[0]["objects"]) == ["dep-a"]

    def test_a_failed_heartbeat_does_not_stop_billing(self, arena: Arena) -> None:
        # Arrange
        session = serving_session({"A": "RUNNING"})
        arena.replies = [httpx.Response(401, text="Unauthorized"), open_usage(10)]

        # Act
        summary = reconcile_usage(session)

        # Assert
        assert "failed: 1; items: 1, settled: 0, events: 1, failed: 0" in summary
        assert len(session.events) == 1

    def test_unconfigured_arena_runs_nothing(
        self, arena: Arena, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(SECRETS, "org_key", "")
        session = serving_session({"A": "RUNNING"})

        summary = reconcile_usage(session)

        assert summary.startswith("Arena is not configured.")
        assert session.statements == []
        assert arena.calls == []

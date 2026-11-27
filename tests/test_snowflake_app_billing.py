# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Native App billing: metering deployments and reconciling usage from the app."""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import app_trial
import httpx
import pytest
from app_billing import DeploymentCalls, meter_deployments
from snowflake.snowpark.exceptions import SnowparkSQLException

from agilerl.snowflake import license as license_mod
from agilerl.snowflake.billing import reconcile
from agilerl.snowflake.license import (
    SNOWFLAKE_DENIED,
    DeploymentClosedError,
    DeploymentRefusedError,
    SnowflakeDeployment,
    SnowflakeLicenseError,
    TrainingDisabledError,
    UnknownUsageError,
)
from agilerl.snowflake.publish import create_chat_function_sql
from snowflake_app.app_code import arena_app
from snowflake_app.app_code.arena_app import (
    IMAGE,
    publish,
    reconcile_usage,
    version_init,
)
from tests.test_snowflake_app_runs import RUN_ID as PUBLISH_RUN_ID
from tests.test_snowflake_app_runs import (
    AppSession,
    Files,
    Install,
    listing_trial,
    publish_session,
)
from tests.test_snowflake_app_runs import Result as AppResult
from tests.test_snowflake_billing import (
    HOUR_13_MS,
    HOUR_MS,
    NOW_MS,
    RUN_ID,
    Result,
    hour_at,
    trained,
    unknown,
)
from tests.test_snowflake_billing import (
    BillingSession as UsageBillingSession,
)
from tests.test_snowflake_license import Arena, reply
from tests.test_snowflake_setup import Row

APP_PREFIX = "AGILERL_APP_ARENA_APP_"
GPU_POOL = f"{APP_PREFIX}GPU_GPU_NV_S"

FORCED_DROP = re.compile(r"DROP SERVICE IF EXISTS app_state\.(\w+) FORCE")
PROVEN_CLAIMS = (
    "SELECT claim_id FROM app_state.arena_claim WHERE proven_at IS NOT NULL "
    "AND expires_at > CURRENT_TIMESTAMP()"
)
SETUP_JSON = "@app_state.runs/_arena/setup.json"


class BillingSession(UsageBillingSession):
    """Refuses a chat function for a gone service, answers the trial query, and keeps the task's and GPU pools' state."""

    def __init__(self) -> None:
        super().__init__()
        self.install = Install()
        self.task = "started"
        self.pools: dict[str, dict[str, object]] = {}
        self.proven_claims: list[Row] = []
        self.setup_doc: dict[str, object] | None = None
        self.file = Files({})

    def sql(self, statement: str) -> Result:
        """Run ``statement``, refusing a chat function for a service that is gone.

        :param statement: SQL to run.
        :return: The statement result.
        """
        if (ended := self.install.ended(statement)) is not None:
            self.statements.append(statement)
            return ended
        if statement == PROVEN_CLAIMS:
            self.statements.append(statement)
            return Result(self.proven_claims)
        if statement == f"LIST {SETUP_JSON}":
            self.statements.append(statement)
            if self.setup_doc is None:
                return Result([])
            self.file.staged[SETUP_JSON] = json.dumps(self.setup_doc)
            return Result([Row(name="runs/_arena/setup.json")])
        if "app_state.arena_claim" in statement or "app_state.arena_key" in statement:
            self.statements.append(statement)
            return Result([])
        if forced := FORCED_DROP.fullmatch(statement):
            self.statements.append(statement)
            self.services.pop(forced.group(1), None)
            return Result([])
        if statement == "DELETE FROM app_state.served_models":
            self.statements.append(statement)
            self.served.clear()
            return Result([])
        if statement == "SELECT app_code.object_prefix() AS PREFIX":
            self.statements.append(statement)
            return Result([Row(PREFIX=APP_PREFIX)])
        if statement == f"SHOW COMPUTE POOLS LIKE '{APP_PREFIX}GPU_%'":
            self.statements.append(statement)
            return Result([Row(name=name, **pool) for name, pool in self.pools.items()])
        if statement.startswith("ALTER COMPUTE POOL ") and statement.endswith(
            " SUSPEND"
        ):
            self.statements.append(statement)
            self.pools[statement.split()[3]]["state"] = "SUSPENDED"
            return Result([])
        if statement in {
            "ALTER TASK app_state.reconcile_usage_task SUSPEND",
            "ALTER TASK app_state.reconcile_usage_task RESUME",
        }:
            for prefix in list(self.hooks):
                if statement.startswith(prefix):
                    self.hooks.pop(prefix)()
            self.statements.append(statement)
            self.task = "suspended" if statement.endswith("SUSPEND") else "started"
            return Result([])
        if statement.startswith("CREATE FUNCTION IF NOT EXISTS app_state."):
            service = statement.split("AS $$SELECT app_state.", 1)[1].split("!", 1)[0]
            if service not in self.services:
                msg = f"Service '{service}' does not exist or not authorized."
                raise SnowparkSQLException(msg)
        return super().sql(statement)


class PublishBillingSession(BillingSession):
    """Billing session that answers publish's other statements from a publish session.

    ``during_wait`` runs once, at the first poll of the publish job, standing
    in for the hourly task.
    """

    def __init__(
        self,
        publish: AppSession,
        during_wait: Callable[[PublishBillingSession], None],
    ) -> None:
        super().__init__()
        self.publish = publish
        self.file = publish.file
        self.during_wait: Callable[[PublishBillingSession], None] | None = during_wait

    def sql(self, statement: str) -> Result | AppResult:
        """Run ``statement`` on the billing tables, or on the publish session.

        :param statement: SQL to run.
        :return: The statement result.
        """
        if self.during_wait is not None and statement.startswith(
            "SHOW SERVICE CONTAINERS IN SERVICE app_state.ARENA_PUBLISH_"
        ):
            during_wait, self.during_wait = self.during_wait, None
            during_wait(self)
        if statement.startswith("DELETE FROM app_state.served_models "):
            self.statements.append(statement)
            return Result([])
        if statement.startswith("INSERT INTO app_state.served_models "):
            self.statements.append(statement)
            service, model, version, run_id, family = re.findall(
                r"'([^']*)'", statement
            )
            self.served.append(
                {
                    "service_name": service,
                    "model": model,
                    "version": version,
                    "run_id": run_id,
                    "gpu_family": family,
                    "gpu_count": int(statement.rsplit(", ", 1)[1].rstrip(")")),
                }
            )
            return Result([])
        billing_prefixes = (
            "SHOW SERVICES IN SCHEMA app_state",
            "CREATE FUNCTION IF NOT EXISTS app_state.",
            "GRANT USAGE ON FUNCTION app_state.",
            "GRANT SERVICE ROLE app_state.",
            "DROP FUNCTION IF EXISTS app_state.",
            "DROP SERVICE IF EXISTS app_state.",
            "SELECT service_name, model, version, run_id, ",
            (
                "SELECT CURRENT_ACCOUNT() AS ACCOUNT, CURRENT_REGION() AS REGION, "
                "app_code.listing_purchased() AS PURCHASED"
            ),
            "SELECT SYSTEM$CREATE_BILLING_EVENTS(",
            "SELECT app_code.object_prefix() AS PREFIX",
            "SHOW COMPUTE POOLS LIKE ",
        )
        if any(
            table in statement
            for table in ("app_state.usage_items", "app_state.billed_hours")
        ) or statement.startswith(billing_prefixes):
            return super().sql(statement)
        return self.publish.sql(statement)


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


def app_user_grants(service: str) -> list[str]:
    """Grants that let ``app_user`` call a served service from SQL."""
    return [
        (
            f"GRANT USAGE ON FUNCTION app_state.{service}_CHAT(ARRAY, FLOAT, INT) "
            "TO APPLICATION ROLE app_user"
        ),
        (
            f"GRANT SERVICE ROLE app_state.{service}!INFERENCE_SERVICE_FUNCTION_USAGE "
            "TO APPLICATION ROLE app_user"
        ),
    ]


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
    def test_chat_function_of_a_missing_service_raises(self) -> None:
        # Arrange
        session = BillingSession()
        session.services = {"LIVE": "RUNNING"}
        statement = create_chat_function_sql("app_state.GONE")

        # Act
        with pytest.raises(SnowparkSQLException, match="GONE"):
            session.sql(statement)

        # Assert
        assert session.functions == []
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
        assert session.grants == app_user_grants("B")
        assert session.functions == ["app_state.B_CHAT"]

    def test_an_ended_trial_on_heartbeat_drops_the_service_through_arena(
        self, arena: Arena
    ) -> None:
        # Arrange
        session = serving_session({"A": "RUNNING"})
        ended = reply(403, {"error": "snowflake_trial_ended", "description": "ended"})
        arena.replies = [ended, reply(200)]
        calls = DeploymentCalls(
            lambda deployment: None,
            lambda deployment_id, now: license_mod.heartbeat_snowflake_deployment(
                deployment_id, now, "https://arena.test", {}, lambda s: None, 0.0
            ),
            lambda deployment_id, now: license_mod.close_snowflake_deployment(
                deployment_id, now, "https://arena.test", {}, lambda s: None, 0.0
            ),
        )

        # Act
        summary = meter_deployments(session, calls, NOW)

        # Assert
        assert "dropped: 1" in summary
        assert session.services == {}
        assert item(session, "dep-a")["last_error"] == license_mod.TRIAL_ENDED

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
        assert session.grants == app_user_grants("B")

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

    def test_app_user_can_call_every_live_or_suspended_served_service(self) -> None:
        # Arrange
        session = serving_session({"A": "RUNNING", "B": "SUSPENDED"})
        session.served.append(served_row("ORPHAN"))
        session.services["ORPHAN"] = "PENDING"
        arena = DeploymentArena()

        # Act
        meter_deployments(session, arena.calls(), NOW)

        # Assert
        assert session.functions == [
            "app_state.A_CHAT",
            "app_state.B_CHAT",
            "app_state.ORPHAN_CHAT",
        ]
        assert session.grants == (
            app_user_grants("A") + app_user_grants("B") + app_user_grants("ORPHAN")
        )
        assert [d.deployment_id for d in arena.opens] == ["ORPHAN-20261116T150000Z"]

    def test_a_recorded_serve_whose_service_never_started_is_left_pending(
        self,
    ) -> None:
        # Arrange
        session = BillingSession()
        session.served.append(served_row("PENDING"))
        arena = DeploymentArena()

        # Act
        summary = meter_deployments(session, arena.calls(), NOW)
        billed = reconcile(session, unknown, trained, NOW_MS)

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

    def test_a_never_seen_404_opens_the_same_deployment_again(self) -> None:
        # Arrange
        session = serving_session({"A": "RUNNING", "B": "RUNNING"})
        arena = DeploymentArena({"dep-a": UnknownUsageError("no deployment")})

        # Act
        summary = meter_deployments(session, arena.calls(), NOW)

        # Assert
        assert summary.endswith(
            "heartbeat: 1, closed: 0, opened: 1, dropped: 0, missing: 1, failed: 0"
        )
        [deployment] = arena.opens
        assert (deployment.deployment_id, deployment.started_at) == (
            "dep-a",
            datetime(2026, 11, 16, 13, 0, tzinfo=timezone.utc),
        )
        assert item(session, "dep-a")["closed_at"] is None
        assert [row["usage_id"] for row in session.items] == ["dep-a", "dep-b"]

    def test_a_refused_repeat_open_closes_the_deployment_now(self) -> None:
        # Arrange
        session = serving_session({"A": "RUNNING"})
        refusal = "Arena answered 409: Inactive."
        arena = DeploymentArena(
            {"dep-a": UnknownUsageError("no deployment")},
            DeploymentRefusedError(refusal),
        )

        # Act
        summary = meter_deployments(session, arena.calls(), NOW)

        # Assert
        assert "opened: 0, dropped: 1, missing: 1" in summary
        assert session.services == {}
        assert [d.deployment_id for d in arena.opens] == ["dep-a"]
        closed = item(session, "dep-a")
        assert (closed["closed_at"], closed["last_error"]) == (NOW_MS, refusal)
        assert len(session.items) == 1

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


def app_paths(arena_root: Path) -> list[str]:
    return [str(arena_root), str(arena_root / "snowflake_app" / "app_code")]


def open_usage(cents: int) -> httpx.Response:
    return reply(
        200,
        {"ok": True, "data": {"state": "open", "billableHours": [hour_at(0, cents)]}},
    )


class TestArenaAppLogging:
    def test_arena_records_reach_the_root_logger_snowflake_captures(self) -> None:
        # Arrange
        # The autouse conftest fixture rewires agilerl, so import in a fresh interpreter.
        arena_root = Path(__file__).resolve().parents[1]
        script = (
            "import logging\n"
            "import snowflake_app.app_code.arena_app\n"
            "from agilerl.snowflake import billing\n"
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
            env={**os.environ, "PYTHONPATH": os.pathsep.join(app_paths(arena_root))},
            capture_output=True,
            text=True,
            check=True,
        )

        # Assert
        assert result.stdout.strip() == (
            "[('agilerl.snowflake.billing', 'ERROR', 'Could not bill run r')]"
        )
        assert "Could not bill" not in result.stderr


class TestArenaAppErrors:
    def test_procedure_errors_reach_the_caller_as_the_message_only(self) -> None:
        # Arrange
        arena_root = Path(__file__).resolve().parents[1]
        script = (
            "import traceback\n"
            "from snowflake_app.app_code import arena_app\n"
            "try:\n"
            "    arena_app.register_reference(None, 'consumer_secret', 'ADD', 'ref-1')\n"
            "except ValueError:\n"
            "    print(repr(traceback.format_exc()))\n"
        )

        # Act
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=arena_root,
            env={**os.environ, "PYTHONPATH": os.pathsep.join(app_paths(arena_root))},
            capture_output=True,
            text=True,
            check=True,
        )

        # Assert
        assert result.stdout.strip() == repr(
            "ValueError: Unknown reference consumer_secret. "
            "Bind source_table, source_view, export_stage or reward_function.\n"
        )

    def test_errors_from_the_keys_module_hide_the_stack_too(self) -> None:
        # Arrange
        # Snowflake loads a handler's module alone, so app_keys must not rely on arena_app.
        arena_root = Path(__file__).resolve().parents[1]
        script = (
            "import traceback\n"
            "import app_keys\n"
            "try:\n"
            "    app_keys.arena_origin(None, 'not a url')\n"
            "except ValueError:\n"
            "    print(repr(traceback.format_exc()))\n"
        )

        # Act
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=arena_root,
            env={**os.environ, "PYTHONPATH": os.pathsep.join(app_paths(arena_root))},
            capture_output=True,
            text=True,
            check=True,
        )

        # Assert
        assert result.stdout.strip() == repr(
            "ValueError: The base URL does not name one host.\nNothing was saved.\n"
        )


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

    def test_a_successful_run_logs_its_summary_at_info(
        self, arena: Arena, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Arrange
        caplog.set_level(logging.INFO, logger=arena_app.logger.name)
        session = serving_session({"A": "RUNNING"})
        arena.replies = [reply(200, {"billableHours": []}), open_usage(957)]

        # Act
        summary = reconcile_usage(session)

        # Assert
        [record] = [r for r in caplog.records if r.name == arena_app.logger.name]
        assert (record.levelname, record.getMessage()) == (
            "INFO",
            f"reconcile_usage: {summary}",
        )
        assert "events: 1, failed: 0, missing: 0, skipped: 0" in summary

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
        [record] = [r for r in caplog.records if r.name == "agilerl.snowflake.billing"]
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

    def test_a_run_during_the_publish_wait_leaves_one_deployment_billed_once(
        self, arena: Arena
    ) -> None:
        # Arrange
        hourly: list[str] = []

        def hourly_run(session: PublishBillingSession) -> None:
            session.services["QWEN_CHECKPOINT_2"] = "PENDING"
            hourly.append(reconcile_usage(session))

        session = PublishBillingSession(publish_session(), hourly_run)
        arena.replies = [
            reply(201, {"billableHours": []}),
            open_usage(957),
            reply(201, {"billableHours": []}),
            reply(200, {"billableHours": []}),
            open_usage(957),
        ]

        # Act
        text = publish(session, PUBLISH_RUN_ID, "qwen", None, None, True, 1, None, None)
        hourly.append(reconcile_usage(session))

        # Assert
        [deployment] = [row for row in session.items if row["kind"] == "deployment"]
        deployment_id = str(deployment["usage_id"])
        assert text.endswith(f"deployment: {deployment_id}")
        opens = [call["json"] for call in arena.calls if call["url"] == DEPLOYMENTS]
        assert len(opens) == 2
        assert opens[0] == opens[1]
        assert opens[0]["deploymentId"] == deployment_id
        assert "opened: 1" in hourly[0]
        assert "events: 1" in hourly[0]
        assert "heartbeat: 1, closed: 0, opened: 0" in hourly[1]
        assert "events: 0" in hourly[1]
        [event] = session.events
        assert json.loads(event["objects"]) == [deployment_id]
        assert event["base_charge"] == 9.57
        assert session.services == {"QWEN_CHECKPOINT_2": "PENDING"}

    def test_a_failed_heartbeat_does_not_stop_billing(self, arena: Arena) -> None:
        # Arrange
        session = serving_session({"A": "RUNNING"})
        arena.replies = [httpx.Response(401, text="Unauthorized"), open_usage(10)]

        # Act
        summary = reconcile_usage(session)

        # Assert
        assert "failed: 1; items: 1, settled: 0, events: 1, failed: 0" in summary
        assert len(session.events) == 1

    def test_an_ended_trial_stops_every_service_and_closes_the_deployment(
        self, arena: Arena, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Arrange
        caplog.set_level(logging.INFO)
        session = serving_session({"A": "RUNNING"})
        session.services.update(
            {
                "ARENA_RUN_1_HEAD": "RUNNING",
                "ARENA_RUN_1_WORKERS": "PENDING",
                "ARENA_RUN_0_HEAD": "DONE",
            }
        )
        session.functions = ["app_state.A_CHAT"]
        session.install = listing_trial(6.98)
        arena.replies = [
            reply(200, {"ok": True}),
            reply(
                200,
                {
                    "ok": True,
                    "data": {"state": "closed", "billableHours": [hour_at(0, 10)]},
                },
            ),
        ]

        # Act
        summary = reconcile_usage(session)

        # Assert
        assert summary.startswith(
            "trial ended, dropped services: 3; deployments: 1, heartbeat: 0, "
            "closed: 1, opened: 0, dropped: 0, missing: 0, failed: 0; "
            "items: 1, settled: 1, events: 1"
        )
        assert session.services == {"ARENA_RUN_0_HEAD": "DONE"}
        assert session.served == []
        assert session.functions == []
        assert [call["url"] for call in arena.calls] == [
            f"{DEPLOYMENTS}/dep-a/close",
            f"{DEPLOYMENTS}/dep-a/usage",
        ]
        assert item(session, "dep-a")["closed_at"] is not None
        assert [
            r.getMessage()
            for r in caplog.records
            if r.name in {arena_app.logger.name, app_trial.logger.name}
            and r.levelname == "INFO"
        ][:3] == [
            "Trial ended unpurchased: dropped service A",
            "Trial ended unpurchased: dropped service ARENA_RUN_1_HEAD",
            "Trial ended unpurchased: dropped service ARENA_RUN_1_WORKERS",
        ]

    @pytest.mark.parametrize(
        ("age_days", "purchased"), [(6.9, False), (30, True)], ids=["trial", "bought"]
    )
    def test_services_keep_running_during_the_trial_or_once_bought(
        self, arena: Arena, age_days: float, purchased: bool
    ) -> None:
        # Arrange
        session = serving_session({"A": "RUNNING"})
        session.services["ARENA_RUN_1_HEAD"] = "RUNNING"
        session.install = Install(listing_trial(age_days).started_at, purchased)
        arena.replies = [reply(200, {"billableHours": []}), open_usage(957)]

        # Act
        summary = reconcile_usage(session)

        # Assert
        assert summary.startswith("deployments: 1, heartbeat: 1, closed: 0")
        assert session.services == {"A": "RUNNING", "ARENA_RUN_1_HEAD": "RUNNING"}
        assert not any(s.startswith("DROP") for s in session.statements)

    def test_unconfigured_arena_runs_nothing(
        self,
        arena: Arena,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.INFO, logger=arena_app.logger.name)
        monkeypatch.setitem(SECRETS, "org_key", "")
        session = serving_session({"A": "RUNNING"})

        summary = reconcile_usage(session)

        assert summary.startswith("Arena is not connected.")
        assert session.statements == [
            "DELETE FROM app_state.arena_claim WHERE expires_at <= CURRENT_TIMESTAMP()",
            (
                "SELECT claim_id, base_url, proven_at IS NOT NULL AS PROVEN "
                "FROM app_state.arena_claim"
            ),
            "SELECT app_code.object_prefix() AS PREFIX",
            f"SHOW COMPUTE POOLS LIKE '{APP_PREFIX}GPU_%'",
            "SHOW SERVICES IN SCHEMA app_state",
            "ALTER TASK app_state.reconcile_usage_task RESUME",
        ]
        assert arena.calls == []
        assert [
            (r.levelname, r.getMessage())
            for r in caplog.records
            if r.name == arena_app.logger.name
        ] == [("INFO", f"reconcile_usage: {summary}")]


def closed_usage(cents: int) -> httpx.Response:
    return reply(
        200,
        {"ok": True, "data": {"state": "closed", "billableHours": [hour_at(0, cents)]}},
    )


class TestReconcileUsageBillingTask:
    def test_a_finished_run_bills_its_last_hours_then_suspends_the_task(
        self, arena: Arena
    ) -> None:
        # Arrange
        session = BillingSession()
        session.add_item(RUN_ID, HOUR_13_MS)
        arena.replies = [closed_usage(25)]

        # Act
        summary = reconcile_usage(session)

        # Assert
        assert "items: 1, settled: 1, events: 1" in summary
        assert summary.endswith(
            "; nothing left to bill, suspended app_state.reconcile_usage_task"
        )
        assert session.task == "suspended"
        [event] = session.events
        assert (event["subclass"], event["base_charge"]) == ("run", 0.25)
        emit = next(
            i
            for i, s in enumerate(session.statements)
            if s.startswith("SELECT SYSTEM$")
        )
        assert emit < session.statements.index(
            "ALTER TASK app_state.reconcile_usage_task SUSPEND"
        )

    def test_a_run_that_failed_before_training_is_not_billed(
        self, arena: Arena
    ) -> None:
        # Arrange
        session = BillingSession()
        session.add_item(RUN_ID, HOUR_13_MS, trained=False)
        arena.replies = [closed_usage(252)]

        # Act
        summary = reconcile_usage(session)

        # Assert
        assert "items: 1, settled: 1, events: 0" in summary
        assert "untrained: 1" in summary
        assert session.events == []
        assert session.items[0]["unbilled_reason"] == (
            "The run ended before training started."
        )
        assert session.task == "suspended"

    def test_an_open_run_keeps_the_task_started(self, arena: Arena) -> None:
        session = BillingSession()
        session.add_item(RUN_ID, HOUR_13_MS)
        arena.replies = [
            reply(200, {"ok": True, "data": {"state": "open", "billableHours": []}})
        ]

        summary = reconcile_usage(session)

        assert "suspended" not in summary
        assert session.task == "started"
        assert "ALTER TASK app_state.reconcile_usage_task SUSPEND" not in (
            session.statements
        )

    def test_a_live_service_keeps_the_task_started_with_nothing_to_bill(
        self, arena: Arena
    ) -> None:
        session = BillingSession()
        session.services = {"ARENA_PUBLISH_1": "RUNNING"}

        summary = reconcile_usage(session)

        assert summary.endswith("unconfirmed: 0, waiting: 0, untrained: 0")
        assert session.task == "started"

    def test_a_run_arena_never_saw_does_not_keep_the_task_started(
        self, arena: Arena
    ) -> None:
        # Arrange
        session = BillingSession()
        session.add_item(RUN_ID, HOUR_13_MS)
        arena.replies = [reply(404)]

        # Act
        summary = reconcile_usage(session)

        # Assert
        assert "settled: 0, events: 0, failed: 0, missing: 1" in summary
        assert item(session, RUN_ID)["settled_at"] is None
        assert session.task == "suspended"

    def test_an_ended_trial_suspends_the_task_once_its_last_run_settles(
        self, arena: Arena
    ) -> None:
        # Arrange
        session = BillingSession()
        session.install = listing_trial(6.98)
        session.services = {"ARENA_RUN_1_HEAD": "RUNNING"}
        session.add_item(RUN_ID, HOUR_13_MS)
        arena.replies = [
            reply(200, {"ok": True, "data": {"state": "open", "billableHours": []}}),
            closed_usage(25),
        ]

        # Act
        first = reconcile_usage(session)
        task_after_sweep = session.task
        second = reconcile_usage(session)

        # Assert
        assert first.startswith("trial ended, dropped services: 1; ")
        assert session.services == {}
        assert task_after_sweep == "started"
        assert "items: 1, settled: 1, events: 1" in second
        assert second.endswith(
            "; nothing left to bill, suspended app_state.reconcile_usage_task"
        )
        assert session.task == "suspended"

    def test_a_run_submitted_while_suspending_resumes_the_task(
        self, arena: Arena
    ) -> None:
        # Arrange
        session = BillingSession()
        session.hooks["ALTER TASK app_state.reconcile_usage_task SUSPEND"] = lambda: (
            session.add_item("arena-20261116150000-def456", NOW_MS)
        )

        # Act
        summary = reconcile_usage(session)

        # Assert
        assert "suspended" not in summary
        assert session.task == "started"
        recheck = (
            "SELECT COUNT(*) AS N FROM app_state.usage_items WHERE settled_at IS NULL "
            "AND (first_seen_at IS NOT NULL OR first_404_at IS NULL)"
        )
        assert session.statements[-2:] == [
            recheck,
            "ALTER TASK app_state.reconcile_usage_task RESUME",
        ]


def gpu_pool(state: str, services: int = 0, jobs: int = 0) -> dict[str, object]:
    return {"state": state, "num_services": services, "num_jobs": jobs}


class TestReconcileUsageGpuPools:
    def test_suspends_a_gpu_pool_left_starting_with_nothing_on_it(
        self, arena: Arena
    ) -> None:
        # Arrange
        session = BillingSession()
        session.pools = {GPU_POOL: gpu_pool("STARTING")}

        # Act
        summary = reconcile_usage(session)

        # Assert
        assert session.pools[GPU_POOL]["state"] == "SUSPENDED"
        assert f"; suspended idle GPU pools {GPU_POOL};" in summary
        assert session.task == "suspended"

    @pytest.mark.parametrize(
        "pool",
        [
            gpu_pool("ACTIVE", services=1),
            gpu_pool("ACTIVE", jobs=1),
            gpu_pool("SUSPENDED"),
        ],
    )
    def test_leaves_a_pool_with_work_or_already_suspended(
        self, arena: Arena, pool: dict[str, object]
    ) -> None:
        session = BillingSession()
        session.pools = {GPU_POOL: dict(pool)}

        summary = reconcile_usage(session)

        assert session.pools[GPU_POOL] == pool
        assert "GPU pools" not in summary
        assert not any(s.startswith("ALTER COMPUTE POOL") for s in session.statements)

    def test_leaves_pools_of_another_app_that_match_the_like_pattern(
        self, arena: Arena
    ) -> None:
        session = BillingSession()
        other = "AGILERL_APP_ARENA_APPXGPU_GPU_NV_S"
        session.pools = {other: gpu_pool("IDLE")}

        reconcile_usage(session)

        assert session.pools[other]["state"] == "IDLE"


class TestVersionInit:
    def test_an_idle_app_leaves_the_task_suspended(self) -> None:
        session = BillingSession()
        session.services = {"ARENA_PUBLISH_1": "DONE"}

        text = version_init(session)

        assert text == "nothing to bill, suspended app_state.reconcile_usage_task"
        assert session.task == "suspended"

    def test_an_unsettled_deployment_resumes_the_task(self) -> None:
        session = BillingSession()
        session.add_item("dep-a", HOUR_13_MS, "deployment", service_name="A")
        session.task = "suspended"

        text = version_init(session)

        assert text == "resumed app_state.reconcile_usage_task"
        assert session.task == "started"

    def test_an_upgrade_points_the_saved_profile_at_this_versions_image(self) -> None:
        # Arrange
        session = BillingSession()
        session.setup_doc = saved_profile(
            "/arena_provider/images/repo/snowflake-job:39.5.0-py313-cu130"
        )

        # Act
        text = version_init(session)

        # Assert
        assert text == (
            f"image set to {IMAGE}; "
            "nothing to bill, suspended app_state.reconcile_usage_task"
        )
        [(path, uploaded)] = session.file.puts.items()
        assert path == SETUP_JSON
        assert json.loads(uploaded) == {**session.setup_doc, "image": IMAGE}

    def test_a_profile_already_on_this_image_is_left_alone(self) -> None:
        session = BillingSession()
        session.setup_doc = saved_profile(IMAGE)

        version_init(session)

        assert session.file.puts == {}

    def test_an_install_before_set_pools_writes_no_profile(self) -> None:
        session = BillingSession()

        version_init(session)

        assert session.file.puts == {}


def saved_profile(image: str) -> dict[str, object]:
    """``setup.json`` as ``set_pools`` writes it, on ``image``."""
    return {
        "image": image,
        "warehouse": "AGILERL_APP_ARENA_APP_WH",
        "stage": "app_state.runs",
        "metrics_table": "app_state.arena_metrics",
        "external_access_integrations": ["AGILERL_APP_ARENA_APP_HF_EAI"],
        "pools": {
            "broker": {
                "pool": "AGILERL_APP_ARENA_APP_BROKER_POOL",
                "family": "CPU_X64_L",
            },
            "head": {"pool": "AGILERL_APP_ARENA_APP_HEAD_POOL", "family": "CPU_X64_SL"},
            "gpu": {
                "GPU_NV_S": {
                    "pool": "AGILERL_APP_ARENA_APP_GPU_GPU_NV_S",
                    "max_gpus": 2,
                    "gpus_per_node": 1,
                    "gpu_memory_gib": 24,
                }
            },
        },
    }

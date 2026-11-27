# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Job service states and container logs for a Snowflake run."""

from __future__ import annotations

import pytest

from agilerl.snowflake.status import run_status, service_logs
from tests.test_snowflake_setup import Row

RUN_ID = "arena-20261109163457-abc123"
STEM = "ARENA_RUN_20261109163457_ABC123"
BROKER = f"{STEM}_BROKER"
HEAD = f"{STEM}_HEAD"
WORKERS = f"{STEM}_WORKERS"
STATUS_COMMAND = f"CALL arena_app.app_code.status('{RUN_ID}')"
READY = [Row(instance_id=0, container_name="main", status="READY")]


class RunSession:
    """Session that answers the status and log statements of one run."""

    def __init__(
        self,
        statuses: dict[str, str],
        containers: list[Row] | None = None,
        live: str = "",
        stored: list[str] | None = None,
    ) -> None:
        self.statuses = statuses
        self.containers = [] if containers is None else containers
        self.live = live
        self.stored = [] if stored is None else stored
        self.statements: list[str] = []

    def sql(self, text: str) -> RunSession:
        self.statements.append(text)
        return self

    def collect(self) -> list[Row]:
        text = self.statements[-1]
        if text.startswith("SHOW JOB SERVICES"):
            return [
                Row(name=name, status=status) for name, status in self.statuses.items()
            ]
        if text.startswith("SHOW SERVICE CONTAINERS"):
            return list(self.containers)
        if "SYSTEM$GET_SERVICE_LOGS" in text:
            return [Row(**{"SYSTEM$GET_SERVICE_LOGS": self.live})]
        if "SPCS_GET_LOGS" in text:
            return [Row(LOG=line) for line in reversed(self.stored)]
        return []


class TestRunStatus:
    def test_a_running_run_lists_every_role_and_container(self) -> None:
        # Arrange
        session = RunSession(
            {BROKER: "RUNNING", HEAD: "RUNNING", WORKERS: "PENDING"}, READY
        )

        # Act
        text = run_status(session, RUN_ID)

        # Assert
        assert text == (
            "broker\trunning\n  0\tmain\tREADY\n"
            "head\trunning\n  0\tmain\tREADY\n"
            "worker\trunning\n  0\tmain\tREADY\n"
            "state: running"
        )
        assert session.statements == [
            f"SHOW JOB SERVICES LIKE '{STEM}%'",
            f"SHOW SERVICE CONTAINERS IN SERVICE {BROKER}",
            f"SHOW SERVICE CONTAINERS IN SERVICE {HEAD}",
            f"SHOW SERVICE CONTAINERS IN SERVICE {WORKERS}",
        ]

    def test_a_finished_run_without_its_broker_succeeded(self) -> None:
        session = RunSession({HEAD: "DONE", WORKERS: "DONE"})

        text = run_status(session, RUN_ID)

        assert text == "broker\tgone\nhead\tdone\nworker\tdone\nstate: succeeded"
        assert f"SHOW SERVICE CONTAINERS IN SERVICE {BROKER}" not in session.statements

    def test_a_failed_worker_fails_the_run(self) -> None:
        session = RunSession({BROKER: "RUNNING", HEAD: "RUNNING", WORKERS: "FAILED"})

        assert run_status(session, RUN_ID).endswith("state: failed")


class TestServiceLogs:
    def test_a_running_service_reads_live_logs(self) -> None:
        session = RunSession({HEAD: "RUNNING"}, READY, live="trainer started")

        text = service_logs(session, RUN_ID, 0, status_command=STATUS_COMMAND)

        assert text == "trainer started"
        assert session.statements[-2:] == [
            f"SHOW SERVICE CONTAINERS IN SERVICE {HEAD}",
            f"SELECT SYSTEM$GET_SERVICE_LOGS('{HEAD}', 0, 'main')",
        ]

    def test_a_pending_container_names_the_status_command(self) -> None:
        pending = [Row(instance_id=0, container_name="main", status="PENDING")]
        session = RunSession({HEAD: "RUNNING", WORKERS: "PENDING"}, pending)

        text = service_logs(session, RUN_ID, 0, "worker", status_command=STATUS_COMMAND)

        assert text == (
            "The worker container has not started yet (state: PENDING). "
            "Its node can take several minutes to start; GPU nodes take longest. "
            f"Check progress with {STATUS_COMMAND}."
        )
        assert not any("SYSTEM$GET_SERVICE_LOGS" in item for item in session.statements)

    def test_an_instance_not_listed_yet_has_not_started(self) -> None:
        session = RunSession({WORKERS: "RUNNING"}, READY)

        text = service_logs(session, RUN_ID, 1, "worker", status_command=STATUS_COMMAND)

        assert text.startswith("The worker container has not started yet")

    def test_a_started_container_with_no_logs_says_so(self) -> None:
        session = RunSession({HEAD: "RUNNING"}, READY)

        text = service_logs(session, RUN_ID, 0, status_command=STATUS_COMMAND)

        assert text == (
            "The head container has started but has written no logs yet. "
            "Check again in a minute."
        )

    def test_a_finished_service_reads_the_event_table(self) -> None:
        session = RunSession(
            {HEAD: "DONE", WORKERS: "FAILED"}, stored=["step 1", "step 2", "OOM"]
        )

        text = service_logs(session, RUN_ID, 1, "worker", status_command=STATUS_COMMAND)

        assert text == "step 1\nstep 2\nOOM"
        assert f"TABLE({WORKERS}!SPCS_GET_LOGS(" in session.statements[-1]

    def test_a_dropped_service_is_reported(self) -> None:
        session = RunSession({HEAD: "DONE"})

        with pytest.raises(ValueError, match=f"{BROKER} was dropped"):
            service_logs(session, RUN_ID, 0, "broker", status_command=STATUS_COMMAND)

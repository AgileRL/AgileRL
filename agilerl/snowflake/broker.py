# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Run Pulsar standalone for one Snowflake training job."""

from __future__ import annotations

import json
import os
import signal
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from agilerl.broker import (
    CGROUP_MEMORY_MAX,
    DEFAULT_PULSAR_HOME,
    HEALTH_TIMEOUT_S,
    POLL_S,
    CommandResult,
    PulsarProcess,
    broker_memory,
    cgroup_memory_bytes,
    ensure_tenant_and_namespace,
    http_status,
    node_ip,
    pulsar_home_from,
    run_command,
    start_process,
    start_standalone,
    stop_pulsar,
    wait_until_healthy,
    write_broker_config,
)
from agilerl.snowflake.job import (
    RUN_MOUNT,
    cpu_memory_gib,
    lowered,
    runs_prefix,
    snowflake_identifier,
)
from agilerl.snowflake.node import (
    STARTUP_TIMEOUT_S,
    exit_on_sigterm,
    read_done,
    write_done,
)
from agilerl.snowflake.session import open_session
from agilerl.snowflake.source import SnowflakeRow, SnowflakeSession

HEAD_GONE_S = 60.0
RUN_POLL_S = 20.0
# Reopening re-reads /snowflake/session/token, which Snowflake keeps fresh.
SESSION_MAX_AGE_S = 1800.0


@dataclass
class BrokerRuntime:
    """Process, HTTP, SQL, clock, and filesystem calls ``run_broker`` uses."""

    environ: Mapping[str, str]
    node_ip: Callable[[], str]
    open_session: Callable[[], SnowflakeSession]
    start_process: Callable[[list[str], Mapping[str, str]], PulsarProcess]
    run_command: Callable[[list[str]], CommandResult]
    http_status: Callable[[str], int]
    sleep: Callable[[float], None]
    monotonic: Callable[[], float]
    run_dir: Path = field(default_factory=lambda: Path(RUN_MOUNT))
    pulsar_home: Path = field(default_factory=lambda: DEFAULT_PULSAR_HOME)
    memory_limit_path: Path = field(default_factory=lambda: CGROUP_MEMORY_MAX)
    health_timeout_s: float = HEALTH_TIMEOUT_S
    head_start_timeout_s: float = STARTUP_TIMEOUT_S
    head_gone_s: float = HEAD_GONE_S
    poll_s: float = POLL_S
    run_poll_s: float = RUN_POLL_S
    session_max_age_s: float = SESSION_MAX_AGE_S


@dataclass
class RotatingSession:
    """One session for the whole run, reopened once it is ``max_age_s`` old."""

    open_session: Callable[[], SnowflakeSession]
    monotonic: Callable[[], float]
    max_age_s: float
    session: SnowflakeSession | None = None
    opened_at: float = 0.0

    def current(self) -> SnowflakeSession:
        """The open session, reopened when it is too old.

        :return: A Snowpark session.
        """
        now = self.monotonic()
        if self.session is None or now - self.opened_at >= self.max_age_s:
            self.close()
            self.session = self.open_session()
            self.opened_at = now
        return self.session

    def close(self) -> None:
        """Close the open session, if any."""
        if self.session is not None:
            self.session.close()
            self.session = None


def main() -> int:
    """Run the broker supervisor from ``/arena/run/job.json``.

    :return: The process exit code.
    """
    signal.signal(signal.SIGTERM, exit_on_sigterm)
    return run_broker()


def memory_limit_bytes(path: Path, family: str) -> int:
    """Cgroup memory limit in bytes, or the broker family size when unset.

    :param path: ``memory.max``.
    :param family: Broker instance family from ``job.json``. Used when the
        file is missing or not a byte count.
    :return: The limit in bytes.
    :raises ValueError: The file is not a byte count and ``family`` is not a
        known CPU family.
    """
    limit = cgroup_memory_bytes(path)
    if limit is not None:
        return limit
    # CPU pools omit memory.max, or write the literal max.
    return cpu_memory_gib(family) * 1024 * 1024 * 1024


def run_broker(runtime: BrokerRuntime | None = None) -> int:
    """Start Pulsar, create ``arena/run``, and exit when the head run finishes.

    :param runtime: Process, HTTP, and SQL calls. ``None`` uses this machine.
    :return: The code in ``.done``, or 1 when the head or Pulsar dies without
        it, or the head never runs before the deadline. That last case also
        writes ``.done`` with 1.
    :raises TimeoutError: The health endpoint never returns 200.
    :raises KeyError: ``job.json`` lacks ``runs_stage``, ``run_id``,
        ``head_service``, or ``broker_family``.
    :raises ValueError: ``run_id`` or the head service name is not a valid
        identifier, or ``memory.max`` is not a byte count and ``broker_family``
        is not a known CPU family.
    :raises RuntimeError: ``pulsar-admin`` failed for a reason other than
        the tenant or namespace already existing.
    """
    current = production_runtime() if runtime is None else runtime
    job = json.loads((current.run_dir / "job.json").read_text(encoding="utf-8"))
    run_prefix = runs_prefix(str(job["runs_stage"]), str(job["run_id"]))
    head_service = str(job["head_service"])
    limit = memory_limit_bytes(current.memory_limit_path, str(job["broker_family"]))
    memory = broker_memory(limit)
    write_broker_config(current.pulsar_home, current.node_ip(), memory)
    process = start_standalone(current)
    try:
        wait_until_healthy(current)
        ensure_tenant_and_namespace(current)
        return _poll_run(current, process, run_prefix, head_service)
    finally:
        stop_pulsar(process)


def service_status_terminal(rows: list[SnowflakeRow]) -> bool:
    """Whether service status text is DONE, FAILED, or CANCELLED.

    :param rows: ``DESCRIBE SERVICE`` rows, or one ``SYSTEM$GET_SERVICE_STATUS`` row.
    :return: True when every reported status is terminal, or when there are no rows.
    """
    if not rows:
        return True
    terminal = frozenset({"DONE", "FAILED", "CANCELLED"})
    statuses = [token for row in rows for token in _row_statuses(lowered(row))]
    if not statuses:
        return False
    return all(token in terminal for token in statuses)


def service_status_running(rows: list[SnowflakeRow]) -> bool:
    """Whether any reported status is RUNNING.

    :param rows: ``DESCRIBE SERVICE`` rows, or one ``SYSTEM$GET_SERVICE_STATUS`` row.
    :return: True once an instance runs.
    """
    return any(
        token == "RUNNING" for row in rows for token in _row_statuses(lowered(row))
    )


def head_service_rows(session: SnowflakeSession, service: str) -> list[SnowflakeRow]:
    """``DESCRIBE SERVICE`` rows for the head job service.

    :param session: Snowpark session.
    :param service: Head service name.
    :return: The rows, or none once the service is dropped.
    """
    name = snowflake_identifier(service, "service")
    try:
        return session.sql(f"DESCRIBE SERVICE {name}").collect()
    except Exception as exc:
        # A dropped service raises, and it will not write .done.
        if "does not exist" in str(exc).lower():
            return []
        raise


def head_service_terminal(session: SnowflakeSession, service: str) -> bool:
    """Whether the head job service is finished or gone.

    :param session: Snowpark session.
    :param service: Head service name.
    :return: True when the service will not write ``.done``.
    """
    return service_status_terminal(head_service_rows(session, service))


def production_runtime() -> BrokerRuntime:
    """Runtime that talks to this process, Pulsar, and Snowflake.

    :return: The default runtime.
    """
    return BrokerRuntime(
        environ=os.environ,
        node_ip=node_ip,
        open_session=lambda: open_session(None),
        start_process=start_process,
        run_command=run_command,
        http_status=http_status,
        sleep=time.sleep,
        monotonic=time.monotonic,
        pulsar_home=pulsar_home_from(os.environ),
        memory_limit_path=CGROUP_MEMORY_MAX,
    )


def _poll_run(
    runtime: BrokerRuntime,
    process: PulsarProcess,
    run_prefix: str,
    head_service: str,
) -> int:
    sessions = RotatingSession(
        runtime.open_session, runtime.monotonic, runtime.session_max_age_s
    )
    try:
        return _watch_head(runtime, sessions, process, run_prefix, head_service)
    finally:
        sessions.close()


def _watch_head(
    runtime: BrokerRuntime,
    sessions: RotatingSession,
    process: PulsarProcess,
    run_prefix: str,
    head_service: str,
) -> int:
    start_deadline = runtime.monotonic() + runtime.head_start_timeout_s
    head_started = False
    terminal_since: float | None = None
    while True:
        session = sessions.current()
        code = read_done(session, run_prefix)
        if code is not None:
            return code
        if process.poll() is not None:
            return 1
        rows = head_service_rows(session, head_service)
        now = runtime.monotonic()
        head_started = head_started or service_status_running(rows)
        if service_status_terminal(rows):
            if terminal_since is None:
                terminal_since = now
            elif now - terminal_since >= runtime.head_gone_s:
                return 1
        else:
            terminal_since = None
            if not head_started and now >= start_deadline:
                # Workers stop waiting once .done is on the stage.
                write_done(session, run_prefix, 1)
                return 1
        runtime.sleep(runtime.run_poll_s)


def _row_statuses(values: dict[str, object]) -> list[str]:
    if "status" in values:
        return _statuses_in(values["status"])
    found: list[str] = []
    for value in values.values():
        if isinstance(value, str) and value.lstrip().startswith("["):
            found.extend(_statuses_in(value))
    return found


def _statuses_in(value: object) -> list[str]:
    text = str(value).strip()
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return []
        return [
            str(item["status"]).strip().upper()
            for item in parsed
            if isinstance(item, dict) and isinstance(item.get("status"), str)
        ]
    if text == "":
        return []
    return [text.upper()]


if __name__ == "__main__":
    raise SystemExit(main())

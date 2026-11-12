# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Run Pulsar standalone for one Snowflake training job."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from agilerl.arena.snowflake.job import (
    RUN_MOUNT,
    lowered,
    runs_prefix,
    snowflake_identifier,
)
from agilerl.arena.snowflake.node import (
    STARTUP_TIMEOUT_S,
    exit_on_sigterm,
    read_done,
    write_done,
)
from agilerl.arena.snowflake.session import open_session
from agilerl.arena.snowflake.source import SnowflakeRow, SnowflakeSession

HEALTH_TIMEOUT_S = 600.0
HEAD_GONE_S = 60.0
POLL_S = 2.0
RUN_POLL_S = 20.0
# Reopening re-reads /snowflake/session/token, which Snowflake keeps fresh.
SESSION_MAX_AGE_S = 1800.0
DEFAULT_PULSAR_HOME = Path("/opt/pulsar")
CGROUP_MEMORY_MAX = Path("/sys/fs/cgroup/memory.max")
JOURNAL_DIR = Path("/pulsar-data/journal")
LEDGER_DIR = Path("/pulsar-data/ledgers")


class PulsarProcess(Protocol):
    """The standalone process while it runs."""

    def poll(self) -> int | None:
        """Exit code, or ``None`` while the process is running."""
        ...

    def terminate(self) -> None:
        """Send SIGTERM."""
        ...

    def wait(self) -> int:
        """Block until the process exits."""
        ...


@dataclass(frozen=True)
class CommandResult:
    """Exit code and combined output of one finished command."""

    code: int
    output: str


@dataclass(frozen=True)
class BrokerMemory:
    """JVM heap, direct memory, and ledger caches, in MiB."""

    heap_mib: int
    direct_mib: int
    ledger_cache_mib: int
    db_write_cache_mib: int
    db_read_cache_mib: int


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


def run_broker(runtime: BrokerRuntime | None = None) -> int:
    """Start Pulsar, create ``arena/run``, and exit when the head run finishes.

    :param runtime: Process, HTTP, and SQL calls. ``None`` uses this machine.
    :return: The code in ``.done``, or 1 when the head or Pulsar dies without
        it, or the head never runs before the deadline. That last case also
        writes ``.done`` with 1.
    :raises TimeoutError: The health endpoint never returns 200.
    :raises KeyError: ``job.json`` lacks ``runs_stage``, ``run_id``, or
        ``head_service``.
    :raises ValueError: ``run_id`` or the head service name is not a valid identifier.
    :raises RuntimeError: ``pulsar-admin`` failed for a reason other than
        the tenant or namespace already existing.
    """
    current = production_runtime() if runtime is None else runtime
    job = json.loads((current.run_dir / "job.json").read_text(encoding="utf-8"))
    run_prefix = runs_prefix(str(job["runs_stage"]), str(job["run_id"]))
    head_service = str(job["head_service"])
    limit = int(current.memory_limit_path.read_text(encoding="utf-8").strip())
    memory = broker_memory(limit)
    write_broker_config(current.pulsar_home, current.node_ip(), memory)
    env = dict(current.environ)
    env["PULSAR_HOME"] = str(current.pulsar_home)
    command = [
        str(current.pulsar_home / "bin" / "pulsar"),
        "standalone",
        "--config",
        str(current.pulsar_home / "conf" / "standalone.conf"),
    ]
    process = current.start_process(command, env)
    try:
        _wait_until_healthy(current)
        _ensure_tenant_and_namespace(current)
        return _poll_run(current, process, run_prefix, head_service)
    finally:
        _stop_pulsar(process)


def broker_memory(limit_bytes: int) -> BrokerMemory:
    """Heap, direct memory, and cache sizes for a cgroup memory limit.

    :param limit_bytes: ``memory.max`` in bytes.
    :return: Sizes in MiB, each at least 256.
    """
    # CPU_X64_L is 116 GiB: heap 24, direct 48, ledger cache 8, each db cache 4.
    memory_parts = 116
    mib = 1024 * 1024
    minimum_mib = 256

    def portion(parts: int) -> int:
        share = limit_bytes * parts // (memory_parts * mib)
        return max(share, minimum_mib)

    db_cache_mib = portion(4)
    return BrokerMemory(
        heap_mib=portion(24),
        direct_mib=portion(48),
        ledger_cache_mib=portion(8),
        db_write_cache_mib=db_cache_mib,
        db_read_cache_mib=db_cache_mib,
    )


def write_broker_config(
    pulsar_home: Path, advertised_address: str, memory: BrokerMemory
) -> None:
    """Write broker, bookie, and JVM settings under the Pulsar install.

    :param pulsar_home: Pulsar install root.
    :param advertised_address: This container's IP.
    :param memory: Heap, direct memory, and cache sizes.
    """
    conf_dir = pulsar_home / "conf"
    conf_dir.mkdir(parents=True, exist_ok=True)
    JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    _write_merged(
        conf_dir / "standalone.conf", _broker_properties(advertised_address, memory)
    )
    _write_merged(conf_dir / "bookkeeper.conf", _bookie_properties(memory))
    _write_merged(conf_dir / "pulsar_env.sh", {"PULSAR_MEM": _pulsar_mem(memory)})


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
    home = os.environ.get("PULSAR_HOME", "").strip()
    return BrokerRuntime(
        environ=os.environ,
        node_ip=lambda: socket.gethostbyname(socket.gethostname()),
        open_session=lambda: open_session(None),
        start_process=_start_process,
        run_command=_run_command,
        http_status=_http_status,
        sleep=time.sleep,
        monotonic=time.monotonic,
        pulsar_home=Path(home) if home else DEFAULT_PULSAR_HOME,
        memory_limit_path=CGROUP_MEMORY_MAX,
    )


def _broker_properties(address: str, memory: BrokerMemory) -> dict[str, str]:
    return {
        "advertisedAddress": address,
        "managedLedgerDefaultEnsembleSize": "1",
        "managedLedgerDefaultWriteQuorum": "1",
        "managedLedgerDefaultAckQuorum": "1",
        "maxMessageSize": "5242880",
        "managedLedgerCacheSizeMB": str(memory.ledger_cache_mib),
        "allowAutoTopicCreation": "true",
        "brokerDeleteInactiveTopicsEnabled": "false",
        "ttlDurationDefaultInSeconds": "0",
        "backlogQuotaDefaultLimitBytes": "-1",
        "defaultRetentionTimeInMinutes": "0",
        "defaultRetentionSizeInMB": "0",
    }


def _bookie_properties(memory: BrokerMemory) -> dict[str, str]:
    return {
        "journalDirectories": JOURNAL_DIR.as_posix(),
        "ledgerDirectories": LEDGER_DIR.as_posix(),
        "journalSyncData": "false",
        "dbStorage_writeCacheMaxSizeMb": str(memory.db_write_cache_mib),
        "dbStorage_readAheadCacheMaxSizeMb": str(memory.db_read_cache_mib),
    }


def _write_merged(path: Path, values: dict[str, str]) -> None:
    existing = path.read_text(encoding="utf-8") if path.is_file() else ""
    path.write_text(_merge_assignments(existing, values), encoding="utf-8")


def _pulsar_mem(memory: BrokerMemory) -> str:
    return (
        f'"-Xms{memory.heap_mib}m -Xmx{memory.heap_mib}m '
        f'-XX:MaxDirectMemorySize={memory.direct_mib}m"'
    )


def _merge_assignments(text: str, values: dict[str, str]) -> str:
    lines = text.splitlines()
    seen: set[str] = set()
    merged: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#") or "=" not in stripped:
            merged.append(line)
            continue
        key = stripped.split("=", 1)[0].strip()
        if key in values:
            merged.append(f"{key}={values[key]}")
            seen.add(key)
        else:
            merged.append(line)
    for key, value in values.items():
        if key not in seen:
            merged.append(f"{key}={value}")
    return "\n".join(merged) + "\n"


def _wait_until_healthy(runtime: BrokerRuntime) -> None:
    url = "http://127.0.0.1:8080/admin/v2/brokers/health"
    deadline = runtime.monotonic() + runtime.health_timeout_s
    while runtime.http_status(url) != 200:
        if runtime.monotonic() >= deadline:
            msg = (
                f"Broker health at {url} did not return 200 "
                f"after {runtime.health_timeout_s:.0f}s"
            )
            raise TimeoutError(msg)
        runtime.sleep(runtime.poll_s)


def _ensure_tenant_and_namespace(runtime: BrokerRuntime) -> None:
    admin = str(runtime.pulsar_home / "bin" / "pulsar-admin")
    admin_url = "http://127.0.0.1:8080"
    _run_admin(
        runtime,
        [
            admin,
            "--admin-url",
            admin_url,
            "tenants",
            "create",
            "arena",
            "--allowed-clusters",
            "standalone",
        ],
    )
    _run_admin(
        runtime,
        [admin, "--admin-url", admin_url, "namespaces", "create", "arena/run"],
    )


def _run_admin(runtime: BrokerRuntime, command: list[str]) -> None:
    result = runtime.run_command(command)
    if result.code == 0 or "already exists" in result.output.lower():
        return
    msg = f"pulsar-admin failed ({result.code}): {result.output}"
    raise RuntimeError(msg)


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


def _stop_pulsar(process: PulsarProcess) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    process.wait()


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


def _http_status(url: str) -> int:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return int(response.status)
    except urllib.error.HTTPError as exc:
        return int(exc.code)
    except OSError:
        return 0


def _start_process(command: list[str], env: Mapping[str, str]) -> PulsarProcess:
    return subprocess.Popen(command, env=dict(env))


def _run_command(command: list[str]) -> CommandResult:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    return CommandResult(completed.returncode, completed.stdout + completed.stderr)


if __name__ == "__main__":
    raise SystemExit(main())

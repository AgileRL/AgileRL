# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""In-job Pulsar supervisor for one Snowflake training run."""

from __future__ import annotations

import json
import os
import runpy
import signal
import socket
import sys
import threading
import time
import warnings
from collections.abc import Iterator, Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from agilerl.arena.snowflake import broker as broker_mod
from agilerl.arena.snowflake.broker import (
    CGROUP_MEMORY_MAX,
    DEFAULT_PULSAR_HOME,
    BrokerMemory,
    BrokerRuntime,
    CommandResult,
    broker_memory,
    head_service_terminal,
    production_runtime,
    run_broker,
    service_status_terminal,
    write_broker_config,
)
from agilerl.arena.snowflake.job import RUN_MOUNT
from agilerl.arena.snowflake.node import exit_on_sigterm

RUN_ID = "arena-20261109163457-abc123"
HEAD = "ARENA_RUN_HEAD"
ADDRESS = "10.0.0.8"
HEALTH_URL = "http://127.0.0.1:8080/admin/v2/brokers/health"
ADMIN_URL = "http://127.0.0.1:8080"
# Half of CPU_X64_L, so the supervisor cannot pass by writing the 116 GiB numbers.
RUN_MEMORY = 58 * 1024**3


class Row:
    """Snowpark ``Row`` stand-in: values only through ``as_dict``."""

    def __init__(self, **values: object) -> None:
        self._values = values

    def as_dict(self) -> dict[str, object]:
        return dict(self._values)


class Result:
    def __init__(self, rows: list[Row]) -> None:
        self.rows = rows

    def collect(self) -> list[Row]:
        return self.rows


class Session:
    """LIST and GET hit a local stage. DESCRIBE SERVICE reads the host's status."""

    def __init__(self, host: Host) -> None:
        self.host = host
        self.file = self

    def sql(self, text: str) -> Result:
        self.host.statements.append(text)
        if text.startswith("LIST "):
            path = self.host.stage_root / text.removeprefix("LIST @")
            return Result([Row(name=path.name)] if path.is_file() else [])
        if text.startswith("DESCRIBE SERVICE "):
            if self.host.service_error is not None:
                raise self.host.service_error
            return Result([Row(status=self.host.service_status)])
        msg = f"unexpected SQL: {text}"
        raise AssertionError(msg)

    def get(self, stage_location: str, target_directory: str) -> None:
        source = self.host.stage_root / stage_location.removeprefix("@")
        (Path(target_directory) / source.name).write_bytes(source.read_bytes())

    def put(
        self,
        local: str,
        stage: str,
        auto_compress: bool = True,
        overwrite: bool = False,
    ) -> None:
        target = self.host.stage_root / stage.removeprefix("@")
        target.mkdir(parents=True, exist_ok=True)
        (target / Path(local).name).write_bytes(Path(local).read_bytes())
        self.host.events.append(f"put {Path(local).name}")

    def close(self) -> None:
        self.host.closed += 1


class Process:
    """Standalone process that stays up until SIGTERM, or exits after ``exit_after`` polls."""

    def __init__(self, host: Host) -> None:
        self.host = host
        self.terminated = False
        self.waited = False

    def poll(self) -> int | None:
        if self.terminated:
            return -15
        if self.host.exit_after is None:
            return None
        if self.host.exit_after > 0:
            self.host.exit_after -= 1
            return None
        return self.host.pulsar_code

    def terminate(self) -> None:
        self.terminated = True
        self.host.events.append("sigterm")

    def wait(self) -> int:
        self.waited = True
        self.host.events.append("wait")
        return -15


class Host:
    """Recorded calls and a local stand-in for the stage, cgroup file, and Pulsar home."""

    def __init__(self, tmp_path: Path) -> None:
        self.run_dir = tmp_path / "run"
        self.run_dir.mkdir()
        (self.run_dir / "job.json").write_text(
            json.dumps(
                {
                    "runs_stage": "arena.runs",
                    "run_id": RUN_ID,
                    "head_service": HEAD,
                }
            ),
            encoding="utf-8",
        )
        self.pulsar_home = tmp_path / "pulsar"
        self.memory_path = tmp_path / "memory.max"
        self.memory_path.write_text(f"{RUN_MEMORY}\n", encoding="utf-8")
        self.stage_root = tmp_path / "stage"
        self.events: list[str] = []
        self.statements: list[str] = []
        self.admin_commands: list[list[str]] = []
        self.pulsar_command: list[str] = []
        self.pulsar_env: dict[str, str] = {}
        self.closed = 0
        self.now = 0.0
        self.health_codes = [200]
        self.admin_results = [CommandResult(0, ""), CommandResult(0, "")]
        self.service_status = "RUNNING"
        self.service_error: Exception | None = None
        self.exit_after: int | None = None
        self.pulsar_code = 0
        self.process: Process | None = None
        self.config_before_process = False

    def put_done(self, code: int) -> None:
        path = self.stage_root / "arena.runs" / RUN_ID / ".done"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{code}\n", encoding="utf-8")

    def done(self) -> str | None:
        path = self.stage_root / "arena.runs" / RUN_ID / ".done"
        return path.read_text(encoding="utf-8") if path.is_file() else None

    def runtime(
        self,
        health_timeout_s: float = 600.0,
        head_gone_s: float = 60.0,
        poll_s: float = 2.0,
        run_poll_s: float = 20.0,
        head_start_timeout_s: float = 2700.0,
        session_max_age_s: float = 1800.0,
    ) -> BrokerRuntime:
        def sleep(seconds: float) -> None:
            self.now += seconds

        def start_process(command: list[str], env: Mapping[str, str]) -> Process:
            conf = self.pulsar_home / "conf" / "standalone.conf"
            jvm = self.pulsar_home / "conf" / "pulsar_env.sh"
            text = conf.read_text(encoding="utf-8")
            mem = jvm.read_text(encoding="utf-8")
            self.config_before_process = (
                f"advertisedAddress={ADDRESS}" in text and "-Xms12288m" in mem
            )
            self.events.append("process")
            self.pulsar_command = command
            self.pulsar_env = dict(env)
            self.process = Process(self)
            return self.process

        def run_command(command: list[str]) -> CommandResult:
            self.admin_commands.append(command)
            self.events.append("tenant" if "tenants" in command else "namespace")
            return self.admin_results.pop(0)

        def http_status(url: str) -> int:
            assert url == HEALTH_URL
            self.events.append("health")
            if len(self.health_codes) > 1:
                return self.health_codes.pop(0)
            return self.health_codes[0]

        return BrokerRuntime(
            environ={"PATH": "/usr/bin"},
            node_ip=lambda: ADDRESS,
            open_session=lambda: self._open(),
            start_process=start_process,
            run_command=run_command,
            http_status=http_status,
            sleep=sleep,
            monotonic=lambda: self.now,
            run_dir=self.run_dir,
            pulsar_home=self.pulsar_home,
            memory_limit_path=self.memory_path,
            health_timeout_s=health_timeout_s,
            head_start_timeout_s=head_start_timeout_s,
            head_gone_s=head_gone_s,
            poll_s=poll_s,
            run_poll_s=run_poll_s,
            session_max_age_s=session_max_age_s,
        )

    def _open(self) -> Session:
        self.events.append("poll")
        return Session(self)


@pytest.fixture
def host(tmp_path: Path) -> Host:
    return Host(tmp_path)


@pytest.fixture(autouse=True)
def ledger_dirs(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Record ledger-directory creates. This machine cannot write ``/pulsar-data``."""
    created: list[Path] = []
    real_mkdir = Path.mkdir

    def mkdir(
        self: Path,
        mode: int = 0o777,
        parents: bool = False,
        exist_ok: bool = False,
    ) -> None:
        if self.as_posix().startswith("/pulsar-data"):
            created.append(self)
            return
        real_mkdir(self, mode, parents=parents, exist_ok=exist_ok)

    monkeypatch.setattr(Path, "mkdir", mkdir)
    return created


class StatusHandler(BaseHTTPRequestHandler):
    """Answers ``/ok`` with 200 and any other path with 503."""

    def do_GET(self) -> None:
        self.send_response(200 if self.path == "/ok" else 503)
        self.end_headers()

    def log_message(self, *args: object) -> None:
        return None


@pytest.fixture
def status_server() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), StatusHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


class TestBrokerMemory:
    def test_scales_with_the_cgroup_limit(self) -> None:
        full = broker_memory(116 * 1024**3)
        half = broker_memory(58 * 1024**3)

        assert full == BrokerMemory(
            heap_mib=24 * 1024,
            direct_mib=48 * 1024,
            ledger_cache_mib=8 * 1024,
            db_write_cache_mib=4 * 1024,
            db_read_cache_mib=4 * 1024,
        )
        assert half == BrokerMemory(
            heap_mib=12 * 1024,
            direct_mib=24 * 1024,
            ledger_cache_mib=4 * 1024,
            db_write_cache_mib=2 * 1024,
            db_read_cache_mib=2 * 1024,
        )

    def test_floors_each_share_at_256_mib(self) -> None:
        memory = broker_memory(1024**3)

        assert memory.heap_mib == 256
        assert memory.direct_mib == 423
        assert memory.ledger_cache_mib == 256
        assert memory.db_write_cache_mib == 256
        assert memory.db_read_cache_mib == 256


class TestWriteBrokerConfig:
    def test_rendered_config_for_a_known_limit(
        self, tmp_path: Path, ledger_dirs: list[Path]
    ) -> None:
        memory = broker_memory(116 * 1024**3)

        write_broker_config(tmp_path, "10.0.0.8", memory)

        conf = (tmp_path / "conf" / "standalone.conf").read_text(encoding="utf-8")
        bookie = (tmp_path / "conf" / "bookkeeper.conf").read_text(encoding="utf-8")
        jvm = (tmp_path / "conf" / "pulsar_env.sh").read_text(encoding="utf-8")
        assert conf == (
            "advertisedAddress=10.0.0.8\n"
            "managedLedgerDefaultEnsembleSize=1\n"
            "managedLedgerDefaultWriteQuorum=1\n"
            "managedLedgerDefaultAckQuorum=1\n"
            "maxMessageSize=5242880\n"
            "managedLedgerCacheSizeMB=8192\n"
            "allowAutoTopicCreation=true\n"
            "brokerDeleteInactiveTopicsEnabled=false\n"
            "ttlDurationDefaultInSeconds=0\n"
            "backlogQuotaDefaultLimitBytes=-1\n"
            "defaultRetentionTimeInMinutes=0\n"
            "defaultRetentionSizeInMB=0\n"
        )
        assert bookie == (
            "journalDirectories=/pulsar-data/journal\n"
            "ledgerDirectories=/pulsar-data/ledgers\n"
            "journalSyncData=false\n"
            "dbStorage_writeCacheMaxSizeMb=4096\n"
            "dbStorage_readAheadCacheMaxSizeMb=4096\n"
        )
        assert jvm == (
            'PULSAR_MEM="-Xms24576m -Xmx24576m -XX:MaxDirectMemorySize=49152m"\n'
        )
        assert ledger_dirs == [
            Path("/pulsar-data/journal"),
            Path("/pulsar-data/ledgers"),
        ]

    def test_keeps_unrelated_settings(self, tmp_path: Path) -> None:
        conf_dir = tmp_path / "conf"
        conf_dir.mkdir()
        (conf_dir / "standalone.conf").write_text(
            "# broker\nbrokerServicePort=6650\nadvertisedAddress=old\n",
            encoding="utf-8",
        )
        (conf_dir / "pulsar_env.sh").write_text(
            'PULSAR_GC="-XX:+UseG1GC"\nPULSAR_MEM="-Xmx1g"\n',
            encoding="utf-8",
        )
        (conf_dir / "bookkeeper.conf").write_text(
            "prometheusStatsHttpPort=8000\njournalSyncData=true\n",
            encoding="utf-8",
        )

        write_broker_config(tmp_path, ADDRESS, broker_memory(116 * 1024**3))

        conf = (conf_dir / "standalone.conf").read_text(encoding="utf-8")
        bookie = (conf_dir / "bookkeeper.conf").read_text(encoding="utf-8")
        jvm = (conf_dir / "pulsar_env.sh").read_text(encoding="utf-8")
        assert conf.startswith("# broker\nbrokerServicePort=6650\n")
        assert conf.count("advertisedAddress=") == 1
        assert "advertisedAddress=10.0.0.8" in conf
        assert 'PULSAR_GC="-XX:+UseG1GC"' in jvm
        assert "-Xmx24576m" in jvm
        assert jvm.count("PULSAR_MEM=") == 1
        assert bookie.startswith("prometheusStatsHttpPort=8000\n")
        assert bookie.count("journalSyncData=") == 1
        assert "journalSyncData=false" in bookie
        assert "ledgerDirectories=/pulsar-data/ledgers" in bookie


class TestServiceStatusTerminal:
    def test_done_failed_and_cancelled(self) -> None:
        assert service_status_terminal([Row(STATUS="DONE")]) is True
        assert service_status_terminal([Row(status="failed")]) is True
        assert service_status_terminal([Row(status="CANCELLED")]) is True

    def test_running_is_not_terminal(self) -> None:
        assert service_status_terminal([Row(status="RUNNING")]) is False
        assert service_status_terminal([Row(name="ARENA_RUN_HEAD")]) is False

    def test_json_status_must_all_be_terminal(self) -> None:
        mixed = json.dumps([{"status": "DONE"}, {"status": "READY"}])
        done = json.dumps([{"status": "DONE"}, {"status": "CANCELLED"}])

        assert service_status_terminal([Row(report=mixed)]) is False
        assert service_status_terminal([Row(report=done)]) is True

    def test_no_rows_means_the_service_is_gone(self) -> None:
        assert service_status_terminal([]) is True

    def test_unreadable_or_empty_statuses_are_not_terminal(self) -> None:
        assert service_status_terminal([Row(report="[not json")]) is False
        assert service_status_terminal([Row(status=" ")]) is False


class TestHeadServiceTerminal:
    def test_missing_service_is_terminal(self, host: Host) -> None:
        host.service_error = RuntimeError(
            "Service 'ARENA_RUN_HEAD' does not exist or not authorized."
        )

        assert head_service_terminal(Session(host), HEAD) is True

        assert host.statements == [f"DESCRIBE SERVICE {HEAD}"]

    def test_other_sql_errors_propagate(self, host: Host) -> None:
        host.service_error = RuntimeError("warehouse suspended")

        with pytest.raises(RuntimeError, match="warehouse suspended"):
            head_service_terminal(Session(host), HEAD)

    def test_rejects_an_unsafe_name(self, host: Host) -> None:
        with pytest.raises(ValueError, match="service"):
            head_service_terminal(Session(host), "HEAD; DROP")


class TestRunBroker:
    def test_starts_in_order(self, host: Host) -> None:
        # Arrange
        host.put_done(0)
        admin = str(host.pulsar_home / "bin" / "pulsar-admin")

        # Act
        code = run_broker(host.runtime())

        # Assert
        assert code == 0
        assert host.config_before_process is True
        assert host.events == [
            "process",
            "health",
            "tenant",
            "namespace",
            "poll",
            "sigterm",
            "wait",
        ]
        assert host.pulsar_command == [
            str(host.pulsar_home / "bin" / "pulsar"),
            "standalone",
            "--config",
            str(host.pulsar_home / "conf" / "standalone.conf"),
        ]
        assert host.pulsar_env["PULSAR_HOME"] == str(host.pulsar_home)
        assert host.admin_commands == [
            [
                admin,
                "--admin-url",
                ADMIN_URL,
                "tenants",
                "create",
                "arena",
                "--allowed-clusters",
                "standalone",
            ],
            [admin, "--admin-url", ADMIN_URL, "namespaces", "create", "arena/run"],
        ]

    def test_health_is_polled_until_it_returns_200(self, host: Host) -> None:
        host.put_done(0)
        host.health_codes = [503, 503, 200]

        code = run_broker(host.runtime(poll_s=5))

        assert code == 0
        assert host.now == 10
        assert host.events[:5] == ["process", "health", "health", "health", "tenant"]

    def test_existing_tenant_and_namespace_are_success(self, host: Host) -> None:
        host.put_done(0)
        host.admin_results = [
            CommandResult(1, "ConflictException: Tenant already exists"),
            CommandResult(1, "Namespace already exists"),
        ]

        assert run_broker(host.runtime()) == 0

    def test_done_zero_stops_the_process_and_returns_zero(self, host: Host) -> None:
        # Arrange
        host.put_done(0)

        # Act
        code = run_broker(host.runtime())

        # Assert
        assert code == 0
        assert host.process is not None
        assert host.process.terminated is True
        assert host.process.waited is True
        assert host.statements[0] == f"LIST @arena.runs/{RUN_ID}/.done"

    def test_done_during_the_head_grace_returns_that_code(self, host: Host) -> None:
        # Arrange
        host.service_status = "DONE"
        runtime = host.runtime(head_gone_s=60, run_poll_s=10)

        def sleep(seconds: float) -> None:
            host.now += seconds
            if host.now >= 20:
                host.put_done(4)

        runtime.sleep = sleep

        # Act
        code = run_broker(runtime)

        # Assert
        assert code == 4
        assert host.now == 20
        assert host.process is not None
        assert host.process.terminated is True

    def test_head_terminal_without_done_exits_one(self, host: Host) -> None:
        # Arrange
        host.service_status = "DONE"

        # Act
        code = run_broker(host.runtime(head_gone_s=60, run_poll_s=60))

        # Assert
        assert code == 1
        assert host.now == 60
        assert host.closed == 1
        assert f"DESCRIBE SERVICE {HEAD}" in host.statements
        assert host.process is not None
        assert host.process.terminated is True
        assert host.process.waited is True

    def test_polls_every_20s_on_one_session(self, host: Host) -> None:
        # Arrange
        runtime = host.runtime()

        def sleep(seconds: float) -> None:
            host.now += seconds
            if host.now >= 60:
                host.put_done(0)

        runtime.sleep = sleep

        # Act
        code = run_broker(runtime)

        # Assert
        assert code == 0
        assert host.now == 60
        assert host.events.count("poll") == 1
        assert host.closed == 1
        assert host.statements.count(f"LIST @arena.runs/{RUN_ID}/.done") == 4

    def test_the_session_is_reopened_once_it_is_too_old(self, host: Host) -> None:
        # Arrange
        runtime = host.runtime(session_max_age_s=40)

        def sleep(seconds: float) -> None:
            host.now += seconds
            if host.now >= 100:
                host.put_done(0)

        runtime.sleep = sleep

        # Act
        code = run_broker(runtime)

        # Assert
        assert code == 0
        assert host.events.count("poll") == 3
        assert host.closed == 3

    def test_head_that_never_runs_writes_a_failed_done(self, host: Host) -> None:
        # Arrange
        host.service_status = "PENDING"

        # Act
        code = run_broker(host.runtime(head_start_timeout_s=100))

        # Assert
        assert code == 1
        assert host.now == 100
        assert host.done() == "1\n"
        assert host.process is not None
        assert host.process.terminated is True

    def test_running_head_is_not_held_to_the_start_deadline(self, host: Host) -> None:
        # Arrange
        statuses = iter(["PENDING", "RUNNING"])
        runtime = host.runtime(head_start_timeout_s=40)

        def sleep(seconds: float) -> None:
            host.now += seconds
            host.service_status = next(statuses, "RUNNING")
            if host.now >= 200:
                host.put_done(0)

        host.service_status = "PENDING"
        runtime.sleep = sleep

        # Act
        code = run_broker(runtime)

        # Assert
        assert code == 0
        assert host.now == 200

    def test_pulsar_exiting_returns_one(self, host: Host) -> None:
        # Arrange
        host.exit_after = 0
        host.pulsar_code = 3

        # Act
        code = run_broker(host.runtime())

        # Assert
        assert code == 1
        assert host.process is not None
        assert host.process.terminated is False
        assert host.now == 0

    def test_health_never_ok_raises_timeout(self, host: Host) -> None:
        # Arrange
        host.health_codes = [503]

        # Act
        with pytest.raises(TimeoutError, match="did not return 200"):
            run_broker(host.runtime(health_timeout_s=0))

        # Assert
        assert host.events == ["process", "health", "sigterm", "wait"]
        assert host.process is not None
        assert host.process.terminated is True

    def test_dropped_head_service_counts_as_terminal(self, host: Host) -> None:
        host.service_error = RuntimeError(
            "Service 'X' does not exist or not authorized."
        )

        code = run_broker(host.runtime(head_gone_s=40))

        assert code == 1
        assert host.now == 40
        assert host.done() is None

    def test_admin_failure_raises(self, host: Host) -> None:
        host.admin_results = [CommandResult(1, "connection refused")]

        with pytest.raises(RuntimeError, match="connection refused"):
            run_broker(host.runtime())

        assert host.process is not None
        assert host.process.terminated is True
        assert "namespace" not in host.events


class TestMain:
    def test_installs_the_sigterm_handler_and_runs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        installed: dict[int, object] = {}
        monkeypatch.setattr(
            broker_mod.signal,
            "signal",
            lambda signum, handler: installed.setdefault(signum, handler),
        )
        monkeypatch.setattr(broker_mod, "run_broker", lambda runtime=None: 5)

        # Act
        code = broker_mod.main()

        # Assert
        assert code == 5
        assert installed == {signal.SIGTERM: exit_on_sigterm}

    def test_module_entrypoint_reads_the_run_job_json(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        installed: dict[int, object] = {}
        monkeypatch.setattr(
            signal,
            "signal",
            lambda signum, handler: installed.setdefault(signum, handler),
        )

        # Act
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*found in sys.modules.*",
                category=RuntimeWarning,
            )
            with pytest.raises(FileNotFoundError, match=r"job\.json"):
                runpy.run_module("agilerl.arena.snowflake.broker", run_name="__main__")

        # Assert
        assert installed == {signal.SIGTERM: exit_on_sigterm}


class TestProductionRuntime:
    def test_pulsar_home_comes_from_the_environment(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("PULSAR_HOME", str(tmp_path))

        runtime = production_runtime()

        assert runtime.pulsar_home == tmp_path
        assert runtime.run_dir == Path(RUN_MOUNT)
        assert runtime.memory_limit_path == CGROUP_MEMORY_MAX
        assert runtime.sleep is time.sleep
        assert runtime.monotonic is time.monotonic
        assert runtime.node_ip()

    def test_an_unset_pulsar_home_uses_the_install_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("PULSAR_HOME", raising=False)

        assert production_runtime().pulsar_home == DEFAULT_PULSAR_HOME

    def test_start_process_passes_the_environment(self) -> None:
        script = "import os, sys; sys.exit(int(os.environ['ARENA_BROKER_CODE']))"

        process = production_runtime().start_process(
            [sys.executable, "-c", script], {**os.environ, "ARENA_BROKER_CODE": "3"}
        )

        assert process.wait() == 3

    def test_run_command_returns_the_code_and_both_streams(self) -> None:
        script = "import sys; print('out'); sys.stderr.write('err'); sys.exit(2)"

        result = production_runtime().run_command([sys.executable, "-c", script])

        assert result == CommandResult(2, "out\nerr")

    def test_http_status_reads_the_response_code(self, status_server: str) -> None:
        runtime = production_runtime()

        assert runtime.http_status(f"{status_server}/ok") == 200
        assert runtime.http_status(f"{status_server}/busy") == 503

    def test_http_status_is_zero_when_nothing_listens(self) -> None:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]

        assert production_runtime().http_status(f"http://127.0.0.1:{port}/") == 0

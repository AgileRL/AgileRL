# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Pulsar sidecar for a training job's Ray head pod."""

from __future__ import annotations

import runpy
import signal
import time
import warnings
from collections.abc import Mapping
from pathlib import Path

import pytest

from agilerl import broker as broker_base
from agilerl import job_broker as job_broker_mod
from agilerl.broker import (
    CGROUP_MEMORY_MAX,
    DATA_ROOT,
    DEFAULT_PULSAR_HOME,
    CommandResult,
    broker_memory,
)
from agilerl.job_broker import (
    BACKLOG_QUOTA_BYTES,
    JobBrokerRuntime,
    production_runtime,
    run_job_broker,
    stop_on_sigterm,
)

ADDRESS = "10.4.0.12"
HEALTH_URL = "http://127.0.0.1:8081/admin/v2/brokers/health"
ADMIN_URL = "http://127.0.0.1:8081"
SIDECAR_MEMORY = 4 * 1024**3


class Process:
    """Standalone process that runs until SIGTERM, or exits after ``exit_after`` polls."""

    def __init__(self, host: Host, exit_after: int | None) -> None:
        self.host = host
        self.exit_after = exit_after
        self.terminated = False

    def poll(self) -> int | None:
        if self.terminated:
            return -15
        if self.exit_after is None:
            return None
        if self.exit_after > 0:
            self.exit_after -= 1
            return None
        return 2

    def terminate(self) -> None:
        self.terminated = True
        self.host.events.append("sigterm")

    def wait(self) -> int:
        self.host.events.append("wait")
        return -15


class Host:
    """Recorded calls and a local stand-in for the data volume, cgroup file, and Pulsar home."""

    def __init__(self, tmp_path: Path) -> None:
        self.pulsar_home = tmp_path / "pulsar"
        self.data_root = tmp_path / "pulsar-data"
        self.memory_path = tmp_path / "memory.max"
        self.memory_path.write_text(f"{SIDECAR_MEMORY}\n", encoding="utf-8")
        self.events: list[str] = []
        self.commands: list[list[str]] = []
        self.admin_commands: list[list[str]] = []
        self.envs: list[dict[str, str]] = []
        self.processes: list[Process] = []
        self.now = 0.0
        self.health_codes = [200]
        self.admin_results = [CommandResult(0, ""), CommandResult(0, "")]
        # One entry per start: polls before the process exits, or None to run.
        self.exits: list[int | None] = [None]
        self.ready_while_running: list[bool] = []

    def runtime(self, health_timeout_s: float = 600.0) -> JobBrokerRuntime:
        def sleep(seconds: float) -> None:
            self.ready_while_running.append((self.data_root / "ready").is_file())
            self.now += seconds

        def start_process(command: list[str], env: Mapping[str, str]) -> Process:
            self.events.append("process")
            self.commands.append(command)
            self.envs.append(dict(env))
            process = Process(self, self.exits.pop(0))
            self.processes.append(process)
            return process

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

        return JobBrokerRuntime(
            environ={"PATH": "/usr/bin"},
            node_ip=lambda: ADDRESS,
            start_process=start_process,
            run_command=run_command,
            http_status=http_status,
            sleep=sleep,
            monotonic=lambda: self.now,
            pulsar_home=self.pulsar_home,
            data_root=self.data_root,
            memory_limit_path=self.memory_path,
            health_timeout_s=health_timeout_s,
            poll_s=2.0,
        )

    def conf(self, name: str) -> str:
        return (self.pulsar_home / "conf" / name).read_text(encoding="utf-8")


@pytest.fixture
def host(tmp_path: Path) -> Host:
    return Host(tmp_path)


class TestRunJobBroker:
    def test_starts_marks_ready_and_stops_when_pulsar_exits(self, host: Host) -> None:
        # Arrange
        host.exits = [3]
        admin = str(host.pulsar_home / "bin" / "pulsar-admin")

        # Act
        code = run_job_broker(host.runtime())

        # Assert
        assert code == 1
        assert host.events == ["process", "health", "tenant", "namespace"]
        assert host.commands == [
            [
                str(host.pulsar_home / "bin" / "pulsar"),
                "standalone",
                "--config",
                str(host.pulsar_home / "conf" / "standalone.conf"),
                "--bookkeeper-dir",
                str(host.data_root / "bookkeeper"),
                "--metadata-dir",
                str(host.data_root / "metadata"),
                "--no-functions-worker",
                "--no-stream-storage",
            ]
        ]
        assert host.envs[0] == {
            "PATH": "/usr/bin",
            "PULSAR_HOME": str(host.pulsar_home),
        }
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
        assert host.ready_while_running == [True, True, True]
        assert not (host.data_root / "ready").exists()

    def test_config_serves_web_on_8081_with_an_8_gib_topic_quota(
        self, host: Host
    ) -> None:
        # Arrange
        host.exits = [0]
        memory = broker_memory(SIDECAR_MEMORY)

        # Act
        run_job_broker(host.runtime())

        # Assert
        conf = host.conf("standalone.conf")
        assert f"advertisedAddress={ADDRESS}\n" in conf
        assert "webServicePort=8081\n" in conf
        assert "backlogQuotaDefaultLimitGB=0\n" in conf
        assert f"backlogQuotaDefaultLimitBytes={8 * 1024**3}\n" in conf
        assert "backlogQuotaDefaultRetentionPolicy=producer_request_hold\n" in conf
        assert BACKLOG_QUOTA_BYTES == 8589934592
        assert (
            f"journalDirectories={(host.data_root / 'journal').as_posix()}\n"
            in host.conf("bookkeeper.conf")
        )
        assert host.conf("pulsar_env.sh") == (
            f'PULSAR_MEM="-Xms{memory.heap_mib}m -Xmx{memory.heap_mib}m '
            f'-XX:MaxDirectMemorySize={memory.direct_mib}m"\n'
        )
        assert memory.heap_mib == 847

    def test_health_is_polled_until_it_returns_200(self, host: Host) -> None:
        host.exits = [None]
        host.health_codes = [0, 503, 200]
        runtime = host.runtime()

        def stop_after_ready(seconds: float) -> None:
            host.now += seconds
            if host.now > 4:
                raise SystemExit(0)

        runtime.sleep = stop_after_ready

        with pytest.raises(SystemExit):
            run_job_broker(runtime)

        assert host.events[:5] == ["process", "health", "health", "health", "tenant"]

    def test_sigterm_stops_pulsar_and_clears_ready(self, host: Host) -> None:
        # Arrange
        runtime = host.runtime()

        def sigterm(_seconds: float) -> None:
            host.ready_while_running.append((host.data_root / "ready").is_file())
            stop_on_sigterm(signal.SIGTERM, None)

        runtime.sleep = sigterm

        # Act
        with pytest.raises(SystemExit) as stopped:
            run_job_broker(runtime)

        # Assert
        assert stopped.value.code == 0
        assert host.ready_while_running == [True]
        assert host.processes[0].terminated is True
        assert host.events[-2:] == ["sigterm", "wait"]
        assert not (host.data_root / "ready").exists()

    def test_a_stale_ready_marker_is_removed_before_start(self, host: Host) -> None:
        host.data_root.mkdir()
        (host.data_root / "ready").touch()
        host.health_codes = [503]

        with pytest.raises(TimeoutError):
            run_job_broker(host.runtime(health_timeout_s=0))

        assert not (host.data_root / "ready").exists()

    def test_a_restart_on_existing_data_keeps_the_tenant_and_namespace(
        self, host: Host
    ) -> None:
        # Arrange
        host.exits = [0]
        host.admin_results = [
            CommandResult(
                1,
                "Reason: Tenant already exist\nTenant already exist",
            ),
            CommandResult(1, "Reason: Namespace already exists"),
        ]

        # Act
        code = run_job_broker(host.runtime())

        # Assert
        assert code == 1
        assert host.events == ["process", "health", "tenant", "namespace"]

    def test_no_memory_limit_is_rejected(self, host: Host) -> None:
        host.memory_path.write_text("max\n", encoding="utf-8")

        with pytest.raises(ValueError, match="no memory limit"):
            run_job_broker(host.runtime())

        assert host.events == []

    def test_admin_failure_stops_pulsar(self, host: Host) -> None:
        host.admin_results = [CommandResult(1, "connection refused")]

        with pytest.raises(RuntimeError, match="connection refused"):
            run_job_broker(host.runtime())

        assert host.processes[0].terminated is True
        assert "namespace" not in host.events
        assert not (host.data_root / "ready").exists()


class TestRecovery:
    def test_unrecoverable_data_is_wiped_and_the_broker_starts_empty(
        self, host: Host
    ) -> None:
        # Arrange
        old_ledger = host.data_root / "bookkeeper" / "ledger.log"
        old_ledger.parent.mkdir(parents=True)
        old_ledger.write_text("corrupt", encoding="utf-8")
        (host.data_root / "metadata").mkdir()
        host.exits = [0, 0]
        host.health_codes = [503, 200]

        # Act
        code = run_job_broker(host.runtime())

        # Assert
        assert code == 1
        assert host.events == [
            "process",
            "health",
            "process",
            "health",
            "tenant",
            "namespace",
        ]
        assert not (host.data_root / "bookkeeper").exists()
        assert not (host.data_root / "metadata").exists()

    def test_a_first_start_that_fails_is_not_retried(self, host: Host) -> None:
        host.exits = [0]
        host.health_codes = [503]

        with pytest.raises(RuntimeError, match="Pulsar exited with 2"):
            run_job_broker(host.runtime())

        assert host.events == ["process", "health"]

    def test_a_second_failure_raises_and_stops_pulsar(self, host: Host) -> None:
        (host.data_root / "metadata").mkdir(parents=True)
        host.exits = [None, None]
        host.health_codes = [503]

        with pytest.raises(TimeoutError, match="8081"):
            run_job_broker(host.runtime(health_timeout_s=0))

        assert host.events == ["process", "health", "sigterm", "wait"] * 2
        assert all(process.terminated for process in host.processes)


class TestMain:
    def test_installs_the_sigterm_handler_and_runs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        installed: dict[int, object] = {}
        monkeypatch.setattr(
            job_broker_mod.signal,
            "signal",
            lambda signum, handler: installed.setdefault(signum, handler),
        )
        monkeypatch.setattr(job_broker_mod, "run_job_broker", lambda runtime=None: 7)

        # Act
        code = job_broker_mod.main()

        # Assert
        assert code == 7
        assert installed == {signal.SIGTERM: stop_on_sigterm}

    def test_module_entrypoint_reads_this_container_limit(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Arrange
        installed: dict[int, object] = {}
        monkeypatch.setattr(
            signal,
            "signal",
            lambda signum, handler: installed.setdefault(signum, handler),
        )
        monkeypatch.setattr(broker_base, "CGROUP_MEMORY_MAX", tmp_path / "missing")

        # Act
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*found in sys.modules.*",
                category=RuntimeWarning,
            )
            with pytest.raises(ValueError, match="no memory limit"):
                runpy.run_module("agilerl.job_broker", run_name="__main__")

        # Assert
        assert list(installed) == [signal.SIGTERM]


class TestProductionRuntime:
    def test_defaults_point_at_the_image_and_the_data_volume(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("PULSAR_HOME", raising=False)

        runtime = production_runtime()

        assert runtime.pulsar_home == DEFAULT_PULSAR_HOME
        assert runtime.data_root == DATA_ROOT
        assert runtime.memory_limit_path == CGROUP_MEMORY_MAX
        assert runtime.sleep is time.sleep
        assert runtime.monotonic is time.monotonic
        assert runtime.http_status is broker_base.http_status

    def test_pulsar_home_comes_from_the_environment(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("PULSAR_HOME", str(tmp_path))

        assert production_runtime().pulsar_home == tmp_path

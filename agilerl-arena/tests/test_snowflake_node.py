# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Ray startup, checkpoint upload, and shutdown on each SPCS job instance."""

from __future__ import annotations

import dataclasses
import json
import os
import runpy
import signal
import socket
import sys
import tempfile
import threading
import types
import warnings
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest
from typing_extensions import Self

from agilerl.arena.snowflake import node as node_mod
from agilerl.arena.snowflake.job import run_services
from agilerl.arena.snowflake.license import (
    COMPLETE_FAILED,
    OPEN_FAILED,
    SNOWFLAKE_DENIED,
    RunRenewalRefusedError,
)
from agilerl.arena.snowflake.node import (
    SIGTERM_EXIT_CODE,
    HeadFinishedError,
    NodeRuntime,
    RunHeartbeat,
    StartupClock,
    earliest_worker_start,
    exit_on_sigterm,
    head_ip_from_rows,
    head_ray_command,
    main,
    production_runtime,
    ray_nodes,
    read_done,
    run_node,
    shared_memory_bytes,
    wait_for_head_ip,
    worker_ray_command,
    write_done,
    write_run_token,
)

RUN_ID = "arena-20261109163457-abc123"
RUN_PREFIX = f"@arena.runs/{RUN_ID}"
SERVICES = run_services(RUN_ID)
HEAD_ENV = {"SNOWFLAKE_SERVICE_NAME": SERVICES.head}
WORKER_ENV = {"SNOWFLAKE_SERVICE_NAME": SERVICES.workers}
BROKER_IP = "10.0.0.7"
PULSAR_URL = f"pulsar://{BROKER_IP}:6650"
WORKER_IP = "10.0.0.5"
WORKER_STARTED = "2026-11-11T18:00:00+00:00"
RUN_TOKEN = "v1.arena-dev-1.payload.signature"
DROP_BROKER = f"DROP SERVICE IF EXISTS {SERVICES.broker} FORCE"


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
    """Answers ``SHOW SERVICE INSTANCES`` from a script; LIST, PUT, and GET hit a local stage."""

    def __init__(self, cluster: Cluster) -> None:
        self.cluster = cluster
        self.file = self

    def sql(self, text: str) -> Result:
        self.cluster.statements.append(text)
        if text.startswith("LIST "):
            listed = text.removeprefix("LIST @")
            directory = self.cluster.stage_root / listed.rsplit("/", 1)[0]
            start = listed.rsplit("/", 1)[1]
            names = (
                sorted(
                    path.name
                    for path in directory.iterdir()
                    if path.is_file() and path.name.startswith(start)
                )
                if directory.is_dir()
                else []
            )
            return Result([Row(name=f"{RUN_ID}/{name}") for name in names])
        if text.startswith("DROP SERVICE "):
            self.cluster.events.append(text)
            return Result([])
        prefix = "SHOW SERVICE INSTANCES IN SERVICE "
        if text.startswith(prefix):
            script = self.cluster.instances.get(text.removeprefix(prefix))
            if script is not None:
                return Result(script.pop(0) if len(script) > 1 else script[0])
        script = self.cluster.instance_rows
        return Result(script.pop(0) if len(script) > 1 else script[0])

    def get(self, stage_location: str, target_directory: str) -> list[object]:
        source = self.cluster.stage_root / stage_location.removeprefix("@")
        (Path(target_directory) / source.name).write_bytes(source.read_bytes())
        return []

    def put(
        self,
        local: str,
        stage: str,
        auto_compress: bool = True,
        overwrite: bool = False,
    ) -> None:
        target = self.cluster.stage_root / stage.removeprefix("@")
        target.mkdir(parents=True, exist_ok=True)
        (target / Path(local).name).write_bytes(Path(local).read_bytes())
        self.cluster.events.append(f"put {Path(local).name}")

    def close(self) -> None:
        self.cluster.closed += 1


class Process:
    """Head command that exits with ``code`` after ``polls`` running polls."""

    def __init__(
        self, code: int, polls: int, on_poll: Callable[[], None], events: list[str]
    ) -> None:
        self.code = code
        self.polls = polls
        self.on_poll = on_poll
        self.events = events
        self.killed = False

    def poll(self) -> int | None:
        if self.killed:
            return -9
        if self.polls > 0:
            self.polls -= 1
            self.on_poll()
            return None
        if "exit" not in self.events:
            self.events.append("exit")
        return self.code

    def kill(self) -> None:
        self.killed = True

    def wait(self) -> int:
        return -9


class Cluster:
    """Recorded calls and a local stand-in for the stage and run mount."""

    def __init__(self, tmp_path: Path) -> None:
        self.run_dir = tmp_path / "run"
        self.run_dir.mkdir()
        self.ckpt_root = tmp_path / "ckpt"
        self.stage_root = tmp_path / "stage"
        self.instance_rows: list[list[Row]] = [
            [Row(instance_id=0, ip_address=BROKER_IP)]
        ]
        self.instances: dict[str, list[list[Row]]] = {}
        self.statements: list[str] = []
        self.events: list[str] = []
        self.closed = 0
        self.now = 0.0
        self.commands: list[tuple[list[str], Mapping[str, str]]] = []
        self.ray_commands: list[tuple[list[str], Mapping[str, str]]] = []
        self.token_path = tmp_path / "arena" / "run_token"
        self.write_job()
        self.put_marker(f".started-{WORKER_IP}", WORKER_STARTED)
        self.put_marker(f".uploaded-{WORKER_IP}", "")

    def write_job(self, **overrides: object) -> None:
        document: dict[str, object] = {
            "run_id": RUN_ID,
            "runs_stage": "arena.runs",
            "nodes": 1,
            "gpus": 4,
            "family": "GPU_NV_M",
            "broker_service": SERVICES.broker,
            "head_service": SERVICES.head,
            "workers_service": SERVICES.workers,
        }
        document.update(overrides)
        (self.run_dir / "job.json").write_text(json.dumps(document), encoding="utf-8")

    def stage_file(self, relative: str) -> Path:
        return self.stage_root / "arena.runs" / RUN_ID / relative

    def put_marker(self, name: str, text: str) -> None:
        path = self.stage_file(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def put_done(self, code: int) -> None:
        self.put_marker(".done", f"{code}\n")

    def done(self) -> int | None:
        return read_done(Session(self), RUN_PREFIX)

    def runtime(
        self,
        environ: Mapping[str, str] | None = None,
        code: int = 0,
        polls: int = 0,
        on_poll: Callable[[], None] = lambda: None,
        gcs_open: Callable[[str], bool] = lambda host: True,
        alive_nodes: Callable[[], int] = lambda: 2,
    ) -> NodeRuntime:
        def sleep(seconds: float) -> None:
            self.now += seconds

        @contextmanager
        def nodes() -> Iterator[Callable[[], int]]:
            yield alive_nodes

        def start_command(command: list[str], env: Mapping[str, str]) -> Process:
            self.commands.append((command, dict(env)))
            self.events.append("command")
            return Process(code, polls, on_poll, self.events)

        def start_ray(args: list[str], env: Mapping[str, str]) -> None:
            self.ray_commands.append((list(args), dict(env)))
            self.events.append(f"ray {' '.join(args[:4])}")

        def gcs(host: str) -> bool:
            opened = gcs_open(host)
            self.events.append(f"gcs {host} {opened}")
            return opened

        chosen = HEAD_ENV if environ is None else environ
        if chosen is not os.environ:
            chosen = dict(chosen)

        return NodeRuntime(
            environ=chosen,
            node_ip=lambda: "10.0.0.4",
            open_session=lambda: Session(self),
            start_ray=start_ray,
            stop_ray=lambda: self.events.append("ray stop"),
            ray_nodes=nodes,
            gcs_open=gcs,
            broker_healthy=lambda ip: True,
            start_command=start_command,
            sleep=sleep,
            monotonic=lambda: self.now,
            shared_memory_bytes=lambda: 1000,
            retry_on=(OSError,),
            run_dir=self.run_dir,
            ckpt_root=self.ckpt_root,
            token_path=self.token_path,
            startup_timeout_s=10,
            uploaded_timeout_s=6,
            head_gone_s=6,
            poll_s=2,
            upload_interval_s=4,
            heartbeat_interval_s=1000,
        )


def _checkpoint(root: Path, number: int) -> None:
    merged = root / f"checkpoint_{number}" / "merged"
    merged.mkdir(parents=True)
    (merged / "model.safetensors").write_text("w", encoding="utf-8")
    (merged / ".complete").write_text("", encoding="utf-8")


def _open_then_closed(open_polls: int) -> Callable[[str], bool]:
    """GCS probe that is open for ``open_polls`` calls, then closed."""
    remaining = iter([True] * open_polls)
    return lambda host: next(remaining, False)


@pytest.fixture
def cluster(tmp_path: Path) -> Cluster:
    return Cluster(tmp_path)


@pytest.fixture(autouse=True)
def _allow_head_license(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    """Head tests that are not about Arena skip the license HTTP calls."""
    if getattr(request.cls, "__name__", "") in {"TestHeadLicense", "TestRunHeartbeat"}:
        return
    monkeypatch.setattr(
        "agilerl.arena.snowflake.node.check_snowflake_license", lambda env: None
    )
    monkeypatch.setattr(
        "agilerl.arena.snowflake.node.open_head_run",
        lambda job, env, started_at, sleep: RUN_TOKEN,
    )
    monkeypatch.setattr(
        "agilerl.arena.snowflake.node.complete_snowflake_run",
        lambda run_id, env, sleep, retry_budget_s, timeout_s: None,
    )


class Completions:
    """Records complete posts, as ``complete`` in the cluster event log."""

    def __init__(self, cluster: Cluster) -> None:
        self.cluster = cluster
        self.calls: list[tuple[str, float, float]] = []
        self.heartbeat_alive: list[bool] = []

    def __call__(
        self,
        run_id: str,
        env: Mapping[str, str],
        sleep: object,
        retry_budget_s: float,
        timeout_s: float,
    ) -> None:
        self.calls.append((run_id, retry_budget_s, timeout_s))
        self.cluster.events.append("complete")
        self.heartbeat_alive.append(
            any(
                thread.name == "arena-heartbeat" and thread.is_alive()
                for thread in threading.enumerate()
            )
        )


class TestRayCommands:
    def test_head_and_worker_use_the_ray_ports(self) -> None:
        head = head_ray_command("10.0.0.4", 4096)
        worker = worker_ray_command("10.0.0.5", "10.0.0.4", 4096)

        assert head[:4] == ["ray", "start", "--head", "--node-ip-address"]
        assert head[head.index("--port") + 1] == "61001"
        assert head[head.index("--ray-client-server-port") + 1] == "61002"
        assert head[head.index("--dashboard-port") + 1] == "61003"
        assert head[head.index("--object-manager-port") + 1] == "61011"
        assert head[head.index("--node-manager-port") + 1] == "61012"
        assert head[head.index("--runtime-env-agent-port") + 1] == "61013"
        assert head[head.index("--dashboard-agent-grpc-port") + 1] == "61014"
        assert head[head.index("--dashboard-agent-listen-port") + 1] == "61015"
        assert head[head.index("--metrics-export-port") + 1] == "61016"
        assert head[head.index("--min-worker-port") + 1] == "61031"
        assert head[head.index("--max-worker-port") + 1] == "61999"
        assert head[head.index("--object-store-memory") + 1] == "4096"
        assert "--disable-usage-stats" in head
        assert head[head.index("--num-cpus") + 1] == "0"
        assert head[head.index("--num-gpus") + 1] == "0"
        assert worker[:4] == ["ray", "start", "--address", "10.0.0.4:61001"]
        assert "--num-cpus" not in worker
        assert "--num-gpus" not in worker
        assert "--head" not in worker
        assert "--ray-client-server-port" not in worker
        assert worker[worker.index("--node-ip-address") + 1] == "10.0.0.5"


class TestHeadIpFromRows:
    def test_instance_zero_ip(self) -> None:
        rows = [
            Row(instance_id=1, ip_address="10.0.0.2"),
            Row(INSTANCE_ID="0", IP_ADDRESS="10.0.0.1"),
        ]

        assert head_ip_from_rows(rows) == "10.0.0.1"

    @pytest.mark.parametrize(
        "rows",
        [
            [],
            [Row(instance_id=1, ip_address="10.0.0.2")],
            [Row(instance_id=0, ip_address="  ")],
            [Row(instance_id=0, ip_address=None)],
        ],
    )
    def test_instance_zero_not_ready(self, rows: list[Row]) -> None:
        assert head_ip_from_rows(rows) is None


def _startup(cluster: Cluster, run_prefix: str | None = None) -> StartupClock:
    runtime = cluster.runtime()
    return StartupClock(
        runtime, Session(cluster), run_prefix, runtime.startup_timeout_s
    )


class TestWaitForHeadIp:
    def test_waits_for_instance_zero_on_the_startup_session(
        self, cluster: Cluster
    ) -> None:
        cluster.instance_rows = [
            [Row(instance_id=1, ip_address="10.0.0.9")],
            [
                Row(instance_id=0, ip_address=None),
                Row(instance_id=1, ip_address="10.0.0.9"),
            ],
            [Row(instance_id=0, ip_address="10.0.0.4")],
        ]

        ip = wait_for_head_ip(_startup(cluster), "ARENA_RUN")

        assert ip == "10.0.0.4"
        assert cluster.statements == ["SHOW SERVICE INSTANCES IN SERVICE ARENA_RUN"] * 3
        assert cluster.now == 4

    def test_times_out_at_the_startup_deadline(self, cluster: Cluster) -> None:
        cluster.instance_rows = [[]]

        with pytest.raises(TimeoutError, match="Instance 0 of ARENA_RUN had no IP"):
            wait_for_head_ip(_startup(cluster), "ARENA_RUN")

        assert cluster.now == 10

    def test_a_worker_stops_when_the_head_writes_done(self, cluster: Cluster) -> None:
        cluster.instance_rows = [[]]
        cluster.put_done(3)

        with pytest.raises(HeadFinishedError) as raised:
            wait_for_head_ip(_startup(cluster, RUN_PREFIX), "ARENA_RUN")

        assert raised.value.code == 3
        assert cluster.now == 0

    def test_an_unsafe_service_name_is_rejected(self, cluster: Cluster) -> None:
        with pytest.raises(ValueError, match="service"):
            wait_for_head_ip(_startup(cluster), "ARENA; DROP")


class TestRunNodeHead:
    def test_single_node_runs_uploads_and_writes_done(self, cluster: Cluster) -> None:
        # Arrange
        runtime = cluster.runtime(
            code=0, polls=1, on_poll=lambda: _checkpoint(cluster.ckpt_root, 1)
        )

        # Act
        code = run_node(["python", "-m", "trainer"], runtime)

        # Assert
        assert code == 0
        assert cluster.done() == 0
        assert cluster.stage_file("checkpoint_1/merged/.complete").is_file()
        assert cluster.commands[0][1]["RAY_ADDRESS"] == "auto"
        assert cluster.commands[0][1]["ARENA_RUN_ID"] == RUN_ID
        assert cluster.token_path.read_text(encoding="utf-8") == RUN_TOKEN
        assert cluster.events[0].startswith("ray ray start --head")
        assert cluster.events[-3:] == ["put .done", "ray stop", DROP_BROKER]
        assert cluster.closed == 5
        assert cluster.ray_commands[0][1]["APP_PULSAR_URL"] == PULSAR_URL
        assert cluster.commands[0][1]["APP_PULSAR_URL"] == PULSAR_URL
        ray_args = cluster.ray_commands[0][0]
        assert ray_args[ray_args.index("--num-cpus") + 1] == "0"
        assert ray_args[ray_args.index("--num-gpus") + 1] == "0"

    def test_object_store_takes_most_of_shm(self, cluster: Cluster) -> None:
        recorded: list[list[str]] = []
        runtime = cluster.runtime()
        runtime.start_ray = lambda args, env: recorded.append(args)

        run_node(["python"], runtime)

        assert recorded[0][recorded[0].index("--object-store-memory") + 1] == "800"

    def test_periodic_upload_while_the_trainer_runs(self, cluster: Cluster) -> None:
        def write_checkpoints() -> None:
            if not (cluster.ckpt_root / "checkpoint_1").exists():
                _checkpoint(cluster.ckpt_root, 1)

        runtime = cluster.runtime(polls=4, on_poll=write_checkpoints)

        run_node(["python"], runtime)

        assert cluster.events[cluster.events.index("command") :] == [
            "command",
            "put model.safetensors",
            "put .complete",
            "exit",
            "put .uploaded",
            "put .done",
            "ray stop",
            DROP_BROKER,
        ]

    def test_trainer_failure_code_reaches_done(self, cluster: Cluster) -> None:
        code = run_node(["python"], cluster.runtime(code=3))

        assert code == 3
        assert cluster.done() == 3

    def test_cluster_timeout_writes_a_failed_done(self, cluster: Cluster) -> None:
        runtime = cluster.runtime(alive_nodes=lambda: 1)

        with pytest.raises(TimeoutError, match="1 of 2 nodes after 10s"):
            run_node(["python"], runtime)

        assert cluster.done() == 1
        assert cluster.commands == []
        assert cluster.events[-2:] == ["ray stop", DROP_BROKER]

    def test_ray_start_failure_writes_a_failed_done_and_opens_no_row(
        self, cluster: Cluster, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        opened: list[datetime] = []
        monkeypatch.setattr(
            node_mod,
            "open_head_run",
            lambda job, env, started_at, sleep: opened.append(started_at),
        )
        runtime = cluster.runtime()

        def start_ray(args: list[str], env: Mapping[str, str]) -> None:
            msg = "ray start failed"
            raise RuntimeError(msg)

        runtime.start_ray = start_ray

        # Act
        with pytest.raises(RuntimeError, match="ray start failed"):
            run_node(["python"], runtime)

        # Assert
        assert opened == []
        assert cluster.done() == 1
        assert cluster.commands == []
        assert cluster.events == ["put .done", "ray stop", DROP_BROKER]

    def test_license_refusal_writes_a_failed_done_before_any_wait(
        self, cluster: Cluster, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        def refuse(env: Mapping[str, str]) -> None:
            raise node_mod.SnowflakeLicenseError(SNOWFLAKE_DENIED)

        monkeypatch.setattr(node_mod, "check_snowflake_license", refuse)

        # Act
        with pytest.raises(node_mod.SnowflakeLicenseError, match="not enabled"):
            run_node(["python"], cluster.runtime())

        # Assert
        assert cluster.done() == 1
        assert cluster.ray_commands == []
        assert not any(s.startswith("SHOW SERVICE") for s in cluster.statements)
        assert cluster.events == ["put .done", "ray stop", DROP_BROKER]

    def test_open_failure_writes_a_failed_done_and_never_trains(
        self, cluster: Cluster, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        completions = Completions(cluster)

        def refuse(
            job: object, env: object, started_at: datetime, sleep: object
        ) -> str:
            raise node_mod.SnowflakeLicenseError(OPEN_FAILED)

        monkeypatch.setattr(node_mod, "open_head_run", refuse)
        monkeypatch.setattr(node_mod, "complete_snowflake_run", completions)

        # Act
        with pytest.raises(node_mod.SnowflakeLicenseError, match="record"):
            run_node(["python"], cluster.runtime())

        # Assert
        assert cluster.done() == 1
        assert cluster.commands == []
        assert completions.calls == []
        assert cluster.events[-2:] == ["ray stop", DROP_BROKER]

    def test_token_write_failure_still_completes_the_open_row(
        self,
        cluster: Cluster,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        # Arrange
        completions = Completions(cluster)
        monkeypatch.setattr(node_mod, "complete_snowflake_run", completions)
        blocker = tmp_path / "not-a-directory"
        blocker.write_text("", encoding="utf-8")
        runtime = cluster.runtime()
        runtime.token_path = blocker / "run_token"

        # Act
        with pytest.raises(OSError, match="not-a-directory"):
            run_node(["python"], runtime)

        # Assert
        assert cluster.commands == []
        assert cluster.done() == 1
        assert completions.calls == [(RUN_ID, 300.0, 30.0)]
        assert cluster.events[-1] == DROP_BROKER

    def test_row_opens_at_the_earliest_worker_start(
        self, cluster: Cluster, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        cluster.write_job(nodes=2)
        cluster.put_marker(".started-10.0.0.6", "2026-11-11T17:59:30+00:00")
        cluster.put_marker(".uploaded-10.0.0.6", "")
        opened: list[datetime] = []

        def record_open(
            job: object, env: object, started_at: datetime, sleep: object
        ) -> str:
            opened.append(started_at)
            return RUN_TOKEN

        monkeypatch.setattr(node_mod, "open_head_run", record_open)

        # Act
        code = run_node(["python"], cluster.runtime(alive_nodes=lambda: 3))

        # Assert
        assert code == 0
        assert opened == [datetime(2026, 11, 11, 17, 59, 30, tzinfo=timezone.utc)]

    def test_heartbeat_runs_until_complete_returns(
        self, cluster: Cluster, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        completions = Completions(cluster)
        monkeypatch.setattr(node_mod, "complete_snowflake_run", completions)

        # Act
        code = run_node(["python"], cluster.runtime())

        # Assert
        assert code == 0
        assert completions.calls == [(RUN_ID, 300.0, 30.0)]
        assert completions.heartbeat_alive == [True]
        assert cluster.events[-3:] == ["ray stop", "complete", DROP_BROKER]

    def test_complete_waits_for_every_worker_upload(
        self, cluster: Cluster, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        cluster.stage_file(f".uploaded-{WORKER_IP}").unlink()
        seen: dict[str, object] = {}

        def complete(
            run_id: str,
            env: object,
            sleep: object,
            retry_budget_s: float,
            timeout_s: float,
        ) -> None:
            seen["done"] = cluster.done()
            seen["now"] = cluster.now

        monkeypatch.setattr(node_mod, "complete_snowflake_run", complete)

        def sleep(seconds: float) -> None:
            cluster.now += seconds
            if cluster.now >= 4:
                cluster.put_marker(f".uploaded-{WORKER_IP}", "")

        runtime = cluster.runtime()
        runtime.sleep = sleep

        # Act
        code = run_node(["python"], runtime)

        # Assert
        assert code == 0
        assert seen == {"done": 0, "now": 4}

    def test_missing_upload_markers_stop_waiting_at_the_deadline(
        self, cluster: Cluster, caplog: pytest.LogCaptureFixture
    ) -> None:
        cluster.stage_file(f".uploaded-{WORKER_IP}").unlink()

        code = run_node(["python"], cluster.runtime())

        assert code == 0
        assert cluster.now == 6
        assert "0 of 1 workers wrote .uploaded-*" in caplog.text
        assert cluster.events[-1] == DROP_BROKER

    def test_failed_complete_is_logged_and_the_run_still_succeeds(
        self,
        cluster: Cluster,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Arrange
        def fail(
            run_id: str,
            env: object,
            sleep: object,
            retry_budget_s: float,
            timeout_s: float,
        ) -> None:
            raise node_mod.SnowflakeLicenseError(COMPLETE_FAILED)

        monkeypatch.setattr(node_mod, "complete_snowflake_run", fail)

        # Act
        code = run_node(["python"], cluster.runtime())

        # Assert
        assert code == 0
        assert cluster.done() == 0
        assert COMPLETE_FAILED in caplog.text
        assert "last heartbeat" in caplog.text
        assert cluster.events[-1] == DROP_BROKER

    def test_sigterm_while_training_completes_first_with_one_short_attempt(
        self, cluster: Cluster, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        completions = Completions(cluster)
        monkeypatch.setattr(node_mod, "complete_snowflake_run", completions)

        def terminated() -> None:
            exit_on_sigterm(15, None)

        runtime = cluster.runtime(polls=1, on_poll=terminated)

        # Act
        with pytest.raises(SystemExit) as raised:
            run_node(["python"], runtime)

        # Assert
        assert raised.value.code == SIGTERM_EXIT_CODE
        assert cluster.done() == 1
        assert completions.calls == [(RUN_ID, 0.0, 10.0)]
        assert cluster.events[-4:] == [
            "complete",
            "put .done",
            "ray stop",
            DROP_BROKER,
        ]

    def test_sigterm_while_waiting_for_uploads_completes_once_quickly(
        self, cluster: Cluster, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        completions = Completions(cluster)
        monkeypatch.setattr(node_mod, "complete_snowflake_run", completions)
        cluster.stage_file(f".uploaded-{WORKER_IP}").unlink()
        runtime = cluster.runtime()

        def sleep(seconds: float) -> None:
            if cluster.done() is not None:
                exit_on_sigterm(15, None)

        runtime.sleep = sleep

        # Act
        with pytest.raises(SystemExit):
            run_node(["python"], runtime)

        # Assert
        assert completions.calls == [(RUN_ID, 0.0, 10.0)]
        assert cluster.events[-2:] == ["complete", DROP_BROKER]

    def test_final_upload_failure_writes_a_failed_done(self, cluster: Cluster) -> None:
        _checkpoint(cluster.ckpt_root, 1)

        class Broken(Session):
            def put(
                self,
                local: str,
                stage: str,
                auto_compress: bool = True,
                overwrite: bool = False,
            ) -> None:
                if Path(local).name != ".done":
                    msg = "stage unavailable"
                    raise OSError(msg)
                super().put(local, stage, auto_compress, overwrite)

        runtime = cluster.runtime()
        runtime.open_session = lambda: Broken(cluster)

        with pytest.raises(OSError, match="stage unavailable"):
            run_node(["python"], runtime)

        assert cluster.done() == 1

    def test_trainer_is_killed_when_a_periodic_upload_raises(
        self, cluster: Cluster
    ) -> None:
        # Arrange
        _checkpoint(cluster.ckpt_root, 1)
        _checkpoint(cluster.ckpt_root, 2)
        processes: list[Process] = []

        class Rejects(Session):
            def put(
                self,
                local: str,
                stage: str,
                auto_compress: bool = True,
                overwrite: bool = False,
            ) -> None:
                if Path(local).name != ".done":
                    msg = "stage name rejected"
                    raise ValueError(msg)
                super().put(local, stage, auto_compress, overwrite)

        runtime = cluster.runtime(polls=10)
        runtime.open_session = lambda: Rejects(cluster)
        start_command = runtime.start_command

        def recording(command: list[str], env: Mapping[str, str]) -> Process:
            processes.append(start_command(command, env))
            return processes[-1]

        runtime.start_command = recording

        # Act
        with pytest.raises(ValueError, match="stage name rejected"):
            run_node(["python"], runtime)

        # Assert
        assert processes[0].killed is True
        assert cluster.done() == 1
        assert cluster.events[-2:] == ["ray stop", DROP_BROKER]

    def test_a_failed_done_put_still_stops_ray(self, cluster: Cluster) -> None:
        class Broken(Session):
            def put(
                self,
                local: str,
                stage: str,
                auto_compress: bool = True,
                overwrite: bool = False,
            ) -> None:
                msg = "stage unavailable"
                raise OSError(msg)

        runtime = cluster.runtime()
        runtime.open_session = lambda: Broken(cluster)

        with pytest.raises(OSError, match="stage unavailable"):
            run_node(["python"], runtime)

        assert cluster.events[-2:] == ["ray stop", DROP_BROKER]
        assert cluster.closed == 3

    def test_job_index_does_not_select_the_head(self, cluster: Cluster) -> None:
        code = run_node(
            ["python"],
            cluster.runtime(
                {"SNOWFLAKE_SERVICE_NAME": SERVICES.head, "SNOWFLAKE_JOB_INDEX": "4"}
            ),
        )

        assert code == 0
        assert cluster.events[0].startswith("ray ray start --head")

    def test_head_waits_until_workers_and_itself_are_alive(
        self, cluster: Cluster
    ) -> None:
        cluster.write_job(nodes=2)
        cluster.put_marker(".started-10.0.0.6", WORKER_STARTED)
        cluster.put_marker(".uploaded-10.0.0.6", "")
        seen = {"n": 0}

        def alive() -> int:
            seen["n"] += 1
            return 2 if seen["n"] < 3 else 3

        run_node(["python"], cluster.runtime(alive_nodes=alive))

        assert seen["n"] == 3

    def test_broker_ip_lookup_uses_the_job_service(self, cluster: Cluster) -> None:
        cluster.instance_rows = [
            [Row(instance_id=1, ip_address="10.0.0.9")],
            [Row(instance_id=0, ip_address="10.0.0.4")],
        ]

        run_node(["python"], cluster.runtime())

        show = f"SHOW SERVICE INSTANCES IN SERVICE {SERVICES.broker}"
        assert cluster.statements[:2] == [show, show]
        assert cluster.ray_commands[0][1]["APP_PULSAR_URL"] == "pulsar://10.0.0.4:6650"

    def test_broker_ip_timeout_does_not_start_ray(self, cluster: Cluster) -> None:
        cluster.instance_rows = [[]]

        with pytest.raises(TimeoutError, match="had no IP"):
            run_node(["python"], cluster.runtime())

        assert cluster.ray_commands == []
        assert cluster.statements
        assert set(cluster.statements) == {
            f"SHOW SERVICE INSTANCES IN SERVICE {SERVICES.broker}",
            DROP_BROKER,
        }
        assert cluster.done() == 1

    def test_broker_health_shares_the_startup_deadline(self, cluster: Cluster) -> None:
        cluster.instance_rows = [
            [Row(instance_id=0, ip_address=None)],
            [Row(instance_id=0, ip_address="10.4.0.1")],
        ]
        runtime = cluster.runtime()
        runtime.broker_healthy = lambda ip: False

        with pytest.raises(
            TimeoutError, match=r"Broker at 10\.4\.0\.1 did not become healthy"
        ):
            run_node(["python"], runtime)

        assert cluster.now == runtime.startup_timeout_s
        assert cluster.ray_commands == []

    def test_pulsar_url_is_on_ray_and_the_trainer_before_process_env(
        self, cluster: Cluster, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("APP_PULSAR_URL", raising=False)

        run_node(["python"], cluster.runtime())

        assert "APP_PULSAR_URL" not in os.environ
        assert cluster.ray_commands[0][1]["APP_PULSAR_URL"] == PULSAR_URL
        assert cluster.commands[0][1]["APP_PULSAR_URL"] == PULSAR_URL
        ray_at = next(
            index
            for index, event in enumerate(cluster.events)
            if event.startswith("ray ray start --head")
        )
        assert ray_at < cluster.events.index("command")

    def test_process_environment_is_set_before_ray_start(
        self, cluster: Cluster, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("APP_PULSAR_URL", raising=False)
        monkeypatch.setenv("SNOWFLAKE_SERVICE_NAME", SERVICES.head)
        seen: dict[str, str] = {}
        runtime = cluster.runtime(environ=os.environ)

        def start_ray(args: list[str], env: Mapping[str, str]) -> None:
            seen["ray"] = os.environ["APP_PULSAR_URL"]
            seen["passed"] = env["APP_PULSAR_URL"]
            cluster.events.append("ray ray start --head")

        runtime.start_ray = start_ray
        try:
            run_node(["python"], runtime)
        finally:
            os.environ.pop("APP_PULSAR_URL", None)

        assert seen == {"ray": PULSAR_URL, "passed": PULSAR_URL}
        assert cluster.commands[0][1]["APP_PULSAR_URL"] == PULSAR_URL


class TestRunNodeWorker:
    def test_waits_for_the_head_gcs_before_ray_start(self, cluster: Cluster) -> None:
        # Arrange
        cluster.instance_rows = [[Row(instance_id=0, ip_address="10.0.0.8")]]
        opened = iter([False, False, True])

        def gcs_open(host: str) -> bool:
            result = next(opened, None)
            if result is None:
                cluster.put_done(0)
                return False
            return result

        # Act
        code = run_node(["python"], cluster.runtime(WORKER_ENV, gcs_open=gcs_open))

        # Assert
        assert code == 0
        ray_start = cluster.events.index("ray ray start --address 10.0.0.8:61001")
        assert cluster.events[:ray_start] == [
            "put .started-10.0.0.4",
            "gcs 10.0.0.8 False",
            "gcs 10.0.0.8 False",
            "gcs 10.0.0.8 True",
        ]
        assert cluster.commands == []

    def test_started_marker_is_the_boot_time_before_any_wait(
        self, cluster: Cluster
    ) -> None:
        # Arrange
        cluster.instance_rows = [[Row(instance_id=0, ip_address="10.0.0.8")]]
        cluster.put_done(0)
        before = datetime.now(timezone.utc)

        # Act
        run_node(["python"], cluster.runtime(WORKER_ENV, gcs_open=_open_then_closed(1)))

        # Assert
        started = datetime.fromisoformat(
            cluster.stage_file(".started-10.0.0.4").read_text(encoding="utf-8")
        )
        assert before <= started <= datetime.now(timezone.utc)
        assert cluster.events[0] == "put .started-10.0.0.4"
        assert cluster.statements[0] == (
            f"SHOW SERVICE INSTANCES IN SERVICE {SERVICES.broker}"
        )

    def test_head_done_during_startup_exits_without_ray(self, cluster: Cluster) -> None:
        # Arrange
        cluster.instance_rows = [[Row(instance_id=0, ip_address="10.0.0.8")]]
        cluster.put_done(7)

        # Act
        code = run_node(
            ["python"], cluster.runtime(WORKER_ENV, gcs_open=lambda host: False)
        )

        # Assert
        assert code == 7
        assert cluster.ray_commands == []
        assert cluster.now == 0
        assert cluster.closed == 1

    def test_uploaded_marker_follows_the_final_upload(self, cluster: Cluster) -> None:
        # Arrange
        cluster.instance_rows = [[Row(instance_id=0, ip_address="10.0.0.8")]]
        _checkpoint(cluster.ckpt_root, 2)
        cluster.put_done(0)

        # Act
        run_node(["python"], cluster.runtime(WORKER_ENV, gcs_open=_open_then_closed(1)))

        # Assert
        assert cluster.stage_file(".uploaded-10.0.0.4").is_file()
        assert cluster.events[-4:] == [
            "put .complete",
            "put .uploaded",
            "put .uploaded-10.0.0.4",
            "ray stop",
        ]

    def test_head_gcs_never_opening_times_out(self, cluster: Cluster) -> None:
        cluster.instance_rows = [[Row(instance_id=0, ip_address="10.0.0.8")]]

        with pytest.raises(TimeoutError, match="did not open after 10s"):
            run_node(
                ["python"], cluster.runtime(WORKER_ENV, gcs_open=lambda host: False)
            )

        assert not any(event.startswith("ray ") for event in cluster.events)
        assert cluster.statements.count(f"LIST {RUN_PREFIX}/.done") == 6

    def test_success_done_uploads_the_worker_checkpoint(self, cluster: Cluster) -> None:
        # Arrange
        cluster.instance_rows = [[Row(instance_id=0, ip_address="10.0.0.8")]]
        _checkpoint(cluster.ckpt_root, 2)
        cluster.put_done(0)

        # Act
        code = run_node(
            ["python"], cluster.runtime(WORKER_ENV, gcs_open=_open_then_closed(1))
        )

        # Assert
        assert code == 0
        assert cluster.stage_file("checkpoint_2/merged/model.safetensors").is_file()
        assert cluster.stage_file("checkpoint_2/merged/.complete").is_file()
        assert cluster.events[-1] == "ray stop"

    def test_done_is_read_only_after_the_head_gcs_closes(
        self, cluster: Cluster
    ) -> None:
        cluster.instance_rows = [[Row(instance_id=0, ip_address="10.0.0.8")]]
        cluster.put_done(5)

        code = run_node(
            ["python"], cluster.runtime(WORKER_ENV, gcs_open=_open_then_closed(3))
        )

        assert code == 5
        assert cluster.statements.count(f"LIST {RUN_PREFIX}/.done") == 1

    def test_head_lost_without_done_raises_after_the_final_upload(
        self, cluster: Cluster
    ) -> None:
        # Arrange
        cluster.instance_rows = [[Row(instance_id=0, ip_address="10.0.0.8")]]
        _checkpoint(cluster.ckpt_root, 1)

        # Act
        with pytest.raises(RuntimeError, match=r"wrote no \.done"):
            run_node(
                ["python"],
                cluster.runtime(WORKER_ENV, gcs_open=_open_then_closed(2)),
            )

        # Assert
        assert cluster.stage_file("checkpoint_1/merged/.complete").is_file()
        assert cluster.events[-1] == "ray stop"
        assert cluster.now >= 6

    def test_done_written_while_the_head_closes_is_read(self, cluster: Cluster) -> None:
        cluster.instance_rows = [[Row(instance_id=0, ip_address="10.0.0.8")]]
        calls = {"n": 0}

        def gcs_open(host: str) -> bool:
            calls["n"] += 1
            if calls["n"] == 5:
                cluster.put_done(2)
            return calls["n"] == 1

        assert run_node(["python"], cluster.runtime(WORKER_ENV, gcs_open=gcs_open)) == 2

    def test_joins_the_head_service_not_its_own_instance_zero(
        self, cluster: Cluster
    ) -> None:
        cluster.instances = {
            SERVICES.broker: [[Row(instance_id=0, ip_address="10.9.0.1")]],
            SERVICES.head: [[Row(instance_id=0, ip_address="10.9.0.8")]],
            SERVICES.workers: [[Row(instance_id=0, ip_address="10.9.0.5")]],
        }
        cluster.put_done(0)

        code = run_node(
            ["python"],
            cluster.runtime(
                {
                    "SNOWFLAKE_SERVICE_NAME": SERVICES.workers,
                    "SNOWFLAKE_JOB_INDEX": "0",
                },
                gcs_open=_open_then_closed(1),
            ),
        )

        assert code == 0
        assert cluster.ray_commands[0][0][:4] == [
            "ray",
            "start",
            "--address",
            "10.9.0.8:61001",
        ]
        assert cluster.ray_commands[0][1]["APP_PULSAR_URL"] == "pulsar://10.9.0.1:6650"
        head_lookup = f"SHOW SERVICE INSTANCES IN SERVICE {SERVICES.head}"
        worker_lookup = f"SHOW SERVICE INSTANCES IN SERVICE {SERVICES.workers}"
        assert head_lookup in cluster.statements
        assert worker_lookup not in cluster.statements


class TestRunNodeValidation:
    def test_empty_command_is_rejected(self, cluster: Cluster) -> None:
        with pytest.raises(ValueError, match="empty"):
            run_node([], cluster.runtime())

    def test_missing_service_name_is_rejected(self, cluster: Cluster) -> None:
        with pytest.raises(KeyError, match="SNOWFLAKE_SERVICE_NAME"):
            run_node(["python"], cluster.runtime({}))

        assert cluster.events == []

    @pytest.mark.parametrize("service", ["ARENA_RUN", ""])
    def test_service_without_a_role_suffix_is_rejected(
        self, cluster: Cluster, service: str
    ) -> None:
        with pytest.raises(ValueError, match="service") as raised:
            run_node(
                ["python"],
                cluster.runtime({"SNOWFLAKE_SERVICE_NAME": service}),
            )

        assert repr(service) in str(raised.value)
        assert cluster.events == []


class TestDoneFile:
    def test_missing_file_is_none(self, cluster: Cluster) -> None:
        assert read_done(Session(cluster), RUN_PREFIX) is None

    def test_written_code_is_read_back(self, cluster: Cluster) -> None:
        write_done(Session(cluster), RUN_PREFIX, 7)

        assert read_done(Session(cluster), RUN_PREFIX) == 7
        assert cluster.stage_file(".done").read_text(encoding="utf-8") == "7\n"


class _HttpResponse:
    def __init__(
        self,
        status_code: int,
        payload: object,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self.headers = {} if headers is None else headers
        self.is_success = 200 <= status_code < 300

    def json(self) -> object:
        return self._payload


def _enabled_body(enabled: bool = True) -> dict[str, object]:
    return {
        "ok": True,
        "data": {
            "schemaVersion": 1,
            "features": {"snowflakeTraining": enabled},
        },
    }


def _token_body(token: str = RUN_TOKEN) -> dict[str, object]:
    return {"ok": True, "data": {"runToken": token}}


def _head_env() -> dict[str, str]:
    return {
        "SNOWFLAKE_SERVICE_NAME": SERVICES.head,
        "SNOWFLAKE_ACCOUNT": "ORG-ACCOUNT",
        "ARENA_ORG_KEY": "arena_org_secret",
        "ARENA_EXTERNAL_USER_ID": "user-1",
        "ARENA_BASE_URL": "https://arena.example.com",
    }


class ArenaCalls:
    """Arena stand-in: one scripted answer per route, recorded in call order."""

    def __init__(
        self,
        cluster: Cluster,
        capabilities: object,
        opened: object,
        completed: object,
    ) -> None:
        self.cluster = cluster
        self.answers = {
            "capabilities": capabilities,
            "open": opened,
            "complete": completed,
        }
        self.order: list[str] = []
        self.posts: list[dict[str, object]] = []
        self.at_complete: dict[str, object] = {}

    def answer(self, route: str) -> object:
        self.order.append(route)
        if route == "complete":
            self.at_complete = {
                "done": self.cluster.done(),
                "checkpoint": self.cluster.stage_file(
                    "checkpoint_1/merged/.complete"
                ).is_file(),
            }
        result = self.answers[route]
        if isinstance(result, Exception):
            raise result
        return result

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = self

        class Client:
            def __enter__(self) -> Self:
                return self

            def __exit__(self, *args: object) -> bool:
                return False

            def get(
                self, url: str, headers: object = None, timeout: object = None
            ) -> object:
                return calls.answer("capabilities")

            def post(
                self,
                url: str,
                headers: object = None,
                json: object = None,
                timeout: object = None,
            ) -> object:
                calls.posts.append({"url": url, "json": json})
                route = "complete" if url.endswith("/complete") else "open"
                return calls.answer(route)

        monkeypatch.setattr("agilerl.arena.snowflake.license.httpx.Client", Client)


class TestHeadLicense:
    def _runtime(self, cluster: Cluster, order: list[str]) -> NodeRuntime:
        runtime = cluster.runtime(
            environ=_head_env(),
            code=0,
            polls=1,
            on_poll=lambda: _checkpoint(cluster.ckpt_root, 1),
        )
        start_ray = runtime.start_ray

        def recording(args: list[str], env: Mapping[str, str]) -> None:
            order.append("ray")
            start_ray(args, env)

        runtime.start_ray = recording
        return runtime

    @pytest.mark.parametrize(
        ("capabilities", "match"),
        [
            (
                httpx.ConnectError("refused"),
                "Could not reach Arena at https://arena.example.com: refused",
            ),
            (_HttpResponse(401, {}), "rejected the org key or external user id"),
            (
                _HttpResponse(200, {"ok": False}),
                (
                    "https://arena.example.com answered HTTP 200 with no content "
                    "type, which is not the Arena API"
                ),
            ),
            (_HttpResponse(200, _enabled_body(False)), "Ask AgileRL to enable"),
        ],
    )
    def test_a_refused_check_writes_a_failed_done_before_ray(
        self,
        cluster: Cluster,
        monkeypatch: pytest.MonkeyPatch,
        capabilities: object,
        match: str,
    ) -> None:
        # Arrange
        arena = ArenaCalls(cluster, capabilities, None, None)
        arena.install(monkeypatch)
        runtime = self._runtime(cluster, arena.order)

        # Act
        with pytest.raises(node_mod.SnowflakeLicenseError, match=match):
            run_node(["python"], runtime)

        # Assert
        assert arena.order == ["capabilities"]
        assert cluster.done() == 1
        assert cluster.ray_commands == []

    def test_open_retries_then_fails_without_training(
        self, cluster: Cluster, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        arena = ArenaCalls(
            cluster,
            _HttpResponse(200, _enabled_body()),
            _HttpResponse(503, {"ok": False}),
            _HttpResponse(200, {"ok": True}),
        )
        arena.install(monkeypatch)
        runtime = self._runtime(cluster, arena.order)

        # Act
        with pytest.raises(
            node_mod.SnowflakeLicenseError, match="record the Snowflake run"
        ):
            run_node(["python"], runtime)

        # Assert
        assert arena.order[:3] == ["capabilities", "ray", "open"]
        assert set(arena.order[2:]) == {"open"}
        assert len(arena.order) > 3
        assert cluster.done() == 1
        assert cluster.commands == []

    def test_open_after_nodes_join_then_complete_after_done(
        self, cluster: Cluster, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        arena = ArenaCalls(
            cluster,
            _HttpResponse(200, _enabled_body()),
            _HttpResponse(200, _token_body()),
            _HttpResponse(200, {"ok": True}),
        )
        arena.install(monkeypatch)
        runtime = self._runtime(cluster, arena.order)

        # Act
        code = run_node(["python"], runtime)

        # Assert
        assert code == 0
        assert arena.order == ["capabilities", "ray", "open", "complete"]
        opened = arena.posts[0]["json"]
        finished = arena.posts[1]["json"]
        assert opened == {
            "runId": RUN_ID,
            "snowflakeAccount": "ORG-ACCOUNT",
            "gpuCount": 4,
            "gpuFamily": "GPU_NV_M",
            "startedAt": WORKER_STARTED,
        }
        assert isinstance(finished, dict)
        assert set(finished) == {"endedAt"}
        assert str(arena.posts[1]["url"]).endswith(
            f"/api/cli/v1/snowflake/runs/{RUN_ID}/complete"
        )
        assert arena.at_complete == {"done": 0, "checkpoint": True}
        assert cluster.token_path.read_text(encoding="utf-8") == RUN_TOKEN
        assert cluster.events[-1] == DROP_BROKER

    def test_flag_off_at_open_asks_agilerl_to_enable_it(
        self, cluster: Cluster, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        flag_off = _HttpResponse(
            403,
            {
                "error": "forbidden",
                "description": "Snowflake training is not enabled for this organization",
            },
        )
        arena = ArenaCalls(cluster, _HttpResponse(200, _enabled_body()), flag_off, None)
        arena.install(monkeypatch)

        # Act
        with pytest.raises(node_mod.SnowflakeLicenseError) as raised:
            run_node(["python"], self._runtime(cluster, arena.order))

        # Assert
        assert str(raised.value) == SNOWFLAKE_DENIED
        assert arena.order == ["capabilities", "ray", "open"]
        assert cluster.done() == 1
        assert cluster.commands == []

    def test_a_rejected_complete_still_succeeds(
        self,
        cluster: Cluster,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Arrange
        arena = ArenaCalls(
            cluster,
            _HttpResponse(200, _enabled_body()),
            _HttpResponse(200, _token_body()),
            _HttpResponse(403, {"ok": False}),
        )
        arena.install(monkeypatch)
        runtime = self._runtime(cluster, arena.order)

        # Act
        code = run_node(["python"], runtime)

        # Assert
        assert code == 0
        assert cluster.done() == 0
        assert arena.order[-1] == "complete"
        assert "rejected the org key" in caplog.text
        assert cluster.stage_file("checkpoint_1/merged/model.safetensors").is_file()


class StopAfter(threading.Event):
    """Stop event whose ``wait`` times out ``beats`` times, then reports set."""

    def __init__(self, beats: int) -> None:
        super().__init__()
        self.beats = beats
        self.waits: list[float] = []

    def wait(self, timeout: float | None = None) -> bool:
        self.waits.append(0.0 if timeout is None else timeout)
        if self.beats == 0:
            return True
        self.beats -= 1
        return False


class TestRunHeartbeat:
    def _heartbeat(self, tmp_path: Path, beats: int) -> RunHeartbeat:
        return RunHeartbeat(
            run_id=RUN_ID,
            env=_head_env(),
            token_path=tmp_path / "arena" / "run_token",
            interval_s=300,
            sleep=lambda seconds: None,
            stop=StopAfter(beats),
        )

    def test_each_beat_rewrites_the_token(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        tokens = iter(["v1.k.first.sig", "v1.k.second.sig"])
        seen: list[str] = []

        def heartbeat(run_id: str, env: Mapping[str, str], sleep: object) -> str:
            seen.append(run_id)
            return next(tokens)

        monkeypatch.setattr(node_mod, "heartbeat_snowflake_run", heartbeat)
        beat = self._heartbeat(tmp_path, beats=2)

        # Act
        beat.run()

        # Assert
        assert seen == [RUN_ID, RUN_ID]
        assert beat.token_path.read_text(encoding="utf-8") == "v1.k.second.sig"
        assert beat.stop.waits == [300, 300, 300]

    def test_a_refusal_stops_renewing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Arrange
        calls: list[str] = []

        def refuse(run_id: str, env: Mapping[str, str], sleep: object) -> str:
            calls.append(run_id)
            msg = "Arena will not renew this Snowflake run. Arena answered 409."
            raise RunRenewalRefusedError(msg)

        monkeypatch.setattr(node_mod, "heartbeat_snowflake_run", refuse)
        beat = self._heartbeat(tmp_path, beats=3)

        # Act
        beat.run()

        # Assert
        assert calls == [RUN_ID]
        assert not beat.token_path.exists()
        assert "token expires" in caplog.text

    def test_a_failed_beat_is_logged_and_the_next_one_still_runs(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Arrange
        answers: list[object] = [
            node_mod.SnowflakeLicenseError("Could not renew the Snowflake run"),
            "v1.k.later.sig",
        ]

        def heartbeat(run_id: str, env: Mapping[str, str], sleep: object) -> str:
            answer = answers.pop(0)
            if isinstance(answer, Exception):
                raise answer
            return str(answer)

        monkeypatch.setattr(node_mod, "heartbeat_snowflake_run", heartbeat)
        beat = self._heartbeat(tmp_path, beats=2)

        # Act
        beat.run()

        # Assert
        assert beat.token_path.read_text(encoding="utf-8") == "v1.k.later.sig"
        assert "Could not renew" in caplog.text

    def test_a_failed_token_write_is_logged_and_the_next_beat_still_runs(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Arrange
        tokens = iter(["v1.k.first.sig", "v1.k.second.sig"])
        monkeypatch.setattr(
            node_mod, "heartbeat_snowflake_run", lambda run_id, env, sleep: next(tokens)
        )
        real_write = node_mod.write_run_token
        writes: list[str] = []

        def write_once_failing(path: Path, token: str) -> None:
            writes.append(token)
            if len(writes) == 1:
                msg = "disk full"
                raise OSError(msg)
            real_write(path, token)

        monkeypatch.setattr(node_mod, "write_run_token", write_once_failing)
        beat = self._heartbeat(tmp_path, beats=2)

        # Act
        beat.run()

        # Assert
        assert writes == ["v1.k.first.sig", "v1.k.second.sig"]
        assert beat.token_path.read_text(encoding="utf-8") == "v1.k.second.sig"
        assert "disk full" in caplog.text


class TestWriteRunToken:
    def test_replaces_the_file_without_a_partial_left(self, tmp_path: Path) -> None:
        path = tmp_path / "arena" / "run_token"

        write_run_token(path, "v1.k.one.sig")
        write_run_token(path, "v1.k.two.sig")

        assert path.read_text(encoding="utf-8") == "v1.k.two.sig"
        assert sorted(child.name for child in path.parent.iterdir()) == ["run_token"]

    def test_head_writes_where_the_trainer_reads(self) -> None:
        trainer_path = Path(tempfile.gettempdir()) / "arena" / "run_token"

        runtime_default = next(
            item.default
            for item in dataclasses.fields(NodeRuntime)
            if item.name == "token_path"
        )

        assert node_mod.RUN_TOKEN_PATH == trainer_path
        assert runtime_default == trainer_path


class TestEarliestWorkerStart:
    def test_earliest_marker_wins(self, cluster: Cluster) -> None:
        cluster.put_marker(".started-10.0.0.6", "2026-11-11T17:00:00+00:00")

        started = earliest_worker_start(Session(cluster), RUN_PREFIX, 2)

        assert started == datetime(2026, 11, 11, 17, tzinfo=timezone.utc)

    def test_a_missing_worker_marker_raises(self, cluster: Cluster) -> None:
        with pytest.raises(RuntimeError, match="Only 1 of 2 workers wrote"):
            earliest_worker_start(Session(cluster), RUN_PREFIX, 2)


class TestMain:
    def test_usage(self) -> None:
        with pytest.raises(SystemExit, match="usage:"):
            main([])

    def test_installs_the_sigterm_handler_and_delegates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        installed: dict[int, object] = {}
        monkeypatch.setattr(
            node_mod.signal,
            "signal",
            lambda signum, handler: installed.setdefault(signum, handler),
        )
        monkeypatch.setattr(node_mod, "run_node", lambda command, runtime=None: 4)

        # Act
        code = main(["python", "-c", "print(1)"])

        # Assert
        assert code == 4
        assert installed == {signal.SIGTERM: exit_on_sigterm}

    def test_sigterm_raises_system_exit_143(self) -> None:
        with pytest.raises(SystemExit) as raised:
            exit_on_sigterm(signal.SIGTERM, None)

        assert raised.value.code == 143

    def test_module_entrypoint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["agilerl.arena.snowflake.node"])

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*found in sys.modules.*",
                category=RuntimeWarning,
            )
            with pytest.raises(SystemExit, match="usage:"):
                runpy.run_module("agilerl.arena.snowflake.node", run_name="__main__")


class TestProductionRuntime:
    @pytest.mark.skipif(sys.platform == "win32", reason="statvfs is POSIX-only")
    def test_shared_memory_bytes(self, tmp_path: Path) -> None:
        assert shared_memory_bytes(str(tmp_path)) > 0

    def test_helpers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Arrange
        errors = types.ModuleType("snowflake.connector.errors")
        errors.InterfaceError = ConnectionError
        errors.OperationalError = TimeoutError
        monkeypatch.setitem(sys.modules, "snowflake", types.ModuleType("snowflake"))
        monkeypatch.setitem(
            sys.modules, "snowflake.connector", types.ModuleType("snowflake.connector")
        )
        monkeypatch.setitem(sys.modules, "snowflake.connector.errors", errors)
        calls: list[tuple[list[str], bool, Mapping[str, str] | None]] = []

        def fake_run(
            args: list[str],
            check: bool = False,
            env: Mapping[str, str] | None = None,
        ) -> None:
            calls.append((args, check, None if env is None else dict(env)))

        monkeypatch.setattr(node_mod.subprocess, "run", fake_run)

        # Act
        runtime = production_runtime()
        node_mod._start_ray(["ray", "start", "--head"], os.environ)
        node_mod._stop_ray()
        node_mod._start_ray(
            ["ray", "start"], {"APP_PULSAR_URL": "pulsar://10.0.0.1:6650"}
        )
        runtime.sleep(0)

        # Assert
        assert runtime.node_ip()
        assert runtime.monotonic() >= 0
        assert runtime.retry_on == (ConnectionError, TimeoutError, OSError)
        assert runtime.ray_nodes is ray_nodes
        assert runtime.broker_healthy is node_mod.broker_is_healthy
        assert calls == [
            (["ray", "start", "--head"], True, None),
            (["ray", "stop"], False, None),
            (["ray", "start"], True, {"APP_PULSAR_URL": "pulsar://10.0.0.1:6650"}),
        ]

    def test_ray_nodes_counts_alive_nodes_on_one_driver(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        events: list[str] = []
        ray = types.ModuleType("ray")
        ray.init = lambda address, log_to_driver: events.append(f"init {address}")
        ray.nodes = lambda: [{"Alive": True}, {"Alive": False}, {"Alive": True}]
        ray.shutdown = lambda: events.append("shutdown")
        monkeypatch.setitem(sys.modules, "ray", ray)

        # Act
        with ray_nodes() as alive_nodes:
            counts = [alive_nodes(), alive_nodes()]

        # Assert
        assert counts == [2, 2]
        assert events == ["init auto", "shutdown"]

    def test_start_command_runs_a_child(self) -> None:
        process = node_mod._start_command(
            [sys.executable, "-c", "import os, sys; sys.exit(int(os.environ['CODE']))"],
            {**os.environ, "CODE": "6"},
        )

        assert process.wait() == 6

    def test_broker_health_reads_the_admin_endpoint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class Response:
            def __init__(self, status: int) -> None:
                self.status = status

            def __enter__(self) -> Self:
                return self

            def __exit__(self, *args: object) -> None:
                return None

        def urlopen(url: str, timeout: float) -> Response:
            assert url == "http://10.0.0.3:8080/admin/v2/brokers/health"
            assert timeout == 5
            return Response(200)

        monkeypatch.setattr(node_mod.urllib.request, "urlopen", urlopen)
        assert node_mod.broker_is_healthy("10.0.0.3") is True

        monkeypatch.setattr(
            node_mod.urllib.request,
            "urlopen",
            lambda url, timeout: Response(503),
        )
        assert node_mod.broker_is_healthy("10.0.0.3") is False

        def refused(url: str, timeout: float) -> Response:
            msg = "refused"
            raise OSError(msg)

        monkeypatch.setattr(node_mod.urllib.request, "urlopen", refused)
        assert node_mod.broker_is_healthy("10.0.0.3") is False

    def test_gcs_probe(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class Open:
            def settimeout(self, seconds: float) -> None:
                return None

            def connect(self, address: tuple[str, int]) -> None:
                return None

            def close(self) -> None:
                return None

        class Closed(Open):
            def connect(self, address: tuple[str, int]) -> None:
                msg = "refused"
                raise OSError(msg)

        monkeypatch.setattr(socket, "socket", lambda: Open())
        assert node_mod._gcs_open("10.0.0.1") is True
        monkeypatch.setattr(socket, "socket", lambda: Closed())
        assert node_mod._gcs_open("10.0.0.1") is False

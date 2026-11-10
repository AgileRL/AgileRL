# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Ray startup, checkpoint upload, and shutdown on each SPCS job instance."""

from __future__ import annotations

import json
import os
import runpy
import socket
import sys
import types
import warnings
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

import pytest

from agilerl.arena.snowflake import node as node_mod
from agilerl.arena.snowflake.node import (
    NodeRuntime,
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
)

RUN_ID = "arena-20261109163457-abc123"
RUN_PREFIX = f"@arena.runs/{RUN_ID}"
WORKER_ENV = {
    "SNOWFLAKE_JOBS_COUNT": "2",
    "SNOWFLAKE_JOB_INDEX": "1",
    "SNOWFLAKE_SERVICE_NAME": "ARENA_RUN",
}


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
            path = self.cluster.stage_root / text.removeprefix("LIST @")
            return Result([Row(name=path.name)] if path.is_file() else [])
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
        (self.run_dir / "job.json").write_text(
            json.dumps({"run_id": RUN_ID, "runs_stage": "arena.runs"}),
            encoding="utf-8",
        )
        self.ckpt_root = tmp_path / "ckpt"
        self.stage_root = tmp_path / "stage"
        self.instance_rows: list[list[Row]] = [[]]
        self.statements: list[str] = []
        self.events: list[str] = []
        self.closed = 0
        self.now = 0.0
        self.commands: list[tuple[list[str], Mapping[str, str]]] = []

    def stage_file(self, relative: str) -> Path:
        return self.stage_root / "arena.runs" / RUN_ID / relative

    def put_done(self, code: int) -> None:
        path = self.stage_file(".done")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{code}\n", encoding="utf-8")

    def done(self) -> int | None:
        return read_done(Session(self), RUN_PREFIX)

    def runtime(
        self,
        environ: Mapping[str, str] | None = None,
        code: int = 0,
        polls: int = 0,
        on_poll: Callable[[], None] = lambda: None,
        gcs_open: Callable[[str], bool] = lambda host: True,
        alive_nodes: Callable[[], int] = lambda: 1,
    ) -> NodeRuntime:
        def sleep(seconds: float) -> None:
            self.now += seconds

        @contextmanager
        def nodes() -> Iterator[Callable[[], int]]:
            yield alive_nodes

        def start_command(command: list[str], env: Mapping[str, str]) -> Process:
            self.commands.append((command, env))
            self.events.append("command")
            return Process(code, polls, on_poll, self.events)

        def gcs(host: str) -> bool:
            opened = gcs_open(host)
            self.events.append(f"gcs {host} {opened}")
            return opened

        return NodeRuntime(
            environ=environ
            or {"SNOWFLAKE_JOBS_COUNT": "1", "SNOWFLAKE_JOB_INDEX": "0"},
            node_ip=lambda: "10.0.0.4",
            open_session=lambda: Session(self),
            start_ray=lambda args: self.events.append(f"ray {' '.join(args[:4])}"),
            stop_ray=lambda: self.events.append("ray stop"),
            ray_nodes=nodes,
            gcs_open=gcs,
            start_command=start_command,
            sleep=sleep,
            monotonic=lambda: self.now,
            shared_memory_bytes=lambda: 1000,
            retry_on=(OSError,),
            run_dir=self.run_dir,
            ckpt_root=self.ckpt_root,
            head_timeout_s=10,
            cluster_timeout_s=10,
            head_gone_s=6,
            poll_s=2,
            upload_interval_s=4,
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
        assert worker[:4] == ["ray", "start", "--address", "10.0.0.4:61001"]
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


class TestWaitForHeadIp:
    def test_waits_for_instance_zero_on_one_session(self, cluster: Cluster) -> None:
        cluster.instance_rows = [
            [Row(instance_id=1, ip_address="10.0.0.9")],
            [
                Row(instance_id=0, ip_address=None),
                Row(instance_id=1, ip_address="10.0.0.9"),
            ],
            [Row(instance_id=0, ip_address="10.0.0.4")],
        ]

        ip = wait_for_head_ip(cluster.runtime(), "ARENA_RUN")

        assert ip == "10.0.0.4"
        assert cluster.statements == ["SHOW SERVICE INSTANCES IN SERVICE ARENA_RUN"] * 3
        assert cluster.closed == 1

    def test_times_out_and_closes_the_session(self, cluster: Cluster) -> None:
        with pytest.raises(TimeoutError, match="Instance 0 had no IP"):
            wait_for_head_ip(cluster.runtime(), "ARENA_RUN")

        assert cluster.closed == 1

    def test_an_unsafe_service_name_is_rejected(self, cluster: Cluster) -> None:
        with pytest.raises(ValueError, match="service"):
            wait_for_head_ip(cluster.runtime(), "ARENA; DROP")


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
        assert cluster.events[0].startswith("ray ray start --head")
        assert cluster.events[-2:] == ["put .done", "ray stop"]
        assert cluster.closed == 2

    def test_object_store_takes_most_of_shm(self, cluster: Cluster) -> None:
        recorded: list[list[str]] = []
        runtime = cluster.runtime()
        runtime.start_ray = recorded.append

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
            "put .done",
            "ray stop",
        ]

    def test_trainer_failure_code_reaches_done(self, cluster: Cluster) -> None:
        code = run_node(["python"], cluster.runtime(code=3))

        assert code == 3
        assert cluster.done() == 3

    def test_cluster_timeout_writes_a_failed_done(self, cluster: Cluster) -> None:
        runtime = cluster.runtime(
            environ={"SNOWFLAKE_JOBS_COUNT": "2", "SNOWFLAKE_JOB_INDEX": "0"},
            alive_nodes=lambda: 1,
        )

        with pytest.raises(TimeoutError, match="1 of 2 nodes"):
            run_node(["python"], runtime)

        assert cluster.done() == 1
        assert cluster.commands == []
        assert cluster.events[-1] == "ray stop"

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
        assert cluster.events[-1] == "ray stop"

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

        assert cluster.events[-1] == "ray stop"
        assert cluster.closed == 1


class TestRunNodeWorker:
    def test_waits_for_the_head_gcs_before_ray_start(self, cluster: Cluster) -> None:
        cluster.instance_rows = [[Row(instance_id=0, ip_address="10.0.0.8")]]
        cluster.put_done(0)
        opened = iter([False, False, True])

        code = run_node(
            ["python"],
            cluster.runtime(WORKER_ENV, gcs_open=lambda host: next(opened, False)),
        )

        assert code == 0
        ray_start = cluster.events.index("ray ray start --address 10.0.0.8:61001")
        assert cluster.events[:ray_start] == [
            "gcs 10.0.0.8 False",
            "gcs 10.0.0.8 False",
            "gcs 10.0.0.8 True",
        ]
        assert cluster.commands == []

    def test_head_gcs_never_opening_times_out(self, cluster: Cluster) -> None:
        cluster.instance_rows = [[Row(instance_id=0, ip_address="10.0.0.8")]]

        with pytest.raises(TimeoutError, match="did not open"):
            run_node(
                ["python"], cluster.runtime(WORKER_ENV, gcs_open=lambda host: False)
            )

        assert not any(event.startswith("ray ") for event in cluster.events)

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


class TestRunNodeValidation:
    def test_empty_command_is_rejected(self, cluster: Cluster) -> None:
        with pytest.raises(ValueError, match="empty"):
            run_node([], cluster.runtime())

    @pytest.mark.parametrize(
        ("environ", "name"),
        [
            ({"SNOWFLAKE_JOB_INDEX": "0"}, "SNOWFLAKE_JOBS_COUNT"),
            ({"SNOWFLAKE_JOBS_COUNT": "1"}, "SNOWFLAKE_JOB_INDEX"),
            (
                {"SNOWFLAKE_JOBS_COUNT": "2", "SNOWFLAKE_JOB_INDEX": "1"},
                "SNOWFLAKE_SERVICE_NAME",
            ),
        ],
    )
    def test_missing_spcs_variable_is_rejected(
        self, cluster: Cluster, environ: dict[str, str], name: str
    ) -> None:
        with pytest.raises(KeyError, match=name):
            run_node(["python"], cluster.runtime(environ))

        assert cluster.events == []

    @pytest.mark.parametrize(
        ("environ", "name"),
        [
            (
                {"SNOWFLAKE_JOBS_COUNT": "0", "SNOWFLAKE_JOB_INDEX": "0"},
                "SNOWFLAKE_JOBS_COUNT",
            ),
            (
                {"SNOWFLAKE_JOBS_COUNT": "no", "SNOWFLAKE_JOB_INDEX": "0"},
                "SNOWFLAKE_JOBS_COUNT",
            ),
            (
                {"SNOWFLAKE_JOBS_COUNT": "1", "SNOWFLAKE_JOB_INDEX": "-1"},
                "SNOWFLAKE_JOB_INDEX",
            ),
            (
                {"SNOWFLAKE_JOBS_COUNT": "2", "SNOWFLAKE_JOB_INDEX": "x"},
                "SNOWFLAKE_JOB_INDEX",
            ),
            (
                {
                    "SNOWFLAKE_JOBS_COUNT": "2",
                    "SNOWFLAKE_JOB_INDEX": "1",
                    "SNOWFLAKE_SERVICE_NAME": "",
                },
                "service",
            ),
        ],
    )
    def test_malformed_spcs_variable_is_rejected(
        self, cluster: Cluster, environ: dict[str, str], name: str
    ) -> None:
        with pytest.raises(ValueError, match=name):
            run_node(["python"], cluster.runtime(environ))


class TestDoneFile:
    def test_missing_file_is_none(self, cluster: Cluster) -> None:
        assert read_done(Session(cluster), RUN_PREFIX) is None

    def test_written_code_is_read_back(self, cluster: Cluster) -> None:
        write_done(Session(cluster), RUN_PREFIX, 7)

        assert read_done(Session(cluster), RUN_PREFIX) == 7
        assert cluster.stage_file(".done").read_text(encoding="utf-8") == "7\n"


class TestMain:
    def test_usage(self) -> None:
        with pytest.raises(SystemExit, match="usage:"):
            main([])

    def test_delegates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(node_mod, "run_node", lambda command, runtime=None: 4)

        assert main(["python", "-c", "print(1)"]) == 4

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
        calls: list[tuple[list[str], bool]] = []

        def fake_run(args: list[str], check: bool = False) -> None:
            calls.append((args, check))

        monkeypatch.setattr(node_mod.subprocess, "run", fake_run)

        # Act
        runtime = production_runtime()
        node_mod._start_ray(["ray", "start", "--head"])
        node_mod._stop_ray()
        runtime.sleep(0)

        # Assert
        assert runtime.node_ip()
        assert runtime.monotonic() >= 0
        assert runtime.retry_on == (ConnectionError, TimeoutError, OSError)
        assert runtime.ray_nodes is ray_nodes
        assert calls == [(["ray", "start", "--head"], True), (["ray", "stop"], False)]

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

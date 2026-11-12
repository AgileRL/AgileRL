# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Start Ray on each SPCS job instance, run the trainer on the head, and upload checkpoints."""

from __future__ import annotations

import json
import logging
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import FrameType
from typing import Protocol

from agilerl.arena.snowflake.job import (
    RUN_MOUNT,
    drop_service_sql,
    lowered,
    put_file,
    runs_prefix,
    snowflake_identifier,
)
from agilerl.arena.snowflake.license import (
    RETRY_BUDGET_S,
    USAGE_TIMEOUT_S,
    RunRenewalRefusedError,
    SnowflakeLicenseError,
    check_snowflake_license,
    complete_snowflake_run,
    heartbeat_snowflake_run,
    open_head_run,
)
from agilerl.arena.snowflake.ports import (
    RAY_CLIENT_SERVER_PORT,
    RAY_DASHBOARD_AGENT_GRPC_PORT,
    RAY_DASHBOARD_AGENT_LISTEN_PORT,
    RAY_DASHBOARD_PORT,
    RAY_GCS_PORT,
    RAY_MAX_WORKER_PORT,
    RAY_METRICS_EXPORT_PORT,
    RAY_MIN_WORKER_PORT,
    RAY_NODE_MANAGER_PORT,
    RAY_OBJECT_MANAGER_PORT,
    RAY_RUNTIME_ENV_AGENT_PORT,
)
from agilerl.arena.snowflake.session import open_session, transient_errors
from agilerl.arena.snowflake.source import SnowflakeRow, SnowflakeSession
from agilerl.arena.snowflake.upload import CKPT_DIR, CheckpointUploader

# One deadline covers every wait before training: broker, head, and all nodes.
STARTUP_TIMEOUT_S = 2700.0
UPLOADED_TIMEOUT_S = 1800.0
HEAD_GONE_S = 60.0
POLL_S = 5.0
UPLOAD_INTERVAL_S = 60.0
HEARTBEAT_INTERVAL_S = 300.0
# The compiled trainer reads the signed run token from this path.
RUN_TOKEN_PATH = Path(tempfile.gettempdir()) / "arena" / "run_token"
# 128 + SIGTERM, the code a shell reports for a process SIGTERM stopped.
SIGTERM_EXIT_CODE = 143
# Snowflake kills the container 30 s after SIGTERM; .done and the broker drop follow.
SIGTERM_COMPLETE_TIMEOUT_S = 10.0
DONE_MARKER = ".done"
STARTED_MARKER = ".started-"
UPLOADED_MARKER = ".uploaded-"

logger = logging.getLogger(__name__)


class ChildProcess(Protocol):
    """The head command while it runs."""

    def poll(self) -> int | None:
        """Exit code, or ``None`` while running."""
        ...

    def kill(self) -> None:
        """Stop the process."""
        ...

    def wait(self) -> int:
        """Block until the process exits."""
        ...


class HeadFinishedError(Exception):
    """The head wrote ``.done`` while this worker was still starting."""

    def __init__(self, code: int) -> None:
        super().__init__(f"The head finished with exit code {code}")
        self.code = code


@dataclass
class NodeRuntime:
    """Process, filesystem, and Snowflake calls ``run_node`` uses."""

    environ: Mapping[str, str]
    node_ip: Callable[[], str]
    open_session: Callable[[], SnowflakeSession]
    start_ray: Callable[[list[str], Mapping[str, str]], None]
    stop_ray: Callable[[], None]
    ray_nodes: Callable[[], AbstractContextManager[Callable[[], int]]]
    gcs_open: Callable[[str], bool]
    broker_healthy: Callable[[str], bool]
    start_command: Callable[[list[str], Mapping[str, str]], ChildProcess]
    sleep: Callable[[float], None]
    monotonic: Callable[[], float]
    shared_memory_bytes: Callable[[], int]
    retry_on: tuple[type[Exception], ...]
    run_dir: Path = field(default_factory=lambda: Path(RUN_MOUNT))
    ckpt_root: Path = CKPT_DIR
    token_path: Path = RUN_TOKEN_PATH
    startup_timeout_s: float = STARTUP_TIMEOUT_S
    uploaded_timeout_s: float = UPLOADED_TIMEOUT_S
    head_gone_s: float = HEAD_GONE_S
    poll_s: float = POLL_S
    upload_interval_s: float = UPLOAD_INTERVAL_S
    heartbeat_interval_s: float = HEARTBEAT_INTERVAL_S


@dataclass
class StartupClock:
    """One deadline shared by every wait before training.

    A worker also reads ``.done`` on each poll, so a failed head stops it early.
    """

    runtime: NodeRuntime
    session: SnowflakeSession
    run_prefix: str | None
    deadline: float

    def wait(
        self, ready: Callable[[], bool], timeout_message: Callable[[], str]
    ) -> None:
        """Poll ``ready`` until it is true.

        :param ready: Condition to wait for.
        :param timeout_message: Error text, built when the deadline passes.
        :raises HeadFinishedError: A worker found ``.done``.
        :raises TimeoutError: The deadline passed first.
        """
        while not ready():
            if self.run_prefix is not None:
                code = read_done(self.session, self.run_prefix)
                if code is not None:
                    raise HeadFinishedError(code)
            if self.runtime.monotonic() >= self.deadline:
                msg = f"{timeout_message()} after {self.runtime.startup_timeout_s:.0f}s"
                raise TimeoutError(msg)
            self.runtime.sleep(self.runtime.poll_s)


@dataclass
class RunHeartbeat:
    """Renews the run token every interval until stopped or Arena refuses."""

    run_id: str
    env: Mapping[str, str]
    token_path: Path
    interval_s: float
    sleep: Callable[[float], None]
    stop: threading.Event = field(default_factory=threading.Event)

    def run(self) -> None:
        """Heartbeat until :attr:`stop` is set or Arena refuses renewal."""
        while not self.stop.wait(self.interval_s):
            try:
                token = heartbeat_snowflake_run(self.run_id, self.env, self.sleep)
            except RunRenewalRefusedError as exc:
                logger.warning("%s The trainer stops when its token expires.", exc)
                return
            except SnowflakeLicenseError as exc:
                logger.warning("%s Trying again in %.0fs.", exc, self.interval_s)
                continue
            try:
                write_run_token(self.token_path, token)
            except OSError as exc:
                logger.warning(
                    "Could not write the run token to %s: %s. Trying again in %.0fs.",
                    self.token_path,
                    exc,
                    self.interval_s,
                )


@dataclass
class HeadUsage:
    """The head's Arena usage row: renewed while open, completed once.

    The heartbeat runs until complete returns, so Arena never sees the run go
    stale while the head waits for worker uploads.
    """

    runtime: NodeRuntime
    run_id: str
    opened: bool = False
    finished: bool = False
    heartbeat: RunHeartbeat | None = None

    def open(self, job: Mapping[str, object], started_at: datetime) -> None:
        """Open the row, write the run token, and start the heartbeat.

        :param job: ``job.json`` fields.
        :param started_at: Earliest GPU worker container start.
        """
        runtime = self.runtime
        token = open_head_run(job, runtime.environ, started_at, runtime.sleep)
        self.opened = True
        write_run_token(runtime.token_path, token)
        self.heartbeat = RunHeartbeat(
            self.run_id,
            runtime.environ,
            runtime.token_path,
            runtime.heartbeat_interval_s,
            runtime.sleep,
        )
        threading.Thread(
            target=self.heartbeat.run, name="arena-heartbeat", daemon=True
        ).start()

    def complete(self, retry_budget_s: float, timeout_s: float) -> None:
        """Post complete once if the row is open, then stop the heartbeat.

        A failed complete is logged; Arena closes the run from its last heartbeat.

        :param retry_budget_s: Seconds of waiting allowed between attempts.
        :param timeout_s: Timeout of each attempt.
        """
        if not self.opened or self.finished:
            return
        self.finished = True
        runtime = self.runtime
        try:
            complete_snowflake_run(
                self.run_id, runtime.environ, runtime.sleep, retry_budget_s, timeout_s
            )
        except SnowflakeLicenseError as exc:
            logger.error("%s Arena closes the run from its last heartbeat.", exc)
        finally:
            if self.heartbeat is not None:
                self.heartbeat.stop.set()

    def complete_on_exit(self, exc: SystemExit) -> None:
        """On SIGTERM, post complete first with one short attempt.

        :param exc: The exit being raised.
        """
        if exc.code == SIGTERM_EXIT_CODE:
            self.complete(0.0, SIGTERM_COMPLETE_TIMEOUT_S)


def main(argv: list[str] | None = None) -> int:
    """Start Ray, run ``argv`` on the head, and exit when the job is finished.

    :param argv: Command to run on the head. ``None`` reads ``sys.argv``.
    :return: The head command's exit code, on every instance.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        msg = (
            "usage: python -m agilerl.arena.snowflake.node "
            "python -m agilerl.arena.snowflake.job_entry /arena/run/job.json"
        )
        raise SystemExit(msg)
    signal.signal(signal.SIGTERM, exit_on_sigterm)
    return run_node(args)


def exit_on_sigterm(_signum: int, _frame: FrameType | None) -> None:
    """Raise ``SystemExit`` so ``finally`` blocks run within Snowflake's 30 s grace.

    :raises SystemExit: Always, with :data:`SIGTERM_EXIT_CODE`.
    """
    raise SystemExit(SIGTERM_EXIT_CODE)


def run_node(command: list[str], runtime: NodeRuntime | None = None) -> int:
    """Start this instance's Ray node, run ``command`` on the head, and upload checkpoints.

    :param command: Argv the head runs after the cluster is up.
    :param runtime: Process and SQL calls. ``None`` uses this machine.
    :return: The head command's exit code.
    :raises KeyError: SPCS did not set ``SNOWFLAKE_SERVICE_NAME``.
    :raises ValueError: The command is empty, or the service is not a head
        or workers service.
    :raises TimeoutError: The broker, the head, or the cluster was not ready
        before the startup deadline.
    :raises SnowflakeLicenseError: Arena refused the run or did not record it.
    :raises RuntimeError: A worker lost the head before it wrote ``.done``.
    """
    if not command:
        msg = "the head command is empty"
        raise ValueError(msg)
    current = production_runtime() if runtime is None else runtime
    role = _service_role(current.environ["SNOWFLAKE_SERVICE_NAME"])
    job = json.loads((current.run_dir / "job.json").read_text(encoding="utf-8"))
    run_prefix = runs_prefix(str(job["runs_stage"]), str(job["run_id"]))
    uploader = CheckpointUploader(
        open_session=current.open_session,
        runs_stage=str(job["runs_stage"]),
        run_id=str(job["run_id"]),
        ckpt_root=current.ckpt_root,
        retry_on=current.retry_on,
        sleep=current.sleep,
    )
    if role == "head":
        return _run_head(command, current, uploader, job, run_prefix)
    return _run_worker(current, uploader, job, run_prefix)


def head_ray_command(node_ip: str, object_store_bytes: int) -> list[str]:
    """``ray start --head`` argv for this instance.

    :param node_ip: Address passed to ``--node-ip-address``.
    :param object_store_bytes: Object store size.
    :return: The argument list.
    """
    return [
        "ray",
        "start",
        "--head",
        "--node-ip-address",
        node_ip,
        "--port",
        str(RAY_GCS_PORT),
        "--ray-client-server-port",
        str(RAY_CLIENT_SERVER_PORT),
        "--dashboard-host",
        "0.0.0.0",
        "--dashboard-port",
        str(RAY_DASHBOARD_PORT),
        "--num-cpus",
        "0",
        "--num-gpus",
        "0",
        *_common_ray_args(object_store_bytes),
    ]


def worker_ray_command(
    node_ip: str, head_ip: str, object_store_bytes: int
) -> list[str]:
    """``ray start`` argv that joins the head.

    :param node_ip: Address passed to ``--node-ip-address``.
    :param head_ip: Head instance IP.
    :param object_store_bytes: Object store size.
    :return: The argument list.
    """
    return [
        "ray",
        "start",
        "--address",
        f"{head_ip}:{RAY_GCS_PORT}",
        "--node-ip-address",
        node_ip,
        *_common_ray_args(object_store_bytes),
    ]


def head_ip_from_rows(rows: list[SnowflakeRow]) -> str | None:
    """IP of instance 0.

    :param rows: ``SHOW SERVICE INSTANCES`` rows.
    :return: The IP, or ``None`` while instance 0 is unlisted or has no IP.
    """
    for row in rows:
        values = lowered(row)
        if int(str(values["instance_id"])) != 0:
            continue
        ip = values.get("ip_address")
        if ip is None or str(ip).strip() == "":
            return None
        return str(ip)
    return None


def wait_for_head_ip(startup: StartupClock, service: str) -> str:
    """Query instances on the startup session until instance 0 has an IP.

    :param startup: Deadline, session, and ``.done`` check.
    :param service: Service name passed to ``SHOW SERVICE INSTANCES``.
    :return: Instance 0's IP.
    :raises HeadFinishedError: A worker found ``.done``.
    :raises TimeoutError: Instance 0 still has no IP at the deadline.
    """
    name = snowflake_identifier(service, "service")
    ip: str | None = None

    def has_ip() -> bool:
        nonlocal ip
        rows = startup.session.sql(
            f"SHOW SERVICE INSTANCES IN SERVICE {name}"
        ).collect()
        ip = head_ip_from_rows(rows)
        return ip is not None

    startup.wait(has_ip, lambda: f"Instance 0 of {name} had no IP")
    return str(ip)


def write_marker(
    session: SnowflakeSession, run_prefix: str, name: str, text: str
) -> None:
    """PUT one small file to the run directory, so it is complete once visible.

    :param session: Snowpark session.
    :param run_prefix: ``@stage/<run_id>``.
    :param name: File name.
    :param text: File contents.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / name
        path.write_text(text, encoding="utf-8")
        put_file(session, path, run_prefix)


def marker_names(session: SnowflakeSession, run_prefix: str, prefix: str) -> list[str]:
    """Names of run-directory files that start with ``prefix``.

    :param session: Snowpark session.
    :param run_prefix: ``@stage/<run_id>``.
    :param prefix: Start of the file name.
    :return: Base names, in listing order.
    """
    rows = session.sql(f"LIST {run_prefix}/{prefix}").collect()
    return [str(lowered(row)["name"]).rsplit("/", 1)[-1] for row in rows]


def read_marker(session: SnowflakeSession, run_prefix: str, name: str) -> str:
    """Contents of one run-directory file.

    :param session: Snowpark session.
    :param run_prefix: ``@stage/<run_id>``.
    :param name: File name.
    :return: The text with surrounding space removed.
    """
    with tempfile.TemporaryDirectory() as tmp:
        session.file.get(f"{run_prefix}/{name}", tmp)
        return (Path(tmp) / name).read_text(encoding="utf-8").strip()


def write_done(session: SnowflakeSession, run_prefix: str, code: int) -> None:
    """PUT ``.done`` with the head's exit code.

    :param session: Snowpark session.
    :param run_prefix: ``@stage/<run_id>``.
    :param code: Exit code every instance returns.
    """
    write_marker(session, run_prefix, DONE_MARKER, f"{code}\n")


def read_done(session: SnowflakeSession, run_prefix: str) -> int | None:
    """Exit code the head PUT to ``.done``, read from the stage.

    :param session: Snowpark session.
    :param run_prefix: ``@stage/<run_id>``.
    :return: The code, or ``None`` before the head writes it.
    """
    if DONE_MARKER not in marker_names(session, run_prefix, DONE_MARKER):
        return None
    return int(read_marker(session, run_prefix, DONE_MARKER))


def earliest_worker_start(
    session: SnowflakeSession, run_prefix: str, workers: int
) -> datetime:
    """Earliest ``.started-*`` time, when Snowflake started billing GPU nodes.

    :param session: Snowpark session.
    :param run_prefix: ``@stage/<run_id>``.
    :param workers: Worker instances in the run.
    :return: The earliest start, timezone-aware.
    :raises RuntimeError: Fewer than ``workers`` markers are on the stage.
    """
    names = marker_names(session, run_prefix, STARTED_MARKER)
    if len(names) < workers:
        msg = f"Only {len(names)} of {workers} workers wrote {STARTED_MARKER}*"
        raise RuntimeError(msg)
    return min(
        datetime.fromisoformat(read_marker(session, run_prefix, name)) for name in names
    )


def write_run_token(path: Path, token: str) -> None:
    """Replace the token file in one rename, so the trainer never reads half a token.

    :param path: Token file.
    :param token: Signed run token.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial")
    partial.write_text(token, encoding="utf-8")
    partial.replace(path)


def shared_memory_bytes(path: str = "/dev/shm") -> int:
    """Capacity of the memory volume.

    :param path: Mount to measure.
    :return: Size in bytes.
    """
    stats = os.statvfs(path)
    return int(stats.f_frsize * stats.f_blocks)


def production_runtime() -> NodeRuntime:
    """Runtime that talks to this process, Ray, and Snowflake.

    :return: The default runtime.
    """
    return NodeRuntime(
        environ=os.environ,
        node_ip=lambda: socket.gethostbyname(socket.gethostname()),
        open_session=lambda: open_session(None),
        start_ray=_start_ray,
        stop_ray=_stop_ray,
        ray_nodes=ray_nodes,
        gcs_open=_gcs_open,
        broker_healthy=broker_is_healthy,
        start_command=_start_command,
        sleep=time.sleep,
        monotonic=time.monotonic,
        shared_memory_bytes=shared_memory_bytes,
        retry_on=transient_errors(),
    )


@contextmanager
def ray_nodes() -> Iterator[Callable[[], int]]:
    """Connect one Ray driver to this cluster and count its alive nodes.

    :return: A context whose value returns the alive node count on each call.
    """
    # Deferred optional package: ray is installed in the job image only.
    import ray

    ray.init(address="auto", log_to_driver=False)
    try:
        yield lambda: sum(1 for node in ray.nodes() if node["Alive"])
    finally:
        ray.shutdown()


def broker_is_healthy(ip: str) -> bool:
    """True when the Pulsar broker health endpoint returns 200.

    :param ip: Broker instance IP.
    :return: Whether ``/admin/v2/brokers/health`` returned 200.
    """
    url = f"http://{ip}:8080/admin/v2/brokers/health"
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return int(response.status) == 200
    except OSError:
        return False


def _common_ray_args(object_store_bytes: int) -> list[str]:
    return [
        "--object-manager-port",
        str(RAY_OBJECT_MANAGER_PORT),
        "--node-manager-port",
        str(RAY_NODE_MANAGER_PORT),
        "--runtime-env-agent-port",
        str(RAY_RUNTIME_ENV_AGENT_PORT),
        "--dashboard-agent-grpc-port",
        str(RAY_DASHBOARD_AGENT_GRPC_PORT),
        "--dashboard-agent-listen-port",
        str(RAY_DASHBOARD_AGENT_LISTEN_PORT),
        "--metrics-export-port",
        str(RAY_METRICS_EXPORT_PORT),
        "--min-worker-port",
        str(RAY_MIN_WORKER_PORT),
        "--max-worker-port",
        str(RAY_MAX_WORKER_PORT),
        "--object-store-memory",
        str(object_store_bytes),
        "--disable-usage-stats",
    ]


def _service_role(service: str) -> str:
    if service.endswith("_HEAD"):
        return "head"
    if service.endswith("_WORKERS"):
        return "worker"
    msg = f"SNOWFLAKE_SERVICE_NAME {service!r} is not a head or workers service"
    raise ValueError(msg)


def _object_store_bytes(runtime: NodeRuntime) -> int:
    # NCCL and vLLM also allocate from /dev/shm.
    return int(runtime.shared_memory_bytes() * 0.8)


def _join_broker(startup: StartupClock, job: Mapping[str, object]) -> Mapping[str, str]:
    runtime = startup.runtime
    broker_ip = wait_for_head_ip(startup, str(job["broker_service"]))
    startup.wait(
        lambda: runtime.broker_healthy(broker_ip),
        lambda: f"Broker at {broker_ip} did not become healthy",
    )
    url = f"pulsar://{broker_ip}:6650"
    if runtime.environ is os.environ:
        os.environ["APP_PULSAR_URL"] = url
        return os.environ
    env = dict(runtime.environ)
    env["APP_PULSAR_URL"] = url
    return env


def _run_head(
    command: list[str],
    runtime: NodeRuntime,
    uploader: CheckpointUploader,
    job: Mapping[str, object],
    run_prefix: str,
) -> int:
    done_code = 1
    usage = HeadUsage(runtime, str(job["run_id"]))
    try:
        try:
            env, started_at = _start_head_cluster(runtime, job, run_prefix)
            usage.open(job, started_at)
            done_code = _train(command, runtime, uploader, env, job)
        except SystemExit as exc:
            usage.complete_on_exit(exc)
            raise
        finally:
            # Workers read .done once the head's GCS closes, so it goes up first.
            try:
                _put_done(runtime, run_prefix, done_code)
            finally:
                runtime.stop_ray()
        _wait_for_uploads(runtime, run_prefix, int(str(job["nodes"])))
    except SystemExit as exc:
        usage.complete_on_exit(exc)
        raise
    finally:
        usage.complete(RETRY_BUDGET_S, USAGE_TIMEOUT_S)
        _drop_broker(runtime, job)
    return done_code


def _start_head_cluster(
    runtime: NodeRuntime, job: Mapping[str, object], run_prefix: str
) -> tuple[Mapping[str, str], datetime]:
    check_snowflake_license(runtime.environ)
    session = runtime.open_session()
    try:
        startup = StartupClock(
            runtime, session, None, runtime.monotonic() + runtime.startup_timeout_s
        )
        env = _join_broker(startup, job)
        runtime.start_ray(
            head_ray_command(runtime.node_ip(), _object_store_bytes(runtime)), env
        )
        workers = int(str(job["nodes"]))
        expected = workers + 1
        alive = 0
        with runtime.ray_nodes() as alive_nodes:

            def joined() -> bool:
                nonlocal alive
                alive = alive_nodes()
                return alive >= expected

            startup.wait(joined, lambda: f"Ray cluster had {alive} of {expected} nodes")
        return env, earliest_worker_start(session, run_prefix, workers)
    finally:
        session.close()


def _train(
    command: list[str],
    runtime: NodeRuntime,
    uploader: CheckpointUploader,
    env: Mapping[str, str],
    job: Mapping[str, object],
) -> int:
    child_env = dict(env)
    child_env["RAY_ADDRESS"] = "auto"
    child_env["ARENA_RUN_ID"] = str(job["run_id"])
    process = runtime.start_command(command, child_env)
    try:
        code = _wait_for_exit(runtime, uploader, process)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    uploader.upload(final=True)
    return code


def _drop_broker(runtime: NodeRuntime, job: Mapping[str, object]) -> None:
    session = runtime.open_session()
    try:
        session.sql(
            drop_service_sql(str(job["broker_service"]), if_exists=True)
        ).collect()
    finally:
        session.close()


def _put_done(runtime: NodeRuntime, run_prefix: str, code: int) -> None:
    session = runtime.open_session()
    try:
        write_done(session, run_prefix, code)
    finally:
        session.close()


def _wait_for_uploads(runtime: NodeRuntime, run_prefix: str, workers: int) -> None:
    deadline = runtime.monotonic() + runtime.uploaded_timeout_s
    session = runtime.open_session()
    try:
        while True:
            uploaded = len(marker_names(session, run_prefix, UPLOADED_MARKER))
            if uploaded >= workers:
                return
            if runtime.monotonic() >= deadline:
                logger.warning(
                    "%d of %d workers wrote %s* after %.0fs",
                    uploaded,
                    workers,
                    UPLOADED_MARKER,
                    runtime.uploaded_timeout_s,
                )
                return
            runtime.sleep(runtime.poll_s)
    finally:
        session.close()


def _run_worker(
    runtime: NodeRuntime,
    uploader: CheckpointUploader,
    job: Mapping[str, object],
    run_prefix: str,
) -> int:
    node_ip = runtime.node_ip()
    session = runtime.open_session()
    try:
        write_marker(
            session,
            run_prefix,
            f"{STARTED_MARKER}{node_ip}",
            datetime.now(timezone.utc).isoformat(),
        )
        startup = StartupClock(
            runtime,
            session,
            run_prefix,
            runtime.monotonic() + runtime.startup_timeout_s,
        )
        try:
            env = _join_broker(startup, job)
            head_ip = wait_for_head_ip(startup, str(job["head_service"]))
            startup.wait(
                lambda: runtime.gcs_open(head_ip),
                lambda: f"Head GCS at {head_ip} did not open",
            )
        except HeadFinishedError as finished:
            return finished.code
    finally:
        session.close()
    runtime.start_ray(
        worker_ray_command(node_ip, head_ip, _object_store_bytes(runtime)), env
    )
    try:
        code = _wait_for_done(runtime, uploader, run_prefix, head_ip)
        uploader.upload(final=True)
        _put_uploaded(runtime, run_prefix, node_ip)
    finally:
        runtime.stop_ray()
    if code is None:
        msg = (
            f"Head GCS at {head_ip} closed {runtime.head_gone_s:.0f}s ago "
            "and the head wrote no .done"
        )
        raise RuntimeError(msg)
    return code


def _put_uploaded(runtime: NodeRuntime, run_prefix: str, node_ip: str) -> None:
    session = runtime.open_session()
    try:
        write_marker(session, run_prefix, f"{UPLOADED_MARKER}{node_ip}", "")
    finally:
        session.close()


def _wait_for_exit(
    runtime: NodeRuntime, uploader: CheckpointUploader, process: ChildProcess
) -> int:
    next_upload = runtime.monotonic() + runtime.upload_interval_s
    while True:
        code = process.poll()
        if code is not None:
            return int(code)
        if runtime.monotonic() >= next_upload:
            uploader.upload(final=False)
            next_upload = runtime.monotonic() + runtime.upload_interval_s
        runtime.sleep(runtime.poll_s)


def _wait_for_done(
    runtime: NodeRuntime,
    uploader: CheckpointUploader,
    run_prefix: str,
    head_ip: str,
) -> int | None:
    next_upload = runtime.monotonic() + runtime.upload_interval_s
    down_since: float | None = None
    while True:
        now = runtime.monotonic()
        if runtime.gcs_open(head_ip):
            down_since = None
        else:
            code = _read_done(runtime, run_prefix)
            if code is not None:
                return code
            if down_since is None:
                down_since = now
            elif now - down_since >= runtime.head_gone_s:
                return None
        if now >= next_upload:
            uploader.upload(final=False)
            next_upload = runtime.monotonic() + runtime.upload_interval_s
        runtime.sleep(runtime.poll_s)


def _read_done(runtime: NodeRuntime, run_prefix: str) -> int | None:
    session = runtime.open_session()
    try:
        return read_done(session, run_prefix)
    finally:
        session.close()


def _start_ray(args: list[str], env: Mapping[str, str]) -> None:
    if env is os.environ:
        subprocess.run(args, check=True)
        return
    subprocess.run(args, check=True, env=dict(env))


def _stop_ray() -> None:
    subprocess.run(["ray", "stop"], check=False)


def _gcs_open(host: str) -> bool:
    sock = socket.socket()
    sock.settimeout(2)
    try:
        sock.connect((host, RAY_GCS_PORT))
    except OSError:
        return False
    else:
        return True
    finally:
        sock.close()


def _start_command(command: list[str], env: Mapping[str, str]) -> ChildProcess:
    return subprocess.Popen(command, env=dict(env))


if __name__ == "__main__":
    raise SystemExit(main())

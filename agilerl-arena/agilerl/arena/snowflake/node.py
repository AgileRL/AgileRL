# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Start Ray on each SPCS job instance, run the trainer on the head, and upload checkpoints."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from agilerl.arena.snowflake.job import (
    RUN_MOUNT,
    lowered,
    put_file,
    runs_prefix,
    snowflake_identifier,
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

HEAD_TIMEOUT_S = 600.0
CLUSTER_TIMEOUT_S = 600.0
HEAD_GONE_S = 60.0
POLL_S = 2.0
UPLOAD_INTERVAL_S = 60.0


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


@dataclass
class NodeRuntime:
    """Process, filesystem, and Snowflake calls ``run_node`` uses."""

    environ: Mapping[str, str]
    node_ip: Callable[[], str]
    open_session: Callable[[], SnowflakeSession]
    start_ray: Callable[[list[str]], None]
    stop_ray: Callable[[], None]
    ray_nodes: Callable[[], AbstractContextManager[Callable[[], int]]]
    gcs_open: Callable[[str], bool]
    start_command: Callable[[list[str], Mapping[str, str]], ChildProcess]
    sleep: Callable[[float], None]
    monotonic: Callable[[], float]
    shared_memory_bytes: Callable[[], int]
    retry_on: tuple[type[Exception], ...]
    run_dir: Path = field(default_factory=lambda: Path(RUN_MOUNT))
    ckpt_root: Path = CKPT_DIR
    head_timeout_s: float = HEAD_TIMEOUT_S
    cluster_timeout_s: float = CLUSTER_TIMEOUT_S
    head_gone_s: float = HEAD_GONE_S
    poll_s: float = POLL_S
    upload_interval_s: float = UPLOAD_INTERVAL_S


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
    return run_node(args)


def run_node(command: list[str], runtime: NodeRuntime | None = None) -> int:
    """Start this instance's Ray node, run ``command`` on the head, and upload checkpoints.

    :param command: Argv the head runs after the cluster is up.
    :param runtime: Process and SQL calls. ``None`` uses this machine.
    :return: The head command's exit code.
    :raises KeyError: SPCS did not set ``SNOWFLAKE_JOBS_COUNT``,
        ``SNOWFLAKE_JOB_INDEX``, or, on a worker, ``SNOWFLAKE_SERVICE_NAME``.
    :raises ValueError: The command is empty or an SPCS variable is malformed.
    :raises TimeoutError: The head never gets an IP or opens its GCS, or a node never joins.
    :raises RuntimeError: A worker lost the head before it wrote ``.done``.
    """
    if not command:
        msg = "the head command is empty"
        raise ValueError(msg)
    current = production_runtime() if runtime is None else runtime
    count = _positive_int(
        current.environ["SNOWFLAKE_JOBS_COUNT"], "SNOWFLAKE_JOBS_COUNT"
    )
    index = _nonnegative_int(
        current.environ["SNOWFLAKE_JOB_INDEX"], "SNOWFLAKE_JOB_INDEX"
    )
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
    node_ip = current.node_ip()
    # NCCL and vLLM also allocate from /dev/shm.
    store_bytes = int(current.shared_memory_bytes() * 0.8)
    if index == 0:
        return _run_head(
            command, current, uploader, run_prefix, node_ip, store_bytes, count
        )
    head_ip = wait_for_head_ip(current, current.environ["SNOWFLAKE_SERVICE_NAME"])
    return _run_worker(current, uploader, run_prefix, node_ip, head_ip, store_bytes)


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
    """IP of instance 0, the head.

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


def wait_for_head_ip(runtime: NodeRuntime, service: str) -> str:
    """Query instances on one session until instance 0 has an IP.

    :param runtime: Clock, sleep, and the session factory.
    :param service: Service name passed to ``SHOW SERVICE INSTANCES``.
    :return: The head IP.
    :raises TimeoutError: Instance 0 still has no IP at the deadline.
    """
    name = snowflake_identifier(service, "service")
    deadline = runtime.monotonic() + runtime.head_timeout_s
    session = runtime.open_session()
    try:
        while True:
            rows = session.sql(f"SHOW SERVICE INSTANCES IN SERVICE {name}").collect()
            ip = head_ip_from_rows(rows)
            if ip is not None:
                return ip
            if runtime.monotonic() >= deadline:
                msg = f"Instance 0 had no IP after {runtime.head_timeout_s:.0f}s"
                raise TimeoutError(msg)
            runtime.sleep(runtime.poll_s)
    finally:
        session.close()


def write_done(session: SnowflakeSession, run_prefix: str, code: int) -> None:
    """PUT ``.done`` with the head's exit code, so it is complete once visible.

    :param session: Snowpark session.
    :param run_prefix: ``@stage/<run_id>``.
    :param code: Exit code every instance returns.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / ".done"
        path.write_text(f"{code}\n", encoding="utf-8")
        put_file(session, path, run_prefix)


def read_done(session: SnowflakeSession, run_prefix: str) -> int | None:
    """Exit code the head PUT to ``.done``, read from the stage.

    :param session: Snowpark session.
    :param run_prefix: ``@stage/<run_id>``.
    :return: The code, or ``None`` before the head writes it.
    """
    location = f"{run_prefix}/.done"
    if not session.sql(f"LIST {location}").collect():
        return None
    with tempfile.TemporaryDirectory() as tmp:
        session.file.get(location, tmp)
        return int((Path(tmp) / ".done").read_text(encoding="utf-8").strip())


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


def _run_head(
    command: list[str],
    runtime: NodeRuntime,
    uploader: CheckpointUploader,
    run_prefix: str,
    node_ip: str,
    object_store_bytes: int,
    count: int,
) -> int:
    done_code = 1
    runtime.start_ray(head_ray_command(node_ip, object_store_bytes))
    try:
        _wait_for_nodes(runtime, count)
        env = dict(runtime.environ)
        env["RAY_ADDRESS"] = "auto"
        process = runtime.start_command(command, env)
        try:
            code = _wait_for_exit(runtime, uploader, process)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
        uploader.upload(final=True)
        done_code = code
    finally:
        # Workers read .done once the head's GCS closes, so it goes up first.
        try:
            session = runtime.open_session()
            try:
                write_done(session, run_prefix, done_code)
            finally:
                session.close()
        finally:
            runtime.stop_ray()
    return done_code


def _run_worker(
    runtime: NodeRuntime,
    uploader: CheckpointUploader,
    run_prefix: str,
    node_ip: str,
    head_ip: str,
    object_store_bytes: int,
) -> int:
    _wait_for_gcs(runtime, head_ip)
    runtime.start_ray(worker_ray_command(node_ip, head_ip, object_store_bytes))
    try:
        code = _wait_for_done(runtime, uploader, run_prefix, head_ip)
        uploader.upload(final=True)
    finally:
        runtime.stop_ray()
    if code is None:
        msg = (
            f"Head GCS at {head_ip} closed {runtime.head_gone_s:.0f}s ago "
            "and the head wrote no .done"
        )
        raise RuntimeError(msg)
    return code


def _wait_for_nodes(runtime: NodeRuntime, expected: int) -> None:
    deadline = runtime.monotonic() + runtime.cluster_timeout_s
    with runtime.ray_nodes() as alive_nodes:
        while True:
            alive = alive_nodes()
            if alive >= expected:
                return
            if runtime.monotonic() >= deadline:
                msg = (
                    f"Ray cluster had {alive} of {expected} nodes "
                    f"after {runtime.cluster_timeout_s:.0f}s"
                )
                raise TimeoutError(msg)
            runtime.sleep(runtime.poll_s)


def _wait_for_gcs(runtime: NodeRuntime, head_ip: str) -> None:
    deadline = runtime.monotonic() + runtime.head_timeout_s
    while not runtime.gcs_open(head_ip):
        if runtime.monotonic() >= deadline:
            msg = f"Head GCS at {head_ip} did not open after {runtime.head_timeout_s:.0f}s"
            raise TimeoutError(msg)
        runtime.sleep(runtime.poll_s)


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


def _positive_int(value: str, name: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        msg = f"{name} must be an integer, got {value!r}"
        raise ValueError(msg) from exc
    if number < 1:
        msg = f"{name} must be at least 1, got {number}"
        raise ValueError(msg)
    return number


def _nonnegative_int(value: str, name: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        msg = f"{name} must be an integer, got {value!r}"
        raise ValueError(msg) from exc
    if number < 0:
        msg = f"{name} must be at least 0, got {number}"
        raise ValueError(msg)
    return number


def _start_ray(args: list[str]) -> None:
    subprocess.run(args, check=True)


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

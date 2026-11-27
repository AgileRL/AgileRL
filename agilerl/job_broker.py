# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Run Pulsar standalone as a sidecar container in a training job's Ray head pod."""

from __future__ import annotations

import logging
import os
import shutil
import signal
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import FrameType

from agilerl.broker import (
    CGROUP_MEMORY_MAX,
    DATA_ROOT,
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

# Ray's metrics export port is 8080, and the head pod shares one network namespace.
WEB_PORT = 8081
BACKLOG_QUOTA_BYTES = 8 * 1024**3
READY_FILE = "ready"

logger = logging.getLogger(__name__)


@dataclass
class JobBrokerRuntime:
    """Process, HTTP, clock, and filesystem calls ``run_job_broker`` uses."""

    environ: Mapping[str, str]
    node_ip: Callable[[], str]
    start_process: Callable[[list[str], Mapping[str, str]], PulsarProcess]
    run_command: Callable[[list[str]], CommandResult]
    http_status: Callable[[str], int]
    sleep: Callable[[float], None]
    monotonic: Callable[[], float]
    pulsar_home: Path = field(default_factory=lambda: DEFAULT_PULSAR_HOME)
    data_root: Path = field(default_factory=lambda: DATA_ROOT)
    memory_limit_path: Path = field(default_factory=lambda: CGROUP_MEMORY_MAX)
    health_timeout_s: float = HEALTH_TIMEOUT_S
    poll_s: float = POLL_S


def main() -> int:
    """Run the job's broker until the pod stops it.

    :return: The process exit code.
    """
    logging.basicConfig(level=logging.INFO)
    signal.signal(signal.SIGTERM, stop_on_sigterm)
    return run_job_broker()


def stop_on_sigterm(_signum: int, _frame: FrameType | None) -> None:
    """Raise ``SystemExit`` so Pulsar is stopped before the container exits.

    :raises SystemExit: Always, with 0.
    """
    raise SystemExit(0)


def broker_overrides() -> dict[str, str]:
    """``standalone.conf`` values for a broker that shares the Ray head pod.

    :return: Web port and per-topic backlog quota settings.
    """
    return {
        "webServicePort": str(WEB_PORT),
        # The broker reads backlogQuotaDefaultLimitGB first when it is above 0,
        # and standalone.conf ships it as 10.
        "backlogQuotaDefaultLimitGB": "0",
        "backlogQuotaDefaultLimitBytes": str(BACKLOG_QUOTA_BYTES),
        # Holds the rollout sender instead of evicting part of a group.
        "backlogQuotaDefaultRetentionPolicy": "producer_request_hold",
    }


def standalone_options(data_root: Path) -> list[str]:
    """``pulsar standalone`` arguments that keep all broker state under ``data_root``.

    :param data_root: The broker's data volume.
    :return: Argv after ``--config``.
    """
    return [
        "--bookkeeper-dir",
        str(data_root / "bookkeeper"),
        "--metadata-dir",
        str(data_root / "metadata"),
        "--no-functions-worker",
        "--no-stream-storage",
    ]


def run_job_broker(runtime: JobBrokerRuntime | None = None) -> int:
    """Start Pulsar, create ``arena/run``, mark it ready, and run until Pulsar exits.

    A restarted container finds the previous broker's data on the same volume
    and recovers it. When that data does not come back healthy, it is wiped and
    the broker starts empty.

    :param runtime: Process, HTTP, and filesystem calls. ``None`` uses this container.
    :return: 1 when Pulsar exits on its own, so the container restarts.
    :raises ValueError: The container has no memory limit.
    :raises TimeoutError: The health endpoint never returns 200.
    :raises RuntimeError: Pulsar exited before it was healthy, or ``pulsar-admin``
        failed for a reason other than the tenant or namespace already existing.
    """
    current = production_runtime() if runtime is None else runtime
    limit = cgroup_memory_bytes(current.memory_limit_path)
    if limit is None:
        msg = (
            f"{current.memory_limit_path} has no memory limit. "
            "Give the broker container a memory limit."
        )
        raise ValueError(msg)
    ready = current.data_root / READY_FILE
    ready.unlink(missing_ok=True)
    write_broker_config(
        current.pulsar_home,
        current.node_ip(),
        broker_memory(limit),
        current.data_root,
        broker_overrides(),
    )
    process = _start_recovered(current)
    try:
        ensure_tenant_and_namespace(current, WEB_PORT)
        ready.touch()
        while process.poll() is None:
            current.sleep(current.poll_s)
        return 1
    finally:
        ready.unlink(missing_ok=True)
        stop_pulsar(process)


def production_runtime() -> JobBrokerRuntime:
    """Runtime that talks to this container and Pulsar.

    :return: The default runtime.
    """
    return JobBrokerRuntime(
        environ=os.environ,
        node_ip=node_ip,
        start_process=start_process,
        run_command=run_command,
        http_status=http_status,
        sleep=time.sleep,
        monotonic=time.monotonic,
        pulsar_home=pulsar_home_from(os.environ),
    )


def _start_recovered(runtime: JobBrokerRuntime) -> PulsarProcess:
    state_dirs = [runtime.data_root / "bookkeeper", runtime.data_root / "metadata"]
    previous = [path for path in state_dirs if path.exists()]
    try:
        return _start_healthy(runtime)
    except (TimeoutError, RuntimeError):
        if not previous:
            raise
    logger.warning(
        "Pulsar did not recover its data in %s; starting with none.", runtime.data_root
    )
    for path in previous:
        shutil.rmtree(path)
    return _start_healthy(runtime)


def _start_healthy(runtime: JobBrokerRuntime) -> PulsarProcess:
    process = start_standalone(runtime, standalone_options(runtime.data_root))
    try:
        wait_until_healthy(runtime, WEB_PORT, process)
    except BaseException:
        stop_pulsar(process)
        raise
    return process


if __name__ == "__main__":
    raise SystemExit(main())

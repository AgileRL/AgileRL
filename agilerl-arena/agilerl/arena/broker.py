# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Run Pulsar standalone in a container: config, start, health, and stop."""

from __future__ import annotations

import socket
import subprocess
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

DEFAULT_PULSAR_HOME = Path("/opt/pulsar")
CGROUP_MEMORY_MAX = Path("/sys/fs/cgroup/memory.max")
DATA_ROOT = Path("/pulsar-data")
WEB_PORT = 8080
HEALTH_TIMEOUT_S = 600.0
POLL_S = 2.0


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


class PulsarCalls(Protocol):
    """Process, HTTP, and clock calls the helpers below make."""

    environ: Mapping[str, str]
    start_process: Callable[[list[str], Mapping[str, str]], PulsarProcess]
    run_command: Callable[[list[str]], CommandResult]
    http_status: Callable[[str], int]
    sleep: Callable[[float], None]
    monotonic: Callable[[], float]
    pulsar_home: Path
    health_timeout_s: float
    poll_s: float


def cgroup_memory_bytes(path: Path) -> int | None:
    """Cgroup memory limit in bytes.

    :param path: ``memory.max``.
    :return: The limit, or ``None`` when the file is missing or holds ``max``.
    """
    try:
        text = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    return int(text) if text.isdecimal() else None


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
    pulsar_home: Path,
    advertised_address: str,
    memory: BrokerMemory,
    data_root: Path = DATA_ROOT,
    broker_overrides: Mapping[str, str] | None = None,
) -> None:
    """Write broker, bookie, and JVM settings under the Pulsar install.

    :param pulsar_home: Pulsar install root.
    :param advertised_address: This container's IP.
    :param memory: Heap, direct memory, and cache sizes.
    :param data_root: Directory for the bookie journal and ledgers.
    :param broker_overrides: ``standalone.conf`` values that replace the defaults here.
    """
    conf_dir = pulsar_home / "conf"
    conf_dir.mkdir(parents=True, exist_ok=True)
    journal_dir = data_root / "journal"
    ledger_dir = data_root / "ledgers"
    journal_dir.mkdir(parents=True, exist_ok=True)
    ledger_dir.mkdir(parents=True, exist_ok=True)
    broker = _broker_properties(advertised_address, memory)
    broker.update(broker_overrides or {})
    _write_merged(conf_dir / "standalone.conf", broker)
    _write_merged(
        conf_dir / "bookkeeper.conf",
        _bookie_properties(journal_dir, ledger_dir, memory),
    )
    _write_merged(conf_dir / "pulsar_env.sh", {"PULSAR_MEM": _pulsar_mem(memory)})


def start_standalone(calls: PulsarCalls, options: Sequence[str] = ()) -> PulsarProcess:
    """Start ``pulsar standalone`` on the config under ``calls.pulsar_home``.

    :param calls: Process calls and the Pulsar install root.
    :param options: Extra ``pulsar standalone`` arguments.
    :return: The running process.
    """
    env = dict(calls.environ)
    env["PULSAR_HOME"] = str(calls.pulsar_home)
    command = [
        str(calls.pulsar_home / "bin" / "pulsar"),
        "standalone",
        "--config",
        str(calls.pulsar_home / "conf" / "standalone.conf"),
        *options,
    ]
    return calls.start_process(command, env)


def health_url(host: str, web_port: int = WEB_PORT) -> str:
    """The broker health endpoint.

    :param host: Broker host or IP.
    :param web_port: Broker web service port.
    :return: The URL of ``/admin/v2/brokers/health``.
    """
    return f"http://{host}:{web_port}/admin/v2/brokers/health"


def wait_until_healthy(
    calls: PulsarCalls,
    web_port: int = WEB_PORT,
    process: PulsarProcess | None = None,
) -> None:
    """Poll the local broker until its health endpoint returns 200.

    :param calls: HTTP and clock calls, the deadline, and the poll interval.
    :param web_port: Broker web service port.
    :param process: The standalone process. When given, its exit ends the wait.
    :raises TimeoutError: Health did not return 200 within ``calls.health_timeout_s``.
    :raises RuntimeError: ``process`` exited first.
    """
    url = health_url("127.0.0.1", web_port)
    deadline = calls.monotonic() + calls.health_timeout_s
    while calls.http_status(url) != 200:
        if process is not None and process.poll() is not None:
            msg = f"Pulsar exited with {process.poll()} before {url} returned 200"
            raise RuntimeError(msg)
        if calls.monotonic() >= deadline:
            msg = (
                f"Broker health at {url} did not return 200 "
                f"after {calls.health_timeout_s:.0f}s"
            )
            raise TimeoutError(msg)
        calls.sleep(calls.poll_s)


def ensure_tenant_and_namespace(calls: PulsarCalls, web_port: int = WEB_PORT) -> None:
    """Create the ``arena`` tenant and the ``arena/run`` namespace.

    :param calls: Command runner and the Pulsar install root.
    :param web_port: Broker web service port.
    :raises RuntimeError: ``pulsar-admin`` failed for a reason other than the
        tenant or namespace already existing.
    """
    admin = str(calls.pulsar_home / "bin" / "pulsar-admin")
    admin_url = f"http://127.0.0.1:{web_port}"
    _run_admin(
        calls,
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
        calls,
        [admin, "--admin-url", admin_url, "namespaces", "create", "arena/run"],
    )


def stop_pulsar(process: PulsarProcess) -> None:
    """Terminate the standalone process and wait for it, unless it has exited.

    :param process: The standalone process.
    """
    if process.poll() is not None:
        return
    process.terminate()
    process.wait()


def pulsar_home_from(environ: Mapping[str, str]) -> Path:
    """Pulsar install root from ``PULSAR_HOME``, or the image's install path.

    :param environ: Process environment.
    :return: The install root.
    """
    home = environ.get("PULSAR_HOME", "").strip()
    return Path(home) if home else DEFAULT_PULSAR_HOME


def node_ip() -> str:
    """This container's IP.

    :return: The address the hostname resolves to.
    """
    return socket.gethostbyname(socket.gethostname())


def http_status(url: str) -> int:
    """Status code of a GET, or 0 when nothing answers.

    :param url: URL to fetch.
    :return: The HTTP status code.
    """
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return int(response.status)
    except urllib.error.HTTPError as exc:
        return int(exc.code)
    except OSError:
        return 0


def start_process(command: list[str], env: Mapping[str, str]) -> PulsarProcess:
    """Start ``command`` with ``env``.

    :param command: Argv.
    :param env: Process environment.
    :return: The running process.
    """
    return subprocess.Popen(command, env=dict(env))


def run_command(command: list[str]) -> CommandResult:
    """Run ``command`` to completion.

    :param command: Argv.
    :return: Exit code and combined stdout and stderr.
    """
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    return CommandResult(completed.returncode, completed.stdout + completed.stderr)


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


def _bookie_properties(
    journal_dir: Path, ledger_dir: Path, memory: BrokerMemory
) -> dict[str, str]:
    return {
        "journalDirectories": journal_dir.as_posix(),
        "ledgerDirectories": ledger_dir.as_posix(),
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


def _run_admin(calls: PulsarCalls, command: list[str]) -> None:
    result = calls.run_command(command)
    # Pulsar 4 reports "Tenant already exist" and "Namespace already exists".
    if result.code == 0 or "already exist" in result.output.lower():
        return
    msg = f"pulsar-admin failed ({result.code}): {result.output}"
    raise RuntimeError(msg)

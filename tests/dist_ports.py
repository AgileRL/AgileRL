# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""MASTER_PORT allocation that does not collide across pytest-xdist workers."""

from __future__ import annotations

import itertools
import os
import socket
import sys

port_counter = itertools.count()


def get_free_port() -> int:
    """Pick a MASTER_PORT that will not collide across xdist workers.

    Bind-to-port-0 / close / reuse is TOCTOU-racy: two workers can be handed
    the same ephemeral port and the loser dies with EADDRINUSE inside
    ``init_process_group``. Carve a disjoint 300-port range per xdist worker
    and walk it with a per-process counter, probing each candidate; fall back
    to an OS-assigned port only if the whole range is occupied.
    """
    worker = os.environ.get("PYTEST_XDIST_WORKER", "gw0")
    try:
        worker_num = int(worker.lstrip("gw"))
    except ValueError:
        worker_num = 0
    base = 20000 + (worker_num % 100) * 300
    for _ in range(300):
        port = base + next(port_counter) % 300
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def gloo_socket_ifname() -> str | None:
    """Loopback interface Gloo should bind, or None when this OS has no ``lo``."""
    if sys.platform == "darwin":
        return "lo0"
    if sys.platform.startswith("linux"):
        return "lo"
    return None


def gloo_rank_env(rank: int, world_size: int, port: int) -> dict[str, str]:
    """Rendezvous env for one Gloo rank. Omits ``GLOO_SOCKET_IFNAME`` on Windows."""
    env = {
        "RANK": str(rank),
        "LOCAL_RANK": str(rank),
        "WORLD_SIZE": str(world_size),
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(port),
    }
    ifname = gloo_socket_ifname()
    if ifname is not None:
        env["GLOO_SOCKET_IFNAME"] = ifname
    return env

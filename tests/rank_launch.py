# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Run a test worker on several ranks of one ``torch.distributed`` process group.

Ranks fork from a forkserver that imported torch, transformers, peft and
agilerl once, so a rank starts in milliseconds instead of re-importing that
stack. Each pytest-xdist worker owns its own forkserver, and ranks fork from
the server's clean state, never from a test's in-process mutations.

Ranks spawn instead on macOS, where forking after the Objective-C runtime has
threads crashes the child on its first Metal call, and on CUDA hosts, where
importing peft (bitsandbytes) initialises CUDA and a forked child cannot use it.
"""

from __future__ import annotations

import multiprocessing
import os
import shutil
import sys
import tempfile
import time
import traceback
from collections.abc import Callable
from multiprocessing.connection import Connection, wait
from multiprocessing.context import BaseContext
from multiprocessing.process import BaseProcess
from typing import Any
from unittest.mock import patch

import torch
import torch.distributed as dist

from tests.dist_ports import get_free_port, gloo_rank_env

RANK_PRELOAD = [
    "torch",
    "torch.distributed",
    "torch.distributed.tensor",
    "transformers",
    "peft",
    "agilerl.algorithms",
    "agilerl.distributed",
    "tests.rank_launch",
]


def rank_context() -> BaseContext:
    """Multiprocessing context for rank processes on this platform."""
    if sys.platform != "linux" or torch.cuda.is_available():
        return multiprocessing.get_context("spawn")
    ctx = multiprocessing.get_context("forkserver")
    ctx.set_forkserver_preload(RANK_PRELOAD)
    return ctx


def run_ranks(
    worker: Callable[..., Any],
    world_size: int = 2,
    *args: Any,
    backend: str | None = "gloo",
    timeout: float = 300.0,
    cpu: bool = False,
) -> list[Any]:
    """Run ``worker(rank, world_size, *args)`` on every rank and return the results.

    Each rank joins a ``backend`` process group before ``worker`` runs and leaves
    it afterwards; ``backend=None`` skips the group. The first rank that raises
    or dies fails the call, and the remaining ranks are killed.

    :param worker: Module-level function, so the rank process can import it.
    :param backend: ``"gloo"``, ``"nccl"`` (rank ``r`` uses ``cuda:r``) or None.
    :param cpu: Hide CUDA from the ranks, so distributed setup picks gloo and CPU.
    :return: Each rank's return value, in rank order.
    """
    ctx = rank_context()
    store_dir = tempfile.mkdtemp(prefix="rank-store-")
    store = os.path.join(store_dir, "store")
    port = get_free_port()
    cpu_env = {"CUDA_VISIBLE_DEVICES": ""} if cpu else {}
    pipes = [ctx.Pipe(duplex=False) for _ in range(world_size)]
    procs = [
        ctx.Process(
            target=rank_main,
            args=(
                dict(os.environ) | cpu_env,
                worker,
                rank,
                world_size,
                port,
                store,
                backend,
                args,
                pipes[rank][1],
            ),
        )
        for rank in range(world_size)
    ]
    # Spawned ranks start with this environment, and can initialise CUDA while
    # unpickling the worker, before ``rank_main`` applies its own.
    with patch.dict(os.environ, cpu_env):
        for proc in procs:
            proc.start()
    for _, sender in pipes:
        sender.close()
    readers = {pipes[rank][0]: rank for rank in range(world_size)}
    exits = {proc.sentinel: rank for rank, proc in enumerate(procs)}
    results: dict[int, Any] = {}
    deadline = time.monotonic() + timeout
    try:
        while len(results) < world_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                missing = sorted(set(range(world_size)) - set(results))
                msg = f"ranks {missing} did not finish within {timeout}s"
                raise TimeoutError(msg)
            for ready in wait([*readers, *exits], timeout=remaining):
                if ready in exits:
                    rank = exits.pop(ready)
                    # The result, or EOF, is read from the pipe. A child of the
                    # rank can still hold the pipe open after the rank exits.
                    if rank in results or pipes[rank][0].poll():
                        continue
                    raise rank_exited(procs[rank], rank)
                rank = readers.pop(ready)
                try:
                    error, payload = ready.recv()
                except EOFError:
                    raise rank_exited(procs[rank], rank) from None
                if error is not None:
                    msg = f"rank {rank} failed:\n{error}"
                    raise AssertionError(msg)
                results[rank] = payload
    finally:
        # After a failure the surviving ranks can block in a collective forever.
        grace = 30.0 if len(results) == world_size else 0.0
        for proc in procs:
            proc.join(timeout=grace)
            if proc.is_alive():
                proc.kill()
                proc.join()
        shutil.rmtree(store_dir, ignore_errors=True)
    return [results[rank] for rank in range(world_size)]


def rank_exited(proc: BaseProcess, rank: int) -> RuntimeError:
    """Error for a rank process that exited without sending a result."""
    proc.join()
    return RuntimeError(
        f"rank {rank} exited with code {proc.exitcode} before reporting"
    )


def rank_main(
    environ: dict[str, str],
    worker: Callable[..., Any],
    rank: int,
    world_size: int,
    port: int,
    store: str,
    backend: str | None,
    args: tuple[Any, ...],
    result: Connection,
) -> None:
    """Body of one rank process: join the group, run ``worker``, report back."""
    # Forkserver children start with the server's environment, not the test's.
    os.environ.clear()
    os.environ.update(environ)
    os.environ.update(gloo_rank_env(rank, world_size, port))
    try:
        if backend == "nccl":
            torch.cuda.set_device(rank)
        if backend is not None:
            dist.init_process_group(
                backend,
                init_method=f"file://{store}",
                rank=rank,
                world_size=world_size,
            )
        payload = worker(rank, world_size, *args)
        # A rank that tears down the group while a peer is still in its
        # handshake makes that peer fail with "Connection closed by peer".
        if dist.is_initialized():
            dist.barrier()
    except BaseException:
        result.send((traceback.format_exc(), None))
    else:
        result.send((None, payload))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

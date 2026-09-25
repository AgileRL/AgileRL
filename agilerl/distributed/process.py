# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Process group, rank, device, and collective helpers.

AgileRL is single-device by default. Multi-GPU LLM training initialises
``torch.distributed`` from the standard launcher environment variables
(``RANK``/``LOCAL_RANK``/``WORLD_SIZE``/``MASTER_ADDR``/``MASTER_PORT``),
which are set by ``torchrun`` or by exporting those variables. There is no
wrapper object: ``torch.distributed`` itself is the single source of truth
for rank/world topology, and these helpers no-op on a single device.
"""

from __future__ import annotations

import datetime
import os
import random
from collections.abc import Sequence
from contextlib import ContextDecorator
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import nn


def distributed_env_present() -> bool:
    """Whether launcher-style rendezvous env vars are set (torchrun)."""
    launcher_envs = {"RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"}
    return launcher_envs.issubset(os.environ)


def is_distributed() -> bool:
    """Whether ``torch.distributed`` is available and initialised."""
    return dist.is_available() and dist.is_initialized()


def init_distributed(timeout_seconds: int = 1800) -> bool:
    """Initialise ``torch.distributed`` from launcher env vars.

    No-op (returns ``False``) on a single device with no launcher env. Safe
    to call repeatedly; if a process group already exists it is reused.

    :param timeout_seconds: Collective timeout for the process group.
    :type timeout_seconds: int
    :return: ``True`` when distributed training is active.
    :rtype: bool
    """
    if not is_distributed():
        if not distributed_env_present():
            return False
        # Compound backend so CPU tensors (object broadcasts, metric
        # aggregation) go over gloo while CUDA tensors use nccl.
        backend = "cpu:gloo,cuda:nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(
            backend=backend,
            timeout=datetime.timedelta(seconds=timeout_seconds),
        )
    if torch.cuda.is_available():
        torch.cuda.set_device(get_local_rank())
    return True


def get_rank() -> int:
    """Global rank (0 on a single device)."""
    return dist.get_rank() if is_distributed() else 0


def get_local_rank() -> int:
    """Rank within the node, used for device selection."""
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return get_rank() % torch.cuda.device_count()
    return 0


def get_world_size() -> int:
    """Number of processes (1 on a single device)."""
    return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    """Whether this is rank 0 (always ``True`` on a single device)."""
    return get_rank() == 0


def barrier() -> None:
    """Synchronise all processes; no-op on a single device."""
    if is_distributed() and dist.get_world_size() > 1:
        dist.barrier()


def broadcast_object_list(objects: list, src: int = 0) -> list:
    """Broadcast a list of picklable objects from ``src`` to all ranks.

    Mutates ``objects`` in place on non-source ranks and returns it. No-op
    on a single device.

    :param objects: Objects to broadcast (same length on every rank).
    :type objects: list
    :param src: Source rank.
    :type src: int
    :return: The broadcast list.
    :rtype: list
    """
    if is_distributed() and dist.get_world_size() > 1:
        dist.broadcast_object_list(objects, src=src)
    return objects


def gather_objects(objects: list) -> list:
    """Flatten object lists gathered from every rank.

    Identity on a single device. Each rank contributes a picklable list;
    the result is concatenation in rank order.

    :param objects: Local objects to gather.
    :type objects: list
    :return: Flattened objects from all ranks.
    :rtype: list
    """
    if not is_distributed():
        return objects
    gathered: list[Any] = [None] * get_world_size()
    dist.all_gather_object(gathered, objects)
    return [item for rank_items in gathered for item in rank_items]


def gather_tensor(
    tensor: torch.Tensor | np.ndarray | float,
) -> torch.Tensor:
    """Gather a tensor from every rank (identity on a single device).

    :param tensor: Tensor (or array/scalar convertible to one) to gather.
    :return: Stacked / concatenated tensors from all ranks.
    """
    tensor = torch.as_tensor(tensor)
    if not is_distributed() or get_world_size() == 1:
        return tensor
    tensor = tensor.detach().to(resolve_device())
    gathered = [torch.empty_like(tensor) for _ in range(get_world_size())]
    dist.all_gather(gathered, tensor)
    return torch.stack(gathered) if tensor.dim() == 0 else torch.cat(gathered)


def allreduce_minmax_int(value: int) -> tuple[int, int]:
    """Return ``(min, max)`` of ``value`` across ranks (``(value, value)`` locally).

    :param value: This rank's value.
    :type value: int
    :return: ``(min, max)`` across ranks.
    :rtype: tuple[int, int]
    """
    value = int(value)
    if not is_distributed() or dist.get_world_size() == 1:
        return value, value
    # One MAX all-reduce yields both: max(-v) == -min(v).
    bounds = torch.tensor([value, -value], device=resolve_device(), dtype=torch.long)
    dist.all_reduce(bounds, op=dist.ReduceOp.MAX)
    max_value, neg_min_value = bounds.tolist()
    return -neg_min_value, max_value


def any_rank(flag: bool) -> bool:
    """True if any data-parallel rank has ``flag`` set.

    :param flag: This rank's flag.
    :type flag: bool
    :return: Whether any rank set ``flag``.
    :rtype: bool
    """
    _, mx = allreduce_minmax_int(int(flag))
    return mx == 1


def all_ranks(flag: bool) -> bool:
    """True only if every data-parallel rank has ``flag`` set.

    :param flag: This rank's flag.
    :type flag: bool
    :return: Whether every rank set ``flag``.
    :rtype: bool
    """
    mn, _ = allreduce_minmax_int(int(flag))
    return mn == 1


def aggregate_metrics_across_gpus(
    metric_tensor: torch.Tensor | np.ndarray | float,
) -> float:
    """Average a metric across ranks (local mean on a single device).

    :param metric_tensor: Metric values on this rank.
    :type metric_tensor: torch.Tensor | np.ndarray | float
    :return: Mean across all ranks.
    :rtype: float
    """
    local_mean = torch.as_tensor(metric_tensor).detach().float().mean()
    if not is_distributed() or dist.get_world_size() == 1:
        return local_mean.item()
    local_mean = local_mean.to(resolve_device())
    dist.all_reduce(local_mean, op=dist.ReduceOp.AVG)
    return local_mean.item()


def aggregate_metrics_dict(
    metrics: dict[str, torch.Tensor | np.ndarray | float],
) -> dict[str, float]:
    """Aggregate all values in a metrics dict across GPUs (or locally).

    :param metrics: Metric values on this rank, by name.
    :type metrics: dict[str, torch.Tensor | np.ndarray | float]
    :return: Mean of each metric across ranks.
    :rtype: dict[str, float]
    """
    return {k: aggregate_metrics_across_gpus(v) for k, v in metrics.items()}


class raise_on_any_rank(ContextDecorator):
    """Join all ranks after the block; raise on every rank if any rank failed.

    Use as ``with raise_on_any_rank():`` or ``@raise_on_any_rank()``. Does not
    recover a hang inside an in-flight NCCL op (``full_tensor`` / DCP).
    """

    def __enter__(self) -> None:
        return None

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        exc: BaseException | None,
        _tb: object,
    ) -> bool:
        if exc is not None and not isinstance(exc, Exception):
            return False
        if not is_distributed() or dist.get_world_size() == 1:
            return False
        failed = torch.tensor([int(exc is not None)], dtype=torch.int64)
        dist.all_reduce(failed, op=dist.ReduceOp.MAX)
        if exc is None and failed.item():
            msg = "Peer rank failed in shard runtime collective"
            raise RuntimeError(msg)
        return False


def sync_grads(params: Sequence[nn.Parameter]) -> None:
    """Average gradients across data-parallel ranks.

    One coalesced all-reduce of every ``.grad`` in ``params`` (SUM, then
    divide by world size so Gloo works). If any rank is missing a grad,
    every rank raises so NCCL cannot hang on mismatched flatten sizes.
    No-op on a single device.

    :param params: Optimizer parameters whose ``.grad`` should be averaged.
    """
    if not params or not is_distributed() or dist.get_world_size() == 1:
        return

    grads = [param.grad for param in params if param.grad is not None]
    missing = len(params) - len(grads)
    if any_rank(missing > 0):
        msg = f"sync_grads: {missing} params have no grad on rank {get_rank()}"
        raise RuntimeError(msg)

    flat = torch._utils._flatten_dense_tensors(grads)
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat /= dist.get_world_size()
    for g, synced in zip(
        grads, torch._utils._unflatten_dense_tensors(flat, grads), strict=True
    ):
        g.copy_(synced)


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (all devices).

    :param seed: Seed value.
    :type seed: int
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str | torch.device | None = None) -> str:
    """Pick the training device.

    Distributed runs are pinned to ``cuda:<local_rank>``, or CPU without CUDA;
    otherwise the requested device (or the best available) is used.

    :param requested: Device requested by the caller, if any.
    :type requested: str | torch.device | None
    :return: Device string.
    :rtype: str
    """
    if is_distributed() or distributed_env_present():
        # Gloo collectives cannot take MPS tensors.
        return f"cuda:{get_local_rank()}" if torch.cuda.is_available() else "cpu"
    if requested is not None:
        return str(requested)
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

"""Torch-native distributed helpers for AgileRL.

AgileRL is single-device by default. Multi-GPU LLM training initialises
``torch.distributed`` from the standard launcher environment variables
(``RANK``/``LOCAL_RANK``/``WORLD_SIZE``/``MASTER_ADDR``/``MASTER_PORT``),
which are set by ``torchrun`` or by an orchestration layer such as Ray
actors. There is no wrapper object: ``torch.distributed`` itself is the
single source of truth for rank/world topology, and these helpers no-op on
a single device.
"""

from __future__ import annotations

import datetime
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist
from torch import nn

try:
    from torch.distributed.fsdp import (
        CPUOffloadPolicy,
        MixedPrecisionPolicy,
        fully_shard,
    )

    HAS_FSDP2 = True
except ImportError:  # torch built without distributed support
    CPUOffloadPolicy = None
    MixedPrecisionPolicy = None
    fully_shard = None
    HAS_FSDP2 = False

_LAUNCHER_ENV_VARS = ("RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")


def distributed_env_present() -> bool:
    """Whether launcher-style rendezvous env vars are set (torchrun / Ray)."""
    return all(var in os.environ for var in _LAUNCHER_ENV_VARS)


def is_distributed() -> bool:
    """Whether ``torch.distributed`` is available and initialised."""
    return dist.is_available() and dist.is_initialized()


def init_distributed(timeout_seconds: int = 1800) -> bool:
    """Initialise ``torch.distributed`` from launcher env vars.

    No-op (returns ``False``) on a single device with no launcher env. Safe
    to call repeatedly; if a process group already exists (e.g. initialised
    by a Ray orchestration layer) it is reused.

    :param timeout_seconds: Collective timeout for the process group.
    :type timeout_seconds: int
    :return: ``True`` when distributed training is active.
    :rtype: bool
    """
    if is_distributed():
        if torch.cuda.is_available():
            torch.cuda.set_device(get_local_rank())
        return True
    if not distributed_env_present():
        return False
    backend = "nccl" if torch.cuda.is_available() else "gloo"
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
    if is_distributed():
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
    if is_distributed():
        dist.broadcast_object_list(objects, src=src)
    return objects


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """Average a tensor across ranks in place; no-op on a single device."""
    if is_distributed():
        dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
    return tensor


def sync_grads(params) -> None:
    """Average gradients across ranks (manual data-parallel synchronisation).

    AgileRL's LLM trainer is LoRA-only, so the trainable gradient set is
    tiny; averaging it explicitly at the gradient-accumulation boundary
    replaces a DDP wrapper outright — no module wrapping, no reducer
    hooks, no unused-parameter bookkeeping. No-op on a single device.

    :param params: Parameters whose ``.grad`` should be averaged.
    """
    if not is_distributed():
        return
    for param in params:
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (all devices)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str | torch.device | None = None) -> str:
    """Pick the training device.

    Distributed CUDA runs are pinned to ``cuda:<local_rank>``; otherwise the
    requested device (or the best available) is used.

    :param requested: Device requested by the caller, if any.
    :type requested: str | torch.device | None
    :return: Device string.
    :rtype: str
    """
    if torch.cuda.is_available() and (is_distributed() or distributed_env_present()):
        return f"cuda:{get_local_rank()}"
    if requested is not None:
        return str(requested)
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class FSDPConfig:
    """Settings for sharding the LLM actor with PyTorch FSDP2 (``fully_shard``).

    :param reshard_after_forward: Free gathered parameters after each
        module's forward (ZeRO-3-like memory profile).
    :type reshard_after_forward: bool
    :param cpu_offload: Offload sharded parameters/gradients to CPU.
    :type cpu_offload: bool
    :param param_dtype: Parameter dtype for the mixed-precision policy
        (``None`` keeps the model's dtype).
    :type param_dtype: torch.dtype | None
    :param reduce_dtype: Gradient-reduction dtype for the mixed-precision
        policy (``None`` keeps the model's dtype).
    :type reduce_dtype: torch.dtype | None
    """

    reshard_after_forward: bool = True
    cpu_offload: bool = False
    param_dtype: torch.dtype | None = None
    reduce_dtype: torch.dtype | None = None


def _transformer_blocks(model: nn.Module) -> list[nn.Module]:
    """Find per-layer transformer blocks to shard individually.

    Uses HuggingFace's ``_no_split_modules`` class-name convention where
    available; falls back to an empty list (root-only sharding).
    """
    no_split: set[str] = set()
    for module in model.modules():
        names = getattr(module, "_no_split_modules", None)
        if names:
            no_split.update(names)
    if not no_split:
        return []
    return [m for m in model.modules() if type(m).__name__ in no_split]


def apply_fsdp2(model: nn.Module, config: FSDPConfig | None = None) -> nn.Module:
    """Shard ``model`` with FSDP2, one group per transformer block plus root.

    The model should already be on the target CUDA device. Parameters are
    swapped to DTensors in place, so any optimizer must be (re)built after
    this call.

    :param model: Model to shard.
    :type model: nn.Module
    :param config: Sharding settings; defaults to :class:`FSDPConfig`'s
        defaults.
    :type config: FSDPConfig | None
    :return: The sharded model (same object).
    :rtype: nn.Module
    """
    if not HAS_FSDP2:
        msg = "FSDP2 requires a torch build with distributed support."
        raise RuntimeError(msg)
    if not is_distributed():
        msg = (
            "FSDP2 sharding requires an initialised process group. Launch "
            "with torchrun (or have your orchestration layer set the "
            "rendezvous env vars) so init_distributed() succeeds."
        )
        raise RuntimeError(msg)
    config = config or FSDPConfig()

    kwargs: dict = {"reshard_after_forward": config.reshard_after_forward}
    if config.cpu_offload:
        kwargs["offload_policy"] = CPUOffloadPolicy()
    if config.param_dtype is not None or config.reduce_dtype is not None:
        kwargs["mp_policy"] = MixedPrecisionPolicy(
            param_dtype=config.param_dtype,
            reduce_dtype=config.reduce_dtype,
        )

    for block in _transformer_blocks(model):
        fully_shard(block, **kwargs)
    fully_shard(model, **kwargs)
    return model


def shard_dataloader_kwargs(dataset, shuffle: bool = True) -> dict:
    """DataLoader kwargs that shard ``dataset`` across ranks.

    Returns ``{"sampler": DistributedSampler(...)}`` when distributed (the
    caller must not also pass ``shuffle``), or ``{"shuffle": shuffle}`` on a
    single device.

    :param dataset: Map-style dataset to shard.
    :param shuffle: Whether to shuffle.
    :type shuffle: bool
    :return: Keyword arguments for ``torch.utils.data.DataLoader``.
    :rtype: dict
    """
    if is_distributed():
        from torch.utils.data.distributed import DistributedSampler

        return {
            "sampler": DistributedSampler(
                dataset,
                num_replicas=get_world_size(),
                rank=get_rank(),
                shuffle=shuffle,
            )
        }
    return {"shuffle": shuffle}

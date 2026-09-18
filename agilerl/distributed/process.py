# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Process group, rank, device, and FSDP2 wrap helpers.

AgileRL is single-device by default. Multi-GPU LLM training initialises
``torch.distributed`` from the standard launcher environment variables
(``RANK``/``LOCAL_RANK``/``WORLD_SIZE``/``MASTER_ADDR``/``MASTER_PORT``),
which are set by ``torchrun`` or by exporting those variables. There is no
wrapper object: ``torch.distributed`` itself is the single source of truth
for rank/world topology, and these helpers no-op on a single device.
"""

from __future__ import annotations

import copy
import datetime
import gc
import os
import random
from collections.abc import Callable, Generator, Iterable, Sequence
from contextlib import ContextDecorator, contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    checkpoint_wrapper,
)
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    FSDPModule,
    MixedPrecisionPolicy,
    fully_shard,
    register_fsdp_forward_method,
    share_comm_ctx,
)
from torch.distributed.tensor import DTensor
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

from agilerl.utils.patching import class_is_patched


def distributed_env_present() -> bool:
    """Whether launcher-style rendezvous env vars are set (torchrun)."""
    launcher_envs = {"RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"}
    return launcher_envs.issubset(os.environ)


def is_distributed() -> bool:
    """Whether ``torch.distributed`` is available and initialised."""
    return dist.is_available() and dist.is_initialized()


def is_fsdp_sharded(model: nn.Module) -> bool:
    """Return ``True`` when ``model`` is an FSDP2-sharded root (``fully_shard``)."""
    return isinstance(model, FSDPModule)


def init_distributed(timeout_seconds: int = 1800) -> bool:
    """Initialise ``torch.distributed`` from launcher env vars.

    No-op (returns ``False``) on a single device with no launcher env. Safe
    to call repeatedly; if a process group already exists it is reused.

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


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """Average a tensor across ranks in place; no-op on a single device."""
    if is_distributed() and dist.get_world_size() > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
    return tensor


def gather_tensor(
    tensor: torch.Tensor | np.ndarray | float,
) -> torch.Tensor:
    """Gather a tensor from every rank (identity on a single device).

    Prefer ``torch.distributed`` when a process group is initialised.

    :param tensor: Tensor (or array/scalar convertible to one) to gather.
    :return: Stacked / concatenated tensors from all ranks.
    """
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)
    if not is_distributed():
        return tensor
    tensor = tensor.detach().to(torch.device(resolve_device()))
    gathered = [torch.empty_like(tensor) for _ in range(get_world_size())]
    dist.all_gather(gathered, tensor)
    return torch.stack(gathered) if tensor.dim() == 0 else torch.cat(gathered)


def allreduce_minmax_int(value: int) -> tuple[int, int]:
    """Return ``(min, max)`` of ``value`` across ranks via torch.distributed."""
    if not is_distributed():
        v = int(value)
        return v, v
    t = torch.tensor(
        [int(value)], device=torch.device(resolve_device()), dtype=torch.long
    )
    gathered = gather_tensor(t)
    return int(gathered.min().item()), int(gathered.max().item())


def any_rank(flag: bool) -> bool:
    """True if any data-parallel rank has ``flag`` set."""
    _, mx = allreduce_minmax_int(int(flag))
    return mx == 1


def all_ranks(flag: bool) -> bool:
    """True only if every data-parallel rank has ``flag`` set."""
    mn, _ = allreduce_minmax_int(int(flag))
    return mn == 1


def aggregate_metrics_across_gpus(
    metric_tensor: torch.Tensor | np.ndarray | float,
) -> float:
    """Average a metric across ranks (local mean on a single device)."""
    if not is_distributed():
        if isinstance(metric_tensor, torch.Tensor):
            return metric_tensor.float().mean().item()
        if isinstance(metric_tensor, np.ndarray):
            return float(np.mean(metric_tensor))
        return float(metric_tensor)
    if not isinstance(metric_tensor, torch.Tensor):
        metric_tensor = torch.as_tensor(metric_tensor)
    local_mean = metric_tensor.detach().float().mean()
    return all_reduce_mean(local_mean.to(torch.device(resolve_device()))).item()


def aggregate_metrics_dict(
    metrics: dict[str, torch.Tensor | np.ndarray | float],
) -> dict[str, float]:
    """Aggregate all values in a metrics dict across GPUs (or locally)."""
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
        err = exc
        flag = 1 if err is not None else 0
        if is_distributed() and dist.get_world_size() > 1:
            flag_tensor = torch.tensor([flag], dtype=torch.int64)
            dist.all_reduce(flag_tensor, op=dist.ReduceOp.MAX)
            if int(flag_tensor.item()) > 0:
                if err is not None:
                    return False
                msg = "Peer rank failed in shard runtime collective"
                raise RuntimeError(msg)
        return False


def set_full_model_state_dict(
    model: nn.Module,
    state_dict: dict[str, Any],
    *,
    strict: bool = False,
) -> None:
    """Scatter a full (unsharded) state dict onto an FSDP2-sharded model."""
    set_model_state_dict(
        model,
        state_dict,
        options=StateDictOptions(full_state_dict=True, strict=strict),
    )


def get_full_model_state_dict(
    model: nn.Module, *, cpu_offload: bool = True
) -> dict[str, Any]:
    """Gather a full (unsharded) state dict from an FSDP2-sharded model.

    Only rank 0 receives the tensors when ``cpu_offload`` is ``True``; other
    ranks get an empty dict. A GPU full state dict is rejected.
    """
    # TODO should really save models in shards here to avoid OOM from all_gather of all params
    if not cpu_offload:
        msg = (
            "get_full_model_state_dict(cpu_offload=False) is not supported "
            "for FSDP2 models: a full GPU state dict would place the entire "
            "model on every rank. Use cpu_offload=True for checkpoints, or "
            "adapter-scoped helpers for clones."
        )
        raise ValueError(msg)
    return get_model_state_dict(
        model,
        options=StateDictOptions(full_state_dict=True, cpu_offload=True),
    )


@contextmanager
def materialize_dtensors(
    *tensors: torch.Tensor | None,
) -> Generator[list[torch.Tensor | None], None, None]:
    """All-gather ``DTensor`` shards to dense locals without swapping module params.

    Prefer this for ephemeral matmuls (fused lm_head logprobs, Liger). Use
    :func:`gather_params` when in-module reads must
    see dense weights (``state_dict`` / PEFT ``save_pretrained``). All ranks
    must enter and exit together. Yields a list parallel to ``tensors``.
    """
    gathered_tensors: list[torch.Tensor | None] = []
    for tensor in tensors:
        if tensor is None or not isinstance(tensor, DTensor):
            gathered_tensors.append(tensor)
        else:
            gathered_tensors.append(tensor.full_tensor())
    yield gathered_tensors


def parameter_owner(param: torch.Tensor) -> tuple[nn.Module, str] | None:
    """Return ``(module, attr_name)`` for a parameter registered on a module."""
    for referrer in gc.get_referrers(param):
        if not isinstance(referrer, dict):
            continue
        attr_names = [
            key
            for key, value in referrer.items()
            if value is param and isinstance(key, str)
        ]
        if not attr_names:
            continue
        for obj in gc.get_referrers(referrer):
            if (
                isinstance(obj, nn.Module)
                and getattr(obj, "_parameters", None) is referrer
            ):
                return obj, attr_names[0]
    return None


@contextmanager
def full_shape_views(
    params: Sequence[torch.Tensor | None],
) -> Generator[None, None, None]:
    """Expose global-shape views on FSDP2 ``DTensor`` params for shape-only reads.

    Each sharded param is temporarily replaced on its owning module with a
    zero-storage view (a scalar expanded to the DTensor global shape) so
    shape, dtype and device reads see the full tensor without
    :meth:`~torch.distributed.tensor.DTensor.full_tensor`. Values must not
    be read inside the block; the original ``DTensor`` is restored on exit
    when the installed view is still the live parameter. Plain tensors pass
    through untouched. Duplicate references are processed once (by identity).
    """
    restores: list[tuple[nn.Module, str, DTensor, nn.Parameter]] = []
    seen: set[int] = set()
    try:
        for param in params:
            if param is None or not isinstance(param, DTensor):
                continue
            if id(param) in seen:
                continue
            owner = parameter_owner(param)
            if owner is None:
                continue
            seen.add(id(param))
            module, name = owner
            view = torch.empty((), dtype=param.dtype, device=param.device).expand(
                tuple(param.shape)
            )
            view_param = nn.Parameter(view, requires_grad=bool(param.requires_grad))
            owned: dict[str, Any] = module._parameters
            owned[name] = view_param
            restores.append((module, name, param, view_param))
        yield
    finally:
        for module, name, original, view_param in reversed(restores):
            owned: dict[str, Any] = module._parameters
            if owned.get(name) is view_param:
                owned[name] = original


@contextmanager
def gather_params(
    params: Sequence[torch.Tensor | None],
) -> Generator[list[torch.Tensor | None], None, None]:
    """Materialize full (unsharded) views of ``params`` for the duration of the context.

    Plain tensors are left unchanged. For FSDP2 ``DTensor`` parameters, each
    tensor is all-gathered with :meth:`~torch.distributed.tensor.DTensor.full_tensor`
    and temporarily installed on its owning module so in-module reads (e.g.
    ``state_dict`` / PEFT ``save_pretrained``) see dense weights. Original
    shards are restored on exit.

    Yields a list parallel to ``params``: dense locals for any gathered
    ``DTensor``, and the original handles otherwise. Callers that hold
    pre-gather tensor references must use the yielded list for math — those
    references still point at the shard. For matmul-only gathers prefer
    :func:`materialize_dtensors` (no module Parameter install).

    Gathered parameters are read-only: writes are discarded when the sharded
    ``DTensor`` is restored. Write into a sharded model with
    :func:`set_full_model_state_dict`. All ranks must enter and exit together.
    """
    restores: list[tuple[nn.Module, str, torch.Tensor]] = []
    gathered_tensors: list[torch.Tensor | None] = []
    try:
        for param in params:
            if param is None or not isinstance(param, DTensor):
                gathered_tensors.append(param)
                continue
            full = param.full_tensor()
            gathered_tensors.append(full)
            owner = parameter_owner(param)
            if owner is None:
                continue
            module, name = owner
            restores.append((module, name, param))
            owned: dict[str, Any] = module._parameters
            owned[name] = nn.Parameter(full, requires_grad=bool(param.requires_grad))
        yield gathered_tensors
    finally:
        for module, name, original in reversed(restores):
            owned: dict[str, Any] = module._parameters
            owned[name] = original


def sync_grads(params: Sequence[nn.Parameter]) -> None:
    """Average gradients across data-parallel ranks.

    One coalesced all-reduce of every ``.grad`` in ``params`` (SUM, then
    divide by world size so Gloo works). If any rank is missing a grad,
    every rank raises so NCCL cannot hang on mismatched flatten sizes.
    No-op on a single device.

    :param params: Optimizer parameters whose ``.grad`` should be averaged.
    """
    if not is_distributed() or dist.get_world_size() <= 1:
        return
    if not params:
        return

    missing = sum(p.grad is None for p in params)
    _, max_missing = allreduce_minmax_int(missing)
    if max_missing > 0:
        msg = f"sync_grads: {missing} params have no grad on rank {get_rank()}"
        raise RuntimeError(msg)

    grads = [p.grad for p in params]
    flat = torch._utils._flatten_dense_tensors(grads)
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat /= dist.get_world_size()
    for g, synced in zip(
        grads, torch._utils._unflatten_dense_tensors(flat, grads), strict=True
    ):
        g.copy_(synced)


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
        module's forward.
    :type reshard_after_forward: bool
    :param cpu_offload: Offload sharded parameters/gradients to CPU.
    :type cpu_offload: bool
    :param optim_cpu_offload: Offload optimizer states (AdamW m+v) to CPU,
        moving them to GPU only during ``step()``.  Parameters and gradients
        stay on GPU.  Mutually exclusive with ``cpu_offload``.  Defaults to
        ``True`` — benchmarked as memory-neutral and faster with colocated vLLM.
    :type optim_cpu_offload: bool
    :param defer_grad_sync: Skip FSDP2 reduce-scatter until the last
        micro-batch of an optimizer step (``FSDPModule.set_requires_gradient_sync``).
        Saves communication; holds unsharded grads between micro-batches.
        ``False`` reduce-scatters every backward (FSDP2 default).
    :type defer_grad_sync: bool
    :param param_dtype: FSDP2 mixed-precision parameter dtype.
    :type param_dtype: torch.dtype
    :param reduce_dtype: Dtype of reduce-scatter / all-reduce.
    :type reduce_dtype: torch.dtype
    :param prefetch_units: How many neighbouring FSDP units to prefetch
        during forward and backward. ``1`` issues all-gather for i+1
        while unit i runs.
    :type prefetch_units: int
    :param wrap_every_n_blocks: Consecutive transformer blocks per FSDP
        unit. ``1`` shards each block. Larger values mean fewer, bigger
        reduce-scatters.
    :type wrap_every_n_blocks: int
    :param param_persistence_threshold: Parameters with fewer elements than
        this stay unsharded. ``0`` shards every parameter.
    :type param_persistence_threshold: int
    """

    reshard_after_forward: bool = True
    cpu_offload: bool = False
    optim_cpu_offload: bool = True
    defer_grad_sync: bool = True
    param_dtype: torch.dtype = torch.bfloat16
    reduce_dtype: torch.dtype = torch.float32
    prefetch_units: int = 1
    wrap_every_n_blocks: int = 1
    param_persistence_threshold: int = 100_000

    def __post_init__(self) -> None:
        self.param_dtype = _coerce_torch_dtype(self.param_dtype)
        self.reduce_dtype = _coerce_torch_dtype(self.reduce_dtype)


def _coerce_torch_dtype(value: torch.dtype | str) -> torch.dtype:
    """Accept a ``torch.dtype`` or a name such as ``bfloat16`` / ``torch.bfloat16``."""
    if isinstance(value, torch.dtype):
        return value
    name = str(value).removeprefix("torch.")
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        msg = f"Unknown torch dtype {value!r}"
        raise ValueError(msg)
    return dtype


def _transformer_blocks(model: nn.Module) -> list[nn.Module]:
    """Outermost HuggingFace ``_no_split_modules`` hits.

    Nested no_split modules (packed experts inside a decoder layer) stay
    leaves of the parent FSDP unit. Empty when the model has no no-split
    names (root-only sharding).
    """
    no_split: set[str] = set()
    for module in model.modules():
        names = getattr(module, "_no_split_modules", None)
        if names:
            no_split.update(names)
    if not no_split:
        return []
    hits = [module for module in model.modules() if type(module).__name__ in no_split]
    nested: set[int] = set()
    for hit in hits:
        for child in hit.modules():
            if child is hit or type(child).__name__ not in no_split:
                continue
            nested.add(id(child))
    return [module for module in hits if id(module) not in nested]


def _unwrap_checkpoint(module: nn.Module) -> nn.Module:
    """Inner module when ``module`` is an activation-checkpoint wrapper."""
    inner = getattr(module, "_checkpoint_wrapped_module", None)
    return inner if isinstance(inner, nn.Module) else module


class FSDPBlockGroup(nn.Module):
    """Consecutive transformer blocks treated as one FSDP2 unit."""

    def __init__(self, blocks: Sequence[nn.Module]) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(list(blocks))
        first = _unwrap_checkpoint(self.blocks[0])
        # HF indexes ``layer.block_type`` / ``layer_idx`` on ModuleList entries.
        self.block_type = getattr(first, "block_type", None)
        self.layer_idx = getattr(first, "layer_idx", None)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        mask = kwargs.get("attention_mask")
        result: torch.Tensor | tuple[torch.Tensor, ...] = hidden_states
        for block in self.blocks:
            hidden = result[0] if isinstance(result, tuple) else result
            if isinstance(mask, dict):
                raw = _unwrap_checkpoint(block)
                block_kwargs = dict(kwargs)
                block_kwargs["attention_mask"] = mask.get(
                    getattr(raw, "block_type", None)
                )
                result = block(hidden, *args, **block_kwargs)
            else:
                result = block(hidden, *args, **kwargs)
        return result


def _replace_consecutive_blocks(
    root: nn.Module, blocks: Sequence[nn.Module]
) -> FSDPBlockGroup:
    """Lift a consecutive ``ModuleList`` span into an :class:`FSDPBlockGroup`."""
    target = list(blocks)
    if not target:
        msg = "blocks must be non-empty"
        raise ValueError(msg)
    for parent in root.modules():
        if not isinstance(parent, nn.ModuleList):
            continue
        children = list(parent)
        span = len(target)
        for start in range(len(children) - span + 1):
            if children[start : start + span] != target:
                continue
            for offset in range(span - 1, -1, -1):
                del parent[start + offset]
            group = FSDPBlockGroup(target)
            parent.insert(start, group)
            return group
    msg = "Could not find consecutive ModuleList span for FSDP block group"
    raise RuntimeError(msg)


def _group_transformer_units(
    model: nn.Module, units: Sequence[nn.Module], every_n: int
) -> list[nn.Module]:
    """Bundle consecutive wrap targets into groups of ``every_n``."""
    if every_n < 1:
        msg = "wrap_every_n_blocks must be >= 1"
        raise ValueError(msg)
    unit_list = list(units)
    if every_n == 1 or len(unit_list) <= 1:
        return unit_list
    grouped: list[nn.Module] = []
    for start in range(0, len(unit_list), every_n):
        chunk = unit_list[start : start + every_n]
        if len(chunk) == 1:
            grouped.append(chunk[0])
            continue
        grouped.append(_replace_consecutive_blocks(model, chunk))
    return grouped


def _replace_child(root: nn.Module, old: nn.Module, new: nn.Module) -> None:
    """Swap ``old`` for ``new`` on its parent under ``root``."""
    for parent in root.modules():
        for name, child in parent.named_children():
            if child is old:
                setattr(parent, name, new)
                return
    msg = f"Could not find parent module for {type(old).__name__}"
    raise RuntimeError(msg)


def _resolve_causal_lm(model: nn.Module) -> nn.Module:
    """Unwrap value-head / PEFT shells to the HuggingFace causal LM."""
    causal = model
    if hasattr(causal, "pretrained_model"):
        causal = causal.pretrained_model
    get_base = getattr(causal, "get_base_model", None)
    if callable(get_base):
        try:
            return get_base()
        except Exception:  # pragma: no cover -- PEFT API edge cases
            pass
    if hasattr(causal, "base_model"):
        causal = causal.base_model
        inner = getattr(causal, "model", None)
        if inner is not None and (
            hasattr(inner, "lm_head")
            or hasattr(inner, "embed_out")
            or hasattr(inner, "model")
        ):
            return inner
    return causal


def _language_model(causal: nn.Module) -> nn.Module | None:
    """Return the transformer body that owns ``embed_tokens`` / ``layers``."""
    inner = getattr(causal, "model", None)
    if inner is not None and (
        hasattr(inner, "embed_tokens")
        or hasattr(inner, "embeddings")
        or hasattr(inner, "layers")
    ):
        return inner
    if hasattr(causal, "embed_tokens") or hasattr(causal, "layers"):
        return causal
    return None


def _owned_parameters(module: nn.Module) -> set[nn.Parameter]:
    """Parameters of ``module`` not already owned by a child FSDP unit."""
    owned = set(module.parameters())
    for child in module.modules():
        if child is module or not isinstance(child, FSDPModule):
            continue
        owned.difference_update(child.parameters())
    return owned


def _persistent_params(module: nn.Module, threshold: int) -> set[nn.Parameter]:
    """Unsharded parameters: ``numel`` below ``threshold``."""
    if threshold <= 0:
        return set()
    return {param for param in _owned_parameters(module) if param.numel() < threshold}


def _cast_params(params: Iterable[nn.Parameter], dtype: torch.dtype) -> None:
    """Put unsharded parameters in the mixed-precision compute dtype."""
    for param in params:
        if param.dtype != dtype:
            param.data = param.data.to(dtype=dtype)


def _shard_unit(
    module: nn.Module, shard_kwargs: dict, persistence_threshold: int
) -> None:
    """``fully_shard`` ``module``, leaving tiny owned parameters replicated."""
    ignored = _persistent_params(module, persistence_threshold)
    if ignored:
        mp_policy = shard_kwargs.get("mp_policy")
        param_dtype = getattr(mp_policy, "param_dtype", None)
        if param_dtype is not None:
            _cast_params(ignored, param_dtype)
        shard_kwargs = dict(shard_kwargs)
        shard_kwargs["ignored_params"] = ignored
    fully_shard(module, **shard_kwargs)


def _shard_embed_and_lm_head(
    model: nn.Module, shard_kwargs: dict, persistence_threshold: int
) -> None:
    """Shard token embeddings (and untied ``lm_head``) as their own FSDP units.

    Matches Prime-RL's layout: embeddings reshards after use; an untied
    ``lm_head`` is a separate unit so it is not held unsharded with the root.
    ``lm_head`` is sharded alone (not joint with final norm) so an
    identity-patched backbone forward does not all-gather the head when only
    the norm runs. An untied head keeps ``reshard_after_forward=False`` so
    the last all-gather stays live into the unembedding. Tied embeddings
    skip the head unit so tying stays intact.
    """
    causal = _resolve_causal_lm(model)
    language = _language_model(causal)
    if language is None:
        return

    embed = getattr(language, "embed_tokens", None) or getattr(
        language, "embeddings", None
    )
    if embed is not None:
        _shard_unit(embed, shard_kwargs, persistence_threshold)

    config = getattr(causal, "config", None)
    if config is not None and bool(getattr(config, "tie_word_embeddings", False)):
        return

    lm_head = getattr(causal, "lm_head", None) or getattr(causal, "embed_out", None)
    if lm_head is not None:
        head_kwargs = dict(shard_kwargs)
        head_kwargs["reshard_after_forward"] = False
        _shard_unit(lm_head, head_kwargs, persistence_threshold)


def _set_prefetch(
    model: nn.Module,
    block_units: Sequence[nn.Module],
    prefetch_units: int = 1,
) -> None:
    """Overlap neighbour FSDP all-gathers with the current unit's compute.

    Dense path only: embed → first block, consecutive transformer blocks,
    last block → ``lm_head``. Forward prefetches the next ``prefetch_units``
    modules. Backward prefetches the previous ``prefetch_units`` so layer
    i-1 gathers while layer i runs backward. ``block_units`` are the
    modules just ``fully_shard``ed (the checkpoint wrappers when
    activation checkpointing is on). Walking ``_transformer_blocks`` after
    wrap would also see inner decoder layers.
    """
    if prefetch_units < 1:
        msg = "prefetch_units must be >= 1"
        raise ValueError(msg)
    units: list[Any] = []
    causal = _resolve_causal_lm(model)
    language = _language_model(causal)
    if language is not None:
        embed = getattr(language, "embed_tokens", None) or getattr(
            language, "embeddings", None
        )
        if isinstance(embed, FSDPModule):
            units.append(embed)
    units.extend(unit for unit in block_units if isinstance(unit, FSDPModule))
    lm_head = getattr(causal, "lm_head", None) or getattr(causal, "embed_out", None)
    if isinstance(lm_head, FSDPModule):
        units.append(lm_head)

    for index, current in enumerate(units):
        nxt = units[index + 1 : index + 1 + prefetch_units]
        if nxt:
            current.set_modules_to_forward_prefetch(list(nxt))
        prev = units[max(0, index - prefetch_units) : index]
        if prev:
            current.set_modules_to_backward_prefetch(list(reversed(prev)))


def _inner_block_types(group: FSDPBlockGroup) -> set[object]:
    """``block_type`` of each inner decoder block in ``group``."""
    types: set[object] = set()
    for block in group.blocks:
        raw = _unwrap_checkpoint(block)
        types.add(getattr(raw, "block_type", None))
    return types


def _mixed_fsdp_groups(module: nn.Module) -> list[FSDPBlockGroup]:
    """Grouped wrap units whose inner blocks do not share one ``block_type``."""
    return [
        child
        for child in module.modules()
        if isinstance(child, FSDPBlockGroup) and len(_inner_block_types(child)) > 1
    ]


def _install_grouped_mask_forward(language: nn.Module) -> None:
    """Pass the full mask dict into mixed-type ``FSDPBlockGroup`` units.

    HuggingFace does ``mapping.get(layer.block_type)`` before calling the layer.
    """
    cls = type(language)
    if class_is_patched(cls, "_agilerl_grouped_fsdp_forward_patched"):
        return
    original = cls.forward

    def wrapped(
        self: nn.Module,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: object | None = None,
        use_cache: bool | None = None,
        attention_mask: torch.Tensor | dict[str, object] | None = None,
        **kwargs: object,
    ) -> object:
        mixed = _mixed_fsdp_groups(self)
        mapping = attention_mask if isinstance(attention_mask, dict) else None
        if mapping is None and mixed:
            needs_linear = any(
                "linear_attention" in _inner_block_types(group) for group in mixed
            )
            embeds = inputs_embeds
            if embeds is None and input_ids is not None:
                embed_fn = getattr(self, "embeddings", None) or getattr(
                    self, "embed_tokens", None
                )
                if callable(embed_fn):
                    embeds = embed_fn(input_ids)
            if needs_linear and embeds is not None:
                # Nemotron hybrid mask mapping; this architecture is not
                # loaded for the rest of this module.
                from agilerl.architectures.nemotron_h.mamba import (
                    block_type_mask_mapping,
                )

                mapping, position_ids = block_type_mask_mapping(
                    self,
                    embeds=embeds,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                )
                if inputs_embeds is None:
                    inputs_embeds = embeds
                    input_ids = None
        restored: list[tuple[FSDPBlockGroup, object]] = []
        if mapping and mixed:
            mapping = dict(mapping)
            payload = dict(mapping)
            mapping["_agilerl_fsdp_group"] = payload
            attention_mask = mapping
            for group in mixed:
                restored.append((group, group.block_type))
                group.block_type = "_agilerl_fsdp_group"
        try:
            return original(
                self,
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                attention_mask=attention_mask,
                **kwargs,
            )
        finally:
            for group, previous in restored:
                group.block_type = previous

    cls.forward = wrapped
    type.__setattr__(cls, "_agilerl_grouped_fsdp_forward_patched", True)


def apply_fsdp2(
    model: nn.Module,
    config: FSDPConfig | None = None,
    *,
    mesh: DeviceMesh | None = None,
    gradient_checkpointing: bool = False,
) -> nn.Module:
    """Shard ``model`` with FSDP2: blocks, embed/(untied) lm_head, root, prefetch.

    Parameters become DTensors in place, so any optimizer must be (re)built
    after this call. Callers must not pass a dense full-model replica on
    CUDA — use :func:`materialize_fsdp2_from_cpu_state` so weights stay on
    CPU/meta until only local shards are allocated on the compute device.

    When ``mesh`` is provided, ``fully_shard`` shards over that device mesh;
    when ``None``, FSDP uses the default process group (flat path).

    When ``gradient_checkpointing`` is on, each transformer block is wrapped
    with ``checkpoint_wrapper`` before ``fully_shard`` so the checkpoint
    boundary sits inside the FSDP unit.

    :param model: Model to shard (CPU or meta parameters).
    :type model: nn.Module
    :param config: Sharding settings; defaults to :class:`FSDPConfig`'s
        defaults.
    :type config: FSDPConfig | None
    :param mesh: Optional FSDP device mesh. Ignored when ``None``.
    :type mesh: DeviceMesh | None
    :param gradient_checkpointing: Wrap each transformer block with
        non-reentrant activation checkpointing before sharding.
    :type gradient_checkpointing: bool
    :return: The sharded model (same object).
    :rtype: nn.Module
    """
    if not is_distributed():
        msg = (
            "FSDP2 sharding requires an initialised process group. Launch "
            "with torchrun (or set RANK, WORLD_SIZE, MASTER_ADDR, and "
            "MASTER_PORT) so init_distributed() succeeds."
        )
        raise RuntimeError(msg)
    config = config or FSDPConfig()
    if config.prefetch_units < 1:
        msg = "FSDPConfig.prefetch_units must be >= 1"
        raise ValueError(msg)
    if config.wrap_every_n_blocks < 1:
        msg = "FSDPConfig.wrap_every_n_blocks must be >= 1"
        raise ValueError(msg)
    if config.param_persistence_threshold < 0:
        msg = "FSDPConfig.param_persistence_threshold must be >= 0"
        raise ValueError(msg)

    kwargs: dict = {
        "reshard_after_forward": config.reshard_after_forward,
        "mp_policy": MixedPrecisionPolicy(
            param_dtype=config.param_dtype,
            reduce_dtype=config.reduce_dtype,
        ),
    }
    if config.cpu_offload:
        kwargs["offload_policy"] = CPUOffloadPolicy()
    if mesh is not None:
        kwargs["mesh"] = mesh

    # Packed-expert skip lives in moe_lora; algorithms.core imports this
    # module via base, so the import stays in this function.
    from agilerl.algorithms.core.llm_ops.moe_lora import _is_packed_experts_module

    wrap_units: list[nn.Module] = []
    for block in _transformer_blocks(model):
        if _is_packed_experts_module(block):
            continue
        unit = block
        if gradient_checkpointing:
            unit = checkpoint_wrapper(
                block,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                preserve_rng_state=False,
            )
            _replace_child(model, block, unit)
        wrap_units.append(unit)
    sharded_blocks = _group_transformer_units(
        model, wrap_units, config.wrap_every_n_blocks
    )
    if _mixed_fsdp_groups(model):
        language = _language_model(_resolve_causal_lm(model))
        if language is not None:
            _install_grouped_mask_forward(language)
    threshold = config.param_persistence_threshold
    for unit in sharded_blocks:
        _shard_unit(unit, kwargs, threshold)
    _shard_embed_and_lm_head(model, kwargs, threshold)
    _shard_unit(model, kwargs, threshold)
    _set_prefetch(model, sharded_blocks, config.prefetch_units)
    # PEFT ``generate`` delegates to ``base_model.generate`` and never enters
    # the FSDP-rooted ``forward``, so root shards (e.g. embed_tokens) stay
    # DTensors against plain ``input_ids``. Register ``generate`` so FSDP2
    # all-gathers the same way as ``forward``. Register ``forward`` so a
    # direct ``module.forward(...)`` still runs root ``_pre_forward``.
    assert register_fsdp_forward_method is not None
    if hasattr(model, "generate"):
        register_fsdp_forward_method(model, "generate")
    register_fsdp_forward_method(model, "forward")
    return model


def _restore_after_to_empty(model: nn.Module) -> None:
    """Restore tying and modules that ``to_empty`` leaves unbound."""
    seen: set[int] = set()
    for module in model.modules():
        tie = getattr(module, "tie_weights", None)
        if not callable(tie):
            continue
        module_id = id(module)
        if module_id in seen:
            continue
        seen.add(module_id)
        try:
            tie()
        except Exception:  # pragma: no cover -- model-specific tie edge cases
            pass


def materialize_fsdp2_from_cpu_state(
    model: nn.Module,
    device: str | torch.device,
    config: FSDPConfig | None = None,
    *,
    mesh: DeviceMesh | None = None,
    gradient_checkpointing: bool = False,
) -> nn.Module:
    """Shard a CPU-resident model without placing a dense full replica on GPU.

    Captures a CPU state dict, moves parameters to meta, applies FSDP2,
    allocates empty sharded storages on ``device`` (or CPU when
    ``config.cpu_offload``), then scatters the CPU state into DTensor shards.

    :param model: Dense actor on CPU (base + LoRA already attached).
    :type model: nn.Module
    :param device: Compute device for sharded parameter storage.
    :type device: str | torch.device
    :param config: FSDP2 settings.
    :type config: FSDPConfig | None
    :param mesh: Optional FSDP device mesh.
    :type mesh: DeviceMesh | None
    :param gradient_checkpointing: Wrap each transformer block with
        non-reentrant activation checkpointing before sharding.
    :type gradient_checkpointing: bool
    :return: The sharded model (same object).
    :rtype: nn.Module
    """
    config = config or FSDPConfig()
    cpu_state = {
        key: value.detach().to("cpu").contiguous()
        for key, value in model.state_dict().items()
    }
    model.to_empty(device="meta")
    apply_fsdp2(
        model,
        config,
        mesh=mesh,
        gradient_checkpointing=gradient_checkpointing,
    )
    target = torch.device("cpu") if config.cpu_offload else torch.device(device)
    model.to_empty(device=target)
    _restore_after_to_empty(model)
    set_full_model_state_dict(model, cpu_state, strict=False)
    del cpu_state
    _share_fsdp_comm_streams(model)
    return model


def _share_fsdp_comm_streams(model: nn.Module) -> None:
    """Create and share all-gather CUDA streams across FSDP units.

    Prefetch of a sibling needs those streams on the target's comm context.
    PEFT calls ``CausalLM.forward`` directly, so the first hook can be embed
    (a false root). Sharing streams here lets that prefetch succeed without
    running root ``_lazy_init`` before the first real forward.
    """
    units = [module for module in model.modules() if isinstance(module, FSDPModule)]
    if not units:
        return
    if len(units) > 1:
        share_comm_ctx(units)
    device = next(model.parameters()).device
    if device.type == "meta":
        return
    comm = units[0]._get_fsdp_state()._comm_ctx
    comm.lazy_init(device)
    for unit in units:
        param_group = unit._get_fsdp_state()._fsdp_param_group
        if param_group is not None:
            param_group.lazy_init()


def shard_dataloader_kwargs(
    dataset: Dataset[object], shuffle: bool = True
) -> dict[str, Any]:
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
        return {
            "sampler": DistributedSampler(
                dataset,
                num_replicas=get_world_size(),
                rank=get_rank(),
                shuffle=shuffle,
            )
        }
    return {"shuffle": shuffle}


class CPUOffloadOptimizer:
    """Wrap an optimizer so optimizer states (AdamW m+v) stay on CPU.

    States are moved to GPU only for ``step()``, then back to CPU.  Parameters
    and gradients remain on GPU throughout, so compute is unaffected.  With
    activation checkpointing, activations and optimizer states are never on
    GPU at the same time: peak memory becomes ``max(activations, opt_states)``
    instead of ``sum``.

    Handles FSDP2 ``DTensor`` optimizer states by swapping ``_local_tensor``
    between CPU and GPU while preserving the DTensor wrapper.
    """

    def __init__(
        self, optimizer: torch.optim.Optimizer, pin_memory: bool = True
    ) -> None:
        self.optimizer = optimizer
        self.pin_memory = pin_memory
        self._initialized = False

    def _move_states(self, device: str) -> None:
        for p in self.optimizer.state:
            state = self.optimizer.state[p]
            for k, v in state.items():
                if isinstance(v, DTensor):
                    local = v._local_tensor
                    if device == "cpu":
                        new_local = local.to("cpu")
                        if self.pin_memory and not new_local.is_pinned():
                            new_local = new_local.pin_memory()
                    else:
                        new_local = local.to(device, non_blocking=True)
                    new_dt = copy.copy(v)
                    new_dt._local_tensor = new_local
                    state[k] = new_dt
                elif isinstance(v, torch.Tensor):
                    if device == "cpu":
                        new_v = v.to("cpu")
                        if self.pin_memory and not new_v.is_pinned():
                            new_v = new_v.pin_memory()
                        state[k] = new_v
                    else:
                        state[k] = v.to(device, non_blocking=True)

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        if not self._initialized:
            result = self.optimizer.step(closure)
            self._move_states("cpu")
            self._initialized = True
            return result
        self._move_states("cuda")
        result = self.optimizer.step(closure)
        self._move_states("cpu")
        return result

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:
        if self._initialized:
            self._move_states("cuda")
        sd = self.optimizer.state_dict()
        if self._initialized:
            self._move_states("cpu")
        return sd

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.optimizer.load_state_dict(state_dict)
        self._move_states("cpu")
        self._initialized = True

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return self.optimizer.param_groups

    @param_groups.setter
    def param_groups(self, value: list[dict[str, Any]]) -> None:
        self.optimizer.param_groups = value

    @property
    def state(self) -> dict[torch.Tensor, Any]:
        return self.optimizer.state

    @property
    def base_optimizer(self) -> torch.optim.Optimizer:
        return self.optimizer

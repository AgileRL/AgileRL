# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""One shared bf16 rounding of each fp32 master weight per optimizer step.

The draw is a function of the value's kept bits and the step index, so every
rank, the rollout copy, and the colocated adapter file produce the same bf16
tensor. The trainer sets that step in :func:`activate_bf16_rounding` at the
start of each ``learn`` and advances it after every optimizer step.
"""

from __future__ import annotations

import importlib.util
import weakref
from contextvars import ContextVar
from typing import TYPE_CHECKING

import torch
import torch.distributed.fsdp._fully_shard._fsdp_collectives as fsdp_collectives
import torch.distributed.fsdp._fully_shard._fsdp_param as fsdp_param
from torch.distributed.tensor import DTensor

HAS_TRITON = importlib.util.find_spec("triton") is not None

if HAS_TRITON:
    from agilerl.utils import stochastic_round_kernel

if TYPE_CHECKING:
    from torch import nn

# None keeps the ordinary round-to-nearest cast.
bf16_round_step: ContextVar[int | None] = ContextVar("bf16_round_step", default=None)

_fsdp_cast_installed = False
_master_storages: set[int] = set()
# Filled only inside a ``learn``, where the optimizer is the one writer of the
# masters and every write advances the step.
# id(master) -> (master, step, bf16 compute weight wired to the master's gradient).
_rounded_weights: dict[int, tuple[torch.Tensor, int, torch.Tensor]] = {}
_reuse_roundings = False
# Plain masters the compute casts rounded, so later steps round them in one batch.
_compute_masters: weakref.WeakValueDictionary[int, torch.Tensor] = (
    weakref.WeakValueDictionary()
)


def stochastic_round_bf16(
    values: torch.Tensor, step: int, out: torch.Tensor | None = None
) -> torch.Tensor:
    """Round finite fp32 values to bf16. The draw depends only on the value and ``step``.

    :param values: FP32 master weights.
    :param step: Optimizer step that selects the shared draw.
    :param out: Optional bf16 tensor of the same shape to write into.
    :return: BF16 tensor with the same shape. Non-finite values use the ordinary cast.
    """
    original = values.detach().float().contiguous()
    salt = (int(step) * 0x9E3779B1) & 0xFFFFFFFF
    # int32 arithmetic wraps mod 2**32; the salt must be the signed equivalent.
    salt -= (salt >> 31) << 32
    if HAS_TRITON and original.is_cuda:
        if out is not None and out.is_contiguous():
            rounded = out
        else:
            rounded = torch.empty_like(original, dtype=torch.bfloat16)
        stochastic_round_kernel.stochastic_round_bf16_into(original, rounded, salt)
    else:
        bits = original.view(torch.int32)
        # Kept bits and the step pick the draw, so one value rounds the same on every rank.
        mixed = (bits >> 16) & 0xFFFF
        mixed ^= salt
        mixed *= 0x7FEB352D
        mixed ^= mixed >> 16
        mixed &= 0xFFFF
        # A carry can only wrap from a NaN, and non-finite values are replaced below.
        mixed += bits
        mixed &= -0x10000
        finite = torch.isfinite(original)
        rounded = torch.where(
            finite,
            mixed.view(torch.float32).to(torch.bfloat16),
            original.to(torch.bfloat16),
        )
    if out is None:
        return rounded
    if rounded is not out:
        out.copy_(rounded)
    return out


def _rounded_cast(tensor: torch.Tensor, step: int) -> torch.Tensor:
    """Cast ``tensor`` to bf16 and overwrite the values with its stochastic rounding.

    The cast keeps autograd's own fp32 gradient path; only the forward values change.
    """
    cast = tensor.to(torch.bfloat16)
    master_local = tensor.to_local() if isinstance(tensor, DTensor) else tensor
    cast_local = cast.to_local() if isinstance(cast, DTensor) else cast
    with torch.no_grad():
        stochastic_round_bf16(master_local, step, out=cast_local)
    return cast


def _round_compute_masters(step: int) -> None:
    """Round every known compute master for ``step`` with one kernel per device."""
    by_device: dict[torch.device, list[torch.Tensor]] = {}
    for master in list(_compute_masters.values()):
        by_device.setdefault(master.device, []).append(master)
    for masters in by_device.values():
        with torch.enable_grad():
            casts = [master.to(torch.bfloat16) for master in masters]
        with torch.no_grad():
            flat = torch.cat([master.detach().reshape(-1) for master in masters])
            parts = stochastic_round_bf16(flat, step).split(
                [cast.numel() for cast in casts]
            )
            torch._foreach_copy_(
                casts,
                [part.view_as(cast) for part, cast in zip(parts, casts, strict=True)],
            )
        for master, cast in zip(masters, casts, strict=True):
            _rounded_weights[id(master)] = (master, step, cast)


def cast_compute_weight(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Cast a weight to the activation dtype for a compute kernel.

    A trainable fp32 master rounds to bf16 with the active step. The backward
    is the ordinary cast. Every other tensor uses round-to-nearest.

    :param tensor: Weight the kernel multiplies.
    :param dtype: Activation dtype.
    :return: ``tensor`` when the dtypes already match, otherwise the cast.
    """
    step = bf16_round_step.get()
    if (
        step is None
        or dtype is not torch.bfloat16
        or tensor.dtype is not torch.float32
        or not tensor.requires_grad
    ):
        return tensor.to(dtype)
    local = tensor.to_local() if isinstance(tensor, DTensor) else tensor
    # Reuse only registered local masters: an FSDP unsharded copy is full-size
    # and must be freed after its layer.
    if (
        not _reuse_roundings
        or torch.is_inference_mode_enabled()
        or local.untyped_storage().data_ptr() not in _master_storages
    ):
        return _rounded_cast(tensor, step)
    cached = _rounded_weights.get(id(tensor))
    if cached is None or cached[1] != step:
        # One graph node per step; every micro-batch backward accumulates through it.
        if _compute_masters.get(id(tensor)) is tensor:
            _round_compute_masters(step)
            cached = _rounded_weights[id(tensor)]
        else:
            if not isinstance(tensor, DTensor):
                _compute_masters[id(tensor)] = tensor
            with torch.enable_grad():
                cached = (tensor, step, _rounded_cast(tensor, step))
            _rounded_weights[id(tensor)] = cached
    return cached[2] if torch.is_grad_enabled() else cached[2].detach()


def cast_exported_adapter(master: torch.Tensor) -> torch.Tensor:
    """Return the active bf16 draw of a trainable fp32 adapter tensor.

    :param master: Adapter tensor about to be written.
    :return: The shared bf16 rounding when a step is active, otherwise ``master``.
    """
    step = bf16_round_step.get()
    # vLLM casts this file to bf16 with round-to-nearest. Frozen weights stay
    # on that cast, matching the trainer.
    if step is None or master.dtype is not torch.float32 or not master.requires_grad:
        return master
    return stochastic_round_bf16(master, step)


def copy_synced_master_(destination: torch.Tensor, master: torch.Tensor) -> None:
    """Copy ``master`` into ``destination``.

    An active step stochastic-rounds an fp32 master into a bf16 destination.
    Otherwise this is ``destination.copy_``.

    :param destination: Buffer the rollout engine reads.
    :param master: Source tensor, fp32 for a master merge.
    """
    step = bf16_round_step.get()
    if (
        step is not None
        and destination.dtype is torch.bfloat16
        and master.dtype is torch.float32
    ):
        stochastic_round_bf16(master, step, out=destination)
        return
    destination.copy_(master)


def register_fp32_masters(module: nn.Module) -> None:
    """Record trainable fp32 storages whose FSDP all-gather cast should be rounded.

    The sharded all-gather input does not require grad. It shares storage with
    the local master, which does.

    :param module: Actor whose masters are sharded.
    """
    _master_storages.clear()
    _rounded_weights.clear()
    for param in module.parameters():
        local = param.to_local() if isinstance(param, DTensor) else param
        if param.requires_grad and local.dtype is torch.float32:
            _master_storages.add(local.untyped_storage().data_ptr())


def activate_bf16_rounding(
    module: nn.Module, step: int, reuse_roundings: bool = False
) -> None:
    """Round ``module``'s trainable fp32 masters to bf16 with ``step``.

    :param module: Actor the trainer forward and the rollout copy read.
    :param step: Optimizer step whose draw those reads share.
    :param reuse_roundings: Round each local master's compute cast once per step
        until :func:`stop_reusing_roundings`. Only the optimizer may write the
        masters meanwhile.
    """
    global _reuse_roundings
    _install_fsdp_cast()
    register_fp32_masters(module)
    bf16_round_step.set(step)
    _reuse_roundings = reuse_roundings


def stop_reusing_roundings() -> None:
    """Round masters at every read again and free the reused roundings."""
    global _reuse_roundings
    _reuse_roundings = False
    _rounded_weights.clear()


def _patch(owner: object, name: str, value: object) -> None:
    """Replace one attribute."""
    # ty rejects assigning these closures to FSDP's typed attributes, and ruff
    # rejects setattr with a literal name.
    setattr(owner, name, value)


def _install_fsdp_cast() -> None:
    """Round FSDP's fp32-to-bf16 all-gather cast when a step is active."""
    global _fsdp_cast_installed
    if _fsdp_cast_installed:
        return
    original_all_gather_inputs = fsdp_param.FSDPParam.all_gather_inputs.fget
    original_get_inputs = fsdp_collectives._get_param_all_gather_inputs

    def _master_shard(param: fsdp_param.FSDPParam) -> torch.Tensor | None:
        """Return the fp32 master shard whose bf16 all-gather input is rounded."""
        if (
            param.param_dtype is not torch.bfloat16
            or param.sharded_state != fsdp_param.ShardedState.SHARDED
            or hasattr(param._sharded_local_tensor, "fsdp_pre_all_gather")
        ):
            return None
        shard = param._sharded_param_data
        if (
            shard.dtype is not torch.float32
            or shard.untyped_storage().data_ptr() not in _master_storages
        ):
            return None
        return shard

    def _all_gather_inputs(param: fsdp_param.FSDPParam) -> list[torch.Tensor]:
        step = bf16_round_step.get()
        if step is None:
            return original_all_gather_inputs(param)
        shard = _master_shard(param)
        if shard is None:
            return original_all_gather_inputs(param)
        # CPU offload keeps the shard on the host.
        return [stochastic_round_bf16(shard.to(param.device, non_blocking=True), step)]

    def _param_all_gather_inputs(
        fsdp_params: list[fsdp_param.FSDPParam],
    ) -> list[list[torch.Tensor]]:
        step = bf16_round_step.get()
        if step is None:
            return original_get_inputs(fsdp_params)
        shards = [
            None if param.offload_to_cpu else _master_shard(param)
            for param in fsdp_params
        ]
        masters = [shard for shard in shards if shard is not None]
        if not masters:
            return original_get_inputs(fsdp_params)
        others = [
            param
            for param, shard in zip(fsdp_params, shards, strict=True)
            if shard is None
        ]
        # FSDP's foreach copy casts the other parameters with round-to-nearest.
        other_inputs = iter(original_get_inputs(others) if others else [])
        # One flat buffer per group, freed after the copy-in, as FSDP's own cast does.
        flat = torch.empty(
            (sum(shard.numel() for shard in masters),),
            device=fsdp_params[0].device,
            dtype=torch.bfloat16,
        )
        splits = iter(flat.split([shard.numel() for shard in masters]))
        inputs = []
        for shard in shards:
            if shard is None:
                inputs.append(next(other_inputs))
            else:
                inputs.append([stochastic_round_bf16(shard, step, out=next(splits))])
        return inputs

    _patch(fsdp_param.FSDPParam, "all_gather_inputs", property(_all_gather_inputs))
    _patch(fsdp_collectives, "_get_param_all_gather_inputs", _param_all_gather_inputs)
    _fsdp_cast_installed = True

# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""One shared bf16 rounding of each fp32 master weight per optimizer step.

The draw is a function of the value's kept bits and the step index, so every
rank and the rollout copy produce the same bf16 tensor. Round-to-nearest stays
in place until :func:`activate_bf16_rounding` sets the step.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING

import torch
import torch.distributed.fsdp._fully_shard._fsdp_collectives as fsdp_collectives
import torch.distributed.fsdp._fully_shard._fsdp_common as fsdp_common
import torch.distributed.fsdp._fully_shard._fsdp_param as fsdp_param
from torch.distributed.tensor import DTensor

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import nn

# None keeps the ordinary round-to-nearest cast.
bf16_round_step: ContextVar[int | None] = ContextVar("bf16_round_step", default=None)

_fsdp_cast_installed = False
_master_storages: set[int] = set()


def stochastic_round_bf16(values: torch.Tensor, step: int) -> torch.Tensor:
    """Round finite fp32 values to bf16. The draw depends only on the value and ``step``.

    :param values: FP32 master weights.
    :param step: Optimizer step that selects the shared draw.
    :return: BF16 tensor with the same shape. Non-finite values use the ordinary cast.
    """
    original = values.detach().float()
    bits = original.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    # Kept bits and the step pick the draw, so one value rounds the same on every rank.
    mixed = (bits >> 16) ^ ((int(step) * 0x9E3779B1) & 0xFFFFFFFF)
    mixed = mixed * 0x7FEB352D
    mixed = mixed ^ (mixed >> 16)
    rounded = (bits + (mixed & 0xFFFF)) & 0xFFFF0000
    signed = torch.where(
        (rounded & 0x80000000) != 0, rounded - 0x100000000, rounded
    ).to(torch.int32)
    out = signed.view(torch.float32).to(torch.bfloat16).view(original.shape)
    finite = torch.isfinite(original)
    return torch.where(finite, out, original.to(torch.bfloat16))


def _round_like(tensor: torch.Tensor, step: int) -> torch.Tensor:
    """Stochastic-round ``tensor``, keeping a DTensor's mesh and placements."""
    if isinstance(tensor, DTensor):
        local = stochastic_round_bf16(tensor.to_local(), step)
        return DTensor.from_local(
            local, tensor.device_mesh, tensor.placements, run_check=False
        )
    return stochastic_round_bf16(tensor, step)


def cast_compute_weight(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Cast a weight to the activation dtype for a compute kernel.

    A trainable fp32 master rounds to bf16 with the active step. The backward
    is the ordinary cast. Every other tensor uses round-to-nearest.

    :param tensor: Weight the kernel multiplies.
    :param dtype: Activation dtype.
    :return: ``tensor`` when the dtypes already match, otherwise the cast.
    """
    step = bf16_round_step.get()
    if tensor.dtype is dtype or step is None or dtype is not torch.bfloat16:
        return tensor if tensor.dtype is dtype else tensor.to(dtype)
    if tensor.dtype is not torch.float32 or not tensor.requires_grad:
        return tensor.to(dtype)
    rounded = _round_like(tensor, step)
    cast = tensor.to(dtype)
    return cast + (rounded - cast).detach()


def cast_synced_master(master: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Cast a gathered fp32 master to the dtype a rollout engine stores.

    Gathered copies are a new storage, so this does not require the training leaf.

    :param master: Master weights the sync is sending.
    :param dtype: Engine dtype.
    :return: Stochastic-rounded bf16 when a step is active, otherwise ``master.to``.
    """
    step = bf16_round_step.get()
    if step is not None and dtype is torch.bfloat16 and master.dtype is torch.float32:
        return stochastic_round_bf16(master, step)
    return master.to(dtype)


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
        destination.copy_(stochastic_round_bf16(master, step))
        return
    destination.copy_(master)


def register_fp32_masters(module: nn.Module) -> None:
    """Record trainable fp32 storages whose FSDP all-gather cast should be rounded.

    The sharded all-gather input does not require grad. It shares storage with
    the local master, which does.

    :param module: Actor whose masters are sharded.
    """
    _master_storages.clear()
    for param in module.parameters():
        local = param.to_local() if isinstance(param, DTensor) else param
        if param.requires_grad and local.dtype is torch.float32:
            _master_storages.add(local.untyped_storage().data_ptr())


def activate_bf16_rounding(module: nn.Module, step: int) -> None:
    """Round ``module``'s trainable fp32 masters to bf16 with ``step``.

    :param module: Actor the trainer forward and the rollout copy read.
    :param step: Optimizer step whose draw those reads share.
    """
    _install_fsdp_cast()
    register_fp32_masters(module)
    bf16_round_step.set(step)


def begin_master_step(
    module: nn.Module,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Snapshot the bf16 value of each trainable fp32 master before the optimizer step.

    :param module: Actor whose masters the step updates.
    :return: ``(live local data, bf16 before the step)`` pairs. Empty when rounding is off.
    """
    step = bf16_round_step.get()
    if step is None:
        return []
    snaps: list[tuple[torch.Tensor, torch.Tensor]] = []
    for param in module.parameters():
        local = param.to_local() if isinstance(param, DTensor) else param
        if not param.requires_grad or local.dtype is not torch.float32:
            continue
        data = local.detach()
        snaps.append((data, stochastic_round_bf16(data, step)))
    return snaps


def finish_master_step(
    snaps: Sequence[tuple[torch.Tensor, torch.Tensor]],
) -> float | None:
    """Advance the shared draw and return the fraction of bf16 values that changed.

    :param snaps: Pairs from :func:`begin_master_step` on this step.
    :return: Fraction of master elements whose bf16 value changed, or ``None`` when rounding is off.
    """
    step = bf16_round_step.get()
    if step is None:
        return None
    new_step = step + 1
    bf16_round_step.set(new_step)
    if not snaps:
        return None
    changed = 0
    total = 0
    for data, before in snaps:
        after = stochastic_round_bf16(data, new_step)
        changed += int(torch.count_nonzero(after != before).item())
        total += before.numel()
    if total == 0:
        return None
    return changed / total


def _patch(owner: object, name: str, value: object) -> None:
    """Replace one attribute."""
    # A nested function is not assignable to the imported FSDP function.
    setattr(owner, name, value)


def _install_fsdp_cast() -> None:
    """Round FSDP's fp32-to-bf16 all-gather cast when a step is active."""
    global _fsdp_cast_installed
    if _fsdp_cast_installed:
        return
    original = fsdp_common._to_dtype_if_needed
    original_all_gather_inputs = fsdp_param.FSDPParam.all_gather_inputs.fget
    original_get_inputs = fsdp_collectives._get_param_all_gather_inputs

    def _cast(tensor: torch.Tensor, dtype: torch.dtype | None) -> torch.Tensor:
        step = bf16_round_step.get()
        if (
            step is not None
            and dtype is torch.bfloat16
            and isinstance(tensor, torch.Tensor)
            and tensor.dtype is torch.float32
            and tensor.untyped_storage().data_ptr() in _master_storages
        ):
            return stochastic_round_bf16(tensor, step)
        return original(tensor, dtype)

    def _all_gather_inputs(param: fsdp_param.FSDPParam) -> list[torch.Tensor]:
        step = bf16_round_step.get()
        # CPU offload copies the host shard onto a new storage before the cast.
        if (
            step is not None
            and param.offload_to_cpu
            and param.param_dtype is torch.bfloat16
            and param.sharded_state == fsdp_param.ShardedState.SHARDED
            and not hasattr(param._sharded_local_tensor, "fsdp_pre_all_gather")
        ):
            shard = param._sharded_param_data
            if (
                shard.dtype is torch.float32
                and shard.untyped_storage().data_ptr() in _master_storages
            ):
                copied = shard.to(param.device, non_blocking=True)
                return [stochastic_round_bf16(copied, step)]
        return original_all_gather_inputs(param)

    def _param_all_gather_inputs(
        fsdp_params: list[fsdp_param.FSDPParam],
    ) -> list[list[torch.Tensor]]:
        # The multi-rank foreach copy casts with round-to-nearest.
        if bf16_round_step.get() is None:
            return original_get_inputs(fsdp_params)
        return [param.all_gather_inputs for param in fsdp_params]

    # Callers imported the name, so patch each module that holds it.
    _patch(fsdp_common, "_to_dtype_if_needed", _cast)
    _patch(fsdp_param, "_to_dtype_if_needed", _cast)
    _patch(fsdp_collectives, "_to_dtype_if_needed", _cast)
    _patch(fsdp_param.FSDPParam, "all_gather_inputs", property(_all_gather_inputs))
    _patch(fsdp_collectives, "_get_param_all_gather_inputs", _param_all_gather_inputs)
    _fsdp_cast_installed = True

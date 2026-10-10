# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""``Shard(0)`` placement of packed-expert weights and packed-expert module discovery."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, distribute_module, distribute_tensor
from torch.distributed.tensor.placement_types import Shard

from agilerl.distributed.ep_mesh import validate_ep_degree
from agilerl.lora.moe.layouts import (
    is_routed_experts_module,
    is_sorted_experts_module,
    routed_projection_names,
)


def packed_expert_counts(model: nn.Module) -> list[int]:
    """Expert counts of each packed-expert module in ``model``.

    :param model: Model to scan for packed expert stacks.
    :return: Expert counts in module order, empty without packed experts.
    """
    counts: list[int] = []
    for module in model.modules():
        projections = routed_projection_names(module)
        if projections is not None:
            up_name, _ = projections
            up = getattr(module, up_name)
            counts.append(int(up.shape[0]))
        elif is_sorted_experts_module(module):
            counts.append(int(cast("torch.Tensor", module.weight).shape[0]))
    return counts


def validate_actor_ep(model: nn.Module, ep: int, world_size: int = 1) -> list[int]:
    """Fail fast when ``ep`` cannot split ``model``'s experts.

    ``ep == 1`` is today's path and always passes. Larger degrees need a
    packed-expert model, a world size divisible by ``ep``, and every expert
    stack divisible by ``ep``.

    :param model: Actor module to check.
    :param ep: Expert-parallel degree.
    :param world_size: Training world size.
    :return: Expert counts of each packed-expert module.
    """
    validate_ep_degree(ep)
    counts = packed_expert_counts(model)
    if ep == 1:
        return counts
    if world_size % ep != 0:
        msg = f"world size {world_size} is not divisible by ep {ep}"
        raise ValueError(msg)
    if not counts:
        msg = f"ep={ep} needs a packed-expert MoE model"
        raise ValueError(msg)
    for count in counts:
        if count % ep != 0:
            msg = f"expert count {count} is not divisible by ep {ep}"
            raise ValueError(msg)
    return counts


def stash_ep(module: nn.Module, device_mesh: DeviceMesh) -> None:
    object.__setattr__(module, "_ep_group", device_mesh.get_group())
    object.__setattr__(module, "_ep_mesh", device_mesh)
    object.__setattr__(module, "_ep_degree", int(device_mesh.size()))


def _partition_experts_fn(_name: str, mod: nn.Module, device_mesh: DeviceMesh) -> None:
    """``Shard(0)`` every direct parameter on the EP mesh; stash the EP group."""
    for param_name, param in list(mod.named_parameters(recurse=False)):
        sharded = nn.Parameter(
            distribute_tensor(param.detach(), device_mesh, [Shard(0)]),
            requires_grad=param.requires_grad,
        )
        mod.register_parameter(param_name, sharded)
    stash_ep(mod, device_mesh)


def shard_experts_on_ep(module: nn.Module, ep_mesh: DeviceMesh | None) -> nn.Module:
    """Place packed expert weights as ``DTensor`` ``Shard(0)`` over ``ep_mesh``."""
    if ep_mesh is None:
        return module
    if getattr(module, "_ep_mesh", None) is not None:
        _partition_experts_fn("", module, ep_mesh)
        return module
    return distribute_module(module, ep_mesh, partition_fn=_partition_experts_fn)


def expert_local_tensor(weight: torch.Tensor) -> torch.Tensor:
    """Dense local expert shard (``to_local`` for ``DTensor``, else identity)."""
    if isinstance(weight, DTensor):
        return weight.to_local()
    return weight


def expert_param_bytes_local(module: nn.Module) -> int:
    """Sum of local (sharded) expert parameter bytes on this rank."""
    total = 0
    for param in module.parameters(recurse=True):
        local = expert_local_tensor(param)
        total += local.numel() * local.element_size()
    return total


def module_ep_degree(module: nn.Module) -> int:
    """EP degree stashed by :func:`shard_experts_on_ep`, else ``1``."""
    return int(getattr(module, "_ep_degree", 1) or 1)


def iter_packed_expert_modules(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Named modules that match the sorted or routed packed-expert layouts."""
    found: list[tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if is_sorted_experts_module(module) or is_routed_experts_module(module):
            found.append((name, module))
    return found


def num_packed_experts(module: nn.Module) -> int:
    """Stacked expert axis of a packed-expert module (global ``DTensor`` shape)."""
    weight = getattr(module, "weight", None)
    if isinstance(weight, torch.Tensor) and weight.ndim == 3:
        return int(weight.shape[0])
    for name in ("gate_up_proj", "up_proj", "down_proj"):
        tensor = getattr(module, name, None)
        if isinstance(tensor, torch.Tensor) and tensor.ndim == 3:
            return int(tensor.shape[0])
    msg = "Packed-expert module has no stacked 3D expert weight."
    raise ValueError(msg)


def packed_expert_count(model: nn.Module) -> int | None:
    """Stacked expert count on the first packed-expert module, if any."""
    modules = iter_packed_expert_modules(model)
    if not modules:
        return None
    return num_packed_experts(modules[0][1])


def assert_packed_experts_ep_sharded(
    model: nn.Module,
    ep: int,
    modules: list[nn.Module] | None = None,
) -> None:
    """Raise if packed-expert weights are not local ``E/ep`` shards."""
    if ep <= 1:
        return
    named: list[tuple[str, nn.Module]]
    if modules is None:
        named = iter_packed_expert_modules(model)
    else:
        named = [(type(module).__name__, module) for module in modules]
    if not named:
        msg = "ep > 1 requires packed expert modules with local E/ep shards."
        raise RuntimeError(msg)
    for name, module in named:
        for param_name, param in module.named_parameters(recurse=False):
            if not isinstance(param, torch.Tensor) or param.ndim != 3:
                continue
            local = expert_local_tensor(param)
            global_e = int(param.shape[0])
            if local.shape[0] * ep != global_e:
                msg = (
                    f"{name}.{param_name} local expert dim {local.shape[0]} "
                    f"* ep {ep} != global {global_e}"
                )
                raise RuntimeError(msg)

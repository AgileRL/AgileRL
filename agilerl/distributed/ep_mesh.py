# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""HSDP, expert-parallel and tensor-parallel ``DeviceMesh`` views."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import DTensor

from agilerl.distributed.process import all_reduce_grads


def validate_ep_degree(ep: int) -> None:
    """Reject a non-positive or non-integer expert-parallel degree."""
    if not isinstance(ep, int) or isinstance(ep, bool):
        msg = f"ep must be an int, got {type(ep).__name__}"
        raise TypeError(msg)
    if ep < 1:
        msg = f"ep must be >= 1, got {ep}"
        raise ValueError(msg)


def ep_data_parallel_size(world_size: int, ep: int) -> int:
    """Leftover data-parallel size after folding ``world_size`` by ``ep``."""
    validate_ep_degree(ep)
    return _fold_world(world_size, ep, "ep", "Expert Parallel")


def tp_data_parallel_size(world_size: int, tp: int) -> int:
    """Distinct batch shards when each ``tp`` consecutive ranks share one."""
    if tp < 1:
        msg = f"tp must be >= 1, got {tp}"
        raise ValueError(msg)
    return _fold_world(world_size, tp, "tp", "Tensor Parallel")


def _fold_world(world_size: int, degree: int, name: str, label: str) -> int:
    if world_size < 1:
        msg = f"world_size must be >= 1, got {world_size}"
        raise ValueError(msg)
    if world_size % degree != 0:
        msg = (
            f"world_size ({world_size}) must be divisible by {name} ({degree}) "
            f"for {label}."
        )
        raise ValueError(msg)
    return world_size // degree


@dataclass
class ParallelMesh:
    """DeviceMesh views for HSDP with expert and tensor parallel.

    ``world`` is ``(replicate, shard)``. Weights shard inside one ``shard``
    group and replicate across groups. ``ep`` and ``tp`` split a shard group
    into consecutive ranks; ``*_replicas`` join the ranks holding the same
    EP or TP shard. EP views are ``None`` at ``ep == 1`` and TP views at
    ``tp == 1``. ``dp_mod_ep`` and ``fsdp_experts`` are ``None`` unless
    experts also FSDP-shard (``ep`` smaller than the shard group).
    """

    world: DeviceMesh
    hsdp: DeviceMesh
    ep: DeviceMesh | None = None
    dp_mod_ep: DeviceMesh | None = None
    ep_replicas: DeviceMesh | None = None
    fsdp_experts: DeviceMesh | None = None
    tp: DeviceMesh | None = None
    tp_replicas: DeviceMesh | None = None

    @property
    def leftover_dp(self) -> int:
        """Ranks per shard group that hold the same experts."""
        ep = 1 if self.ep is None else int(self.ep.size())
        return int(self.world.size(1)) // ep

    def sync_grads(
        self, params: Sequence[nn.Parameter], reduce_dtype: torch.dtype
    ) -> None:
        """Average EP and TP shard grads over the ranks that hold each shard.

        An EP owner's grad already sums every EP-group rank's tokens, so the
        sum over replicas divides by the full world. TP ranks share tokens,
        so TP shard grads take a plain mean over replicas. FSDP reduces its
        own shards.

        :param params: Optimizer parameters; dense params are skipped.
        :param reduce_dtype: Dtype of the reduce buffer.
        :raises RuntimeError: A trainable DTensor is on no FSDP, EP or TP mesh.
        """
        fsdp_meshes = [
            mesh for mesh in (self.hsdp, self.fsdp_experts) if mesh is not None
        ]
        reductions: list[tuple[DeviceMesh, DeviceMesh, int, list[nn.Parameter]]] = []
        if self.ep is not None and self.ep_replicas is not None:
            reductions.append((self.ep, self.ep_replicas, self.world.size(), []))
        if self.tp is not None and self.tp_replicas is not None:
            reductions.append((self.tp, self.tp_replicas, self.tp_replicas.size(), []))
        for param in params:
            if not isinstance(param, DTensor) or not param.requires_grad:
                continue
            if param.device_mesh in fsdp_meshes:
                continue
            sharded = next(
                (
                    group
                    for mesh, _, _, group in reductions
                    if param.device_mesh == mesh
                ),
                None,
            )
            if sharded is None:
                msg = (
                    f"Trainable DTensor of shape {tuple(param.shape)} is on mesh "
                    f"{param.device_mesh.mesh_dim_names}, which is not an FSDP, "
                    "EP or TP mesh; its grad would never sync across replicas."
                )
                raise RuntimeError(msg)
            sharded.append(param)
        for _mesh, replicas, divisor, sharded in reductions:
            if sharded:
                all_reduce_grads(
                    sharded,
                    divisor=divisor,
                    group=replicas.get_group(),
                    reduce_dtype=reduce_dtype,
                )


def build_parallel_mesh(
    world_size: int | None = None,
    ep: int = 1,
    tp: int = 1,
    shard_group_size: int | None = None,
    device_type: str | None = None,
) -> ParallelMesh | None:
    """Build HSDP / EP / TP mesh views, or ``None`` for plain FSDP over the world.

    :param world_size: Trainer ranks; defaults to the process-group world.
    :param ep: Expert-parallel degree.
    :param tp: Tensor-parallel degree.
    :param shard_group_size: Ranks per weight-shard group; ``None`` is the world.
    :param device_type: Mesh device type; defaults to CUDA when available.
    :return: Mesh views, or ``None`` for ``ep == tp == 1`` with one shard
        group spanning the world.
    """
    if ep == 1 and tp == 1 and shard_group_size is None:
        return None
    if not dist.is_available() or not dist.is_initialized():
        msg = (
            "Expert / tensor parallel and HSDP require an initialised process "
            "group. Launch with torchrun (or set rendezvous env vars) before "
            "building the mesh."
        )
        raise RuntimeError(msg)
    if world_size is None:
        world_size = dist.get_world_size()
    shard = world_size if shard_group_size is None else shard_group_size
    if world_size % shard != 0:
        msg = (
            f"world_size ({world_size}) must be divisible by shard_group_size ({shard})"
        )
        raise ValueError(msg)
    leftover_dp = ep_data_parallel_size(shard, ep)
    tp_groups = tp_data_parallel_size(shard, tp)
    replicate = world_size // shard
    if replicate == 1 and ep == 1 and tp == 1:
        return None
    if device_type is None:
        device_type = "cuda" if torch.cuda.is_available() else "cpu"

    world = init_device_mesh(
        device_type, (replicate, shard), mesh_dim_names=("replicate", "shard")
    )
    mesh = ParallelMesh(world=world, hsdp=world if replicate > 1 else world["shard"])
    # Each unflatten / flatten creates process groups on every rank, so every
    # rank must take the same branches in the same order.
    if ep > 1:
        mesh.ep, mesh.dp_mod_ep, mesh.ep_replicas = _split_shard_axis(
            world, leftover_dp, ep, ("dp", "ep")
        )
        if mesh.dp_mod_ep is not None:
            # The mesh fully_shard gives EP DTensors it shards on dp_mod_ep.
            mesh.fsdp_experts = DeviceMesh._concatenate([mesh.dp_mod_ep, mesh.ep])
    if tp > 1:
        mesh.tp, _, mesh.tp_replicas = _split_shard_axis(
            world, tp_groups, tp, ("tp_dp", "tp")
        )
    return mesh


def _split_shard_axis(
    world: DeviceMesh, outer: int, inner: int, names: tuple[str, str]
) -> tuple[DeviceMesh, DeviceMesh | None, DeviceMesh]:
    """Split ``world``'s shard axis into ``(outer, inner)`` consecutive ranks.

    No size-1 outer dim is created, and replicas reuse the outer or
    replicate group when one already holds exactly those ranks.

    :param world: ``(replicate, shard)`` root mesh.
    :param outer: Size of the outer split.
    :param inner: Size of the inner split (consecutive ranks).
    :param names: Outer and inner dim names.
    :return: Inner mesh; outer mesh with the replicate dim, ``None`` when
        ``outer == 1``; mesh joining the ranks that hold the same inner shard.
    """
    outer_name, inner_name = names
    if outer == 1:
        split = world._unflatten(1, (inner,), (inner_name,))
        return split[inner_name], None, world["replicate"]
    split = world._unflatten(1, (outer, inner), names)
    if world.size(0) == 1:
        return split[inner_name], split[outer_name], split[outer_name]
    outer_mesh = split["replicate", outer_name]
    return split[inner_name], outer_mesh, outer_mesh._flatten(f"{inner_name}_replicas")

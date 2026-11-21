# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Named DeviceMesh views for HSDP, expert, tensor, and context parallel."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import DTensor

from agilerl.distributed.context_parallel.sequence import validate_cp_degree
from agilerl.distributed.process import all_reduce_grads


class MeshAxisName(str, Enum):
    """Names for axes of a ``DeviceMesh``."""

    # Format as the plain value on Python 3.10, which has no StrEnum.
    __str__ = str.__str__
    __format__ = str.__format__

    DP_REPLICATE = "dp_replicate"
    DP_SHARD = "dp_shard"
    DP_SHARD_CP = "dp_shard_cp"
    CP = "cp"
    TP = "tp"
    EP = "ep"
    EDP_SHARD = "edp_shard"


DENSE_AXIS_ORDER = (
    MeshAxisName.DP_REPLICATE,
    MeshAxisName.DP_SHARD,
    MeshAxisName.CP,
    MeshAxisName.TP,
)
SPARSE_AXIS_ORDER = (
    MeshAxisName.DP_REPLICATE,
    MeshAxisName.EDP_SHARD,
    MeshAxisName.EP,
)
# Size-1 axes are omitted from the DeviceMesh except FSDP dims, which
# fully_shard still needs at degree 1.
REQUIRED_DENSE_AXES = frozenset({MeshAxisName.DP_SHARD})
REQUIRED_SPARSE_AXES = frozenset({MeshAxisName.EDP_SHARD, MeshAxisName.EP})
# Process groups are process-wide; agent clones share one mesh per world
# group and degrees.
PARALLEL_MESHES: dict[tuple[object, ...], ParallelMesh] = {}


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


def _kept_axes(
    names: tuple[str, ...],
    degrees: tuple[int, ...],
    required: frozenset[str],
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    kept_names = []
    kept_degrees = []
    for name, degree in zip(names, degrees, strict=True):
        if degree > 1 or name in required:
            kept_names.append(str(name))
            kept_degrees.append(degree)
    return tuple(kept_names), tuple(kept_degrees)


def _unflatten_world(
    world: DeviceMesh,
    names: tuple[str, ...],
    degrees: tuple[int, ...],
) -> DeviceMesh:
    return world._unflatten(0, degrees, names)


def _flatten_except(mesh: DeviceMesh, keep: str, flat_name: str) -> DeviceMesh:
    names = tuple(n for n in (mesh.mesh_dim_names or ()) if n != keep)
    if len(names) == 1:
        return mesh[names[0]]
    return mesh[names]._flatten(flat_name)


def _axis_mesh(mesh: DeviceMesh, name: str, degree: int) -> DeviceMesh | None:
    if degree <= 1:
        return None
    return mesh[name]


def _fsdp_mesh(dense: DeviceMesh, degrees: dict[MeshAxisName, int]) -> DeviceMesh:
    hsdp = degrees[MeshAxisName.DP_REPLICATE] > 1
    if degrees[MeshAxisName.CP] == 1:
        if hsdp:
            return dense[MeshAxisName.DP_REPLICATE, MeshAxisName.DP_SHARD]
        return dense[MeshAxisName.DP_SHARD]
    shard = dense[MeshAxisName.DP_SHARD, MeshAxisName.CP]._flatten(
        MeshAxisName.DP_SHARD_CP
    )
    if hsdp:
        # One root with ``dense``, so TP DTensors still concatenate with it in fully_shard.
        return DeviceMesh._concatenate([dense[MeshAxisName.DP_REPLICATE], shard])
    return shard


def _on_dense_fsdp(param: DTensor, dp_shard: DeviceMesh) -> bool:
    mesh = param.device_mesh
    if mesh is dp_shard:
        return True
    names = mesh.mesh_dim_names or ()
    on_shard_axis = MeshAxisName.DP_SHARD in names or MeshAxisName.DP_SHARD_CP in names
    return on_shard_axis and MeshAxisName.EP not in names


def _on_expert_fsdp(param: DTensor) -> bool:
    return MeshAxisName.EDP_SHARD in (param.device_mesh.mesh_dim_names or ())


@dataclass
class ParallelMesh:
    """Named DeviceMesh views for HSDP with expert, tensor, and context parallel.

    Dense FSDP uses ``dp_shard``: the ``dp_shard`` axis, flattened with
    ``cp`` into ``dp_shard_cp`` when ``cp > 1``, and 2-D with
    ``dp_replicate`` when HSDP. ``ep``, ``tp``, and ``cp`` are ``None``
    when that degree is 1. ``edp_shard`` is leftover FSDP in the EP region,
    2-D with ``dp_replicate`` when HSDP, and ``None`` when its shard axis
    would be size 1. ``*_replicas`` join ranks that hold
    the same EP or TP shard.
    """

    dp_shard: DeviceMesh
    ep: DeviceMesh | None = None
    tp: DeviceMesh | None = None
    cp: DeviceMesh | None = None
    edp_shard: DeviceMesh | None = None
    ep_replicas: DeviceMesh | None = None
    tp_replicas: DeviceMesh | None = None

    def sync_grads(
        self, params: Sequence[nn.Parameter], reduce_dtype: torch.dtype
    ) -> None:
        """Average EP and TP-only shard grads over the ranks that hold each shard.

        CP ranks each hold one slice of a sequence's grad, so every reduction
        sums over CP and averages over data parallel. An EP owner's grad
        already sums every EP-group rank's tokens, so the sum over replicas
        divides by world / cp. TP-only shards (a trainable tied embedding,
        which FSDP leaves out) share tokens across TP, so they divide by
        replicas / cp. Dense FSDP reduces its own shards over ``dp_shard_cp``
        (sum over CP, mean over data parallel), TP DTensors under FSDP
        included. Their grads are already equal across TP ranks: replicated
        params sum over TP in their region and every TP rank runs the same
        tokens outside it.

        :param params: Optimizer parameters; dense params are skipped.
        :param reduce_dtype: Dtype of the reduce buffer.
        :raises RuntimeError: A trainable DTensor is on no FSDP, EP or TP mesh.
        """
        cp = 1 if self.cp is None else int(self.cp.size())
        reductions: list[tuple[DeviceMesh, DeviceMesh, int, list[nn.Parameter]]] = []
        if self.ep is not None and self.ep_replicas is not None:
            reductions.append(
                (self.ep, self.ep_replicas, dist.get_world_size() // cp, [])
            )
        if self.tp is not None and self.tp_replicas is not None:
            reductions.append(
                (self.tp, self.tp_replicas, self.tp_replicas.size() // cp, [])
            )
        for param in params:
            if not isinstance(param, DTensor) or not param.requires_grad:
                continue
            if _on_dense_fsdp(param, self.dp_shard) or _on_expert_fsdp(param):
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


def _mesh_degrees(
    world_size: int, shard_group_size: int | None, ep: int, tp: int, cp: int
) -> tuple[int, int, int]:
    """Split the world into replicate, dense-shard and expert-shard degrees.

    :param world_size: Trainer ranks.
    :param shard_group_size: Ranks per weight-shard group; ``None`` is the world.
    :param ep: Expert-parallel degree.
    :param tp: Tensor-parallel degree.
    :param cp: Context-parallel degree.
    :return: ``(dp_replicate, dp_shard, edp_shard)``.
    :raises ValueError: A degree does not divide its region.
    """
    inner = world_size if shard_group_size is None else shard_group_size
    if world_size % inner != 0:
        msg = (
            f"world_size ({world_size}) must be divisible by shard_group_size ({inner})"
        )
        raise ValueError(msg)
    dp_replicate = world_size // inner
    if inner % (cp * tp) != 0:
        msg = f"shard group ({inner}) must be divisible by cp * tp ({cp} * {tp})"
        raise ValueError(msg)
    dp_shard = inner // (cp * tp)
    inner_region = dp_shard * cp * tp
    if inner_region % ep != 0:
        msg = (
            f"ep ({ep}) must divide dp_shard * cp * tp "
            f"({dp_shard} * {cp} * {tp} = {inner_region})"
        )
        raise ValueError(msg)
    return dp_replicate, dp_shard, inner_region // ep


def build_parallel_mesh(
    world_size: int | None = None,
    ep: int = 1,
    tp: int = 1,
    cp: int = 1,
    shard_group_size: int | None = None,
    device_type: str | None = None,
) -> ParallelMesh | None:
    """Build named HSDP / EP / TP / CP mesh views, or ``None`` for plain FSDP.

    :param world_size: Trainer ranks; defaults to the process-group world.
    :param ep: Expert-parallel degree.
    :param tp: Tensor-parallel degree.
    :param cp: Context-parallel degree. ``1`` keeps full sequences on every
        rank.
    :param shard_group_size: Ranks per weight-shard group; ``None`` is the world.
    :param device_type: Mesh device type; defaults to CUDA when available.
    :return: Mesh views, shared by every call with the same world group and
        degrees, or ``None`` for ``ep == tp == cp == 1`` with one shard group
        spanning the world.
    """
    validate_cp_degree(cp)
    if ep == 1 and tp == 1 and cp == 1 and shard_group_size is None:
        return None
    if not dist.is_available() or not dist.is_initialized():
        msg = (
            "Expert / tensor / context parallel and HSDP require an initialised "
            "process group. Launch with torchrun (or set rendezvous env vars) "
            "before building the mesh."
        )
        raise RuntimeError(msg)
    if world_size is None:
        world_size = dist.get_world_size()
    dp_replicate, _, _ = _mesh_degrees(world_size, shard_group_size, ep, tp, cp)
    if dp_replicate == 1 and ep == 1 and tp == 1 and cp == 1:
        return None
    if device_type is None:
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
    key = (dist.group.WORLD, world_size, shard_group_size, ep, tp, cp, device_type)
    if key not in PARALLEL_MESHES:
        PARALLEL_MESHES[key] = _new_parallel_mesh(
            world_size, shard_group_size, ep, tp, cp, device_type
        )
    return PARALLEL_MESHES[key]


def _new_parallel_mesh(
    world_size: int,
    shard_group_size: int | None,
    ep: int,
    tp: int,
    cp: int,
    device_type: str,
) -> ParallelMesh:
    dp_replicate, dp_shard, edp_shard = _mesh_degrees(
        world_size, shard_group_size, ep, tp, cp
    )
    world = init_device_mesh(device_type, (world_size,), mesh_dim_names=("world",))
    # Each unflatten creates process groups on every rank, so every rank
    # must take the same branches in the same order.
    dense_degrees = {
        MeshAxisName.DP_REPLICATE: dp_replicate,
        MeshAxisName.DP_SHARD: dp_shard,
        MeshAxisName.CP: cp,
        MeshAxisName.TP: tp,
    }
    dense_names, dense_sizes = _kept_axes(
        DENSE_AXIS_ORDER,
        tuple(dense_degrees[name] for name in DENSE_AXIS_ORDER),
        REQUIRED_DENSE_AXES,
    )
    dense = _unflatten_world(world, dense_names, dense_sizes)
    mesh = ParallelMesh(
        dp_shard=_fsdp_mesh(dense, dense_degrees),
        tp=_axis_mesh(dense, MeshAxisName.TP, tp),
        cp=_axis_mesh(dense, MeshAxisName.CP, cp),
        tp_replicas=(
            _flatten_except(dense, MeshAxisName.TP, "tp_replicas") if tp > 1 else None
        ),
    )
    if ep > 1:
        sparse_degrees = {
            MeshAxisName.DP_REPLICATE: dp_replicate,
            MeshAxisName.EDP_SHARD: edp_shard,
            MeshAxisName.EP: ep,
        }
        sparse_names, sparse_sizes = _kept_axes(
            SPARSE_AXIS_ORDER,
            tuple(sparse_degrees[name] for name in SPARSE_AXIS_ORDER),
            REQUIRED_SPARSE_AXES,
        )
        sparse = _unflatten_world(world, sparse_names, sparse_sizes)
        mesh.ep = sparse[MeshAxisName.EP]
        if edp_shard > 1 and dp_replicate > 1:
            mesh.edp_shard = sparse[MeshAxisName.DP_REPLICATE, MeshAxisName.EDP_SHARD]
        elif edp_shard > 1:
            mesh.edp_shard = sparse[MeshAxisName.EDP_SHARD]
        mesh.ep_replicas = _flatten_except(sparse, MeshAxisName.EP, "ep_replicas")
    return mesh

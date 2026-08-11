# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Hybrid FSDP / Context Parallel / Expert Parallel mesh planning.

Prime-RL-shaped topology when ``cp > 1`` and ``ep > 1``:

```text
ep % cp == 0
dp_in_ep = ep // cp
dp_mod_ep = world_size // ep
world dims = (dp_mod_ep, dp_in_ep, cp)
ep mesh = flatten(dp_in_ep × cp)
hsdp / dp_shard_cp = flatten(full world)   # non-expert FSDP
```

Degenerate 2-GPU case ``cp=2, ep=2``: ``dp_mod_ep=1``, ``dp_in_ep=1``, ``cp=2``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist

DeviceMesh: Any
init_device_mesh: Any
try:
    from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
except ImportError:  # pragma: no cover
    DeviceMesh = None
    init_device_mesh = None


@dataclass(frozen=True)
class HybridMeshLayout:
    """Derived sizes for flat / CP-only / EP-only / CP×EP worlds."""

    world_size: int
    cp: int
    ep: int
    dp_shard: int
    dp_mod_ep: int
    dp_in_ep: int

    @property
    def cp_enabled(self) -> bool:
        return self.cp > 1

    @property
    def ep_enabled(self) -> bool:
        return self.ep > 1


def compute_hybrid_layout(
    world_size: int, *, cp: int = 1, ep: int = 1
) -> HybridMeshLayout:
    """Validate and derive mesh dim sizes for ``(world_size, cp, ep)``."""
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    if cp < 1:
        raise ValueError(f"cp must be >= 1, got {cp}")
    if ep < 1:
        raise ValueError(f"ep must be >= 1, got {ep}")
    if world_size % cp != 0:
        raise ValueError(
            f"world_size ({world_size}) must be divisible by cp ({cp})"
        )
    if world_size % ep != 0:
        raise ValueError(
            f"world_size ({world_size}) must be divisible by ep ({ep})"
        )
    if cp > 1 and ep > 1:
        if ep % cp != 0:
            raise ValueError(
                f"ep ({ep}) must be divisible by cp ({cp}) for CP×EP compose "
                "(Prime-RL: ep % cp == 0)."
            )
        dp_in_ep = ep // cp
        dp_mod_ep = world_size // ep
        dp_shard = dp_mod_ep * dp_in_ep
        if (dp_shard * cp) % ep != 0:
            raise ValueError(
                f"(dp_shard * cp) ({dp_shard * cp}) must be divisible by ep ({ep})"
            )
        return HybridMeshLayout(
            world_size=world_size,
            cp=cp,
            ep=ep,
            dp_shard=dp_shard,
            dp_mod_ep=dp_mod_ep,
            dp_in_ep=dp_in_ep,
        )
    if ep > 1:
        dp_mod_ep = world_size // ep
        return HybridMeshLayout(
            world_size=world_size,
            cp=1,
            ep=ep,
            dp_shard=world_size,
            dp_mod_ep=dp_mod_ep,
            dp_in_ep=ep,
        )
    # CP-only or flat
    dp_shard = world_size // cp
    return HybridMeshLayout(
        world_size=world_size,
        cp=cp,
        ep=1,
        dp_shard=dp_shard,
        dp_mod_ep=dp_shard,
        dp_in_ep=1,
    )


def validate_hybrid_parallel_config(
    *,
    world_size: int,
    cp: int = 1,
    ep: int = 1,
) -> HybridMeshLayout:
    """Public validator used by algo init; returns the layout."""
    return compute_hybrid_layout(world_size, cp=cp, ep=ep)


@dataclass
class HybridParallelMesh:
    """DeviceMesh views for FSDP + optional CP + optional EP.

    Exposes the same attributes as :class:`~agilerl.utils.expert_parallel.ExpertParallelMesh`
    (``layout``-compatible fields via ``ep_layout``, ``world``, ``ep``, ``dp_mod_ep``,
    ``hsdp``) plus CP accessors used by Ulysses/ring.
    """

    hybrid_layout: HybridMeshLayout
    world: Any
    ep: Any | None
    dp_mod_ep: Any | None
    hsdp: Any
    cp: Any | None
    dp_shard_cp: Any
    _submeshes: dict[str, Any] = field(default_factory=dict, repr=False)

    # ExpertParallelMesh-compatible alias
    @property
    def layout(self) -> Any:
        """EP-shaped layout view for callers expecting ``ExpertParallelMesh.layout``."""
        from agilerl.utils.expert_parallel import EpMeshLayout

        hl = self.hybrid_layout
        return EpMeshLayout(
            world_size=hl.world_size,
            ep=hl.ep,
            dp_shard=hl.dp_shard if hl.ep > 1 else hl.world_size,
            dp_shard_mod_ep=hl.dp_mod_ep if hl.ep > 1 else hl.world_size,
            dp_shard_in_ep=hl.ep if hl.ep > 1 else 1,
        )

    @property
    def ep_group(self) -> dist.ProcessGroup:
        if self.ep is None:
            raise RuntimeError("No EP mesh when ep == 1")
        return self.ep.get_group()

    @property
    def ep_degree(self) -> int:
        return self.hybrid_layout.ep

    def get_mesh(self, name: str) -> Any:
        if name in self._submeshes:
            return self._submeshes[name]
        if name == "hsdp":
            return self.hsdp
        if name == "dp_shard_cp":
            return self.dp_shard_cp
        if name == "cp":
            if self.cp is None:
                raise RuntimeError("No CP mesh when cp == 1")
            return self.cp
        if name == "ep":
            if self.ep is None:
                raise RuntimeError("No EP mesh when ep == 1")
            return self.ep
        if name == "dp_mod_ep":
            if self.dp_mod_ep is None:
                raise RuntimeError("No dp_mod_ep mesh when ep == 1")
            return self.dp_mod_ep
        return self.world[name]

    def cp_group(self) -> dist.ProcessGroup:
        return self.get_mesh("cp").get_group()

    def cp_rank(self) -> int:
        return dist.get_rank(self.cp_group())

    @property
    def dp_size(self) -> int:
        """Batch / sampler world excluding the CP dim."""
        return self.hybrid_layout.dp_shard

    @property
    def world_mesh(self) -> Any:
        return self.world

    @property
    def cp_enabled(self) -> bool:
        return self.hybrid_layout.cp_enabled

    def build_mesh(self, device_type: str | None = None) -> Any:
        """Meshes are built in :func:`build_hybrid_parallel_mesh`; return world."""
        del device_type
        return self.world


def build_hybrid_parallel_mesh(
    world_size: int | None = None,
    *,
    cp: int = 1,
    ep: int = 1,
    device_type: str | None = None,
) -> HybridParallelMesh | None:
    """Build hybrid mesh views, or ``None`` when ``cp == 1`` and ``ep == 1``.

    Requires an initialised process group when either degree is ``> 1``.
    """
    if cp <= 1 and ep <= 1:
        return None
    if init_device_mesh is None or DeviceMesh is None:
        raise RuntimeError(
            "Hybrid parallel mesh requires torch.distributed DeviceMesh support."
        )
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "Hybrid parallel mesh requires an initialised process group. "
            "Launch with torchrun before building the mesh."
        )
    if world_size is None:
        world_size = dist.get_world_size()
    layout = compute_hybrid_layout(world_size, cp=cp, ep=ep)
    if device_type is None:
        device_type = "cuda" if torch.cuda.is_available() else "cpu"

    if layout.cp_enabled and layout.ep_enabled:
        dims = [layout.dp_mod_ep, layout.dp_in_ep, layout.cp]
        names = ["dp_shard_mod_ep", "dp_shard_in_ep", "cp"]
        world = init_device_mesh(device_type, tuple(dims), mesh_dim_names=tuple(names))
        cp_mesh = world["cp"]
        # EP = flatten(dp_in_ep × cp)
        if layout.dp_in_ep > 1:
            ep_root = world[("dp_shard_in_ep", "cp")]
            ep_mesh = ep_root._flatten(mesh_dim_name="ep")
        else:
            # Same ranks as CP; reuse for expert Shard(0) + A2A.
            ep_mesh = cp_mesh
        dp_mod_ep = world["dp_shard_mod_ep"]
        hsdp = world._flatten(mesh_dim_name="hsdp")
        dp_shard_cp = hsdp
        sub = {
            "cp": cp_mesh,
            "ep": ep_mesh,
            "dp_mod_ep": dp_mod_ep,
            "hsdp": hsdp,
            "dp_shard_cp": dp_shard_cp,
            "dp_cp": dp_shard_cp,
        }
        return HybridParallelMesh(
            hybrid_layout=layout,
            world=world,
            ep=ep_mesh,
            dp_mod_ep=dp_mod_ep,
            hsdp=hsdp,
            cp=cp_mesh,
            dp_shard_cp=dp_shard_cp,
            _submeshes=sub,
        )

    if layout.ep_enabled:
        dims = [layout.dp_mod_ep, layout.dp_in_ep]
        names = ["dp_shard_mod_ep", "dp_shard_in_ep"]
        world = init_device_mesh(device_type, tuple(dims), mesh_dim_names=tuple(names))
        ep_mesh = world["dp_shard_in_ep"]
        if ep_mesh.ndim > 1:
            ep_mesh = ep_mesh._flatten(mesh_dim_name="ep")
        dp_mod_ep = world["dp_shard_mod_ep"]
        hsdp = world._flatten(mesh_dim_name="hsdp") if world.ndim > 1 else world
        sub = {
            "ep": ep_mesh,
            "dp_mod_ep": dp_mod_ep,
            "hsdp": hsdp,
            "dp_shard_cp": hsdp,
        }
        return HybridParallelMesh(
            hybrid_layout=layout,
            world=world,
            ep=ep_mesh,
            dp_mod_ep=dp_mod_ep,
            hsdp=hsdp,
            cp=None,
            dp_shard_cp=hsdp,
            _submeshes=sub,
        )

    # CP-only
    dims = [layout.dp_shard, layout.cp]
    names = ["dp_shard", "cp"]
    world = init_device_mesh(device_type, tuple(dims), mesh_dim_names=tuple(names))
    cp_mesh = world["cp"]
    dp_shard_cp = world._flatten(mesh_dim_name="dp_shard_cp")
    sub = {
        "dp": world["dp_shard"],
        "cp": cp_mesh,
        "dp_shard_cp": dp_shard_cp,
        "dp_cp": dp_shard_cp,
        "hsdp": dp_shard_cp,
    }
    return HybridParallelMesh(
        hybrid_layout=layout,
        world=world,
        ep=None,
        dp_mod_ep=None,
        hsdp=dp_shard_cp,
        cp=cp_mesh,
        dp_shard_cp=dp_shard_cp,
        _submeshes=sub,
    )

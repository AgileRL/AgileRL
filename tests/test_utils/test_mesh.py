# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Rank-group contracts for named ``build_parallel_mesh`` views."""

from __future__ import annotations

import copy
import os
from typing import Any
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from agilerl.distributed import (
    FSDPConfig,
    build_parallel_mesh,
    ep_data_parallel_size,
    fsdp_blocks,
    gather_for_cp,
    shard_for_cp,
    sync_grads,
    tp_data_parallel_size,
)
from agilerl.distributed.fsdp_blocks import apply_fsdp2
from tests.rank_launch import run_ranks

_DIST_ENV = ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")


@pytest.fixture(autouse=True)
def _clean_dist_state():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    saved = {var: os.environ.pop(var, None) for var in _DIST_ENV}
    yield
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    for var, value in saved.items():
        if value is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = value


def _groups(mesh: DeviceMesh) -> set[tuple[int, ...]]:
    local = tuple(sorted(dist.get_process_group_ranks(mesh.get_group())))
    gathered: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)
    return set(gathered)


def _consecutive_groups(world_size: int, degree: int) -> set[tuple[int, ...]]:
    return {
        tuple(range(start, start + degree)) for start in range(0, world_size, degree)
    }


def _strided_groups(world_size: int, stride: int) -> set[tuple[int, ...]]:
    return {tuple(range(offset, world_size, stride)) for offset in range(stride)}


def _hsdp_replica_groups(
    world_size: int, shard_group_size: int
) -> set[tuple[int, ...]]:
    n_groups = world_size // shard_group_size
    return {
        tuple(rank + group * shard_group_size for group in range(n_groups))
        for rank in range(shard_group_size)
    }


def _dense_product(mesh, world_size: int) -> None:
    tp = 1 if mesh.tp is None else int(mesh.tp.size())
    assert int(mesh.dp_shard.size()) * tp == world_size


def _inner_fsdp_groups(mesh) -> set[tuple[int, ...]]:
    if mesh.dp_shard.ndim == 2:
        return _groups(mesh.dp_shard[mesh.dp_shard.mesh_dim_names[-1]])
    return _groups(mesh.dp_shard)


def _plain_fsdp_worker(rank: int, world_size: int) -> None:
    assert build_parallel_mesh(world_size=world_size, device_type="cpu") is None
    assert (
        build_parallel_mesh(
            world_size=world_size, shard_group_size=world_size, device_type="cpu"
        )
        is None
    )


def _default_device_worker(
    _rank: int, world_size: int
) -> dict[str, tuple[str, tuple[int, ...], tuple[str, ...]]]:
    """Device type, shape and axis names of the dense and expert meshes."""
    mesh = build_parallel_mesh(world_size=world_size, ep=2)
    assert mesh is not None
    assert mesh.ep is not None
    return {
        name: (view.device_type, tuple(view.mesh.shape), view.mesh_dim_names)
        for name, view in (("dp_shard", mesh.dp_shard), ("ep", mesh.ep))
    }


def _ep_groups_worker(rank: int, world_size: int) -> None:
    ep = 2
    mesh = build_parallel_mesh(world_size=world_size, ep=ep, device_type="cpu")

    assert mesh is not None
    assert mesh.ep is not None
    assert _groups(mesh.ep) == _consecutive_groups(world_size, ep)
    assert mesh.edp_shard is not None
    assert int(mesh.edp_shard.size()) == world_size // ep
    assert _groups(mesh.edp_shard) == _strided_groups(world_size, ep)
    assert mesh.ep_replicas is not None
    assert int(mesh.ep_replicas.size()) == world_size // ep
    assert mesh.tp is None
    _dense_product(mesh, world_size)


def _ep_spans_inner_worker(rank: int, world_size: int) -> None:
    mesh = build_parallel_mesh(world_size=world_size, ep=world_size, device_type="cpu")

    assert mesh is not None
    assert mesh.ep is not None
    assert _groups(mesh.ep) == {tuple(range(world_size))}
    assert mesh.edp_shard is None
    assert mesh.ep_replicas is not None
    assert int(mesh.ep_replicas.size()) == 1
    _dense_product(mesh, world_size)


def _tp_groups_worker(rank: int, world_size: int) -> None:
    tp = 2
    mesh = build_parallel_mesh(world_size=world_size, tp=tp, device_type="cpu")

    assert mesh is not None
    assert mesh.tp is not None
    assert _groups(mesh.tp) == _consecutive_groups(world_size, tp)
    assert mesh.ep is None
    assert mesh.edp_shard is None
    inner_groups = _inner_fsdp_groups(mesh)
    assert {len(group) for group in inner_groups} == {world_size // tp}
    assert inner_groups == _strided_groups(world_size, tp)
    assert mesh.dp_shard.ndim == 1
    _dense_product(mesh, world_size)


def _hsdp_replica_worker(rank: int, world_size: int) -> None:
    shard_group_size = 2
    mesh = build_parallel_mesh(
        world_size=world_size,
        shard_group_size=shard_group_size,
        device_type="cpu",
    )

    assert mesh is not None
    assert mesh.dp_shard.ndim == 2
    assert mesh.dp_shard.mesh_dim_names == ("dp_replicate", "dp_shard")
    assert _groups(mesh.dp_shard["dp_replicate"]) == _hsdp_replica_groups(
        world_size, shard_group_size
    )
    assert mesh.ep is None
    assert mesh.tp is None
    _dense_product(mesh, world_size)


def _ep_and_tp_worker(rank: int, world_size: int) -> None:
    ep, tp = 2, 2
    mesh = build_parallel_mesh(world_size=world_size, ep=ep, tp=tp, device_type="cpu")

    assert mesh is not None
    assert mesh.ep is not None
    assert mesh.tp is not None
    assert _groups(mesh.ep) == _consecutive_groups(world_size, ep)
    assert _groups(mesh.tp) == _consecutive_groups(world_size, tp)
    assert mesh.edp_shard is not None
    assert int(mesh.edp_shard.size()) == world_size // ep
    assert _inner_fsdp_groups(mesh) == _strided_groups(world_size, tp)
    _dense_product(mesh, world_size)


def _cp_pairs_worker(rank: int, world_size: int) -> None:
    mesh = build_parallel_mesh(world_size=world_size, cp=2, device_type="cpu")

    assert mesh is not None
    assert mesh.cp is not None
    assert int(mesh.cp.size()) == 2
    assert _groups(mesh.cp) == _consecutive_groups(world_size, 2)
    assert mesh.tp is None
    assert mesh.ep is None
    _dense_product(mesh, world_size)


def _cp_tp_fold_worker(rank: int, world_size: int) -> None:
    mesh = build_parallel_mesh(world_size=world_size, tp=2, cp=2, device_type="cpu")

    assert mesh is not None
    assert mesh.dp_shard.mesh_dim_names == ("dp_shard_cp",)
    assert _inner_fsdp_groups(mesh) == _strided_groups(world_size, 2)
    assert _groups(mesh.tp) == _consecutive_groups(world_size, 2)
    assert _groups(mesh.cp) == {(0, 2), (1, 3), (4, 6), (5, 7)}
    _dense_product(mesh, world_size)


def _hsdp_cp_fold_worker(rank: int, world_size: int) -> None:
    mesh = build_parallel_mesh(
        world_size=world_size, cp=2, shard_group_size=4, device_type="cpu"
    )

    assert mesh is not None
    assert mesh.dp_shard.mesh_dim_names == ("dp_replicate", "dp_shard_cp")
    assert _inner_fsdp_groups(mesh) == _consecutive_groups(world_size, 4)
    assert _groups(mesh.dp_shard["dp_replicate"]) == _hsdp_replica_groups(world_size, 4)
    _dense_product(mesh, world_size)


def _mesh_reuse_worker(rank: int, world_size: int) -> None:
    # Act
    first = build_parallel_mesh(world_size=world_size, tp=2, cp=2, device_type="cpu")
    second = build_parallel_mesh(world_size=world_size, tp=2, cp=2, device_type="cpu")
    other = build_parallel_mesh(world_size=world_size, cp=2, device_type="cpu")

    # Assert
    assert first is not None
    assert second is first
    assert other is not None
    assert second.dp_shard.get_group() is first.dp_shard.get_group()
    assert second.tp.get_group() is first.tp.get_group()
    assert second.cp.get_group() is first.cp.get_group()
    assert other.dp_shard.get_group() is not first.dp_shard.get_group()
    assert other.cp.get_group() is not first.cp.get_group()


def _cp_grad_sync_worker(rank: int, world_size: int) -> None:
    torch.manual_seed(0)
    model = nn.Linear(4, 4)
    reference = copy.deepcopy(model)
    inputs = torch.randn(2, 1, 8, 4)
    probes = torch.randn(2, 1, 8, 4)
    mesh = build_parallel_mesh(world_size=world_size, cp=2, device_type="cpu")
    assert mesh is not None
    assert mesh.cp is not None
    dp_rank = rank // 2
    cp_rank = int(mesh.cp.get_local_rank())
    with patch.object(fsdp_blocks, "PARAM_PERSISTENCE_THRESHOLD", 0):
        apply_fsdp2(
            model,
            FSDPConfig(cp=2, param_dtype="float32", reduce_dtype="float32"),
            mesh=mesh.dp_shard,
        )

    local = shard_for_cp(inputs[dp_rank], cp_rank, 2)
    output = gather_for_cp(model(local), mesh.cp.get_group())
    (output * probes[dp_rank]).sum().backward()
    params = list(model.parameters())
    sync_grads([param for param in params if not isinstance(param, DTensor)], 2)
    mesh.sync_grads(params, torch.float32)
    reference_loss = sum(
        (reference(inputs[index]) * probes[index]).sum() for index in range(2)
    )
    (reference_loss / 2).backward()

    for name, param in model.named_parameters():
        expected = reference.get_parameter(name).grad
        assert param.grad is not None
        got = param.grad.full_tensor()
        assert torch.allclose(got, expected, atol=1e-6, rtol=1e-6), name


def _tp_dense_grad_sync_worker(rank: int, world_size: int) -> None:
    # Arrange
    torch.manual_seed(0)
    model = nn.Linear(4, 4)
    inputs = torch.randn(world_size // 2, 3, 4)
    mesh = build_parallel_mesh(world_size=world_size, tp=2, device_type="cpu")
    assert mesh is not None
    with patch.object(fsdp_blocks, "PARAM_PERSISTENCE_THRESHOLD", 0):
        apply_fsdp2(
            model,
            FSDPConfig(tp=2, param_dtype="float32", reduce_dtype="float32"),
            mesh=mesh.dp_shard,
        )
    model(inputs[rank // 2]).sum().backward()
    params = list(model.parameters())
    before = [param.grad.to_local().clone() for param in params]

    # Act
    mesh.sync_grads(params, torch.float32)

    # Assert
    for param, expected in zip(params, before, strict=True):
        assert torch.equal(param.grad.to_local(), expected)


def _tp_only_grad_sync_worker(rank: int, world_size: int) -> None:
    # Arrange
    mesh = build_parallel_mesh(world_size=world_size, tp=2, device_type="cpu")
    assert mesh is not None
    assert mesh.tp is not None
    tp_rank = int(mesh.tp.get_local_rank())
    weight = nn.Parameter(distribute_tensor(torch.zeros(2, 4), mesh.tp, [Shard(1)]))
    grads = {
        dp_rank: torch.arange(8.0).reshape(2, 4) * (dp_rank + 1)
        for dp_rank in range(world_size // 2)
    }
    local = grads[rank // 2][:, tp_rank * 2 : tp_rank * 2 + 2]
    weight.grad = DTensor.from_local(local.clone(), mesh.tp, [Shard(1)])

    # Act
    mesh.sync_grads([weight], torch.float32)

    # Assert
    mean = sum(grads.values()) / len(grads)
    assert torch.allclose(weight.grad.full_tensor(), mean, atol=0, rtol=0)


def _hsdp_ep_worker(rank: int, world_size: int) -> None:
    mesh = build_parallel_mesh(
        world_size=world_size, ep=2, shard_group_size=4, device_type="cpu"
    )

    assert mesh is not None
    assert mesh.edp_shard is not None
    assert mesh.edp_shard.mesh_dim_names == ("dp_replicate", "edp_shard")
    assert int(mesh.edp_shard.size()) == world_size // 2
    assert _groups(mesh.edp_shard["dp_replicate"]) == _hsdp_replica_groups(
        world_size, 4
    )


def _hsdp_ep_spans_inner_worker(rank: int, world_size: int) -> None:
    mesh = build_parallel_mesh(
        world_size=world_size, ep=4, shard_group_size=4, device_type="cpu"
    )

    assert mesh is not None
    assert mesh.edp_shard is None
    assert mesh.ep_replicas is not None
    assert int(mesh.ep_replicas.size()) == world_size // 4


def _ep_does_not_divide_inner_worker(rank: int, world_size: int) -> None:
    with pytest.raises(ValueError, match="must divide dp_shard"):
        build_parallel_mesh(world_size=world_size, ep=3, device_type="cpu")


@pytest.mark.multi_rank
class TestBuildParallelMesh:
    def test_plain_fsdp_returns_none(self):
        run_ranks(_plain_fsdp_worker)

    def test_ep_groups_match_consecutive_last_axis(self):
        run_ranks(_ep_groups_worker, 4)

    def test_edp_shard_is_none_when_ep_spans_the_inner_group(self):
        run_ranks(_ep_spans_inner_worker)

    def test_tp_groups_are_consecutive_and_dense_fsdp_excludes_tp(self):
        run_ranks(_tp_groups_worker, 4)

    def test_hsdp_replica_groups_match_consecutive_shard_split(self):
        run_ranks(_hsdp_replica_worker, 4)

    def test_ep_groups_unchanged_when_tp_splits_dense_fsdp(self):
        run_ranks(_ep_and_tp_worker, 4)

    def test_cp_two_builds_titan_pairs(self):
        run_ranks(_cp_pairs_worker, 4)

    def test_dense_fsdp_spans_dp_shard_and_cp_per_tp_rank(self):
        run_ranks(_cp_tp_fold_worker, 8)

    def test_hsdp_dense_fsdp_folds_cp_into_the_shard_axis(self):
        run_ranks(_hsdp_cp_fold_worker, 8)

    def test_hsdp_expert_fsdp_spans_dp_replicate_and_edp_shard(self):
        run_ranks(_hsdp_ep_worker, 8)

    def test_hsdp_edp_shard_is_none_when_ep_spans_the_shard_group(self):
        run_ranks(_hsdp_ep_spans_inner_worker, 8)

    def test_ep_that_does_not_divide_inner_raises(self):
        run_ranks(_ep_does_not_divide_inner_worker, 4)

    def test_same_degrees_reuse_the_process_groups(self):
        run_ranks(_mesh_reuse_worker, 8)


@pytest.mark.multi_rank
class TestParallelMeshSyncGrads:
    def test_cp_ranks_sum_into_the_full_sequence_grad(self):
        run_ranks(_cp_grad_sync_worker, 4)

    def test_dense_fsdp_grads_are_unchanged_under_tp(self):
        run_ranks(_tp_dense_grad_sync_worker, 4)

    def test_tp_only_shard_grads_average_over_data_parallel(self):
        run_ranks(_tp_only_grad_sync_worker, 4)


class TestTpDataParallelSize:
    def test_rejects_tp_below_one(self):
        with pytest.raises(ValueError, match="tp must be >= 1"):
            tp_data_parallel_size(4, 0)


class TestFoldWorld:
    def test_rejects_non_positive_world_size(self):
        with pytest.raises(ValueError, match="world_size must be >= 1"):
            ep_data_parallel_size(0, 1)


class TestBuildParallelMeshGuards:
    def test_requires_an_initialised_process_group(self):
        with pytest.raises(RuntimeError, match="initialised process group"):
            build_parallel_mesh(world_size=2, ep=2, device_type="cpu")


class TestBuildParallelMeshShardGroup:
    def test_rejects_shard_group_that_does_not_divide_world(self):
        with (
            patch.object(dist, "is_available", return_value=True),
            patch.object(dist, "is_initialized", return_value=True),
        ):
            with pytest.raises(ValueError, match="divisible by shard_group_size"):
                build_parallel_mesh(world_size=4, shard_group_size=3, device_type="cpu")

    def test_rejects_shard_group_that_cp_and_tp_do_not_divide(self):
        with (
            patch.object(dist, "is_available", return_value=True),
            patch.object(dist, "is_initialized", return_value=True),
        ):
            with pytest.raises(ValueError, match=r"divisible by cp \* tp \(2 \* 2\)"):
                build_parallel_mesh(world_size=6, cp=2, tp=2, device_type="cpu")


class TestBuildParallelMeshCp:
    def test_rejects_cp_below_one(self):
        with pytest.raises(ValueError, match="cp must be >= 1, got 0"):
            build_parallel_mesh(world_size=2, cp=0, device_type="cpu")


class TestBuildParallelMeshDefaultDevice:
    @pytest.mark.multi_rank
    def test_infers_cpu_when_device_type_is_omitted(self):
        expected = {
            "dp_shard": ("cpu", (2,), ("dp_shard",)),
            "ep": ("cpu", (2,), ("ep",)),
        }

        meshes = run_ranks(_default_device_worker, 2, cpu=True)

        assert meshes == [expected, expected]

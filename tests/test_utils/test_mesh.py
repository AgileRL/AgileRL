# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Rank-group contracts for named ``build_parallel_mesh`` views."""

from __future__ import annotations

import os
import sys
from typing import Any
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import DeviceMesh

from agilerl.distributed import (
    build_parallel_mesh,
    ep_data_parallel_size,
    tp_data_parallel_size,
)
from tests.dist_ports import (
    get_free_port,
    gloo_filestore,
    gloo_init_method,
    gloo_rank_env,
)

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


def _gloo_available() -> bool:
    if sys.platform == "win32":
        return False
    return dist.is_available()


requires_gloo = pytest.mark.skipif(not _gloo_available(), reason="gloo unavailable")


def _init_gloo(rank: int, world_size: int, port: int) -> None:
    os.environ.update(gloo_rank_env(rank, world_size, port))
    dist.init_process_group(
        backend="gloo",
        init_method=gloo_init_method(port),
        rank=rank,
        world_size=world_size,
    )


def _spawn_ranks(worker, world_size: int = 2, timeout: float = 300.0) -> None:
    last_error: AssertionError | None = None
    for _ in range(3):
        port = get_free_port()
        ctx = mp.get_context("spawn")
        queue: mp.Queue = ctx.Queue()
        procs = [
            ctx.Process(target=worker, args=(rank, world_size, port, queue))
            for rank in range(world_size)
        ]
        with gloo_filestore():
            for proc in procs:
                proc.start()
            try:
                results = [queue.get(timeout=timeout) for _ in range(world_size)]
                for proc in procs:
                    proc.join(timeout=timeout)
                    assert proc.exitcode == 0, f"rank exited {proc.exitcode}"
                for rank, status, err in sorted(results):
                    assert status == "ok", f"rank {rank}: {err}"
                return
            except AssertionError as exc:
                if "DistNetworkError" not in str(exc):
                    raise
                last_error = exc
            finally:
                for proc in procs:
                    if proc.is_alive():
                        proc.kill()
                        proc.join(timeout=5)
    assert last_error is not None
    raise last_error


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


def _hsdp_replica_groups(world_size: int, shard_group_size: int) -> set[tuple[int, ...]]:
    n_groups = world_size // shard_group_size
    return {
        tuple(rank + group * shard_group_size for group in range(n_groups))
        for rank in range(shard_group_size)
    }


def _dense_product(mesh, world_size: int) -> None:
    tp = 1 if mesh.tp is None else int(mesh.tp.size())
    cp = 1 if mesh.cp is None else int(mesh.cp.size())
    if mesh.dp_shard.ndim == 2:
        dp_replicate = int(mesh.dp_shard.size(0))
        dp_shard = int(mesh.dp_shard.size(1))
    else:
        dp_replicate = 1
        dp_shard = int(mesh.dp_shard.size())
    assert dp_replicate * dp_shard * cp * tp * 1 == world_size


def _inner_fsdp_groups(mesh) -> set[tuple[int, ...]]:
    if mesh.dp_shard.ndim == 2:
        return _groups(mesh.dp_shard["dp_shard"])
    return _groups(mesh.dp_shard)


def _ok(result_queue: Any, rank: int) -> None:
    result_queue.put((rank, "ok", None))


def _fail(result_queue: Any, rank: int, exc: BaseException) -> None:
    result_queue.put((rank, "err", repr(exc)))


def _plain_fsdp_worker(
    rank: int, world_size: int, port: int, result_queue: Any
) -> None:
    try:
        _init_gloo(rank, world_size, port)

        assert build_parallel_mesh(world_size=world_size, device_type="cpu") is None
        assert (
            build_parallel_mesh(
                world_size=world_size, shard_group_size=world_size, device_type="cpu"
            )
            is None
        )

        _ok(result_queue, rank)
    except Exception as exc:
        _fail(result_queue, rank, exc)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _ep_groups_worker(rank: int, world_size: int, port: int, result_queue: Any) -> None:
    try:
        _init_gloo(rank, world_size, port)
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

        _ok(result_queue, rank)
    except Exception as exc:
        _fail(result_queue, rank, exc)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _ep_spans_inner_worker(
    rank: int, world_size: int, port: int, result_queue: Any
) -> None:
    try:
        _init_gloo(rank, world_size, port)
        mesh = build_parallel_mesh(
            world_size=world_size, ep=world_size, device_type="cpu"
        )

        assert mesh is not None
        assert mesh.ep is not None
        assert _groups(mesh.ep) == {tuple(range(world_size))}
        assert mesh.edp_shard is None
        assert mesh.ep_replicas is not None
        assert int(mesh.ep_replicas.size()) == 1
        _dense_product(mesh, world_size)

        _ok(result_queue, rank)
    except Exception as exc:
        _fail(result_queue, rank, exc)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _tp_groups_worker(rank: int, world_size: int, port: int, result_queue: Any) -> None:
    try:
        _init_gloo(rank, world_size, port)
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

        _ok(result_queue, rank)
    except Exception as exc:
        _fail(result_queue, rank, exc)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _hsdp_replica_worker(
    rank: int, world_size: int, port: int, result_queue: Any
) -> None:
    try:
        _init_gloo(rank, world_size, port)
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

        _ok(result_queue, rank)
    except Exception as exc:
        _fail(result_queue, rank, exc)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _ep_and_tp_worker(
    rank: int, world_size: int, port: int, result_queue: Any
) -> None:
    try:
        _init_gloo(rank, world_size, port)
        ep, tp = 2, 2
        mesh = build_parallel_mesh(
            world_size=world_size, ep=ep, tp=tp, device_type="cpu"
        )

        assert mesh is not None
        assert mesh.ep is not None
        assert mesh.tp is not None
        assert _groups(mesh.ep) == _consecutive_groups(world_size, ep)
        assert _groups(mesh.tp) == _consecutive_groups(world_size, tp)
        assert mesh.edp_shard is not None
        assert int(mesh.edp_shard.size()) == world_size // ep
        assert _inner_fsdp_groups(mesh) == _strided_groups(world_size, tp)
        _dense_product(mesh, world_size)

        _ok(result_queue, rank)
    except Exception as exc:
        _fail(result_queue, rank, exc)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _cp_pairs_worker(rank: int, world_size: int, port: int, result_queue: Any) -> None:
    try:
        _init_gloo(rank, world_size, port)
        mesh = build_parallel_mesh(world_size=world_size, cp=2, device_type="cpu")

        assert mesh is not None
        assert mesh.cp is not None
        assert int(mesh.cp.size()) == 2
        assert _groups(mesh.cp) == _consecutive_groups(world_size, 2)
        assert mesh.tp is None
        assert mesh.ep is None
        _dense_product(mesh, world_size)

        _ok(result_queue, rank)
    except Exception as exc:
        _fail(result_queue, rank, exc)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _ep_does_not_divide_inner_worker(
    rank: int, world_size: int, port: int, result_queue: Any
) -> None:
    try:
        _init_gloo(rank, world_size, port)
        with pytest.raises(ValueError, match="must divide dp_shard"):
            build_parallel_mesh(world_size=world_size, ep=3, device_type="cpu")
        _ok(result_queue, rank)
    except Exception as exc:
        _fail(result_queue, rank, exc)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@requires_gloo
class TestBuildParallelMesh:
    def test_plain_fsdp_returns_none(self):
        _spawn_ranks(_plain_fsdp_worker)

    def test_ep_groups_match_consecutive_last_axis(self):
        _spawn_ranks(_ep_groups_worker, world_size=4)

    def test_edp_shard_is_none_when_ep_spans_the_inner_group(self):
        _spawn_ranks(_ep_spans_inner_worker)

    def test_tp_groups_are_consecutive_and_dense_fsdp_excludes_tp(self):
        _spawn_ranks(_tp_groups_worker, world_size=4)

    def test_hsdp_replica_groups_match_consecutive_shard_split(self):
        _spawn_ranks(_hsdp_replica_worker, world_size=4)

    def test_ep_groups_unchanged_when_tp_splits_dense_fsdp(self):
        _spawn_ranks(_ep_and_tp_worker, world_size=4)

    def test_cp_two_builds_titan_pairs(self):
        _spawn_ranks(_cp_pairs_worker, world_size=4)

    def test_ep_that_does_not_divide_inner_raises(self):
        _spawn_ranks(_ep_does_not_divide_inner_worker, world_size=4)


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


class TestBuildParallelMeshDefaultDevice:
    def test_infers_cpu_when_device_type_is_omitted(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        with (
            patch.object(dist, "is_available", return_value=True),
            patch.object(dist, "is_initialized", return_value=True),
            patch(
                "agilerl.distributed.mesh.init_device_mesh",
                side_effect=AssertionError("stop"),
            ) as init_mesh,
        ):
            with pytest.raises(AssertionError, match="stop"):
                build_parallel_mesh(world_size=2, ep=2)

        init_mesh.assert_called_once_with(
            "cpu", (2,), mesh_dim_names=("world",)
        )

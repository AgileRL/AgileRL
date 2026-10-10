# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for expert parallel mesh, sharding, dispatch, and parity.

CPU/gloo tests cover mesh construction, shard placement, token round-trip,
sliced expert restore, and FSDP materialization. CUDA tests check routed and
sorted packed-expert forward+backward exactly match a dense reference; each
rank must feed a disjoint batch shard, since expert grads sum over the
global batch (feeding every rank the same batch would double grads versus
the reference).
"""

from __future__ import annotations

import copy
import gc
import os
import sys
import tempfile
import weakref
from functools import partial
from types import MethodType, SimpleNamespace
from typing import Any, ClassVar
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import save_file
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, distribute_tensor
from torch.distributed.tensor.placement_types import Replicate, Shard
from torch.utils.checkpoint import checkpoint, noop_context_fn
from transformers import NemotronHConfig
from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHMLP

from agilerl.distributed import FSDPConfig, ep_forward, expert_parallel
from agilerl.distributed.ep_forward import (
    _call_with_local_params,
    _comm_stream,
    _gathered_expert_block,
    _hand_over,
    _install_routed_ep_forward,
    _install_sorted_ep_forward,
    _local_param_dict,
    _on_comm_stream,
    _route_replicated_span,
    _routed_up_weight,
    _routing_override,
    _run_local_experts,
    _sorted_weight,
    _token_adapter_ids,
    _wait_for,
)
from agilerl.distributed.ep_mesh import _split_shard_axis
from agilerl.distributed.expert_parallel import (
    LocalExperts,
    ParallelMesh,
    TokenAdapterIds,
    TokenDispatchState,
    _outer_expert_wrappers,
    _shard_lora_linear_on_ep,
    _shard_wrapper_adapters,
    apply_expert_parallel,
    assert_packed_experts_ep_sharded,
    build_parallel_mesh,
    expert_local_tensor,
    expert_param_bytes_local,
    iter_packed_expert_modules,
    num_packed_experts,
    packed_expert_count,
    reference_dispatch_combine,
    routed_counts_contexts,
    scatter_scaled_expert_rows,
    shard_experts_on_ep,
    token_combine,
    token_dispatch,
    tp_data_parallel_size,
)
from agilerl.distributed.fsdp import (
    _copy_indexed_weights,
    _ep_expert_live_keys,
    _scatter_ep_expert_slices,
    _write_ep_expert_slice,
    _write_full_tensor,
    materialize_fsdp2_from_cpu_state,
)
from agilerl.distributed.runtime import DPRuntime, FSDPRuntime
from agilerl.lora.fused import ROUTING_STATE
from agilerl.lora.moe import (
    set_routed_experts_chunk_bytes,
    set_routed_experts_recompute,
)
from agilerl.lora.moe import wrappers as moe_wrappers
from agilerl.utils.llm_utils import make_llm_optimizer

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


class TestScatterScaledExpertRows:
    def test_matches_product(self):
        torch.manual_seed(0)
        rows, hidden, tokens = 20, 4, 6
        index = torch.randint(0, tokens, (rows,))
        hidden_states = torch.randn(tokens, hidden)

        def run(inplace_path: bool):
            leaf = torch.randn(rows, hidden, requires_grad=True)
            weights = torch.rand(rows, 1, requires_grad=True)
            combined = leaf * 1
            if inplace_path:
                out = scatter_scaled_expert_rows(
                    combined, weights, index, hidden_states, 64 * 1024 * 1024
                )
            else:
                out = torch.zeros_like(hidden_states)
                out.index_add_(0, index, combined * weights)
            out.sum().backward()
            return out.detach(), leaf.grad.detach(), weights.grad.detach()

        torch.manual_seed(1)
        reference = run(False)
        torch.manual_seed(1)
        scaled = run(True)
        assert torch.allclose(scaled[0], reference[0])
        assert torch.allclose(scaled[1], reference[1])
        assert torch.allclose(scaled[2], reference[2])

    def test_multi_chunk_bf16_accumulates_in_fp32(self):
        # Arrange
        torch.manual_seed(0)
        rows, hidden, tokens = 23, 4, 5
        # 3 fp32 rows per chunk: 8 chunks, the last one partial.
        chunk_bytes = 3 * hidden * 4
        index = torch.randint(0, tokens, (rows,))
        hidden_states = torch.randn(tokens, hidden, dtype=torch.bfloat16)
        leaf = torch.randn(rows, hidden, dtype=torch.bfloat16, requires_grad=True)
        weights = torch.rand(rows, 1, requires_grad=True)
        upstream = torch.randn(tokens, hidden, dtype=torch.bfloat16)
        ref_leaf = leaf.detach().clone().requires_grad_(True)
        ref_weights = weights.detach().clone().requires_grad_(True)
        reference = torch.zeros(tokens, hidden).index_add_(
            0, index, ref_leaf.float() * ref_weights
        )
        (reference.to(torch.bfloat16) * upstream).sum().backward()

        # Act
        out = scatter_scaled_expert_rows(
            leaf * 1, weights, index, hidden_states, chunk_bytes
        )
        (out * upstream).sum().backward()

        # Assert
        assert out.dtype == torch.bfloat16
        assert torch.equal(out, reference.to(torch.bfloat16))
        assert torch.equal(leaf.grad, ref_leaf.grad)
        assert torch.allclose(weights.grad, ref_weights.grad, rtol=1e-6, atol=1e-6)

    def test_zero_rows_returns_zeros(self):
        hidden_states = torch.randn(3, 4, dtype=torch.bfloat16)

        out = scatter_scaled_expert_rows(
            torch.empty(0, 4, dtype=torch.bfloat16),
            torch.empty(0, 1),
            torch.empty(0, dtype=torch.long),
            hidden_states,
            64 * 1024 * 1024,
        )

        assert out.dtype == torch.bfloat16
        assert torch.equal(out, torch.zeros(3, 4, dtype=torch.bfloat16))


def _gloo_available() -> bool:
    if sys.platform == "win32":
        return False
    return dist.is_available()


requires_gloo = pytest.mark.skipif(not _gloo_available(), reason="gloo unavailable")
requires_2_cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="needs 2 CUDA devices",
)


def _init_gloo(rank: int, world_size: int, port: int) -> None:
    os.environ.update(
        {
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
        }
    )
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)


def _init_nccl(rank: int, world_size: int, port: int) -> None:
    os.environ.update(
        {
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
        }
    )
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def _free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _spawn_ranks(worker, world_size: int = 2, timeout: float = 300.0) -> None:
    port = _free_port()
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    procs = [
        ctx.Process(target=worker, args=(rank, world_size, port, queue))
        for rank in range(world_size)
    ]
    for proc in procs:
        proc.start()
    results = [queue.get(timeout=timeout) for _ in range(world_size)]
    for proc in procs:
        proc.join(timeout=timeout)
        assert proc.exitcode == 0, f"rank exited {proc.exitcode}"
    for rank, status, err in sorted(results):
        assert status == "ok", f"rank {rank}: {err}"


class TestDataParallelFold:
    def test_dp_runtime_is_process_world(self):
        runtime = DPRuntime()

        assert runtime.data_parallel_world(4) == 4
        assert runtime.data_parallel_rank(3) == 3

    def test_fsdp_ep_one_is_process_world(self):
        runtime = FSDPRuntime(FSDPConfig(ep=1))

        assert runtime.data_parallel_world(4) == 4
        assert runtime.data_parallel_rank(3) == 3

    def test_fsdp_ep_gives_every_rank_its_own_shard(self):
        runtime = FSDPRuntime(FSDPConfig(ep=8, tp=1))

        assert runtime.data_parallel_world(16) == 16
        assert [runtime.data_parallel_rank(rank) for rank in range(16)] == list(
            range(16)
        )

    def test_fsdp_tp_folds_world(self):
        runtime = FSDPRuntime(FSDPConfig(ep=2, tp=2))

        assert runtime.data_parallel_world(4) == 2
        assert runtime.data_parallel_rank(3) == 1

    def test_fsdp_tp_rejects_indivisible_world(self):
        runtime = FSDPRuntime(FSDPConfig(tp=2))

        with pytest.raises(ValueError, match="divisible by tp"):
            runtime.data_parallel_world(3)


class _PackedExperts(nn.Module):
    def __init__(self, num_experts: int, out_features: int, in_features: int):
        super().__init__()
        self.weight = nn.Parameter(
            torch.arange(
                num_experts * out_features * in_features, dtype=torch.float32
            ).reshape(num_experts, out_features, in_features)
        )


def _shard_worker(rank: int, world_size: int, port: int, result_queue: Any) -> None:
    try:
        _init_gloo(rank, world_size, port)
        ep = world_size
        num_experts = 4
        mesh = build_parallel_mesh(world_size=world_size, ep=ep, device_type="cpu")
        assert mesh is not None
        assert mesh.leftover_dp == 1
        assert tuple(mesh.world.mesh_dim_names) == ("replicate", "shard")
        assert mesh.ep.size() == ep
        assert mesh.ep.ndim == 1
        assert mesh.dp_mod_ep is None
        assert mesh.hsdp.size() == mesh.world.size()
        assert mesh.hsdp is not mesh.ep

        module = _PackedExperts(num_experts, out_features=3, in_features=2)
        for param in module.parameters():
            dist.broadcast(param.data, src=0)

        shard_experts_on_ep(module, mesh.ep)
        local = expert_local_tensor(module.weight)
        local_e = num_experts // ep
        assert local.shape[0] == local_e

        full = module.weight.full_tensor().cpu()
        expected = torch.arange(num_experts * 3 * 2, dtype=torch.float32).reshape(
            num_experts, 3, 2
        )
        assert torch.allclose(full, expected)
        start = rank * local_e
        assert torch.allclose(local.cpu(), expected[start : start + local_e])
        bytes_local = expert_param_bytes_local(module)
        bytes_full = expected.numel() * expected.element_size()
        assert bytes_local == bytes_full // ep

        result_queue.put((rank, "ok", None))
    except Exception as exc:  # pragma: no cover
        result_queue.put((rank, "err", repr(exc)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@requires_gloo
class TestExpertShardPlacement:
    def test_shard0_local_experts_and_full_tensor_roundtrip(self):
        _spawn_ranks(_shard_worker)

    def test_reshard_slices_densified_full_e_replica(self):
        _spawn_ranks(_densify_reshard_worker)


def _dp_ep_mesh_worker(
    rank: int, world_size: int, port: int, result_queue: Any
) -> None:
    try:
        _init_gloo(rank, world_size, port)
        ep = 2
        leftover_dp = world_size // ep
        mesh = build_parallel_mesh(world_size=world_size, ep=ep, device_type="cpu")

        assert mesh is not None
        assert mesh.leftover_dp == leftover_dp
        assert tuple(mesh.world.mesh_dim_names) == ("replicate", "shard")
        assert mesh.ep.size() == ep
        assert mesh.dp_mod_ep.size() == leftover_dp
        assert mesh.ep_replicas.size() == leftover_dp
        assert mesh.fsdp_experts.mesh_dim_names == ("dp", "ep")
        assert mesh.fsdp_experts.size() == world_size
        assert mesh.hsdp.size() == mesh.world.size()
        assert mesh.hsdp.ndim == 1
        assert mesh.hsdp is not mesh.ep
        assert mesh.tp is None
        assert mesh.tp_replicas is None

        result_queue.put((rank, "ok", None))
    except Exception as exc:  # pragma: no cover
        result_queue.put((rank, "err", repr(exc)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _size_one_dp_mesh_worker(
    rank: int, world_size: int, port: int, result_queue: Any
) -> None:
    try:
        _init_gloo(rank, world_size, port)
        ep = world_size
        mesh = build_parallel_mesh(world_size=world_size, ep=ep, device_type="cpu")

        assert mesh is not None
        assert mesh.leftover_dp == 1
        assert tuple(mesh.world.mesh_dim_names) == ("replicate", "shard")
        assert mesh.ep.size() == ep
        assert mesh.ep.mesh_dim_names == ("ep",)
        assert mesh.dp_mod_ep is None
        assert mesh.fsdp_experts is None
        assert mesh.ep_replicas.size() == 1
        assert mesh.hsdp.size() == mesh.world.size()
        assert mesh.hsdp.ndim == 1
        assert mesh.hsdp != mesh.ep
        assert mesh.tp is None
        assert mesh.tp_replicas is None

        result_queue.put((rank, "ok", None))
    except Exception as exc:  # pragma: no cover
        result_queue.put((rank, "err", repr(exc)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@requires_gloo
class TestBuildParallelMesh:
    def test_ep_spanning_shard_group_builds_no_dp_mesh(self):
        _spawn_ranks(_size_one_dp_mesh_worker)

    def test_leftover_dp_gt_one_names_dp_and_ep(self):
        _spawn_ranks(_dp_ep_mesh_worker, world_size=4)

    def test_tensor_parallel_only_builds_no_ep_views(self):
        _spawn_ranks(_tp_only_mesh_worker, world_size=4)

    def test_hsdp_only_builds_no_ep_or_tp_views(self):
        _spawn_ranks(_hsdp_only_mesh_worker, world_size=4)


def _tp_only_mesh_worker(
    rank: int, world_size: int, port: int, result_queue: Any
) -> None:
    try:
        _init_gloo(rank, world_size, port)

        mesh = build_parallel_mesh(world_size=world_size, tp=2, device_type="cpu")

        assert mesh is not None
        assert mesh.tp.size() == 2
        assert mesh.tp_replicas.size() == world_size // 2
        assert mesh.ep is None
        assert mesh.dp_mod_ep is None
        assert mesh.ep_replicas is None
        assert mesh.fsdp_experts is None
        result_queue.put((rank, "ok", None))
    except Exception as exc:  # pragma: no cover
        result_queue.put((rank, "err", repr(exc)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _hsdp_only_mesh_worker(
    rank: int, world_size: int, port: int, result_queue: Any
) -> None:
    try:
        _init_gloo(rank, world_size, port)

        mesh = build_parallel_mesh(
            world_size=world_size, shard_group_size=2, device_type="cpu"
        )

        assert mesh is not None
        assert mesh.hsdp.mesh_dim_names == ("replicate", "shard")
        assert mesh.leftover_dp == 2
        assert mesh.ep is None
        assert mesh.ep_replicas is None
        assert mesh.tp is None
        assert mesh.tp_replicas is None
        result_queue.put((rank, "ok", None))
    except Exception as exc:  # pragma: no cover
        result_queue.put((rank, "err", repr(exc)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _densify_reshard_worker(
    rank: int, world_size: int, port: int, result_queue: Any
) -> None:
    try:
        _init_gloo(rank, world_size, port)
        ep = world_size
        num_experts = 4
        mesh = build_parallel_mesh(world_size=world_size, ep=ep, device_type="cpu")
        assert mesh is not None
        module = _PackedExperts(num_experts, out_features=3, in_features=2)
        for param in module.parameters():
            dist.broadcast(param.data, src=0)
        shard_experts_on_ep(module, mesh.ep)
        full = module.weight.full_tensor().detach().cpu().contiguous()
        module.register_parameter("weight", nn.Parameter(full.clone()))
        shard_experts_on_ep(module, mesh.ep)
        local = expert_local_tensor(module.weight)
        local_e = num_experts // ep
        assert local.shape[0] == local_e
        start = rank * local_e
        assert torch.allclose(local.cpu(), full[start : start + local_e])

        result_queue.put((rank, "ok", None))
    except Exception as exc:  # pragma: no cover
        result_queue.put((rank, "err", repr(exc)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class TestAssertPackedExpertsEpSharded:
    def test_full_e_dense_raises(self):
        class PackedExperts(nn.Module):
            def __init__(self):
                super().__init__()
                self.up_proj = nn.Parameter(torch.ones(4, 8, 4))
                self.down_proj = nn.Parameter(torch.ones(4, 4, 8))
                self.act_fn = nn.SiLU()

            def forward(self, hidden_states, top_k_index, top_k_weights):
                return hidden_states

        with pytest.raises(RuntimeError, match="local expert dim"):
            assert_packed_experts_ep_sharded(PackedExperts(), ep=2)

    def test_ep_without_packed_modules_raises(self):
        with pytest.raises(RuntimeError, match="packed expert"):
            assert_packed_experts_ep_sharded(nn.Linear(2, 2), ep=2)

    def test_saved_refs_checked_when_forward_no_longer_matches(self):
        class PackedExperts(nn.Module):
            def __init__(self):
                super().__init__()
                self.up_proj = nn.Parameter(torch.ones(4, 8, 4))
                self.down_proj = nn.Parameter(torch.ones(4, 4, 8))
                self.act_fn = nn.SiLU()

            def forward(self, *args, **kwargs):
                return args[0]

        experts = PackedExperts()
        assert iter_packed_expert_modules(experts) == []

        with pytest.raises(RuntimeError, match="local expert dim"):
            assert_packed_experts_ep_sharded(experts, ep=2, modules=[experts])


def _a2a_worker(rank: int, world_size: int, port: int, result_queue: Any) -> None:
    try:
        _init_gloo(rank, world_size, port)
        ep = world_size
        num_experts = 4
        num_local = num_experts // ep
        hidden = 8

        torch.manual_seed(rank + 1)
        n_tokens = 12
        tokens = torch.randn(n_tokens, hidden)
        expert_ids = torch.randint(
            0,
            num_experts,
            (n_tokens,),
            generator=torch.Generator().manual_seed(rank + 7),
        )
        order = torch.argsort(expert_ids, stable=True)
        sorted_tokens = tokens[order]
        sorted_ids = expert_ids[order]
        counts = torch.bincount(sorted_ids, minlength=num_experts).to(torch.long)

        mesh = build_parallel_mesh(world_size=world_size, ep=ep, device_type="cpu")
        assert mesh is not None
        local_tokens, local_counts, state = token_dispatch(
            sorted_tokens,
            counts,
            ep_group=mesh.ep.get_group(),
            ep_degree=ep,
            num_local_experts=num_local,
        )
        assert local_counts.numel() == num_local
        assert local_tokens.shape[0] == int(local_counts.sum().item())
        assert int(local_counts.sum().item()) == sum(state.output_splits)

        combined = token_combine(local_tokens, state)
        assert combined.shape == sorted_tokens.shape
        max_err = (combined - sorted_tokens).abs().max().item()
        assert max_err == 0.0, f"round-trip max abs err {max_err}"

        ref = reference_dispatch_combine(
            tokens, expert_ids, ep_degree=ep, num_experts=num_experts
        )
        assert torch.allclose(ref, tokens, atol=0, rtol=0)

        result_queue.put((rank, "ok", None))
    except Exception as exc:  # pragma: no cover
        result_queue.put((rank, "err", repr(exc)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class TestTokenDispatchCombine:
    def test_all_to_all_roundtrip_restores_sorted_tokens(self):
        _spawn_ranks(_a2a_worker)


class _RoutedExperts(nn.Module):
    def __init__(self, num_experts: int, hidden: int, intermediate: int):
        super().__init__()
        self.up_proj = nn.Parameter(torch.randn(num_experts, intermediate, hidden))
        self.down_proj = nn.Parameter(torch.randn(num_experts, hidden, intermediate))
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states, top_k_index, top_k_weights):
        return hidden_states


def _wrap_worker(rank: int, world_size: int, port: int, result_queue: Any) -> None:
    try:
        _init_gloo(rank, world_size, port)
        from agilerl.lora.moe import (
            install_packed_expert_grouped_gemm,
        )

        torch.manual_seed(0)
        experts = _RoutedExperts(num_experts=4, hidden=4, intermediate=8)
        for param in experts.parameters():
            dist.broadcast(param.data, src=0)
        install_packed_expert_grouped_gemm(experts)

        torch.manual_seed(rank + 3)
        hidden = torch.randn(6, 4)
        logits = torch.randn(6, 4)
        top_k_weights, top_k_index = torch.softmax(logits, dim=-1).topk(2, dim=-1)
        with torch.no_grad():
            dense = experts(hidden, top_k_index, top_k_weights)

        mesh = build_parallel_mesh(
            world_size=world_size, ep=world_size, device_type="cpu"
        )
        assert mesh is not None
        apply_expert_parallel(experts, mesh.ep)
        out = experts(hidden, top_k_index, top_k_weights)
        assert torch.allclose(out, dense, atol=1e-5, rtol=1e-5)

        result_queue.put((rank, "ok", None))
    except Exception as exc:  # pragma: no cover
        result_queue.put((rank, "err", repr(exc)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class TestCallWithLocalParams:
    def test_bound_routed_forward_receives_hidden_index_and_weights(self):
        class Experts(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(2, 3, 4))

            def forward(self, hidden_states, top_k_index, top_k_weights):
                return (
                    hidden_states * top_k_weights.sum()
                    + top_k_index.to(hidden_states.dtype).sum()
                )

        torch.manual_seed(0)
        module = Experts()
        hidden = torch.randn(5, 4)
        index = torch.arange(5).unsqueeze(-1)
        weights = torch.ones(5, 1)
        expected = module(hidden, index, weights)

        out = _call_with_local_params(
            module, module.forward, _local_param_dict(module), hidden, index, weights
        )

        assert torch.equal(out, expected)


@requires_gloo
class TestApplyExpertParallel:
    def test_routed_forward_matches_dense_on_this_ranks_tokens(self):
        _spawn_ranks(_wrap_worker)


def _tiny_moe_actor(
    num_experts: int, hidden: int = 4, intermediate: int = 8
) -> nn.Module:
    class PackedExperts(nn.Module):
        def __init__(self):
            super().__init__()
            self.up_proj = nn.Parameter(torch.ones(num_experts, intermediate, hidden))
            self.down_proj = nn.Parameter(torch.ones(num_experts, hidden, intermediate))
            self.act_fn = nn.SiLU()

        def forward(self, hidden_states, top_k_index, top_k_weights):
            return hidden_states

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(hidden, hidden)
            self.experts = PackedExperts()

    class Model(nn.Module):
        _no_split_modules: ClassVar = ["Block"]

        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Block()])

    return Model()


def _materialize_and_capture_meshes(
    model: nn.Module, config: FSDPConfig
) -> tuple[ParallelMesh, dict[str, Any]]:
    from agilerl.distributed import fsdp as fsdp_mod

    captured: dict[str, Any] = {}
    real_apply = fsdp_mod.apply_fsdp2

    def _apply(
        sharded: nn.Module,
        apply_config: FSDPConfig | None = None,
        *,
        mesh: Any = None,
        expert_mesh: Any = None,
        gradient_checkpointing: bool = False,
    ):
        captured["mesh"] = mesh
        captured["expert_mesh"] = expert_mesh
        return real_apply(
            sharded,
            apply_config,
            mesh=mesh,
            expert_mesh=expert_mesh,
            gradient_checkpointing=gradient_checkpointing,
        )

    with patch.object(fsdp_mod, "apply_fsdp2", side_effect=_apply):
        mesh = _materialize(model, config)
    return mesh, captured


def _materialize(model: nn.Module, config: FSDPConfig) -> ParallelMesh | None:
    mesh = build_parallel_mesh(
        ep=config.ep,
        tp=config.tp,
        shard_group_size=config.shard_group_size,
        device_type="cpu",
    )
    materialize_fsdp2_from_cpu_state(model, "cpu", config, parallel_mesh=mesh)
    return mesh


def _materialize_ep_worker(
    rank: int, world_size: int, port: int, result_queue: Any
) -> None:
    try:
        _init_gloo(rank, world_size, port)
        num_experts = 4
        model = _tiny_moe_actor(num_experts)
        for param in model.parameters():
            dist.broadcast(param.data, src=0)
        original_up = model.layers[0].experts.up_proj.detach().cpu().clone()
        original_down = model.layers[0].experts.down_proj.detach().cpu().clone()

        ep_mesh, captured = _materialize_and_capture_meshes(
            model,
            FSDPConfig(
                ep=world_size, wrap_every_n_blocks=1, routed_expert_chunk_mib=64
            ),
        )
        assert ep_mesh.leftover_dp == 1
        assert tuple(ep_mesh.world.mesh_dim_names) == ("replicate", "shard")
        assert ep_mesh.hsdp.size() == ep_mesh.world.size()
        assert captured["mesh"] is ep_mesh.hsdp
        assert captured["expert_mesh"] is None
        assert captured["mesh"] is not ep_mesh.ep

        experts = model.layers[0].experts
        local = expert_local_tensor(experts.up_proj)
        local_e = num_experts // world_size
        assert isinstance(experts.up_proj, DTensor), type(experts.up_proj)
        assert local.shape[0] == local_e, tuple(local.shape)
        assert torch.allclose(experts.up_proj.full_tensor().cpu(), original_up)
        assert torch.allclose(experts.down_proj.full_tensor().cpu(), original_down)
        assert (
            expert_param_bytes_local(experts)
            == (
                experts.up_proj.full_tensor().numel()
                + experts.down_proj.full_tensor().numel()
            )
            * experts.up_proj.full_tensor().element_size()
            // world_size
        )

        result_queue.put((rank, "ok", None))
    except Exception as exc:  # pragma: no cover
        result_queue.put((rank, "err", repr(exc)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _materialize_dp_ep_worker(
    rank: int, world_size: int, port: int, result_queue: Any
) -> None:
    try:
        _init_gloo(rank, world_size, port)
        ep = 2
        num_experts = 4
        model = _tiny_moe_actor(num_experts)
        for param in model.parameters():
            dist.broadcast(param.data, src=0)
        original_up = model.layers[0].experts.up_proj.detach().cpu().clone()
        original_down = model.layers[0].experts.down_proj.detach().cpu().clone()

        ep_mesh, captured = _materialize_and_capture_meshes(
            model, FSDPConfig(ep=ep, wrap_every_n_blocks=1, routed_expert_chunk_mib=64)
        )
        assert ep_mesh.leftover_dp == world_size // ep
        assert captured["mesh"] is ep_mesh.hsdp
        assert captured["expert_mesh"] is ep_mesh.dp_mod_ep
        assert captured["mesh"] is not ep_mesh.ep

        experts = model.layers[0].experts
        local = expert_local_tensor(experts.up_proj)
        local_e = num_experts // ep
        assert isinstance(experts.up_proj, DTensor), type(experts.up_proj)
        assert local.shape[0] == local_e, tuple(local.shape)
        assert torch.allclose(experts.up_proj.full_tensor().cpu(), original_up)
        assert torch.allclose(experts.down_proj.full_tensor().cpu(), original_down)

        result_queue.put((rank, "ok", None))
    except Exception as exc:  # pragma: no cover
        result_queue.put((rank, "err", repr(exc)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@requires_gloo
class TestMaterializeExpertParallel:
    def test_materialize_keeps_local_expert_count_at_e_over_ep(self):
        _spawn_ranks(_materialize_ep_worker)

    def test_materialize_uses_hsdp_and_dp_mod_ep_when_leftover_dp_gt_one(self):
        _spawn_ranks(_materialize_dp_ep_worker, world_size=4)

    def test_dp_gt_one_experts_restore_via_sliced_write(self):
        _spawn_ranks(_dp_gt_one_slice_worker, world_size=4)


def _check_expert_dim0_world_sharded(
    rank: int, world_size: int, port: int, result_queue: Any, ep: int
) -> None:
    """Every expert param shards dim 0 across all ranks, so no single-axis
    gather (FSDP forward included) rebuilds full-E on one rank.
    """
    try:
        _init_gloo(rank, world_size, port)
        num_experts = 8
        # Above the FSDP persistence threshold so experts take the real
        # wrapped path instead of staying replicated.
        torch.manual_seed(0)
        model = _tiny_moe_actor(num_experts, hidden=256, intermediate=512)
        for param in model.parameters():
            dist.broadcast(param.data, src=0)
        _materialize(
            model, FSDPConfig(ep=ep, wrap_every_n_blocks=1, routed_expert_chunk_mib=64)
        )
        for _name, module in iter_packed_expert_modules(model):
            for param in module.parameters(recurse=False):
                assert isinstance(param, DTensor), type(param)
                assert param.shape[0] == num_experts, tuple(param.shape)
                local = expert_local_tensor(param)
                assert local.shape[0] == num_experts // world_size, tuple(local.shape)
        result_queue.put((rank, "ok", None))
    except Exception as exc:  # pragma: no cover
        result_queue.put((rank, "err", repr(exc)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _unit_size_dp2_ep2_worker(
    rank: int, world_size: int, port: int, result_queue: Any
) -> None:
    _check_expert_dim0_world_sharded(rank, world_size, port, result_queue, ep=2)


def _unit_size_dp1_ep4_worker(
    rank: int, world_size: int, port: int, result_queue: Any
) -> None:
    _check_expert_dim0_world_sharded(rank, world_size, port, result_queue, ep=4)


@requires_gloo
class TestExpertDim0WorldSharded:
    def test_dp2_ep2_experts_shard_dim0_across_world(self):
        _spawn_ranks(_unit_size_dp2_ep2_worker, world_size=4)

    def test_dp1_ep4_experts_shard_dim0_across_world(self):
        _spawn_ranks(_unit_size_dp1_ep4_worker, world_size=4)


def _resume_slice_worker(
    rank: int, world_size: int, port: int, result_queue: Any
) -> None:
    """Resume restores experts via the sliced write, dense params as before."""
    try:
        from agilerl.distributed import fsdp as fsdp_mod
        from agilerl.distributed.runtime import FSDPRuntime

        _init_gloo(rank, world_size, port)
        num_experts = 8
        # Above the FSDP persistence threshold so experts take the real
        # wrapped path instead of staying replicated.
        torch.manual_seed(0)
        model = _tiny_moe_actor(num_experts, hidden=256, intermediate=512)
        for param in model.parameters():
            dist.broadcast(param.data, src=0)
        shapes = {name: tuple(param.shape) for name, param in model.named_parameters()}
        _materialize(
            model, FSDPConfig(ep=2, wrap_every_n_blocks=1, routed_expert_chunk_mib=64)
        )
        state = {
            name: torch.full(
                shapes[name], 7.0 if "experts" in name else 3.0, dtype=torch.float32
            )
            for name in shapes
        }
        full_ids: set[int] = set()
        slice_ids: set[int] = set()
        real_full = fsdp_mod._write_full_tensor
        real_slice = fsdp_mod._write_ep_expert_slice

        def spy_full(dest: nn.Parameter, value: torch.Tensor) -> None:
            full_ids.add(id(dest))
            real_full(dest, value)

        def spy_slice(dest: nn.Parameter, value: torch.Tensor) -> None:
            slice_ids.add(id(dest))
            real_slice(dest, value)

        with (
            patch.object(fsdp_mod, "_write_full_tensor", spy_full),
            patch.object(fsdp_mod, "_write_ep_expert_slice", spy_slice),
        ):
            FSDPRuntime(FSDPConfig()).import_model_state(model, state, strict=True)
        experts = {
            id(param)
            for _name, module in iter_packed_expert_modules(model)
            for param in module.parameters(recurse=False)
        }
        assert experts, "no packed experts found"
        assert experts <= slice_ids, "resume must restore experts via sliced write"
        assert not experts & full_ids, "resume must not stage full experts"
        assert full_ids, "dense params must still use the full-tensor path"
        assert torch.equal(
            model.layers[0].experts.up_proj.full_tensor().cpu(),
            torch.full((num_experts, 512, 256), 7.0),
        )
        assert torch.equal(
            model.layers[0].lin.weight.detach().cpu(),
            torch.full((256, 256), 3.0),
        )
        result_queue.put((rank, "ok", None))
    except Exception as exc:  # pragma: no cover
        result_queue.put((rank, "err", repr(exc)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@requires_gloo
class TestResumeExpertSlices:
    def test_dp2_ep2_resume_restores_experts_via_sliced_write(self):
        _spawn_ranks(_resume_slice_worker, world_size=4)


def _dp_gt_one_slice_worker(
    rank: int, world_size: int, port: int, result_queue: Any
) -> None:
    """Dense restore at dp>1 writes experts via the sliced path only."""
    try:
        from agilerl.distributed import fsdp as fsdp_mod

        _init_gloo(rank, world_size, port)
        num_experts = 8
        # Above the FSDP persistence threshold so experts take the real
        # wrapped path instead of staying replicated.
        torch.manual_seed(0)
        model = _tiny_moe_actor(num_experts, hidden=256, intermediate=512)
        for param in model.parameters():
            dist.broadcast(param.data, src=0)
        original_up = model.layers[0].experts.up_proj.detach().cpu().clone()
        original_down = model.layers[0].experts.down_proj.detach().cpu().clone()
        full_ids: set[int] = set()
        slice_ids: set[int] = set()
        real_full = fsdp_mod._write_full_tensor
        real_slice = fsdp_mod._write_ep_expert_slice

        def spy_full(dest: nn.Parameter, value: torch.Tensor) -> None:
            full_ids.add(id(dest))
            real_full(dest, value)

        def spy_slice(dest: nn.Parameter, value: torch.Tensor) -> None:
            slice_ids.add(id(dest))
            real_slice(dest, value)

        with (
            patch.object(fsdp_mod, "_write_full_tensor", spy_full),
            patch.object(fsdp_mod, "_write_ep_expert_slice", spy_slice),
        ):
            _materialize(
                model,
                FSDPConfig(ep=2, wrap_every_n_blocks=1, routed_expert_chunk_mib=64),
            )
        experts = {
            id(param)
            for _name, module in iter_packed_expert_modules(model)
            for param in module.parameters(recurse=False)
        }
        assert experts, "no packed experts found"
        assert experts <= slice_ids, "experts must restore via the sliced write"
        assert not experts & full_ids, "experts must not take the full-tensor path"
        assert torch.equal(
            model.layers[0].experts.up_proj.full_tensor().cpu(), original_up
        )
        assert torch.equal(
            model.layers[0].experts.down_proj.full_tensor().cpu(), original_down
        )
        result_queue.put((rank, "ok", None))
    except Exception as exc:  # pragma: no cover
        result_queue.put((rank, "err", repr(exc)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _routed_parity_worker(
    rank: int, world_size: int, port: int, result_queue: Any
) -> None:
    """Routed MoE forward+backward matches a dense reference, ep=2."""
    try:
        _init_nccl(rank, world_size, port)
        from agilerl.lora.moe import routed_experts_local_forward

        torch.manual_seed(1)
        global_hidden = torch.randn(64, 16)
        global_index = torch.randint(0, 4, (64, 2))
        global_weights = torch.rand(64, 2)
        global_weights = global_weights / global_weights.sum(dim=-1, keepdim=True)
        hidden = global_hidden[rank * 32 : (rank + 1) * 32].cuda()
        top_k_index = global_index[rank * 32 : (rank + 1) * 32].cuda()
        top_k_weights = global_weights[rank * 32 : (rank + 1) * 32].cuda()

        torch.manual_seed(0)
        ref = _RoutedExperts(num_experts=4, hidden=16, intermediate=8).cuda()
        ref.forward = MethodType(routed_experts_local_forward, ref)
        ref_out = ref(global_hidden.cuda(), global_index.cuda(), global_weights.cuda())
        ref_out.sum().backward()
        ref_grads = {n: p.grad.clone() for n, p in ref.named_parameters()}

        torch.manual_seed(0)
        module = _RoutedExperts(num_experts=4, hidden=16, intermediate=8).cuda()
        module.forward = MethodType(routed_experts_local_forward, module)
        ep_mesh = build_parallel_mesh(ep=2)
        assert ep_mesh is not None
        assert apply_expert_parallel(module, ep_mesh.ep) == 1
        out = module(hidden, top_k_index, top_k_weights)
        assert torch.allclose(out, ref_out[rank * 32 : (rank + 1) * 32], atol=1e-4)
        out.sum().backward()
        for name, param in module.named_parameters():
            local_grad = expert_local_tensor(param.grad)
            local_ref = ref_grads[name].chunk(2, dim=0)[rank]
            assert torch.allclose(local_grad, local_ref, atol=1e-4), name

        result_queue.put((rank, "ok", None))
    except Exception as exc:  # pragma: no cover
        result_queue.put((rank, "err", repr(exc)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.gpu
@requires_2_cuda
class TestRoutedExpertParallelParity:
    def test_forward_and_param_grads_match_dense_reference(self):
        _spawn_ranks(_routed_parity_worker)


class _SortedExperts(nn.Module):
    def __init__(self, num_experts: int, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_experts, out_features, in_features))

    def forward(self, inputs, expert_size):
        return inputs


def _sorted_parity_worker(
    rank: int, world_size: int, port: int, result_queue: Any
) -> None:
    """Sorted MoE forward+backward matches a dense reference, ep=2."""
    try:
        _init_nccl(rank, world_size, port)
        from agilerl.lora.moe.grouped_gemm import grouped_linear

        torch.manual_seed(1)
        global_inputs = torch.randn(64, 16)
        torch.manual_seed(2)
        global_upstream = torch.randn(64, 8)
        inputs = global_inputs[rank * 32 : (rank + 1) * 32].cuda()
        counts = torch.tensor(
            [16, 16, 0, 0] if rank == 0 else [0, 0, 16, 16], device="cuda"
        )

        torch.manual_seed(0)
        ref = _SortedExperts(num_experts=4, in_features=16, out_features=8).cuda()
        ref.forward = MethodType(grouped_linear, ref)
        ref_out = ref(global_inputs.cuda(), [16, 16, 16, 16])
        # Materialize the upstream grad: torch._grouped_mm backward rejects
        # the zero-stride expanded grad a bare sum() would feed it.
        (ref_out * global_upstream.cuda()).sum().backward()
        ref_grad = ref.weight.grad.clone()

        torch.manual_seed(0)
        module = _SortedExperts(num_experts=4, in_features=16, out_features=8).cuda()
        module.forward = MethodType(grouped_linear, module)
        ep_mesh = build_parallel_mesh(ep=2)
        assert ep_mesh is not None
        assert apply_expert_parallel(module, ep_mesh.ep) == 1
        out = module(inputs, counts)
        assert torch.allclose(out, ref_out[rank * 32 : (rank + 1) * 32], atol=1e-4)
        upstream = global_upstream[rank * 32 : (rank + 1) * 32].cuda()
        (out * upstream).sum().backward()
        local_grad = expert_local_tensor(module.weight.grad)
        local_ref = ref_grad.chunk(2, dim=0)[rank]
        assert torch.allclose(local_grad, local_ref, atol=1e-4)

        result_queue.put((rank, "ok", None))
    except Exception as exc:  # pragma: no cover
        result_queue.put((rank, "err", repr(exc)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.gpu
@requires_2_cuda
class TestSortedExpertParallelParity:
    def test_forward_and_param_grads_match_dense_reference(self):
        _spawn_ranks(_sorted_parity_worker)


class _LoraExpertsBlock(nn.Module):
    """Packed-experts submodule holding the targeted parameters."""

    def __init__(self, num_experts: int = 4, hidden: int = 8, intermediate: int = 6):
        super().__init__()
        self.num_experts = num_experts
        self.gate_up_proj = nn.Parameter(
            torch.randn(num_experts, 2 * intermediate, hidden) * 0.1
        )
        self.down_proj = nn.Parameter(
            torch.randn(num_experts, hidden, intermediate) * 0.1
        )
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states, top_k_index, top_k_weights):
        return hidden_states


class _LoraRoutedExperts(nn.Module):
    """Qwen3-MoE-convention packed experts for PEFT adapter tests."""

    def __init__(self, num_experts: int = 4, hidden: int = 8, intermediate: int = 6):
        super().__init__()
        # Parameters live nested, as in real models: PEFT rejects
        # target_parameters on the top-level module (cyclic module graph).
        self.experts = _LoraExpertsBlock(num_experts, hidden, intermediate)

    def forward(self, hidden_states, top_k_index, top_k_weights):
        return self.experts(hidden_states, top_k_index, top_k_weights)


def _lora_chain_worker(
    rank: int, world_size: int, port: int, result_queue: Any
) -> None:
    """Every wrapper-chain link's adapters shard; frozen params skip the grad hook."""
    try:
        _init_gloo(rank, world_size, port)
        from peft import LoraConfig, inject_adapter_in_model

        from agilerl.distributed.expert_parallel import (
            _chain_links,
            _outer_expert_wrappers,
        )
        from agilerl.lora.moe import (
            upgrade_moe_param_wrappers,
        )

        torch.manual_seed(0)
        model = _LoraRoutedExperts()
        model = inject_adapter_in_model(
            LoraConfig(
                r=2,
                lora_alpha=4,
                lora_dropout=0.0,
                target_modules=[],
                target_parameters=["experts.gate_up_proj", "experts.down_proj"],
                init_lora_weights=False,
            ),
            model,
            adapter_name="actor",
        )
        upgraded = upgrade_moe_param_wrappers(model)
        assert upgraded == 1, f"upgraded={upgraded}"

        mesh = build_parallel_mesh(
            world_size=world_size, ep=world_size, device_type="cpu"
        )
        assert mesh is not None, "no mesh"
        applied = apply_expert_parallel(model, mesh.ep)
        assert applied == 1, f"applied={applied}"

        local_e = 4 // world_size
        links = 0
        for _name, module in iter_packed_expert_modules(model):
            for wrapper in _outer_expert_wrappers(model, module):
                for link in _chain_links(wrapper):
                    links += 1
                    for adapter in link.lora_A.values():
                        assert isinstance(adapter.weight, DTensor), (
                            type(link).__name__,
                            tuple(adapter.weight.shape),
                        )
                        local = expert_local_tensor(adapter.weight)
                        assert local.shape[0] == local_e * 2, tuple(local.shape)
                    for adapter in link.lora_B.values():
                        assert isinstance(adapter.weight, DTensor), (
                            type(link).__name__,
                            tuple(adapter.weight.shape),
                        )
        assert links >= 1, "expected at least one wrapper link"

        # Frozen DTensor params must not take the grad hook; trainable ones do.
        from torch.distributed.tensor import distribute_tensor
        from torch.distributed.tensor.placement_types import Shard

        class Toy(nn.Module):
            def __init__(self, frozen: bool):
                super().__init__()
                self.weight = nn.Parameter(
                    distribute_tensor(torch.randn(4, 8), mesh.ep, [Shard(0)]),
                    requires_grad=not frozen,
                )

            def forward(self, x):
                return x @ self.weight.T

        for frozen in (True, False):
            toy = Toy(frozen)
            assert isinstance(toy.weight, DTensor), type(toy.weight).__name__
            params = _local_param_dict(toy)
            assert params["weight"].requires_grad is (not frozen), (
                frozen,
                params["weight"].requires_grad,
            )
            out = _call_with_local_params(
                toy, toy.forward, _local_param_dict(toy), torch.randn(3, 8)
            )
            if frozen:
                assert not out.requires_grad, "frozen fwd needs no grad"
            else:
                out.sum().backward()
                assert toy.weight.grad is not None, "trainable param missing grad"

        result_queue.put((rank, "ok", None))
    except Exception as exc:  # pragma: no cover
        result_queue.put((rank, "err", repr(exc)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@requires_gloo
class TestExpertLoraChainSharding:
    def test_all_chain_links_shard_adapters_and_backward(self):
        _spawn_ranks(_lora_chain_worker)


class _EpLoraBlock(nn.Module):
    """Transformer block with a dense linear and PEFT-wrapped MoE experts."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(8, 8)
        self.moe = _LoraRoutedExperts()


class _EpLoraMoeModel(nn.Module):
    _no_split_modules: ClassVar = ["_EpLoraBlock"]

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_EpLoraBlock()])


def _lora_a_dim0(model: nn.Module) -> dict[str, int]:
    """Unsharded LoRA A dim 0, keyed by parameter name."""
    return {
        name: int(param.shape[0])
        for name, param in model.named_parameters()
        if "lora_A" in name and name.endswith("weight")
    }


def _ep_lora_fsdp_worker(
    rank: int, world_size: int, port: int, result_queue: Any
) -> None:
    """EP LoRA stays on the EP mesh under FSDP, including leftover dp > 1."""
    try:
        _init_gloo(rank, world_size, port)
        from peft import LoraConfig, inject_adapter_in_model

        from agilerl.distributed import fsdp as fsdp_mod
        from agilerl.distributed.fsdp_blocks import apply_fsdp2
        from agilerl.lora.moe import (
            upgrade_moe_param_wrappers,
        )

        torch.manual_seed(0)
        model = _EpLoraMoeModel()
        model = inject_adapter_in_model(
            LoraConfig(
                r=2,
                lora_alpha=4,
                lora_dropout=0.0,
                target_modules=[],
                target_parameters=[
                    "moe.experts.gate_up_proj",
                    "moe.experts.down_proj",
                ],
                init_lora_weights=False,
            ),
            model,
            adapter_name="actor",
        )
        upgraded = upgrade_moe_param_wrappers(model)
        assert upgraded == 1, f"upgraded={upgraded}"
        for param in model.parameters():
            dist.broadcast(param.data, src=0)
        unsharded_lora_a = _lora_a_dim0(model)
        assert unsharded_lora_a, "expected LoRA A weights"
        ep = 2
        mesh = build_parallel_mesh(world_size=world_size, ep=ep, device_type="cpu")
        assert mesh is not None, "no mesh"
        applied = apply_expert_parallel(model, mesh.ep)
        assert applied == 1, f"applied={applied}"
        before = {
            id(param) for param in model.parameters() if isinstance(param, DTensor)
        }
        assert before, "expected EP-sharded base and adapter params"
        expert_mesh = None if mesh.leftover_dp == 1 else mesh.dp_mod_ep
        apply_fsdp2(
            model,
            # Zero threshold: every parameter is managed, so stray EP
            # DTensors reach fully_shard (production adapters exceed the
            # default persistence threshold too).
            FSDPConfig(ep=ep, wrap_every_n_blocks=1, param_persistence_threshold=0),
            mesh=mesh.hsdp,
            expert_mesh=expert_mesh,
        )
        if mesh.leftover_dp == 1:
            after = {
                id(param)
                for param in model.parameters()
                if isinstance(param, DTensor) and param.device_mesh is mesh.ep
            }
            assert after == before, "adapters must skip FSDP untouched"
        else:
            found = False
            for name, param in model.named_parameters():
                if name not in unsharded_lora_a:
                    continue
                found = True
                assert isinstance(param, DTensor), name
                assert param.device_mesh is mesh.ep, name
                local = expert_local_tensor(param)
                unsharded_dim0 = unsharded_lora_a[name]
                ep_local = unsharded_dim0 // ep
                world_local = unsharded_dim0 // (ep * mesh.leftover_dp)
                assert ep_local != world_local, (unsharded_dim0, ep, mesh.leftover_dp)
                assert local.shape[0] == ep_local, (
                    name,
                    tuple(local.shape),
                    unsharded_dim0,
                    ep,
                )
                fsdp_mod._init_lora_parameter(
                    param,
                    name,
                    tuple(int(size) for size in param.shape),
                    tuple(param.placements),
                    param.device_mesh,
                )
                filled = expert_local_tensor(param)
                assert torch.isfinite(filled).all(), name
            assert found, "LoRA A missing after FSDP"
        result_queue.put((rank, "ok", None))
    except Exception as exc:  # pragma: no cover
        result_queue.put((rank, "err", repr(exc)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@requires_gloo
class TestExpertLoraFsdp:
    def test_dp1_ep2_adapters_skip_fsdp(self):
        _spawn_ranks(_ep_lora_fsdp_worker)

    def test_dp2_ep2_lora_a_local_dim_is_unsharded_over_ep(self):
        _spawn_ranks(_ep_lora_fsdp_worker, world_size=4)


def _arange_packed_moe(num_experts: int) -> nn.Module:
    """Packed experts whose row values are ``arange(num_experts)``."""

    class PackedExperts(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            rows = torch.arange(num_experts, dtype=torch.float32).view(
                num_experts, 1, 1
            )
            self.up_proj = nn.Parameter(rows.clone())
            self.down_proj = nn.Parameter(rows.clone())
            self.act_fn = nn.SiLU()

        def forward(self, hidden_states, top_k_index, top_k_weights):
            return hidden_states

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(1, 1)
            self.experts = PackedExperts()

    class Model(nn.Module):
        _no_split_modules: ClassVar = ["Block"]

        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([Block()])

    return Model()


def _ep_block_dp_half_expert(rank: int, ep: int, num_experts: int, dp: int) -> int:
    """Expert index this rank stores when each rank keeps one row."""
    block = num_experts // ep
    local_e = block // dp
    ep_index = rank % ep
    dp_index = rank // ep
    return ep_index * block + dp_index * local_e


def _dp_ep_packed_rows_worker(
    rank: int, world_size: int, port: int, result_queue: Any
) -> None:
    """Local packed-expert rows are the dp half of this rank's EP block."""
    try:
        _init_gloo(rank, world_size, port)
        from agilerl.distributed.fsdp_blocks import apply_fsdp2

        ep = 2
        num_experts = 4
        mesh = build_parallel_mesh(world_size=world_size, ep=ep, device_type="cpu")
        assert mesh is not None, "no mesh"
        dp = mesh.leftover_dp
        assert dp == world_size // ep
        local_e = num_experts // (ep * dp)
        model = _arange_packed_moe(num_experts)
        for param in model.parameters():
            dist.broadcast(param.data, src=0)
        applied = apply_expert_parallel(model, mesh.ep)
        assert applied == 1, f"applied={applied}"
        expert_mesh = None if dp == 1 else mesh.dp_mod_ep
        apply_fsdp2(
            model,
            FSDPConfig(ep=ep, wrap_every_n_blocks=1, param_persistence_threshold=0),
            mesh=mesh.hsdp,
            expert_mesh=expert_mesh,
        )
        expected_expert = (0, 2, 1, 3)[rank]
        assert expected_expert == _ep_block_dp_half_expert(rank, ep, num_experts, dp)
        found = False
        for _name, module in iter_packed_expert_modules(model):
            for param in module.parameters(recurse=False):
                found = True
                local = expert_local_tensor(param)
                assert local.shape[0] == local_e, tuple(local.shape)
                expected = torch.full(
                    tuple(local.shape), expected_expert, dtype=local.dtype
                )
                assert torch.equal(local.detach().cpu(), expected), (
                    rank,
                    tuple(local.shape),
                    local.detach().cpu(),
                    expected,
                )
        assert found, "packed experts missing after FSDP"
        experts = next(module for _name, module in iter_packed_expert_modules(model))

        class BaseHolder(nn.Module):
            def __init__(self, base: nn.Module) -> None:
                super().__init__()
                self.base_layer = base

            def get_base_layer(self) -> nn.Module:
                return self.base_layer

        with _gathered_expert_block(BaseHolder(experts)):
            for param in experts.parameters(recurse=False):
                gathered = expert_local_tensor(param)
                assert gathered.shape[0] == num_experts // ep, tuple(gathered.shape)
        for param in experts.parameters(recurse=False):
            restored = expert_local_tensor(param)
            assert restored.shape[0] == local_e, tuple(restored.shape)

        full = torch.arange(num_experts, dtype=torch.float32).view(num_experts, 1, 1)
        keys = [f"experts.{index}.weight" for index in range(num_experts)]
        block = num_experts // ep
        ep_index = rank % ep
        index_slices = (
            slice(ep_index * block, (ep_index + 1) * block),
            slice(0, 1),
            slice(0, 1),
        )
        dest = torch.empty(local_e, 1, 1)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.safetensors")
            save_file(
                {key: full[index].clone() for index, key in enumerate(keys)},
                path,
            )
            with patch("torch.distributed.get_rank", return_value=rank):
                _copy_indexed_weights(
                    dict.fromkeys(keys, path),
                    keys,
                    (num_experts, 1, 1),
                    index_slices,
                    dest,
                )
        assert torch.equal(dest, full[expected_expert : expected_expert + local_e])
        result_queue.put((rank, "ok", None))
    except Exception as exc:  # pragma: no cover
        result_queue.put((rank, "err", repr(exc)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@requires_gloo
class TestCopyIndexedWeightsDpEp:
    def test_dp2_ep2_local_rows_are_ep_block_dp_half(self):
        _spawn_ranks(_dp_ep_packed_rows_worker, world_size=4)


def _slice_writer_worker(
    rank: int, world_size: int, port: int, result_queue: Any
) -> None:
    try:
        _init_gloo(rank, world_size, port)
        ep = world_size
        mesh = build_parallel_mesh(world_size=world_size, ep=ep, device_type="cpu")
        assert mesh is not None
        num_experts, rows, cols = 4, 3, 2
        dest = nn.Parameter(
            distribute_tensor(torch.empty(num_experts, rows, cols), mesh.ep, [Shard(0)])
        )
        full = torch.arange(num_experts * rows * cols, dtype=torch.float32).reshape(
            num_experts, rows, cols
        )
        _write_ep_expert_slice(dest, full)
        local_e = num_experts // ep
        start = rank * local_e
        assert torch.allclose(dest.to_local().cpu(), full[start : start + local_e])
        assert torch.allclose(dest.full_tensor().cpu(), full)

        out_f, lora_rank = 5, 2
        dest_b = nn.Parameter(
            distribute_tensor(
                torch.empty(out_f, lora_rank, num_experts), mesh.ep, [Shard(0)]
            )
        )
        snap_b = torch.arange(
            out_f * lora_rank * num_experts, dtype=torch.float32
        ).reshape(out_f, lora_rank * num_experts)
        _write_ep_expert_slice(dest_b, snap_b)
        assert torch.allclose(
            dest_b.full_tensor().cpu(),
            snap_b.reshape(out_f, lora_rank, num_experts),
        )

        cols_sharded = nn.Parameter(
            distribute_tensor(torch.empty(4, 8), mesh.ep, [Shard(1)])
        )
        snap_cols = torch.arange(32, dtype=torch.float32).reshape(4, 8)
        _write_ep_expert_slice(cols_sharded, snap_cols)
        assert torch.allclose(cols_sharded.full_tensor().cpu(), snap_cols)

        with pytest.raises(ValueError, match="does not match destination shape"):
            _write_ep_expert_slice(dest, full[:1])

        result_queue.put((rank, "ok", None))
    except Exception as exc:  # pragma: no cover
        result_queue.put((rank, "err", repr(exc)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _slice_peak_worker(
    rank: int, world_size: int, port: int, result_queue: Any
) -> None:
    try:
        _init_nccl(rank, world_size, port)
        mesh = build_parallel_mesh(ep=world_size)
        assert mesh is not None
        num_experts, rows, cols = 32, 128, 128
        full = torch.arange(num_experts * rows * cols, dtype=torch.float32).reshape(
            num_experts, rows, cols
        )
        sliced = nn.Parameter(
            distribute_tensor(
                torch.empty(num_experts, rows, cols, device="cuda"),
                mesh.ep,
                [Shard(0)],
            )
        )
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        _write_ep_expert_slice(sliced, full)
        torch.cuda.synchronize()
        peak_sliced = torch.cuda.max_memory_allocated()
        reference = nn.Parameter(
            distribute_tensor(
                torch.empty(num_experts, rows, cols, device="cuda"),
                mesh.ep,
                [Shard(0)],
            )
        )
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        _write_full_tensor(reference, full)
        torch.cuda.synchronize()
        peak_full = torch.cuda.max_memory_allocated()
        assert peak_sliced < peak_full, (peak_sliced, peak_full)
        local_e = num_experts // world_size
        start = rank * local_e
        assert torch.allclose(sliced.to_local().cpu(), full[start : start + local_e])

        result_queue.put((rank, "ok", None))
    except Exception as exc:  # pragma: no cover
        result_queue.put((rank, "err", repr(exc)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@requires_gloo
class TestEpExpertSliceWriter:
    def test_chunk_equality_reshape_and_errors(self):
        _spawn_ranks(_slice_writer_worker)


@pytest.mark.gpu
@requires_2_cuda
class TestEpExpertSlicePeak:
    def test_sliced_write_allocates_less_than_full_scatter(self):
        _spawn_ranks(_slice_peak_worker)


class TestEpExpertLiveKeys:
    def test_resolves_packed_params_and_ignores_rest(self):
        model = _tiny_moe_actor(4)
        packed = list(iter_packed_expert_modules(model))
        assert packed, "expected packed experts"

        assert _ep_expert_live_keys(model, packed) == frozenset(
            {
                "layers.0.experts.up_proj",
                "layers.0.experts.down_proj",
            }
        )

    def test_resolves_through_checkpoint_wrappers(self):
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            checkpoint_wrapper,
        )

        model = _tiny_moe_actor(4)
        model.layers[0] = checkpoint_wrapper(model.layers[0])
        packed = list(iter_packed_expert_modules(model))
        assert packed, "expected packed experts"

        assert _ep_expert_live_keys(model, packed) == frozenset(
            {
                "layers.0._checkpoint_wrapped_module.experts.up_proj",
                "layers.0._checkpoint_wrapped_module.experts.down_proj",
            }
        )


def _lora_routed_model(adapters: tuple[str, ...]) -> nn.Module:
    """Upgraded PEFT routed experts with frozen base weights."""
    from peft import LoraConfig, inject_adapter_in_model

    from agilerl.lora.moe import upgrade_moe_param_wrappers

    torch.manual_seed(0)
    model: nn.Module = _LoraRoutedExperts()
    for name in adapters:
        model = inject_adapter_in_model(
            LoraConfig(
                r=2,
                lora_alpha=4,
                lora_dropout=0.0,
                target_modules=[],
                target_parameters=["experts.gate_up_proj", "experts.down_proj"],
                init_lora_weights=False,
            ),
            model,
            adapter_name=name,
        )
    assert upgrade_moe_param_wrappers(model) == 1
    for name, param in model.named_parameters():
        param.requires_grad_("lora" in name)
    return model


def _requires_grad_worker(
    rank: int, world_size: int, port: int, result_queue: Any
) -> None:
    try:
        _init_gloo(rank, world_size, port)
        mesh = build_parallel_mesh(
            world_size=world_size, ep=world_size, device_type="cpu"
        )
        assert mesh is not None
        experts = _RoutedExperts(num_experts=4, hidden=4, intermediate=8)
        experts.up_proj.requires_grad_(False)
        model = _lora_routed_model(("actor", "reference"))
        for name, param in model.named_parameters():
            if "reference" in name:
                param.requires_grad_(False)
        expected = {name: p.requires_grad for name, p in model.named_parameters()}

        assert apply_expert_parallel(experts, mesh.ep) == 1
        assert apply_expert_parallel(model, mesh.ep) == 1

        assert isinstance(experts.up_proj, DTensor)
        assert experts.up_proj.requires_grad is False
        assert experts.down_proj.requires_grad is True
        actual = {name: p.requires_grad for name, p in model.named_parameters()}
        assert actual == expected, (actual, expected)
        assert any(isinstance(p, DTensor) for p in model.parameters())
        assert not any(
            p.requires_grad
            for name, p in model.named_parameters()
            if "lora" not in name
        )
        result_queue.put((rank, "ok", None))
    except Exception as exc:  # pragma: no cover
        result_queue.put((rank, "err", repr(exc)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@requires_gloo
class TestApplyExpertParallelRequiresGrad:
    def test_frozen_stays_frozen_and_trainable_stays_trainable(self):
        _spawn_ranks(_requires_grad_worker)


def _local_param_dict_worker(
    rank: int, world_size: int, port: int, result_queue: Any
) -> None:
    try:
        _init_gloo(rank, world_size, port)
        mesh = build_parallel_mesh(
            world_size=world_size, ep=world_size, device_type="cpu"
        )
        assert mesh is not None
        module = _RoutedExperts(num_experts=4, hidden=4, intermediate=8)
        module.up_proj.requires_grad_(False)
        shard_experts_on_ep(module, mesh.ep)
        up_ptr = module.up_proj.to_local().data_ptr()
        down_ptr = module.down_proj.to_local().data_ptr()

        views = _local_param_dict(module)
        copies = _local_param_dict(module, gathered=True)

        assert views["up_proj"].data_ptr() == up_ptr
        assert not views["up_proj"].requires_grad
        assert copies["up_proj"].data_ptr() != up_ptr
        assert torch.equal(copies["up_proj"], views["up_proj"])
        for params in (views, copies):
            assert params["down_proj"].data_ptr() != down_ptr
            assert params["down_proj"].requires_grad
            assert params["down_proj"].is_leaf
        result_queue.put((rank, "ok", None))
    except Exception as exc:  # pragma: no cover
        result_queue.put((rank, "err", repr(exc)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@requires_gloo
class TestLocalParamDict:
    def test_frozen_views_unless_gathered_trainable_leaf_copies(self):
        _spawn_ranks(_local_param_dict_worker)


def _checkpoint_worker(
    rank: int, world_size: int, port: int, result_queue: Any
) -> None:
    """The expert kernel's saved input is dropped under non-reentrant checkpointing."""
    try:
        _init_gloo(rank, world_size, port)
        from agilerl.lora.moe import routed_experts_local_forward

        mesh = build_parallel_mesh(
            world_size=world_size, ep=world_size, device_type="cpu"
        )
        assert mesh is not None
        torch.manual_seed(0)
        base = _RoutedExperts(num_experts=4, hidden=8, intermediate=16)
        for param in base.parameters():
            dist.broadcast(param.data, src=0)
        kernel_inputs: list[weakref.ref] = []

        def recording_forward(self, hidden_states, top_k_index, top_k_weights):
            kernel_inputs.append(weakref.ref(hidden_states))
            return routed_experts_local_forward(
                self, hidden_states, top_k_index, top_k_weights
            )

        def build() -> nn.Module:
            module = copy.deepcopy(base)
            module.forward = MethodType(recording_forward, module)
            assert apply_expert_parallel(module, mesh.ep) == 1
            return module

        plain, checkpointed = build(), build()
        torch.manual_seed(rank + 3)
        hidden = torch.randn(6, 8)
        top_k_weights, top_k_index = torch.softmax(torch.randn(6, 4), -1).topk(2, -1)
        upstream = torch.randn(6, 8)

        plain_hidden = hidden.clone().requires_grad_(True)
        plain_out = plain(plain_hidden, top_k_index, top_k_weights)
        gc.collect()
        plain_alive = [ref() is not None for ref in kernel_inputs]
        (plain_out * upstream).sum().backward()

        kernel_inputs.clear()
        ckpt_hidden = hidden.clone().requires_grad_(True)
        ckpt_out = checkpoint(
            checkpointed, ckpt_hidden, top_k_index, top_k_weights, use_reentrant=False
        )
        gc.collect()
        ckpt_alive = [ref() is not None for ref in kernel_inputs]
        (ckpt_out * upstream).sum().backward()

        assert plain_alive == [True], plain_alive
        assert ckpt_alive == [False], ckpt_alive
        assert torch.allclose(ckpt_out, plain_out, rtol=0, atol=0)
        assert torch.allclose(ckpt_hidden.grad, plain_hidden.grad, rtol=0, atol=0)
        for (name, ref_param), param in zip(
            plain.named_parameters(), checkpointed.parameters(), strict=True
        ):
            assert torch.allclose(
                expert_local_tensor(param.grad),
                expert_local_tensor(ref_param.grad),
                rtol=0,
                atol=0,
            ), name
        result_queue.put((rank, "ok", None))
    except Exception as exc:  # pragma: no cover
        result_queue.put((rank, "err", repr(exc)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@requires_gloo
class TestRoutedExpertParallelCheckpointing:
    def test_saved_kernel_input_dropped_and_grads_match(self):
        _spawn_ranks(_checkpoint_worker)


def _mixed_routing_worker(
    rank: int, world_size: int, port: int, result_queue: Any, token_blocks: int = 1
) -> None:
    """Per-row actor/critic routing through EP matches the dense model."""
    try:
        _init_gloo(rank, world_size, port)
        from agilerl.lora.fused import (
            patch_lora_for_fused_forward,
            set_fused_adapter_routing,
        )

        mesh = build_parallel_mesh(
            world_size=world_size, ep=world_size, device_type="cpu"
        )
        assert mesh is not None
        model = _lora_routed_model(("actor", "critic"))
        reference = copy.deepcopy(model)
        torch.manual_seed(1)
        tokens = 6 * world_size
        global_hidden = torch.randn(tokens, 8)
        global_weights, global_index = torch.softmax(torch.randn(tokens, 4), -1).topk(
            2, -1
        )
        global_upstream = torch.randn(tokens, 8)
        rows = slice(rank * 6, (rank + 1) * 6)

        patch_lora_for_fused_forward(reference)
        set_fused_adapter_routing(reference, ["actor", "critic"] * (tokens // 2))
        ref_hidden = global_hidden.clone().requires_grad_(True)
        ref_out = reference(ref_hidden, global_index, global_weights)
        (ref_out * global_upstream).sum().backward()
        ref_grads = {
            name: param.grad
            for name, param in reference.named_parameters()
            if param.requires_grad
        }

        assert apply_expert_parallel(model, mesh.ep, token_blocks=token_blocks) == 1
        set_fused_adapter_routing(model, ["actor", "critic"] * 3)
        hidden = global_hidden[rows].clone().requires_grad_(True)
        out = model(hidden, global_index[rows], global_weights[rows])
        (out * global_upstream[rows]).sum().backward()

        assert torch.allclose(out, ref_out[rows], rtol=1e-5, atol=1e-6)
        assert torch.allclose(hidden.grad, ref_hidden.grad[rows], rtol=1e-5, atol=1e-6)
        checked = 0
        for name, param in model.named_parameters():
            if "lora_A" not in name:
                continue
            checked += 1
            local_ref = ref_grads[name].chunk(world_size, dim=0)[rank]
            assert torch.allclose(
                expert_local_tensor(param.grad), local_ref, rtol=1e-5, atol=1e-6
            ), name
        assert checked == 4, checked
        result_queue.put((rank, "ok", None))
    except Exception as exc:  # pragma: no cover
        result_queue.put((rank, "err", repr(exc)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@requires_gloo
class TestRoutedExpertParallelMixedRouting:
    def test_actor_critic_rows_match_dense_reference(self):
        _spawn_ranks(_mixed_routing_worker)

    def test_actor_critic_rows_in_token_blocks_match_dense_reference(self):
        _spawn_ranks(partial(_mixed_routing_worker, token_blocks=3))


def _routed_case(
    tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hidden states, top-2 routing over 4 experts, and an upstream grad."""
    torch.manual_seed(1)
    hidden = torch.randn(tokens, 8)
    weights, index = torch.softmax(torch.randn(tokens, 4), -1).topk(2, -1)
    return hidden, index, weights, torch.randn(tokens, 8)


def _run_routed(
    model: nn.Module,
    hidden: torch.Tensor,
    index: torch.Tensor,
    weights: torch.Tensor,
    upstream: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward, backward through ``upstream``; output, hidden grad, weight grad."""
    hidden = hidden.clone().requires_grad_(True)
    weights = weights.clone().requires_grad_(True)
    out = model(hidden, index, weights)
    (out * upstream).sum().backward()
    return out.detach(), hidden.grad, weights.grad


def _local_lora_grads(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: expert_local_tensor(param.grad)
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def _reference_local_grads(
    reference: nn.Module, rank: int, ep: int
) -> dict[str, torch.Tensor]:
    """Dense LoRA grads cut to this EP rank's experts (``B`` as ``[out, r, E]``)."""
    grads: dict[str, torch.Tensor] = {}
    for name, param in reference.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_B" in name:
            out_features = param.shape[0]
            grads[name] = param.grad.reshape(out_features, -1, 4).chunk(ep, dim=2)[rank]
        else:
            grads[name] = param.grad.chunk(ep, dim=0)[rank]
    return grads


def _token_blocks_worker(
    rank: int, world_size: int, port: int, result_queue: Any
) -> None:
    """Routed EP in 1 and 3 uneven token blocks matches a dense reference."""
    try:
        _init_gloo(rank, world_size, port)
        # Arrange
        mesh = build_parallel_mesh(
            world_size=world_size, ep=world_size, device_type="cpu"
        )
        assert mesh is not None
        base = _lora_routed_model(("actor",))
        reference = copy.deepcopy(base)
        # Rank 1's two tokens leave its first block empty.
        sizes = [7, 2]
        hidden, index, weights, upstream = _routed_case(sum(sizes))
        rows = slice(sum(sizes[:rank]), sum(sizes[: rank + 1]))
        ref_out, ref_hidden_grad, ref_weight_grad = _run_routed(
            reference, hidden, index, weights, upstream
        )
        ref_grads = _reference_local_grads(reference, rank, world_size)
        kernel = moe_wrappers.routed_experts_local_forward
        grouping: list[bool | None] = []

        def recording_kernel(*args: Any, **kwargs: Any) -> torch.Tensor:
            grouping.append(kwargs.get("already_grouped"))
            return kernel(*args, **kwargs)

        # Act
        results = {}
        with patch.object(
            moe_wrappers, "routed_experts_local_forward", recording_kernel
        ):
            for blocks in (1, 3):
                model = copy.deepcopy(base)
                assert apply_expert_parallel(model, mesh.ep, token_blocks=blocks) == 1
                outputs = _run_routed(
                    model, hidden[rows], index[rows], weights[rows], upstream[rows]
                )
                results[blocks] = (*outputs, _local_lora_grads(model))

        # Assert
        for blocks, (out, hidden_grad, weight_grad, grads) in results.items():
            assert torch.allclose(out, ref_out[rows], rtol=1e-5, atol=1e-6), blocks
            assert torch.allclose(
                hidden_grad, ref_hidden_grad[rows], rtol=1e-5, atol=1e-6
            ), blocks
            assert torch.allclose(
                weight_grad, ref_weight_grad[rows], rtol=1e-5, atol=1e-6
            ), blocks
            assert grads.keys() == ref_grads.keys()
            for name, grad in grads.items():
                assert torch.allclose(grad, ref_grads[name], rtol=1e-5, atol=1e-6), (
                    blocks,
                    name,
                )
        assert torch.allclose(results[1][0], results[3][0], rtol=0, atol=1e-6)
        assert grouping, "routed LoRA kernel never ran"
        assert all(flag is True for flag in grouping), grouping
        result_queue.put((rank, "ok", None))
    except Exception as exc:
        result_queue.put((rank, "err", repr(exc)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@requires_gloo
class TestRoutedExpertParallelTokenBlocks:
    def test_one_and_three_blocks_match_dense_reference(self):
        _spawn_ranks(_token_blocks_worker)


def _counts_replay_worker(
    rank: int, world_size: int, port: int, result_queue: Any, token_blocks: int
) -> None:
    """A checkpoint recompute replays the forward's routed EP counts with identical results."""
    try:
        _init_gloo(rank, world_size, port)
        # Arrange
        mesh = build_parallel_mesh(
            world_size=world_size, ep=world_size, device_type="cpu"
        )
        assert mesh is not None
        base = _lora_routed_model(("actor",))
        sizes = [7, 5]
        hidden, index, weights, upstream = _routed_case(sum(sizes))
        rows = slice(sum(sizes[:rank]), sum(sizes[: rank + 1]))
        exchange = ep_forward.exchange_expert_counts
        exchanges: list[torch.Tensor] = []

        def recording_exchange(*args: Any, **kwargs: Any) -> torch.Tensor:
            exchanges.append(args[0])
            return exchange(*args, **kwargs)

        def run(context_fn: Any) -> tuple[Any, ...]:
            model = copy.deepcopy(base)
            assert apply_expert_parallel(model, mesh.ep, token_blocks=token_blocks) == 1
            hidden_in = hidden[rows].clone().requires_grad_(True)
            weights_in = weights[rows].clone().requires_grad_(True)
            exchanges.clear()
            with patch.object(ep_forward, "exchange_expert_counts", recording_exchange):
                out = checkpoint(
                    model,
                    hidden_in,
                    index[rows],
                    weights_in,
                    use_reentrant=False,
                    context_fn=context_fn,
                )
                forward_exchanges = len(exchanges)
                (out * upstream[rows]).sum().backward()
            return (
                out.detach(),
                hidden_in.grad,
                weights_in.grad,
                _local_lora_grads(model),
                forward_exchanges,
                len(exchanges),
            )

        # Act
        plain = run(noop_context_fn)
        replayed = run(routed_counts_contexts)

        # Assert
        for expected, actual in zip(plain[:3], replayed[:3], strict=True):
            assert torch.equal(expected, actual)
        assert plain[3].keys() == replayed[3].keys()
        for name, grad in plain[3].items():
            assert torch.equal(grad, replayed[3][name]), name
        assert plain[4:] == (1, 2), plain[4:]
        assert replayed[4:] == (1, 1), replayed[4:]
        result_queue.put((rank, "ok", None))
    except Exception as exc:
        result_queue.put((rank, "err", repr(exc)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _counts_replay_mismatch_worker(
    rank: int, world_size: int, port: int, result_queue: Any
) -> None:
    """A recompute whose routing differs from its forward fails before any token moves."""
    try:
        _init_gloo(rank, world_size, port)
        # Arrange
        mesh = build_parallel_mesh(
            world_size=world_size, ep=world_size, device_type="cpu"
        )
        assert mesh is not None
        model = _lora_routed_model(("actor",))
        assert apply_expert_parallel(model, mesh.ep) == 1
        hidden, index, weights, upstream = _routed_case(6)
        recompute_index = torch.zeros_like(index)
        recompute_index[:, 1] = 1
        assert not torch.equal(
            torch.bincount(index.reshape(-1), minlength=4),
            torch.bincount(recompute_index.reshape(-1), minlength=4),
        )
        routings = iter((index, recompute_index))

        def block(hidden_in: torch.Tensor, weights_in: torch.Tensor) -> torch.Tensor:
            return model(hidden_in, next(routings), weights_in)

        out = checkpoint(
            block,
            hidden.clone().requires_grad_(True),
            weights.clone().requires_grad_(True),
            use_reentrant=False,
            context_fn=routed_counts_contexts,
        )

        # Act / Assert
        with pytest.raises(RuntimeError, match="Recompute routed tokens differently"):
            (out * upstream).sum().backward()
        result_queue.put((rank, "ok", None))
    except Exception as exc:
        result_queue.put((rank, "err", repr(exc)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@requires_gloo
class TestRoutedCountsContexts:
    def test_recompute_reuses_forward_counts_with_identical_results(self):
        _spawn_ranks(partial(_counts_replay_worker, token_blocks=1))

    def test_recompute_reuses_forward_counts_in_token_blocks(self):
        _spawn_ranks(partial(_counts_replay_worker, token_blocks=3))

    def test_recompute_with_different_routing_raises(self):
        _spawn_ranks(_counts_replay_mismatch_worker)


def _row_chunks_worker(
    rank: int, world_size: int, port: int, result_queue: Any
) -> None:
    """Routed EP in one-row expert and combine chunks, with and without recompute, matches a dense reference."""
    try:
        _init_gloo(rank, world_size, port)
        # Arrange
        mesh = build_parallel_mesh(
            world_size=world_size, ep=world_size, device_type="cpu"
        )
        assert mesh is not None
        base = _lora_routed_model(("actor",))
        reference = copy.deepcopy(base)
        sizes = [6, 5]
        hidden, index, weights, upstream = _routed_case(sum(sizes))
        rows = slice(sum(sizes[:rank]), sum(sizes[: rank + 1]))
        ref_out, ref_hidden_grad, ref_weight_grad = _run_routed(
            reference, hidden, index, weights, upstream
        )
        ref_grads = _reference_local_grads(reference, rank, world_size)
        kernel = moe_wrappers.routed_experts_local_forward
        calls: list[tuple[bool, int]] = []

        def recording_kernel(*args: Any, **kwargs: Any) -> torch.Tensor:
            calls.append((kwargs["recompute"], kwargs["chunk_bytes"]))
            return kernel(*args, **kwargs)

        # Act
        results = {}
        with patch.object(
            moe_wrappers, "routed_experts_local_forward", recording_kernel
        ):
            for recompute in (False, True):
                model = copy.deepcopy(base)
                assert apply_expert_parallel(model, mesh.ep) == 1
                set_routed_experts_recompute(model, recompute)
                set_routed_experts_chunk_bytes(model, 1)
                outputs = _run_routed(
                    model, hidden[rows], index[rows], weights[rows], upstream[rows]
                )
                results[recompute] = (*outputs, _local_lora_grads(model))

        # Assert
        # fp32; chunking only reorders the per-expert sums.
        tolerance = {"rtol": 1e-5, "atol": 1e-6}
        for recompute, (out, hidden_grad, weight_grad, grads) in results.items():
            assert torch.allclose(out, ref_out[rows], **tolerance), recompute
            assert torch.allclose(hidden_grad, ref_hidden_grad[rows], **tolerance), (
                recompute
            )
            assert torch.allclose(weight_grad, ref_weight_grad[rows], **tolerance), (
                recompute
            )
            assert grads.keys() == ref_grads.keys()
            for name, grad in grads.items():
                assert torch.allclose(grad, ref_grads[name], **tolerance), (
                    recompute,
                    name,
                )
        assert set(calls) == {(False, 1), (True, 1)}
        result_queue.put((rank, "ok", None))
    except Exception as exc:
        result_queue.put((rank, "err", repr(exc)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@requires_gloo
class TestRoutedExpertParallelRowChunks:
    def test_one_row_chunks_match_dense_reference_with_and_without_recompute(self):
        _spawn_ranks(_row_chunks_worker)


def _replicated_tokens_worker(
    rank: int, world_size: int, port: int, result_queue: Any
) -> None:
    """TP ranks that hold the same tokens dispatch each once and match dense."""
    try:
        _init_gloo(rank, world_size, port)
        # Arrange
        tp = 2
        top_k = 2
        mesh = build_parallel_mesh(
            world_size=world_size, ep=world_size, tp=tp, device_type="cpu"
        )
        assert mesh is not None
        base = _lora_routed_model(("actor",))
        reference = copy.deepcopy(base)
        tokens = 9
        hidden, index, weights, upstream = _routed_case(tokens)
        ref_out, ref_hidden_grad, ref_weight_grad = _run_routed(
            reference, hidden, index, weights, upstream
        )
        # Expert grads count every TP rank's copy of a token.
        expected_grads = {
            name: tp * grad
            for name, grad in _reference_local_grads(
                reference, rank, world_size
            ).items()
        }
        all_to_all = dist.all_to_all_single
        rows_sent: list[int] = []

        def recording_all_to_all(
            output: torch.Tensor, input: torch.Tensor, *args: Any, **kwargs: Any
        ) -> Any:
            if input.is_floating_point():
                rows_sent.append(int(input.shape[0]))
            return all_to_all(output, input, *args, **kwargs)

        # Act
        results = {}
        for label, tp_mesh in (("duplicated", None), ("replicated", mesh.tp)):
            model = copy.deepcopy(base)
            assert (
                apply_expert_parallel(model, mesh.ep, tp_mesh=tp_mesh, token_blocks=3)
                == 1
            )
            rows_sent.clear()
            with patch.object(dist, "all_to_all_single", recording_all_to_all):
                outputs = _run_routed(model, hidden, index, weights, upstream)
            sent = torch.tensor(sum(rows_sent))
            dist.all_reduce(sent)
            results[label] = (*outputs, _local_lora_grads(model), int(sent))

        # Assert
        for label, (out, hidden_grad, weight_grad, grads, _) in results.items():
            assert torch.allclose(out, ref_out, rtol=1e-5, atol=1e-6), label
            assert torch.allclose(hidden_grad, ref_hidden_grad, rtol=1e-5, atol=1e-6), (
                label
            )
            assert torch.allclose(weight_grad, ref_weight_grad, rtol=1e-5, atol=1e-6), (
                label
            )
            for name, grad in grads.items():
                assert torch.allclose(
                    grad, expected_grads[name], rtol=1e-5, atol=1e-6
                ), (label, name)
        # Forward and backward each send every routed row out and back.
        sent = {label: result[4] for label, result in results.items()}
        assert sent == {
            "replicated": 4 * tokens * top_k,
            "duplicated": 4 * tp * tokens * top_k,
        }, sent
        result_queue.put((rank, "ok", None))
    except Exception as exc:
        result_queue.put((rank, "err", repr(exc)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@requires_gloo
class TestRoutedExpertParallelReplicatedTokens:
    def test_tp_ranks_dispatch_each_token_once_and_match_dense(self):
        _spawn_ranks(_replicated_tokens_worker)


class _LoopedSortedExperts(nn.Module):
    """Sorted experts with a per-expert matmul loop, so the kernel runs on CPU."""

    def __init__(self, num_experts: int, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_experts, out_features, in_features))

    def forward(self, inputs, expert_size):
        pieces = inputs.split([int(rows) for rows in expert_size])
        return torch.cat(
            [
                piece @ weight.T
                for piece, weight in zip(pieces, self.weight, strict=True)
            ]
        )


def _sorted_replicated_rows_worker(
    rank: int, world_size: int, port: int, result_queue: Any
) -> None:
    """TP ranks holding the same sorted rows dispatch each once and match dense."""
    try:
        _init_gloo(rank, world_size, port)
        # Arrange
        tp = 2
        mesh = build_parallel_mesh(
            world_size=world_size, ep=world_size, tp=tp, device_type="cpu"
        )
        assert mesh is not None
        torch.manual_seed(0)
        base = _LoopedSortedExperts(num_experts=4, in_features=8, out_features=6)
        reference = copy.deepcopy(base)
        # Rank 0's span ends inside expert 2's rows.
        counts = torch.tensor([3, 0, 4, 2])
        rows = int(counts.sum())
        torch.manual_seed(1)
        inputs = torch.randn(rows, 8)
        upstream = torch.randn(rows, 6)
        ref_inputs = inputs.clone().requires_grad_(True)
        ref_out = reference(ref_inputs, counts)
        (ref_out * upstream).sum().backward()
        # Expert grads count every TP rank's copy of a row.
        expected_grad = tp * reference.weight.grad.chunk(world_size, dim=0)[rank]
        all_to_all = dist.all_to_all_single
        rows_sent: list[int] = []

        def recording_all_to_all(
            output: torch.Tensor, input: torch.Tensor, *args: Any, **kwargs: Any
        ) -> Any:
            if input.is_floating_point():
                rows_sent.append(int(input.shape[0]))
            return all_to_all(output, input, *args, **kwargs)

        # Act
        results = {}
        for label, tp_mesh in (("duplicated", None), ("replicated", mesh.tp)):
            module = copy.deepcopy(base)
            assert apply_expert_parallel(module, mesh.ep, tp_mesh=tp_mesh) == 1
            module_inputs = inputs.clone().requires_grad_(True)
            rows_sent.clear()
            with patch.object(dist, "all_to_all_single", recording_all_to_all):
                out = module(module_inputs, counts)
                (out * upstream).sum().backward()
            sent = torch.tensor(sum(rows_sent))
            dist.all_reduce(sent)
            results[label] = (
                out.detach(),
                module_inputs.grad,
                expert_local_tensor(module.weight.grad),
                int(sent),
            )

        # Assert
        for label, (out, input_grad, weight_grad, _) in results.items():
            assert torch.allclose(out, ref_out.detach(), rtol=1e-5, atol=1e-6), label
            assert torch.allclose(input_grad, ref_inputs.grad, rtol=1e-5, atol=1e-6), (
                label
            )
            assert torch.allclose(weight_grad, expected_grad, rtol=1e-5, atol=1e-6), (
                label
            )
        # Forward and backward each send every row out and back.
        sent = {label: result[3] for label, result in results.items()}
        assert sent == {"replicated": 4 * rows, "duplicated": 4 * tp * rows}, sent
        result_queue.put((rank, "ok", None))
    except Exception as exc:
        result_queue.put((rank, "err", repr(exc)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@requires_gloo
class TestSortedExpertParallelReplicatedRows:
    def test_tp_ranks_dispatch_each_row_once_and_match_dense(self):
        _spawn_ranks(_sorted_replicated_rows_worker)


class _ForwardLoraBlock(nn.Module):
    def __init__(self, moe: nn.Module):
        super().__init__()
        self.lin = nn.Linear(8, 8)
        self.moe = moe

    def forward(self, hidden_states, top_k_index, top_k_weights):
        return self.moe(self.lin(hidden_states), top_k_index, top_k_weights)


class _ForwardLoraMoeModel(nn.Module):
    _no_split_modules: ClassVar = ["_ForwardLoraBlock"]

    def __init__(self, moe: nn.Module):
        super().__init__()
        self.layers = nn.ModuleList([_ForwardLoraBlock(moe)])

    def forward(self, hidden_states, top_k_index, top_k_weights):
        return self.layers[0](hidden_states, top_k_index, top_k_weights)


def _fsdp_gathered_frozen_worker(
    rank: int, world_size: int, port: int, result_queue: Any
) -> None:
    """FSDP-gathered frozen experts survive reshard into backward under checkpointing."""
    try:
        _init_gloo(rank, world_size, port)
        from agilerl.distributed.fsdp_blocks import apply_fsdp2

        ep = 2
        mesh = build_parallel_mesh(world_size=world_size, ep=ep, device_type="cpu")
        assert mesh is not None
        assert mesh.leftover_dp == 2
        model = _ForwardLoraMoeModel(_lora_routed_model(("actor",)))
        for param in model.parameters():
            dist.broadcast(param.data, src=0)
        reference = copy.deepcopy(model)
        torch.manual_seed(rank + 3)
        hidden = torch.randn(6, 8)
        top_k_weights, top_k_index = torch.softmax(torch.randn(6, 4), -1).topk(2, -1)
        upstream = torch.randn(6, 8)
        ref_hidden = hidden.clone().requires_grad_(True)
        (reference(ref_hidden, top_k_index, top_k_weights) * upstream).sum().backward()

        assert apply_expert_parallel(model, mesh.ep) == 1
        apply_fsdp2(
            model,
            FSDPConfig(
                ep=ep,
                wrap_every_n_blocks=1,
                param_persistence_threshold=0,
                param_dtype="float32",
                reduce_dtype="float32",
            ),
            mesh=mesh.hsdp,
            expert_mesh=mesh.dp_mod_ep,
            gradient_checkpointing=True,
        )
        experts = next(
            module for module in model.modules() if "gate_up_proj" in module._parameters
        )
        ep_hidden = hidden.clone().requires_grad_(True)
        out = model(ep_hidden, top_k_index, top_k_weights)
        (out * upstream).sum().backward()

        assert callable(getattr(experts, "unshard", None)), type(experts).__name__
        assert not any(param.requires_grad for param in experts.parameters())
        assert torch.allclose(ep_hidden.grad, ref_hidden.grad, rtol=1e-5, atol=1e-7)
        result_queue.put((rank, "ok", None))
    except Exception as exc:  # pragma: no cover
        result_queue.put((rank, "err", repr(exc)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@requires_gloo
class TestFsdpGatheredFrozenExperts:
    def test_checkpointed_backward_matches_dense_reference(self):
        _spawn_ranks(_fsdp_gathered_frozen_worker, world_size=4)


class _HsdpMoeBlock(nn.Module):
    """Dense linear, NemotronH MLP, then PEFT-wrapped routed experts."""

    def __init__(self, moe: nn.Module):
        super().__init__()
        self.lin = nn.Linear(8, 8)
        mlp_config = NemotronHConfig(
            hidden_size=8,
            intermediate_size=8,
            vocab_size=32,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=4,
            layers_block_type=["mlp"],
            mlp_bias=True,
        )
        self.mlp = NemotronHMLP(mlp_config, intermediate_size=8)
        self.moe = moe

    def forward(self, hidden_states, top_k_index, top_k_weights):
        hidden = self.lin(hidden_states)
        hidden = hidden + self.mlp(hidden)
        return self.moe(hidden, top_k_index, top_k_weights)


class _HsdpMoeModel(nn.Module):
    _no_split_modules: ClassVar = ["_HsdpMoeBlock"]

    def __init__(self, moe: nn.Module):
        super().__init__()
        self.layers = nn.ModuleList([_HsdpMoeBlock(moe)])

    def forward(self, hidden_states, top_k_index, top_k_weights):
        return self.layers[0](hidden_states, top_k_index, top_k_weights)


def _full_value(param: torch.Tensor) -> torch.Tensor:
    return param.full_tensor() if isinstance(param, DTensor) else param


def _spans_shard_groups(mesh: Any, dim: int, shard_group_size: int) -> bool:
    ranks = dist.get_process_group_ranks(mesh.get_group(dim))
    return len({member // shard_group_size for member in ranks}) > 1


def _replica_group(param: torch.Tensor, mesh: ParallelMesh) -> Any:
    """Ranks that hold the same local values of ``param``."""
    if not isinstance(param, DTensor):
        return dist.group.WORLD
    if param.device_mesh is mesh.ep:
        return mesh.ep_replicas.get_group()
    if param.device_mesh is mesh.tp:
        return mesh.tp_replicas.get_group()
    return mesh.world["replicate"].get_group()


def _hsdp_ep_step_worker(
    rank: int,
    world_size: int,
    port: int,
    result_queue: Any,
    tp: int,
    shard_group_size: int = 2,
    train_base_experts: bool = False,
) -> None:
    """One FSDPRuntime step on HSDP + EP (+ TP) matches a dense SGD step."""
    try:
        _init_gloo(rank, world_size, port)
        # Arrange
        ep, rows_per_shard = 2, 6
        model = _HsdpMoeModel(_lora_routed_model(("actor",)))
        model.layers[0].lin.weight.requires_grad_(False)
        base_experts = [
            name
            for name, param in model.named_parameters()
            if "experts" in name and "lora" not in name and param.dim() == 3
        ]
        assert base_experts
        if train_base_experts:
            for name in base_experts:
                model.get_parameter(name).requires_grad_(True)
        for param in model.parameters():
            dist.broadcast(param.data, src=0)
        reference = copy.deepcopy(model)
        initial = {
            name: param.detach().clone() for name, param in model.named_parameters()
        }
        batch_shards = world_size // tp
        torch.manual_seed(1)
        tokens = batch_shards * rows_per_shard
        hidden = torch.randn(tokens, 8)
        weights, index = torch.softmax(torch.randn(tokens, 4), -1).topk(2, -1)
        upstream = torch.randn(tokens, 8)
        rows = slice((rank // tp) * rows_per_shard, (rank // tp + 1) * rows_per_shard)

        ref_out = reference(hidden, index, weights)
        ref_loss = (ref_out * upstream).sum() / tokens
        ref_loss.backward()
        expected = {
            name: param.detach() - param.grad
            for name, param in reference.named_parameters()
            if param.requires_grad
        }

        config = FSDPConfig(
            ep=ep,
            tp=tp,
            shard_group_size=shard_group_size,
            wrap_every_n_blocks=1,
            param_persistence_threshold=0,
            param_dtype="float32",
            reduce_dtype="float32",
            routed_expert_chunk_mib=64,
        )
        mesh = build_parallel_mesh(
            world_size=world_size,
            ep=ep,
            tp=tp,
            shard_group_size=shard_group_size,
            device_type="cpu",
        )
        assert mesh is not None
        materialize_fsdp2_from_cpu_state(model, "cpu", config, parallel_mesh=mesh)
        runtime = FSDPRuntime(config)
        runtime.parallel_mesh = mesh
        trainable = [param for param in model.parameters() if param.requires_grad]
        by_mesh: dict[Any, list[nn.Parameter]] = {}
        for param in trainable:
            key = param.device_mesh if isinstance(param, DTensor) else None
            by_mesh.setdefault(key, []).append(param)
        sgd = torch.optim.SGD([{"params": group} for group in by_mesh.values()], lr=1.0)
        optimizer = SimpleNamespace(
            _single_optimizer=lambda: sgd, step=sgd.step, zero_grad=sgd.zero_grad
        )

        # Act
        out = model(hidden[rows], index[rows], weights[rows])
        loss = (out * upstream[rows]).sum() / rows_per_shard
        step = runtime.backward(
            loss,
            optimizer,
            gradient_accumulation_steps=1,
            actor=model,
            max_grad_norm=1e6,
        )

        # Assert
        assert step is not None
        assert torch.allclose(out, ref_out[rows], rtol=1e-5, atol=1e-6)
        shard_losses = loss.detach().clone()
        dist.all_reduce(shard_losses)
        assert torch.allclose(shard_losses / world_size, ref_loss, rtol=1e-5)
        assert mesh.leftover_dp == shard_group_size // ep
        checked = 0
        for name, param in model.named_parameters():
            value = _full_value(param.detach())
            if not param.requires_grad:
                assert param.grad is None, name
                assert torch.equal(value, initial[name].reshape(value.shape)), name
                if isinstance(param, DTensor):
                    for dim, placement in enumerate(param.placements):
                        if _spans_shard_groups(
                            param.device_mesh, dim, shard_group_size
                        ):
                            assert isinstance(placement, Replicate), (name, dim)
                continue
            checked += 1
            target = expected[name].reshape(value.shape)
            assert torch.allclose(value, target, rtol=1e-5, atol=1e-6), name
            local = param.to_local() if isinstance(param, DTensor) else param
            group = _replica_group(param, mesh)
            gathered = [
                torch.empty_like(local) for _ in range(dist.get_world_size(group))
            ]
            dist.all_gather(gathered, local.detach().contiguous(), group=group)
            assert all(torch.equal(copy_, gathered[0]) for copy_ in gathered), name
        assert checked == len(expected), (checked, sorted(expected))
        experts = next(
            module for module in model.modules() if "gate_up_proj" in module._parameters
        )
        assert isinstance(experts.gate_up_proj, DTensor)
        assert "ep" in experts.gate_up_proj.device_mesh.mesh_dim_names
        if tp > 1:
            up = model.layers[0].mlp.up_proj.weight
            assert isinstance(up, DTensor)
            assert up.device_mesh is mesh.tp
        result_queue.put((rank, "ok", None))
    except Exception as exc:  # pragma: no cover
        result_queue.put((rank, "err", repr(exc)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _hsdp_ep2_tp1_worker(
    rank: int, world_size: int, port: int, result_queue: Any
) -> None:
    _hsdp_ep_step_worker(rank, world_size, port, result_queue, tp=1)


def _hsdp_ep2_tp2_worker(
    rank: int, world_size: int, port: int, result_queue: Any
) -> None:
    _hsdp_ep_step_worker(rank, world_size, port, result_queue, tp=2)


def _fsdp_experts_ep2_worker(
    rank: int, world_size: int, port: int, result_queue: Any
) -> None:
    _hsdp_ep_step_worker(
        rank,
        world_size,
        port,
        result_queue,
        tp=1,
        shard_group_size=4,
        train_base_experts=True,
    )


@requires_gloo
class TestFSDPRuntimeBackwardHsdpExpertParallel:
    def test_replicated_shard_groups_match_dense_step(self):
        _spawn_ranks(_hsdp_ep2_tp1_worker, world_size=4)

    def test_tensor_parallel_pairs_share_tokens_and_match_dense_step(self):
        _spawn_ranks(_hsdp_ep2_tp2_worker, world_size=4)

    def test_fsdp_sharded_base_experts_and_lora_match_dense_step(self):
        _spawn_ranks(_fsdp_experts_ep2_worker, world_size=4)


def _ep_adamw_step(model: nn.Module, fused: bool) -> list[nn.Parameter]:
    """Run one FSDPRuntime AdamW step on ``model``; return its trainable params."""
    config = FSDPConfig(
        ep=2,
        wrap_every_n_blocks=1,
        param_persistence_threshold=0,
        param_dtype="float32",
        reduce_dtype="float32",
        routed_expert_chunk_mib=64,
    )
    mesh = build_parallel_mesh(
        world_size=dist.get_world_size(), ep=2, device_type="cpu"
    )
    materialize_fsdp2_from_cpu_state(model, "cpu", config, parallel_mesh=mesh)
    runtime = FSDPRuntime(config)
    runtime.parallel_mesh = mesh
    optimizer = make_llm_optimizer(model, lr=0.1, lr_critic=None, fused=fused)
    torch.manual_seed(dist.get_rank() + 1)
    hidden = torch.randn(6, 8)
    weights, index = torch.softmax(torch.randn(6, 4), -1).topk(2, -1)
    upstream = torch.randn(6, 8)
    loss = (model(hidden, index, weights) * upstream).sum()
    runtime.backward(
        loss, optimizer, gradient_accumulation_steps=1, actor=model, max_grad_norm=1e6
    )
    return [param for param in model.parameters() if param.requires_grad]


def _ep_fused_adamw_worker(
    rank: int, world_size: int, port: int, result_queue: Any
) -> None:
    try:
        _init_gloo(rank, world_size, port)
        # Arrange
        model = _HsdpMoeModel(_lora_routed_model(("actor",)))
        for param in model.parameters():
            dist.broadcast(param.data, src=0)
        reference = copy.deepcopy(model)

        # Act
        fused = _ep_adamw_step(model, fused=True)
        unfused = _ep_adamw_step(reference, fused=False)

        # Assert
        assert any(isinstance(param, DTensor) for param in fused)
        for param, expected in zip(fused, unfused, strict=True):
            assert torch.allclose(
                _full_value(param.detach()), _full_value(expected.detach()), atol=1e-6
            )
        result_queue.put((rank, "ok", None))
    except Exception as exc:  # pragma: no cover
        result_queue.put((rank, "err", repr(exc)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@requires_gloo
class TestFSDPRuntimeBackwardExpertParallelFusedAdamW:
    def test_fused_step_on_ep_lora_matches_unfused_step(self):
        _spawn_ranks(_ep_fused_adamw_worker)


def _sync_grads_unknown_mesh_worker(
    rank: int, world_size: int, port: int, result_queue: Any
) -> None:
    try:
        _init_gloo(rank, world_size, port)
        # Arrange
        mesh = build_parallel_mesh(world_size=world_size, ep=2, device_type="cpu")
        expert = nn.Parameter(distribute_tensor(torch.zeros(2, 3), mesh.ep, [Shard(0)]))
        expert.grad = distribute_tensor(torch.full((2, 3), 4.0), mesh.ep, [Shard(0)])
        other = init_device_mesh("cpu", (world_size,), mesh_dim_names=("other",))
        stray = nn.Parameter(distribute_tensor(torch.zeros(2, 3), other, [Shard(0)]))
        stray.grad = distribute_tensor(torch.ones(2, 3), other, [Shard(0)])

        # Act
        mesh.sync_grads([expert], torch.float32)
        try:
            mesh.sync_grads([expert, stray], torch.float32)
        except RuntimeError as exc:
            error = str(exc)
        else:
            error = ""
        stray.requires_grad_(False)
        mesh.sync_grads([stray], torch.float32)

        # Assert
        assert torch.equal(expert.grad.to_local(), torch.full((1, 3), 2.0))
        assert "('other',), which is not an FSDP, EP or TP mesh" in error
        result_queue.put((rank, "ok", None))
    except Exception as exc:  # pragma: no cover
        result_queue.put((rank, "err", repr(exc)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@requires_gloo
class TestParallelMeshSyncGrads:
    def test_trainable_dtensor_on_unknown_mesh_raises(self):
        _spawn_ranks(_sync_grads_unknown_mesh_worker)


def _shard_group_spans_world_worker(
    rank: int, world_size: int, port: int, result_queue: Any
) -> None:
    try:
        _init_gloo(rank, world_size, port)
        model = _tiny_moe_actor(4)
        for param in model.parameters():
            dist.broadcast(param.data, src=0)
        original = model.layers[0].lin.weight.detach().clone()
        config = FSDPConfig(
            shard_group_size=world_size,
            wrap_every_n_blocks=1,
            param_persistence_threshold=0,
            routed_expert_chunk_mib=64,
        )

        # fully_shard's default mesh follows the host accelerator (MPS on macOS).
        with patch("torch._C._get_accelerator", return_value=torch.device("cpu")):
            mesh = _materialize(model, config)

        weight = model.layers[0].lin.weight
        assert mesh is None
        assert isinstance(weight, DTensor), type(weight)
        assert weight.device_mesh.size() == world_size, weight.device_mesh
        assert torch.equal(weight.full_tensor(), original)
        result_queue.put((rank, "ok", None))
    except Exception as exc:  # pragma: no cover
        result_queue.put((rank, "err", repr(exc)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@requires_gloo
class TestMaterializeShardGroupSpanningWorld:
    def test_plain_fsdp_over_world(self):
        _spawn_ranks(_shard_group_spans_world_worker)


class TestTpDataParallelSize:
    def test_rejects_tp_below_one(self):
        with pytest.raises(ValueError, match="tp must be >= 1"):
            tp_data_parallel_size(4, 0)


class TestFoldWorld:
    def test_rejects_non_positive_world_size(self):
        with pytest.raises(ValueError, match="world_size must be >= 1"):
            expert_parallel.ep_data_parallel_size(0, 1)


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
        fake_world = MagicMock()
        fake_world.__getitem__.return_value = MagicMock()
        inner, replicas = MagicMock(), MagicMock()
        with (
            patch.object(dist, "is_available", return_value=True),
            patch.object(dist, "is_initialized", return_value=True),
            patch(
                "agilerl.distributed.ep_mesh.init_device_mesh",
                return_value=fake_world,
            ) as init_mesh,
            patch(
                "agilerl.distributed.ep_mesh._split_shard_axis",
                return_value=(inner, None, replicas),
            ),
        ):
            mesh = build_parallel_mesh(world_size=2, ep=2)

        init_mesh.assert_called_once_with(
            "cpu", (1, 2), mesh_dim_names=("replicate", "shard")
        )
        assert mesh is not None
        assert mesh.ep is inner


class TestSplitShardAxis:
    def test_flattens_replicas_when_replicate_is_above_one(self):
        world = MagicMock()
        world.size.return_value = 2
        inner = object()
        outer_mesh = MagicMock()
        replicas = object()
        outer_mesh._flatten.return_value = replicas
        split = MagicMock()
        split.__getitem__.side_effect = lambda key: {
            "ep": inner,
            ("replicate", "dp"): outer_mesh,
        }[key]
        world._unflatten.return_value = split

        got_inner, got_outer, got_replicas = _split_shard_axis(
            world, 2, 2, ("dp", "ep")
        )

        world._unflatten.assert_called_once_with(1, (2, 2), ("dp", "ep"))
        outer_mesh._flatten.assert_called_once_with("ep_replicas")
        assert got_inner is inner
        assert got_outer is outer_mesh
        assert got_replicas is replicas


class TestShardExpertsOnEp:
    def test_none_mesh_returns_the_module(self):
        module = nn.Linear(2, 2)

        assert shard_experts_on_ep(module, None) is module


class TestTokenDispatchGuards:
    def test_rejects_count_length_mismatch(self):
        with pytest.raises(ValueError, match="num_tokens_per_expert length"):
            token_dispatch(
                torch.zeros(2, 4),
                torch.zeros(3),
                ep_group=object(),
                ep_degree=2,
                num_local_experts=2,
            )


class TestTokenAdapterIds:
    def test_rejects_routing_that_does_not_tile_tokens(self):
        with pytest.raises(ValueError, match="Fused adapter routing covers"):
            _token_adapter_ids(
                ["actor", "critic"], n_tokens=3, device=torch.device("cpu")
            )


class TestPackedExpertLookup:
    def test_num_packed_experts_rejects_a_module_without_3d_weights(self):
        with pytest.raises(ValueError, match="no stacked 3D expert weight"):
            num_packed_experts(nn.Linear(4, 4))

    def test_packed_expert_count_is_none_on_a_dense_model(self):
        assert packed_expert_count(nn.Linear(4, 4)) is None

    def test_packed_expert_count_reads_the_first_stack(self):
        assert packed_expert_count(_tiny_moe_actor(4)) == 4


class TestAssertPackedExpertsEpShardedGuards:
    def test_ep_one_returns(self):
        assert_packed_experts_ep_sharded(nn.Linear(2, 2), ep=1)

    def test_skips_non_3d_parameters(self):
        class Packed(nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = nn.Parameter(torch.ones(4))

        module = Packed()
        assert_packed_experts_ep_sharded(module, ep=2, modules=[module])


class TestShardLoraLinearOnEp:
    def test_returns_when_weight_is_missing(self):
        _shard_lora_linear_on_ep(nn.Module(), MagicMock(), expert_dim=0)

    def test_lora_b_rejects_a_non_2d_weight(self):
        linear = nn.Module()
        linear.weight = nn.Parameter(torch.ones(2, 2, 2))
        mesh = MagicMock()
        mesh.size.return_value = 2

        with pytest.raises(ValueError, match="Expected 2D LoRA B weight"):
            _shard_lora_linear_on_ep(linear, mesh, expert_dim=1)

    def test_lora_b_requires_rank_and_expert_count(self):
        linear = nn.Module()
        linear.weight = nn.Parameter(torch.ones(4, 8))
        mesh = MagicMock()
        mesh.size.return_value = 2

        with pytest.raises(ValueError, match="requires lora_rank and num_experts"):
            _shard_lora_linear_on_ep(linear, mesh, expert_dim=1)

    def test_lora_b_rejects_shape_that_is_not_out_by_er(self):
        linear = nn.Module()
        linear.weight = nn.Parameter(torch.ones(4, 7))
        mesh = MagicMock()
        mesh.size.return_value = 2

        with pytest.raises(ValueError, match=r"\[out, E\*r\]"):
            _shard_lora_linear_on_ep(
                linear, mesh, expert_dim=1, lora_rank=2, num_experts=4
            )

    def test_lora_b_rejects_expert_count_not_divisible_by_ep(self):
        linear = nn.Module()
        linear.weight = nn.Parameter(torch.ones(4, 6))
        mesh = MagicMock()
        mesh.size.return_value = 2

        with pytest.raises(ValueError, match="must be divisible by"):
            _shard_lora_linear_on_ep(
                linear, mesh, expert_dim=1, lora_rank=2, num_experts=3
            )

    def test_lora_a_rejects_a_non_2d_weight(self):
        linear = nn.Module()
        linear.weight = nn.Parameter(torch.ones(2, 2, 2))
        mesh = MagicMock()
        mesh.size.return_value = 2

        with pytest.raises(ValueError, match="Expected 2D LoRA weight"):
            _shard_lora_linear_on_ep(linear, mesh, expert_dim=0)

    def test_lora_a_rejects_a_dim_not_divisible_by_ep(self):
        linear = nn.Module()
        linear.weight = nn.Parameter(torch.ones(3, 4))
        mesh = MagicMock()
        mesh.size.return_value = 2

        with pytest.raises(ValueError, match="must be divisible by"):
            _shard_lora_linear_on_ep(linear, mesh, expert_dim=0)


class TestOuterExpertWrappers:
    def test_includes_a_module_whose_base_layer_is_the_experts(self):
        experts = nn.Linear(2, 2)

        class Holder(nn.Module):
            def __init__(self):
                super().__init__()
                self.experts = experts
                self.wrapper = nn.Identity()
                self.wrapper.base_layer = experts

        found = _outer_expert_wrappers(Holder(), experts)

        assert any(getattr(module, "base_layer", None) is experts for module in found)


class TestEpCommHelpers:
    def test_on_comm_stream_runs_the_body_on_the_side_stream(self):
        comm = MagicMock()
        comm.device = torch.device("cpu")
        current = MagicMock()
        entered = []

        class _StreamCtx:
            def __enter__(self):
                entered.append(True)
                return self

            def __exit__(self, *_exc):
                return False

        with (
            patch("torch.cuda.current_stream", return_value=current),
            patch("torch.cuda.stream", return_value=_StreamCtx()),
        ):
            with _on_comm_stream(comm):
                entered.append("body")

        comm.wait_stream.assert_called_once_with(current)
        assert entered == [True, "body"]

    def test_hand_over_records_streams_and_returns_an_event(self):
        comm = MagicMock()
        comm.device = torch.device("cpu")
        event = object()
        comm.record_event.return_value = event
        current = MagicMock()
        sent = MagicMock()
        received = MagicMock()

        with patch("torch.cuda.current_stream", return_value=current):
            out = _hand_over(comm, [sent], [received])

        sent.record_stream.assert_called_once_with(comm)
        received.record_stream.assert_called_once_with(current)
        assert out is event

    def test_wait_for_an_event_joins_the_current_stream(self):
        event = object()
        current = MagicMock()

        with patch("torch.cuda.current_stream", return_value=current):
            _wait_for(event)

        current.wait_event.assert_called_once_with(event)

    def test_comm_stream_opens_one_stream_per_cuda_device(self):
        device = torch.device("cpu")
        hidden = SimpleNamespace(is_cuda=True, device=device)
        created = []

        def _stream(dev):
            created.append(dev)
            return "side"

        streams: dict = {}
        with patch("torch.cuda.Stream", side_effect=_stream):
            first = _comm_stream(streams, hidden)
            second = _comm_stream(streams, hidden)

        assert first == "side"
        assert second == "side"
        assert created == [device]


class TestRoutingOverride:
    def test_clears_routing_that_was_unset_before_the_body(self):
        module = nn.Linear(2, 2)
        ROUTING_STATE.pop(module, None)

        with _routing_override(module, ["actor"]):
            assert ROUTING_STATE[module] == ["actor"]

        assert module not in ROUTING_STATE


class TestRunLocalExperts:
    def test_empty_rows_keep_trainable_params_in_the_graph(self):
        experts_mod = nn.Linear(2, 2)
        param = torch.ones(2, 2, requires_grad=True)
        state = TokenDispatchState(
            input_splits=[],
            output_splits=[],
            ep_group=object(),
            ep_degree=2,
            num_local_experts=1,
            permute_indices=torch.zeros(0, dtype=torch.long),
            num_tokens_per_local_expert=torch.zeros(1, dtype=torch.long),
        )
        experts = LocalExperts(
            module=experts_mod,
            forward=experts_mod.forward,
            params={"weight": param},
            kwargs={},
        )

        out = _run_local_experts(
            experts,
            state,
            torch.zeros(0, 2),
            None,
            ["actor"],
            torch.long,
        )

        assert out.shape == (0, 2)
        (out.sum() + 0).backward()
        assert param.grad is not None


class TestRoutedAndSortedWeightLookup:
    def test_routed_up_weight_rejects_an_unknown_layout(self):
        with pytest.raises(RuntimeError, match="supported packed layout"):
            _routed_up_weight(nn.Linear(2, 2))

    def test_sorted_weight_rejects_a_module_without_weight(self):
        with pytest.raises(RuntimeError, match="no stacked weight"):
            _sorted_weight(nn.Module())


class TestInstallEpForwardIdempotent:
    def test_routed_install_is_a_no_op_when_already_wrapped(self):
        module = nn.Linear(2, 2)
        object.__setattr__(module, "_agilerl_ep_forward", True)

        _install_routed_ep_forward(module, token_blocks=1, tp_group=None)

        assert "forward" not in module.__dict__

    def test_sorted_install_is_a_no_op_when_already_wrapped(self):
        module = nn.Linear(2, 2)
        object.__setattr__(module, "_agilerl_ep_forward", True)

        _install_sorted_ep_forward(module, tp_group=None)

        assert "forward" not in module.__dict__


class TestShardWrapperAdapters:
    def test_skips_a_wrapper_that_was_already_seen(self):
        wrapper = nn.Linear(2, 2)
        seen = {id(wrapper)}

        _shard_wrapper_adapters(wrapper, MagicMock(), num_experts=4, seen=seen)

        assert seen == {id(wrapper)}


class TestApplyExpertParallelNoneMesh:
    def test_none_mesh_shards_nothing(self):
        assert apply_expert_parallel(nn.Linear(2, 2), None) == 0


class TestScatterEpExpertSlices:
    def test_missing_keys_raise(self):
        model = nn.Linear(2, 2)

        with pytest.raises(RuntimeError, match="Missing expert keys"):
            _scatter_ep_expert_slices(model, {}, frozenset({"weight", "bias"}))

    def test_missing_keys_preview_truncates_after_eight(self):
        model = nn.Linear(2, 2)
        keys = frozenset(f"w{i}" for i in range(10))

        with pytest.raises(RuntimeError, match="…"):
            _scatter_ep_expert_slices(model, {}, keys)


class TestCopyIndexedWeightsShapeMismatch:
    def test_even_split_narrow_that_does_not_match_dest_raises(self, tmp_path):
        keys = ["e0", "e1", "e2", "e3"]
        tensors = {key: torch.ones(2, 2) for key in keys}
        path = tmp_path / "weights.safetensors"
        save_file(tensors, str(path))
        dest = torch.empty(1, 2, 2)
        original_narrow = torch.Tensor.narrow

        def _wrong_narrow(self, dim, start, length):
            if self.ndim == 3:
                return torch.zeros(length, 9, 9)
            return original_narrow(self, dim, start, length)

        with (
            patch("torch.distributed.get_rank", return_value=0),
            patch.object(torch.Tensor, "narrow", _wrong_narrow),
        ):
            with pytest.raises(RuntimeError, match="does not match"):
                _copy_indexed_weights(
                    dict.fromkeys(keys, str(path)),
                    keys,
                    (4, 2, 2),
                    None,
                    dest,
                )


class TestMaterializeNeedsMesh:
    def test_ep_without_a_mesh_raises(self):
        with pytest.raises(ValueError, match="needs a ParallelMesh"):
            materialize_fsdp2_from_cpu_state(
                nn.Linear(2, 2), "cpu", FSDPConfig(ep=2), parallel_mesh=None
            )


class TestMaterializeNeedsChunkSize:
    def test_unset_routed_expert_chunk_raises_and_keeps_the_weights(self):
        # Arrange
        model = nn.Linear(2, 2)
        weight = model.weight.detach().clone()

        # Act / Assert
        with pytest.raises(ValueError, match="routed_expert_chunk_mib is unset"):
            materialize_fsdp2_from_cpu_state(
                model, "cpu", FSDPConfig(routed_expert_chunk_mib=None)
            )
        assert torch.equal(model.weight, weight)


class TestRouteReplicatedSpanAdapterIds:
    def test_slices_adapter_ids_to_this_ranks_span(self):
        captured = {}

        def run_blocks(hidden, index, weights, adapter_ids):
            captured["ids"] = adapter_ids
            return hidden

        class _Fn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, tensor, *rest):
                return tensor[:1] if tensor.shape[0] == 2 else tensor

            @staticmethod
            def backward(ctx, grad, *rest):
                return (grad, *([None] * len(rest)))

        ids = TokenAdapterIds(["actor", "critic"], torch.tensor([0, 1]))
        with (
            patch(
                "agilerl.distributed.ep_forward.replicated_row_span",
                return_value=(0, 1),
            ),
            patch(
                "agilerl.distributed.ep_forward.SliceReplicatedRows",
                _Fn,
            ),
            patch(
                "agilerl.distributed.ep_forward.GatherReplicatedRows",
                _Fn,
            ),
        ):
            hidden = torch.ones(2, 2)
            _route_replicated_span(
                run_blocks,
                hidden,
                torch.zeros(2, 1, dtype=torch.long),
                torch.ones(2, 1),
                ids,
                object(),
            )

        assert captured["ids"].names == ["actor", "critic"]
        assert torch.equal(captured["ids"].ids, torch.tensor([0]))


class TestMaterializeEpWithoutPackedExperts:
    def test_ep_on_a_dense_model_raises(self):
        mesh = MagicMock()
        mesh.hsdp = None
        mesh.ep = MagicMock()
        mesh.tp = None

        with pytest.raises(RuntimeError, match="no packed"):
            materialize_fsdp2_from_cpu_state(
                nn.Linear(2, 2),
                "cpu",
                FSDPConfig(ep=2, routed_expert_chunk_mib=64),
                parallel_mesh=mesh,
            )


class TestEpForwardBypassesWhenDegreeIsOne:
    def test_routed_forward_falls_back_without_an_ep_group(self):
        experts = _tiny_moe_actor(4).layers[0].experts
        hidden = torch.ones(2, 4)
        index = torch.zeros(2, 1, dtype=torch.long)
        weights = torch.ones(2, 1)
        expected = experts(hidden, index, weights)

        _install_routed_ep_forward(experts, token_blocks=1, tp_group=None)
        out = experts(hidden, index, weights)

        assert torch.equal(out, expected)

    def test_sorted_forward_falls_back_without_an_ep_group(self):
        class Sorted(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(4, 3, 2))

            def forward(self, inputs, expert_size):
                return inputs

        module = Sorted()
        x = torch.ones(3, 2)
        expected = module(x, [1, 1, 1, 0])

        _install_sorted_ep_forward(module, tp_group=None)
        out = module(x, torch.tensor([1, 1, 1, 0]))

        assert torch.equal(out, expected)


class TestSortedEpForwardListCounts:
    def test_converts_a_python_list_when_ep_is_active(self):
        class Sorted(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(4, 3, 2))

            def forward(self, inputs, expert_size):
                return inputs

        module = Sorted()
        captured = {}

        def _dispatch(inputs, counts, ep_group, ep_degree, num_local_experts):
            captured["counts"] = counts.clone()
            return (
                inputs,
                counts,
                TokenDispatchState(
                    input_splits=[],
                    output_splits=[],
                    ep_group=ep_group,
                    ep_degree=ep_degree,
                    num_local_experts=num_local_experts,
                    permute_indices=torch.zeros(0, dtype=torch.long),
                    num_tokens_per_local_expert=torch.zeros(1, dtype=torch.long),
                ),
            )

        _install_sorted_ep_forward(module, tp_group=None)
        object.__setattr__(module, "_ep_group", object())
        with (
            patch(
                "agilerl.distributed.ep_forward.module_ep_degree",
                return_value=2,
            ),
            patch(
                "agilerl.distributed.ep_forward.token_dispatch",
                side_effect=_dispatch,
            ),
            patch(
                "agilerl.distributed.ep_forward._gathered_expert_block",
            ) as gathered,
            patch(
                "agilerl.distributed.ep_forward.token_combine",
                side_effect=lambda rows, _state: rows,
            ),
        ):
            gathered.return_value.__enter__.return_value = False
            gathered.return_value.__exit__.return_value = False
            out = module(torch.ones(3, 2), [1, 1, 1, 0])

        assert torch.equal(out, torch.ones(3, 2))
        assert torch.equal(captured["counts"], torch.tensor([1, 1, 1, 0]))

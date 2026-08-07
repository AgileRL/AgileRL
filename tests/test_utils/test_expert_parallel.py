"""Behavior tests for Expert Parallel mesh, shard placement, and token A2A.

U1–U3 / U6 from ``docs/migration/expert-parallel-plan.md``. Multi-rank CPU cases
use a real gloo process group via ``torch.multiprocessing``. When ≥2 CUDA devices
are available, ``TestCudaEpSmoke`` runs NCCL ``ep=2`` shard + A2A + local MoE GEMM
parity (Phase 0–2). Phase 3 FSDP ``mesh=`` compose is not covered here.
"""

from __future__ import annotations

import copy
import os
import sys
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import nn

from agilerl.utils.distributed import FSDPConfig
from agilerl.utils.expert_parallel import (
    build_expert_parallel_mesh,
    compute_ep_mesh_layout,
    expert_local_tensor,
    expert_param_bytes_local,
    reference_dispatch_combine,
    shard_experts_on_ep,
    token_combine,
    token_dispatch,
    validate_ep_config,
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


def _cuda_multi_gpu_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.device_count() >= 2


requires_cuda_multi_gpu = pytest.mark.skipif(
    not _cuda_multi_gpu_available(),
    reason="requires >=2 CUDA devices for NCCL EP smoke",
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
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def _free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _spawn_ranks(worker, world_size: int = 2, timeout: float = 120.0) -> None:
    """Spawn ``world_size`` workers; fail if any rank reports an error."""
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


# ---------------------------------------------------------------------------
# U1 — mesh layout
# ---------------------------------------------------------------------------


class TestEpMeshLayout:
    def test_world_size_and_ep_yield_mod_and_in_ep_sizes(self):
        layout = compute_ep_mesh_layout(world_size=8, ep=2)
        assert layout.dp_shard == 8
        assert layout.dp_shard_mod_ep == 4
        assert layout.dp_shard_in_ep == 2
        assert layout.ep_enabled is True

    def test_ep_one_is_disabled(self):
        layout = compute_ep_mesh_layout(world_size=4, ep=1)
        assert layout.dp_shard_mod_ep == 4
        assert layout.dp_shard_in_ep == 1
        assert layout.ep_enabled is False

    def test_invalid_world_size_raises(self):
        with pytest.raises(ValueError, match="divisible by ep"):
            compute_ep_mesh_layout(world_size=3, ep=2)

    def test_invalid_ep_raises(self):
        with pytest.raises(ValueError, match="ep must be >= 1"):
            compute_ep_mesh_layout(world_size=4, ep=0)

    def test_build_returns_none_when_ep_one(self):
        assert build_expert_parallel_mesh(world_size=2, ep=1) is None


# ---------------------------------------------------------------------------
# U6 — config validation
# ---------------------------------------------------------------------------


class TestEpConfigValidation:
    def test_ep_gt_one_without_fsdp_raises(self):
        with pytest.raises(ValueError, match="requires fsdp_config"):
            validate_ep_config(2, fsdp_config=None)

    def test_ep_gt_one_with_fsdp_ok(self):
        validate_ep_config(2, fsdp_config=FSDPConfig(), world_size=2, num_experts=4)

    def test_ep_one_skips_fsdp_requirement(self):
        validate_ep_config(1, fsdp_config=None)

    def test_num_experts_not_divisible_raises(self):
        with pytest.raises(ValueError, match="num_experts"):
            validate_ep_config(
                2, fsdp_config=FSDPConfig(), world_size=2, num_experts=3
            )

    def test_non_moe_raises(self):
        with pytest.raises(ValueError, match="packed-expert MoE"):
            validate_ep_config(2, fsdp_config=FSDPConfig(), is_moe=False)

    def test_world_size_not_divisible_raises(self):
        with pytest.raises(ValueError, match="world_size"):
            validate_ep_config(2, fsdp_config=FSDPConfig(), world_size=3)

    def test_bool_ep_rejected(self):
        with pytest.raises(TypeError, match="ep must be an int"):
            validate_ep_config(True, fsdp_config=FSDPConfig())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# U2 — Shard(0) placement (2-rank gloo)
# ---------------------------------------------------------------------------


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
        mesh = build_expert_parallel_mesh(world_size=world_size, ep=ep, device_type="cpu")
        assert mesh is not None
        assert mesh.layout.dp_shard_mod_ep == 1
        assert mesh.ep.size() == ep

        module = _PackedExperts(num_experts, out_features=3, in_features=2)
        # Rank 0 owns the dense init; broadcast so Shard placement is consistent.
        for param in module.parameters():
            dist.broadcast(param.data, src=0)

        shard_experts_on_ep(module, mesh.ep)
        local = expert_local_tensor(module.weight)
        local_e = num_experts // ep
        assert local.shape[0] == local_e

        full = module.weight.full_tensor().cpu()
        expected = torch.arange(
            num_experts * 3 * 2, dtype=torch.float32
        ).reshape(num_experts, 3, 2)
        assert torch.allclose(full, expected)

        # Local slice matches the contiguous expert block for this rank.
        start = rank * local_e
        assert torch.allclose(local.cpu(), expected[start : start + local_e])

        bytes_local = expert_param_bytes_local(module)
        bytes_full = expected.numel() * expected.element_size()
        assert bytes_local == bytes_full // ep

        result_queue.put((rank, "ok", None))
    except Exception as exc:  # pragma: no cover - surfaced via queue
        result_queue.put((rank, "err", repr(exc)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@requires_gloo
class TestExpertShardPlacement:
    def test_shard0_local_experts_and_full_tensor_roundtrip(self):
        world_size = 2
        port = _free_port()
        ctx = mp.get_context("spawn")
        queue: mp.Queue = ctx.Queue()
        procs = [
            ctx.Process(target=_shard_worker, args=(rank, world_size, port, queue))
            for rank in range(world_size)
        ]
        for proc in procs:
            proc.start()
        results = [queue.get(timeout=60) for _ in range(world_size)]
        for proc in procs:
            proc.join(timeout=60)
            assert proc.exitcode == 0, f"rank exited {proc.exitcode}"
        for rank, status, err in sorted(results):
            assert status == "ok", f"rank {rank}: {err}"


# ---------------------------------------------------------------------------
# U3 — token dispatch / combine (2-rank gloo)
# ---------------------------------------------------------------------------


def _a2a_worker(rank: int, world_size: int, port: int, result_queue: Any) -> None:
    try:
        _init_gloo(rank, world_size, port)
        ep = world_size
        num_experts = 4
        num_local = num_experts // ep
        hidden = 8

        # Prime-RL data mesh includes EP ranks: each rank holds a different
        # microbatch. Dispatch gathers tokens for local experts from all ranks;
        # combine restores this rank's original expert-sorted rows.
        torch.manual_seed(rank + 1)
        n_tokens = 12
        tokens = torch.randn(n_tokens, hidden)
        expert_ids = torch.randint(0, num_experts, (n_tokens,), generator=torch.Generator().manual_seed(rank + 7))
        order = torch.argsort(expert_ids, stable=True)
        sorted_tokens = tokens[order]
        sorted_ids = expert_ids[order]
        counts = torch.bincount(sorted_ids, minlength=num_experts).to(torch.long)

        mesh = build_expert_parallel_mesh(world_size=world_size, ep=ep, device_type="cpu")
        assert mesh is not None
        group = mesh.ep_group

        local_tokens, local_counts, state = token_dispatch(
            sorted_tokens,
            counts,
            ep_group=group,
            ep_degree=ep,
            num_local_experts=num_local,
        )
        assert local_counts.numel() == num_local
        assert local_tokens.shape[0] == int(local_counts.sum().item())
        # Local buffer holds tokens for this rank's experts from every EP peer.
        expected_local = 0
        for peer in range(ep):
            # Peer counts are unknown here; only check consistency with state.
            _ = peer
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


@requires_gloo
class TestTokenDispatchCombine:
    def test_a2a_roundtrip_matches_sorted_input(self):
        world_size = 2
        port = _free_port()
        ctx = mp.get_context("spawn")
        queue: mp.Queue = ctx.Queue()
        procs = [
            ctx.Process(target=_a2a_worker, args=(rank, world_size, port, queue))
            for rank in range(world_size)
        ]
        for proc in procs:
            proc.start()
        results = [queue.get(timeout=60) for _ in range(world_size)]
        for proc in procs:
            proc.join(timeout=60)
            assert proc.exitcode == 0, f"rank exited {proc.exitcode}"
        for rank, status, err in sorted(results):
            assert status == "ok", f"rank {rank}: {err}"


class TestReferenceDispatchCombine:
    def test_single_process_reference_is_identity(self):
        tokens = torch.randn(10, 4)
        expert_ids = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 1, 2])
        out = reference_dispatch_combine(
            tokens, expert_ids, ep_degree=2, num_experts=4
        )
        assert torch.allclose(out, tokens)


# ---------------------------------------------------------------------------
# U4 — split LoRA on local expert shards
# ---------------------------------------------------------------------------


class TestSplitLoraLocalExperts:
    def test_local_lora_delta_matches_full_e_slice(self):
        """LoRA delta on a local ``E/ep`` shard matches slicing a full-E reference."""
        from agilerl.algorithms.core.llm_ops.moe_lora import _grouped_linear

        num_experts = 4
        local_e = num_experts // 2
        rank = 2
        hidden = 8
        out_features = 6
        torch.manual_seed(0)
        # Stacked PEFT-style A ``[E*r, in]`` (expert-major); B ``[out, E*r]``
        # viewed as ``[out, r, E]`` (rank-major on the stacked axis).
        weight_a = torch.randn(num_experts * rank, hidden)
        weight_b = torch.randn(out_features, num_experts * rank)
        counts_full = torch.tensor([2, 1, 3, 2])
        x_full = torch.randn(int(counts_full.sum()), hidden)

        a3 = weight_a.view(num_experts, rank, hidden)
        b3 = weight_b.view(out_features, rank, num_experts)

        def delta(x, counts, a_exp, b_exp):
            down = _grouped_linear(x, a_exp, counts)
            return _grouped_linear(down, b_exp.permute(2, 0, 1), counts)

        ref = delta(x_full, counts_full, a3, b3)
        local_counts = counts_full[:local_e]
        end = int(local_counts.sum())
        local = delta(x_full[:end], local_counts, a3[:local_e], b3[:, :, :local_e])
        assert torch.allclose(local, ref[:end], atol=1e-5, rtol=1e-5)

    def test_ep_attach_order_constant_locked(self):
        from agilerl.utils.expert_parallel import EP_ATTACH_ORDER

        assert EP_ATTACH_ORDER[0] == "cpu_dense"
        assert "ep_shard" in EP_ATTACH_ORDER
        assert EP_ATTACH_ORDER.index("ep_shard") < EP_ATTACH_ORDER.index(
            "fsdp2_materialize"
        )


# ---------------------------------------------------------------------------
# CUDA multi-GPU smoke (ep=2, NCCL) — Phase 0–2; no FSDP mesh compose
# ---------------------------------------------------------------------------

_CUDA_NUM_EXPERTS = 4
_CUDA_HIDDEN = 8
_CUDA_INTERMEDIATE = 6
_CUDA_TOP_K = 2


class _CudaRoutedExperts(nn.Module):
    """Minimal Qwen3-style packed experts for CUDA EP smoke (no HF download)."""

    def __init__(self) -> None:
        super().__init__()
        self.num_experts = _CUDA_NUM_EXPERTS
        self.gate_up_proj = nn.Parameter(
            torch.randn(_CUDA_NUM_EXPERTS, 2 * _CUDA_INTERMEDIATE, _CUDA_HIDDEN) * 0.1
        )
        self.down_proj = nn.Parameter(
            torch.randn(_CUDA_NUM_EXPERTS, _CUDA_HIDDEN, _CUDA_INTERMEDIATE) * 0.1
        )
        self.act_fn = F.silu

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)
        for expert_idx in range(self.num_experts):
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current = hidden_states[token_idx]
            gate, up = F.linear(current, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current = F.linear(self.act_fn(gate) * up, self.down_proj[expert_idx])
            current = current * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, current)
        return final


def _cuda_shard_worker(
    rank: int, world_size: int, port: int, result_queue: Any
) -> None:
    try:
        _init_nccl(rank, world_size, port)
        device = torch.device(f"cuda:{rank}")
        ep = world_size
        num_experts = _CUDA_NUM_EXPERTS
        mesh = build_expert_parallel_mesh(
            world_size=world_size, ep=ep, device_type="cuda"
        )
        assert mesh is not None
        assert mesh.ep.size() == ep

        module = _PackedExperts(num_experts, out_features=3, in_features=2).to(device)
        for param in module.parameters():
            dist.broadcast(param.data, src=0)

        shard_experts_on_ep(module, mesh.ep)
        local = expert_local_tensor(module.weight)
        local_e = num_experts // ep
        assert local.shape[0] == local_e
        assert local.device.type == "cuda"

        full = module.weight.full_tensor()
        expected = (
            torch.arange(num_experts * 3 * 2, dtype=torch.float32, device=device)
            .reshape(num_experts, 3, 2)
        )
        assert torch.allclose(full, expected)

        start = rank * local_e
        assert torch.allclose(local, expected[start : start + local_e])
        bytes_local = expert_param_bytes_local(module)
        bytes_full = expected.numel() * expected.element_size()
        assert bytes_local == bytes_full // ep

        result_queue.put((rank, "ok", None))
    except Exception as exc:  # pragma: no cover - surfaced via queue
        result_queue.put((rank, "err", repr(exc)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _cuda_a2a_worker(
    rank: int, world_size: int, port: int, result_queue: Any
) -> None:
    try:
        _init_nccl(rank, world_size, port)
        device = torch.device(f"cuda:{rank}")
        ep = world_size
        num_experts = _CUDA_NUM_EXPERTS
        num_local = num_experts // ep
        hidden = _CUDA_HIDDEN

        torch.manual_seed(rank + 1)
        n_tokens = 12
        tokens = torch.randn(n_tokens, hidden, device=device)
        expert_ids = torch.randint(
            0,
            num_experts,
            (n_tokens,),
            device=device,
            generator=torch.Generator(device=device).manual_seed(rank + 7),
        )
        order = torch.argsort(expert_ids, stable=True)
        sorted_tokens = tokens[order]
        sorted_ids = expert_ids[order]
        counts = torch.bincount(sorted_ids, minlength=num_experts).to(torch.long)

        mesh = build_expert_parallel_mesh(
            world_size=world_size, ep=ep, device_type="cuda"
        )
        assert mesh is not None
        local_tokens, local_counts, state = token_dispatch(
            sorted_tokens,
            counts,
            ep_group=mesh.ep_group,
            ep_degree=ep,
            num_local_experts=num_local,
        )
        assert local_tokens.device.type == "cuda"
        assert local_counts.numel() == num_local
        assert int(local_counts.sum().item()) == sum(state.output_splits)

        combined = token_combine(local_tokens, state)
        assert combined.shape == sorted_tokens.shape
        max_err = (combined - sorted_tokens).abs().max().item()
        assert max_err == 0.0, f"CUDA A2A round-trip max abs err {max_err}"

        result_queue.put((rank, "ok", None))
    except Exception as exc:  # pragma: no cover
        result_queue.put((rank, "err", repr(exc)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _cuda_moe_parity_worker(
    rank: int, world_size: int, port: int, result_queue: Any
) -> None:
    """ep=2 local-shard MoE + A2A matches dense ep=1 on this rank's microbatch."""
    try:
        from agilerl.algorithms.core.llm_ops.moe_lora import (
            _routed_experts_local_forward,
        )

        _init_nccl(rank, world_size, port)
        device = torch.device(f"cuda:{rank}")
        ep = world_size

        torch.manual_seed(0)
        experts = _CudaRoutedExperts().to(device)
        for param in experts.parameters():
            dist.broadcast(param.data, src=0)

        # Distinct microbatch per rank (DP folds into EP for token A2A).
        torch.manual_seed(rank + 42)
        hidden = torch.randn(16, _CUDA_HIDDEN, device=device)
        logits = torch.randn(16, _CUDA_NUM_EXPERTS, device=device)
        top_k_weights, top_k_index = torch.softmax(logits, dim=-1).topk(
            _CUDA_TOP_K, dim=-1
        )

        dense = copy.deepcopy(experts)
        with torch.no_grad():
            ref = _routed_experts_local_forward(
                dense, hidden, top_k_index, top_k_weights
            )

        mesh = build_expert_parallel_mesh(
            world_size=world_size, ep=ep, device_type="cuda"
        )
        assert mesh is not None
        shard_experts_on_ep(experts, mesh.ep)
        local_e = _CUDA_NUM_EXPERTS // ep
        assert expert_local_tensor(experts.gate_up_proj).shape[0] == local_e
        bytes_full = sum(
            p.full_tensor().numel() * p.full_tensor().element_size()
            for p in experts.parameters()
        )
        assert expert_param_bytes_local(experts) == bytes_full // ep

        out = _routed_experts_local_forward(
            experts, hidden, top_k_index, top_k_weights
        )
        assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4), (
            f"rank {rank} ep=2 vs dense max abs "
            f"{(out - ref).abs().max().item()}"
        )

        loss = out.square().mean()
        assert torch.isfinite(loss).item()
        loss.backward()
        for name, param in experts.named_parameters():
            if param.grad is None:
                continue
            local_grad = expert_local_tensor(param.grad)
            assert torch.isfinite(local_grad).all(), f"non-finite grad on {name}"

        result_queue.put((rank, "ok", None))
    except Exception as exc:  # pragma: no cover
        result_queue.put((rank, "err", repr(exc)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _cuda_lora_ep_worker(
    rank: int, world_size: int, port: int, result_queue: Any
) -> None:
    """Split LoRA under EP: finite loss; ep=2 matches dense upgraded forward."""
    try:
        from peft import LoraConfig, inject_adapter_in_model

        from agilerl.algorithms.core.llm_ops.moe_lora import upgrade_moe_param_wrappers
        from agilerl.utils.expert_parallel import apply_expert_parallel

        _init_nccl(rank, world_size, port)
        device = torch.device(f"cuda:{rank}")
        ep = world_size

        torch.manual_seed(0)
        host = nn.Module()
        host.experts = _CudaRoutedExperts()
        lora_cfg = LoraConfig(
            r=2,
            lora_alpha=4,
            target_modules=[],
            target_parameters=["experts.gate_up_proj", "experts.down_proj"],
            lora_dropout=0.0,
            init_lora_weights=False,
        )
        peft_model = inject_adapter_in_model(lora_cfg, host, adapter_name="actor")
        assert upgrade_moe_param_wrappers(peft_model) > 0
        peft_model = peft_model.to(device)
        for param in peft_model.parameters():
            dist.broadcast(param.data, src=0)

        torch.manual_seed(rank + 7)
        hidden = torch.randn(12, _CUDA_HIDDEN, device=device)
        logits = torch.randn(12, _CUDA_NUM_EXPERTS, device=device)
        top_k_weights, top_k_index = torch.softmax(logits, dim=-1).topk(
            _CUDA_TOP_K, dim=-1
        )

        dense = copy.deepcopy(peft_model)
        with torch.no_grad():
            ref = dense.experts(hidden, top_k_index, top_k_weights)

        mesh = build_expert_parallel_mesh(
            world_size=world_size, ep=ep, device_type="cuda"
        )
        assert mesh is not None
        n_sharded = apply_expert_parallel(peft_model, mesh.ep)
        assert n_sharded >= 1

        out = peft_model.experts(hidden, top_k_index, top_k_weights)
        assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4), (
            f"rank {rank} LoRA ep=2 vs dense max abs "
            f"{(out - ref).abs().max().item()}"
        )
        loss = out.square().mean()
        assert torch.isfinite(loss).item()
        loss.backward()

        result_queue.put((rank, "ok", None))
    except Exception as exc:  # pragma: no cover
        result_queue.put((rank, "err", repr(exc)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@requires_cuda_multi_gpu
class TestCudaEpSmoke:
    """NCCL ep=2 smoke on real GPUs (strongest path without Phase 3 FSDP mesh)."""

    def test_cuda_shard0_local_experts_and_bytes(self):
        _spawn_ranks(_cuda_shard_worker)

    def test_cuda_a2a_roundtrip(self):
        _spawn_ranks(_cuda_a2a_worker)

    def test_cuda_routed_ep2_matches_dense_finite_loss(self):
        _spawn_ranks(_cuda_moe_parity_worker)

    def test_cuda_split_lora_ep2_matches_dense_finite_loss(self):
        _spawn_ranks(_cuda_lora_ep_worker)

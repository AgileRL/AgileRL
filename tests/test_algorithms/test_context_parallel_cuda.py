"""Multi-GPU CUDA smoke for context-parallel Ulysses (Phase 0 collectives).

Proves NCCL ``cp=2`` shard/gather + Ulysses all-to-all against a ``cp=1``
golden using a pure-torch varlen attention stub. Full HF FA2 logprob parity
is gated on ``flash-attn`` (skipped when the wheel is unavailable).
"""

from __future__ import annotations

import os
import socket
from typing import Callable

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from agilerl.utils.cp import (
    ParallelDims,
    flash_attn_available,
    gather_for_cp,
    setup_cp_params,
    shard_for_cp,
    substitute_cp_attention,
)
from agilerl.utils.ulysses_attn import ulysses_flash_attn_varlen_func

pytestmark = pytest.mark.gpu

_WORLD_SIZE = 2
# Plan N1 starting band for bf16-ish abs error on attention / logprobs.
_ABS_TOL = 1e-2


def _require_two_cuda_gpus() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < _WORLD_SIZE:
        pytest.skip(f"need >= {_WORLD_SIZE} CUDA GPUs for CP multi-GPU smoke")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _spawn_nccl(worker: Callable, *worker_args) -> None:
    """Run ``worker(rank, world_size, port, *worker_args)`` on 2 NCCL ranks."""
    _require_two_cuda_gpus()
    port = _free_port()
    mp.spawn(
        _spawn_entry,
        args=(worker, port, worker_args),
        nprocs=_WORLD_SIZE,
        join=True,
    )


def _spawn_entry(rank: int, worker, port: int, worker_args) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(_WORLD_SIZE)
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=_WORLD_SIZE,
        device_id=torch.device(f"cuda:{rank}"),
    )
    try:
        worker(rank, _WORLD_SIZE, *worker_args)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _reference_varlen_flash(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    *,
    causal: bool = True,
    **kwargs,
) -> torch.Tensor:
    """Pure-torch stand-in for ``flash_attn_varlen_func`` (``[S, H, D]``)."""
    del max_seqlen_q, max_seqlen_k, kwargs
    if not causal:
        raise AssertionError("CP smoke stub only supports causal attention")
    if not torch.equal(cu_seqlens_q, cu_seqlens_k):
        raise AssertionError("CP smoke stub expects shared cu_seqlens for Q/K")

    outs: list[torch.Tensor] = []
    for i in range(cu_seqlens_q.numel() - 1):
        start = int(cu_seqlens_q[i].item())
        end = int(cu_seqlens_q[i + 1].item())
        # [S, H, D] -> [1, H, S, D] for SDPA
        qi = q[start:end].transpose(0, 1).unsqueeze(0)
        ki = k[start:end].transpose(0, 1).unsqueeze(0)
        vi = v[start:end].transpose(0, 1).unsqueeze(0)
        oi = F.scaled_dot_product_attention(qi, ki, vi, is_causal=True)
        outs.append(oi.squeeze(0).transpose(0, 1))
    return torch.cat(outs, dim=0)


def _worker_shard_gather_roundtrip(rank: int, world_size: int) -> None:
    device = torch.device(f"cuda:{rank}")
    dims = ParallelDims.from_world(world_size=world_size, cp=world_size)
    dims.build_mesh(device_type="cuda")
    cp_group = dims.cp_group()
    cp_rank = dims.cp_rank()
    assert cp_rank == rank

    full = torch.arange(16, device=device, dtype=torch.float32).view(1, 16)
    local = shard_for_cp(full, cp_rank=cp_rank, cp_world_size=dims.cp)
    gathered = gather_for_cp(local, cp_group)
    torch.testing.assert_close(gathered, full, atol=0.0, rtol=0.0)


def _worker_ulysses_stub_parity(rank: int, world_size: int) -> None:
    """``cp=2`` Ulysses + stub FA vs ``cp=1`` full-sequence stub (same QKV)."""
    device = torch.device(f"cuda:{rank}")
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    seq_len = 32
    n_heads = 4
    head_dim = 16
    assert seq_len % world_size == 0
    assert n_heads % world_size == 0

    q = torch.randn(seq_len, n_heads, head_dim, device=device, dtype=torch.float16)
    k = torch.randn(seq_len, n_heads, head_dim, device=device, dtype=torch.float16)
    v = torch.randn(seq_len, n_heads, head_dim, device=device, dtype=torch.float16)
    dist.broadcast(q, src=0)
    dist.broadcast(k, src=0)
    dist.broadcast(v, src=0)

    cu_seqlens = torch.tensor([0, seq_len], device=device, dtype=torch.int32)
    max_seqlen = seq_len

    golden = _reference_varlen_flash(
        q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, causal=True
    )

    dims = ParallelDims.from_world(world_size=world_size, cp=world_size)
    dims.build_mesh(device_type="cuda")
    cp_group = dims.cp_group()
    cp_rank = dims.cp_rank()

    q_local = shard_for_cp(q.unsqueeze(0), cp_rank=cp_rank, cp_world_size=dims.cp).squeeze(0)
    k_local = shard_for_cp(k.unsqueeze(0), cp_rank=cp_rank, cp_world_size=dims.cp).squeeze(0)
    v_local = shard_for_cp(v.unsqueeze(0), cp_rank=cp_rank, cp_world_size=dims.cp).squeeze(0)

    out_local = ulysses_flash_attn_varlen_func(
        _reference_varlen_flash,
        q_local,
        k_local,
        v_local,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        causal=True,
        cp_group=cp_group,
        cp_size=dims.cp,
    )
    # Restore full-sequence layout for comparison with the cp=1 golden.
    out_full = gather_for_cp(out_local.unsqueeze(0), cp_group).squeeze(0)
    torch.testing.assert_close(out_full, golden, atol=_ABS_TOL, rtol=1e-3)


def _worker_setup_cp_params_shard(rank: int, world_size: int) -> None:
    device = torch.device(f"cuda:{rank}")
    dims = ParallelDims.from_world(world_size=world_size, cp=world_size)
    dims.build_mesh(device_type="cuda")
    seq_len = 16
    input_ids = torch.arange(seq_len, device=device, dtype=torch.long).view(1, seq_len)
    position_ids = input_ids.clone()
    seq_lens = torch.tensor([seq_len], device=device, dtype=torch.int32)
    local_ids, local_pos = setup_cp_params(
        input_ids,
        position_ids,
        cp_rank=dims.cp_rank(),
        cp_world_size=dims.cp,
        cp_group=dims.cp_group(),
        seq_lens=seq_lens,
        cp_style="ulysses",
    )
    assert local_ids.shape == (1, seq_len // world_size)
    assert local_pos.shape == (1, seq_len // world_size)
    gathered = gather_for_cp(local_ids.to(dtype=torch.float32), dims.cp_group())
    torch.testing.assert_close(
        gathered.to(dtype=torch.long), input_ids, atol=0, rtol=0
    )


class TestContextParallelCudaSmoke:
    def test_shard_gather_roundtrip_nccl(self):
        _spawn_nccl(_worker_shard_gather_roundtrip)

    def test_ulysses_stub_attn_matches_cp1_golden(self):
        _spawn_nccl(_worker_ulysses_stub_parity)

    def test_setup_cp_params_shards_ids(self):
        _spawn_nccl(_worker_setup_cp_params_shard)

    def test_ring_substitute_wires_hf_adapter(self):
        _require_two_cuda_gpus()
        if not flash_attn_available():
            pytest.skip("flash-attn required for ring CP substitute")
        from agilerl.utils.cp import ring_flash_attn_available

        if not ring_flash_attn_available():
            pytest.skip("ring-flash-attn unavailable after transformers compat shim")

        # Lightweight: patch the ring entrypoints instead of NCCL init here.
        from unittest.mock import MagicMock, patch

        import agilerl.utils.ring_attn_compat  # noqa: F401 — before ring_flash_attn

        with (
            patch("ring_flash_attn.substitute_hf_flash_attn") as sub,
            patch("ring_flash_attn.adapters.hf_adapter.use_ring_attn") as use_ring,
        ):
            substitute_cp_attention("ring", MagicMock())
            sub.assert_called_once()
            use_ring.assert_called_once_with(False)

    def test_hf_fa2_logprob_parity_requires_flash_attn(self):
        """N1 full HF path — skip until a torch/CUDA-matched FA2 wheel exists."""
        if flash_attn_available():
            pytest.skip(
                "flash-attn is installed; extend this test with HF CausalLM "
                "cp=2 vs cp=1 logprob parity (plan N1)"
            )
        pytest.skip(
            "flash-attn (FA2) missing: no wheel for torch "
            f"{torch.__version__} / CUDA {torch.version.cuda} / "
            f"CPython {torch.version}; nvcc unavailable to build from sdist. "
            "Ulysses HF monkeypatch imports flash_attn_varlen_func."
        )

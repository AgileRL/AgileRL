"""Isolate the wall-clock + peak-memory cost of the fp32 promotion inside
``LLMAlgorithm._memory_efficient_logits`` and ``_fused_linear_logprobs_no_grad``.

Background
----------
``_memory_efficient_logits`` always promotes a ``(_chunk_rows, T, V)`` slice
to fp32 before ``amax``/``logsumexp``. That cast was introduced in
``2176925a`` to reduce bf16 numerical error vs an fp32 reference. The
counterpart no-grad path ``_fused_linear_logprobs_no_grad`` exposes the
promotion as an opt-in ``cast_to_fp32`` flag, default ``False``, so the
two paths are *asymmetric* by default.

This bench measures the cost of the fp32 promotion on realistic LLM
shapes so we can decide whether to (a) unify on fp32 (worth it), (b)
unify on bf16 (cast is cheap to drop), or (c) expose it as a flag.

What it does
------------
For each shape, run all four code paths under ``no_grad``:

* ``mel_fp32``  — current ``_memory_efficient_logits`` (fp32 cast inside chunk)
* ``mel_bf16``  — patched copy with the ``.float()`` removed
* ``flp_fp32``  — ``_fused_linear_logprobs_no_grad(..., cast_to_fp32=True)``
* ``flp_bf16``  — ``_fused_linear_logprobs_no_grad(..., cast_to_fp32=False)``  (current default)

Reports per shape, best-of-N: wall-clock time (CUDA events) and peak
device memory delta vs entry. Also prints max abs deviation vs a true
fp32 reference computed on the same inputs.

Usage
-----
    python benchmarking/benchmarking_fp32_cast_cost.py
    python benchmarking/benchmarking_fp32_cast_cost.py --shapes 8,512,128256 16,1024,128256
"""

from __future__ import annotations

import argparse
import gc
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from agilerl.algorithms.core.base import LLMAlgorithm


# ---------------------------------------------------------------------------
# Patched bf16 variant of _memory_efficient_logits (drops the .float() cast)
# ---------------------------------------------------------------------------


def memory_efficient_logits_bf16(
    logits: torch.Tensor,
    index: torch.Tensor,
    _chunk_rows: int = 8,
) -> torch.Tensor:
    """``_memory_efficient_logits`` with the fp32 promotion removed."""
    orig_dtype = logits.dtype
    B = logits.shape[0]

    def _chunk(lg: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        max_lg = lg.amax(dim=-1, keepdim=True)
        shifted = lg - max_lg
        target = shifted.gather(dim=-1, index=idx.unsqueeze(-1)).squeeze(-1)
        log_z = torch.logsumexp(shifted, dim=-1)
        return (target - log_z).to(orig_dtype)

    if B <= _chunk_rows:
        return _chunk(logits, index)
    out = []
    for s in range(0, B, _chunk_rows):
        e = min(s + _chunk_rows, B)
        out.append(_chunk(logits[s:e], index[s:e]))
    return torch.cat(out, dim=0)


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _peak_mem_gb() -> float:
    return torch.cuda.max_memory_allocated() / (1024**3)


def _sync():
    torch.cuda.synchronize()


@dataclass
class TimingResult:
    name: str
    ms_best: float
    ms_mean: float
    peak_gb: float
    max_abs_err_vs_fp32: float


def run_timed(
    name: str, fn, n_warmup: int, n_iters: int, reference: torch.Tensor
) -> TimingResult:
    """Time `fn` (no args) with CUDA events; track best/mean ms and peak mem."""
    for _ in range(n_warmup):
        fn()
    _sync()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    times_ms: list[float] = []
    out_for_err: torch.Tensor | None = None
    for _ in range(n_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        _sync()
        times_ms.append(start.elapsed_time(end))
        if out_for_err is None:
            out_for_err = out
    peak = _peak_mem_gb()
    err = (out_for_err.float() - reference).abs().max().item()
    return TimingResult(
        name=name,
        ms_best=min(times_ms),
        ms_mean=sum(times_ms) / len(times_ms),
        peak_gb=peak,
        max_abs_err_vs_fp32=err,
    )


# ---------------------------------------------------------------------------
# Single-shape harness
# ---------------------------------------------------------------------------


def benchmark_shape(
    B: int,
    T: int,
    V: int,
    H: int,
    n_warmup: int,
    n_iters: int,
    mel_chunk: int,
    flp_chunk: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> list[TimingResult]:
    torch.manual_seed(0)
    hidden = torch.randn(B, T, H, dtype=dtype, device=device)
    W = torch.randn(V, H, dtype=dtype, device=device) * (H**-0.5)
    bias = None
    target_ids = torch.randint(0, V, (B, T), device=device)

    # fp32 reference (very expensive at large V — we only run it once)
    with torch.no_grad():
        logits_fp32 = hidden.float() @ W.float().t()  # (B, T, V)
        ref = (
            F.log_softmax(logits_fp32, dim=-1)
            .gather(dim=-1, index=target_ids.unsqueeze(-1))
            .squeeze(-1)
        )
        del logits_fp32

    # Materialise bf16 (B, T, V) once for the MEL paths.
    logits_bf16 = hidden @ W.t()  # (B, T, V) bf16

    results: list[TimingResult] = []

    # MEL: current fp32-cast path
    results.append(
        run_timed(
            name="mel_fp32_cast",
            fn=lambda: LLMAlgorithm._memory_efficient_logits(
                logits_bf16, target_ids, _chunk_rows=mel_chunk
            ),
            n_warmup=n_warmup,
            n_iters=n_iters,
            reference=ref,
        )
    )
    # MEL: bf16-only patched variant
    results.append(
        run_timed(
            name="mel_bf16_only",
            fn=lambda: memory_efficient_logits_bf16(
                logits_bf16, target_ids, _chunk_rows=mel_chunk
            ),
            n_warmup=n_warmup,
            n_iters=n_iters,
            reference=ref,
        )
    )
    del logits_bf16

    # FLP: cast_to_fp32=True
    results.append(
        run_timed(
            name="flp_cast_fp32",
            fn=lambda: LLMAlgorithm._fused_linear_logprobs_no_grad(
                hidden,
                W,
                bias,
                target_ids,
                temperature=1.0,
                cast_to_fp32=True,
                _chunk_rows=flp_chunk,
            ),
            n_warmup=n_warmup,
            n_iters=n_iters,
            reference=ref,
        )
    )
    # FLP: cast_to_fp32=False (current default)
    results.append(
        run_timed(
            name="flp_bf16",
            fn=lambda: LLMAlgorithm._fused_linear_logprobs_no_grad(
                hidden,
                W,
                bias,
                target_ids,
                temperature=1.0,
                cast_to_fp32=False,
                _chunk_rows=flp_chunk,
            ),
            n_warmup=n_warmup,
            n_iters=n_iters,
            reference=ref,
        )
    )

    return results


def _parse_shape(s: str) -> tuple[int, int, int]:
    parts = [int(p) for p in s.split(",")]
    if len(parts) != 3:
        raise ValueError(f"shape must be B,T,V; got {s}")
    return parts[0], parts[1], parts[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shapes",
        nargs="+",
        default=[
            "1,256,32000",  # tiny: small model, short context
            "8,512,32000",  # small model, medium context
            "4,1024,128256",  # Llama-3 vocab, longer ctx
            "8,1024,128256",  # Llama-3, larger batch
        ],
        help="B,T,V tuples",
    )
    parser.add_argument("--hidden", type=int, default=4096, help="H")
    parser.add_argument("--n-warmup", type=int, default=2)
    parser.add_argument("--n-iters", type=int, default=10)
    parser.add_argument(
        "--mel-chunk",
        type=int,
        default=8,
        help="_chunk_rows for _memory_efficient_logits",
    )
    parser.add_argument(
        "--flp-chunk",
        type=int,
        default=1024,
        help="_chunk_rows for _fused_linear_logprobs_no_grad",
    )
    args = parser.parse_args()

    assert torch.cuda.is_available(), "this bench requires CUDA"
    print(f"device: {torch.cuda.get_device_name(0)}  H={args.hidden}  dtype=bf16")
    print(f"mel_chunk={args.mel_chunk}  flp_chunk={args.flp_chunk}")
    print(f"warmup={args.n_warmup}  iters={args.n_iters}")

    for shape in args.shapes:
        B, T, V = _parse_shape(shape)
        print()
        print(f"=== shape B={B} T={T} V={V}  (B*T={B * T}) ===")
        with torch.inference_mode():
            results = benchmark_shape(
                B=B,
                T=T,
                V=V,
                H=args.hidden,
                n_warmup=args.n_warmup,
                n_iters=args.n_iters,
                mel_chunk=args.mel_chunk,
                flp_chunk=args.flp_chunk,
            )
        header = f"{'path':18s} {'best ms':>10s} {'mean ms':>10s} {'peak GB':>10s} {'max|err| vs fp32':>20s}"
        print(header)
        print("-" * len(header))
        for r in results:
            print(
                f"{r.name:18s} {r.ms_best:10.3f} {r.ms_mean:10.3f} "
                f"{r.peak_gb:10.3f} {r.max_abs_err_vs_fp32:20.6f}"
            )
        # Quick summary deltas
        by = {r.name: r for r in results}
        mel_delta_ms = by["mel_fp32_cast"].ms_best - by["mel_bf16_only"].ms_best
        flp_delta_ms = by["flp_cast_fp32"].ms_best - by["flp_bf16"].ms_best
        mel_delta_gb = by["mel_fp32_cast"].peak_gb - by["mel_bf16_only"].peak_gb
        flp_delta_gb = by["flp_cast_fp32"].peak_gb - by["flp_bf16"].peak_gb
        print(
            f"\nMEL fp32-cast cost vs bf16:  Δt={mel_delta_ms:+.3f} ms   "
            f"Δmem={mel_delta_gb:+.3f} GB"
        )
        print(
            f"FLP fp32-cast cost vs bf16:  Δt={flp_delta_ms:+.3f} ms   "
            f"Δmem={flp_delta_gb:+.3f} GB"
        )


if __name__ == "__main__":
    main()

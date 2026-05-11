"""Compare ``cast_logprobs_to_fp32=True`` vs ``False`` on a real agent.

Drives the rollout-side no-grad logprob pass through both code paths
(``use_fused_linear_logprobs`` False / True) and toggles
``agent.cast_logprobs_to_fp32`` between measurements. Reports peak GPU
memory and wall-clock for the 4-cell cross product.

The microbench in ``benchmarking_fp32_cast_cost.py`` isolates just the
two kernels with synthetic inputs. This bench wraps the full
``_fused_forward_no_grad`` path (model forward + identity-patch
machinery + chunked kernel) so the numbers include the overhead a real
GRPO/PPO learn() step pays.

Usage
-----
    python benchmarking/benchmarking_cast_logprobs_fp32.py
    python benchmarking/benchmarking_cast_logprobs_fp32.py --batch 4 --seq-len 1024
"""

from __future__ import annotations

import argparse
import gc

import torch
from peft import LoraConfig
from transformers import AutoTokenizer

from agilerl.algorithms.grpo import GRPO


def measure_no_grad_peak(agent: GRPO, ids: torch.Tensor, batch_size: int) -> float:
    """Run ``_fused_forward_no_grad`` once, return peak GPU mem (GB)."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.no_grad():
        agent._fused_forward_no_grad(ids, batch_size=batch_size)
    end.record()
    torch.cuda.synchronize()
    peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
    elapsed_ms = start.elapsed_time(end)
    return peak_gb, elapsed_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--n-warmup", type=int, default=1)
    parser.add_argument("--n-iters", type=int, default=3)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    pad = tokenizer.pad_token_id or tokenizer.eos_token_id

    lora_config = LoraConfig(
        r=16,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
    )

    # Build once; toggle live attrs between measurements so weights stay fixed.
    agent = GRPO(
        actor_network=None,
        model_name=args.model,
        lr=1e-5,
        pad_token_id=pad,
        pad_token=tokenizer.pad_token or tokenizer.eos_token,
        device="cuda",
        group_size=4,
        lora_config=lora_config,
        use_vllm=False,
        beta=0.0,
        update_epochs=1,
        use_fused_linear_logprobs=False,
        cast_logprobs_to_fp32=True,
    )
    agent.actor.eval()

    torch.manual_seed(42)
    ids = torch.randint(
        0, tokenizer.vocab_size, (args.batch, args.seq_len), device="cuda"
    )

    cells = [
        # (label, use_fused_linear_logprobs, cast_logprobs_to_fp32)
        ("unfused/fp32", False, True),
        ("unfused/bf16", False, False),
        ("fused/fp32", True, True),
        ("fused/bf16", True, False),
    ]

    # Warmup all four to settle allocator / autotune.
    for _ in range(args.n_warmup):
        for _, use_fused, cast in cells:
            agent.use_fused_linear_logprobs = use_fused
            agent.cast_logprobs_to_fp32 = cast
            measure_no_grad_peak(agent, ids, batch_size=args.batch)

    results: dict[str, list[tuple[float, float]]] = {label: [] for label, _, _ in cells}
    for i in range(args.n_iters):
        for label, use_fused, cast in cells:
            agent.use_fused_linear_logprobs = use_fused
            agent.cast_logprobs_to_fp32 = cast
            peak_gb, t_ms = measure_no_grad_peak(agent, ids, batch_size=args.batch)
            results[label].append((peak_gb, t_ms))
            print(f"iter {i} {label:14s}: peak={peak_gb:6.3f} GB  t={t_ms:7.1f} ms")
        print()

    print()
    print(f"shape: B={args.batch}  T={args.seq_len}  V={tokenizer.vocab_size}")
    print(
        f"{'cell':14s} {'best ms':>10s} {'mean ms':>10s} {'best GB':>10s} {'mean GB':>10s}"
    )
    print("-" * 60)
    bests: dict[str, tuple[float, float]] = {}
    for label, _, _ in cells:
        peaks = [p for p, _ in results[label]]
        times = [t for _, t in results[label]]
        b_peak, b_ms = min(peaks), min(times)
        m_peak, m_ms = sum(peaks) / len(peaks), sum(times) / len(times)
        bests[label] = (b_peak, b_ms)
        print(f"{label:14s} {b_ms:10.1f} {m_ms:10.1f} {b_peak:10.3f} {m_peak:10.3f}")

    print()
    print("Cast cost (fp32 vs bf16) — best of N:")
    for prefix in ("unfused", "fused"):
        peak32, t32 = bests[f"{prefix}/fp32"]
        peak16, t16 = bests[f"{prefix}/bf16"]
        print(
            f"  {prefix:>7s}:  Δmem={peak32 - peak16:+.3f} GB  "
            f"Δt={t32 - t16:+.1f} ms  ({(t32 / t16 - 1) * 100:+.1f}% wall)"
        )


if __name__ == "__main__":
    main()

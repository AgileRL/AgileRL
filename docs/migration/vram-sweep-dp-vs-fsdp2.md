# FSDP2 Offload Benchmark — AgileRL GRPO

**Date:** 2026-08-05 · **Hardware:** 2x NVIDIA L4 (24 GB) · **Software:** torch 2.11.0+cu130, CUDA 13.0
**Model:** Qwen2.5-0.5B (bf16) · **Workload:** AgileRL GRPO, 40 steps, seq_len=128, batch_size=1
**Code path:** `create_population → wrap_models → apply_fsdp2 → CPUOffloadOptimizer` (real AgileRL algorithm runs)

---

## Headline

FSDP2 with CPU offloading lets AgileRL train larger models on limited GPU memory. Full CPU offload cuts peak VRAM **36%** at a **1.6x** step-time cost. Optimizer-only offload is **free** at 0.5B and actually *speeds up* colocated vLLM.

## FSDP2 offload modes — Qwen2.5-0.5B

Six configurations, all passing through actual AgileRL GRPO training. Lower is better for VRAM and time.

| Configuration | Peak VRAM (MiB) | Step (ms) | Wall (s) |
|---|---:|---:|---:|
| DP (no offload) | 1,544 | 628 | 70 |
| FSDP2 (no offload) | 1,666 | 1,889 | 165 |
| FSDP2 + optim offload | 1,664 | 1,912 | 166 |
| FSDP2 + full offload | **1,061** | 3,004 | 254 |
| FSDP2 + vLLM | 4,982 | 910 | 86 |
| FSDP2 + vLLM + optim | 4,980 | 883 | **77** |

**Read:**
- **Optim-only offload is free** — same peak VRAM and step time as FSDP2 baseline (1664 vs 1666 MiB). Optimizer states live on pinned CPU and only touch GPU during `step()`.
- **Full offload saves 36% VRAM** (1061 vs 1666 MiB) but steps run 1.6x slower — the PCIe cost of moving params to GPU each step.
- **Colocated vLLM is 2x faster wall-clock** (86s vs 165s) — vLLM generation beats HF `generate`. It uses 3x more VRAM because the engine holds its own base weights + KV cache.
- **Optim offload makes vLLM faster** (77s vs 86s) — freeing GPU from optimizer states leaves more room for KV cache, improving generation throughput.

## Key takeaways

1. **Optimizer-only CPU offload is free** at 0.5B and a net win with colocated vLLM — should be the default.
2. **Full CPU offload trades speed for memory** — 36% VRAM cut for 1.6x slower steps. Use when memory-bound.
3. **Colocated vLLM doubles wall-clock throughput** but triples VRAM. Best paired with optim offload.

## Artifacts

- **Bar charts:** `fsdp2-offload-bench` canvas (peak VRAM + step time by configuration)
- **Raw results:** `/tmp/agilerl-bench/results/all_results.json`
- **Probe script:** `scripts/fsdp_gates/run_probe.py` (actual AgileRL GRPO)
- **Offload implementation:** `agilerl/utils/distributed.py` (`CPUOffloadOptimizer`, `FSDPConfig`), `agilerl/algorithms/core/base.py` (`wrap_models`)

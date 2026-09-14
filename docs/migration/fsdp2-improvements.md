# FSDP2 improvements

**Date:** 2026-09-11 · **Hardware:** 2× NVIDIA L4 (24 GB) · **NCCL:** Socket / ens7
**Code:** `apply_fsdp2` → `_set_prefetch` → `set_modules_to_forward_prefetch` /
`set_modules_to_backward_prefetch`. Timers from `agilerl.fsdp_profile` (commit
`b64f096a`) plus SFT `microbatch_forward`.
**Probe:** `demos/llm/debugging/sft_fsdp_prefetch_bench.py` (1 warmup + 4 measured
`learn()` steps, LoRA r=16, FSDP2 `reshard_after_forward=True`).

---

## Headline

Explicit backward prefetch is wired on the same unit chain as forward
(embed → blocks → `lm_head`). FSDP2 already prefetches the previous unit from
reverse post-forward order, so a singleton explicit list does not change
Qwen 0.5B SFT step time.

On a larger dense model the overlap is real: **Granite 2B backward dropped
14%** when prefetch was on versus forced off (1003 ms → 863 ms).

Peak CUDA memory is **unchanged**. Rank 0 after warmup, 4 measured steps:

| Model | Prefetch | Peak alloc (MiB) | Peak reserved (MiB) |
|---|---|---:|---:|
| Qwen 0.5B, GC on, seq 512 | off | 1,831 | 2,624 |
| Qwen 0.5B, GC on, seq 512 | on | 1,831 | 2,624 |
| Granite 2B, GC off, seq 512 | off | 6,373 | 6,910 |
| Granite 2B, GC off, seq 512 | on | 6,373 | 6,910 |

Singleton reverse prefetch does not raise the high-water mark versus prefetch off
on these runs. Activations dominate; one extra unsharded layer does not show up
in `max_memory_allocated` / `max_memory_reserved`.

## Qwen2.5-0.5B (oss `sft.yaml` model)

2 GPUs, seq 512, batch 2, gradient checkpointing on. Mean CUDA-synced stage
time over 4 measured steps, rank 0.

| Config | Forward (ms) | Backward (ms) | Wall (s) | `wait_unshard` bwd (ms, 4 steps) |
|---|---:|---:|---:|---:|
| Forward prefetch only (FSDP2 default bwd) | 294.1 | 279.1 | 4.518 | 60.3 |
| Explicit reverse pair | 302.6 | 285.1 | 4.630 | 59.5 |

No win. Layer all-gather wait is ~0.5 ms; backward is ~280 ms of compute.

Same model, checkpointing off, seq 1024 (prefetch disabled vs explicit):

| Config | Forward (ms) | Backward (ms) | Wall (s) |
|---|---:|---:|---:|
| Prefetch off | 348.2 | 366.6 | 5.319 |
| Explicit reverse pair | 356.0 | 359.7 | 5.300 |

~2% faster backward, inside run-to-run noise on this size.

## Granite 3.3 2B

2 GPUs, seq 512, batch 2, gradient checkpointing **off**. Prefetch off clears
FSDP2's default reverse-order prefetch and the explicit list.

| Config | Forward (ms) | Backward (ms) | Wall (s) | Peak alloc (MiB) | Peak reserved (MiB) |
|---|---:|---:|---:|---:|---:|
| Prefetch off | 870.6 | **1002.7** | 10.344 | 6,373 | 6,910 |
| Explicit reverse pair | 876.4 | **863.1** | 9.810 | 6,373 | 6,910 |

Backward **−14%**. Peak alloc/reserved match to the MiB. `wait_unshard` CPU
sums stay ~90 ms because those wrappers do not CUDA-sync; the overlap shows
up in the synced `microbatch_backward` timer.

## Activation checkpoint wrap order

`apply_fsdp2` wraps each transformer block with `checkpoint_wrapper`
(`NO_REENTRANT`, `preserve_rng_state=False`) then `fully_shard`s the wrapper.
HuggingFace `gradient_checkpointing_enable` still runs on the DP path only.

Same probe as above (GC on, seq 512, prefetch on). Sequential before/after on
this host.

| Model | Wrap | Forward (ms) | Backward (ms) | Wall (s) | Peak alloc (MiB) | Peak reserved (MiB) |
|---|---|---:|---:|---:|---:|---:|
| Qwen 0.5B | HF after FSDP | 296.1 | 284.4 | 4.569 | 1,831 | 2,624 |
| Qwen 0.5B | checkpoint then shard | 307.4 | 282.9 | 4.588 | 1,831 | 2,624 |
| Granite 2B | HF after FSDP | 889.3 | 879.6 | 9.660 | 3,576 | 4,218 |
| Granite 2B | checkpoint then shard | 882.9 | 897.8 | 9.676 | 3,572 | 4,202 |

No step-time win. Peak alloc is flat (Qwen identical; Granite −4 MiB). Reserved
on Granite dropped 16 MiB. Deltas sit in run-to-run noise. This is wrap-order
correctness for a later per-block `compile`, not a compile speedup.

Granite GC on vs the GC-off prefetch table above: peak alloc 3,576 vs 6,373 MiB
(activation save). That is AC vs no-AC, not wrap order.

## What the debugs show

- `microbatch_forward` / `microbatch_backward` CUDA-sync; use these for
  step-time claims.
- Collectives (`all_gather_unshard`, `wait_unshard`) use `sync=False` so
  prefetch can overlap. Autograd unshards do not inherit `parent=`, so
  backward waits land as `parent=none`.
- Optimizer (~66 ms on 0.5B, ~116 ms on 2B) is unchanged by prefetch.

## Reproduce

```bash
cd oss/agilerl
CUDA_VISIBLE_DEVICES=0,1 NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=ens7 NCCL_NET=Socket \
  PYTHONUNBUFFERED=1 \
  torchrun --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:29502 \
  demos/llm/debugging/sft_fsdp_prefetch_bench.py --no-gc --seq-len 1024 --bwd-prefetch on
```

`--bwd-prefetch off` is bench-only (clears default + explicit lists). GPU
Nemotron remains a cluster check.

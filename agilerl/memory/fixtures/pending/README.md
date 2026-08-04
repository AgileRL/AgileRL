# Pending measurements

Sweeps that cost GPU time but cannot join the curated set yet. `curated_profiles()`
globs `*.json` non-recursively, so nothing here is loaded by the estimator or
checked by the drift test.

Promote one by fixing the reason it is here, re-running
`python -m agilerl.memory.profiling.refit`, and moving it up a directory.

### `...@NVIDIA-A100-SXM4-40GB.nf4.json`

Qwen2.5-0.5B on an A100, 42 points, nf4 on both trainer and engine.
Generation is fine (0.71% mean). Training refits to **20.8% mean**, against
~5% for the same model unquantized.

### What the gap is — measured 2026-07-30

**It is not a dequantized weight copy.** It scales with tokens, so it is
activation memory. Paired bf16/nf4 runs at identical corners on an A100
(Qwen2.5-0.5B, group 8, rank 32, all-linear), reading the device delta:

| tokens (mb x seq) | bf16 | nf4 | penalty |
|---|---|---|---|
| 512 | 1820 MiB | 1720 MiB | **-100 MiB** (nf4 cheaper) |
| 8,192 | 2536 MiB | 2868 MiB | +332 MiB |
| 32,768 | 6190 MiB | 8466 MiB | **+2276 MiB** |

**The trade inverts.** nf4 buys back ~512 MiB of weights on this model, so
below roughly 2,700 tokens per micro-batch it is a net win and above it a net
loss. "Quantize to make it fit" can make long-context training *stop* fitting.

`bitsandbytes`' `MatMul4Bit.backward` workspace is directly visible in the
allocator trace — 76 MiB at the 8k corner, under
`_functions.py:backward <- function.py:apply`. The rest shows up as activations
(+309 MiB) and pre-window allocations (+381 MiB).

**A hypothesis recorded here earlier was wrong and is retracted.**
`prepare_model_for_kbit_training` calls `enable_input_require_grads()` only when
`use_reentrant` is absent or true — and the framework passes
`gradient_checkpointing_kwargs={"use_reentrant": False}` on *both* the quantized
and unquantized paths (`base.py:4290`, `:3301`, `:3310`). So it is never called
here, and checkpointing is identical either way. That was not the mechanism.

**Why the term is still not modelled.** Three paired points give
`-201 MiB + 78,630 B/token`, but that fit misses its own middle point by 82 MiB,
and it is one model. Shipping it would be fitting noise. The estimator instead
*warns* that quantized training is a lower bound at long context.

**What would settle it:** paired bf16/nf4 runs across a second and third model
(different hidden/intermediate ratio, e.g. SmolLM2-1.7B and Qwen2.5-3B) at four
or more token counts each. If the per-token coefficient tracks a geometric width
across models, it is real and modellable; if it does not, it is a bnb workspace
constant that needs its own lookup.

### `...@NVIDIA-A100-SXM4-40GB.longctx.json`

Qwen2.5-0.5B on an A100, 38 points, sequence lengths out to 32k.

Training is good — 5.1% mean, 10.7% worst, after re-basing onto the sweep's
median device floor — and it is the only evidence we have that the model
extrapolates to 32k (those corners land within 0.2–9.7%).

Generation is not: several points were measured before `wait_for_idle` landed
and opened on a floor left by the previous subprocess. Training points can be
re-based onto the median floor; generation points are measured with the engine
awake, so they have no baseline to subtract and cannot be repaired offline.
The re-run that would have replaced them was killed at point 12 of 21 — which
is what motivated sweep checkpointing.

### `google__gemma-4-E2B-it@...pinned.json`

Re-swept 2026-07-30 with completion length pinned, so the measurement is now
reproducible and comparable to what the estimator predicts. It is held back
because that made the error *visible*, not because the measurement is bad —
this is the honest number.

    training bias +14.5%, mean 14.5%, worst 47.0%   (was +7.8% / 8.4% / 18.6%
                                                     when completions varied)

The error scales sharply with tokens, which is the useful part:

    seq 4096, mb 8   +41% to +47%
    seq 4096, mb 1   +17%
    seq  512, mb 8   +11%

At the worst corner (seq 4096, mb 8, group 16, rank 64) the card peaked at
**40170 MiB of 40960** — essentially an OOM — against a 21140 MiB prediction.
Qwen2.5-1.5B at *identical* knobs used 13380 MiB, so Gemma costs **2.9x** for
the same workload.

The unexplained 18.8 GiB is roughly five blocks' worth of activations where the
model charges one, which points at gradient checkpointing not applying at the
granularity assumed — plausibly Gemma 4's per-layer structure (altup/laurel,
the PLE path) not being wrapped as one checkpoint segment per decoder layer.

**Next step:** allocator snapshot at seq 4096 / mb 8 with the 2M trace cap, and
count how many distinct `decoder_layer` frames hold live tensors at the peak.
One means checkpointing works as modelled; five means it does not.

Note the other three models re-swept at the same time went *in*, and improved:
Qwen2.5-0.5B +0.9% bias, Qwen2.5-1.5B +2.4%, OLMoE +4.9% (worst 13.5% -> 7.8%).

## The worst training points: five explanations ruled out — 2026-08-03

Across the ten curated profiles the estimator sits at **2.66% mean / 12.80%
worst** on 203 training points (generation is far better, ~0.3–1%). The worst
points cluster at `seq_len=512`, high LoRA rank, on the small Qwen models, and
are all over-predictions. None of the obvious causes survives contact with the
full dataset. Regressing the post-calibration residual on each candidate:

| candidate basis | R² |
|---|---|
| optimizer moments (per trainable param) | 0.001 |
| logit tile (`chunk_rows x vocab x 4`) | 0.060 |
| no-grad instant binding x logit tile | 0.002 |
| activations | 0.017 |
| base weights | 0.147 |
| **all six together** | **0.166** |

All six leave residual scatter at 316 MiB against 346 MiB unexplained, so
there is no single missing term. Also ruled out:

- **Fixture vintage.** The corpus mixes formats, but old-format profiles are
  marginally *better* (2.46% vs 2.79% mean, 9.09% vs 12.80% worst), so
  methodology drift is not the driver.
- **A constant absolute floor.** Mean absolute residual spans 108→647 MiB
  across profiles whose totals span 3478→23380 MiB — a 6.0x spread against
  6.7x, correlation +0.756. The error is proportional, not a floor, and
  relative error is uniform at 1.5–3.6% over a 6.7x size range.

The optimizer-moment story is worth spelling out because it is seductive and
wrong. `torch.optim` really does allocate `exp_avg`/`exp_avg_sq` inside the
first `step()`, so a fresh agent's first update — which is what the harness
measures — genuinely lacks them. On one profile the overshoot at r=8 and r=64
implies a slope of exactly 8 bytes per trainable parameter, which is exactly
two fp32 moments. But that slope was fitted through **two points on one
model**, and two points always define a line: across all 203 it explains
R²=0.001, and subtracting the moment charge makes the corpus *worse*
(2.66% -> 3.85% mean) because it helps high-rank points and hurts the rest.

**Next step, and it needs a GPU.** The remaining error is either outlier noise
or a bias too small to separate from it at n=1 per point. Distinguish them by
re-measuring one worst point (Qwen2.5-0.5B, seq 512, mb 8, group 4, rank 64 —
+12.8% on the A100, +11.5% on the L4) five or more times: if the spread covers
the error, it is noise and the honest fix is to widen the stated band rather
than add terms. If the point is tightly reproducible, it is a real bias and an
allocator snapshot there will name it.

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

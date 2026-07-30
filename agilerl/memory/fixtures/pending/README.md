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

### What the gap is — characterised 2026-07-30

**It is not a dequantized weight copy.** Regressing the gap against tokens over
all 21 points:

    gap = 361 MiB  +  43,200 bytes/token        (R^2 = 0.78)

A whole-model bf16 dequant copy would be 682 MiB and *flat*; one layer's worth
would be 28 MiB and flat. Neither matches — the term is dominated by a
per-token component, i.e. it is activation memory.

**The mechanism is in `prepare_model_for_kbit_training`**, which does two
things beyond quantizing:

1. Upcasts every non-`Params4bit` parameter to fp32 — layernorms and the
   lm_head. *Already modelled*, via `weight_bytes(..., kbit_prepared=True)`.
2. Calls `enable_input_require_grads()`, putting the **embedding output** into
   the autograd graph. This is the unmodelled one. Under LoRA-only training
   nothing upstream of an adapter needs a gradient, so those activations are
   freed as the forward proceeds; once the embedding requires grad the whole
   chain is retained instead. That is a per-token cost, which is what the
   regression sees.

**Why it is not modelled yet.** Pinning the coefficient means knowing which
tensors flip from freed to retained, and the data here cannot settle it: these
21 points were measured *before* `wait_for_idle` landed, so they carry the
contaminated-floor noise (hence R^2 = 0.78, and a gap ranging 190–2278 MiB).

**The experiment to run** (needs a GPU; both zones were stocked out when this
was written): sweep Qwen2.5-0.5B at nf4 with the drain-wait in place, and take
an allocator snapshot at one corner with and without `enable_input_require_grads`.
The difference between those two snapshots is the term.

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

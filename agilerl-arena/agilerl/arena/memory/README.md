# GPU memory estimation for LLM RL

A first-principles model of peak GPU memory occupancy for the LLM RL stack,
as one closed-form function:

```
peak(model_spec, device_spec, knobs) -> {component: bytes}   # per phase
```

Everything is derived from three inputs: the training manifest (the same
document a submission carries), the resource class the run would be scheduled
on, and the checkpoint's own `config.json` geometry — plus a small table of
measured per-device constants (`MEASURED_CUDA_CONTEXT_BYTES` in `specs.py`,
`ENGINE_PROCESS_OVERHEAD_BYTES` in `formulas.py`). **There is no per-model
profiling step and no fitted correction** — a new model needs nothing but its
config. `manifest.py` is the front door: it turns the manifest and the
resource class into the estimator's working `RunConfig`, so the gate reads
exactly the knobs the run would train with, never a hand-assembled copy.

### Geometry is not the whole checkpoint

`config.json` describes the decoder; a checkpoint can hold considerably more
than the decoder. Gemma 4's per-layer-embedding tables and its vision/audio
towers are 5.0% of E2B and 3.7% of E4B, and no reading of the config recovers
them exactly — `multimodal_tower_params` assumes plain transformer encoders
while the real towers are convolutional.

So `ModelSpec.n_params` carries the checkpoint's own parameter total when the
caller has it (`arena memory estimate` reads it from the Hub's safetensors
metadata — still metadata, still no weight download), and `param_counts` books the
difference to `ParamCounts.unattributed`. Weight bytes then come out exact
rather than analytic. It is deliberately *not* spread across the modelled
groups: bitsandbytes and PEFT dispatch on decoder `nn.Linear` modules and
reach none of it, so inflating `attention`/`mlp` to make the total agree would
corrupt the quantized and LoRA terms to fix the weight term.

Above 2% the estimate says so, because the activation, KV and LoRA terms are
still derived from the geometry and are correspondingly low. Six of the eight
validated checkpoints reconcile to within 0.006% and never see the warning.

Seven modules, and the only runtime dependencies are what `agilerl-arena`
already costs (pydantic, click, pyyaml) — so the same calculation runs in the
CLI, in a backend service, and client-side in a browser, without installing
the training stack. That is why it lives in `agilerl-arena` rather than
`agilerl`: memory estimation and manifest validation are the two things every
caller needs before a run exists, and `agilerl` depends on the arena package,
not the other way round.

`huggingface-hub` and `transformers` are needed only to resolve a *model id*
(geometry plus the checkpoint's parameter count) and are imported lazily inside
the resolving functions — `arena memory estimate --config path/to/config.json`
works with neither installed. Install them with `agilerl-arena[hub]`.

| module | role |
|---|---|
| `specs.py` | parse `config.json` -> geometry; knob and device schemas |
| `manifest.py` | manifest + resource class -> the estimator's `RunConfig` |
| `formulas.py` | the arithmetic (parameter counts, KV, activations, tiles) |
| `estimator.py` | assemble the two phase bars |
| `advice.py` | rank the cheapest knob changes when a bar is over budget |
| `solver.py` | invert one knob: largest value that still fits |
| `cli.py` | pre-submission gate (`arena memory estimate`) and knob solve (`arena memory solve`) |

`arena memory solve KNOB` holds every other input fixed and binary-searches
one field. That is the serving question — how long a context an L4 can
honour — and the same entry point later picks a micro-batch or a concurrent
sequence cap.

```
arena memory solve max_model_len --inference --gpu "NVIDIA L4" \
    --model Qwen/Qwen2.5-7B-Instruct
```

`--inference` is a dedicated serving GPU (utilization 0.9, 8 sequences, no
trainer residual). `--max-num-seqs 1` is the longest single-request context.
Without `--inference`, pass a training manifest and the solve is that run.
Invertible knobs: `max_model_len`, `max_num_seqs`, `micro_batch_size_per_gpu`.

### Why there is no calibration layer

There was one, fitted per (model, device) against 406 measurements, and it was
removed. It bought about a point of mean training error on models already
measured (2.66% against 3.69%) and **made things worse on unseen ones**:
leave-one-model-out put the pooled coefficients at 6.8–9.2% against 4.8% for
the bare formulas. For a platform that must size arbitrary checkpoints, that
is a liability, not an asset. Accuracy since then has come from finding
missing physical terms, not from fitting.

The measurement rig that found those terms lives in the hub's
`scripts/memory_profiling`, together with the measurements it collected and a
`validate` entry point that replays them through the core on CPU. It is how
the model was built and how a formula edit gets checked for drift, but it is
not something a caller imports and it does not ship in this package. When a
new model is added to the platform, the rig's `NEW_MODEL_PROTOCOL.md` is the
process for deciding whether this estimator already covers it, and what to
measure when it does not.

## Why two independent bars

Training and generation never hold peak memory simultaneously, even
colocated on one device, and **only one base copy is ever GPU-resident**.
Measured across one rollout+learn cycle (Qwen2.5-1.5B, 3.01 GiB of weights,
L4, vLLM 0.23):

| stage | device used | trainer base |
|---|---|---|
| after construct (engine asleep) | 4.05 GiB | on GPU |
| after wake (rollout) | 10.26 GiB | **on CPU** |
| after sleep (training ready) | **0.91 GiB** | on CPU |
| during `learn` | ~4.8 GiB | back on GPU |

The two sides alternate: vLLM sleep level 1 hands back its weights and KV
pool (−9.35 GiB at the sleep boundary) and the trainer offloads its base to
host RAM for the whole rollout (`use_memory_efficient_params`). Each side
*owns* a base copy, but the idle one lives in host RAM. So the sizing
question is two separate peaks against the same capacity, plus a small
residual for whatever the dormant side leaves behind. No phase modelling, no
`max()` over interleavings.

## Algorithms

The algorithm decides how many adapters exist, how many carry gradients, and
how wide the fused no-grad forward is — the last of which multiplies
activation memory directly, because the pass repeats the batch once per
consulted adapter.

| algorithm | fused no-grad rows | resident adapters | trained | live grad graphs |
|---|---|---|---|---|
| SFT | — (no no-grad pass) | 1 | 1 | 1 |
| GRPO / GSPO / CISPO / REINFORCE | 2 (+ reference) | 2 | 1 | 1 |
| GRPO at `beta=0` | 1 | 2 | 1 | 1 |
| DPO | 1 (sequential, not fused) | 2 | 1 | **2** |
| PPO | 3 (+ critic) | 3 | 2 | 1 |

PPO's critic is a full LoRA adapter *plus* a `PPOValueHead` of
`Linear(hidden -> 1)` held in `modules_to_save`, so it costs a third fused
row, a third resident adapter set, and a second set of gradients and
optimizer moments.

DPO is structured differently from the rollout algorithms, and the terms
follow its code path. It forwards the chosen and rejected sequences in
*separate* passes — reference logprobs are precomputed sequentially under
`no_grad`, so nothing fuses and the no-grad row multiplier is 1 — but one
loss depends on both gradient forwards, so **both graphs' saved state**
(checkpoint boundaries, loss hidden states, fp32 input casts) is resident at
the backward: twice the micro-batch on the saved side, while the per-block
recompute transient still runs one graph at a time. Its reference is the
preference baseline, always consulted regardless of `beta` (which is DPO's
temperature, not a KL coefficient). SFT and DPO also train from a fixed
dataset — no vLLM engine exists, so their generation bar is empty and no
sleeping-engine residual enters the training bar.

The grpo family is what the validation corpus measures; the ppo/sft/dpo
structure above follows the implemented code paths but carries no measured
ground truth yet, and the estimate says so in its warnings. The rig already
drives all four (`python -m memory_profiling.sweep --algorithm {grpo,ppo,sft,dpo}`).

## Distributed and orchestration terms — analytic, unmeasured

The framework's multi-GPU LLM path is DeepSpeed under accelerate, driven by
`training.training_gpus_per_agent` in the manifest. The terms are logical —
stated so they can be checked against data, not fitted from it:

- **ZeRO-2** (the default stage): optimizer state (fp32 moments + master
  copy, 12 B per trainable param) and gradient storage shard across the
  data-parallel group. Weights do not: every GPU holds the full frozen base.
- **ZeRO-3** additionally shards *parameters* — for a LoRA run the frozen
  base is where the memory is, so per-GPU weights become `base / N` plus a
  gather working set of roughly two modules in flight (the largest of the
  embedding table and one decoder layer, doubled for prefetch).
- **Data parallelism** shards the update itself: the replay buffer splits
  `batch_size x group_size` rows across learner shards, so per-GPU rollout
  tensors and the trajectories available to a micro-batch divide by N.
- **Ray orchestration** costs a measured ~50 MiB per GPU actor
  (`RAY_ACTOR_OVERHEAD_BYTES`), charged on both phases when the run is
  orchestrated — every Arena submission is. The separately observed 3.2 GiB
  gap on one orchestrated agilerl-ray job remains unexplained, lives in the
  orchestration path rather than this model, and is deliberately not
  charged; finding it is open work.

Communication buffers, bucket sizes and prefetch depth are taken at their
defaults, and no ZeRO measurement backs any of this yet — the estimate warns
whenever these terms are active.

At `beta=0` the KL term leaves the loss, so the reference row has nothing to
feed and is skipped. Two caveats the estimate carries as warnings rather
than assuming away:

- The fused no-grad pass currently builds that row unconditionally, so an
  unpatched run still pays for it. This models the intended behaviour.
- The saving often does not move the bar. Activations are
  `max(gradient pass, no-grad pass)`, and under gradient checkpointing the
  gradient pass carries a saved hidden state per layer, so it usually
  dominates — `beta=0` shrinks the no-grad pass without shrinking the peak.

## Attention backend

Which backend runs is a *software* fact, not a device capability, and the
framework resolves `auto` to FlashAttention-2 only when the `flash_attn`
package is importable — it is not in the `llm` extra, so a stock install
gets SDPA.

`eager` materialises a `rows x heads x S x S` score matrix — three copies of
it, in fact (scores, the saved softmax output, and its gradient). Nothing else
does. `flex_attention` and FlashAttention-2 tile it, and **SDPA does not build
one even on a windowed model**. Generation is untouched either way: vLLM owns
its own attention backend.

That last point was contested and is now settled by measurement. The reasoning
against it was sound — the flash kernel cannot take an explicit mask, so a
sliding window ought to push the dispatch onto the math backend — and an
allocator snapshot on Gemma 4 E2B was read as confirming it. Paired runs on
both Gemma sizes at that snapshot's own point say otherwise, each carrying its
own `eager` control so the plumbing is verified per model:

| model | `eager` | `sdpa` | `flex_attention` | eager − sdpa | sdpa − flex |
|---|---|---|---|---|---|
| Gemma 4 E2B | 26,386 | 18,760 | 18,604 | **+7,626** | +156 |
| Gemma 4 E4B | 34,642 | 27,748 | 27,770 | **+6,894** | −22 |

MiB, micro-batch 8, seq 4096, where the term predicts 6,442 MiB. `eager` sits
about 7 GiB above both on each model — so the term is real and this section's
arithmetic for it is right — while SDPA and FlexAttention land within 156 MiB
of each other. A 6.4 GiB allocation does not hide in 156 MiB.

Two consequences worth stating plainly:

- **The backend is not a memory decision unless you are on `eager`.** Choosing
  `flex_attention` for a windowed model is a correctness matter (sliding-window
  support), not a memory one, and the advice engine only offers the switch
  when the resolved backend is `eager`.
- **The `eager` term is too small by about half.** Against these six points the
  estimator under-predicts `sdpa` and `flex` by 10–11% and `eager` by 19–23%,
  so whatever `eager` costs beyond three copies is unmodelled. That is now
  visible precisely because the phantom SDPA term is gone.

Removing the phantom costs the corpus mean — 3.42% to 3.89% training, worst
11.86% to 17.77% — and that is the trade being made deliberately. The term was
cancelling a real under-prediction: Gemma 4 E2B's error at seq 4096 ran −9.4%
at micro-batch 1 and +10.2% at micro-batch 8 with it, and −15.6%/−12.1%
without. A sign flip across micro-batch is the signature of a term scaling as
`rows x S²` that should not be there; a consistent bias is something that can
be hunted. `python -m memory_profiling.residuals --decompose` is what showed
this, and it is why the estimate is now honest and slightly worse rather than
flattering and structurally wrong.

## Hybrid state-space models

Nemotron-H, Granite 4 H, Falcon-H1 and Qwen3-Next replace most attention
layers with a recurrent mixer. The distinction is the entire point of the
architecture and it is a memory distinction: **an attention layer's KV grows
with context, a recurrent layer's state does not.** Counting every layer as
attention overstates Nemotron-Nano-9B-v2's KV cache by **14x** — it keeps 4
attention layers out of 56 — and Granite 4.0 H Small's by 10x.

Four config spellings, all parsed:

| model | declares it as | attention | recurrent |
|---|---|---|---|
| Nemotron-Nano-9B-v2 | `hybrid_override_pattern` (`M`/`*`/`-`) | 4 | 27 |
| Granite 4.0 H Small | `layer_types` list | 4 | 36 |
| Qwen3-Next-80B | `full_attention_interval` | 12 | 36 |
| Falcon-H1-1.5B | *nothing* — see below | 24 | 24 |

Falcon-H1 states no layout because it has none: attention and the SSM run in
**parallel inside every block**, so it pays for both a KV cache and a Mamba
state on all 24 layers.

The state itself follows vLLM's `mamba2_state_shape`: per recurrent layer, a
causal-conv window of `(conv_dim, d_conv - 1)` and a recurrent state of
`(n_heads, d_head, d_state)`. Only `conv_dim` differs by family — Mamba-2 uses
`d_inner + 2 * n_groups * d_state`, gated-delta-net (Qwen3-Next)
`2 * k_heads * k_dim + v_heads * v_dim` — so it is resolved at parse time and
one formula covers both.

### `mamba_cache_mode`, and why `align` exists

Prefix caching needs a state to resume from at block boundaries, which undoes
the architecture's main memory advantage. vLLM's
`MambaSpec.max_memory_usage_bytes` keeps per sequence:

| mode | state blocks per sequence | when |
|---|---|---|
| `none` | 1 | prefix caching off |
| `align` | **2** | cache only the last token of each scheduler step, on a block boundary |
| `all` | `ceil(max_model_len / block_size)` | default once prefix caching is on |

On Nemotron-Nano-9B-v2 at 8k context and 16 slots that is 1.1 GiB, 2.2 GiB and
**14.1 GiB** — `all` puts the context scaling straight back, and `align` is
what buys it off at a bounded two blocks.

### Page alignment moves the block size

Both caches share one block pool, so their page sizes must match. vLLM raises
the *attention* block size until an attention page holds a Mamba page, then
pads the Mamba page to match exactly. On Falcon-H1 it logs:

```
Setting attention block size to 1568 tokens to ensure that attention page size is >= mamba page size.
Padding mamba page size by 0.71% to ensure that mamba page size and attention page size are exactly equal.
```

`aligned_kv_block_size` reproduces that 1568 from the config alone
(1,594,368 bytes of state per layer over 1,024 bytes of KV per token is 1,557,
rounded up to a multiple of 16). It matters twice: it is what makes `all` mode
merely expensive rather than absurd, and it means every sequence's KV rounds up
to a page of *hundreds to thousands* of tokens rather than 16 — which dominates
the cache when sequences are short.

### Measured against Nemotron-Nano-9B-v2

Validated on an A100 40GB against vLLM 0.26, at `gpu_memory_utilization=0.9`
and 16 sequence slots:

| quantity | model | vLLM / measured |
|---|---|---|
| Mamba page per layer | 5,316,608 B | 5,316,608 B |
| aligned attention block size | 1,312 tokens | **1,312** (logged) |
| KV bytes per token | 16,384 | 16,384 (4 attention layers) |
| weight bytes | 16,953 MiB | 17,079 MiB (−0.7%) |
| device footprint, 4k context | 37,479 MiB | 36,503 MiB (**+2.67%**) |
| device footprint, 8k context | 38,686 MiB | 37,255 MiB (**+3.84%**) |

For an architecture family the estimator had never seen, with nothing fitted
to it, +2.7% is in line with the transformer corpus (3.19% mean). Counting all
56 layers as attention would have put KV at 229,376 bytes per token and the
pool at 71 GiB on a 40 GB card.

The run also corrected a real error. The recurrent state's dtype is resolved
**per model** by vLLM, not from the config: Nemotron-H returns
`(bfloat16, float32)` and Falcon-H1 `(bfloat16, bfloat16)`. That doubles
Nemotron's page and its block size with it — predicting 672 where vLLM logged
1,312 — so `mamba_ssm_state_dtype` defaults to fp32, the larger of the two,
because a memory estimate must not read low. The rounding granularity is 32,
the attention backend's kernel block alignment, not vLLM's default 16.

Two things remain unvalidated, and the estimate still warns:

- **The 8k agreement is partly luck.** At that context the profiling peak
  drives the computed KV pool to zero while the real pool is 7,982 MiB; the
  total lands close because the clamp removes a term that was too large. The
  4k point is the honest one.
- **The sleeping residual is nondeterministic**, which two points made look
  like context scaling (1,146 MiB at 4k against 1,914 MiB at 8k). Sweeping
  four contexts refutes it: the residual takes two discrete values — 2,434 MiB
  at 2k *and* 16k, 1,686 MiB at 4k *and* 8k — with no monotone trend. Two runs
  at an *identical* config then differed by 748 MiB, which settles it. See
  "The sleeping floor is noisy".

Activation and weight terms for a recurrent *layer* still have no profiled
ground truth of their own — only the totals above constrain them.

## The sleeping floor is noisy, and it is the OLMoE residual

The rig samples a *sleeping-engine floor* per point — NVML at the moment the
training window opens — and re-bases training points onto the sweep median so
the fitted quantity is what the trainer adds, not whatever the device happened
to hold. Generation points are not re-based; their target is the raw peak.

That floor is not stable. Across a single model's stored points it spans:

| model | floor range | generation error |
|---|---|---|
| Qwen2.5-7B | 1,310–1,450 MiB | 0.68% |
| SmolLM2-1.7B | 849–925 | 3.18% |
| gemma-4-E2B | 1,316–1,480 | 5.56% |
| **OLMoE-1B-7B** | **1,388–5,646** | **3.65%** |

OLMoE's floor moves by 4.3 GiB across points of the same model on the same
device, where every other model moves by a few hundred MiB. Its signed
generation error correlates with its own point's floor at **r = −0.99**: the
higher the floor, the more the estimate reads low. Re-basing OLMoE's
generation targets on their own floors, the way training points are re-based,
halves its error — **3.65% to 1.76%**.

So the OLMoE generation residual is substantially *measurement variance*, not
a missing term. It survived a long hunt — the start-up profiling peak, an
MoE-specific dispatch factor, the CUDA-graph pool and a budget-filling
model were each tried and refuted — because none of them was the cause.

Generation is deliberately **not** re-based in response. Doing so helps OLMoE
and costs every model whose floor is stable, taking the pooled mean from
3.19% to 3.38%: the raw peak is the right target when the floor is trustworthy.
The rig now drains the trainer allocator before sampling the floor instead, on
the grounds that NVML charges reserved bytes and the rollout leaves reclaimable
segments behind. That removes one known contaminant; it is **not** shown to
remove the whole spread, and the 203 stored points predate it either way.

Part of the spread is not reclaimable at all. Probing the engine process
directly — `collective_rpc` into the worker, so the reads are its allocator and
not the driver's — `empty_cache()` there moves neither the worker's reserved
bytes nor the device's, and two runs at an identical config still land 748 MiB
apart. Whatever that is, it is pinned by vLLM's CuMem allocator and varies by
run, so no amount of draining will make the floor repeatable. A floor that
moves by ~750 MiB between identical runs is the noise level any
generation-phase claim has to clear.

## Allocator reserve

Every other term in this document is an **allocation** size. The device is
charged the PyTorch caching allocator's **reservation**, which is larger: it
rounds segments up, and it cannot always hand a freed block to a
differently-shaped request. `torch.cuda.max_memory_reserved /
max_memory_allocated` over stored training points runs 1.5–6.4%.

This is what closed the one model whose device peak the torch-visible
attribution could not explain. An allocator snapshot of Gemma 4 E2B in the
training window accounts for 12,807 MiB live; the sleeping engine measures
1,458 MiB (independently confirmed by loading the engine, sleeping it and
reading NVML: 1,430 MiB); at that run's 8.6% process-wide reserve ratio the
device should see 15,366 MiB, against 15,062 measured.

It is deliberately *not* scoped to the churning activation terms. On
leave-one-model-out the whole-torch-side scope holds a held-out optimum of
1.50–2.75% and improves 6 of 8 models; scoping it to activations alone spreads
0.00–3.75% and improves 4. Segment rounding applies to every allocation, and
the large weight tensors are the ones most often rounded up.

The constant is stack-dependent, and the 2026-08-21 A100 re-sweeps re-judged
it at **5%**. On the production ray-worker image (torch 2.11.0+cu130) the
trainer-only reserved-over-allocated ratio — reconstructed per point by
netting the engine's CuMem phantom out of the torch counters — runs 8–25%,
while the model's *allocation* arithmetic reconstructs to within a few
percent at the very corners that read low. The whole re-sweep under-read was
this constant. 0.05 is the largest value every stored fixture's drift bands
tolerate (0.06 tips SmolLM2@L4 over the 8% mean band); the pooled corpus pays
4.41% → 4.85% training mean for it, all of it on the over-prediction side,
and both under-reading re-sweeps clear the systematic-under-prediction
admission line.

**Training only.** Generation runs inside vLLM's CuMem pool, which is reserved
up front at `gpu_memory_utilization`; there is no incremental slack to charge,
and the held-out optimum is 0.0% for every one of the eight models. Adding it
there costs 3.16% → 3.72%.

This is the term the removed windowed-mask phantom had been standing in for —
but only partly. It recovers 3.86% → 3.57% mean and 17.83% → 16.36% worst; the
mask form fit better still (3.06%) and was wrong, which is the distinction the
snapshot exists to draw. Direct attribution of that snapshot found **49 MiB**
of simultaneous mask residency — three bool `(1,1,4096,4096)` masks, allocated
and freed per layer — against the 2,240 MiB the term claimed.

## Training bar

| Component | Scales with | Grounding in the framework |
|---|---|---|
| Base weights (frozen) | `P x bytes(dtype)`, or realised quantized size | trainer always holds its own full copy; nf4/int8 keep `lm_head` unquantized and upcast norms + `lm_head` to fp32 (`base.py` k-bit prep) |
| LoRA adapters | `rank`, target scope, x2 with the reference adapter, x3 for PPO | **full FT does not exist** — the framework trains adapters only |
| Gradients + optimizer state | trainable (adapter) params only | plain `torch.optim.AdamW`; DeepSpeed-config optimizer is the only alternative. LoRA-only makes this a rounding error next to the base — the opposite of the classic 16-bytes/param intuition |
| Activations | `max(grad pass, no-grad logprob pass)` | grad pass: checkpoint boundaries (`rows x S x H x L`) + one block's recompute + the `(rows, S, H)` hidden the fused-logprob autograd saves. No-grad pass: actor+reference(+critic) rows fused into one wider forward (`_fused_forward_no_grad`), micro-batched by the same per-GPU cap |
| Logit workspace | `chunk_rows x V x (act_bytes + 4)` | the fused chunked path (`llm_ops/fused_logprobs.py`) never materialises `B x S x V`. Two tiles are live under autograd — the matmul output and the fp32 cast the log-softmax makes — measured at 255.6 + 127.8 MiB against a 441-row chunk and a 151936 vocab. The no-grad pass builds no graph, so only the fp32 tile stands |
| Overhead | measured per-device constant | CUDA context (A100 501 MiB, L4 226 MiB — a 2x spread, so one constant would bias whole fleets), held rollout tensors (MB-scale) |
| Caching-allocator slack | 5% of the torch-side subtotal | the device is charged reserved, not allocated, bytes; measured 8–25% on the cu130 image, and 5% is what the drift bands tolerate. Training only — vLLM's CuMem pool is reserved up front. See "Allocator reserve" |

What is *not* a training memory knob, and worth teaching in the UI:

- **`beta`, mostly**: it never changes what is resident — the reference
  adapter is built at init either way — and it only shrinks the *peak* when
  the no-grad pass is the binding side of the `max()`, which under gradient
  checkpointing it usually is not. See the Algorithms section.
- **The reference model**: an adapter copy, not a second base. Turning off
  `use_separate_reference_adapter` saves only adapter-sized bytes and pins
  the reference to the initial policy.
- **Gradient accumulation**: changes micro-batch for a fixed effective
  batch — it is the *micro-batch* that is the memory lever.
- **Optimizer choice / full FT**: not in the framework.

## Generation bar

vLLM self-limits to `gpu_memory_utilization x total`. Inside that budget:
engine weight copy (vLLM owns its base across sleep/wake; possibly a
different variant than the trainer via `vllm_model_name_or_path`), prefill
transients for one scheduler step (`resolve_vllm_max_num_batched_tokens`
tokens), sampler buffers, the ~2 GiB CUDA-graph pool (`enforce_eager` skips
it), LoRA slots, and — as the *remainder* — the KV pool (or the exact
`kv_cache_memory_bytes` pin).

The estimator reports KV two ways: the **pool** (supply, what vLLM
allocates) and the **worst-case demand** (`concurrency x S x 2 x L x n_kv x
head_dim x bytes`). Demand above pool is a throughput cliff (preemption +
recompute), not an OOM — the widget should render it as a warning, distinct
from the red over-capacity state.

Outside the engine budget the device also carries the CUDA context and, when
colocated, the trainer residual (optimizer state stays on-device while the
base sits in host RAM).

## Validation, not calibration

Tabulating memory is infeasible (six knobs x four levels is 4096 runs per
model, and sequence length is continuous) and unnecessary: the analytic form
carries the shape. It once also carried fitted per-model constants; see "Why
there is no calibration layer" above for why they were removed.

What remains is a validation corpus: 423 measurements over 8 models
(dense, MoE, multimodal) and 2 devices, from sweeps of the same 16
corners of `seq_len x micro_batch x group_size x lora_rank` plus 5 interior
points, with completion length pinned so runs are byte-reproducible. Every
point in it is consistent with the current tree: when the fused-loss kernel
rewrite moved the loss instant, the 143 training points measured under the
previous kernel were deleted rather than gated around, and the 2026-08-21
re-sweep campaign then refilled the training axes on the production
ray-worker image — Qwen2.5-0.5B and Gemma 4 E2B replaced wholesale,
SmolLM2-1.7B gaining the A100 coverage it never had (the per-fixture record
is in the rig's git history and its README's analysis notes).

`memory_profiling.validate` replays all of them through the current core on
CPU. Constants drift with framework changes — anything touching the fused
kernels, checkpointing or the vLLM wiring moves them — and this is the only
thing between a formula edit and silent wrongness, since the estimator keeps
returning plausible numbers either way.

## Porting notes

- Component `key` strings are the contract with the widget and the
  platform backend; change them only with a `schema_version` bump.
- `estimate_run(...).model_dump(by_alias=True)` is exactly the widget
  payload; `advise()` is the "you are N GiB over, cheapest fixes" list,
  computed by re-running the estimator under candidate knob mutations so it
  can never disagree with the bars.

## Measuring memory under vLLM is a minefield — read this before touching the harness

Getting a trustworthy number here is harder than the arithmetic it feeds.
Every item below is a bug this harness hit and now guards against; the first
two produced a confident, self-consistent, **wrong** result for a whole
sweep, so treat them as load-bearing.

- **Absolute `torch.cuda` stats are meaningless under CuMem.** vLLM's sleep
  allocator releases physical pages that torch still counts. Measured at the
  same instant with the engine asleep: torch reports **9.33 GiB allocated**
  while the device shows **0.91 GiB used**. Only the *delta* across a window
  is meaningful — the phantom is constant, so it cancels in a subtraction and
  not in an absolute reading. The training peak uses
  `max(nvml_poll, window_baseline + (torch_max_reserved − torch_reserved_at_entry))`.
- **`learn()` sleeps the engine itself**, so a measurement window opened
  around `learn` samples the still-awake engine and reports the *rollout*
  footprint as the training peak. Sleep explicitly first (idempotent), then
  open the window. Getting this wrong overstated training by **5×**
  (12.08 vs 2.34 GiB on a 0.5B).
- **Two wrong measurements can agree.** The two bugs above both inflated the
  training figure and landed within a few percent of each other, which read
  as corroboration. Cross-check any device-level number against a physical
  sanity bound — a 0.5B model cannot need 12 GiB to train a LoRA adapter.
- **One process per point.** vLLM's CuMem allocator is process-global and
  permits one engine per process, so `run_sweep` spawns a fresh subprocess
  per point. A single-process loop dies on the second point with "CuMem
  allocator can only be used for one instance per process".
- **NVML polling can still miss a spike.** The 10 ms poll is a lower bound;
  the torch-growth term above covers the brief backward peak between samples.
- **flashinfer JIT needs `curand.h`.** On a box without full CUDA toolkit
  headers, set `VLLM_USE_FLASHINFER_SAMPLER=0`.

## What the measurements taught the model

Every correction below came from a sweep disagreeing with the analytic core,
and each one is a *modelling* fix — verified by checking that it also
improves models it was not derived from.

- **Generation is bounded by prompt tokens in flight, not by the context
  budget.** vLLM sizes its KV pool as "budget minus the start-up profiling
  step", and that step is transient. So a 4096-context engine sits ~1.9 GiB
  *below* the same engine at 512 context — its larger profiling step bought
  it a smaller pool. Modelling this (`max_prompt_len`) took generation from
  12.4% worst-case to 1.7%, and the *uncalibrated* error from 5.5% to 0.7%:
  the analytic model became right, rather than the fit covering for it.
- **LoRA adapters are fp32 even on a bf16 base**, because PEFT defaults to
  `autocast_adapter_dtype=True`. The measured cost of raising rank 8 -> 64 is
  ~20 bytes per added parameter, which only resolves as 4 (actor) + 4
  (reference) + 4 (gradient) + 8 (two AdamW moments).
- **Micro-batches are capped by available trajectories.** Requesting a
  micro-batch larger than the update has rows costs nothing, so `group_size`
  bounds the gradient batch.
- **A per-block transient peaks at the widest block, not the mean one.**
  Gradient checkpointing recomputes one block at a time and the engine runs
  layer by layer, so on a heterogeneous stack (Gemma 4's double-wide-MLP
  KV-shared layers, its 512-wide global heads) the binding block realises
  the maxima, and `block_recompute_bytes` and the one recomputed block's
  LoRA casts charge `peak_mlp_width_factor`/`peak_qkv_dim`. Parameter
  counts keep the means — a sum over layers is what they integrate.
  Uniform stacks are unchanged; E2B's old fixture alone moved −11.1% →
  −5.7% signed on this.

Two things that did *not* work, recorded so they are not retried:

- **Non-negative least squares for the residual.** "Unmodelled memory can't
  be negative" sounds principled but is wrong — the analytic core can
  over-count too, and a negative coefficient is how the fit corrects that.
  Forcing non-negativity made the generation holdout 8x worse.
- **Fitting the colocated engine floor.** It was an artifact (see above); the
  floor does not exist.

## Accuracy

No fit is applied, so every stored point is a held-out prediction: the
replay of the full corpus through the closed-form core is the accuracy
claim. Mean absolute error per fixture (2026-08-21 core):

| model | device | training | generation |
|---|---|---|---|
| Qwen2.5-0.5B-Instruct | A100 | 4.5% | 1.3% |
| Qwen2.5-0.5B-Instruct (sft / dpo / nf4) | A100 | 5.3% / 4.9% / 4.9% | — / — / 1.6% |
| Qwen2.5-0.5B-Instruct | L4 | 4.1% | 2.5% |
| Qwen2.5-1.5B-Instruct | A100 | 2.8% | 2.8% |
| Qwen2.5-1.5B-Instruct | L4 | 3.6% | 2.8% |
| Qwen2.5-3B-Instruct | A100 | 4.6% | 2.0% |
| Qwen2.5-7B-Instruct | A100 | 5.8% | 0.7% |
| SmolLM2-1.7B-Instruct | A100 | 6.2% | 1.8% |
| SmolLM2-1.7B-Instruct | L4 | 7.7% | 3.2% |
| OLMoE-1B-7B | A100 | 3.0% | 1.6% |
| Gemma 4 E2B (sdpa) | A100 | 1.5% | 7.0% |
| Gemma 4 E4B | A100 | 5.5% | 7.3% |

Gemma is the hardest architectural case in the set — MQA (a single KV head),
`head_dim` 256 against a 512-wide global head, a 262k vocab, double-wide
MLPs on the KV-shared layers, and 28 of 35 layers windowed — so it exercises
the geometry paths the Qwen models never touch. Its training is now the best
fixture in the corpus and its generation the worst: the engine on the
production image decouples from the `gpu_memory_utilization` budget (see the
open list), which no budget arithmetic reproduces.

The knob axes span sequence length, micro-batch, group size, LoRA rank *and*
`gpu_memory_utilization` (0.21→0.56 on the A100 SmolLM2 sweep, 0.30/0.60
holdouts on the L4 fixtures), so the corpus is not pinned to the engine
budget it was measured under.

The worst single point runs higher (+18% over-prediction on SmolLM2's
backward-bound mb8 g16 corner). The extreme corners cannot all be captured
by a closed form at once; `WORST_BAND` exists as a drift alarm rather than
an accuracy claim, and the residual worst points now sit on the
over-prediction side.

## What the totals do not tell you

The sweep compares one predicted peak against one measured peak, so it
validates sums, not splits — the bars could each be wrong and cancel. An
allocator snapshot (`--snapshot`, then
the rig's `snapshot` tool) attributes the *peak instant*
per call site. On Qwen2.5-0.5B at seq=4096, micro-batch 8, group 16:

| component | predicted | observed |
|---|---|---|
| base weights | 0.92 | 0.95 |
| gradients + optimizer | 0.05 | 0.03 |
| activations | 2.55 | 3.49 |
| adapters (LoRA-path buffers) | 0.03 | 0.59 |
| logit workspace | 0.50 | 0.00 |

Weights and optimizer state are essentially exact, which localises the
unexplained superlinear residual at large corners to the activation and
LoRA-forward path rather than leaving it as a mystery in the intercept. It
also exposes an over-count: the chunked logit tiles are not live at the peak
instant, so charging two full tiles adds ~0.5 GiB the run never pays at the
same time as the activation peak.

Two things make the snapshot awkward, both worth knowing before reaching
for it: recording must be scoped to the training window (otherwise the peak
belongs to vLLM's start-up KV allocation), and it must be read from the
event trace rather than the final block list (by the time `learn` returns,
every activation has been freed).

## Status

Calculation core, advice engine and the CLI gate are implemented and tested
(CPU-only tests). **The corpus rule: every point in it was measured under
(or is consistent with) the current working tree.** The estimator models
exactly what `oss/agilerl/agilerl` on this branch does — no per-fixture
gating, no version switches; a fixture's `framework_versions` is
provenance only. When the fused-loss kernel was rewritten (hoisted fp32
head copy, fp32 tile pair in the chunk backward, forward cast-skip —
`algorithms/core/llm_ops/fused_logprobs.py`), the 143 stored training
points whose measured targets reflect the *previous* kernel — identified
as the points whose replay shifts more than 2pp of target between the two
physics — were deleted from the corpus rather than gated around; the
2026-08-21 re-sweep campaign then refilled the training axes on the
production ray-worker image, pre-registered blind predictions first (the
per-fixture accounting lives in the rig's git history and its README's
analysis notes).

Accuracy against the 178 training and 245 generation measurements now
stored, over 8 models and 2 devices — including the nf4 sweep, the
cluster-collected SFT and DPO sweeps, and the three re-sweeps (21 training
+ 21 generation A100 points each, blind-scored against pre-registered
predictions before any model change: Qwen2.5-0.5B −3.5% signed training,
SmolLM2-1.7B +0.8%, Gemma 4 E2B −4.2% training / −4.5% generation):

| phase | n | mean | worst |
|---|---|---|---|
| training | 178 | 4.63% | 18.0% (over) |
| generation | 245 | 2.88% | 13.4% |

The blind scores are the production-path claim: they were computed the way
the CLI computes — `config.json` parsed fresh, weights reconciled against
the checkpoint's parameter count, no realised sizes — and committed before
the sweeps ran.

Two of the three re-sweeps landed under the −2% systematic-under-prediction
admission line, and closing them changed the model twice (see "Allocator
reserve" for the 5% re-judgement, and "What the measurements taught the
model" for the widest-block rule). After both, the re-swept fixtures replay
at 4.5% / 6.2% / 1.5% training mean — Gemma 4 E2B, the family whose
"checkpointing-granularity gap" once ran 26.9 GiB against 17 predicted, is
now the best-predicted training fixture in the corpus — and the pooled
worst point moved from a 15.3% under-prediction to an 18.0%
over-prediction, which is the direction the gate can afford.

Generation's earlier history stands: the MoE dispatch copies are excluded
engine-side (vLLM's fused-MoE routes in-kernel; curated OLMoE generation
1.56% mean), and the CUDA-graph pool constant follows vLLM's own logged
40–93 MiB rather than a 2 GiB guess — at 2 GiB it drove the computed pool
negative on seven measured-fine configurations and raised false OOM
warnings against them.

Two MoEs the estimator had never seen (granite-3.1-1b-a400m at 32 experts and
3b-a800m at 40, against OLMoE's 64) were measured to test that. Built the way
the CLI builds them — live `config.json`, checkpoint parameter count, no
constant touched by these models — the training bar came out **13.31% mean and
22.81% worst, every point low**, against 3.6% on the corpus. Generation was
fine. That gap was the expert gather outliving its checkpoint block, and
closing it took the held-out MoEs to 7.26% / 12.86% for nothing on the corpus.

The rest is not a knob effect. Sliced by `seq_len`, `micro_batch`,
`group_size` or `lora_rank` the signed error is flat to within a point;
sliced by model it still spreads — SmolLM2's backward instant over-charges
(255 KiB per grad token measured against 316 charged) where Qwen's matches
(156/160) — and that spread is per-model allocation behaviour the closed
form does not carry. Fitting it would rebuild the calibration layer that was
removed. The remaining attributions need allocator timelines at named
corners, not more sweeping.

Open, in rough priority order:

- **Quantized training is now validated end to end, on one model.** The
  terms landed in three steps. Paired bf16/nf4 runs at four token counts on
  Qwen2.5-0.5B/1.5B/3B and Gemma 4 E4B found the missing constant — k-bit
  preparation upcasts a *tied* `lm_head`, which is the embedding table — and
  the per-token bitsandbytes workspace
  (`40 KiB/token + 1.5 bytes per (n_layers x hidden)`, R² 0.982, held to
  12.9 KiB/token leave-one-model-out). Replaying the 21-point nf4 sweep then
  exposed the last piece: a profiled *realised* weight size measures the
  load, and k-bit prep's upcasts happen after it, so `weight_bytes` now adds
  them on top of a realised quantized size instead of assuming them inside
  it (validate.py also replayed quantized points against the bf16 variant —
  fixed). The nf4 sweep now scores **4.3% mean / 10.9% worst / +0.2% signed**
  against 3.4% for the same model unquantized, and is promoted into the
  corpus. Still one model on one device; the 1.5B/3B/E4B paired points check
  the *delta*, not the absolute bar.
- **The bf16/nf4 trade inverts, and where depends on model size.** nf4 buys
  weight bytes and costs activation bytes. Measured crossovers: ~3,600 tokens
  on Qwen2.5-0.5B, ~14,400 on the 1.5B, ~19,700 on the 3B — a bigger model has
  more weight saving to spend before the per-token cost exhausts it. On
  Gemma 4 E4B nf4 is *more expensive at every context measured*, because its
  262k-vocab tied embedding is over half the parameters and none of it
  quantizes. Quantizing to make a run fit can make a long-context run stop
  fitting, and on a wide-vocab tied model it never helps at all.
- **Gemma 4's training gap is closed; its engine is now the open item.**
  The SDPA timelines count **one** live decoder block at the training peak,
  exactly as modelled, and the re-sweep decomposed what remained into two
  named pieces — the stack-level allocator reserve (re-judged at 5%, see
  "Allocator reserve") and the widest-block rule (`peak_mlp_width_factor` /
  `peak_qkv_dim`: E2B's double-wide-MLP global-attention blocks are what a
  per-block transient binds on). E2B training replays at 1.5% mean / 3.7%
  worst. Its **generation** is the residual problem: on the production
  ray-worker image the engine's measured residency does not track
  `gpu_memory_utilization` at all — flat at weights + ~3.9 GiB across
  budgets 0.28–0.44, i.e. 2.3 GiB *above* the budget at the low end —
  where the same engine under the local vLLM 0.26 filled whatever it was
  given, and engine *construction* transiently peaks at 1.76× the serving
  bar (24 GiB at 16 slots × 4096). Three independent instruments agree
  (generation NVML, the training-window CuMem phantom, the startup phase).
  The bar warns on every multimodal checkpoint; deciding *which* engine
  allocation does this (encoder cache and multimodal profiling are the
  candidates) needs an engine-side trace. OLMoE's remaining gap is
  training-side only: mb8/g16 over-prediction from the known
  `MOE_GATHER_RESIDENT_BLOCKS = 4` trade — corpus-neutral and
  held-out-optimal against granite, so the constant stands.
- **DP / ZeRO-3 / Ray terms are analytic and unmeasured** (see "Distributed
  and orchestration terms"): the shapes are stated so a measurement can
  falsify them, and the 3.2 GiB orchestration observation is still
  unexplained.
- **sft and dpo are measured and in the corpus; ppo is half-measured.** The
  SFT and DPO sweeps blind-scored at −33% and −25% signed, and closing them
  named three mechanisms, each verified against this tree's source: the
  engine-less trainer's own library workspaces (583 MiB,
  `TRAINER_LIB_OVERHEAD_BYTES`); the fp32 `(vocab, hidden)` lm_head copy
  the fused-logprob loop hoists (`fp32_lm_head_operands`,
  `algorithms/core/llm_ops/fused_logprobs.py`), plus two fp32 chunk tiles
  at the loss instant (the SFT allocator timeline shows the peaks
  decompose as exactly 519 + n x 256 MiB on Qwen2.5-0.5B, and the
  Gemma-4-E2B timeline reads 2,069 MiB against 2,048 predicted — the same
  kernel, every algorithm); and PEFT's fp32 input casts being
  *recompute-only* wherever the forward runs under `_amp_ctx` — the cast
  skip (`_lora_input_cast_ctx`, `algorithms/core/base.py`) covers the
  original forward, checkpoint recompute runs after the flag is restored,
  so one graph's single-block casts reappear during backward and DPO's
  two live graphs never double them. PPO's generation points are well
  predicted (2.90% mean) once the model knows
  `use_memory_efficient_params=False` keeps the whole trainer resident
  through the rollout; its *training* points are parked — the rig sent one
  prompt and PPO samples one completion per prompt, so every update had one
  trajectory and the batch axes were never exercised.
- **First-step peaks are not steady-state peaks.** Every fixture is measured
  at `warmup_steps=0`, where the Adam moments do not exist yet: the PPO
  sweep's measured rank slope is ~230 MiB r8→r64 against ~1.1 GiB predicted,
  and the Qwen re-sweep shows the same signature at token-heavy corners
  (~12 B per added adapter parameter measured against the 20 B steady state
  charges). The estimator deliberately predicts steady state —
  over-prediction is the safe direction — but the split should be closed by
  re-measuring one sweep at `--warmup-steps 1`, not by softening the terms.
  The warmup-1 probe was pre-registered in the re-sweep campaign and cut
  for cluster time; it remains the cheapest decisive run if the cluster is
  ever up for other reasons.
- **vLLM's fused-MoE workspace is unmodelled.** With the dispatch copies
  gone the remaining OLMoE generation error is −0.5% to −5.3%, worst where
  `max_num_batched_tokens` is largest — the shape of the kernel's chunked
  intermediate caches (`VLLM_FUSED_MOE_CHUNK_SIZE`). The estimate warns on
  MoE generation instead of guessing the chunk size.
- **SmolLM2's backward instant is over-charged.** At the mb8/g16 s4096
  corner its trainer-only allocation reconstructs to 11.2 GiB against 13.6
  modelled (+18% on the total after the reserve trade) — an
  allocation-side over-charge in the recompute/cast terms that Qwen does
  not share. Over-prediction, inside the bands; attribution needs an
  allocator timeline at that corner, not a discount.
- **The purge-thinned fixtures that were not re-swept still hold 3–5
  training points each** (Qwen2.5-1.5B/3B/7B, the L4 Qwen2.5-0.5B, Gemma 4
  E4B) — the backward-bound corners only. The re-sweeps refilled
  Qwen2.5-0.5B, SmolLM2 and E2B on A100; for the rest the per-fixture mean
  drift band applies only to phases holding 8+ points, with the worst-point
  and pooled-mean bands guarding everything. E4B's generation points are
  additionally 0.26-engine vintage where production runs 0.24 — the two
  stacks demonstrably diverge on its sibling — so treat E4B generation
  accuracy claims as indicative.
- **The dense long-context generation under-read did not reproduce on the
  production stack.** The historical −13.4% cluster (SmolLM2-1.7B on L4 at
  16 slots × 4096) survives only in the L4 fixture; the A100 re-sweep at
  the same knobs and gmu up to 0.56 reads −0.4..−0.8%. The L4 points stay
  as measured; whether the gap was the engine vintage or the device class
  is undecidable without an L4 re-sweep, which nothing currently justifies.
- The Arena widget itself, on this same core.

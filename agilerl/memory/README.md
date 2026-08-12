# GPU memory estimation for LLM RL

A first-principles model of peak GPU memory occupancy for the LLM RL stack,
as one closed-form function:

```
peak(model_spec, device_spec, knobs) -> {component: bytes}   # per phase
```

Everything is derived from three inputs: the checkpoint's own `config.json`
geometry, the run's knobs, and a small table of measured per-device constants
(`MEASURED_CUDA_CONTEXT_BYTES` in `specs.py`,
`ENGINE_PROCESS_OVERHEAD_BYTES` in `formulas.py`). **There is no per-model
profiling step and no fitted correction** — a new model needs nothing but its
config.

Five modules, no torch dependency, so the same calculation runs in the CLI,
in a backend service, and client-side in a browser:

| module | role |
|---|---|
| `specs.py` | parse `config.json` -> geometry; knob and device schemas |
| `formulas.py` | the arithmetic (parameter counts, KV, activations, tiles) |
| `estimator.py` | assemble the two phase bars |
| `advice.py` | rank the cheapest knob changes when a bar is over budget |
| `preflight.py` | CLI entry point (`python -m agilerl.memory.preflight`) |

### Why there is no calibration layer

There was one, fitted per (model, device) against 406 measurements, and it was
removed. It bought about a point of mean training error on models already
measured (2.66% against 3.69%) and **made things worse on unseen ones**:
leave-one-model-out put the pooled coefficients at 6.8–9.2% against 4.8% for
the bare formulas. For a platform that must size arbitrary checkpoints, that
is a liability, not an asset. Accuracy since then has come from finding
missing physical terms, not from fitting.

The measurement rig that found those terms lives in `tools/memory_profiling`,
outside the shipped package. It is how the model was built and how it is
guarded against drift (`python -m tools.memory_profiling.validate` replays all
406 stored measurements through the current core, no GPU), but nothing a
caller imports.

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

| algorithm | fused rows | resident adapters | trained |
|---|---|---|---|
| SFT | 1 (actor) | 1 | 1 |
| GRPO / GSPO / CISPO / REINFORCE / DPO | 2 (+ reference) | 2 | 1 |
| GRPO at `beta=0` | 1 | 2 | 1 |
| PPO | 3 (+ critic) | 3 | 2 |

PPO's critic is a full LoRA adapter *plus* a `PPOValueHead` of
`Linear(hidden -> 1)` held in `modules_to_save`, so it costs a third fused
row, a third resident adapter set, and a second set of gradients and
optimizer moments.

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

Measured on Gemma 4 E2B (28 of 35 layers windowed, 8 heads) at seq 4096,
micro-batch 8, where a materialised score matrix would be 2.15 GiB:

| backend | training peak | generation peak |
|---|---|---|
| `sdpa` | 14.78 GiB | 17.52 GiB |
| `flex_attention` | 15.03 GiB | 17.48 GiB |

SDPA came out *below* FlexAttention, so on torch 2.11 / transformers 5.11 it
stays O(S) even for a windowed mask, and only `eager` materialises scores.
Generation is untouched either way — vLLM owns its own attention backend.

The practical reading: the two O(S) backends are within 2% on memory, so
there is no memory argument for taking on `flash_attn` as a dependency. The
case for it rests on Hopper-class throughput and nothing measured here.

## Training bar

| Component | Scales with | Grounding in the framework |
|---|---|---|
| Base weights (frozen) | `P x bytes(dtype)`, or realised quantized size | trainer always holds its own full copy; nf4/int8 keep `lm_head` unquantized and upcast norms + `lm_head` to fp32 (`base.py` k-bit prep) |
| LoRA adapters | `rank`, target scope, x2 with the reference adapter, x3 for PPO | **full FT does not exist** — the framework trains adapters only |
| Gradients + optimizer state | trainable (adapter) params only | plain `torch.optim.AdamW`; DeepSpeed-config optimizer is the only alternative. LoRA-only makes this a rounding error next to the base — the opposite of the classic 16-bytes/param intuition |
| Activations | `max(grad pass, no-grad logprob pass)` | grad pass: checkpoint boundaries (`rows x S x H x L`) + one block's recompute + the `(rows, S, H)` hidden the fused-logprob autograd saves. No-grad pass: actor+reference(+critic) rows fused into one wider forward (`_fused_forward_no_grad`), micro-batched by the same per-GPU cap |
| Logit workspace | `chunk_rows x V x (act_bytes + 4)` | the fused chunked path (`llm_ops/fused_logprobs.py`) never materialises `B x S x V`. Two tiles are live under autograd — the matmul output and the fp32 cast the log-softmax makes — measured at 255.6 + 127.8 MiB against a 441-row chunk and a 151936 vocab. The no-grad pass builds no graph, so only the fp32 tile stands |
| Overhead | measured per-device constant | CUDA context (A100 501 MiB, L4 226 MiB — a 2x spread, so one constant would bias whole fleets), held rollout tensors (MB-scale), allocator slack |

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

What remains is a validation corpus: 406 measurements over 8 models
(dense, MoE, multimodal MoE) and 2 devices, each sweeping the same 16 corners
of `seq_len x micro_batch x group_size x lora_rank` plus 5 interior points,
with completion length pinned so runs are byte-reproducible.

    python -m tools.memory_profiling.validate

replays all of them through the current core on CPU. Constants drift with
framework changes — anything touching the fused kernels, checkpointing or the
vLLM wiring moves them — and this is the only thing between a formula edit and
silent wrongness, since the estimator keeps returning plausible numbers either
way. `tools/memory_profiling/test_fixtures.py` runs it in CI.

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

Two things that did *not* work, recorded so they are not retried:

- **Non-negative least squares for the residual.** "Unmodelled memory can't
  be negative" sounds principled but is wrong — the analytic core can
  over-count too, and a negative coefficient is how the fit corrects that.
  Forcing non-negativity made the generation holdout 8x worse.
- **Fitting the colocated engine floor.** It was an artifact (see above); the
  floor does not exist.

## Accuracy

Held-out knob combinations — the accuracy claim, since "will my run fit"
depends on configurations that were never measured:

| model | device | training | generation |
|---|---|---|---|
| Qwen2.5-0.5B-Instruct | L4 | 6.6% | 0.5% |
| Qwen2.5-0.5B-Instruct | A100 | 6.9% | 0.4% |
| Qwen2.5-1.5B-Instruct | L4 | 2.8% | 0.9% |
| Qwen2.5-1.5B-Instruct | A100 | 2.7% | 0.4% |
| Qwen2.5-3B-Instruct | A100 | 0.7% | 1.0% |
| Qwen2.5-7B-Instruct | A100 | 4.3% | 7.4% |
| SmolLM2-1.7B-Instruct | L4 | 3.3% | 1.0% |
| Gemma 4 E2B | A100 | 1.9% | 1.0% |

Gemma is the hardest architectural case in the set — MQA (a single KV head),
`head_dim` 256 against `hidden/heads` of 192, a 262k vocab, and 28 of 35
layers windowed — so it exercises the geometry paths the Qwen models never
touch.

Holdouts span sequence length, micro-batch, group size, LoRA rank *and*
`gpu_memory_utilization` (0.30/0.60 against corners fitted at 0.45), so a
profile is not pinned to the engine budget it was measured under.

The worst single point runs higher (~13% on the smallest Qwen config). The
residual model is linear in a handful of basis terms while the true residual
is convex in batch x sequence, so the extreme corners cannot all be fitted at
once; `WORST_POINT_BAND` exists as a drift alarm rather than an accuracy
claim.

## What the totals do not tell you

The sweep compares one predicted peak against one measured peak, so it
validates sums, not splits — the bars could each be wrong and cancel. An
allocator snapshot (`--snapshot`, then
`python -m tools.memory_profiling.snapshot`) attributes the *peak instant*
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

Calculation core, advice, preflight CLI, profiling harness, offline refit
and snapshot attribution are implemented and tested (58 CPU-only tests,
including a fixture regression check that replays every stored
measurement). Seven (model, device) calibrations span 0.5B to 7B across an
L4 and an A100.

Known gaps, in rough priority order:

- The logit workspace is over-counted by ~0.5 GiB (see above); correcting it
  needs a re-fit, which is free thanks to the stored measurements.
- The activation model under-predicts at large `micro_batch x seq_len`. The
  snapshot says where it lives; the shape of the term is still open.
- MoE is modelled (weights and optimizer on total experts, activations on
  active) and the parameter counts check out against published totals, but
  no MoE model has been measured end to end yet.
- Quantized (nf4) variants are supported by the schema and still unmeasured
  in practice.
- The Arena widget itself, on this same core.

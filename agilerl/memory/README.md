# GPU memory estimation for LLM RL

A first-principles model of peak GPU memory occupancy for the LLM RL stack,
built as a closed-form function calibrated per curated model:

```
peak(model_spec, device_spec, knobs) -> {component: bytes}   # per phase
```

One calculation core, two surfaces: the Arena sizing widget (two live stacked
bars) and the CLI preflight (`python -m agilerl.memory.preflight`). The core
(`specs.py`, `formulas.py`, `estimator.py`, `calibration.py`, `advice.py`) is
pure python with no torch dependency, so it ports line-for-line to a browser
runtime; per-model calibration ships as one self-contained JSON under
`fixtures/`.

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

## Training bar

| Component | Scales with | Grounding in the framework |
|---|---|---|
| Base weights (frozen) | `P x bytes(dtype)`, or realised quantized size | trainer always holds its own full copy; nf4/int8 keep `lm_head` unquantized and upcast norms + `lm_head` to fp32 (`base.py` k-bit prep) |
| LoRA adapters | `rank`, target scope, x2 with the separate reference adapter | **full FT does not exist** — the framework trains adapters only |
| Gradients + optimizer state | trainable (adapter) params only | plain `torch.optim.AdamW`; DeepSpeed-config optimizer is the only alternative. LoRA-only makes this a rounding error next to the base — the opposite of the classic 16-bytes/param intuition |
| Activations | `max(grad pass, no-grad logprob pass)` | grad pass: checkpoint boundaries (`rows x S x H x L`) + one block's recompute + the `(rows, S, H)` hidden the fused-logprob autograd saves. No-grad pass: actor+reference(+value) rows fused into one wider forward (`_fused_forward_no_grad`), micro-batched by the same per-GPU cap |
| Logit workspace | `2 x chunk_rows x V x 4` | the fused chunked path (`llm_ops/fused_logprobs.py`) never materialises `B x S x V`; auto-tune caps the tile at 256 MiB |
| Overhead | intercept | CUDA context, held rollout tensors (MB-scale), allocator slack; absorbed by the calibration intercept |

What is *not* a training memory knob, and worth teaching in the UI:

- **`beta`**: the reference forward runs and the reference adapter exists
  regardless of beta (KL is always computed as a metric). Memory-neutral.
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

## Calibration: function + constants, not a lookup tensor

Tabulating is infeasible (six knobs x four levels is 4096 runs per model and
sequence length is continuous) and unnecessary — the analytic form carries
the shape, so profiling only pins constants:

```
predicted = analytic + intercept + sum(slope_i x basis_i(knobs))
```

with named, interpretable basis terms (`grad_tokens`, `nograd_tokens`,
`batched_tokens`, `kv_tokens`, `seq_len`). ~16 corner points fit the
residual; 3 interior points are held out to validate *interpolation across
the knob space* (the generalisation question is not "unseen models" — every
curated model is profiled — but "unseen knob combinations").

Calibration is per (model, device): kernel selection and workspace sizes
differ across GPU architectures. A profile measured on one device applied to
another is possible but should be surfaced as lower confidence.

## Profiling protocol (`profiling/`)

- **NVML polling is mandatory for the generation phase.**
  `torch.cuda.max_memory_allocated` only sees the torch caching allocator;
  colocated vLLM allocates weights and KV through CuMem (the sleep/wake
  mechanism), which bypasses torch entirely. `NvmlPeakSampler` polls
  device-level used-bytes on a background thread; torch stats are recorded
  for the training phase as a cross-check.
- Each sweep point runs the real path — `GRPO.get_action` (colocated vLLM,
  sleep mode) then `GRPO.learn` — with synthetic fixed-length prompts;
  content is irrelevant to memory, shape is everything.
- Realised weight bytes are measured per variant at load by summing
  parameter/buffer storages (nominal bits-per-weight lies: scales, held-out
  layers, and fp32 upcasts all cut into the saving).
- Because the memory optimizations (chunked logprobs, gradient
  checkpointing, FlashAttention) are always on, the sweep has no
  optimization on/off axis.
- Constants drift with framework changes (anything touching the fused
  kernels, checkpointing, or the vLLM wiring moves them). Two defences:
  fixtures keep their raw points so `python -m agilerl.memory.profiling.refit`
  re-derives the constants against the current core with no GPU, and
  `tests/test_memory/test_fixtures.py` replays every stored point through the
  estimator in CI, failing if any drifts outside the band.

## Porting notes

- Component `key` strings and the `ModelProfile` JSON schema are the
  contract with the widget; change them only with a `schema_version` bump.
- `estimate_run(...).model_dump(by_alias=True)` is exactly the widget
  payload; `advise()` is the "you are N GiB over, cheapest fixes" list,
  computed by re-running the estimator under candidate knob mutations so it
  can never disagree with the bars.
- Fitting (`profiling/sweep.py`) needs numpy; applying a fit does not.

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

## Status

Prototype with one calibrated model. Calculation core + advice + preflight
CLI + profiling harness implemented and unit-tested (33 CPU-only tests); the
sweep runs end-to-end on an L4 and the first fixture
(`fixtures/Qwen__Qwen2.5-0.5B-Instruct.json`) validates inside the target
band. Next: broaden the curated set (+ nf4 / vision-stripped variants),
add a re-fit that attributes the engine floor analytically (so the intercept
drops to true overhead), a cut-down CI drift check, and the two-bar Arena
widget on this same core.

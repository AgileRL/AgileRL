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
colocated on one device: vLLM sleeps (level 1 — base weights to host RAM, KV
pool freed) during `learn`, and the trainer offloads its base to CPU
(`use_memory_efficient_params`) before waking vLLM for rollout
(`LLMAlgorithm._prepare_vllm_for_generation` / `_prepare_vllm_for_training`).
So the sizing question is two separate peaks against the same capacity, plus
a small cross-phase residual on each side (what the dormant side leaves
resident). No phase modelling, no `max()` over interleavings.

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
  kernels, checkpointing, or the vLLM wiring moves them). A cut-down sweep
  on one small model in CI should assert holdout error stays inside the
  band.

## Porting notes

- Component `key` strings and the `ModelProfile` JSON schema are the
  contract with the widget; change them only with a `schema_version` bump.
- `estimate_run(...).model_dump(by_alias=True)` is exactly the widget
  payload; `advise()` is the "you are N GiB over, cheapest fixes" list,
  computed by re-running the estimator under candidate knob mutations so it
  can never disagree with the bars.
- Fitting (`profiling/sweep.py`) needs numpy; applying a fit does not.

## First calibration finding — the colocated training floor

Profiling `Qwen/Qwen2.5-0.5B-Instruct` on an L4 (vLLM 0.23, sleep level 1)
surfaced the single most decision-relevant fact the widget can teach:
**colocated training does not get the vLLM engine's memory back.** Every
training point sat on a ~12 GiB device floor for a model whose weights are
under 1 GiB, flat across all knobs — almost exactly the engine's
`gpu_memory_utilization` (0.45 × 23 GiB ≈ 10.4 GiB) reservation. Sleep
level 1 offloads the base to host RAM and frees the KV *contents*, but the
reservation stays mapped on the device rather than returning to the
co-resident trainer.

Consequences baked into the estimator:

- The training bar attributes this to a labelled **"Sleeping engine
  reservation"** segment (usually the dominant colocated-training term), not
  to overhead. An uncalibrated colocated estimate now includes it
  analytically (`gpu_memory_utilization × total`) so it never tells a user a
  run fits when it won't.
- It means the practical colocated headroom for training is
  `total − gpu_memory_utilization × total`, i.e. the *same* budget split the
  generation phase lives under — the two bars are more symmetric than the
  "training gets the whole GPU back" intuition suggests. Lowering
  `gpu_memory_utilization` helps *both* phases.

Holdout accuracy on this first profile: training 4.8%, generation 0.37% —
inside the 10% target band. (One generation *fit* point sits at ~12%, a
single noisy measurement; the held-out points are the accuracy signal.)

## Profiling gotchas found while running the first sweep

- **One process per point.** vLLM's CuMem allocator (which backs sleep mode)
  is process-global and permits one engine per process, so `run_sweep`
  spawns a fresh subprocess per point (`python -m
  agilerl.memory.profiling.harness`). A single-process loop dies on the
  second point with "CuMem allocator can only be used for one instance per
  process".
- **NVML polling undercounts the training spike.** The 10 ms poll missed the
  brief backward-pass peak (torch's exact `max_memory_reserved` came in
  ~0.7 GiB higher). The training calibration target is therefore
  `max(nvml_poll, non-torch baseline + torch_max_reserved)`; generation
  stays pure-NVML (vLLM's CuMem is invisible to torch and its allocation is
  not spiky).
- **flashinfer JIT needs `curand.h`.** On a box without the full CUDA
  toolkit headers, set `VLLM_USE_FLASHINFER_SAMPLER=0`.

## Status

Prototype with one calibrated model. Calculation core + advice + preflight
CLI + profiling harness implemented and unit-tested (33 CPU-only tests); the
sweep runs end-to-end on an L4 and the first fixture
(`fixtures/Qwen__Qwen2.5-0.5B-Instruct.json`) validates inside the target
band. Next: broaden the curated set (+ nf4 / vision-stripped variants),
add a re-fit that attributes the engine floor analytically (so the intercept
drops to true overhead), a cut-down CI drift check, and the two-bar Arena
widget on this same core.

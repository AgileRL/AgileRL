# Auto-config for LLM RL training — design

**Goal** (from the "Dynamic vLLM/micro batch assignment" ticket): given the hardware we
are running on, the model, the context length, and the environment's workload profile,
automatically assign vLLM engine parameters, micro-batch sizes, parallelism layout, and
the GPU memory split between trainer and rollout engine — so a run *fits* and is *fast*
without the user hand-tuning a dozen interacting knobs.

This doc is the result of a survey of our own stack (agilerl + agilerl-integration) and
the state of the art (prime-rl, TRL, verl, OpenRLHF, NeMo-RL, AReaL, SkyRL, slime, and
the leading single-GPU fine-tuning stacks). It covers:

1. What inputs an auto-configurator needs.
2. The complete knob inventory (what exists, what is dead, what is hardcoded and must
   become a knob first).
3. The memory and throughput models that turn inputs into knob values.
4. Two distinct optimization processes: **colocated** and **async/decoupled**.
5. A phased delivery plan: hygiene → static analytic config → probe calibration →
   telemetry-driven adaptation (arena integration).

---

## 1. What the field does (survey conclusions)

No competitor ships a true auto-tuner. What exists, and what is worth copying:

| Library | What they auto-derive | What they leave manual | Lesson |
|---|---|---|---|
| **TRL** | Batch-system closure (`generation_batch_size = per_device_bs × world × steps_per_generation`, divisibility validation); engine `max_num_seqs` from training batch geometry; async `max_inflight_tasks = staleness × global_batch` | `vllm_gpu_memory_utilization` fixed 0.3 colocated / 0.9 server; `max_num_batched_tokens` pinned 4096 | Close the over-determined batch system with validators; derive engine concurrency from batch geometry, not vLLM defaults |
| **prime-rl** | All *topology arithmetic* (FSDP shard degree, inference DP fill, api_server_count, packer worker counts, LoRA rank rounding); env workers = 1 per 256 concurrent rollouts | **Zero memory-based sizing** ("you need to make sure the model will fit… we will not go into details"); GPU split user-set; fitting via a `--bench` probe + ~60 persisted baseline JSONs keyed `(GPU, model, seq_len, AC, LoRA)` + a curated example ladder | Token-budget packing **replaces** micro-batch/grad-accum knobs entirely (FFD bins of `seq_len` tokens; accumulation is emergent). Maintain an empirical baseline table; standardize a cheap bench probe |
| **verl** | Nothing shipped (the auto-tune feature request is open) | Everything; but the *perf-tuning guide ordering* is a ready-made search order, and `use_dynamic_bsz` token budgets (`ppo_max_token_len_per_gpu ≥ 2×(max_prompt+max_response)`, forward-only budgets ≈ 2× that) replace micro-batch counts | Split knobs into **algorithmic/global** (convergence) vs **performance/local** (throughput-only) — auto-config may only touch the second class |
| **Single-GPU FT stacks** | The most complete pipeline: analytic memory model (weights/LoRA/KV per-token bytes, GQA-aware) decides *feasibility and regime*; **empirical tier tables** keyed on KV-headroom-GB and total-VRAM-GB pick the actual values (`max_num_seqs` 8→256, `max_num_batched_tokens` 2048→8192, standby util 0.75→0.925 by GPU size); budget from **free** memory converted to fraction-of-total; explicit CUDA-graph reservation (0.15–1.0 GiB); OOM retry ladder (`seqs ×0.75`, `util ×0.85`); runtime micro-batch autotune from `mem_get_info` with a quadratic logit-memory model | Multi-GPU, async | Hybrid analytic + lookup beats pure analytics; reserve trainer activations *before* giving vLLM its share; tier the sleep-mode util target by GPU size because the remainder must cover an *absolute* trainer footprint |
| **AReaL / SkyRL / NeMo-RL** | NeMo-RL: buffer = `prompts_per_step × max_age × 2`; SkyRL: hard placement-consistency validators | GPU split (AReaL: fixed 75:25 inference:training, chosen empirically, authors note it "could benefit from dynamic adjustment") | Async split is workload-dependent; start from a ratio table, close the loop with telemetry |

Shared doctrine across verl/OpenRLHF/TRL: **prefer more inference-engine replicas (DP)
over deeper TP** until the model stops fitting; minimum TP from
`TP ≥ 2 × param_bytes / (mem_per_gpu × gpu_mem_util)`.

---

## 2. Inputs to the auto-configurator

Everything below is obtainable today; items marked ⚠ need new plumbing.

**Hardware** (probe locally, or accept the platform `--resource-spec` JSON in Ray mode):
- `torch.cuda.device_count()`, `torch.cuda.get_device_properties(i)` (total VRAM, SM
  arch → bf16/fp8 support, name → A100-vs-H100 default split), `torch.cuda.mem_get_info()`
  (free VRAM at config time — the single most important number we currently never read).
- Host RAM (for CPU-offload/pinned-buffer decisions), ⚠ interconnect class
  (NVLink vs PCIe; conservative default: PCIe unless probed).

**Model** (from the HF config — no weights needed):
- `hidden_size, num_hidden_layers, num_attention_heads, num_key_value_heads, head_dim,
  intermediate_size, vocab_size, tie_word_embeddings`, attention layout (full vs
  sliding-window/hybrid — Gemma-style 5:1 SWA:full changes KV math ~6×), MoE fields.
- Param count `P` (safetensors metadata or config-derived), trainer dtype, quantization
  plan (bf16 / bnb-8bit / bnb-4bit), LoRA config (r, alpha, target modules → `P_lora`).

**Workload profile** (from env metadata + algo args):
- `max_prompt_tokens` (dataset p95/max — the dataset gym already tokenizes, so
  percentiles are computable at init ⚠), `max_output_tokens`, `max_turns`,
  multi-turn vs single-turn, shared-prefix structure (system prompt length).
- Derived classification: `prefill_fraction = E[prompt_tokens] / E[prompt+completion]`.
  - **prefill-heavy** (long prompts, short completions — preference data, judge envs,
    reasoning-over-context): prefill_fraction ≳ 0.7
  - **decode-heavy** (short prompts, long/multi-turn generations — math reasoning,
    agentic envs): prefill_fraction ≲ 0.3
  - **balanced** otherwise.
- Batch plan: `data_batch_size_per_gpu × group_size` (the real rollout/learn batch),
  target effective batch.

**Topology**: colocated vs async/decoupled; pop_size; weight-sync backend (filesystem /
NCCL); DDP vs FSDP2 (constrained: colocated zero-copy weight sharing ⇒ DDP, base
unsharded; FSDP2 is for decoupled trainers).

---

## 3. Knob inventory

### 3.1 Knobs that exist today (agilerl lib, colocated path)

| Knob | Default | Should be derived from |
|---|---|---|
| `VLLMConfig.gpu_memory_utilization` | 0.3 | VRAM, model footprint, trainer peak, sleep mode — **the** colocated split knob (fraction of *total*, gated by free-at-init) |
| `VLLMConfig.kv_cache_memory_bytes` | None | Exact KV pin; bypasses vLLM profiling — the most precise rollout-memory knob we have; auto-config should usually compute and set this directly |
| `VLLMConfig.max_num_seqs` | 8 | `data_batch_size_per_gpu × group_size` (colocated) / rollout slots (async), KV budget, env profile |
| `VLLMConfig.tensor_parallel_size` | 1 | model size vs per-GPU VRAM, GPU count, interconnect |
| `VLLMConfig.sleep_mode` | False | topology (colocated ⇒ True), population layout |
| `VLLMConfig.dtype` / `quantization` | None | model dtype / quantization plan |
| `max_model_len` | **1024** | env context profile: `max_prompt_p100 + max_output_tokens` (+ multi-turn budget). Never inherit the model's native 128k — vLLM's "one full-length request" check eats the KV budget |
| `max_output_tokens` / `min_output_tokens` | None | env generation-length profile |
| `batch_size` (algo) | 16 | desired effective batch — *algorithmic knob, not auto-config's to change* |
| `micro_batch_size_per_gpu` | None → derived (and `create_population` defaults it to full `BATCH_SIZE`) | **prime trainer-activation knob**: VRAM, stitched seq len, vocab, fused-loss flags, gradient checkpointing |
| `group_size` | 8 | algorithmic (reward variance) — but capacity must be *re-validated* when HPO mutates it |
| env `data_batch_size_per_gpu` | 8 | effective-batch plan; co-moves with `max_num_seqs` |
| `gradient_checkpointing` | True | activation memory vs ~30% speed |
| `use_liger_loss` / `use_fused_linear_logprobs` | False | should auto-enable when installed (docs already claim this — doc/code drift) |
| `cast_logprobs_to_fp32` | True | vocab size, VRAM (≈18 GB swing on the unfused path) |
| `use_separate_reference_adapter` | True | semantics choice; memory layer must know it multiplies the no-grad fused batch ×N |
| `use_memory_efficient_params` | True | compute from whether trainer-peak + woken engine fit, not a static default (costs 2 full PCIe weight transfers per iteration) |
| `lora_config` (r/alpha/targets) | r16/α32/all-linear | task policy; feeds `P_lora`, `max_lora_rank` |
| `hf_generate_chunk_size` | 1 | non-vLLM fallback only |
| env `max_context_length` | None | must equal `max_model_len` — needs a single source of truth |

### 3.2 Hardcoded/dead items that must become knobs first (Phase 0)

1. **`max_num_batched_tokens = max_num_seqs × max_model_len`** (hardcoded at engine
   creation). At 64 seqs × 32k ctx that is a 2M-token prefill budget: it disables
   chunked-prefill admission control and inflates the profile-run activation
   reservation (the dummy run scales with this value). Must be independent and
   env-profile-derived. TRL pins 4096 for exactly this reason; the FT stacks tier it
   2048→8192 by KV headroom.
2. **Sleep level 2 hardcoded** — must be `{1, 2, standby}`; bnb-4bit requires the
   standby patch (level-2 re-quantizes to garbage, level-1 frees too little).
3. **`enable_prefix_caching` never set** — GRPO duplicates each prompt `group_size`
   times with `n=1`; dedupe currently rides on vLLM's default. Make explicit
   (multi-turn and grouped envs benefit massively; the prefix cache is also keyed by
   LoRA name, so in-place adapter updates require `reset_prefix_cache()` — we already
   do this in the sync path).
4. **`swap_space` / `enforce_eager` declared on `VLLMConfig` but never forwarded to
   `LLM()`** — wire or delete (note: `swap_space` is deprecated/ignored in vLLM V1
   anyway; preemption is recompute. Delete it. `enforce_eager` is a real
   low-VRAM lever; wire it).
5. **Eval temperature 0.01 hardcoded.**
6. **Trainer load dtype** is fp16-under-accelerator regardless of mixed-precision
   config; `attn_implementation="sdpa"` hardcoded (flash-attention-2 never
   auto-selected).
7. **Liger `chunk_size=1`** and logprob `_chunk_rows` (1 / 1024) hardcoded —
   VRAM-derivable.
8. **`batch_size` (optimizer bookkeeping) vs `data_batch_size_per_gpu × group_size`
   (the actual learn batch) are never reconciled** — one derived quantity, validated,
   with HPO mutation hooks re-running the derivation (today a `group_size` mutation
   silently invalidates capacity assumptions).
9. **No hardware introspection anywhere** — the stack reads `device_count()` only.
10. **No packing/length-sorting** — pad-to-longest everywhere; the preference env even
    pads every pair to full `max_context_length`.
11. Per-`learn` `gc.collect()+empty_cache()` — measurable cost on fast loops; make
    conditional on memory pressure.

### 3.3 Knobs at the Ray/async layer (agilerl-integration, async-rollout branch)

| Knob | Today | Auto-config target |
|---|---|---|
| GPU split | rigid: rollout = exactly 1 GPU/agent, trainer = floor-divided rest, hard `total ≥ 2×pop` | ratio derived from workload profile + measured tokens/s (§6) |
| trainer micro-batch | **hardcoded `min(per_rank, 1)`** with a FIXME | trainer memory solve (§5) |
| `gpu_memory_utilization` | 0.9 injected for async (vs 0.3 colocated) — the codified topology split | per-engine solve; 0.85–0.92 fine for dedicated GPUs |
| `max_num_seqs` | 16 | rollout slots (`rollout_batch_size × group_size`), with the documented ~2× slot:seq oversubscription to keep KV saturated through the long tail |
| `rollout_engines_per_agent` | must be 1 | unlock >1; engines = ceil(required concurrency / per-engine concurrency) |
| `weight_sync_interval` / `max_rollout_version_lag` | 2 / ∞ | staleness policy by env episode length (cf. prime-rl 8 single-turn / 16 agentic) |
| `buffer_occupancy_multiplier`, `memory_size` | 1 / 10000 | `batch × max_lag × 2` (NeMo-RL formula) |
| env workers | already auto-sized from CPU budget | keep; precedent for the pattern |
| timeouts (`POP_BATCH`, generation, NCCL) | static env vars | scale with `max_turns × max_output_tokens / concurrency` |

vLLM TP is hardcoded 1 on the NCCL sync path and trainer workers are pinned to 1 GPU
each — both become real knobs under the FSDP2 move.

---

## 4. Memory models

All formulas per GPU. `P` = params, `P_lora` = trainable adapter params, `W` = shard
world size, `b_x` = bytes/element. Validate constants against the FT stacks' calibrated
tables: 4-bit ≈ `P×0.625` bytes (16/5 factor incl. quant scales), 8-bit ≈ `P×1.25`;
CUDA context + NCCL + allocator slack ≈ **1.5–2.5 GB** per GPU, always reserved.

### 4.1 vLLM (rollout) side

- **Weights**: `P × b_engine / TP` (0 extra in colocated mode once zero-copy
  base-weight sharing is in: the trainer aliases the engine's tensors).
- **KV cache**: `bytes/token = 2 × (num_kv_heads/TP) × head_dim × b_kv ×
  n_full_attn_layers` (+ a bounded `≈ window` term per sliding-window layer for hybrid
  models — naive full-attention math over-estimates Gemma-class KV ~6×). `b_kv` = 2
  bf16, 1 fp8 (`kv_cache_dtype="fp8"` ⇒ ~2× concurrency for decode-heavy envs; validate
  logprob sensitivity for importance-ratio algorithms before defaulting on).
  vLLM's own `kv_cache_utils` (`estimate_max_model_len`,
  `get_max_concurrency_for_kv_cache_config`, `max_memory_usage_bytes`) implement this
  including hybrid/MLA corner cases — **import them, don't reimplement**.
- **Activation/profile peak** ∝ `max_num_batched_tokens` (the profile dummy run uses
  exactly that many tokens; whatever it reserves is lost to KV forever).
- **CUDA graphs**: charged against the KV budget by the profiler in recent vLLM;
  reserve explicitly when computing our own budget: `clamp(172 × hidden × layers × 4,
  0.15 GiB, 1.0 GiB)` (FT-stack calibration), or `enforce_eager=True` on tiny-headroom
  GPUs at a 10–30% decode cost.
- **LoRA slots**: static buffers ∝ `max_loras × max_lora_rank` per target module — set
  `max_lora_rank = r` exactly and `max_loras = 1` (colocated) / agents-per-engine
  (async).
- Three failure modes the configurator must preempt: (a) `total×util > free_at_init`
  (startup check — colocated trainer footprint), (b) `util` too low to cover
  weights+activations+graphs ⇒ 0 blocks, (c) KV remainder < one full `max_model_len`
  request.

**Demand side**: required concurrent KV tokens ≈ `concurrent_seqs × expected_live_tokens`
where `concurrent_seqs = data_batch × group_size` (colocated) or
`rollout_batch × group_size × oversubscription(≈2)` (async). For GRPO groups all
sequences grow together — budget for `group × max_len`, not the average, or accept
preemption (recompute) on the tail.

### 4.2 Trainer side — colocated (DDP, LoRA, base shared with vLLM)

```
M_steady  = P×b_base(shared→0 extra) + P_lora×b + grads(P_lora×b) + opt(P_lora×8)   # AdamW fp32 moments
M_learn   = M_steady + A_live(micro_bs, s_train, …) + C_overhead
A_live    ≈ micro_bs × s_train × hidden × c_arch × b_act × gc_factor   # gc_factor ≈ 1 layer w/ ckpt
          + logits term:  fused → chunk_size × V × b   |   unfused → micro_bs × s_train × V × (2 + 4·fp32cast)
```

`s_train` is the **stitched** sequence length — multi-turn sliding-window stitching
means training sequences can exceed `max_model_len`; the trainer bound is the stitched
length and today no knob caps it (Phase 0 item). The no-grad ref/old-logprob pass runs
at `micro_bs × N_adapters` (separate-reference-adapter and value head multiply it 2–3×).

Time-slicing gives two constraints instead of one:

```
Phase G (generate):  M_steady_minus_activations + KV + engine_act_peak + graphs ≤ VRAM − headroom
Phase L (learn):     M_steady + A_live(micro_bs)            (KV freed by standby) ≤ VRAM − headroom
```

Solve Phase L for the largest `micro_bs`; solve Phase G for the KV budget → set
`kv_cache_memory_bytes` exactly (and `gpu_memory_utilization` just low enough to pass
the startup free-memory check). `use_memory_efficient_params` (CPU round-trip of
trainer params during generation) becomes a *derived* escape hatch: only when Phase G
fails without it — and it is moot once zero-copy sharing lands.

### 4.3 Trainer side — decoupled (DDP or FSDP2)

DDP (default while adapters-only and base fits):
`M = P×b + P_lora×(b + 8) + A_live + C` — comm is one ~MB all-reduce; strictly better
than FSDP2 for LoRA whenever the base fits replicated (our ≤8B on A100 envelope:
almost always).

FSDP2 (per-block `fully_shard`, `reshard_after_forward=True`, bf16 param / fp32 reduce):

```
M_peak ≈ P/W×b_orig + P_t/W×b_orig + 2×P_t/W×4          # sharded params/grads/Adam
       + E×b_param                                       # embeds+lm_head group unsharded
       + 3×G_max×b_param + G_max×b_reduce                # transient unsharded blocks (G_max = largest block)
       + A_live + C_overhead
```

Key facts that shape the decision rules: with LoRA, FSDP2 still **all-gathers the
entire frozen base every forward** (3–4 orders of magnitude more wire bytes than DDP);
`ignored_params=base` gives the hybrid (adapters sharded, base replicated, zero base
comm); **bnb 4-bit bases cannot be FSDP2-sharded** (no `fsdp_pre_all_gather` on
`Params4bit`; torchao NF4 has it) — bnb + doesn't-fit ⇒ NF4 or more GPUs. Decision rule:

```
if M_ddp_peak ≤ budget:           DDP   (or fully_shard+ignored_params for sharded ckpt ergonomics)
elif activations bind:            stay DDP; AC ladder / fused loss / smaller micro first
else (base doesn't fit):          FSDP2 per-block, reshard=True; reshard=int(intra-node) multi-node
```

Validation: `FSDPMemTracker` under `FakeTensorMode` predicts FSDP2 peaks **without a
GPU** — the right pre-flight check for the chosen config.

### 4.4 The micro-batch endgame: token budgets, not sequence counts

RL rollouts are variable-length; a fixed `micro_batch_size_per_gpu` is sized for the
worst case and wastes the average case (pad-to-longest today makes it worse). Both
prime-rl (FFD bin packing into `seq_len`-token bins; grad-accum emergent; global
token-count loss normalization with an FSDP gradient-divide correction) and verl
(`use_dynamic_bsz`, `ppo_max_token_len_per_gpu ≥ 2×(max_prompt+max_response)`,
forward-only budgets ≈ 2×) converge on the same answer: **micro-batch by token budget**.
This makes trainer memory ~independent of the rollout length distribution — i.e. it
removes the hardest input from the sizing problem. Phase 2 deliverable; until then the
analytic solve above sizes the sequence-count knob.

---

## 5. The colocated optimization process

Inputs → decisions, in order (each step emits a `(knob, value, reason)` record):

1. **Context budget**: `max_model_len = prompt_p100 + max_output_tokens` (multi-turn:
   the windowed budget); propagate as the *single* source of truth to env filter, engine,
   GenerationConfig, and the new training-side stitched-length cap.
2. **Trainer dtype/quant plan** (given): compute `M_steady`.
3. **Loss-path flags**: liger/fused logprobs auto-on when installed and supported;
   chunk sizes from VRAM headroom; `cast_logprobs_to_fp32` off on the fused path for
   large-V models when headroom is tight.
4. **Solve Phase L** for `micro_batch_size_per_gpu` (largest that fits, floor 1; if
   floor 1 doesn't fit → escalate: gradient checkpointing on (already default) →
   fused loss mandatory → smaller `s_train` cap → fail with actionable message).
   Derive `gradient_accumulation_steps = learn_batch / micro` and validate
   divisibility against `data_batch × group_size`.
5. **Solve Phase G** for KV: `kv_bytes = VRAM − headroom − M_resident_during_gen −
   engine_act_peak(max_num_batched_tokens) − graph_reserve`. Set
   `kv_cache_memory_bytes` directly; set `gpu_memory_utilization` to the matching
   fraction for the startup check.
6. **Engine workload knobs** by profile:
   - decode-heavy: `max_num_batched_tokens` 2–8k; `max_num_seqs ≥ batch×group`
     (check vLLM's logged "Maximum concurrency ≥" demand); fp8 KV as the 2× lever;
     prefix caching on (multi-turn re-prefill dedupe).
   - prefill-heavy: `max_num_batched_tokens` 8–16k (A100: 8k — large values reduce
     throughput on A100 per vLLM's own defaults); prefix caching on if shared
     templates; `max_num_seqs` modest.
7. **Sleep level**: standby for bnb-quantized engines or zero-copy-shared bases;
   level 2 only for full-dtype engines without sharing (weights cheaply reloadable —
   they're rewritten by the next weight sync anyway, TRL's argument).
8. **TP**: 1 unless `P×b_engine > ~0.6 × VRAM`; verl floor
   `TP ≥ 2×P_bytes/(VRAM×util)`; prefer engine replicas over TP (doctrine), noting
   TP>1 helps ≥7B and *hurts* ≤1.5B.
9. **Validate** everything against §4.1's three failure modes + batch divisibility +
   `max_num_seqs ≥` demand (or log accepted queueing), then emit the decision report.

Capability ladder when it doesn't fit (in order, mirroring prime-rl's documented
recipe and the torchtune ladder): fused loss → micro=1 → AC already on → fp8 KV /
smaller `max_num_batched_tokens` → `enforce_eager` → 4-bit engine quant → smaller
`max_model_len` (warn loudly: changes the task) → refuse with the closest-fitting
config printed.

## 6. The async/decoupled optimization process

Different problem: no time-slicing — instead a **GPU partition** and a **pipeline
balance**.

1. **Per-rollout-engine sizing** is the colocated §5 steps 6–8 with the whole GPU:
   `gpu_memory_utilization` 0.85–0.92, KV from the engine's own profiling (no pin
   needed), `max_num_seqs` from slots×oversubscription(2×), `max_loras` =
   agents-on-engine, `max_lora_rank` = r.
2. **Per-trainer-GPU sizing**: §4.3 (DDP vs FSDP2 decision + micro-batch/token-budget
   solve). FSDP2 unlocks trainer models that don't fit per-GPU — the decoupled-only
   sharding path.
3. **The split** (the genuinely new decision): choose `R` rollout GPUs : `T` trainer
   GPUs to balance token *production* against token *consumption*:
   - tokens produced per engine ≈ measured `gen_tokens/s` (decode-bound);
     tokens consumed per trainer GPU ≈ measured `learn_tokens/s` (≈ 5 forward-equiv
     per token for GRPO: old-logprob + ref no-grad passes + fwd/bwd, ÷ update_epochs).
   - v0 (no measurements yet): ratio table by profile — decode-heavy/agentic 2:1 → 3:1
     rollout:trainer (AReaL's 75:25), balanced 1:1, prefill-heavy 1:1 → 1:2. Floor: the
     existing `≥ 2×pop` assertion generalizes to `≥ (1+1)×pop`.
   - v1: rebalance from telemetry (`buffer wait time` high ⇒ rollout-bound ⇒ shift a
     GPU to rollout or add an engine; `occupancy at cap`/eviction-by-lag ⇒
     trainer-bound). prime-rl's two human signals (`wait_for_batch` vs
     `wait_for_ckpt`) are exactly the gauges to automate against.
4. **Staleness policy**: `weight_sync_interval` and `max_rollout_version_lag` from
   episode shape — near-on-policy (1/1) for short episodes; lag 8 single-turn → 16
   agentic (prime-rl's ladder) when episodes span many sync periods. Buffer
   `memory_size = batch × max_lag × 2`; `buffer_occupancy_multiplier` stays 1.
5. **Engines per agent**: unlock `rollout_engines_per_agent > 1`; engines =
   `ceil(required_concurrency / per_engine_max_concurrency)` where per-engine
   concurrency comes from vLLM's logged KV capacity at `max_model_len`.
6. **Timeout scaling**: `pop_batch_timeout ∝ max_turns × max_output_tokens /
   per-engine decode rate` — replace the static env vars.

Auto-config plugs in at **manifest/spec-build time** (precedents already exist in
ConfigBuilder: the async 0.9 util injection, `use_vllm` coercion, on-policy
batch auto-correction, env-worker CPU auto-sizing) and emits the same decision report,
which the platform can render and override per-field.

---

## 7. Delivery plan

**Phase 0 — knob hygiene (small PRs, immediate):** §3.2 items. Wire/delete dead
fields, expose `max_num_batched_tokens` / sleep level / prefix caching / chunk sizes,
unify context length, reconcile the batch dichotomy + HPO revalidation hook, fix the
fp16-under-accelerator dtype and `create_population` micro default. None of this
changes behavior for users who set values explicitly.

**Phase 1 — static analytic auto-config:** a pure function
`resolve_llm_run_config(hardware, model, workload, topology) → (ResolvedConfig,
DecisionReport)` in the agilerl lib, implementing §5/§6 with the §4 closed forms +
tier-table fallbacks. Called from algo init when knobs are `"auto"`/unset, and from the
integration ConfigBuilder for manifests. Every derived value logged with its formula
inputs; `--dry-run` prints the report (prime-rl's validator-network pattern).
Conservative by design: prediction errors fail *toward* fitting.

**Phase 2 — probe calibration:** (a) trainer micro-batch autotune at first learn from
live `mem_get_info` (the FT-stack approach: quadratic logit model, once per run, with
distributed lockstep — all ranks must agree, the known accelerate
`find_executable_batch_size` pitfall); (b) `FakeTensorMode + FSDPMemTracker` pre-flight
for FSDP2 configs (no GPU time); (c) OOM backoff ladder on engine init (`seqs ×0.75`,
`util ×0.85`, never under standby); (d) a standardized `--bench` probe (4 fake-data
steps → tokens/s, MFU, peak mem) persisting baselines keyed
`(gpu, model, seq_len, quant, lora, ac)` — the empirical table that sharpens Phase 1's
priors over time. (e) token-budget micro-batching + FFD packing to retire the
sequence-count knob entirely.

**Phase 3 — telemetry loop + arena:** emit the DecisionReport as run metadata on the
existing Pulsar/OTel channel (run_id-keyed, same as MetricsHandler series) so the arena
UI can show *what was tuned, to what, and why*. Add the missing gauges: tokens/s
(generate and learn separately), vLLM KV-utilization %, prefill/decode token split,
buffer queue-latency (envelopes already carry `event_ts` — unused), NCCL sync duration,
per-phase peak memory. Close the loop conservatively: between-iteration adjustments
only for restart-free knobs (sampling, batch routing, engine assignment via
`AgentRuntimeConfig`), engine-restart knobs only at run boundaries or via the existing
stop/start lifecycle. The async GPU split becomes telemetry-driven here (§6.3 v1).

**Out of scope for now:** runtime vLLM engine re-sizing without restart; speculative
decoding (opt-in for copy-heavy envs only — acceptance collapses at RL sampling
temperatures); P/D disaggregation (relevant only at much larger scale).

---

## 8. Invariants the configurator must always enforce

1. `batch_size % num_processes == 0`; `learn_batch % micro == 0`;
   `num_samples % group_size == 0`; `generation batch contains whole groups`.
2. `max_num_batched_tokens ≥ max_num_seqs` (vLLM), and `≥ max_model_len` when chunked
   prefill is off (it never is, in V1).
3. Colocated: `total × gpu_memory_utilization ≤ free_at_engine_init`; KV ≥ one
   full-`max_model_len` request; sum of colocated stages' fractions ≤ 1.0.
4. Async: `total_gpus ≥ (trainer_floor + rollout_floor) × pop_size`; placement-
   consistency (SkyRL-style) between engines × TP × DP and allocated GPUs.
5. `max_lora_rank ≥ r` (rounded up to vLLM's legal set {8,16,32,…,512}).
6. Stitched training length ≤ trainer activation budget (new cap from Phase 0).
7. HPO mutations of capacity-coupled params (`group_size`, lr is fine) re-run
   validation; reject or re-derive, never silently proceed.
8. Decision report completeness: every knob the resolver touched appears with reason;
   every knob it *didn't* touch because the user pinned it appears as `user-pinned`.

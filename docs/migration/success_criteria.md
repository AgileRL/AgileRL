# FSDP2 production success criteria

Pass/fail gates for shipping `migration/fsdp` (FSDP2 + torchrun LLM path) and
replacing DeepSpeed ZeRO on LLM training in production.

**Do not push or declare production-ready until every gate below is green**, or
has an explicit waiver recorded in `blockers.md` with owner + expiry.

## Scope

| In scope | Out of scope |
| --- | --- |
| LLM algorithms: **GRPO, DPO, SFT, LLMPPO, LLMREINFORCE** | Classic RL (Accelerate stays) |
| Flat DP + optional FSDP2 (`fsdp_config`) | DeviceMesh / TP / PP / EP (v2) |
| Compare FSDP2 vs DeepSpeed ZeRO (zero3 SoT) | Tool masking / OpenEnv |
| Learning parity + throughput + memory | Unrelated refactors |

**Baselines**

| Role | Worktree / branch | Launch |
| --- | --- | --- |
| Behaviour + DeepSpeed baseline | `/home/mike/wt-zero3` (`feature/zero3-support`) | `accelerate launch` + DeepSpeed plugin / ZeRO stage used in production today |
| Candidate | `/home/mike/wt-fsdp` (`migration/fsdp`) | `torchrun --nproc_per_node N …` + optional `FSDPConfig` |

Same model, tokenizer, dataset shard, LoRA config, seeds, batch/microbatch,
grad accum, max steps, and GPU count for every paired run. Change **only** the
distributed backend.

**Modes to exercise on the candidate (where applicable)**

1. Single GPU, no FSDP (`fsdp_config=None`)
2. Multi-GPU DP without FSDP (`sync_grads` path)
3. Multi-GPU FSDP2 full shard (`fsdp_config` set) — **primary production target**

---

## Verdict rules (apply to all gates)

| Dimension | Pass | Fail |
| --- | --- | --- |
| **Learning** | Paired curves match within tolerances below; no NaN/Inf; eval metric not worse than DeepSpeed beyond tolerance | Systematic loss divergence, silent wrong grads, hang, or eval regression beyond tolerance |
| **Speed** | FSDP2 step time / tokens-sec **≥ 0.90×** DeepSpeed on the same hardware (≤10% slower). Prefer ≥1.0× | Sustained >10% slower without an accepted memory trade-off note |
| **Memory** | Peak allocated VRAM ≤ DeepSpeed + 10%, **or** higher VRAM with documented ≥10% throughput win and no OOM on the prod shape | OOM on a shape DeepSpeed completes; unexplained >10% VRAM regression with no speed win |
| **Correctness / ops** | Checkpoint save→load, adapter export, vLLM sync, clone/tournament paths work under FSDP2 | Corrupt adapters, rank desync, NCCL hang, or export that DeepSpeed got right |

Record every paired run with: commit SHAs, GPU SKU/count, model id, config
path, seed, wall time, tokens/sec (or steps/sec), peak VRAM, final loss / eval,
and artifact paths.

---

## Gate 0 — Static / package done-checks

Prerequisite; no GPU required.

```bash
# From /home/mike/wt-fsdp — must match AGENTS.md
rg 'from accelerate|import accelerate|from deepspeed|import deepspeed' \
  agilerl/algorithms/core/base.py \
  agilerl/algorithms/grpo.py agilerl/algorithms/dpo.py agilerl/algorithms/sft.py \
  agilerl/algorithms/ppo_llm.py agilerl/algorithms/reinforce_llm.py \
  agilerl/utils/llm_utils.py agilerl/utils/distributed.py \
  demos/llm

rg 'from deepspeed|import deepspeed' agilerl demos configs pyproject.toml

rg 'dist\.get_rank|dist\.get_world_size|LOCAL_RANK|WORLD_SIZE' \
  agilerl/utils/distributed.py agilerl/algorithms agilerl/utils/llm_utils.py
```

| Check | Pass criteria |
| --- | --- |
| DeepSpeed imports | None under `agilerl/`, `demos/`, `configs/`, `pyproject.toml` (migration docs OK) |
| Accelerate on LLM path | No Accelerate on LLM algo modules / `llm_utils` / `distributed` / `demos/llm` (classic RL in shared `base.py` OK) |
| Rank helpers | Training/data use `agilerl.utils.distributed`; raw `dist.get_*` / env only in bootstrap/logging inside `distributed.py` (and documented vLLM rendezvous bootstrap) |
| Deps | `deepspeed` removed from LLM extras; `uv.lock` refreshed; `accelerate` remains for classic RL |
| Remap ledger | `docs/migration/fsdp2-remap-ledger.md` complete; open risks only in `blockers.md` |
| Unit/CPU tests | LLM unit suite green under `agilerl[llm]` as far as CI allows without multi-GPU |

---

## Gate 1 — Smoke / launch matrix

Confirm the candidate boots and finishes short runs without hang or crash.

| ID | Run | Hardware | Pass |
| --- | --- | --- | --- |
| S1 | `torchrun --nproc_per_node=2 demos/llm/debugging/debugging_llm.py` | ≥2 GPU | Completes; hit-rate / logs sane |
| S2 | Same demo, `fsdp_config` enabled (full shard) | ≥2 GPU | Completes; no NCCL hang |
| S3 | `debugging_llm_training_matrix.py` (reasoning / preference / multiturn × pop 1 and tournament) | ≥1–2 GPU | All matrix cases complete |
| S4 | Stage demos (`debugging_llm_stage_{1,2,3}.py`) if still shipped | ≥2 GPU | Complete under torchrun |
| S5 | Classic RL Accelerate demo smoke (e.g. existing non-LLM train path) | ≥1 GPU | Still imports/runs — no accidental strip |

Fail on first hang, collective desync, or import of DeepSpeed on the LLM path.

---

## Gate 2 — Learning parity vs DeepSpeed (per algorithm)

For each affected algorithm, run **paired** short- and medium-horizon jobs:
DeepSpeed on `wt-zero3` vs FSDP2 on `wt-fsdp`, identical HPs.

### 2a — Deterministic / near-bitwise short window (optional but preferred)

Where a fixed synthetic or tiny model + fixed seed is available (see harness
golden pattern in `wt-harness`):

| Metric | Tolerance |
| --- | --- |
| Per-step loss | `abs ≤ 1e-4` (match harness `LOSS_ATOL`) |
| Grad norm | `abs ≤ 1e-4` |
| Param checksum (quantized) | Exact match after gather/`full_tensor()` |

If true bitwise parity is impossible under NCCL nondeterminism, document the
floor and fall through to **2b** with the same seeds and reporting.

### 2b — Algorithm learning runs (required)

Use the debugging configs under `demos/llm/debugging/configs/` where they
exist; otherwise the closest production YAML under
`configs/training/llm_finetuning/`. Prefer a **small HF causal LM + LoRA** for
speed, then one **production-shaped** model per algorithm that matters in prod.

| ID | Algorithm | Loop / demo | DeepSpeed baseline | FSDP2 candidate | Learning pass |
| --- | --- | --- | --- | --- | --- |
| L1 | **GRPO** | `finetune_llm_reasoning` + `grpo_constant_target.yaml` (and/or grid nav) | ZeRO stage used in prod (typically 3) | `fsdp_config` full shard | Eval hit-rate / reward within **±5% relative** of DeepSpeed at matched step; loss MAE over last 20% of steps ≤ **5%** of DeepSpeed loss scale |
| L2 | **LLMPPO** | multiturn / constant-target PPO configs | same | same | Same tolerances on eval + loss |
| L3 | **LLMREINFORCE** | reinforce debug or `reinforce_llm.yaml` | same | same | Same tolerances |
| L4 | **DPO** | `finetune_llm_preference` | same | same | Preference accuracy / chosen-rejected margin within ±5% rel; loss as above |
| L5 | **SFT** | `finetune_llm_sft` / SFT demo path | same | same | Train loss MAE ≤5% over last 20%; held-out perplexity / token-acc within ±5% rel |

**Minimum length:** enough steps that DeepSpeed shows a clear learning trend
(not just init noise). For tiny probes, use the debug config
`max_sample_steps` / eval interval as written; for prod-shaped models, ≥1
epoch or a fixed step budget agreed before the run (record it).

**Seeds:** ≥2 seeds for tiny/debug runs; ≥1 seed for expensive prod-shaped
runs if wall-clock constrained — note the waiver.

**Hard fails (any algorithm):** NaN/Inf loss on any rank; early-return barrier
deadlock; cross-rank seq padding assert; nonfinite allreduce raise that
DeepSpeed does not hit on the same data.

### 2c — Inventory behavioural spot-checks (ALGORITHMIC survival)

These are not full trainings; short scripts or assertions during L1–L5:

| Behaviour | How to confirm |
| --- | --- |
| LoRA-only gather on save/export | Peak VRAM during save stays far below full-model gather; adapter files loadable |
| Cross-rank completion `T` lockstep (GRPO) | Uneven completion lengths still train; no hang |
| Nonfinite loss all-rank check (GRPO) | Injected NaN on one rank aborts all ranks cleanly |
| `fp32_lm_head_operands` / Liger head path | No NaN spike vs DeepSpeed on same fused path |
| Adapter load → next optimizer step | Loaded LoRA weights affect the **immediate** next step (covers removed `_refresh_deepspeed_master_weights`) |
| Fused logprob + frozen lm_head | Forward/backward completes; trainable lm_head policy matches ledger (error or verified path) |

---

## Gate 3 — Speed and memory vs DeepSpeed

Run after Gate 2 configs are stable. Same hardware, model, sequence length,
batch, and step count. Warm up ≥10 steps; measure steady-state.

| ID | Workload | Metrics | Pass |
| --- | --- | --- | --- |
| P1 | GRPO FSDP2 vs ZeRO-3, 2 GPU | ms/step, tokens/sec, peak VRAM | Speed ≥0.90×; VRAM ≤1.10× or justified trade-off |
| P2 | Same as P1 at production GPU count (e.g. 4 or 8) | same | same |
| P3 | SFT or DPO FSDP2 vs ZeRO (whichever is the heavier prod job) | same | same |
| P4 | LLMPPO with value head (if prod uses it) | same | same |
| P5 | Colocated vLLM rollout + learn (GRPO/PPO) if prod uses colocation | learn step time + rollout time + peak VRAM | No regression >10% on either phase; sleep/wake handoff stable |

Report both **mean** and **p95** step time. A single slow first step after
unshard does not fail the gate; sustained regression does.

---

## Gate 4 — Checkpoint, clone, tournament, vLLM

| ID | Scenario | Pass |
| --- | --- | --- |
| C1 | Save under FSDP2 → load on FSDP2 (same world size) | Loss/eval continues; no silent adapter clobber |
| C2 | Save under FSDP2 → load single-GPU / export `save_pretrained` | Weights match gathered reference (checksum or max abs diff ≤ 1e-5 on LoRA tensors) |
| C3 | DeepSpeed checkpoint compatibility | Documented: either supported with converter, or **explicit break** with migration note — no silent partial load |
| C4 | `clone_llm` / population member copy under FSDP2 | Clone trains independently; no shared DTensor alias bugs |
| C5 | Tournament + mutation path (`debugging_llm_training_matrix` pop>1) | Completes; winners’ adapters valid |
| C6 | vLLM adapter sync (LoRA-only gather) | Rollout policy matches trainer after sync; MoE key map if applicable |
| C7 | Resume mid-run (optimizer + step index) | Resumed curve overlays paused curve within Gate 2 loss tolerance |

---

## Gate 5 — High-risk inventory / MoE

Must clear or stay in `blockers.md` with a production waiver.

| ID | Risk (from inventory / blockers) | Required verification | Pass |
| --- | --- | --- | --- |
| R1 | Adapter load without `refresh_fp32_params` | After FSDP2 wrap, load adapter, run 1 step; compare param delta to DeepSpeed | Immediate learning signal matches |
| R2 | Fused logprob lm_head gather / trainable head | Frozen head train OK; trainable head either works under FSDP2 or hard-errors (no silent wrong grads) | Matches documented policy |
| R3 | MoE packed expert LoRA (`blockers.md` #3) | MoE model + `fsdp_config` + `torchrun`: PEFT attach, train ≥N steps, vLLM export | No crash; expert weight reads correct; export loads in vLLM |
| R4 | Gradient checkpointing under FSDP2 | Train with CKPT on; compare loss to DeepSpeed CKPT run | Within Gate 2 tolerances; no empty-shard CKPT failure |

**If R3 fails:** keep MoE + FSDP2 out of production; leave DeepSpeed or block
MoE until fixed. Do not ship a partial MoE path silently.

---

## Gate 6 — Production soak

Final gate before push / prod cutover.

| ID | Run | Pass |
| --- | --- | --- |
| Prod1 | One full production-shaped job per **actively used** algorithm (minimum: the primary GRPO/reasoning job) | Completes without restart; eval within Gate 2 tolerance vs a DeepSpeed reference of equal budget |
| Prod2 | Multi-hour or full scheduled job (or ≥ max(2h, 1 prod epoch)) | No memory leak (VRAM flat after warmup); no progressive slowdown >10% |
| Prod3 | Failure injection: kill one rank mid-run | Clean abort or documented restart procedure works |
| Prod4 | Docs / launch: demos + LLM tutorials say `torchrun`; DeepSpeed accelerate YAML not the LLM default | Reviewer sign-off |
| Prod5 | Classic RL regression: one Accelerate training job green | No collateral damage |

---

## Algorithm coverage checklist

Tick before production:

- [ ] **GRPO** — L1, P1/P2, C6 if colocated, Prod1 if primary
- [ ] **LLMPPO** — L2, P4 if used, value-head config if used
- [ ] **LLMREINFORCE** — L3 (or waiver: not used in prod)
- [ ] **DPO** — L4, P3 if preference is a prod path
- [ ] **SFT** — L5, P3 if SFT is a prod path
- [ ] **MoE split-LoRA** — R3 or explicit “not in prod” waiver in `blockers.md`

---

## Sign-off

| Role | Name | Date | Notes |
| --- | --- | --- | --- |
| Implementation | | | Gates 0–1 |
| Learning + perf | | | Gates 2–3 artifact links |
| MoE / inventory risks | | | Gate 5 / `blockers.md` |
| Production owner | | | Gate 6; approve push / cutover |

**Push criterion:** Gates 0–4 green for every **in-production** algorithm; Gate 5
resolved or waived; Gate 6 Prod1+Prod4 green. Speed must not be compromised
beyond the 0.90× floor without a written production-owner exception.

## Experiment harness

Runnable scaffolding (dry-run by default; **do not execute the suite until
ready**) lives in [`gates/`](gates/README.md):

- `gates/suite.yaml` — job → gate mapping with hardware profile waivers
- `scripts/fsdp_gates/launch.py` — orchestrator + nvidia-smi sampling
- `scripts/fsdp_gates/compare.py` — paired DeepSpeed vs FSDP2 pass/fail
- `scripts/fsdp_gates/run_probe.py` / `run_probe_ds.py` — instrumented tiny probes

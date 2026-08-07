# Expert Parallel Plan — AgileRL

> **Worktree:** `/home/mike/wt-expert-parallel`  
> **Branch:** `migration/expert-parallel`  
> **Do not implement on other worktrees** — especially not `/home/mike/AgileRL` (main FSDP checkout), `/home/mike/wt-context-parallel`, or `/home/mike/wt-zero3`. Context Parallel lives in `wt-context-parallel`; this plan owns EP only.

## Context

AgileRL LLM training is flat data-parallel plus optional FSDP2: `fully_shard` on the
default process group, no `DeviceMesh`, stock HF + PEFT. MoE models use **packed 3D
expert weights** with PEFT `target_parameters` and AgileRL’s split-expert LoRA
(`moe_lora.py`) so adapters stay rank-`r` per expert instead of materializing full
`B @ A` deltas. Under FSDP2, ZeRO-3 leaf modules are gone; attach/upgrade gathers
expert params explicitly (`blockers.md` inventory risk #3).

Prime-RL scales MoE with **Expert Parallel (EP)**: expert weights are `Shard(0)`’d
across an `ep` mesh dim, tokens are routed with all-to-all dispatch/combine, and
FSDP wraps experts on a *smaller* mesh (`dp_shard_mod_ep`) while attention / router
/ non-expert params stay on the full HSDP mesh. Their production recipes commonly
pair `ep=2`–`8` with FSDP (+ optional CP). EP is the lever for **expert-parameter
memory and MoE FSDP all-gather volume**, not sequence length (that is CP).

This plan ports a **trainer-side EP path** into AgileRL for HF packed-expert MoE +
split LoRA, leveraging Prime-RL’s mesh / EP / LoRA-under-EP patterns, without
implementing CP, PP, or dense TP in this worktree.

## Goal

Enable `ep > 1` so each rank holds roughly `1/ep` of the experts (plus FSDP
sharding of remaining dims), routes tokens via EP all-to-all, and trains MoE split
LoRA with numerics matching `ep=1` within bf16 tolerance — **without densifying the
full MoE on any single GPU**.

Success looks like: GRPO (or SFT) on FSDP2 + MoE + split LoRA, `ep=2`, expert
param memory scaling with `ep`, per-token logprobs / loss matching `ep=1` on the
same batch, and vLLM adapter export still loading on the colocated rollout path.

## Current state

**AgileRL** (`/home/mike/wt-expert-parallel`)

| Piece | Where |
|---|---|
| Flat `init_distributed` / rank helpers | `agilerl/utils/distributed.py` |
| `FSDPConfig` + `apply_fsdp2` (**no `mesh=`**) | `distributed.py` (~199–367) |
| CPU → empty local shards → scatter | `materialize_fsdp2_from_cpu_state` / `wrap_models` in `base.py` |
| Packed-expert LoRA attach + gather | `base.py` (~4293–4383): `gather_params(expert_params)` around PEFT + `upgrade_moe_param_wrappers` |
| Split LoRA wrappers | `agilerl/algorithms/core/llm_ops/moe_lora.py` (`SortedExpertsLoraWrapper`, `RoutedExpertsLoraWrapper`) |
| Partitioned-weight guard | `RoutedExpertsLoraWrapper` raises if expert weights are still FSDP `DTensor`s |
| Partitioned LoRA fallback | `_split_lora_delta` routes through adapter Linears when A/B are `DTensor` |
| vLLM MoE key map / export | `llm_utils.py` (`expert_lora_vllm_key_map`, `save_peft_adapter_for_vllm_rollout`); `vllm_colocate.py` |
| MoE under FSDP2 open risk | `docs/migration/blockers.md` #3; remap notes in `fsdp2-remap-ledger.md` |
| DeviceMesh / TP / PP / EP deferred | `docs/migration/success_criteria.md` (v2) |
| Unit coverage (non-EP) | `tests/test_algorithms/test_llm_ops/test_moe_lora.py` |

**Prime-RL (reference)**

| Piece | Where |
|---|---|
| `ep`, `ep_comm_backend` (`torch` \| `deepep`), `resolve_ep("auto")` | `packages/prime-rl-configs/.../trainer.py`; `parallel_dims.resolve_ep` |
| Mesh with EP: `dp_shard_mod_ep`, `dp_shard_in_ep`, `ep`, `hsdp` | `src/prime_rl/trainer/parallel_dims.py` |
| `ExpertParallel` / DeepEP / MXFP8 A2A | `torchtitan…ExpertParallel`; `trainer/distributed/expert_parallel.py` |
| `@expert_parallel` on expert GEMMs + `ep_group` | `trainer/models/layers/moe.py` |
| LoRA under EP (`to_local`, permute, grouped_mm) | `trainer/models/layers/lora/multi_moe.py` |
| FSDP: experts on `dp_mod_ep`, blocks on `hsdp` | `trainer/model.py` `setup_fsdp` / `apply_ep` |
| Docs recipe | `docs/scaling.md` (Expert Parallelism section) |

**Context Parallel worktree (do not implement here)**

| Piece | Where |
|---|---|
| CP plan / Ulysses / `cp` knobs | `/home/mike/wt-context-parallel/docs/migration/context-parallel-plan.md` |

## What EP means here

Expert Parallel (stage 1 definition for AgileRL):

1. **Shard experts across ranks.** Packed expert weight tensors shaped
   `[num_experts, …]` become `DTensor`s with `Shard(0)` on the expert axis over
   an `ep` process group. Each rank owns `num_experts // ep` experts locally
   (`to_local()`).
2. **Tokens are routed.** The router still runs on every rank (replicated or
   FSDP-sharded like today). After top-k, tokens destined for remote experts are
   exchanged.
3. **All-to-all dispatch / combine.** Before expert GEMMs: all-to-all so each
   rank receives only tokens for its local experts. After expert GEMMs: reverse
   all-to-all to return outputs to the originating ranks, then weighted combine
   as today.
4. **FSDP still shards what EP does not.** Non-expert params (attention, router,
   embeddings, lm_head, LoRA on dense layers) use the full FSDP mesh. Expert
   modules use a **reduced** FSDP mesh so EP and FSDP do not double-count the
   same degree (Prime-RL `dp_shard_mod_ep`).

```text
ep=1 (today):  every rank holds all experts; no token A2A for experts
ep>1:          each rank holds E/ep experts; token A2A around local expert compute
```

## Design choices (lock these in)

1. **Stock HF packed-expert path first.** AgileRL stays HF + PEFT — no Prime-RL
   custom `GroupedExperts` stack in stage 1. EP wraps the same modules
   `moe_lora.py` already recognizes (`_is_sorted_experts_module` /
   `_is_routed_experts_module` conventions). Custom / `trust_remote_code` MoE
   layouts that fail those detectors stay unsupported until explicitly added.
2. **Torch EP backend first; DeepEP later.** Stage 1 uses torch
   all-to-all (Prime-RL `ep_comm_backend="torch"` / torchtitan `ExpertParallel`
   semantics). DeepEP / MXFP8 A2A are optional later stages — do not block merge
   on them.
3. **FSDP mesh composition matches Prime-RL’s EP split.** When `ep > 1`:
   - Build a minimal `DeviceMesh` with `dp_shard_mod_ep` × `dp_shard_in_ep`
     (and `cp=1` degenerate).
   - `ep` view = flatten(`dp_shard_in_ep`) (with CP deferred, this is just `ep`).
   - Non-expert FSDP → mesh spanning full shard world (`hsdp` / `dp_shard_cp`).
   - Expert modules → `fully_shard(..., mesh=dp_mod_ep)` **after** EP
     `Shard(0)` parallelize.
4. **`ep > 1` requires `fsdp_config`.** Pure EP without FSDP would still leave
   attention / router / embeddings fully replicated; for any model that needs EP
   for memory, that violates the hygiene invariant. Reject at init.
5. **Split LoRA must run on local expert shards.** Today
   `RoutedExpertsLoraWrapper` errors if packed expert weights are FSDP
   `DTensor`s and expects attach-time gathers. Under EP that model is wrong:
   forward must use `to_local()` expert slices + **local** LoRA slices
   (`[local_E, r, …]`), never gather all experts onto one GPU for the train
   step. Prime-RL `multi_moe.py` is the reference (`to_local` + permute when
   torch EP).
6. **Attach / upgrade gathers stay LoRA-sized or expert-local — never full MoE
   densify on CUDA.** Prefer meta/CPU attach, or gather only the local expert
   shard / LoRA tensors needed for PEFT bookkeeping. Forbidden: “gather all
   experts to CUDA so PEFT can see a dense 3D weight” under `ep > 1`.
7. **CP composition is deferred; do not implement CP here.** Document mesh
   hooks so a later merge with `wt-context-parallel` can add `cp` the Prime-RL
   way (`ep % cp == 0`, `ep` borrows CP ranks). Stage 1 validates only `cp=1`.
   Do not edit the CP worktree from this branch.
8. **vLLM stays full-expert on the inference replica.** Train-only EP; rollout
   loads a complete adapter (gathered LoRA, remapped keys). Importance sampling
   already covers train/infer skew. Do not require infer-side EP for stage 1.
9. **Flat user knobs, not nested configs.** Constructor: `ep: int = 1`.
   INIT_HP: `EP: 2` beside `FSDP: true | {…}` (and eventually `CP` from the
   other plan). Keep `FSDPConfig` as the sole policy dataclass. No
   `ExpertParallelConfig`, `ParallelConfig`, or `DistributionConfig`. Do not put
   `ep` in `HyperparameterConfig` / mutation specs. Optional later:
   `ep_comm_backend: str = "torch"` only when a second backend exists.
10. **CPU/GPU hygiene (hard invariant).** When any trainer parallelism is on
    (`fsdp_config` set and/or `ep > 1`), **never materialize a dense full-MoE
    (or full-model) replica on a single GPU** — not at load, wrap, clone,
    checkpoint, LoRA attach, weight sync, debug gather, or EP setup.

    Concrete rules:

    - **`ep > 1` requires `fsdp_config`.**
    - **Init / wrap stays on the existing path:** CPU (or meta) dense actor →
      `materialize_fsdp2_from_cpu_state` → empty local shards on device →
      scatter. Do not `model.to(cuda)` / `device_map="cuda"` before shard.
    - **EP shards experts; A2A moves tokens/activations**, never full expert
      weight replicas across the EP group for the forward.
    - **Allowed temporary unshards only:** per-module FSDP forward all-gather
      of *that module’s* shard group; LoRA-sized / touched-param gathers; rank-0
      **CPU** full state for checkpoint/export
      (`get_state_dict(..., cpu_offload=True)`). Forbidden: training-step
      `gather_full_params` of all experts onto CUDA, GPU full-state dict,
      clone/broadcast that densifies MoE on device.
11. **No ZeRO-3 leaf modules.** FSDP2 has no `set_z3_leaf_modules`. EP +
    explicit local-shard MoE forwards replace the old leaf workaround. Clearing
    `blockers.md` #3 for the non-EP path remains a prerequisite or parallel
    verification (Gate R3); EP must not reintroduce “treat experts as leaves
    and hope.”

## Mesh (Prime-RL-shaped, AgileRL-minimal)

For stage 1 (no CP, no PP, no `dp_replicate`):

```text
dp_shard == world_size
ep divides dp_shard
dp_shard_mod_ep = dp_shard // ep
dp_shard_in_ep  = ep
```

Named views:

| Mesh | Dims | Role |
|---|---|---|
| `dp` / `dp_shard_mod_ep` (+ in-ep when flattened for data) | data-parallel batch world | Batch / sampler size — **EP ranks that share a microbatch still see the same tokens** (router runs everywhere; experts are sharded) |
| `hsdp` / `dp_shard_cp` | full `dp_shard` (× `cp=1`) | FSDP for attention, router, embeddings, lm_head, dense LoRA |
| `dp_mod_ep` | `dp_shard_mod_ep` | FSDP for expert modules only (Prime-RL) |
| `ep` | `dp_shard_in_ep` (= `ep` when `cp=1`) | Expert `Shard(0)` + token all-to-all group |

`ep=1` keeps today’s flat process-group FSDP path (no mandatory DeviceMesh) —
same “no Phase-0 mesh theater” discipline as the CP plan: prove expert-shard +
A2A numerics first; add `fully_shard(mesh=)` when composing with FSDP+EP.

Constraints to validate at config time:

- `ep > 1` requires `fsdp_config`
- `world_size % ep == 0`
- `num_experts % ep == 0` (reject otherwise in stage 1; no uneven expert packs)
- Model must expose packed-expert modules recognized by `moe_lora` detectors
  (or a thin registry of supported HF MoE families)
- Non-MoE models: `ep > 1` fails loud (or hard-force `ep=1` only if we add
  `ep="auto"` later — stage 1 can require explicit int and MoE)
- Do **not** require CP-related `seq_len % (2*cp)` checks in this worktree

### Deferred CP×EP composition (document only)

When CP lands from the other worktree, follow Prime-RL:

```text
ep % cp == 0
(dp_shard * cp) % ep == 0
ep mesh = flatten(dp_shard_in_ep × cp)
```

Do not implement CP, Ulysses, or `cp` knobs on `migration/expert-parallel`.

## Train-step shape (target)

Per microbatch when `ep > 1`:

1. All ranks in the DP/FSDP world run the shared non-expert stack (attention,
   norms) under FSDP on `hsdp`.
2. Router produces top-k indices / weights on every rank (same tokens per DP
   replica; EP peers that are pure expert shards of one logical replica share
   the microbatch — match Prime-RL’s EP island semantics).
3. **Dispatch:** all-to-all tokens (and metadata) over `ep` so each rank holds
   only rows for its local experts; build local `num_tokens_per_expert` /
   sorted-row counts for the **local** expert count.
4. **Local experts + split LoRA:** `to_local()` packed weights and LoRA A/B;
   run sorted/grouped GEMM paths from `moe_lora.py` over **local** `E' = E/ep`
   experts (update counts / offsets accordingly).
5. **Combine:** all-to-all expert outputs back; weighted `index_add_` combine
   as in `RoutedExpertsLoraWrapper` today.
6. Backward through combine → local experts/LoRA → dispatch; FSDP reduces grads
   on the appropriate meshes (`dp_mod_ep` for experts, `hsdp` for the rest).
7. **No Prime-RL `fsdp_gradient_divide_factor` un-divide** while AgileRL keeps
   local mean loss — same lock as the CP plan.

## Interaction with packed 3D weights + PEFT split LoRA

| Concern | Stage-1 rule |
|---|---|
| PEFT `target_parameters` 3D layout | Keep; LoRA A/B stay stacked on expert dim, then EP-`Shard(0)` with base experts |
| `upgrade_moe_param_wrappers` | Must understand EP-local shapes; upgrade after EP parallelize + FSDP, or upgrade on CPU meta and re-bind — pick one order and test (lock in Phase 1) |
| `_is_partitioned` / gather error | Replace “must gather all experts” with “use `to_local()` under EP; gather only when exporting” |
| `_split_lora_delta` DTensor path | Prefer `to_local()` + grouped path over the memory-heavy Linear fallback under EP |
| Attach-time `gather_params(expert_params)` | Revisit under EP: cannot gather full expert tensors to CUDA; attach on CPU/meta or local shards only |
| `blockers.md` #3 | Non-EP MoE+FSDP2 correctness remains Gate R3; EP work must not paper over it with full densify |

## Grad sync / optimizer under EP+FSDP

- **Expert grads:** FSDP reduce-scatter on `dp_mod_ep` only (EP peers do *not*
  all-reduce full expert grads into one dense expert replica).
- **Non-expert / dense LoRA grads:** existing FSDP or `sync_grads` behavior on
  the full shard mesh.
- **Optimizer construction after wrap:** unchanged AgileRL rule — build AdamW
  (etc.) after FSDP+EP so state matches DTensor shards. No need for Prime-RL
  Muon expert mesh param groups in stage 1 unless AgileRL already uses Muon.
- **Grad norms / clipping:** if any global norm spans expert + non-expert
  params, ensure EP ranks contribute correctly (Prime-RL passes
  `ep_enabled=` into clip). Port an equivalent only if AgileRL’s clip path
  double-counts or drops expert grads under EP — verify in numerics tests.
- **Do not** average expert LoRA grads across the EP group as if every rank
  owned the same expert (that would be wrong under `Shard(0)`).

## vLLM rollout / adapter export under EP

| Path | Rule |
|---|---|
| Train | EP-sharded experts + local LoRA shards |
| Export / colocated sync | Gather **LoRA-only** (and only as needed) to build the PEFT adapter files; prefer CPU gather |
| Key map | Keep `expert_lora_vllm_key_map` — keys are logical full-expert indices; gathered LoRA must be reassembled to full `E` ordering before remap |
| `patch_vllm_3d_moe_lora_flag` | Unchanged; inference engine still sees fused MoE + full adapter |
| Base weights in vLLM | Rollout engine holds the full base MoE (or its own TP); train EP does not shard the vLLM replica |
| Forbidden | Loading EP local shards directly into vLLM without assemble; CUDA full-base gather for sync |

## Config knobs

| Knob | Surface | Default | Notes |
|---|---|---|---|
| `ep` | LLM algo ctor | `1` | Explicit int in stage 1 |
| `EP` | INIT_HP / YAML beside `FSDP` | absent → 1 | Flat; no nested block |
| `fsdp_config` / `FSDP` | existing | required if `ep > 1` | |
| `ep_comm_backend` | defer | `"torch"` | Add only when DeepEP lands |

Reject examples (loud `ValueError`):

- `ep > 1` without `fsdp_config`
- `world_size % ep != 0` or `num_experts % ep != 0`
- `ep > 1` on non-MoE / unsupported packed layout
- `ep > 1` with features that force full expert densify (document any temporary
  rejects: e.g. multi-adapter expert LoRA already rejected today)

## Phased plan

### Phase 0 — EP group + shard placement proof (no product config)

**Exit criteria**

- Minimal DeviceMesh / process group helpers for `ep` and `dp_mod_ep` (or
  equivalent) living under `agilerl/utils/` (names TBD).
- Unit test: `Shard(0)` on a fake `[E, out, in]` parameter yields
  `local_E = E/ep` per rank; `full_tensor()` round-trips on CPU.
- Tiny all-to-all dispatch/combine round-trip on random tokens + expert ids
  (`ep=2`, 2 GPU) matches single-process reference.

### Phase 1 — HF packed experts forward under EP (LoRA off)

**Exit criteria**

- Hook/wrapper around supported HF packed-expert modules: local GEMM + torch
  A2A; `ep=2` vs `ep=1` same-batch hidden states / logprobs within bf16 tol.
- No CUDA full-MoE materialize (assert peak expert param bytes ∝ `1/ep`).
- Document locked **attach order**: EP parallelize vs PEFT attach vs FSDP wrap.

**Locked attach order** (`EP_ATTACH_ORDER` in `agilerl/utils/expert_parallel.py`):

1. `cpu_dense` — load / build actor on CPU (or meta)
2. `peft_attach` — PEFT LoRA on CPU (no CUDA full-expert gather when `ep > 1`)
3. `upgrade_moe_wrappers` — split-LoRA class swap on CPU
4. `ep_shard` — `Shard(0)` packed experts (+ LoRA expert axes) on `ep` mesh
5. `fsdp2_materialize` — `materialize_fsdp2_from_cpu_state` (experts → `dp_mod_ep` in Phase 3)

### Phase 2 — Split LoRA under EP  ← critical path for AgileRL

**Exit criteria**

- `SortedExpertsLoraWrapper` / `RoutedExpertsLoraWrapper` operate on
  `to_local()` base + LoRA; remove/replace the “must gather packed experts”
  train-step error for the EP path.
- Forward + backward: `ep=2` LoRA grads vs `ep=1` within bf16 tol (same batch).
- Attach/upgrade path never densifies full experts on CUDA when `ep > 1`.

### Phase 3 — FSDP2 compose + hygiene  ← stage 1 done when green

**Exit criteria**

- `apply_fsdp2` (or sibling) accepts mesh: experts on `dp_mod_ep`, blocks on
  `hsdp`; `ep=1` preserves today’s flat PG behavior.
- Stay on `materialize_fsdp2_from_cpu_state`; hygiene gate on shard param bytes.
- 2-GPU smoke: MoE + FSDP + `ep=2` GRPO (or SFT) ≥N steps; loss finite; no NCCL hang.
- Numerics: FSDP `ep=2` vs FSDP `ep=1` same-batch logprob/loss gate.

### Phase 4 — Flat config + vLLM export

**Exit criteria**

- `ep: int = 1` ctor + INIT_HP `EP` passthrough next to `FSDP`.
- Fail-loud validation suite green.
- Adapter export: gathered LoRA → `expert_lora_vllm_key_map` → vLLM load;
  colocated sync smoke if MoE+vLLM is in scope for the model under test.

### Phase 5 — Product surface / docs (stage 1b)

- Docs: when to set `EP`, interaction with FSDP, memory expectations, unsupported
  layouts; point at CP plan for sequence scaling.
- Optional gate script under `scripts/fsdp_gates/` for `ep ∈ {1,2}` × MoE probe.
- Revisit `blockers.md` #3 with EP+FSDP evidence; close or waive explicitly.

### Later (separate plans — not this worktree)

- DeepEP / MXFP8 A2A backends
- CP×EP mesh merge (implement CP only in `wt-context-parallel`)
- Dense TP, PP
- `ep="auto"` resolution (Prime-RL style)

## File touch list (implementation later — not this PR)

| Area | Likely files |
|---|---|
| Mesh / EP helpers | `agilerl/utils/distributed.py` (or new `agilerl/utils/expert_parallel.py`) |
| FSDP mesh apply | `distributed.py` `apply_fsdp2` / `wrap_models` call sites in `base.py` |
| MoE LoRA under EP | `agilerl/algorithms/core/llm_ops/moe_lora.py` |
| Attach / upgrade order | `base.py` `_initialize_actors` MoE gather block |
| Export / vLLM | `llm_utils.py` gather/assemble for full-E LoRA; `vllm_colocate.py` if needed |
| Config surface | LLM algo `__init__` + INIT_HP plumbing (GRPO/SFT/… as applicable) |
| Tests | `tests/test_algorithms/test_llm_ops/test_moe_lora.py` + new `test_expert_parallel*.py` |
| Docs / gates | this plan; `blockers.md`; optional `scripts/fsdp_gates/` |

## Interaction with other work

| Workstream | Interaction |
|---|---|
| FSDP2 migration | EP adds `mesh=` only when `ep > 1`; `ep=1` keeps flat PG FSDP |
| Packed expert LoRA / blockers #3 | Must verify or waive non-EP MoE+FSDP2; EP redesigns gather assumptions |
| Context Parallel (`wt-context-parallel`) | Out of scope here; mesh comments reserve Prime-RL CP×EP composition |
| Activation offload | Complementary (activations); EP reduces expert **param** memory |
| Sequence packing / Liger | No special EP packing rules beyond existing FSDP lockstep; do not couple to CP packing work |

## Testing and evaluation plan

Behavior-first, following existing MoE LoRA and FSDP2 unit styles
(`test_moe_lora.py`, `test_fsdp2_offload_units.py`).

### Unit tests

| ID | Test | Pass |
|---|---|---|
| U1 | `TestEpMeshLayout` — `world_size`, `ep` → `dp_shard_mod_ep`, `ep` group size; invalid dims raise | Exact sizes; clear errors |
| U2 | `TestExpertShardPlacement` — `Shard(0)` on `[E, …]`; each rank `local_E = E/ep`; CPU `full_tensor` round-trip | Shape + allclose |
| U3 | `TestTokenDispatchCombine` — A2A round-trip preserves token→expert routing vs single-rank reference | Max abs err ≤ bf16 tol |
| U4 | `TestSplitLoraLocalExperts` — LoRA delta on local shards matches reference slice of full-E LoRA | allclose |
| U5 | `TestNoDenseMoEOnCuda` — wrap/attach under mocked or 2-GPU EP never allocates full `E` expert weight bytes on CUDA | Peak expert param bytes ≤ `(E/ep)·…·dtype` (+FSDP slack) |
| U6 | Config validation — `ep>1` without FSDP; bad `num_experts`; non-MoE | Raises |

### Numerics

| ID | Comparison | Pass |
|---|---|---|
| N1 | Forward logprobs: `ep=2` vs `ep=1`, **same batch**, FSDP on, MoE+split LoRA | Per-token max \|Δ\| within bf16 tol (record floor if NCCL noise) |
| N2 | One backward: loss + LoRA grad norms / checksums (gathered on CPU) | Same tolerance class as FSDP2 Gate 2a where feasible |
| N3 | Short train (N steps): loss curve overlay `ep=2` vs `ep=1` | MAE over last 20% ≤ 5% of loss scale (or documented NCCL floor) |

Golden rule (same as CP plan): compare `ep=2` vs `ep=1` on the **same logical
batch**, not “2-GPU EP” vs “2-GPU pure DP with different data split” unless the
DP semantics are intentionally identical.

### Memory

| ID | Metric | Pass |
|---|---|---|
| M1 | Expert **parameter** bytes resident after wrap | Scales ≈ `1/ep` vs `ep=1` (same world size / FSDP settings) |
| M2 | Peak allocated during forward | Expert weight peak still ≈ `1/ep`; expect **higher** activation traffic from A2A — record bytes/step, do not fail solely on A2A activation cost |
| M3 | Hygiene: no full-MoE CUDA densify during attach, step, export | Export may CPU-gather LoRA; peak CUDA during export stays LoRA-sized |

### Multi-GPU smoke matrix

| ID | Setup | Pass |
|---|---|---|
| S1 | MoE probe, 2 GPU, FSDP, `ep=1` | Completes (regression baseline) |
| S2 | MoE probe, 2 GPU, FSDP, `ep=2` | Completes; no hang |
| S3 | MoE + split LoRA, FSDP, `ep=2`, ≥N train steps | Finite loss; optimizer steps |
| S4 | Dense (non-MoE) + `ep=2` | Reject at init |
| S5 | MoE + `ep=2` without `fsdp_config` | Reject at init |
| S6 | Optional: colocated vLLM sync after S3 | Adapter loads; rollout runs |

### Regression vs current non-EP MoE LoRA

| ID | Check | Pass |
|---|---|---|
| R1 | Existing `test_moe_lora.py` suite | Green with `ep=1` defaults |
| R2 | Non-EP FSDP MoE path (blocker #3 / Gate R3) | Still passes or remains explicitly open in `blockers.md` — EP merge must not worsen |
| R3 | vLLM key map unit tests | Green; EP export assemble covered by new test |

### Gates before merge

| Gate | Requirement |
|---|---|
| G0 | Unit suite U1–U6 green in CI (CPU / 1-GPU where possible) |
| G1 | N1 + M1 on ≥2 GPU |
| G2 | S1–S5 smoke matrix green |
| G3 | Phase 3 FSDP+EP numerics N2 (or N3) green |
| G4 | Phase 4 config + export path green; docs in this file match flags |
| G5 | Explicit sign-off that CP was **not** implemented in this worktree; no edits under `wt-context-parallel` / `AgileRL` main |

Do not merge with only “A2A runs” — **N1 + M1 + S3** are the minimum product bar.

## Risks

- **HF MoE ≠ Prime-RL custom MoE.** Stock transformers packed experts do not use
  `@expert_parallel`; wrappers must intercept the right module forwards without
  forking an entire modeling stack — highest delivery risk.
- **Attach-order / PEFT assumptions.** PEFT and `upgrade_moe_param_wrappers`
  assume dense 3D expert views; EP forces local shards. Wrong order → silent
  wrong LoRA or OOM densify.
- **blockers.md #3 interaction.** If non-EP FSDP2 MoE LoRA is still broken,
  EP-on-FSDP will inherit the bug; verify or waive before advertising MoE+EP.
- **Collective lockstep.** EP A2A + FSDP collectives amplify desync risk when
  per-rank token counts or shapes diverge — validate counts before dispatch.
- **Load imbalance.** Skewed expert assignment makes A2A and local GEMM uneven;
  stage 1 accepts imbalance (measure); capacity-factor / dropless policies are
  out of scope unless HF already implements them.
- **Train/infer skew.** EP train vs full-expert vLLM increases IS gap; measure
  before calling MoE+EP+vLLM production-ready.
- **Scope creep into DeepEP / CP / custom models.** Stage 1 is torch A2A + HF
  packed experts + split LoRA + FSDP meshes only.

## Out of scope

- Implementing **Context Parallel** (Ulysses/ring, `cp` knobs, sequence shard) —
  owned by `/home/mike/wt-context-parallel`
- Dense trainer **TP**, **PP**, `dp_replicate` HSDP islands (unless a trivial
  mesh stub is required for Prime-RL-shaped math — prefer not)
- DeepEP / MXFP8 all-to-all backends (later)
- Custom Prime-RL modeling stack as a hard dependency for stage 1
- Infer-side expert parallel in vLLM
- Flipping `ep > 1` on by default / `ep="auto"`
- EP without FSDP (full non-expert replica) — rejected by hygiene invariant
- Product code in this documentation-only change set

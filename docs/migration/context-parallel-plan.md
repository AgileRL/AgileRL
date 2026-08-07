# Context Parallel Plan — AgileRL

> **Worktree:** `/home/mike/wt-context-parallel`
> **Branch:** `migration/context-parallel`
> **Do not implement on other worktrees** (`/home/mike/AgileRL`, `/home/mike/wt-expert-parallel`, `/home/mike/wt-zero3`, or any other checkout). All CP code and doc edits land here only.

## Context

AgileRL LLM training is flat data-parallel plus optional FSDP2: `fully_shard` on the
default process group, no `DeviceMesh`, stock HF + PEFT, fused logprob / packing on a
**full local sequence**. Long context is currently fought with gradient checkpointing
and (separately) activation offload — see `activation-offload-plan.md`, which explicitly
defers CP.

Prime-RL trains at 32k–131k by sharding the **sequence** across a `cp` mesh dim, with
two attention styles (`ulysses` / `ring`), FSDP on a mesh that includes CP ranks, and
loss/token reductions over `dp_cp`. Their production recipes use `cp=2`–`8` alongside
FSDP + EP for MoE; CP is the lever for context length, not model width. Docs:
`prime-rl/docs/scaling.md` § Context Parallelism.

This plan ports the **trainer-side CP path** into AgileRL for **both** Ulysses and
ring attention, leveraging Prime-RL’s mesh and train-step patterns, without taking
EP, PP, or dense TP in the same program. Implementation is sequenced: Ulysses is the
stage-1 primary path; ring is a first-class stage-2 deliverable in this same plan
(not a vague deferral).

## Goal

Enable `cp > 1` so a single logical microbatch’s tokens are split across CP ranks,
cutting activation / attention memory roughly with `cp`, while numerics match `cp=1`
within bf16 tolerance — for **both** `cp_style="ulysses"` and `cp_style="ring"`.

**Stage-1 success (Ulysses):** GRPO on FSDP2 + FA2, **B=1 varlen/packed** forward under
Ulysses, `cp=2`, mid-seq memory win, per-token logprobs matching `cp=1` on the same
batch.

**Stage-2 success (ring):** Same GRPO contract with `cp_style="ring"`, FA2 +
`ring-flash-attn` llama3-style path, numerics match `cp=1` and (where heads allow)
match Ulysses within bf16 tolerance on the same batch. Packing/SFT/docs sweeps are
stage 1b / 2b — not exit criteria for the first green of each style.

## Current state

**AgileRL** (`/home/mike/wt-context-parallel`)

| Piece | Where |
|---|---|
| Flat `init_distributed` / rank helpers | `agilerl/utils/distributed.py` |
| `FSDPConfig` + `apply_fsdp2` (**no `mesh=`**) | `distributed.py` (~199–367) |
| `wrap_models` → FSDP → optim → HF grad checkpoint | `base.py` (~3271–3317) |
| Fused / packed / no-grad forwards | `base.py` (`_fused_model_pass`, `_fused_packed_forward`, `_get_logprobs`, …) |
| Packing (within-rank FA2 varlen / flex) | `agilerl/utils/llm_packing.py`, `_packing_mode` in `base.py` |
| Cross-rank equal-`T` padding for FSDP/Liger | `llm_utils.py` (`needs_cross_rank_seq_padding`) |
| Batch size ÷ `world_size` | `base.py` `_configure_batch_size_per_process` |
| Colocated vLLM TP forced to 1 | `base.py` (~2777–2790, `_configure_vllm`) |
| DeviceMesh / TP / PP / EP deferred | `docs/migration/success_criteria.md` (v2) |

**Prime-RL (reference — read-only)**

| Piece | Where |
|---|---|
| `cp`, `cp_style` (`ring` \| `ulysses`) + validators | `packages/prime-rl-configs/.../trainer.py` |
| `ParallelDims` mesh: `dp`, `dp_shard_cp`, `dp_cp`, `hsdp` | `src/prime_rl/trainer/parallel_dims.py` |
| `seq_len_divisor` (= `2 * cp`, PyTorch CP load-balance note) | `parallel_dims.seq_len_divisor` |
| Shard / gather / `setup_cp_params` | `src/prime_rl/utils/cp.py` |
| Ulysses all-to-all FA (custom + HF monkeypatch) | `trainer/models/layers/ulysses_attn.py` |
| Ring FA (llama3-style + FA3/FA4 adapters) | `ring-flash-attn` + `trainer/models/layers/ring_attn.py`, `attn.substitute_ring_attn` |
| RL/SFT train-step CP wiring | `trainer/rl/train.py`, `trainer/sft/train.py` |
| FSDP on `hsdp` (includes CP) | `trainer/model.py` `setup_fsdp` |
| Docs recipe | `docs/scaling.md` (CP section + memory-tight recipe) |

## Design choices (lock these in)

1. **Ulysses first, ring second — both in this plan.** Ulysses all-to-alls Q/K/V so
   local FA runs on the full sequence with `H/cp` heads. It matches stock HF softmax
   attention with a monkeypatch, and is what Prime-RL recommends for most models
   (`docs/scaling.md`). Ring (llama3-style via `ring-flash-attn`) keeps full heads
   and communicates K/V around the CP ring; it is stage 2, with explicit deps, FA
   constraints, and a model support matrix below — not “later / maybe.”
2. **Default `cp_style="ulysses"`.** Prime-RL’s config default is historically
   `ring`; AgileRL defaults to `ulysses` because it is stage 1, works with stock
   FA2 + HF, and matches Prime-RL’s recommendation for most models. Users opt into
   `ring` after stage 2 ships.
3. **FSDP mesh includes CP** (Prime-RL `hsdp` / `dp_shard_cp`), not “FSDP on DP only.”
   CP ranks co-shard parameters for the same logical microbatch. Stage 1–2 keep
   AgileRL’s local per-sequence mean loss: FSDP AVG on `dp_shard_cp` is enough —
   **do not** copy Prime-RL’s `grad.mul_(fsdp_gradient_divide_factor)` or global
   token renorm until the loss reduction is rewritten end-to-end.
4. **Data-parallel size excludes CP.** Batch split and dataloader world use
   `dp_size = world_size // cp`. CP ranks in a group see the **same** microbatch,
   then shard the sequence.
5. **Stock HF + PEFT path only.** No custom modeling stack in this plan. Softmax
   models: attention monkeypatch + tensor shard in the train step. Models that need
   `set_context_parallel_attributes` (Mamba / linear-attn / sparse MLA) are **out of
   scope** until those architectures are first-class trainers.
6. **EP / PP / dense TP out of scope for this plan.** CP + FSDP only. Do not
   pre-build unused `ep`/`pp` fields — add mesh dims when EP lands.
7. **vLLM stays full-sequence, colocated TP=1.** Train-only CP; importance sampling
   already exists for train/infer logprob skew. Do not block CP on infer CP/TP.
8. **Flat user knobs, not nested configs.** Constructor: `cp: int = 1`,
   `cp_style: Literal["ulysses", "ring"] = "ulysses"`. INIT_HP: `CP` / `CP_STYLE`
   beside existing `FSDP: true | {…}`. Keep `FSDPConfig` as the sole policy
   dataclass. No `ContextParallelConfig`, `ParallelConfig`, or `DistributionConfig`.
   Do not put `cp` in `HyperparameterConfig` / mutation specs.
9. **CPU/GPU hygiene (hard invariant).** When any trainer parallelism is on
   (`fsdp_config` set and/or `cp > 1`), **never materialize a dense full-model
   replica on a single GPU** — not at load, wrap, clone, checkpoint, LoRA
   attach, weight sync, or debug gather. (Concrete rules below.)

## Style comparison

| | Ulysses | Ring (llama3 / `ring-flash-attn`) |
|---|---|---|
| Mechanism | All-to-all heads ↔ seq; local FA on full `S`, `H/cp` heads | Keep heads; all-gather / ring K/V; local FA on shard + remote KV |
| Attention kernel | Stock `flash_attn_varlen_func` (FA2) after a2a | `llama3_flash_attn_varlen_func` (FA2); Prime-RL also has FA3/FA4 adapters for *custom* attn |
| HF patch surface | `ALL_ATTENTION_FUNCTIONS["flash_attention_2"]` + `_flash_attention_forward` | `ring_flash_attn.substitute_hf_flash_attn` (+ AgileRL must also patch `ALL_ATTENTION_FUNCTIONS` on newer transformers) |
| Head constraint | `H % cp == 0`; GQA: `H_kv % cp == 0` **or** `cp % H_kv == 0` (KV replicate) | No head÷cp requirement from Ulysses math; still FA2 softmax-only |
| Seq length | `N % cp == 0` | `N % cp == 0`; prefer `N % (2*cp) == 0` when using zigzag / load-balanced llama3 (match Prime-RL `seq_len_divisor`) |
| Hybrid / Mamba / linear attn | Kernel-agnostic in principle; **out of scope** here (needs model hooks) | **Rejected** — softmax ring only (`assert_cp_style_supports_model`) |
| Comms pattern | 2× all-to-all per layer (fwd) | All-gather (or ring) of K/V chunks per layer |
| When to pick | Default; most dense softmax HF models | When Ulysses head divisibility fails, or to A/B throughput; some families only make sense with ring in Prime-RL (e.g. GLM-5 custom) — AgileRL HF path: optional style |

### Model support matrix (AgileRL HF+PEFT)

| Model class | Ulysses | Ring | Notes |
|---|---|---|---|
| Dense softmax CausalLM (Llama/Qwen2/Qwen3/… FA2) | ✅ stage 1 | ✅ stage 2 | Primary target |
| GQA with `H_kv < cp` | ✅ if KV replicate | ✅ (no head÷cp) | Lock KV replicate for Ulysses (see critic locks) |
| SWA / softcap (Gemma2-style) | ❌ reject | ❌ reject | Patch asserts `softcap is None`; SWA needs separate validation |
| SDPA / eager / flex attn | ❌ | ❌ | `cp>1` requires FA2 |
| Liger fused CE + CP | ❌ until contracted | ❌ | Gate until shapes/logprob gather defined |
| Hybrid Mamba / DeltaNet / linear attn | ❌ out of scope | ❌ reject | Needs `set_context_parallel_attributes`; ring unsupported in Prime-RL |
| Sparse MLA / DSA | ❌ out of scope | ❌ out of scope | Prime-RL custom hooks only |
| VLM / multimodal | ❌ out of scope | ❌ | Prime-RL forces ulysses for VLM; not in AgileRL scope |
| MoE (EP) | ❌ this plan | ❌ | Separate EP plan; CP may compose later |

### FA / dependency constraints (from Prime-RL)

| Dep | AgileRL stage | Constraint |
|---|---|---|
| `flash-attn` (FA2) | Stage 1+ | **Required** for `cp>1`. Pin in `agilerl[llm]` (or llm+cp extra). Refuse `cp>1` if import/`_attn_implementation` is not FA2. |
| `ring-flash-attn` (≥0.1.8, as Prime-RL) | Stage 2 | **Required** only when `cp_style="ring"`. Lazy-import; fail loud at init if missing. Import order: apply any compat patch **before** importing `ring_flash_attn` (Prime-RL `prime_rl._compat` pattern — port a minimal shim if needed). |
| FA3 (`flash_attn_interface`) | Not required | Prime-RL ring/ulysses support FA3 on *custom* stacks. AgileRL HF path stays on FA2 for both styles in this plan. |
| FA4 (`flash_attn.cute`) | Not required | Same — out of scope for AgileRL CP v1/v2. |
| `attn_implementation` | Both | `cp>1` ⇒ `"flash_attention_2"` only. Reject `sdpa` / `eager` / `flex_attention` / FA3/FA4 strings until explicitly added. |

Prime-RL config rule (reference): `cp>1` requires `attn in {flash_attention_2,3,4,auto}` and custom/`auto` impl. AgileRL maps that to: **stock HF + FA2 only**, both styles.

## Mesh / ParallelDims (Prime-RL-shaped, AgileRL-minimal)

For this plan (no EP, no PP, no `dp_replicate`):

```text
dp_shard * cp == world_size
```

Implement a small helper (name suggestion: `ParallelDims` or `build_cp_meshes`) in
`agilerl/utils/` — **not** a full TorchTitan port. Only build a `DeviceMesh` when
`cp > 1` (and FSDP needs `mesh=`). `cp=1` keeps today’s flat process-group FSDP.

Named views:

| Mesh | Dims | Role |
|---|---|---|
| `dp` | `dp_shard` | Batch / sampler world (excludes CP) |
| `dp_shard_cp` | `dp_shard` × `cp` | Flattened FSDP shard mesh (`hsdp` when no replicate) |
| `cp` | `cp` | Sequence shard + Ulysses/ring collectives |
| `dp_cp` | `dp_shard` × `cp` | Loss / metric reductions that must include every token owner |

`cp=1` builds a degenerate path equivalent to today’s flat world so the default is
a numerical no-op (no mandatory DeviceMesh).

Constraints to validate at config / model-init time:

- `cp > 1` requires `fsdp_config` (no full-weight replica mode)
- `world_size % cp == 0`
- `cp_style in {"ulysses", "ring"}`
- Packed / unpadded token length `N % cp == 0` (both styles)
- Ring: also enforce `N % (2 * cp) == 0` when using llama3 zigzag / load-balanced
  path (align with Prime-RL `seq_len_divisor`); pad packed rows to that multiple
- Ulysses: `num_attention_heads % cp == 0`; GQA: implement KV replicate when
  `cp % H_kv == 0`, else reject
- Ring: reject models with linear-attn / Mamba layers (Prime-RL
  `assert_cp_style_supports_model`); for AgileRL HF dense softmax this is usually N/A
- `cp > 1` requires installed FA2 and `_attn_implementation == "flash_attention_2"`
- `cp_style="ring"` requires importable `ring_flash_attn`
- `cp > 1` ⇒ effective forward `B == 1` (packed row or single sequence); reject
  fused multi-adapter batching / value-head `repeat(N)` under CP

## Config knobs

| Knob | Type | Default | Surface |
|---|---|---|---|
| `cp` | `int` | `1` | LLM algo `__init__` (GRPO/SFT/… via `LLMAlgorithm`) |
| `cp_style` | `"ulysses" \| "ring"` | `"ulysses"` | Same; ignored when `cp==1` |
| INIT_HP `CP` | int | unset → 1 | `utils.py` / population init beside `FSDP` |
| INIT_HP `CP_STYLE` | str | unset → `"ulysses"` | Same |

Validation (fail loud at init, before wrap):

1. `cp > 1` and `fsdp_config is None` → error (hygiene)
2. `world_size % cp != 0` → error
3. Missing FA2 / wrong attn impl → error
4. `cp_style="ring"` without `ring-flash-attn` → error
5. Ulysses head / GQA rules → error
6. Liger + `cp>1` → error until contracted
7. Packing mode `flex` + `cp>1` → error until FA2 packing path is CP-safe
8. `cp_style` unknown → error

Do **not** nest these under `FSDPConfig`. FSDP policy stays independent; CP is a
topology/train-step concern that *composes* with FSDP via mesh.

## Train-step shape (target)

Shared by both styles when `cp > 1`:

1. CP peers share an identical microbatch (same tokens, same shuffle, same early-exit).
2. **Shift-then-shard** for next-token targets (Prime-RL): build labels/masks aligned
   so hidden[t] scores token[t+1] **before** `shard_for_cp` — never apply AgileRL’s
   `hidden[:, :-1]` / `ids[:, 1:]` on a CP shard alone (drops boundary tokens).
3. Publish full-seq attention params **every** forward (grad, no-grad, checkpoint
   recompute):
   - Ulysses: `cu_seqlens` / `max_seqlen` into `ULYSSES_PARAMS`
   - Ring: `update_ring_flash_attn_params(cu_seqlens, cp_group)` → `DATA_PARAMS`
     including `local_k_slice`
4. `shard_for_cp` on ids / positions / labels / action masks / temperatures.
5. Forward `B=1` on local `S/cp` tokens; patched attention restores full-seq semantics.
6. **Gather logprobs** (grad: differentiable gather; old/ref: no-grad gather) then run
   full-sequence GRPO (or SFT) loss on every CP rank. Shard-local loss is out for
   stages 1–2.
7. Backward through style-specific attention collectives + FSDP on `dp_shard_cp`
   (no Prime-RL grad un-divide).

Style selection at wrap / first CP enable:

```text
if cp > 1:
    if cp_style == "ulysses":
        substitute_hf_ulysses_attn(cp_group)   # + ALL_ATTENTION_FUNCTIONS
    elif cp_style == "ring":
        substitute_hf_flash_attn(cp_group, heads_k_stride=1)
        # ensure ALL_ATTENTION_FUNCTIONS["flash_attention_2"] routes correctly
```

Only one style patched per process; switching mid-run is unsupported.

## Loss / grad reduction

AgileRL today: per-sequence (or per-token then mean) losses computed locally after
full-seq logprobs; FSDP2 uses AVG reduction over the shard mesh.

Under CP with **gather-logprobs + full-seq loss on every CP rank**:

- Every CP rank in a group computes the **same** scalar loss → FSDP AVG over
  `dp_shard_cp` would over-count by `cp` if we naïvely treated CP ranks as independent
  data owners. Two acceptable stage-1/2 fixes (pick one and lock in code review):
  1. **Preferred for minimal change:** keep full-seq loss on all CP ranks but run
     FSDP / grad sync in a way that matches today’s mean-loss semantics — e.g. scale
     the loss by `1/cp` **or** use a reduction mesh that does not double-count
     identical CP replicas. Document the chosen scale in the PR.
  2. **Do not** import Prime-RL’s `fsdp_gradient_divide_factor` + global token renorm
     without rewriting AgileRL’s loss to token-sum / world-token-normalize form.

`dp_cp` reductions are for metrics (token counts, summed CE) that must see every
token once — use them when aggregating *sharded* local stats; with full gather +
identical loss, prefer logging from one CP rank per DP group (e.g. `cp_rank==0`).

Golden check: `cp=2` vs `cp=1` **same batch**, same DP size — loss and per-token
logprobs within bf16 tol; grad norms match within tol after the chosen scale.

## Packing / varlen / fused logprob

- **Varlen / packed FA2:** pack on the **full** sequence on every CP rank (identical
  pack), publish full `cu_seqlens`, then shard the packed row. Pad packed length to
  a multiple of `cp` (and `2*cp` for ring zigzag).
- **Flex packing:** out of CP until FA2 path is green.
- **Fused logprobs:** still run on local hidden shard then **gather** logprob
  tensors along seq before GRPO advantages / clips. Chunked fused LM-head stays
  valid on the local shard (chunk size independent of CP) as long as targets were
  shift-then-shard aligned.
- **Fused multi-adapter / value-head batch stacking:** disabled when `cp>1`
  (sequential ref/old/actor forwards).
- **Cross-rank equal-T padding** (`needs_cross_rank_seq_padding`): still applies
  across **DP** ranks; CP peers must already share identical `T` (same microbatch).
  Pad to `cp` / `2*cp` multiple before shard.

## vLLM interaction

- CP is **train-only**. Rollouts and vLLM engines keep full sequences.
- Colocated inference: keep existing **TP=1** force in `_configure_vllm`.
- Weight sync / LoRA gather paths must obey hygiene (CPU or LoRA-sized gathers only).
- Expect larger train/infer logprob skew at long context; rely on existing importance
  sampling. Measure IS ratios before advertising production CP+vLLM (see risks).
- Do not implement infer-side CP or train–infer TP alignment in this plan.

## CPU/GPU hygiene (hard invariant)

When `fsdp_config` is set and/or `cp > 1`, **never materialize a dense full-model
replica on a single GPU** — load, wrap, clone, checkpoint, LoRA attach, weight sync,
or debug gather.

Concrete rules:

- **`cp > 1` requires `fsdp_config`.** Pure CP without FSDP would keep a full weight
  replica on every rank; reject at init.
- **Init / wrap stays on the existing path:** CPU (or meta) dense actor →
  `materialize_fsdp2_from_cpu_state` → empty local shards on device → scatter
  (`base.py` `wrap_models`, `distributed.py`). Do not `model.to(cuda)` /
  `device_map="cuda"` before shard. `apply_fsdp2` already documents this contract.
- **FSDP mesh includes CP** so param shards stay `1/(dp_shard·cp)` per rank;
  Ulysses/ring collectives move **activations** (Q/K/V), never full weights.
- **Allowed temporary unshards only:** per-module FSDP forward all-gather
  (reshard after forward when configured); LoRA-sized / touched-param gathers;
  rank-0 **CPU** full state for checkpoint/export
  (`get_state_dict(..., cpu_offload=True)`). Forbidden: training-step
  `gather_full_params` onto CUDA, GPU full-state dict, clone/broadcast that
  densifies the actor on device.
- **CP microbatch identity** means ranks share the same *tokens*, not the same
  *dense weights*. Sequence shard ≠ weight replicate.
- **Style-agnostic:** ring’s KV all-gather is activation traffic; it must not become
  an excuse for parameter densification.

## Critic locks (must hold before coding)

From post-plan review ([scope](fe2d1954-93ae-4ff5-9e36-301ca7334a6a),
[correctness](4323b2d4-4e2b-42a4-b0c6-39bbbede92e9),
[feasibility](8a2fb622-5e1c-4e3c-8d1f-c795b8bf697c)), extended for ring:

1. **No Phase-0 mesh theater.** Prove Ulysses+shard numerics first; add
   `fully_shard(mesh=)` only when wiring FSDP+CP. `cp=1` users keep today’s flat PG
   path (no mandatory DeviceMesh).
2. **Finish / productize activation offload as the default long-context lever;** CP
   is an additive escape hatch, not AO’s substitute.
3. **Add + pin `flash-attn` (FA2)** in `agilerl[llm]`; refuse `cp>1` without it.
   Stage 2: pin `ring-flash-attn` for `cp_style="ring"`.
4. **Patch `ALL_ATTENTION_FUNCTIONS["flash_attention_2"]`** (transformers ≥4.x/5.x
   routing); module-attr patch alone is a silent no-op. Applies to **both** styles.
5. **Forward shape under CP: `B==1` varlen** (pack or unpad). Multi-row padded GRPO
   is out until a separate padded path exists. Disable fused multi-adapter /
   value-head batch stacking when `cp>1`.
6. **Logprob contract: shift-then-shard + gather-logprobs + full-seq loss** on all
   CP ranks (both styles).
7. **Grad math: no `× fsdp_gradient_divide_factor`** with current AgileRL mean loss;
   lock an explicit `1/cp` (or equivalent) scale instead of copying Prime-RL renorm.
8. **Topology:** `dp_size`/`dp_rank` for batch, sampler, env increment; `cp` for
   shard/collectives; CP peers share RNG shuffle. Golden test = `cp=2` vs `cp=1`
   **same batch** (not 2gpu cp=2 vs 2gpu cp=1 DP).
9. **GQA (Ulysses):** implement KV replicate (`_replicate_kv_heads`) when
   `cp % H_kv == 0`; else reject. Do not leave this binary open past Phase 0.
10. **Reject list (both styles unless noted):** SDPA/flex, SWA/softcap, Liger+CP,
    EP, CP without FSDP, left-pad train rows, exotic `trust_remote_code` attn,
    FA3/FA4 CP paths, hybrid/Mamba. **Ring-only reject:** linear-attn/Mamba models.

## Phased implementation

### Phase 0 — Ulysses + shard proof (no FSDP, no product config)

**Work**

- Add/pin FA2; port `shard_for_cp` / `gather_for_cp` / `gather_for_cp_wo_grad`
- Port Ulysses helpers + `substitute_hf_ulysses_attn` with
  `ALL_ATTENTION_FUNCTIONS` patch
- Tiny CausalLM, `B=1` varlen/packed, `cp=2` vs `cp=1` same batch

**Exit criteria**

- max \|Δlogprob\| and loss within bf16 tol (not “patch runs”)
- Missing `ALL_ATTENTION_FUNCTIONS` patch fails a negative test
- GQA policy locked (KV replicate implemented or reject path tested)

### Phase 1 — GRPO `B=1` wire (may still be non-FSDP for numerics)

**Work**

- Shift-then-shard; publish `cu_seqlens` every forward; gather logprobs; full-seq GRPO
- Sequential ref/old/actor forwards when `cp>1`
- `dp_size` for batch/sampler; identical CP-peer shuffle

**Exit criteria**

- GRPO loss/`cp=2` vs `cp=1` same-batch tol on ≥2 GPUs
- No fused multi-adapter path entered under CP (assert / unit)

### Phase 2 — FSDP2 compose + hygiene  ← **Ulysses stage 1 done when green**

**Work**

- Minimal `DeviceMesh` only as needed for `fully_shard(..., mesh=dp_shard_cp)`
- Stay on `materialize_fsdp2_from_cpu_state`; `cp>1` requires `fsdp_config`
- Loss scale / reduction locked; no Prime-RL grad un-divide
- Hygiene gate on shard param bytes

**Exit criteria**

- 2-GPU GRPO FSDP `cp=2` ulysses: mid-seq mem win vs `cp=1`, per-token logprob match
- Peak CUDA param bytes after wrap scale with `1/world_size`, not full model
- Checkpoint CPU export + clone path smoke under CP (no dense CUDA gather)

### Phase 3 — Flat config + style knob surface

**Work**

- `cp: int = 1`, `cp_style: str = "ulysses"` ctor + INIT_HP `CP` / `CP_STYLE`
- Fail loud: FA2, heads, `N % cp`, packing/Liger/SWA rejects, `cp>1` without FSDP,
  ring without `ring-flash-attn` (even if ring not wired yet — validate early)
- One smoke script — not a sweep framework

**Exit criteria**

- Config unit tests green; invalid combos raise before NCCL work

### Phase 4 — Ring attention (stage 2 primary)  ← **ring stage 2 done when green**

**Work**

- Add/pin `ring-flash-attn>=0.1.8`; compat import shim if required
- Wire `cp_style="ring"`: `substitute_hf_flash_attn` + `ALL_ATTENTION_FUNCTIONS`
  routing; `update_ring_flash_attn_params` in `setup_cp_attention_params`
- Enforce ring seq divisor (`2*cp` when zigzag); reject hybrid/linear-attn
- Reuse the same shard/gather/shift-then-shard/GRPO/FSDP mesh path as Ulysses

**Exit criteria**

- Numerics: `cp=2` ring vs `cp=1` same batch within bf16 tol
- Cross-style: ring vs ulysses on the same batch within bf16 tol (models where
  both are legal)
- Memory: mid-seq and long-seq VRAM ≤ ulysses or documented tradeoff; no hygiene
  regression
- Multi-GPU smoke: `cp=2` (and `cp=4` if ≥4 GPUs) with FSDP

### Phase 5 — Packing + FA2 varlen (stage 1b / 2b)

**Work**

- Make `_fused_packed_forward` / `pack_padded_batch` CP-safe for both styles
- Pad packed length to `cp` / `2*cp` multiples
- Keep flex-attention packing out of CP

**Exit criteria**

- Packed GRPO smoke `cp=2` ulysses + ring; logprob match vs unpacked `cp=1`

### Phase 6 — SFT + product surface

**Work**

- `SFT._sft_loss`: shard labels / loss mask; gather or careful `dp_cp` reduce
- Docs: when to set `cp` / `cp_style`, interaction with FSDP and activation offload,
  head checklist, FA2 + ring-flash-attn deps
- Optional: `scripts/fsdp_gates/` CP sweep
  (`cp ∈ {1,2}` × `cp_style ∈ {ulysses,ring}` × `seq_len ∈ {4096,16384}`)

**Exit criteria**

- SFT smoke under FSDP+CP both styles; docs merged; gate IDs recorded

### Later (separate plans — not this document’s implementation)

- Expert parallel (EP) mesh dims and MoE
- FA3/FA4 CP paths; Prime-RL custom ring adapters
- Hybrid/Mamba / sparse MLA CP hooks
- Infer CP / vLLM TP alignment
- Padded multi-row Ulysses (B>1)

## File-level touch list (AgileRL)

| Area | Files (expected) | Change |
|---|---|---|
| CP primitives | **new** `agilerl/utils/cp.py` (or `agilerl/utils/context_parallel.py`) | `shard_for_cp`, gather, `setup_cp_attention_params`, style enum, model support asserts |
| Ulysses | **new** `agilerl/utils/ulysses_attn.py` (or under `agilerl/algorithms/core/llm_ops/`) | a2a helpers, KV replicate, HF + `ALL_ATTENTION_FUNCTIONS` patch |
| Ring | **new** `agilerl/utils/ring_attn.py` (thin wrapper) | call into `ring_flash_attn`; ensure HF registry patch; no FA3/FA4 port unless needed |
| Mesh | `agilerl/utils/distributed.py` + small **new** mesh helper | `ParallelDims`-lite; `apply_fsdp2(..., mesh=)`; `dp_size` helpers |
| FSDP apply | `agilerl/utils/distributed.py` | Pass `device_mesh=dp_shard_cp` when `cp>1` |
| Train wiring | `agilerl/algorithms/core/base.py` | knobs, wrap-time substitute, shift-then-shard, publish params, gather logprobs, disable fused batching under CP |
| Batch / DP | `base.py` `_configure_batch_size_per_process` | divide by `dp_size` not `world_size` when CP on |
| Packing | `agilerl/utils/llm_packing.py`, `base.py` `_fused_packed_forward` | full-seq pack → shard; pad to cp/2cp |
| Algo ctors | `grpo.py`, `sft.py`, `dpo.py`, `ppo_llm.py`, `reinforce_llm.py` | plumb `cp` / `cp_style` |
| INIT_HP | `agilerl/utils/utils.py` | `CP`, `CP_STYLE` beside `FSDP` |
| Deps | `pyproject.toml` / extras | pin `flash-attn`; optional/required `ring-flash-attn` for ring |
| Tests | **new** `tests/test_algorithms/test_context_parallel*.py` (+ extend FSDP hygiene tests) | see testing plan |
| Gates / docs | `docs/migration/gates/*`, this plan, user-facing LLM scaling notes | CP gate IDs; style matrix |
| Smoke | `scripts/` or `demos/llm/debugging/` | `torchrun` CP smoke both styles |

Do not edit Prime-RL. Do not implement in other AgileRL worktrees.

## Interaction with other work

| Workstream | Interaction |
|---|---|
| FSDP2 migration | Mesh only when `cp>1` needs it; `cp=1` keeps today’s flat process-group FSDP |
| Activation offload | Prefer finishing AO productization first; CP is complementary (live activation width), not a substitute. Do not block AO on mesh work |
| Sequence packing | Phase 5; do not enable packing+CP before Phase 2/4 green for the style under test |
| Liger / fused losses | Explicitly gate until logprob gather / shape contracts are defined |
| MoE / EP | Out of scope; separate plan — no placeholder mesh dims |
| Expert-parallel / ZeRO3 worktrees | No shared edits; CP lands only in `wt-context-parallel` |

## Testing and evaluation plan

Follow behavior-first patterns from `tests/test_algorithms/test_fsdp2_offload_units.py`
and the gate style in `docs/migration/gates/` + `docs/migration/success_criteria.md`.

### Unit tests (CPU / 1-GPU mocked where possible)

| ID | Test | Assert |
|---|---|---|
| U1 | `TestParallelDimsMath` | `dp_shard * cp == world_size`; submesh sizes; `cp=1` no-op; reject `world_size % cp != 0` |
| U2 | `TestShardForCp` | Round-trip shard+gather; wrong `S % cp` raises; shift-then-shard boundary tokens present after gather |
| U3 | `TestShardForCpRingDivisor` | Ring path rejects / pads to `2*cp` when zigzag enabled |
| U4 | `TestUlyssesMonkeypatch` | `ALL_ATTENTION_FUNCTIONS["flash_attention_2"]` replaced; negative test: attr-only patch fails correctness helper |
| U5 | `TestRingMonkeypatch` | `ring_flash_attn` substitute + registry routing; import without compat fails loudly if shim required |
| U6 | `TestCpStyleSelection` | `ulysses` vs `ring` selects correct substitute; unknown style raises; `cp==1` skips patch |
| U7 | `TestCpConfigValidation` | FA2 required; `cp>1` without `fsdp_config` raises; Liger/flex/SWA rejects; ring without dep raises |
| U8 | `TestGqaKvReplicate` | Ulysses KV replicate layout + grad sum; reject when `cp % H_kv != 0` and `H_kv % cp != 0` |

### Numerics (multi-GPU)

Golden rule: compare **same logical batch**, not “2-GPU DP vs 2-GPU CP.”

| ID | Comparison | Tolerance (bf16) | Notes |
|---|---|---|---|
| N1 | `cp=1` vs `cp=2` ulysses | max \|Δlogprob\| and \|Δloss\| within agreed bf16 band (start: `1e-2` abs on logprobs or document measured floor) | Tiny HF CausalLM + LoRA; B=1 varlen |
| N2 | `cp=1` vs `cp=2` ring | same | Requires `ring-flash-attn` |
| N3 | `cp=2` ulysses vs `cp=2` ring | same | Only models legal for both |
| N4 | Grad norm parity after locked loss scale | relative ≤ few % | Catches forgetting `1/cp` scale |
| N5 | Checkpoint recompute path | logprob match with grad checkpoint on | Ensures `ULYSSES_PARAMS` / ring `DATA_PARAMS` republished |

### Memory / perf eval

| ID | Shape | Metrics | Pass idea |
|---|---|---|---|
| M1 | Mid-seq (e.g. 4k–8k), `cp=1` vs `cp=2` ulysses | peak allocated / reserved VRAM, step_ms | CP peak activations/attn memory ↓ vs `cp=1` at same per-rank logical seq; no unexplained full-model param spike |
| M2 | Long-seq (e.g. 16k+ as hardware allows), both styles | same | `cp=2` completes where `cp=1` OOMs **or** clear VRAM win; record step_ms tradeoff |
| M3 | Ring vs ulysses at same `cp` | VRAM + step_ms | Document winner; no hygiene fail |
| M4 | Hygiene wrap probe | CUDA param bytes after wrap | Scales ≈ `1/world_size` |

Use `torch.cuda.max_memory_allocated` + optional `nvidia-smi` sampler pattern from
`docs/migration/gates/README.md`.

### Multi-GPU smoke matrix

| World | FSDP | CP | Style | Expect |
|---|---|---|---|---|
| 1 | off | 1 | n/a | baseline (no CP) |
| 2 | on | 1 | n/a | today’s FSDP |
| 2 | on | 2 | ulysses | stage-1 primary smoke |
| 2 | on | 2 | ring | stage-2 primary smoke |
| 2 | **off** | 2 | either | **must error** at init |
| 4 | on | 2 | ulysses | dp_shard=2, cp=2 |
| 4 | on | 4 | ulysses (+ ring if heads allow) | dp_shard=1, cp=4 |
| 4 | on | 3 | either | **must error** (`world_size % cp`) |

Run with `torchrun --nproc_per_node=N`. On 2×T4 gates hardware, prioritize world=2
rows; waive world=4 with note in `blockers.md` style if no larger box.

### Integration

| ID | Area | Check |
|---|---|---|
| I1 | GRPO | Full short run FSDP+CP ulysses; then ring |
| I2 | SFT | Phase 6; shard labels / loss finite; checkpoint reload |
| I3 | Packing | Phase 5; FA2 varlen pack+CP both styles |
| I4 | Checkpoint / clone | Save CPU full state; load under CP; clone/tournament path does not densify CUDA |
| I5 | vLLM colocate | TP remains 1; weight sync works; record IS ratio smoke (not a merge blocker for stage 1) |
| I6 | LoRA + PEFT | Adapter forward under both styles; export loadable |

### Merge gates (tie to `docs/migration/gates`)

Add CP jobs to the gate suite (names illustrative — match `suite.yaml` style):

| Gate | Meaning | Merge blocker for |
|---|---|---|
| **CP0** | Unit suite U1–U8 green in CI (CPU / skipped GPU) | any CP merge |
| **CP1** | Phase 0–2 exit criteria (Ulysses+FSDP numerics + hygiene + mid-seq mem) | Ulysses stage 1 |
| **CP2** | Phase 4 exit criteria (ring numerics + smoke + deps) | Ring stage 2 |
| **CP3** | Packing I3 + SFT I2 for enabled style | packing/SFT follow-ups |
| **CP4** | Optional perf sweep M1–M3 artifacts under `/tmp/…` | docs claim / “production CP” advertising |

Do **not** declare production-ready CP until CP1 (and CP2 if ring is advertised) are
green or explicitly waived with owner + expiry (same discipline as
`success_criteria.md`).

### Explicit non-goals / out of scope

- EP, PP, dense trainer TP, `dp_replicate` HSDP
- Custom hybrid/Mamba/VLM/sparse-MLA CP hooks
- Infer-side context parallel / vLLM CP
- FA3/FA4 AgileRL CP paths (Prime-RL custom adapters)
- Changing activation-offload defaults
- Flipping `cp > 1` on by default
- CP without FSDP (full weight replica per rank)
- Blind copy of Prime-RL token renorm / `fsdp_gradient_divide_factor`
- Multi-row padded `B>1` Ulysses
- Nested `ContextParallelConfig` / mutation of `cp` via hyperparam search
- Implementing or editing code in `/home/mike/AgileRL`, `wt-expert-parallel`, or `wt-zero3`

## Risks / open questions

| Risk | Mitigation |
|---|---|
| **Collective lockstep:** non-divisible `T` deadlocks Ulysses a2a or ring KV | Validate `N % cp` (+ `2*cp` for ring) before forward; pad packed rows |
| **Hygiene regressions** on clone/checkpoint/LoRA/MoE attach | Extend `TestFSDP2NoDenseCudaMaterialize`; forbid GPU full gather in CP paths |
| **LoRA + DTensor + head patch** | Ulysses/ring must not break PEFT or fused LM head identity patch; gather only touched/LoRA-sized params |
| **GQA `H_kv < cp`** | KV replicate on Ulysses; ring as escape hatch when heads block Ulysses |
| **Loss double-count across CP ranks** | Lock scale (`1/cp` or mesh choice) + N4 grad-norm golden |
| **Train/infer skew** with long CP train vs full-seq vLLM | Measure IS before prod claims |
| **`ring-flash-attn` / FA2 version skew** | Pin versions beside Prime-RL’s known-good floor (`ring-flash-attn>=0.1.8`, FA2 wheel compatible with torch) |
| **Transformers attention routing drift** | Always patch `ALL_ATTENTION_FUNCTIONS`; add version-sensitive tests |
| **Scope creep into custom models** | HF-first only; resist porting Prime-RL’s full custom stack |
| **Open: exact bf16 atol** for N1–N3 | Measure on tiny model in Phase 0; record floor in test constants |
| **Open: loss scale choice** (loss `* 1/cp` vs reduction group) | Decide in Phase 2 PR description; one approach only |
| **Open: zigzag mandatory for ring?** | Prefer match Prime-RL `2*cp` padding; confirm against installed `ring-flash-attn` llama3 defaults in Phase 4 |

## Out of scope (summary)

- EP, PP, dense trainer TP
- Custom hybrid/Mamba/VLM CP hooks; infer CP
- FA3/FA4 CP on AgileRL; Prime-RL custom-only model impl requirement
- Changing AO defaults; default-on `cp>1`
- CP without FSDP
- Token-renorm loss rewrite (unless a follow-up plan owns it)

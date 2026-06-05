# Scoping: per-token policy-version provenance for async / off-policy RL

Status: **scoping / design** (not implemented). Inert while colocated; built to
avoid retrofitting token-level provenance onto a batch-level buffer later
(the "architectural debt" the HF async-RL landscape post warns against).

## Goal

Tag every generated token with the **integer policy version** that produced it,
so that — once generation and training overlap (decoupled Ray async) — the
trainer can:

1. **Measure staleness** — distribution of `current_version - token_version`
   (mean / p95 / max lag), correlated with the existing `vllm_is_*` mismatch
   metrics.
2. **Gate by staleness** — hard-drop (or down-weight) tokens whose lag exceeds a
   `max_staleness` threshold *before* the loss (the "depth bounding" half of the
   recommended hybrid).
3. **Truncated importance sampling (TIS)** — we already ship this (the clamped
   `clamp(exp(old - sampling), max=cap)` reweight); the version tag is what makes
   it staleness-aware and enables per-token admission.

Per the landscape post, production systems (PRIME-RL, AReaL, open-instruct,
NeMo-RL, TorchForge, verl, SLIME) converge on **per-sample/per-token integer
version tags + bounded-queue depth gating + clipped TIS**. We already have TIS;
this scopes the provenance + gating.

## Granularity: per-token (not per-turn)

A per-*turn* tag cannot represent in-flight weight swaps (PipelineRL "never-stop":
weights change *between forward passes during a single sequence*), so the token
is the only piecewise-constant unit. It also costs us nothing extra: our
`sampling_logps` is **already a per-token vector** scattered onto the action
mask, so `model_version` is a parallel per-token `int32` vector on the identical
path.

## The plumbing already exists — this mirrors `sampling_logps` exactly

`model_version` rides the same path WS8 (`ActionResult`) and the sampling-mismatch
work built for `sampling_logps`:

| Stage | `sampling_logps` (today) | `model_version` (proposed) |
|---|---|---|
| Source | vLLM logprob per generated token | trainer's `policy_version` counter |
| Stamp | `_generate_with_vllm_colocate` | same, broadcast current version to tokens |
| Carrier | `ActionResult.sampling_logps` | new `ActionResult.model_versions` field |
| Multi-turn accum | `SyncMultiTurnVecEnv._sampling_logps_buf` | new `_model_versions_buf` (same shape) |
| Rollout return | `get_trajectories()` → `all_sampling_logps` | `→ all_model_versions` |
| Into trainer | `learn(..., sampling_logps=...)` | `learn(..., model_versions=...)` |
| Align | scatter onto action-mask `True` positions | identical scatter |

Adding a field to `ActionResult` (a `NamedTuple`) does not break attribute-access
callers — exactly why WS8 chose A2.

## Version source (the counter)

- `LLMAlgorithm.policy_version: int`, starts at 0.
- **Incremented on each trainer→engine adapter sync** — the LoRA→vLLM sync point
  (`_move_lora_to_vllm` / the `[agilerl lora-sync]` hook visible in benchmark
  logs). Each sync = a new policy version that subsequent rollouts sample from.
- Decoupled async: the counter lives on the trainer; the rollout engine holds
  `(version, adapter)` and stamps generations with the version it currently has.

## Data shape

`model_versions`: `list[torch.Tensor | None] | None` — one 1-D `int32` tensor per
trajectory (generated tokens, chronological; concatenated across turns), aligned
to `action_mask.sum()` exactly like `sampling_logps`. Storage ≈ one int per
action token (negligible, parallel to the float `sampling_logps` already stored).

## What `learn()` does with it

- **Metrics (always):** `staleness = current_version - model_versions` over action
  tokens → log `staleness_mean / p95 / max`, and `frac_stale` past threshold.
- **Admission gate (opt-in `max_staleness`):** zero the action mask for tokens
  with `staleness > max_staleness` before the loss (the loss already masks, so
  this is a mask edit — no special-casing downstream).
- **TIS (already shipped):** unchanged; the gate + metrics layer on top.

## Phasing

- **Phase 0 — provenance plumbing (cheap, no-op while colocated).**
  Add the `policy_version` counter (+ bump on sync), the `ActionResult.model_versions`
  field, the env buffer, the `learn()` param, alignment helper, and staleness
  metrics. Lag is always 0 colocated, so this changes no numerics — it just
  establishes provenance so async doesn't require a re-plumb.
- **Phase 1 — queue-based async (generate-then-train).** Version is constant per
  generation call (no mid-sequence swap), so per-token = the call's version
  broadcast — **needs no engine change**. Wire the Ray experience queue to carry
  version; add the trainer-side `max_staleness` drop gate; log the staleness
  distribution. This is the regime most frameworks ship.
- **Phase 2 — true in-flight (PipelineRL-style).** Requires (a) engine support to
  tag each *decode step* with the active weight version, and (b) using the
  **recorded per-token logprobs as `π_old`** instead of the trainer's recompute
  (with weights changing mid-sequence there is no single coherent `π_old` to
  recompute). Bigger lift; defer until in-flight updates are on the roadmap.

## Boundaries / caveats

- **Inert until async.** Colocated weight-sharing (today's benchmark) is lag-0;
  build everything as a no-op when `model_versions is None` or staleness ≡ 0.
- **Staleness gate is split.** Bounded-queue *admission* (`max_async_level` /
  `max_staleness_steps`) is an orchestration concern in the Ray integration repo
  (the versioned experience queue); the trainer-side per-token drop is the
  loss-level safety net. Both consume the same version tag.
- **Phase 2 changes the loss reference.** Folding recorded logprobs in as `π_old`
  is a semantic change to `_compute_policy_loss` — scope separately.

## Files (mirrors WS8 + the sampling-mismatch work)

- `agilerl/algorithms/core/base.py` — `policy_version` counter + bump in the LoRA
  sync; stamp version in `_generate_with_vllm_colocate`; `ActionResult` gains
  `model_versions`; an `_align_token_versions` helper (parallel to the existing
  `_align_sampling_logprobs`).
- `agilerl/algorithms/grpo.py` — `learn(..., model_versions=...)`; staleness
  metrics; optional `max_staleness` drop gate.
- `agilerl/llm_envs/sync_vec_env.py` — `_model_versions_buf` + step/get_trajectories
  plumbing (mirror `_sampling_logps_buf`).
- `agilerl/rollouts/on_policy.py` — forward `action_result.model_versions`.
- `agilerl/training/train_llm.py` — pass-through (parallel to `sampling_logps`).
- Ray integration (separate repo) — version-carrying experience queue + bounded
  admission gate (Phase 1+).

# Qwen2.5-0.5B GRPO ConstantTarget — CP/EP smoke results

**Date:** 2026-08-07  
**Host:** mikeyp-l4x2 (2× NVIDIA L4 24GB)  
**Python:** `/tmp/agilerl-venv` — torch `2.11.0+cu130`, flash-attn `2.8.3`, transformers `5.14.1`, ring-flash-attn `0.1.8` (+ compat shim)  
**Harness:** `scripts/fsdp_gates/run_probe.py` + `demos/llm/debugging/configs/grpo_constant_target.yaml`  
**Model:** `Qwen/Qwen2.5-0.5B` (cached)  
**GPU lock:** `/tmp/agilerl-smoke/GPU_LOCK.md`

Metric note: probe logs `train/mean_score` (ConstantTarget reward; 1.0 = hit).  
`summary.json` `final_eval` is derived from the last logged score.

---

## Summary table

| Job | Worktree | Config | Exit | Wall | Learning signal | Outcome |
|-----|----------|--------|------|------|-----------------|---------|
| `Q05B-FSDP-base` | CP | FSDP, `CP=1` | **0** | **158s** | `mean_score` → **1.0**, `final_eval=1.0` | **Learning OK** |
| `Q05B-CP2-ulysses-fix` | CP | FSDP, `CP=2`, `CP_STYLE=ulysses` | **0** | **248s** | `mean_score` −1→**1.0**, `final_eval=1.0` | **Learning OK** |
| `Q05B-CP2-ring-fix` | CP | FSDP, `CP=2`, `CP_STYLE=ring` | **0** | **248s** | `mean_score` −1→**1.0**, `final_eval=1.0` | **Learning OK** |
| `Q05B-FSDP-ep1` | EP | FSDP, `EP=1` | **0** | **158s** | `mean_score` → **1.0** | **Learning OK** |
| `Q05B-EP2-dense` | EP | FSDP, `EP=2` on dense Qwen | **1** | 16s | none | **FAIL** expected: not MoE |

---

## Per-job detail (CP)

### 1. `cp/Q05B-FSDP-base` — FSDP baseline (CP=1)

- **Artifact:** `/tmp/agilerl-smoke/cp/Q05B-FSDP-base/`
- **Exit / wall:** 0 / ~158s
- **summary.json:** `final_eval=1.0`, `final_loss=0.0`
- **Verdict:** Clear ConstantTarget success under FSDP2.

### 2. `cp/Q05B-CP2-ulysses-fix` — Context Parallel Ulysses

- **Artifact:** `/tmp/agilerl-smoke/cp/Q05B-CP2-ulysses-fix/`
- **Command:**
  ```bash
  /tmp/agilerl-venv/bin/torchrun --standalone --nproc_per_node=2 \
    scripts/fsdp_gates/run_probe.py \
    --job-id Q05B-CP2-ulysses-fix \
    --artifact-dir /tmp/agilerl-smoke/cp/Q05B-CP2-ulysses-fix \
    --config demos/llm/debugging/configs/grpo_constant_target.yaml \
    --model-name Qwen/Qwen2.5-0.5B \
    --max-steps 120 --warmup-steps 2 --fsdp --log-stdout \
    --init-hp CP=2 --init-hp CP_STYLE=ulysses \
    --init-hp BATCH_SIZE=8 --init-hp MICRO_BATCH_SIZE_PER_GPU=1 --init-hp GROUP_SIZE=2
  ```
- **Exit / wall:** 0 / **248s** (`summary.wall_s`≈248s)
- **Metrics** (`train_metrics.jsonl`):
  - early: `mean_score=-1.0` → `-0.75` → `-0.125` → `0.375`
  - step ~224+: `mean_score=1.0`, `mean_kl≈49.5`, `mean_loss=0.0`
- **summary.json:** `final_eval=1.0`, `final_loss=0.0`, peak alloc ≈1.6GB
- **Fixes that unblocked:**
  - Stock FA2 fallback when Ulysses `cu_seqlens` unpublished (HF generate / rollouts)
  - Global `position_ids` forced + recomputed after CP pad; omit shard `attention_mask` on CP train forward
  - Broadcast rollouts across CP group (CUDA sampling not bit-identical)
  - CP peers share RNG (`seed += rank // cp`); dp-sized step accounting
  - No extra `loss/cp` under FSDP AVG on identical CP replicas
- **Note:** Transient KL spike (~3.5e5) mid-run then recovers to ~50 (baseline-like). Prefer `max-steps 120` for this batch; 60 was marginal.
- **Verdict:** **Learning OK** for Ulysses `CP=2`.

### 3. `cp/Q05B-CP2-ring-fix` — Context Parallel ring

- **Artifact:** `/tmp/agilerl-smoke/cp/Q05B-CP2-ring-fix/`
- **Command:** same as Ulysses with `CP_STYLE=ring`
- **Exit / wall:** 0 / **248s**
- **Metrics:** same learning pattern — `mean_score` → **1.0**, `final_eval=1.0`, `mean_kl≈49.5`
- **Fixes that unblocked:**
  - `agilerl/utils/ring_attn_compat.py` shims `is_flash_attn_greater_or_equal_2_10` for transformers 5.14 + ring-flash-attn 0.1.8
  - `substitute_cp_attention("ring")` → `substitute_hf_flash_attn` + `use_ring_attn` gated around train forwards
- **Verdict:** **Learning OK** for ring `CP=2`.

---

## Verdicts

| Question | Answer |
|----------|--------|
| Does AgileRL FSDP GRPO still learn on Qwen 0.5B ConstantTarget? | **Yes** (~2.5 min baseline) |
| Does Ulysses `CP=2` learn? | **Yes** (`final_eval=1.0`, ~4 min @ max-steps 120) |
| Does ring `CP=2` learn? | **Yes** (`final_eval=1.0`, ~4 min @ max-steps 120) |
| Does `EP=2` learn on this model? | **No** — correctly rejected (not MoE) |

---

## Raw artifacts

```
/tmp/agilerl-smoke/
  GPU_LOCK.md
  RESULTS.md                 ← this file
  cp/Q05B-FSDP-base/         (ok)
  cp/Q05B-CP2-ulysses-fix/  (ok)
  cp/Q05B-CP2-ring-fix/     (ok)
```

# Expert Parallel — hub MoE GRPO ConstantTarget smoke

**Date:** 2026-08-07  
**Host:** mikeyp-l4x2 (2× NVIDIA L4 24GB)  
**Worktree:** `/home/mike/wt-expert-parallel` (`migration/expert-parallel`)  
**Python:** `/tmp/agilerl-venv` — torch `2.11.0+cu130`  
**Harness:** `scripts/fsdp_gates/run_probe.py` + `demos/llm/debugging/configs/grpo_constant_target_hub_moe.yaml`  
**GPU lock:** `/tmp/agilerl-smoke/gpu.lock.d`

Metric note: probe logs `train/mean_score` (ConstantTarget reward; 1.0 = hit).  
Each GRPO outer iter under this config advances **256** sample-steps
(`BATCH_SIZE=16 × GROUP_SIZE=8 × world_size=2`).

---

## Summary

| Job | Model | Config | Exit | Wall | Learning signal | Outcome |
|-----|-------|--------|------|------|-----------------|---------|
| `hub-moe-ep2` | `PrimeIntellect/qwen3-moe-tiny` | FSDP, `EP=2` | **0** | **714s** | score **-0.51 → 1.0** by step 1280 | **Learning OK** |

---

## Model

**Hub packed-expert MoE** (not the local random fixture):

| Field | Value |
|-------|-------|
| Hub id | `PrimeIntellect/qwen3-moe-tiny` |
| Download cache | `/tmp/hf-home/models--PrimeIntellect--qwen3-moe-tiny` (**~1.3 GB**) |
| Architecture | `Qwen3MoeForCausalLM` |
| Experts | `num_experts=16`, `num_experts_per_tok=4` (divisible by `EP=2`) |
| Depth / width | 24 layers, `hidden_size=1024`, `moe_intermediate_size=256` |
| Vocab | 151936 (HF tokenizer; digit `"3"` → token id **18**) |
| Packed experts | `gate_up_proj` `(16, 512, 1024)`, `down_proj` `(16, 1024, 256)` |

Fallback hub ids were not needed (`shibatch/tinyqwen3gatedmoe3m` 401; preferred model fit in `/tmp`).

### ConstantTarget learning prior

Raw hub weights have ~0 ConstantTarget hit-rate on prompt `"11"` at T=1.0
(`P("3")≈3e-5`). The probe materializes a `/tmp/hf-home/ctprior-*` snapshot with
`digit_logit_boost=10.0` on ASCII digit lm_head rows (~0.09 measured hit-rate /
`P("3")≈0.12`) before FSDP/PEFT wrap. HF tokenizer + `ConstantTargetEnv(target_digit="3")`
(substring reward on decoded text).

---

## Green job: `ep/hub-moe-ep2`

- **Artifact:** `/tmp/agilerl-smoke/ep/hub-moe-ep2/`
- **Command:**
  ```bash
  cd /home/mike/wt-expert-parallel
  export PYTHONPATH=/home/mike/wt-expert-parallel/agilerl-arena:/home/mike/wt-expert-parallel
  export HF_HOME=/tmp/hf-home TMPDIR=/tmp UV_CACHE_DIR=/tmp/uv-cache
  export TRANSFORMERS_CACHE=/tmp/hf-home/transformers
  while ! mkdir /tmp/agilerl-smoke/gpu.lock.d 2>/dev/null; do sleep 20; done
  trap 'rmdir /tmp/agilerl-smoke/gpu.lock.d 2>/dev/null' EXIT
  /tmp/agilerl-venv/bin/torchrun --standalone --nproc_per_node=2 \
    scripts/fsdp_gates/run_probe.py \
    --job-id hub-moe-ep2 \
    --artifact-dir /tmp/agilerl-smoke/ep/hub-moe-ep2 \
    --config demos/llm/debugging/configs/grpo_constant_target_hub_moe.yaml \
    --model-name PrimeIntellect/qwen3-moe-tiny \
    --max-steps 1280 --warmup-steps 0 --fsdp --log-stdout \
    --init-hp EP=2
  ```
- **Exit / wall:** 0 / **714.3s**
- **Peak CUDA alloc:** ≈1.92 GB (reserved ≈2.37 GB)
- **Metrics** (`train_metrics.jsonl` / stdout tables):

  | global step | mean_score | mean_loss | mean_kl |
  |------------:|-----------:|----------:|--------:|
  | 256 | **-0.5078** | -0.043 | 1.41 |
  | 512 | **+0.0859** | -0.047 | 2.34 |
  | 768 | -0.0234 | -0.083 | 1.85 |
  | 1024 | **+0.7812** | 0.107 | 4.04 |
  | 1280 | **+1.0000** | 0.077 | 1.93 |

- **summary.json:** `final_eval=1.0`, `final_loss≈0.077`, `status=ok`
- **Verdict:** FSDP2 + EP=2 product learning path green on a **real hub** packed-expert MoE.

---

## Config / code changes for hub MoE

| Piece | Role |
|-------|------|
| `grpo_constant_target_hub_moe.yaml` | `EP`-ready GRPO: `BATCH=16`, `GROUP=8`, `BETA=0.04`, `LR=1e-3`, SDPA, attention LoRA (`q/k/v/o`, r=16) |
| `run_probe.py` `_materialize_digit_prior_model` | Pre-FSDP `/tmp` snapshot with digit lm_head prior from `DEBUG.digit_logit_boost` |
| LoRA targets | Attention only — full trainable `lm_head` on vocab 151936 caused KL blow-up / NaN loss |

---

## Iterations (failed / discarded)

1. `digit_logit_boost=8` + `target_logit_boost=0.5` + `modules_to_save: [lm_head]` → exit 0 but score trivially 1.0 from step 512 (prior too strong; loss/KL=0).
2. Same with `target_logit_boost=0` + trainable `lm_head` → NaN loss by ~step 256 (`mean_kl≈179`).

---

## Residual risks

- Digit prior is required for short ConstantTarget smokes on this hub checkpoint.
- `world_size=2`, `ep=2` ⇒ `dp_mod_ep` size 1. Larger worlds still need coverage.
- vLLM / colocated rollout under EP not exercised.
- Shared 2×L4 host: take `/tmp/agilerl-smoke/gpu.lock.d` before torchrun.

---

## Raw artifacts

```
/tmp/agilerl-smoke/ep/hub-moe-ep2/   (ok, score -0.51 → 1.0)
/tmp/hf-home/models--PrimeIntellect--qwen3-moe-tiny/   (~1.3 GB)
/tmp/hf-home/ctprior-f4f43a3728/   (digit_logit_boost=10 snapshot used by green run)
```

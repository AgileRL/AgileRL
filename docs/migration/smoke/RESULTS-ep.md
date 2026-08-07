# Expert Parallel — GRPO ConstantTarget smoke results

**Date:** 2026-08-07  
**Host:** mikeyp-l4x2 (2× NVIDIA L4 24GB)  
**Worktree:** `/home/mike/wt-expert-parallel` (`migration/expert-parallel`)  
**Python:** `/tmp/agilerl-venv` — torch `2.11.0+cu130`  
**Harness:** `scripts/fsdp_gates/run_probe.py` + `demos/llm/debugging/configs/grpo_constant_target_moe.yaml`  
**GPU lock:** `/tmp/agilerl-smoke/GPU_LOCK.md` (single torchrun; CP priority then EP)

Metric note: probe logs `train/mean_score` (ConstantTarget reward; 1.0 = hit).  
Each GRPO outer iter under this config advances **512** sample-steps
(`BATCH_SIZE=32 × GROUP_SIZE=8 × world_size=2`).

---

## Summary

| Job | Config | Exit | Wall | Learning signal | Outcome |
|-----|--------|------|------|-----------------|---------|
| `Q05B-FSDP-ep1` | FSDP, `EP=1`, Qwen2.5-0.5B | **0** | ~158s | score → **1.0** | Learning OK (dense, no expert shard) |
| `Q05B-EP2-dense` | FSDP, `EP=2`, Qwen2.5-0.5B | **1** | ~16s | none | **Expected fail** — not packed-expert MoE |
| `Q-EP2-moe-green` | FSDP, `EP=2`, tiny Qwen3-MoE fixture | **0** | **56s** | score **+0.066 → 1.0** by step 1024 | **Learning OK** |

---

## MoE choice

**Local randomly-init packed-expert fixture** (no multi-GB download):

- Path: `demos/llm/debugging/fixtures/tiny_qwen3_moe`
- Builder: `demos/llm/debugging/fixtures/build_tiny_qwen3_moe.py`
- Architecture: `Qwen3MoeForCausalLM` — `num_experts=4`, `num_experts_per_tok=2`,
  2 layers, `hidden_size=64`, vocab **7** (TinyDigit ids: digits 0–4, pad=5, eos=6)
- ~350KB `model.safetensors`; recognized by `moe_lora` routed-experts detectors
  (`gate_up_proj` / `down_proj`)
- Mild lm_head prior (~0.5 ConstantTarget hit-rate at T=1.0) so short GRPO gets
  positive rewards without densifying a hub MoE

---

## Green job: `ep/Q-EP2-moe-green`

- **Artifact:** `/tmp/agilerl-smoke/ep/Q-EP2-moe-green/`
- **Command:**
  ```bash
  cd /home/mike/wt-expert-parallel
  export PYTHONPATH=/home/mike/wt-expert-parallel/agilerl-arena:/home/mike/wt-expert-parallel
  export TMPDIR=/tmp UV_CACHE_DIR=/tmp/uv-cache HF_HOME=/tmp/hf-home
  # acquire /tmp/agilerl-smoke/gpu.lock.d per GPU_LOCK.md first
  /tmp/agilerl-venv/bin/torchrun --standalone --nproc_per_node=2 \
    scripts/fsdp_gates/run_probe.py \
    --job-id Q-EP2-moe-green \
    --artifact-dir /tmp/agilerl-smoke/ep/Q-EP2-moe-green \
    --config demos/llm/debugging/configs/grpo_constant_target_moe.yaml \
    --model-name demos/llm/debugging/fixtures/tiny_qwen3_moe \
    --max-steps 2048 --warmup-steps 0 --fsdp --log-stdout \
    --init-hp EP=2
  ```
- **Exit / wall:** 0 / **55.6s**
- **Metrics** (`train_metrics.jsonl`):
  - step 512: `mean_score=+0.0664`, `mean_loss=-0.176`, `mean_kl=1.01`
  - step 1024: `mean_score=+1.0000`, `mean_loss=0.0`, `mean_kl=0.28`
  - step 1536–2048: `mean_score=1.0` held
- **summary.json:** `final_eval=1.0`, `final_loss=0.0`, peak alloc ≈17MB, `status=ok`
- **Verdict:** FSDP2 + EP=2 product learning path green on packed-expert MoE.

---

## Phase-3 plumbing that landed

| Piece | Change |
|-------|--------|
| `apply_fsdp2(..., ep_mesh=)` | Experts `fully_shard(mesh=dp_mod_ep)` then blocks/embed/root on `hsdp`; `ep=1` keeps flat PG |
| `materialize_fsdp2_from_cpu_state` | Passes `ep_mesh` into `apply_fsdp2` after `apply_expert_parallel` |
| `install_ep_routed_forward` | EP-aware A2A forward on base packed experts (attention-only LoRA path) |
| Probe / YAML | MoE config, `modules_to_save: [lm_head]`, `USE_SEPARATE_REFERENCE_ADAPTER=false`, SDPA |

---

## Residual risks

- Green smoke uses a **tiny random-init** MoE + trainable `lm_head`, not a hub-scale MoE.
- `world_size=2`, `ep=2` ⇒ `dp_mod_ep` size **1** (experts EP-sharded only; no extra FSDP shard on experts). Larger worlds (`dp_mod_ep>1`) still need coverage.
- vLLM export / colocated rollout under EP not exercised (`USE_VLLM=false`).
- Shared 2×L4 host: always take `/tmp/agilerl-smoke/gpu.lock.d` before torchrun.

---

## Raw artifacts

```
/tmp/agilerl-smoke/ep/
  RESULTS-ep.md              ← this file
  Q-EP2-moe-green/           (ok, score → 1.0)
  Q05B-FSDP-ep1/             (ok, dense EP=1)
  Q05B-EP2-dense/            (expected fail)
  Q-EP2-moe-learn*/          (earlier tuning runs)
```

# FSDP2 Optimization Plan

## Context

AgileRL migrated to FSDP2 (`fully_shard`) and has a working `CPUOffloadOptimizer`. Benchmarks show optim offload is free at 0.5B, full param offload saves 36% VRAM but costs 1.6x step time. This plan adds the missing high-impact optimizations identified by comparing with Prime-RL.

## Current state (already done)

- `CPUOffloadOptimizer` — optim states on pinned CPU, move to GPU for `step()` (`distributed.py`)
- `CPUOffloadPolicy` — full param+grad+state offload (`distributed.py`, `--cpu-offload`)
- FSDP2 per-layer sharding via `fully_shard` (`apply_fsdp2`)
- `reshard_after_forward` config knob on `FSDPConfig`
- `MixedPrecisionPolicy` with separate `param_dtype` / `reduce_dtype`
- Fused chunked logprobs (`fused_logprobs.py`)
- `save_on_cpu` activation offloading context (`base.py:4426-4444`)

## What to implement

### 1. Flip `optim_cpu_offload` default to True

**What:** Change `FSDPConfig.optim_cpu_offload` default from `False` → `True`.

**Where:** `agilerl/utils/distributed.py` — `FSDPConfig` dataclass

**Why:** Benchmark shows it's free at 0.5B (1664 vs 1666 MiB, 1912 vs 1889ms) and speeds up colocated vLLM (77s vs 86s). Prime-RL defaults this to True.

**Risk:** Users who relied on "FSDP=True means no offload" get a behavior change. Can override with `{"optim_cpu_offload": False}`.

---

### 2. Activation checkpointing + fix wrap order + wire activation offload

**What:** Three changes in one PR:

#### 2a. Move AC before FSDP in `wrap_models`

**Current order** (`base.py:3230-3256`):
```
apply_fsdp2 → rebuild optimizer → CPUOffloadOptimizer → gradient_checkpointing_enable
```

**Target order** (matching Prime-RL `model.py:1329-1335`):
```
apply_activation_checkpointing → apply_fsdp2 → rebuild optimizer → CPUOffloadOptimizer
```

AC must wrap blocks *before* `fully_shard` so the checkpoint boundary sits inside the FSDP unit. The current post-FSDP `gradient_checkpointing_enable` should be replaced with pre-FSDP `checkpoint_wrapper`.

**Where:**
- `agilerl/utils/distributed.py` — add `ActivationCheckpointConfig` dataclass + `apply_activation_checkpointing()` function
- `agilerl/algorithms/core/base.py` — reorder `wrap_models()`, remove post-FSDP `gradient_checkpointing_enable` when AC config is set

**How:**
```python
@dataclass
class ActivationCheckpointConfig:
    mode: Literal["full", "selective"] = "full"
    freq: int = 1

def apply_activation_checkpointing(model: nn.Module, cfg: ActivationCheckpointConfig) -> None:
    blocks = _transformer_blocks(model)
    for i, block in enumerate(blocks):
        if i % cfg.freq != 0:
            continue
        wrapped = checkpoint_wrapper(block, preserve_rng_state=False)
        # re-register on parent ModuleList
```

Selective AC requires custom model layers (Prime-RL pattern). For HF models, start with full block AC only.

#### 2b. Wire existing `save_on_cpu` as the activation offload context

**Current state:** `base.py:4426-4444` already has `_activation_offload_ctx` using `torch.autograd.graph.save_on_cpu(pin_memory=True)`. This works but isn't wired to a config.

**What to do:**
- Add `ActivationOffloadingConfig` dataclass (`pin_memory: bool = True`)
- Add `ac_offloading: ActivationOffloadingConfig | None = None` to `FSDPConfig`
- Wire the existing `_activation_offload_ctx` to activate when `ac_offloading` is set
- Auto-enable AC when `ac_offloading` is set (Prime-RL validator pattern)
- Wrap **forward only** — `loss.backward()` runs outside the context (Prime-RL pattern)

**Where:**
- `agilerl/utils/distributed.py` — `ActivationOffloadingConfig` dataclass
- `agilerl/algorithms/core/base.py` — wire context around forward calls in `_fused_model_pass` / `_packed_forward`

**Why `save_on_cpu` is enough for now:** Prime-RL's `OffloadActivates` adds size filtering and param/buffer exclusion, but hardcodes `use_streams=False` (streams cause leaks). The `save_on_cpu` approach is simpler and achieves the same core benefit. Port the sophisticated version later if profiling shows we're offloading tensors we shouldn't.

---

## Config end state

```python
@dataclass
class FSDPConfig:
    reshard_after_forward: bool = True       # exists
    cpu_offload: bool = False                # exists (escape hatch)
    optim_cpu_offload: bool = True           # CHANGE: False → True
    param_dtype: torch.dtype | None = None  # exists
    reduce_dtype: torch.dtype | None = None # exists
    ac: ActivationCheckpointConfig | None = None        # NEW
    ac_offloading: ActivationOffloadingConfig | None = None  # NEW
```

## Default stack

```
optim_cpu_offload (on by default)
  + activation checkpointing (full mode, opt-in)
  + activation offloading (save_on_cpu, opt-in)
  = peak VRAM ≈ max(activations, opt_states)

param offload (cpu_offload) = nuclear option, off by default
```

## Tests

- Smoke gate: `--fsdp --ac full --ac-offload` 40 steps, peak VRAM < baseline
- Smoke gate: `--fsdp` alone still works (optim offload on by default)
- Unit: `FSDPConfig().optim_cpu_offload is True`
- Unit: AC applied before FSDP (check wrap order via mock)

## Out of scope

- FSDP prefetch
- torch.compile per-layer
- Meta device + DCP loading
- Deterministic GC
- Selective AC for custom models
- Broader LoRA targets (constructor already defaults to all-linear)
- Porting Prime-RL's `inject_prime_lm_head`
- Expert parallelism / DeepEP

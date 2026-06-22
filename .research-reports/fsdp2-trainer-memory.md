# FSDP2 (`fully_shard`) trainer-side memory model

Research slice for AgileRL auto-config: predicting TRAINER memory under the torch-native FSDP2 backend.

Sources read directly:

- PyTorch trunk clone at `/Users/michaeldoherty/git/Misc/pytorch` (commit `8fa0e605718`, 2026-05-26) — `torch/distributed/fsdp/_fully_shard/*`, `test/distributed/_composable/fsdp/*`, `torch/distributed/_tools/*`, `docs/source/distributed.fsdp.fully_shard.md`.
- torchtitan main (GitHub API): `torchtitan/distributed/fsdp.py`, `torchtitan/config` (job config), historic `scripts/estimate/estimation.py` (v0.2.2).
- torchtune main (GitHub API): `recipes/lora_finetune_distributed.py`, `torchtune/training/_distributed.py`; torchao `torchao/quantization/quantize_/workflows/nf4/nf4_tensor.py`.
- prime-rl local clone: `src/prime_rl/trainer/model.py` (a production RL trainer using FSDP2 — useful corroborating wrapping policy).
- AgileRL worktree (current state): trainer goes through HF Accelerate (`agilerl/utils/llm_utils.py:313` still has the DeepSpeed contract; the DDP/FSDP2 migration replaces it). Accelerate 1.7.0 in the repo venv exposes `FullyShardedDataParallelPlugin(fsdp_version=2, reshard_after_forward=..., mixed_precision_policy=..., cpu_offload=..., activation_checkpointing=...)` which maps onto everything below.

All `file:line` references into PyTorch are relative to `/Users/michaeldoherty/git/Misc/pytorch`.

---

## 1. FSDP2 public knobs

Entry point: `fully_shard(module_or_list, *, mesh, reshard_after_forward, shard_placement_fn, mp_policy, offload_policy, ignored_params, dp_mesh_dims)` — `torch/distributed/fsdp/_fully_shard/_fully_shard.py:98-108`.

### 1.1 Communication grouping (the implicit, most important knob)

Each `fully_shard` call creates ONE communication group ("param group") containing all parameters of that module not already claimed by an earlier (deeper) call. One all-gather per group per forward, one reduce-scatter per group per backward (`_fully_shard.py:134-143`; `docs/source/distributed.fsdp.fully_shard.md:58-133`). There is **no `bucket_cap_mb`** — the unsharded-peak granularity *is* the wrapping granularity. Calling `fully_shard` only on the root degenerates to "all-gather everything → forward → backward → reduce-scatter everything": full unsharded model resident, zero overlap. The canonical policy (torchtitan `torchtitan/distributed/fsdp.py::apply_fsdp`, prime-rl `src/prime_rl/trainer/model.py:607-680`, torchtune `torchtune/training/_distributed.py::shard_model`) is:

- `fully_shard(block)` per transformer block (bottom-up),
- `fully_shard(embed_tokens)` as its own group,
- `fully_shard([final_norm, lm_head], reshard_after_forward=False)` grouped (and skipped/merged with embeddings when weights are tied — tied params must live in one group to avoid double all-gather),
- `fully_shard(model)` last; leftovers form the root group.

`fully_shard([a, b, ...])` (list form) groups several modules into one collective — used for norm+lm_head and for chunked-loss patterns (`_fully_shard.py:145-157`).

### 1.2 `reshard_after_forward: bool | int | None` (`_fully_shard.py:180-203`)

- `True`: free the group's unsharded params after its forward; re-all-gather in backward. Lowest memory, 1.5× param comm volume (AG fwd + AG bwd + RS).
- `False`: keep unsharded params resident from forward through backward. No backward all-gather. Memory cost: the whole model unsharded resident at end of forward (see §2.3).
- `None` (default): `True` for non-root groups, `False` for the **root** group — set during lazy init (`_fsdp_state.py:203-205`, `_fully_shard.py:189-190`). Root is needed first in backward anyway, so resharding it only adds latency.
- `int k`: reshard post-forward to a *smaller world size* `k` instead of fully (`_fsdp_init.py:152-182`). `k` must divide the shard-mesh size; `k=1` ⇒ `False`, `k=world` ⇒ `True`. Between forward and backward the group holds `numel/k` bytes per rank (a cloned slice of the all-gather output, `_fsdp_param.py:707-744`), and the backward all-gather runs over the `k`-size group (intra-node `k = torch.cuda.device_count()` is the suggested choice — turns the backward AG into NVLink-only). Not yet supported with `dp_mesh_dims` SPMD meshes (`_fully_shard.py:249-258`).
- Runtime mutators: `set_reshard_after_forward(bool, recurse=True)` (`_fully_shard.py:453`), `set_reshard_after_backward(bool)` (`_fully_shard.py:490`, for grad accumulation: keep params unsharded across microbatches), `module.reshard()` / `module.unshard(async_op)` manual control (`_fully_shard.py:335-373`).

### 1.3 `MixedPrecisionPolicy` (`_fsdp_api.py:13-54`)

`MixedPrecisionPolicy(param_dtype=None, reduce_dtype=None, output_dtype=None, cast_forward_inputs=True)`.

- `param_dtype`: dtype of the **unsharded** (all-gathered) params, hence the compute dtype and AG payload dtype. The **sharded** params stay in original dtype — this is the implicit "master weights" copy: *"FSDP does not require any extra memory to keep a high-precision copy of the parameters for the optimizer step"* (`_fsdp_api.py:21-24`).
- `reduce_dtype`: dtype of the gradient reduce-scatter/all-reduce payload. If `None`, falls back to gradient compute dtype (= `param_dtype`). After reduction the sharded grad is cast back to the original dtype (`_fsdp_collectives.py:696`), so sharded `.grad` is in **orig dtype** (fp32 if the model was loaded fp32).
- Clamping: `param_dtype` is ignored for non-floating params and when equal to orig dtype; `reduce_dtype` is ignored when equal to `param_dtype` (`_fsdp_param.py:584-598`).
- All trainable params in one group must share orig dtype and reduce dtype (`_fsdp_param_group.py:262-283`); **frozen params may have different dtypes** — mixed-dtype groups all-gather as a flat `uint8` buffer (`_fsdp_collectives.py:343-346`, `789-801`). This is what lets a quantized frozen base and bf16 trainable adapters share one group.
- Reference defaults: torchtitan `mixed_precision_param="bfloat16"`, `mixed_precision_reduce="float32"`, `cast_forward_inputs=False`; prime-rl same (`model.py:609`). torchtune LoRA recipes instead load the whole model in bf16 with **no** mp_policy (pure-bf16: halves sharded param/grad/optimizer bytes at some numerics risk; fp32 reduce unavailable then).

### 1.4 `OffloadPolicy` / `CPUOffloadPolicy(pin_memory=True)` (`_fsdp_api.py:174-199`)

CPU offload moves **sharded params, sharded grads, and optimizer states** to CPU (optimizer step runs on CPU). GPU holds only the transient unsharded group(s): sharded params are H2D-copied before all-gather, reduced grads D2H-copied after reduce-scatter (`_fsdp_api.py:183-190`, D2H copy at `_fsdp_collectives.py:711-743`). `pin_memory` enables overlap but pins host RAM. Throughput cost is large (CPU Adam + PCIe); for auto-config treat it as the last resort tier. Note prime-rl deliberately implements its own `CPUOffloadOptimizer` instead ("keeps weights on GPU", `src/prime_rl/trainer/optim.py:19-22`) — offloading only optimizer state is often the better trade.

### 1.5 `shard_placement_fn` (`_fully_shard.py:204-220`)

Per-parameter override of shard dim/mesh. Default `Shard(0)` (dim-0 chunking). Returning `Shard(i)` for i>0 requires even divisibility; used e.g. for MoE experts (`Shard(1)` when FSDP degree > num_experts, torchtitan `fsdp.py`) and for tying parameter layouts to vLLM TP layouts if you ever want zero-copy alignment. Nonzero shard dims add a transient chunk-cat copy in AG copy-out (`_fsdp_collectives.py:465-471, 493-518`) and a chunk-cat before reduce-scatter (`_fsdp_collectives.py:579-590`) — small extra transient memory.

### 1.6 Prefetching: `set_modules_to_forward_prefetch` / `set_modules_to_backward_prefetch` (`_fully_shard.py:513-551`)

Default behavior already overlaps one group ahead: implicit forward prefetch (CPU runahead issues next AG on the AG stream) and explicit reverse-post-forward-order backward prefetch of 1 module. These APIs override the target list; *"Passing a list with at least length two is required for more aggressive overlap and will use more reserved memory"* — each extra prefetched module adds one more unsharded group to the peak (§2.3). Auto-config: leave at default (1-ahead) unless comm-bound and memory-rich.

### 1.7 `ignored_params: set[nn.Parameter]` (`_fully_shard.py:227-229`)

Ignored params are **not sharded, not moved, not gradient-reduced** — they stay as plain replicated tensors. For LoRA this enables a hybrid topology: `fully_shard(model, ignored_params=base_params)` shards only adapters while the frozen base stays replicated with **zero** FSDP communication (see §3.3).

### 1.8 Other runtime knobs relevant to memory/accounting

- `set_requires_gradient_sync(bool)` (`_fully_shard.py:413`): grad accumulation **without** reduction. Memory consequence: gradients are then accumulated **unsharded** (full model size per rank) in `reduce_dtype` (`_fsdp_api.py:38-39`, `_fsdp_param.py:800-819`). This is expensive; the cheap accumulation is to reduce every microbatch (sharded fp32 grads accumulate at `N/W`) — FSDP2's per-microbatch RS costs comm, not memory.
- `set_is_last_backward(bool)` (`_fully_shard.py:403`), `set_unshard_in_backward(False)` for params not needed in backward (embedding) (`_fully_shard.py:707`), `set_reduce_scatter_unused_params` (conditional-routing models, `_fully_shard.py:680`), `set_post_optim_event` (`_fully_shard.py:622`), custom comms / symmetric memory / PG allocators (`_fully_shard.py:553-771`).
- HSDP: pass a 2-D mesh `(replicate, shard)` or `dp_mesh_dims=DataParallelMeshDims(shard=..., replicate=...)` (`_fsdp_api.py:131-171`). Sharded sizes below then use `W = shard-dim size` only, plus an extra all-reduce of the reduced grads across the replicate dim.

### 1.9 Composition with activation checkpointing

Apply AC to the block **first**, then `fully_shard` the same (or an enclosing) block — the test pattern is `checkpoint(mlp); fully_shard(mlp)` (`test/distributed/_composable/fsdp/test_fully_shard_frozen.py:77-81`); torchtune does `set_activation_checkpointing` then `shard_model` (`recipes/lora_finetune_distributed.py:529-552`). Semantics under recompute: the recomputed forward re-fires FSDP's pre-forward (unshard — a no-op if backward prefetch already unsharded the group), and the post-forward hook detects backward via `is_bw()` and skips resharding/order-recording (`_fsdp_param_group.py:545-554`). Net memory effect: with `reshard_after_forward=True` + per-block AC, each block is unsharded exactly while being (re)computed; no interaction penalty. AC composes per-module, orthogonal to FSDP grouping.

### 1.10 Composition with torch.compile

Dynamo never traces FSDP hooks — they are skipped (`test/distributed/_composable/fsdp/test_fully_shard_compile.py::test_disable_compiling_hooks`); managed modules are flagged `_is_fsdp_managed_module` / `_fsdp_use_orig_params = True` for Dynamo (`_fully_shard.py:288-290`). Practical patterns: (a) torchtitan-style **compile each transformer block individually, then `fully_shard` it** — graph breaks at FSDP boundaries are then free; (b) whole-model `model.compile()` works with graph breaks at hooks; (c) full-graph backward needs compiled autograd (still experimental). Compile does not change the FSDP memory model below; it changes activation memory only via the AC/partitioner path (`memory_budget`, §5.2).

---

## 2. The memory model (formulas)

Notation: `P` = total param count; `P_t` = trainable param count; `W` = FSDP shard world size; `G_max` = parameter count of the **largest fully_shard group** (typically one transformer block; for tied-embed models often the embed+lm_head group — check both!); `b_orig`, `b_param`, `b_reduce` = bytes/element of original dtype, `param_dtype`, `reduce_dtype`; `n_opt` = optimizer state multiplier in elements (Adam/AdamW = 2; SGD-momentum = 1; plus a transient for `foreach` ops); `mb` = micro-batch size, `s` = sequence length, `h` = hidden size, `L` = layer count, `V` = vocab size.

Per-parameter dim-0 sharding pads dim 0 up to a multiple of `W` (`_fsdp_param.py:298-316`), so per-rank shard of one param = `ceil(dim0/W) * prod(other dims)`. Padding overhead is bounded by `(W-1)/dim0` per tensor — negligible for real LLM weights but can matter for many tiny tensors (per-layer norms with `dim0=h ≫ W` are still fine). Treat sharded numel ≈ `P/W` with a ~0.1–1% safety factor.

### 2.1 Persistent (steady-state, between steps)

```
M_params_sharded = P/W · b_orig                       # DTensor shards, original dtype
M_grads_sharded  = P_t/W · b_orig                     # after RS + cast back to orig dtype
M_optim_sharded  = n_opt · P_t/W · 4                  # Adam moments, fp32 if orig fp32
                   (n_opt · P_t/W · b_orig if pure-bf16 loading)
```

Evidence: sharded grads are viewed out of the RS output and cast to orig dtype (`_fsdp_collectives.py:696-754`); optimizer states follow `sharded_param` dtype. The memory test asserts post-step memory = `3 × model_sharded_numel × 4` (1× params + 2× Adam states) `+ 1× sharded grads` until `zero_grad` (`test_fully_shard_memory.py:213-239`).

So for full fine-tune, fp32 master + bf16 compute (`param_dtype=bf16`): persistent ≈ `P/W·(4+4+8) = 16·P/W` bytes — identical to ZeRO-3's `16N/W` rule. Pure-bf16 load: `P/W·(2+2+4) = 8·P/W`.

### 2.2 Transient comm buffers and unsharded params

Mechanics (`_fsdp_collectives.py:325-378, 431-519`): per group, forward unshard allocates (i) one flat AG output buffer `G·b_param·1` (input slice is a view of it at rank offset — actually input copy-in is a separate `G/W·b_param` staging buffer, line 419-425, minor), then copy-out allocates (ii) per-param unsharded storages totalling `G·b_param`, then frees (i). In forward with implicit prefetch, the current flat buffer is intentionally kept alive until the *next* group's copy-out (`_fsdp_param_group.py:404-420`), and the next group's AG buffer is already allocated → **3 unsharded group-copies coexist in forward**. In backward the AG result is freed right after copy-out → 2 param copies, but gradients add 2 more (autograd-produced unsharded grads + flat RS input buffer of `G·b_reduce`); the RS output (`G/W·b_reduce`) is accounted within sharded grads since grads view into it (`test_fully_shard_memory.py:182-190` NOTE).

From the assertions in `test_fully_shard_memory.py` (world=2, fp32, no MP — generalized to dtypes):

```
# reshard_after_forward=True (per-block wrapping, root kept unsharded)
M_peak_fwd  = M_persistent_params + nonblock_unsharded·b_param + 3·G_max·b_param + A_live
M_peak_bwd  = M_persistent       + nonblock_unsharded·b_param
              + 2·G_max·b_param              # AG buffer (prefetched next) + copy-out (current)
              + G_max·b_param                # unsharded grads of current block (computed in param_dtype)
              + G_max·b_reduce               # flat reduce-scatter input
              + A_live
# test form (fp32, b_param=b_reduce=4): 4·G_max + nonblock, lines 182-209

# reshard_after_forward=False
M_peak_fwd  = P/W·b_orig + P·b_param + G_max·b_param + A_live          # whole model unsharded + 1 copy-out
M_peak_bwd  = P/W·b_orig + P·b_param + 1.5·G_max·(b_param/b_reduce mix) + A_live
# test form: model_sharded + model_unsharded + 1.5·max_unsharded (RS in+out), lines 200-208

# reshard_after_forward=k (int): between fwd and bwd each group holds G/k·b_param instead of 0 (True) or G·b_param (False)
M_between   = Σ_groups G/k·b_param  = P/k·b_param   (plus root group unsharded)
```

Extra prefetch: each additional module in `set_modules_to_forward/backward_prefetch` adds `+G·b_param` to the respective peak.

Gradient accumulation: with per-microbatch sync (default) no extra term. With `set_requires_gradient_sync(False)`: `+ P_t·b_reduce` **unsharded** accumulated grads persist across microbatches (`_fsdp_param.py:800-819`). With `set_reshard_after_backward(False)`: params stay unsharded between microbatches: `+ P·b_param`.

HSDP all-reduce holds one extra `G/W_shard·b_reduce` buffer alive across layers (`_fsdp_param_group.py:249-259`).

CPU offload: GPU persistent terms drop to ≈0; GPU peak ≈ `3·G_max·b_param + G_max·b_reduce + A_live`; CPU RAM gains `P/W·b_orig + P_t/W·(b_orig + 4·n_opt)` (pinned). Init-time GPU transient ≈ `1.5·G_max·4` (`test_fully_shard_memory.py:139-141`).

### 2.3 Activation memory `A_live`

FSDP does not change activation math; activations are full-size per rank (DP shards the batch, not the activations). For a Llama-style block (SDPA/FlashAttention — no materialized attention matrix), saved-for-backward bytes per layer per token are approximately:

```
A_layer ≈ mb · s · h · b_act · c_arch
  c_arch ≈ 2 (attn in/out, qkv) + 4·r_ff (SwiGLU up/gate/act) + norms ≈ 12–20 for r_ff≈3.5, GQA
           (classic no-flash fp16 estimate: s·b·h·(34 + 5·a·s/h) bytes — use only without SDPA)

No AC:        A_live ≈ L · A_layer                                   + cross-entropy/logits term
Full AC:      A_live ≈ L · (mb·s·h·b_act)            # one boundary input per layer
              + A_layer                              # recompute live set of one block
Selective every-k layers (torchtitan selective_ac_option=int k):
              A_live ≈ (L/k)·A_layer + L·(mb·s·h·b_act) ≈ full-AC + 1/k of no-AC
Selective per-op (save matmul/SDPA outputs only): ≈ 0.3–0.5 × no-AC (empirical; torchtitan _save_list policy)
```

The **logits term dominates for RL fine-tuning**: `mb·s·V·b_act` for logits + up to `mb·s·V·4` for fp32 softmax/log-softmax, plus a same-size grad buffer. For Qwen-style `V≈150k`, one sequence of 4k tokens is ~1.2 GB in bf16 before fp32 upcast. Chunked/fused losses (Liger-style, which AgileRL already uses — `fused_loss` with chunked logits/log-softmax, repo commit `0a9ad78b`) cap this at `chunk_size·V` terms; the auto-config should model logits memory only when fused loss is OFF, else `n_chunks`-divided. Because RL rollouts are variable-length, use `s = max_context` (prefill-heavy) or `max_prompt+max_completion` for the worst-case micro-batch; decode-heavy multi-turn workloads have larger `s·mb` activation pressure relative to model size, pushing toward AC-on and smaller `mb`.

### 2.4 Putting it together (auto-config predictor, reshard=True, per-block wrap)

```
M_trainer_peak ≈ P/W·b_orig                          # sharded params
               + P_t/W·b_orig                        # sharded grads
               + n_opt·P_t/W·4                       # optimizer
               + E·b_param                           # root/non-block group unsharded (embeds+lm_head)
               + (3 + n_extra_prefetch)·G_max·b_param
               + G_max·b_reduce
               + A_live(mb, s, h, L, V, AC, fused_loss)
               + C_overhead                          # CUDA ctx + NCCL + allocator frag: budget 1.5–2.5 GB/GPU
```

Solve for `mb` (largest s.t. `M_trainer_peak ≤ VRAM_budget_trainer`); set grad-accum = `ceil(global_batch / (mb·W))`. In COLOCATED mode, `VRAM_budget_trainer = VRAM − M_vllm_resident_during_training` (with sleep/standby, vLLM weights stay resident if shared zero-copy; KV freed). FSDP2's deterministic allocator behavior (no `recordStream`; avoids FSDP1's non-determinism, `docs/source/distributed.fsdp.fully_shard.md:147-150`) makes this prediction usefully tight — torchtitan reported FSDP2 ~7% lower peak than FSDP1.

Validation tool: **`FSDPMemTracker`** (`torch/distributed/_tools/fsdp2_mem_tracker.py:122`) — a TorchDispatchMode that categorizes FSDP2 memory per module/state, works under `FakeTensorMode` (no GPU needed!). torchtitan's removed-from-main estimation script (`scripts/estimate/estimation.py` @ v0.2.2) ran exactly this: fake-mode two-iteration dry run, `display_modulewise_snapshots`, compare tracker peak vs `active_bytes.all.peak`. **Recommendation: AgileRL's auto-config should use the closed-form above for the search, then verify the chosen config with one fake-mode FSDPMemTracker dry run.**

---

## 3. LoRA + FSDP2

### 3.1 What is sharded, what optimizer state exists

Everything passed to `fully_shard` is sharded — frozen base and trainable adapters alike become dim-0 DTensor shards in the same group. Frozen/trainable can mix freely in one group (FSDP2's per-param sharding "relaxes constraints around frozen parameters" vs FSDP1, docs:145); only **trainable** params must share orig/reduce dtype (`_fsdp_param_group.py:265-283`).

- **All-gather: full group including frozen base, every forward (and every backward if reshard=True).** Frozen weights are needed for compute, so the unsharded-peak terms in §2.2 are unchanged by freezing.
- **Reduce-scatter: trainable-only.** Post-backward collects only params with grads; RS payload numel = adapter numel (verified: `test_fully_shard_frozen.py:96-103` asserts RS numel = bias-only; collection logic `_fsdp_param_group.py:603-624`).
- **Persistent memory:** `M_grads = P_lora/W·b_orig`, `M_optim = n_opt·P_lora/W·4` — both negligible (`P_lora` ~0.1–1% of `P`). Sharded base params: `P/W·b_orig`.
- **Does reshard matter?** Yes — exactly as much as for full FT, because the dominant transient is all-gathering the *frozen base*. `reshard_after_forward=True` keeps only ~3 blocks unsharded; `False` keeps the whole bf16 model unsharded through backward (then FSDP2 sharding bought you almost nothing memory-wise — only optimizer/grad sharding, which LoRA already made tiny). For LoRA: reshard=True if you need the memory; if the unsharded model fits, you shouldn't be using FSDP at all (→ §4).

torchtune's distributed LoRA recipe: shards each `layers.{i}` block + root via `shard_model` (`torchtune/training/_distributed.py:661-706`), `fsdp_reshard_after_forward=True` default (`recipes/lora_finetune_distributed.py:326`), model loaded bf16, no mp_policy; NOTE comment: "register (pre-)forward hooks with fully_shard instead of wrapping nn.Module" (`recipes/lora_finetune_distributed.py:489`).

### 3.2 QLoRA (4-bit base) under FSDP2 — the torchtune/torchao mechanism

FSDP2 supports tensor-subclass extension hooks `fsdp_pre_all_gather` / `fsdp_post_all_gather` (`_fsdp_param.py:601-655`, must define both). torchao's `NF4Tensor` implements them (`torchao/quantization/quantize_/workflows/nf4/nf4_tensor.py:986-1041`):

- `fsdp_pre_all_gather` returns the **quantized payloads** `(quantized_scalers, quantization_factor, quantized_data)` — so the AG moves ~0.5 byte/param, not bf16.
- `fsdp_post_all_gather` reconstructs an **NF4Tensor as the unsharded param** (stays quantized; the mixed-dtype group goes through the uint8 flat-buffer path, `_fsdp_collectives.py:343-346`). Dequant to bf16 happens inside `linear_nf4` per matmul, not held.

Memory consequence vs bf16-base LoRA: sharded base `≈0.55·P/W` bytes; unsharded transient per block `≈3·0.55·G_max` instead of `3·2·G_max`; adapter terms unchanged. QLoRA+FSDP2 trainer memory is therefore ≈ 27% of bf16-base LoRA on both persistent and transient param terms — activations unchanged. Caveats: 2-D weights only (`nf4_tensor.py:1028-1031`); early FSDP2+NF4 dispatch gaps existed (torchtune issue #1072) — fixed by these hooks; dequant kernel cost per re-gathered block is the throughput tax. **bitsandbytes `Params4bit` (AgileRL's current bnb path) has no `fsdp_pre_all_gather` implementation** — bnb-quantized bases cannot be sharded by FSDP2 directly; the workable topologies are DDP/replicated base (bnb) or NF4 (torchao) if sharding the base is required. torchtitan does not do QLoRA at all (pretraining-focused); its quantization story is fp8/MXFP8 compute via `Float8Linear` (`torchao/float8/fsdp_utils.py` implements the same AG hooks for fp8 all-gather).

### 3.3 The `ignored_params` hybrid

`fully_shard(model, ignored_params=set(base_params))` ⇒ adapters sharded (DTensor), base left replicated and untouched (no AG, no RS for base, `_fully_shard.py:227-229`). Memory = full base per rank + `P_lora/W` adapter terms; comm = adapter RS only (~DDP-of-adapters but with sharded optimizer state). This is the right FSDP2 spelling of "base fits, adapters trained" — including the colocated case where the base is zero-copy shared with vLLM and must NOT be sharded or moved. It also sidesteps the bnb-can't-shard problem entirely.

---

## 4. DDP vs FSDP2 for LoRA-only training

DDP memory per rank (every rank identical): `P·b_model` params + `P_lora·b` grads (bucketed; default `bucket_cap_mb=25`, `torch/nn/parallel/distributed.py:31`; `gradient_as_bucket_view=True` avoids the 2× grad copy) + `n_opt·P_lora·4` optimizer. Comm per step: one all-reduce of `P_lora` (only `requires_grad` params are bucketed). With `P_lora` ~10–50 MB this is a **single sub-millisecond-amortized collective** — effectively free even on PCIe-only boxes.

FSDP2-for-LoRA changes exactly one persistent term — base params `P·b → P/W·b` — and pays for it with per-block all-gathers of the *entire frozen base* every forward (+ backward if resharding): comm volume `≈ 2·P·b_param` per step vs DDP's `P_lora·b`, i.e. **3–4 orders of magnitude more bytes on the wire**, plus latency exposure on weak interconnects (multi-node Ethernet, or PCIe boxes without NVLink).

**Decision rule (auto-config):**

```
M_ddp_peak = P·b_model + P_lora·(b + 4·n_opt) + A_live(mb_min, ...) + C_overhead
if M_ddp_peak ≤ VRAM_budget_trainer:           → DDP (or fully_shard+ignored_params if you
                                                  want DTensor checkpoints / sharded adapter state)
elif activations are the binding constraint:   → stay DDP, enable full AC / fused loss / smaller mb first
else (base params don't fit replicated):       → FSDP2, reshard_after_forward=True, per-block wrap;
                                                  consider reshard=int(intra-node) on multi-node
QLoRA wrinkle: 4-bit base shrinks P·b_model 4×, so DDP wins far more often;
               and bnb base *cannot* be FSDP2-sharded anyway (§3.2) — bnb + won't-fit ⇒ NF4 or more GPUs.
```

Crossover intuition: sharding base params saves `P·b·(1−1/W)`; it only matters when that quantity rescues an otherwise-infeasible config or buys a ≥2× micro-batch increase that AC couldn't. For AgileRL's typical envelope (≤8B base, 4-bit or bf16, LoRA, A100-40/80), DDP is strictly better in almost all colocated configs — FSDP2 earns its keep at bf16 ≥13B on 40 GB cards, or full fine-tuning, or when grad+optim states are non-trivial (`P_t` large). Note also: in COLOCATED mode the time-sliced vLLM engine wants the base resident and unsharded for zero-copy sharing — sharding the base conflicts with that design; FSDP2-with-sharded-base fits the ASYNC/decoupled topology better, where the trainer GPUs don't host vLLM.

(HSDP is the middle ground if you outgrow one node with FSDP2: shard intra-node, replicate inter-node — set via 2-D mesh, §1.8.)

---

## 5. Ecosystem heuristics for auto-configuration

### 5.1 torchtitan

- **No automatic micro-batch picking.** `training.local_batch_size` (default 8) is user-set; `global_batch_size` defaults to `local_batch_size × dp_degree`; grad-accum is derived. Auto-config must do its own solve (§2.4).
- **Wrapping policy is the codified heuristic** (`torchtitan/distributed/fsdp.py::apply_fsdp`): per-block, embeddings separate, `[norm, lm_head]` grouped with `reshard_after_forward=False` ("FSDP would prefetch them immediately"); weight-tied models group embed+norm+lm_head; **PP ⇒ reshard_after_forward=False** (`get_fsdp_reshard_after_forward_policy`: "default" = `not pp_enabled` — backward follows forward immediately per microbatch, resharding is pure overhead).
- **Activation checkpointing config**: `mode ∈ {none, selective, full, memory_budget}`, default `selective` with `selective_ac_option="2"` (checkpoint every 2nd block); `"op"` mode saves high-arithmetic-intensity ops (matmuls/SDPA via `_save_list`, only every other matmul) and recomputes the rest; `memory_budget` mode (requires compile) exposes the partitioner's automatic compute↔memory tradeoff with `memory_budget` default 0.5 and a pareto-visualization flag. The sensible AC ladder for an auto-config: none → selective-op → selective-k (k=2,4) → full.
- **Memory estimation**: `--memory_estimation.enable` (historic `scripts/estimate/estimation.py`, v0.2.2) = `FSDPMemTracker` under `FakeTensorMode`, FSDP/HSDP only (no TP/PP/CP), reports per-module snapshots and tracker-vs-CUDA peak.
- PyTorch additionally ships an **auto-SAC ILP stack**: `torch/distributed/_tools/{sac_estimator,sac_ilp,ilp_utils}.py` — estimates per-op recompute time/memory and solves a knapsack for the best selective-AC policy under a memory budget. Research-grade but directly reusable machinery.

### 5.2 torchtune

- **No predictive estimator**; runtime utilities only: `training.get_memory_stats`/`log_memory_stats` (alloc/reserved/peak logging) plus the memory-optimization toolkit: per-layer AC, **activation offloading** (`enable_activation_offloading`, stream-overlapped CPU offload of saved activations — composes with AC), torchao `CPUOffloadOptimizer` ("expect RAM ≈ 4× model size"), optimizer-in-backward (frees grad memory eagerly; FSDP2 supports it via post-accumulate-grad hooks — `test_fully_shard_memory.py:345-357` and the D2H special-casing at `_fsdp_collectives.py:711-733`), 8-bit optimizers, and `fsdp_cpu_offload`. The torchtune docs' decision table (memory_optimizations.html) is the closest thing to a published heuristic ladder: LoRA/QLoRA → AC → activation offload → optimizer-in-backward/8-bit → CPU offload.
- LoRA recipe defaults: `fsdp_reshard_after_forward=True`, `fsdp_cpu_offload=False`, LoRA weights init on CPU when offloading (`recipes/lora_finetune_distributed.py:568`).

### 5.3 Distillation for AgileRL's auto-config

1. Choose topology first: adapters-only + base fits ⇒ DDP (or ignored_params FSDP2); else FSDP2 per-block, reshard=True, bf16 param_dtype / fp32 reduce_dtype, root+lm_head unresharded.
2. Compute `M_trainer_peak` closed-form (§2.4) over candidate `(mb, AC mode)` pairs; prefer raising `mb` before easing AC for prefill-heavy workloads (activations scale with `mb·s`); always force fused/chunked loss for large-V models.
3. Verify winner with a `FakeTensorMode` + `FSDPMemTracker` dry run (no GPU time needed) before launching.
4. Keep 1.5–2.5 GB/GPU headroom for CUDA context, NCCL buffers, and allocator fragmentation; under colocated sleep/wake add the vLLM resident set to the budget.

# AgileRL LLM RL Stack — Exhaustive Config-Knob Inventory (memory & throughput)

Repo audited: `/Users/michaeldoherty/git/AgileRL/.claude/worktrees/auto-config-llm-batch` (main, HEAD `6da75e08`).
Branch comparison: `deepspeed-fsdp` in `/Users/michaeldoherty/git/AgileRL` (see §13).

All `file:line` references are relative to the worktree root unless prefixed with `[deepspeed-fsdp]`.

Legend for the "Set by" column: **U** = user-settable constructor/config arg; **H** = hardcoded; **D** = derived internally; **DEAD** = declared but never used.

---

## 1. vLLM engine configuration (colocated rollout engine)

### 1.1 `VLLMConfig` dataclass — `agilerl/utils/algo_utils.py:1416-1495`

| Knob | file:line | Default | Controls | Memory impact | Throughput impact | Set by | Should be derived from |
|---|---|---|---|---|---|---|---|
| `tensor_parallel_size` | algo_utils.py:1474 | `1` | vLLM TP degree; also defines trainer rank→TP subgroup layout (base.py:4749-4764) | Splits engine weights/KV across GPUs | TP comm overhead vs per-GPU batch | U | GPU count, model size vs VRAM, interconnect (NVLink vs PCIe) |
| `gpu_memory_utilization` | algo_utils.py:1475 | `0.3` | Fraction of total GPU memory vLLM may claim (weights + KV + activations) | THE primary trainer/rollout memory split in colocated mode | More KV cache → more concurrent seqs → higher gen throughput | U | VRAM, model size+dtype, trainer peak (micro-batch × seq len × optimizer), sleep_mode on/off |
| `max_num_seqs` | algo_utils.py:1476 | `8` | Max concurrent sequences in vLLM scheduler | KV-cache working set | Decode batch size; should be ≥ `data_batch_size_per_gpu × group_size` to avoid queuing (docstring says ≥ group_size, algo_utils.py:1425-1427) | U | rollout batch (prompts×group), KV budget, env profile (decode-heavy wants this high) |
| `swap_space` | algo_utils.py:1477 | `None` | **DEAD** — declared but never forwarded to `LLM(...)` (absent from llm_kwargs base.py:4773-4797) | none (no-op) | none | DEAD | CPU RAM size; should either be wired or removed |
| `enforce_eager` | algo_utils.py:1478 | `None` | **DEAD** — declared, never forwarded | CUDA-graph memory (~hundreds MB–GBs) would be saved if wired & True | Eager mode slows decode ~10-20% | DEAD | VRAM headroom; small GPUs want True |
| `sleep_mode` | algo_utils.py:1479 | `False` | Enables vLLM `enable_sleep_mode` + time-slice sleep/wake around learn/generate | When True, frees vLLM weights+KV during training (level-2 sleep) | Sleep/wake cycles + full weight re-load each iteration cost wall time | U | topology mode: must be True for colocated when trainer needs the GPU; incompatible with populations on one device (warning algo_utils.py:1489-1495) |
| `dtype` | algo_utils.py:1480 | `None` (vLLM picks) | Engine weight dtype (`"bfloat16"`/`"float16"`) | Engine weight footprint | Marginal | U | model config dtype; should match trainer dtype to keep weight-sync lossless |
| `quantization` | algo_utils.py:1481 | `None` | vLLM quantization method (`"awq"`, `"gptq"`, …) | Big engine weight reduction | Dequant overhead | U | model size vs VRAM; quantization initiative |
| `stop_sequences` | algo_utils.py:1482 | `None` | `SamplingParams.stop` | Shorter completions → less KV + shorter train sequences | Faster generation | U | env profile (e.g. `</answer>` tags) |
| `presence_penalty` / `frequency_penalty` | algo_utils.py:1483-1484 | `0.0` | SamplingParams | Indirect (completion length) | Indirect | U | env profile |
| `kv_cache_memory_bytes` | algo_utils.py:1487 | `None` | Pins KV cache size in bytes; skips vLLM's auto-profiling (`determine_available_memory` early return) | Exact KV cache budget — the most precise existing rollout-memory knob; also required for multi-process-per-GPU safety (docstring algo_utils.py:1449-1470) | Bounds concurrent tokens | U | VRAM − trainer peak − engine weights; an auto-config layer should compute this directly |

### 1.2 Hardcoded engine kwargs — `LLMAlgorithm._configure_vllm`, `agilerl/algorithms/core/base.py:4723-4816`

| Item | file:line | Value | Notes |
|---|---|---|---|
| `max_model_len` | base.py:4778 | `self.max_model_len` (algo arg, default 1024) | Engine context length = trainer context length; single shared value |
| `max_num_batched_tokens` | base.py:4781-4782 | **`max_num_seqs * max_model_len`** (H) | Derived heuristic, not user-settable. At defaults 8×1024=8k is fine, but e.g. 64 seqs × 32k ctx → 2M token prefill budget: effectively disables chunked-prefill admission control and inflates activation/profiling memory. Should be its own knob derived from env profile (prefill-heavy wants larger; decode-heavy wants ~4-8k) |
| `distributed_executor_backend` | base.py:4779 | `"external_launcher"` (H) | Required for accelerate-launched colocated SPMD |
| `seed` | base.py:4780 | `process_index // tp_size` (D) | Per-TP-group sampling seed |
| `model_impl` | base.py:4783 | `"vllm"` (H) | |
| `enable_sleep_mode` | base.py:4784 | from `vllm_config.sleep_mode` | |
| `enable_prefix_caching` | — | **never passed** (vLLM default: on in v1) | Implicit. GRPO sends each prompt `group_size` times as separate requests (base.py:3946-3950, `n=1` hardcoded at base.py:3993); prefill dedupe across the group relies entirely on vLLM's default prefix caching. Should be an explicit knob (prefill-heavy envs benefit massively; tiny-VRAM cases may want it off) |
| `enforce_eager`, `swap_space`, `cpu_offload_gb`, `max_seq_len_to_capture`, `block_size` | — | never passed | All would matter for auto-sizing |
| immediate `llm.sleep(level=2)` after init | base.py:4812-4813 | H | Sleep level **2** hardcoded — discards weights. Known-bad for bnb 4-bit (re-quantize garbage); the validated fix (standby patch keeping `"weights"` allocs resident) lives outside main |
| `VLLM_ATTENTION_BACKEND` | base.py:4801-4810 | env var pass-through | Only env-var-driven engine option acknowledged |

### 1.3 Generation kwargs (vLLM path) — `_generate_with_vllm_colocate`, base.py:3874-4084

| Item | file:line | Value | Notes |
|---|---|---|---|
| `n` | base.py:3993 | `1` (H) | Group fan-out done by duplicating prompts instead of `n=group_size`; trades vLLM-native dedupe for explicit replication |
| per-prompt `max_tokens` | base.py:3923-3931, 4007-4010 | `min(max_output_tokens or max_model_len, max_model_len − prompt_len)`, raised to `min(min_output_tokens, room)` | Derived; the only place prompt length feeds output budget |
| sampling: `temperature, top_p, top_k, min_p, repetition_penalty, min_tokens` | base.py:3992-4001 | from algo args | GRPO defaults: temp 0.9 (grpo.py:202), top_p 0.95 (204), top_k 50 (205), min_p 0.0 (206), repetition_penalty 1.0 (203) |
| eval temperature | grpo.py:507-509 (also ppo_llm.py:411) | **`0.01` hardcoded** when `training=False` | "Almost deterministic" eval; not configurable |
| TP all-gather of prompts + full-batch generate on every rank, then slice | base.py:3952-3990, 4024-4034 | H | Every rank in a TP group generates the whole group batch and keeps its slice |

### 1.4 HF-generation fallback (no vLLM)

| Knob | file:line | Default | Notes |
|---|---|---|---|
| `hf_generate_chunk_size` | grpo.py:213,386-394; ppo_llm.py:186,310-317 | `None → 1` | Prompts per HF `generate` chunk; inner loop is still per-prompt with the group as the batch dim (grpo.py:451-501). Memory: group_size×seq KV per generate call. Ignored under vLLM (warning) |
| `GenerationConfig` | grpo.py:395-406 | `do_sample=True, max_length=max_model_len, max_new_tokens=max_output_tokens, min_new_tokens, pad_token_id, repetition_penalty, top_p, top_k, min_p` | HF path mirror of SamplingParams |

---

## 2. Sleep/wake + colocated memory time-slicing

| Mechanism | file:line | Behavior | Knob? |
|---|---|---|---|
| `_prepare_vllm_for_training` | base.py:4911-4921 | `torch.cuda.empty_cache()` then `llm.sleep(level=2)` on main process | level hardcoded; sleep level should be a knob (level-1 vs level-2 vs standby-patch) derived from quantization + VRAM |
| `_prepare_vllm_for_generation` | base.py:4923-4933 | `empty_cache()` → `llm.wake_up()` → optionally move trainer params to CPU → `_move_model_to_vllm()` | |
| `use_memory_efficient_params` | base.py:2066 (default `True`), grpo.py:207 | During generation, moves the **entire trainer model to CPU** (`move_params_to_cpu`, llm_utils.py:403-414) and back for training (`_memory_efficient_params` ctx, base.py:3392-3412) | U. Only allowed with vLLM **and** sleep_mode (guards base.py:2124-2138). Incompatible with ZeRO-3 (warns + no-ops, base.py:3401-3408). Costs 2 full PCIe transfers of model weights per iteration — should be derived from (VRAM − vLLM budget − trainer peak) |
| Weight sync trainer→vLLM | `_move_model_to_vllm` base.py:3839-3872 | `merge_adapter(["actor"])` → per-parameter `llm_model.load_weights([(name, param)])` → `unmerge_adapter()` → `reset_prefix_cache()` (base.py:3871) | All hardcoded. Full dense-weight copy each learn→generate transition (LoRA merge/unmerge each time). Throughput cost grows with model size. The colocated zero-copy base-weight sharing redesign exists on the quant branch (PR #522 lineage), not on this main |
| Per-`learn` cache scrubbing | grpo.py:525-528, sft.py:213-216 | `gc.collect()` + `torch.cuda.empty_cache()` (+ mps) at the top of every learn call | H; measurable throughput cost on fast iteration loops |

---

## 3. Batch-size hierarchy / gradient accumulation

The effective hierarchy in the reasoning flow:

```
env.data_batch_size_per_gpu (prompts per rank per step, default 8; llm_envs/base.py:74)
  × group_size (GRPO fan-out, default 8; grpo.py:201)
  = num_samples entering learn() per rank        ← actual rollout/train sequence count
algo batch_size (default 16; grpo.py:195)        ← used ONLY for DeepSpeed micro/accum bookkeeping
  ÷ num_processes = batch_size_per_process (base.py:4236)
  ÷ gradient_accumulation_steps (from DS config) = micro_batch_size_per_gpu (base.py:4255-4257)
learn() minibatch loop step = min(num_samples, micro_batch_size_per_gpu) (grpo.py:596-600)
```

| Knob | file:line | Default | Controls | Memory | Throughput | Set by | Derive from |
|---|---|---|---|---|---|---|---|
| `batch_size` (algo) | grpo.py:195; base.py:2046 | `16` | Total nominal optimizer batch across ranks; drives DS `train_micro_batch_size_per_gpu` & `gradient_accumulation_steps` reconciliation | Indirect | Indirect | U | desired effective batch; group_size; world size |
| `micro_batch_size_per_gpu` | grpo.py:209; base.py:2063, 4210-4304 | `None` → derived; **create_population defaults it to `BATCH_SIZE`** (utils.py:147-152, with an inline `NOTE we should take a look into deepspeed auto batch-sizing`) | Per-rank per-forward sequence count in learn(); under DS also written into `ds_config["train_micro_batch_size_per_gpu"]` (base.py:4268, 4295) | **Primary trainer activation-memory knob** (activations ∝ micro_bs × seq_len; logits ∝ micro_bs × seq_len × V on unfused path) | More accumulation = more steps per optimizer update | U | VRAM, seq len (max_model_len), vocab size, liger/fused flags, gradient checkpointing |
| `gradient_accumulation_steps` | DS config (benchmarking/configs/ds_config.json: `2`); reconciled at base.py:4238-4303 | DS-config-owned | Derived as `batch_size / num_processes / micro_batch_size_per_gpu` (base.py:4296-4303) when micro specified; else read from DS config | none (just step count) | linear | U (DS json) / D | batch_size, micro, world size |
| divisibility constraints | base.py:4230-4234, 4279-4287 | — | `batch_size % num_processes == 0`; `batch_size % (micro × num_processes) == 0` — hard errors | — | — | — | auto-config must respect these |
| **GRPO group/batch mismatch** | grpo.py:553-559 | — | `num_samples % group_size == 0` enforced in learn; but `num_samples = data_batch_size_per_gpu × group_size` is set by the **env**, while `batch_size` (the DS bookkeeping value) is a separate arg — nothing reconciles them. DS loss averaging assumes its micro/accum mapping matches the actual loop, which is only true if the user keeps `batch_size == data_batch_size_per_gpu × group_size × num_processes` by hand | — | — | implicit | should be a single derived quantity |
| `update_epochs` | grpo.py:200 | `1` | Passes over the rollout per learn() | none | linear ×epochs | U | algo choice (off-policy drift) |
| `group_size` | grpo.py:201 | `8` | Completions per prompt (GRPO advantage groups); also generation fan-out | rollout KV ∝ group; learn batch ∝ group | generation tokens ∝ group | U (+ **HPO-mutable**, see §10) | reward variance needs vs token budget |
| `data_batch_size_per_gpu` (env) | llm_envs/base.py:74 | `8` | Prompts per rank per env step (DataLoader batch size; llm_envs/base.py:101-114) | rollout + learn batch | step granularity | U | with group_size and micro: the master scheduling quantity |
| `batch_size` (multiturn loop) | train_llm.py:1279, 1345 | `INIT_HP["BATCH_SIZE"]` | Number of parallel multi-turn envs: `SyncMultiTurnVecEnv(env_factory, batch_size, group_size)` → batch×group simultaneous trajectories all generated in one vLLM batch per turn | rollout KV ∝ batch×group×ctx | env parallelism | U | max_num_seqs, KV budget, env profile |
| DPO/SFT micro-batch | dpo.py:243-246; sft.py:238-241 | `min(num_samples, micro_batch_size_per_gpu)` | Same micro pattern; DPO doubles it again internally (chosen+rejected forward) | DPO: 2× sequences per step | — | D | |

---

## 4. Context length & how it flows

| Stage | Knob | file:line | Default | Notes |
|---|---|---|---|---|
| Algorithm | `max_model_len` | grpo.py:212, 383-385; ppo_llm.py:185; reinforce_llm.py | `1024` | Single value used for: vLLM engine `max_model_len` (base.py:4778), vLLM `max_num_batched_tokens` product (base.py:4781), HF `GenerationConfig.max_length` (grpo.py:398), per-prompt output budget (base.py:3924). Falls back to `max_output_tokens` if None (grpo.py:383-385); one of the two must be set (grpo.py:374-378) |
| Algorithm | `max_output_tokens` | grpo.py:210, 379-381 | `None` → max_model_len | Completion cap. Also passed to Liger GRPO as normalizer (grpo.py:1079) |
| Algorithm | `min_output_tokens` | grpo.py:211 | `None` → 0 | `SamplingParams.min_tokens` (base.py:3999-4001); can override room cap (base.py:3929-3930) |
| Env (dataset gyms) | `max_context_length` | llm_envs/base.py:75, 87, 188-215 | `None` | **Filters out** training samples whose prompt exceeds `max_context_length − min_completion_length` (base.py:201-205). Completely decoupled from algo `max_model_len` — user must pass the same number to both (benchmark does: `max_context_length=init_hp["MAX_MODEL_LEN"]`, benchmarking_llm_reasoning.py:124) |
| Env (preference) | padding to `max_context_length` | llm_envs/preference.py:81-97 | — | When set, **every** chosen/rejected pair is padded to the full `max_context_length` (`padding="max_length"`) → DPO trains on fixed max-size tensors regardless of actual lengths. Without it, pads to batch max (preference.py:99-122). Memory-wasteful hardcode |
| Env (SFT) | `padding="longest"` | llm_envs/sft.py:85-87 | — | Dynamic per-batch padding (better) |
| Multi-turn wrapper | `max_model_len`, `max_output_tokens` | llm_envs/token_observation.py:25-34, 117-122 | `None` | Sliding-window prompt truncation: keeps initial user segment, drops oldest middle turns, records `stitch_prefix_ids` so the full sequence is reassembled for training (`max_prompt_tokens_for_sliding_window` llm_utils.py:66-90; stitching base.py:4066-4073, llm_utils.py:417-552). Means **training sequences can exceed the engine context** — trainer activation memory governed by full stitched length, not max_model_len |
| Guard | prompt > max_model_len | base.py:3925-3927 | — | hard ValueError |

**No packing anywhere.** Sequences are stacked + right-padded to the longest sequence of the batch (`stack_and_pad_experiences`, algo_utils.py:1286-1316; pad value = `pad_token_id`). Mixed-length groups burn quadratic attention on pads (masked but computed in sdpa). A packing/sorting-by-length option is an obvious missing throughput knob.

---

## 5. Trainer model memory: dtype, LoRA, checkpointing, value/reference heads

| Knob | file:line | Default | Controls | Memory | Set by | Derive from |
|---|---|---|---|---|---|---|
| model load dtype | llm_utils.py:200-204 | **bf16 without accelerator, fp16 with accelerator** + `attn_implementation="sdpa"` | base-model weights | model_size × 2 bytes | H (overridable via `model_config` dict, grpo.py:192) | hardware bf16 support; DS bf16 config (note: fp16-under-accelerator clashes with bf16 DS configs); flash-attention availability |
| `model_config` | grpo.py:192; base.py:2071 | `None` | full kwargs passthrough to `from_pretrained` (quantization_config, attn impl, etc.) — the only current entry point for bnb 4/8-bit trainer quantization | arbitrary | U | model + VRAM |
| `lora_config` | base.py:2055, 2102-2116 | `None` → warns + defaults `r=16, alpha=32, target_modules="all-linear", dropout=0.05` | trainable param count → optimizer state size, gradient size, weight-sync cost | LoRA params ≈ 2·r·d per layer; AdamW states ×2 | U (also via INIT_HP `LORA_R=16, LORA_ALPHA=64, LORA_DROPOUT=0.0, TARGET_MODULES` — utils.py:81-99; note alpha default differs 32 vs 64) | model size, task; mostly user-policy |
| liger forces `exclude_modules=["lm_head"]` | base.py:2117-2122 | — | LoRA never applied to lm_head with liger | — | H | — |
| `gradient_checkpointing` | grpo.py:223; base.py:2072 (default `True`); enabled at wrap_models base.py:2720-2732 with `use_reentrant=False` | `True` | recompute activations | huge activation savings; ~30% slower fwd+bwd | U | micro_bs × seq_len vs VRAM — prime auto-config target |
| `torch_compiler` | base.py:2073, 3342-3357 | `None` | torch.compile mode | — | kernel fusion speedups | U | incompatible with DeepSpeed (skipped) and with gradient checkpointing (disables ckpt!) — base.py:3343-3356 |
| `use_value_head` (LLMPPO) | base.py:2058; ppo_llm | `False` (True for PPO) | adds `AutoModelForCausalLMWithValueHead` + critic adapter; **doubles** fused-forward batch (`ids.repeat(2,1)`, base.py:3601-3604) | 2× sequences per training forward | algo-fixed | — |
| `use_separate_reference_adapter` | grpo.py:229 (default **True**); base.py:2056 | GRPO/PPO/REINFORCE: True; DPO: True; SFT: False | Reference = dedicated frozen LoRA adapter copy vs disabling actor adapter (`disable_adapter_layers`, base.py:3124). Deprecation warning says prefer False (base.py:2252-2265) | adapter copy is small; but with separate adapter the no-grad pass batches reference+actor(+critic) as `ids.repeat(N,1)` (base.py:3654-3668) → N× no-grad forward batch; with False, reference logprobs are a **second full forward** (base.py:3678-3685) | U | — (semantics choice); memory layer should know N multiplies the no-grad batch |
| reference refresh | `set_reference_policy` base.py:3002-3031 | per dataset epoch (train_llm.py:706,1032) | copies actor→reference adapter, or **merges actor LoRA into dense base in place** (base.py:3033-3103) when no separate adapter | merge path mutates base weights (matters for zero-copy sharing) | cheap | D | — |
| **No full reference model is ever kept** — reference is always adapter-based (LoRA-disable trick or adapter snapshot). Zero extra full-model memory. | | | | | | |

---

## 6. Loss / log-prob memory paths (the (B,T,V) problem)

| Knob | file:line | Default | Controls | Memory | Set by | Derive from |
|---|---|---|---|---|---|---|
| `use_liger_loss` | grpo.py:225 (False), dpo.py:117, sft.py:124, ppo_llm.py:206; gate base.py:2088-2094 | `False` | Liger chunked fused-linear loss: GRPO/GSPO/CISPO via `LigerFusedLinearGRPOFunction` (grpo.py:1063-1088), DPO via `_LigerDPOWithAlpha` (fused_loss.py:440-485), SFT via `LigerCrossEntropyLoss` (sft.py:169-172), PPO via custom turn-aware chunk loop (fused_loss.py:200-240) | avoids materializing (B,T,V) logits **with grad** — the dominant train-time peak for large vocabs | U | should default on when liger installed (docs/llm_finetuning/fused_logprobs.rst:17-23 already describes a `None`→auto default that **does not match this code** — doc/code drift) |
| `use_fused_linear_logprobs` | grpo.py:235 (False); base.py:2075, 3469-3473, 3714 | `False` | no-grad rollout/reference logprobs via lm_head-identity patch + chunked matmul (`_logprobs_from_hidden_fused`, base.py:4146-4208) | avoids no-grad (B,T,V); ~0.3GB vs ~5GB per doc example (fused_logprobs.rst:92-109) | U | same auto-on policy |
| `cast_logprobs_to_fp32` | grpo.py:236; base.py:2076, 2020-2037 | `True` | fp32 promotion inside logprob reduction | docstring: disabling saves ~18 GB on unfused path at B=8,T=2048,V≈152k; ~6 MB on fused | U | vocab size, dtype tolerance |
| `_chunk_rows` (unfused logits) | base.py:4091 | `1` | rows per fp32 logprob chunk | bounds fp32 workspace to (1,T,V) | H (private default) | VRAM headroom — explicitly invites tuning in docstring (base.py:4096-4099) |
| `_chunk_rows` (fused hidden) | base.py:4153 | `1024` | flat (B·T) rows per matmul chunk | workspace = chunk×V | H | VRAM |
| Liger `chunk_size` | grpo.py:1086; fused_loss.py:226 (PPO), fused_loss.py:457 (DPO) | `1` | sequences per Liger chunk | workspace ∝ chunk×T×V | H | VRAM — should be a knob |
| grad-path no chunking | `_fused_forward` base.py:3566-3617 | — | training forward always one chunk ("preserve gradient-checkpoint routing", base.py:3577-3580) → micro_batch_size_per_gpu **is** the grad-time chunk | — | — | micro must be sized for this |
| no-grad chunking | `_fused_forward_no_grad` base.py:3619-3689; `_fused_model_pass(batch_size=...)` base.py:3431-3564 | micro_batch_size_per_gpu | old/ref logprob pass chunked; preallocated output buffers avoid 2× cat peak (base.py:3533-3563) | — | — | — |
| `temperature` division of logits | base.py:3515, 3761 | from algo | training logits divided by sampling temperature | — | — | inherits sampling knob | — |
| `calc_position_embeddings` | grpo.py:208 | `True` | explicit position_ids from attention mask | tiny | tiny | U | model family |

---

## 7. Optimizer & schedule

| Knob | file:line | Default | Notes |
|---|---|---|---|
| optimizer class | base.py:3145-3161 | **AdamW** (or `DummyOptimizer` when DS config defines its own optimizer) | No optimizer choice knob; no weight_decay/betas exposure for LLM path (`init_llm_optimizer` optimizer_wrapper.py:67-102 passes only param groups + lr). Optimizer state = 2× fp32 per LoRA param (small for LoRA). Muon etc. would need the DS-native hook (separate ticket) |
| `lr` / `lr_critic` | grpo.py:197 (5e-7); ppo_llm.py:171-172 (actor 5e-7, critic 5e-5) | | `align_deepspeed_lr` overwrites DS config lr (llm_utils.py:611-636) |
| `cosine_lr_schedule_config` | grpo.py:215; algo_utils.py:1408-1413 (`num_epochs`, `warmup_proportion`) | `None` | disabled when DS config has an optimizer (base.py:2180-2190) |
| `max_grad_norm` | grpo.py:199 (0.1); ppo_llm 1.0 | | DS: overwrites `ds_config["gradient_clipping"]` (base.py:2173-2179) + mutation hook resync (base.py:4818-4840). Non-DS: `clip_grad_norm_` per group (base.py:3784-3791) |
| backward path | `_backward_pass` base.py:3774-3794 | | DS: `accelerator.backward` (engine handles accum/clip). Plain: backward→clip→step→zero per minibatch (no accumulation support outside DS on main!) |

---

## 8. Accelerator / DeepSpeed plumbing (main)

| Item | file:line | Notes |
|---|---|---|
| `create_llm_accelerator` | llm_utils.py:313-357 | Returns None on 0 GPUs; **requires** a DeepSpeed plugin otherwise (raises with setup instructions). Only hardware introspection in the entire stack: `torch.cuda.device_count()` (llm_utils.py:339). **No `mem_get_info`, no `get_device_properties`, no interconnect detection anywhere** (verified by grep) |
| ZeRO stage | read from DS config base.py:2192; stage-3 warning base.py:2193-2201 | stage is user-owned in the accelerate/DS json. Reference DS config: stage 2, `offload_optimizer: cpu`, bf16, micro 8, accum 2, 2e8 buckets (benchmarking/configs/ds_config.json) |
| ZeRO-3 gathers | `gather_if_zero3` llm_utils.py:138-166 used for checkpoint/weight-sync/merge paths | |
| per-agent accelerators | `get_llm_accelerator` llm_utils.py:360-387 | population idx 0 reuses, idx>0 fresh `Accelerator()` |
| mixed precision (no accelerator) | `_amp_ctx` base.py:3414-3429 | autocast bf16 hardcoded when CUDA bf16 supported |
| device resolution | grpo.py:238-248 | `cuda:{process_index}` under accelerator |

---

## 9. Per-algorithm default table (memory/throughput-relevant args only)

| Arg | GRPO (grpo.py:186-237) | LLMPPO (ppo_llm.py:156-206) | LLMREINFORCE (reinforce_llm.py:140-186) | DPO (dpo.py:92-120) | SFT (sft.py:102-126) | ILQL (ilql.py:84-108) |
|---|---|---|---|---|---|---|
| batch_size | 16 | 16 | 16 | 16 | 16 | 64 |
| micro_batch_size_per_gpu | None | None | None | None | None | — |
| max_model_len | 1024 | 1024 | 1024 | — | — | — |
| max_output_tokens | None | None | None | — | — | — |
| group_size | 8 | — (1 implicit) | — | — | — | — |
| temperature | 0.9 | 1.0 | 1.0 | — | — | — |
| update_epochs | 1 | 1 | 1 | 1 | 1 | — |
| gradient_checkpointing | True | True | True | True | True | — |
| use_liger_loss | False | False | False | False | False | — |
| use_fused_linear_logprobs | False | False | False | — | — | — |
| use_separate_reference_adapter | True | True | True | True | False | — |
| use_memory_efficient_params | True | True | True | — | — | — |
| use_vllm | False | False | False | — | — | — |
| use_value_head | False | True | False | False | False | (own Q/V heads) |
| gamma / gae_lambda | — | 1.0 / 1.0 | gamma 1.0 | — | — | 0.99 |
| extra | loss_type grpo/gspo/cispo (GSPO/CISPO are thin wrappers, gspo.py:16-18, cispo.py) | action_granularity "auto", turn_level_clip True, vf_coef 0.5 | — | beta 0.1, nll_alpha 1.0 | — | legacy offline stack, not vLLM/accelerate-integrated |

PPO turn-mode pooling (`pool_by_turns`, llm_utils.py:256-310) is a Python per-turn loop — throughput-relevant for many-turn episodes (O(num_turns) kernel launches).

---

## 10. HPO / evolutionary interactions

| Item | file:line | Notes |
|---|---|---|
| What HPO can mutate | `rl_hyperparam_mutation` hpo/mutation.py:412-452 | Any attribute registered as `RLParameter` in `hp_config` — plain `setattr(individual, attr, value)` with grow/shrink factors 1.2/0.8 (registry.py:110-134). LR mutation triggers optimizer reinit (mutation.py:441-450) |
| What's registered in the shipped configs | configs/training/llm_finetuning/grpo.yaml | `lr` (1e-7→1e-5), `beta` (1e-4→1e-2), **`group_size` (4→12)** — i.e. **HPO directly mutates a memory/throughput knob**: group_size changes rollout KV demand and learn-batch size with **no revalidation** of `max_num_seqs`, batch divisibility, or vLLM capacity after mutation |
| What HPO cannot touch | mutation.py:528-534, train_llm.py:81-97 | architecture/parameter/activation mutations hard-disabled for LLMs |
| batch_size mutation hazard | base.py:4210 only runs at `__init__` | if `batch_size`/`micro_batch_size_per_gpu` were registered as RLParameters, mutation would NOT re-run `_configure_batch_size_per_process` → DS config silently stale |
| Populations | utils.py:726-816; VLLMConfig.__post_init__ algo_utils.py:1489-1495 | each agent = full model copy (+ optional own vLLM); sleep_mode warns it can't be used with populations on a single device. POP_SIZE is a brute multiplier on everything |
| Cloning (tournament) | base.py:2773-2932 | clone via temp-dir save/load of adapters; vLLM/tp_group handles shared by reference (base.py:2886-2898) |

---

## 11. Hardware introspection — current state

- `torch.cuda.device_count()` — llm_utils.py:339 (GPU presence check only).
- `torch.cuda.is_bf16_supported()` — base.py:3425 (autocast dtype choice).
- `torch.backends.mps.is_available()` — device fallback (grpo.py:245) and cache clears.
- **Nothing else.** No `torch.cuda.mem_get_info`, no `get_device_properties`, no NVLink/interconnect probing, no model-size estimation, no free-memory-driven sizing. Every memory split is a static user number. (vLLM internally profiles for KV sizing — the only dynamic memory measurement in the stack, and `kv_cache_memory_bytes` can bypass it.)

---

## 12. Hardcoded / implicit things that SHOULD be knobs (auto-config targets)

1. **`max_num_batched_tokens = max_num_seqs × max_model_len`** (base.py:4781) — should be independent and env-profile-driven (prefill-heavy vs decode-heavy).
2. **vLLM sleep level 2** (base.py:4813, 4917) — should be `{1, 2, standby}` derived from quantization (bnb requires standby/level-1 semantics) and VRAM.
3. **`enable_prefix_caching`** never set — group_size-duplicate prompts depend on vLLM's default; make explicit, derive from env profile (multi-turn/long shared prompts → on).
4. **`enforce_eager` / `swap_space` declared but never forwarded** (algo_utils.py:1477-1478 vs base.py:4773-4797) — wire or delete.
5. **Eval temperature 0.01** (grpo.py:509, ppo_llm.py:411).
6. **`n=1` + prompt duplication for groups** (base.py:3993, 3946) — using `SamplingParams.n=group_size` would let vLLM share prefill KV natively.
7. **Trainer load dtype fp16-under-accelerator** (llm_utils.py:202) — should follow mixed-precision config (bf16 DS configs get an fp16 model today unless `model_config` overrides).
8. **`attn_implementation="sdpa"`** (llm_utils.py:203) — flash-attention-2 not auto-selected.
9. **Liger `chunk_size=1`** (grpo.py:1086, fused_loss.py:226,457) and **`_chunk_rows`** (base.py:4091 `1`; base.py:4153 `1024`) — VRAM-derivable.
10. **Fused/liger flags default False** while docs describe auto-on (fused_logprobs.rst:17-23) — resolve drift; auto-on when installed.
11. **Padding-only batching, no packing/length-sorting** (algo_utils.py:1286) — biggest untapped throughput knob for mixed-length groups; preference env even pads to full `max_context_length` (preference.py:87).
12. **`gc.collect()+empty_cache()` every learn()** (grpo.py:525-528).
13. **env `max_context_length` vs algo `max_model_len` duplication** — single source of truth needed; today the benchmark manually passes the same INIT_HP value to both (benchmarking_llm_reasoning.py:124 vs utils.py:783).
14. **`batch_size` (DS bookkeeping) vs actual learn batch (`data_batch_size_per_gpu × group_size`)** — nothing ties them; should be one derived quantity (utils.py:152 already carries a TODO for DS auto batch-sizing).
15. **Weight-sync strategy** — full dense merge→copy→unmerge per cycle (base.py:3839-3872); LoRA-adapter-only sync / zero-copy base sharing exist only off-main.
16. **`use_memory_efficient_params` CPU round-trip** — should be computed from whether trainer-peak + woken-vLLM fit, not defaulted True.
17. **Optimizer choice / weight decay** — AdamW hardcoded (base.py:3161).
18. **micro default = full batch** in `create_population` (utils.py:147-152) — pathological for big models; should derive from VRAM.
19. **HPO mutation of `group_size` without capacity revalidation** (mutation.py:438; grpo.yaml MIN/MAX_GROUP_SIZE).
20. **Sliding-window stitched sequences can exceed `max_model_len` at train time** (token_observation.py:205-272) — trainer memory bound is the stitched length; no cap knob exists for the training-side sequence length.
21. **vLLM TP rank seeding / external launcher / model_impl** (base.py:4779-4783) — fine as hardcodes, but an async/decoupled mode needs a different executor backend; today colocated-external-launcher is the only wired topology in this repo (the Ray decoupled topology lives in agilerl-integration).

---

## 13. `deepspeed-fsdp` branch: what the torch-native backend changes

Commits on top of main: `1f357a25` (Remove DeepSpeed: torch-native DDP/FSDP2), `feb1505d`, `5b6c66bd`, `844a5ccd`, `b2eb2a21`. Net −3,945/+1,642 lines; `agilerl/algorithms/core/base.py` −1,028 lines churn.

Knob-relevant deltas:

| Area | main (DeepSpeed) | deepspeed-fsdp (DDP/FSDP2) |
|---|---|---|
| Accelerator factory | `create_llm_accelerator(deepspeed_plugin=...)`, requires DS plugin | `create_llm_accelerator(fsdp_plugin=None, gradient_accumulation_steps=None)`; default = plain `Accelerator()` → DDP; FSDP2 only via plugin/config with `fsdp_version: 2` enforced (raises on FSDP1); passing `deepspeed_plugin` raises |
| Grad accumulation | DS json `gradient_accumulation_steps` reconciled into ds_config | owned by `Accelerator(gradient_accumulation_steps=…)` / accelerate yaml; `_configure_batch_size_per_process` derives `self.gradient_accumulation_steps = batch_size_per_process // micro_batch_size_per_gpu`; `_backward_pass` counts micro-batches itself (`self._micro_batch_count % accumulation_steps`), clips with `max_grad_norm` then steps — accumulation now works without DS |
| ZeRO stages / offload | stage 1-3 + cpu offload via DS json | gone. Replacements: DDP (whole weights per rank — required by colocated vLLM zero-copy weight sharing) or FSDP2 `fsdp_reshard_after_forward: true` (ZeRO-3-like) + `fsdp_cpu_ram_efficient_loading`. **No optimizer-CPU-offload equivalent** — an offload knob disappears |
| FSDP2 vs colocated vLLM | — | explicitly documented as incompatible with colocated weight sharing ("use this config for decoupled trainer/rollout setups or HF generation only", configs/accelerate/fsdp2_accelerate_config.yaml header) → **topology mode now constrains the sharding knob**: COLOCATED ⇒ DDP; ASYNC/DECOUPLED ⇒ DDP or FSDP2 |
| Param gathering | `gather_if_zero3` (deepspeed.zero.GatheredParameters) | `gather_full_params` (FSDP2 unshard/reshard; read-only) + `load_full_state_dict` / `get_state_dict` via `torch.distributed.checkpoint` |
| Knob surface for FSDP2 (accelerate yaml) | — | `fsdp_version: 2`, `fsdp_reshard_after_forward`, `fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP`, `fsdp_state_dict_type: FULL_STATE_DICT`, `fsdp_cpu_ram_efficient_loading`, `mixed_precision: bf16`, `gradient_accumulation_steps`, `num_processes` |
| Removed | `align_deepspeed_lr`, `_sync_deepspeed_gradient_clipping`, DS gradient_clipping override, DummyOptimizer-from-DS-config path | optimizer is always AgileRL-owned AdamW |

For the auto-config effort: on the new backend the trainer-side knob set collapses to {DDP vs FSDP2, reshard_after_forward, micro_batch_size_per_gpu, gradient_accumulation_steps, mixed_precision, gradient_checkpointing, liger/fused flags} — simpler to model than ZeRO stages, but introduces the hard coupling *colocated ⇒ no sharding ⇒ model must fit whole per GPU alongside the vLLM budget*.

---

## 14. Compact master list (knob → primary derivation inputs)

- `vllm_config.gpu_memory_utilization` (0.3) ← VRAM, model+dtype, trainer peak, sleep_mode
- `vllm_config.kv_cache_memory_bytes` (None) ← VRAM − engine weights − trainer peak (exact-pin alternative)
- `vllm_config.max_num_seqs` (8) ← data_batch×group, KV budget, decode-vs-prefill profile
- `max_num_batched_tokens` (H: seqs×len) ← env profile, VRAM
- `vllm_config.tensor_parallel_size` (1) ← model size vs per-GPU VRAM, GPU count, interconnect
- `vllm_config.sleep_mode` (False) + sleep level (H: 2) ← topology=colocated, quantization
- `use_memory_efficient_params` (True) ← does trainer fit beside woken engine
- `vllm_config.dtype`/`quantization` (None) ← model dtype/quant plan
- `max_model_len` (1024) ← env context profile (prompt+gen percentiles)
- `max_output_tokens`/`min_output_tokens` (None) ← env gen-length profile
- `batch_size` (16) / `micro_batch_size_per_gpu` (None→derived) / grad-accum ← VRAM, seq len, vocab, liger flags, world size, target effective batch
- `group_size` (8) ← reward-variance vs token budget; must co-move with max_num_seqs & batch
- `data_batch_size_per_gpu` (8) ← effective-batch plan
- `gradient_checkpointing` (True) ← activation memory vs speed
- `use_liger_loss`/`use_fused_linear_logprobs` (False) /`cast_logprobs_to_fp32` (True) ← vocab size, VRAM
- liger/logprob chunk sizes (H: 1 / 1024) ← VRAM headroom
- `lora_config` r/alpha/targets (16/32 all-linear) ← model size, task
- `use_separate_reference_adapter` (True) ← no-grad batch multiplier awareness
- ZeRO stage / offload (DS json) → branch: DDP vs FSDP2 reshard ← model size vs VRAM, topology mode
- `hf_generate_chunk_size` (1) ← only non-vLLM path
- env `max_context_length` (None) ← must equal max_model_len (single source of truth)
- sampling (temp 0.9, top_p .95, top_k 50, min_p 0, penalties 0) ← task policy, minor memory effect via lengths

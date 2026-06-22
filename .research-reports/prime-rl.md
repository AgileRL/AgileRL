# prime-rl (Prime Intellect async RL trainer) — config system, auto-sizing, packing, FSDP, and implicit sizing tables

Repo: `/Users/michaeldoherty/git/Misc/prime-rl` (read-only reference clone), HEAD `e5c91695` (2026-06-04).
All `file:line` references below are relative to the repo root at that commit.

prime-rl is an **async-first, three-process** RL stack:

- **Trainer** (`src/prime_rl/trainer/rl/train.py`) — torchrun process group, FSDP2 (+EP/CP/HSDP), consumes packed micro-batches.
- **Orchestrator** (`src/prime_rl/orchestrator/orchestrator.py`) — single asyncio process; schedules rollouts against the inference pool via OpenAI-compatible HTTP, assembles batches, ships them to the trainer over a filesystem or ZMQ transport.
- **Inference** (`src/prime_rl/inference/server.py`) — patched vLLM server(s) with `/update_weights`, `/load_lora_adapter`, `/init_broadcaster` endpoints.

There is **no Ray**: process orchestration is the `rl` entrypoint (subprocesses + `CUDA_VISIBLE_DEVICES` partitioning) on one node, or generated SLURM sbatch scripts multi-node. Weight sync is full-weights (filesystem snapshots or layerwise NCCL broadcast), *not* LoRA-adapter-diff sync (LoRA runs broadcast merged weights via filesystem; NCCL+LoRA is explicitly rejected, `packages/prime-rl-configs/src/prime_rl/configs/trainer.py:641-646`).

---

## 1. Config system — every perf-relevant knob

Configs are Pydantic models in a separate package `packages/prime-rl-configs/src/prime_rl/configs/{trainer,orchestrator,inference,rl,sft,shared}.py`, loaded via `pydantic-config` (TOML files composed left-to-right with `@`, CLI dotted overrides win; `docs/configuration.md:23-42`). The top-level `RLConfig` (`configs/rl.py:180`) nests `trainer` / `orchestrator` / `inference` plus *shared* fields (`model`, `seq_len`, `max_steps`, `wandb`, `ckpt`, `weight_broadcast`, `deployment`, `slurm`) that are propagated into sub-configs pre-validation by `propagate_shared_fields` (`packages/prime-rl-configs/src/prime_rl/utils/validation.py:10-195`) with a "fill-if-absent + conflict-on-disagreement" rule. Cross-process consistency is then *validated* (same model name, `trainer.model.seq_len >= orchestrator.seq_len` at `validation.py:285-293`, same max_steps/ckpt interval/broadcast type).

### 1.1 Trainer knobs (`configs/trainer.py`)

`ModelConfig` (trainer side):

| Knob | Default | Ref |
|---|---|---|
| `seq_len` | 2048 | trainer.py:119 |
| `attn` | `flash_attention_2` (eager/sdpa/fa2/fa3/fa4) | trainer.py:122, alias map :27 |
| `compile` (sub-config; per-block torch.compile, `fullgraph` flag) | None (off) | trainer.py:125, :61-63 |
| `ac` — activation checkpointing: `mode = full\|selective`, `freq` (every N layers), `targets` (norm/attn_proj/mlp/mla_up_proj/routed_experts/linear_attn) | None (off) | trainer.py:35-50, :128 |
| `ac_offloading` — `pin_memory=True`, `max_inflight_activations=5` | None (off); enabling it auto-enables `ac` (trainer.py:225-230) | trainer.py:53-58, :131 |
| `fsdp_cpu_offload` (params+grads+optim to CPU, pinned) | False | trainer.py:134 |
| `optim_cpu_offload` (optimizer state only) | False; mutually exclusive with fsdp_cpu_offload (trainer.py:238-242) | trainer.py:137 |
| `reshard_after_forward` | **True** | trainer.py:140 |
| `dp_replicate` (HSDP replication degree) | 1 | trainer.py:143 |
| `ep` (expert parallel degree) + `ep_comm_backend` (`torch`/`deepep`) + `deepep_num_sms=20`, `deepep_token_chunk_size` | 1 / torch | trainer.py:146-156 |
| `cp` (context parallel) + `cp_style` (`ring`/`ulysses`) | 1 / ring | trainer.py:158-162 |
| `impl` (`hf`/`custom`/`auto`) | auto (custom if supported) | trainer.py:164 |
| `optimization_dtype` / `reduce_dtype` | float32 / float32 (params compute in bf16 via FSDP mp_policy) | trainer.py:167-171 |
| `moe_use_grouped_mm` | True (needs SM90+) | trainer.py:173 |
| `fp8` (DeepGEMM blockwise FP8 training; SM90 + custom impl) | False | trainer.py:176 |
| `freeze_moe_router` | False | trainer.py:182 |
| `lora` — rank 16, alpha 32, dropout 0, target_modules covering attn+MLP+experts | None | trainer.py:79-104, :185 |
| `fused_lm_head_token_chunk_size` | `"disabled"`; `"auto"` → **8192** for RL (validator trainer.py:656-661); ints rejected for SFT (sft.py:349-357) | trainer.py:191 |

`TrainerConfig` extras: `matmul_precision="high"` (TF32; `"highest"` required on ROCm, trainer.py:532), `gc.interval=50` (deterministic synchronized GC to avoid stragglers, trainer.py:30-32), `dist_timeout_seconds=600` (trainer.py:553), `max_concurrent_runs=1` (multi-tenant LoRA training, trainer.py:562), `weight_broadcast` (filesystem default / nccl, trainer.py:454-490), `rollout_transport` (filesystem default / zmq with `hwm=10`, shared.py:243-264), optimizer union (`adamw` default lr=1e-6 wd=0.01 max_norm=1.0; `sgd`/`muon`/`sign_sgd`, trainer.py:310-360), scheduler union (constant default / linear / cosine).

**Notable absence: the RL trainer has NO `micro_batch_size` and NO `gradient_accumulation` knob.** Both are *emergent* from sequence packing (§4): a micro-batch is one packed bin of ≤`seq_len` tokens at batch dim 1, and the number of micro-batches per rank per step (= grad-accum steps) is whatever the orchestrator's batch packs into.

### 1.2 Orchestrator knobs (`configs/orchestrator.py`)

| Knob | Default | Ref |
|---|---|---|
| `batch_size` (rollouts per trainer step) | None → resolved to **128** if neither batch knob set | orchestrator.py:594, :847-848 |
| `token_batch_size` (tokens per trainer step; mutually exclusive with `batch_size`) | None | orchestrator.py:597, :843-845 |
| `oversampling_factor` (rollout-mode only; multiplies batch_size to derive in-flight capacity) | None (=1.0) | orchestrator.py:600 |
| `max_inflight_rollouts` | auto: `max(group_size, batch_size*oversampling_factor)`; **required explicit** for token-mode | orchestrator.py:603, :850-872 |
| `group_size` (GRPO rollouts per example; alias `rollouts_per_example`) | 1 | orchestrator.py:606 |
| `seq_len` (orchestrator-side cap; samples > this are dropped/truncated) | 2048 | orchestrator.py:609 |
| `num_train_workers` (trainer DP ranks the packer shards for; "TODO should be automatic from ZMQ connections") | 1, auto-set by deployment (§3) | orchestrator.py:612-614 |
| `max_off_policy_steps` (**the async/staleness dial**) | **8** | orchestrator.py:619-620 |
| `tasks_per_minute` (per-env-worker rate limit) | None | orchestrator.py:591 |
| `pool_size` (renderer tokenization slots) | None | orchestrator.py:522 |
| Per-env: `num_workers="auto"`, `max_retries=3`, `ratio`, `max_total_completion_tokens=-1`, `timeout` | | orchestrator.py:162-178 |
| Sampling: `temperature=1.0`, `max_completion_tokens` (None = generate to context limit) | | orchestrator.py:45-57 |
| Filters: pre/post-batch `gibberish`/`repetition`/`zero_advantage` (monitor-mode by default) | | orchestrator.py:549-565 |

Batching resolution invariants (`resolve_batching`, orchestrator.py:839-885): exactly one of `batch_size`/`token_batch_size`; `batch_size % group_size == 0`; `max_inflight_rollouts >= group_size`; conflict check between explicit `max_inflight_rollouts` and `oversampling_factor*batch_size`.

### 1.3 Inference knobs (`configs/inference.py`) — direct vLLM passthrough

| Knob | Default | Ref |
|---|---|---|
| `parallel.tp` / `parallel.dp` | 1 / 1 | inference.py:27-31 |
| `gpu_memory_utilization` | **0.9** | inference.py:304 |
| `model.max_model_len` | None (model config value) | inference.py:48 |
| `model.dtype` | auto | inference.py:45 |
| `model.enforce_eager` | False (hybrid cudagraphs) | inference.py:51 |
| `enable_prefix_caching` | None (vLLM default; force-enabled by KV offload, inference.py:372-380) | inference.py:301 |
| `api_server_count` | 1, heavily auto-adjusted (§2) | inference.py:307 |
| `data_parallel_size_local` | None, auto-derived multi-node | inference.py:310 |
| `enable_lora` / `max_loras=8` / `max_cpu_loras=100` / `max_lora_rank` | off | inference.py:283-296 (the 100 default is a deliberate workaround: adapters are added under new names, never swapped in-place, so stale in-flight requests don't crash — comment :289-291) |
| `enable_expert_parallel`, `all2all_backend` (5 options, default `allgather_reducescatter`), `enable_eplb`, `enable_dbo`, `use_deep_gemm` | off | inference.py:319-332 |
| `kv_cache_offload` — `native` (vLLM OffloadingConnector / TieringOffloadingSpec) or `mooncake` distributed store; `cpu.num_bytes` required, optional `disk.path` | None | inference.py:89-153, :336 |
| `enable_fp32_lm_head` | **True** (monkey-patched fp32 logits GEMM for logprob precision under FP8/bf16 — matters for the importance-ratio KL) | inference.py:345-346 |
| `logprobs_mode` | hardwired `processed_logprobs` | inference.py:518 |
| `vllm_extra` (arbitrary vLLM namespace overrides, e.g. `headless`) | {} | inference.py:348 |
| `deployment` — `single_node` / `multi_node{num_nodes}` / `disaggregated{num_prefill_nodes,num_decode_nodes,num_*_replicas,prefill/decode_env_overrides}` + router (`consistent_hash` default policy) | single_node | inference.py:187-265 |

### 1.4 SFT trainer (`configs/sft.py`) — the only place with explicit micro-batch

`data.batch_size=128` (global, in samples-worth-of-packed-sequences), `data.seq_len=128`, `data.micro_batch_size=1`, `pack_function = "cat"|"stack"` (sft.py:28-47). `grad_accum_steps = (batch_size * cp) / (world_size * micro_batch_size)`, asserted divisible (`src/prime_rl/trainer/sft/train.py:100-106`). CP forces `micro_batch_size==1` and `pack_function=="cat"` (sft.py:276-301). `loss_impl`: `liger | torch | liger_fused | quack_fused` (sft.py:228-229).

---

## 2. Auto-sizing: what is derived vs. user-set

**Headline: prime-rl does essentially ZERO memory-based auto-sizing.** There is no GPU-VRAM introspection feeding config, no heuristic table keyed on model size, no OOM-retry loop. `torch.cuda.mem_get_info()` appears only in benchmark *reporting* (peak-mem as % of total, `src/prime_rl/trainer/utils.py:246,279-280`; `src/prime_rl/trainer/sft/train.py:334`), and the benchmark harness merely *recognizes* OOM in output for triage (`benchmarks/scripts/run_single_benchmark.py:36-46`: greps for `"torch.OutOfMemoryError: CUDA out of memory."`). Memory fitting is delegated to the human + the `--bench` workflow + example configs (§6). The docs say so outright: "You need to make sure that the model will fit into the available GPU memory. We will not go into the details on how to do this" (`docs/inference.md:75`).

What IS auto-derived is **topology arithmetic** — a substantial validator network that fills in consistent parallelism/worker counts from a small set of user inputs:

1. **FSDP shard degree is always derived**: `dp_shard = -1` → `world_size // (dp_replicate * cp * pp)` (`src/prime_rl/trainer/parallel_dims.py:62-64`, `get_parallel_dims` :273-292). The user only ever sets `dp_replicate`/`cp`/`ep`; world size comes from torchrun.
2. **Trainer DP worker count → orchestrator packer**: single-node `orchestrator.num_train_workers = num_train_gpus // cp` (`configs/rl.py:467-471`); multi-node `= num_train_nodes * gpus_per_node` (rl.py:489).
3. **Inference DP fill**: single-node, if `num_infer_gpus != dp*tp`, then `dp = num_infer_gpus // tp` (asserting divisibility) (`configs/rl.py:474-480`). Multi-node without EP: `dp_per_node = gpus_per_node // tp`; auto-sets `parallel.dp`, `data_parallel_size_local`, `api_server_count` (rl.py:537-548). With EP: validates `dp*tp == total inference GPUs` and derives `data_parallel_size_local = gpus_per_node // tp` (rl.py:501-531).
4. **`api_server_count` auto**: raised to DP size so the NCCL broadcast group doesn't deadlock waiting for workers that don't exist (comment rl.py:481-486); forced to 1 when LoRA (vLLM limitation), 0 when headless (`configs/inference.py:428-446`).
5. **NCCL broadcast world size**: `inference_world_size = total_infer_nodes * gpus_per_node` (with an explicit comment about a double-counting bug they fixed, rl.py:550-563; disaggregated path rl.py:594-598). Marked "TODO: Should not be configurable, but auto-inferred" (trainer.py:480-482).
6. **`nodes_per_fsdp_group` → `trainer.model.dp_replicate = num_train_nodes / nodes_per_fsdp_group`** — HSDP island sizing from a node count (rl.py:491-499; same for SFT, sft.py:385-394).
7. **LoRA cross-propagation**: trainer LoRA on → orchestrator student LoRA rank/alpha inherited, adapter name auto-generated `r{rank}-a{alpha}`, `inference.enable_lora=true`, `inference.max_lora_rank = rank` *rounded up to vLLM's legal set* `(8,16,32,64,128,256,320,512)` (`configs/rl.py:367-419`; rounding `configs/inference.py:410-426`, set at :158).
8. **Env worker autoscaling — the one workload-derived heuristic**: `num_workers = "auto"` → **1 worker per 256 concurrent rollouts**: train envs `max(1, ceil(max_inflight_rollouts / 256))` (`configs/orchestrator.py:879-883`); eval envs `max(1, ceil(num_examples * group_size / 256))`, 4 if unbounded (:327-334). Doc on the field: "auto scales to 1 worker per 256 concurrent rollouts" (:162-163).
9. **`max_inflight_rollouts` auto** = `max(group_size, int(batch_size * oversampling_factor))` (:858-869) — this is the inference concurrency knob, i.e. the orchestrator-side "batch size" presented to vLLM.
10. **Disaggregated P/D auto-setup**: forces NIXL transfer, EP on, `dp = api_server_count = data_parallel_size_local = gpus_per_node/tp` (`configs/inference.py:382-400`); prefill/decode roles get different all2all backends *in the sbatch template* (prefill `deepep_high_throughput`, decode `deepep_low_latency`, `src/prime_rl/templates/multi_node_rl.sbatch.j2:268,282`); orchestrator P/D metrics roles auto-ordered (rl.py:585-593).
11. **`fused_lm_head_token_chunk_size = "auto"` → 8192** (trainer.py:656-661) — the only "auto" that picks a numeric memory/perf value.
12. **Parser/renderer auto**: tool-call & reasoning parsers resolved from the model name (`configs/inference.py:69-81`); renderer `"auto"` resolves via `MODEL_RENDERER_MAP` and hard-fails at config time when unmappable rather than silently degrading (`configs/orchestrator.py:803-837`).

**Warnings / fail-fast in lieu of fitting checks**: every misfit is a `ValueError` at validation (`--dry-run` runs the full validator network: rl.py:227-228, docs/training.md:82). Examples: GPU budget overrun (`configs/rl.py:144-152` single-node, sft.py:147-151), NCCL needing ≥2 GPUs (rl.py:253-261), `seq_len % (2*cp) != 0` (`parallel_dims.py:262-266,285-291`), CP×attention-kernel compatibility (trainer.py:210-223), DeepEP auto-disabling grad clipping with a warning (trainer.py:567-576), VLM forcing bf16 (trainer.py:578-586). Runtime warnings: orchestrator warns when off-policy cancellation triggers ("Consider increasing it", `dispatcher.py:277-283`) and when <10% of a batch is trainable (orchestrator.py:552-556); packer warns on token-budget timeout (`packer.py:290-292`).

---

## 3. Async orchestration: GPU split, replicas, weight sync, staleness

### 3.1 Trainer:inference GPU split — always user-set

- **Single node**: `deployment.num_train_gpus` / `num_infer_gpus` (defaults **1/1**, `configs/rl.py:135-152`). The `rl` entrypoint slices `CUDA_VISIBLE_DEVICES` in order — *inference first, trainer next, teacher last* (`src/prime_rl/entrypoints/rl.py:87-104,155-163,240-255`; docs/scaling.md:44). No heuristic; the docs just show an example 6-infer/2-train split (`docs/scaling.md:35-43`).
- **Multi-node (SLURM)**: `num_train_nodes` / `num_infer_nodes` / `num_infer_replicas` (`configs/rl.py:155-172`); total inference nodes = `num_infer_nodes * num_infer_replicas`. The generated sbatch (`templates/multi_node_rl.sbatch.j2:4-50`) allocates `num_train_nodes + num_infer_nodes*num_infer_replicas` nodes and exports the wiring (`INFERENCE_TP`, `INFERENCE_DP_LOCAL=$((GPUS_PER_NODE / INFERENCE_TP))`, router/prefill/decode ports). `num_infer_nodes = 0` is allowed for trainer-only benchmarking (requires fake data, rl.py:239-250).
- **Inference replica count** = `num_infer_replicas` (independent vLLM stacks each behind its own `vllm-router`); within a replica, engine count = DP size. None of this is auto-tuned; the docs give qualitative guidance only ("High `dp` … highest throughput; higher `tp` … lower latency, saturates faster" `docs/inference.md:73`).

### 3.2 Weight sync mechanics

Two transports (discriminated union, trainer.py:454-490):

- **`filesystem`** (default): trainer writes full HF-format weights to `<output>/broadcasts/step_N/` + `STABLE` marker each step (`trainer/rl/broadcast/filesystem.py:38-104`), orchestrator's `WeightWatcher` polls (1 s interval) and POSTs `/update_weights` to every inference server (`orchestrator/watcher.py:49-114`). Old broadcast dirs are GC'd (`filesystem.py:111`). LoRA: merged weights written, adapter loaded under a fresh name (`max_cpu_loras=100` exists because of this).
- **`nccl`**: trainer rank-0 joins a `StatelessProcessGroup` of size `1 + inference_world_size` and broadcasts the state dict **layer-by-layer**, dtype-grouped and flattened (`trainer/rl/broadcast/nccl.py:34-110,141-162`), with optional **FP8 kernel-format quantized transfer** (`quantize_in_weight_transfer`, used for GLM-5 FP8 rollouts; gated to `nccl` + custom impl, rl.py:263-277). The final broadcast is skipped when it would have no receiver (train.py:262-275).

### 3.3 Staleness control (their "async_level")

Two mechanisms, both orchestrator-side:

1. **Hard pipeline lead — `TARGET_LAG = 1` (constant, not user-configurable)**: "Maximum batches the orchestrator may run ahead of the trainer" (`orchestrator/orchestrator.py:106-109`). `update_dispatch_gate()` computes `lead = (progress.step + 1) - policy.version` and pauses the dispatcher (an asyncio Event) when `lead > 1`; resumed when the watcher advances the policy (`orchestrator.py:816-833`). This pins the pipeline to exactly one-step-off-policy *at the batch level* — documented as "asynchronous by default … inference is exactly one step behind the trainer" (`docs/algorithms.md:25-35`).
2. **Per-rollout staleness — `max_off_policy_steps = 8` (user knob)**: each weight update bumps `off_policy_steps` on every in-flight rollout; groups exceeding the limit are cancelled mid-flight and re-queued as errors (`orchestrator/dispatcher.py:262-283`, drop logic :568-641). This is the dial for long agentic rollouts that span many policy versions ("bump for throughput, lower for tighter on-policyness", `docs/training.md:61`). Examples set 8 (math, 32k ctx) vs **16** (SWE agentic, 128k ctx).

Supporting machinery: rollout groups pin a single inference client for prefix-cache reuse with a `cache_salt = policy_version` so caches invalidate on weight update (dispatcher.py:392-417); KV-cache-friendly sticky routing via `X-Session-ID: trajectory_id` header + `consistent_hash` router policy (orchestrator.py:729-733, docs/inference.md:177-188); the trainer-side loss compensates for staleness via importance-ratio clipping + mismatch-KL regularization with `mismatch_kl` as the monitored early-warning metric (docs/algorithms.md, docs/training.md:117).

**None of the async split is auto-tuned.** The implicit feedback loop is human: `time/wait_for_batch` high → orchestrator (inference) is the bottleneck, `time/wait_for_ckpt` high → trainer is the bottleneck (`docs/training.md:122-127`).

---

## 4. Sequence packing / variable-length micro-batching (RL trainer)

This is prime-rl's core throughput design and the answer to "how do micro-batches form from variable-length rollouts":

1. **Trainer-side packer, master rank only.** The orchestrator ships raw variable-length `TrainingSample`s (prompt/completion token ids, logprobs, masks). The trainer's rank-0 `Packer` (`src/prime_rl/trainer/rl/packer.py`) converts them into per-DP-rank micro-batch lists and fans them out via the micro-batch transport.
2. **Token-budget step sizing (multi-run mode)**: a step is ready when buffered tokens ≥ `token_budget = seq_len * dp_world_size` (`packer.py:224-228,296`); samples are selected round-robin across runs up to the budget (`packer.py:230-267`). Single-run mode packs the orchestrator's whole batch as one step (`packer.py:92-115`).
3. **Bin packing = First-Fit-Decreasing**: `packed_samples_into_micro_bs` sorts samples by `(lora_idx, -length)` and FFD-packs them into bins of capacity `seq_len` — "minimize potential padding while never truncating" (`src/prime_rl/trainer/batch.py:144-214`). Each bin becomes one micro-batch of shape `[1, packed_len]` (sample count per bin is variable); position_ids restart per sample inside the bin and downstream attention uses `cu_seqlens` derived from position-id resets (`src/prime_rl/utils/sequence.py:6-35`, flash-attn varlen). Per-token temperatures let mixed-temperature samples share a bin (batch.py:152). Multimodal samples are never packed (one sample = one micro-batch, batch.py:163-167).
4. **Padding is minimal and purposeful**: bins are padded only to `pad_to_multiple_of = cp` (DataLoader wiring `trainer/rl/train.py:233-241`, `pad_micro_batch` batch.py:217-259); CP additionally requires `seq_len % (2*cp) == 0` (`parallel_dims.py:262-266`).
5. **DP distribution + FSDP lockstep**: micro-batch count is padded to a multiple of `num_train_workers` with zero-loss dummy batches, multimodal and text bins are grouped so all ranks see the same modality at each accumulation index (FSDP all-gather hang avoidance), then dealt round-robin (`batch.py:270-320`).
6. **Implicit gradient accumulation**: the trainer just iterates `for micro_step, micro_batch in enumerate(micro_batches)` calling `loss.backward()` each time, one optimizer step per orchestrator batch (`trainer/rl/train.py:370-491,555`). Accum depth per step ≈ `total_batch_tokens / (seq_len * dp_world_size)` — emergent, never configured.
7. **Global token-mean loss normalization**: loss is divided by the *global* (all-reduced over dp×cp) unmasked-token count, then FSDP's per-rank gradient averaging is undone by multiplying grads by `fsdp_gradient_divide_factor = dp_replicate*dp_shard*cp` (`train.py:353-361,537-541`; `parallel_dims.py:253-255`) — so unevenly-packed ranks don't skew the gradient.
8. **Orchestrator-side token batching option**: `token_batch_size` makes the *step* itself token-defined — the sink cuts a cohort when cumulative `final_input_tokens + final_output_tokens` crosses the threshold (`orchestrator/train_sink.py:254-267`, readiness check :140-148). Pairs naturally with the token-budget packer; requires explicit `max_inflight_rollouts`.
9. **Sample-size guard**: the packer evicts runs that ship samples exceeding `seq_len` or with inconsistent mask/logprob lengths (`packer.py:149-184`); `prepare_sample` truncates to `seq_len` as last resort (batch.py:80-94).

SFT packing is simpler: stream-pack documents into `seq_len * micro_batch_size` windows via `CatDataset` (concat, default) or `StackDataset` (`trainer/sft/data.py:349-377,609-614`).

---

## 5. FSDP2 usage and exposed knobs

Setup in `setup_fsdp` (`src/prime_rl/trainer/model.py:608-714`), per-transformer-block `fully_shard` (FSDP2):

- **MixedPrecisionPolicy hardcoded**: `param_dtype=torch.bfloat16`, `reduce_dtype` from config (default fp32) (model.py:609). Master weights kept in `optimization_dtype` (default fp32).
- **Offload**: `CPUOffloadPolicy(pin_memory=True)` iff `fsdp_cpu_offload` (model.py:610) — enabling it also turns on a gloo group for distributed ops (`train.py:111-113`). Separately, `optim_cpu_offload` keeps params on GPU and offloads only optimizer state (`trainer/optim.py`).
- **`reshard_after_forward`** is user-exposed (default True) and applied per block and to the root (model.py:615,680); the **lm_head+final-norm group is always `reshard_after_forward=False`** as a last-layer optimization, skipped when embeddings are tied (model.py:654-673).
- **HSDP**: `dp_replicate > 1` builds an `hsdp` mesh `(dp_replicate, dp_shard_cp)` (`parallel_dims.py:143-151,196-203`); multi-node convenience knob `nodes_per_fsdp_group` derives `dp_replicate` (§2.6).
- **EP integration**: MoE expert modules are sharded on a separate `dp_shard_mod_ep` mesh (`ep` borrows from `dp_shard×cp`, constraint `ep % cp == 0 && (dp_shard*cp) % ep == 0`, parallel_dims.py:72-74; mesh build :82-152); with EP, FSDP forward/backward prefetch lists are wired manually because DeepEP d2h syncs break automatic prefetch (model.py:683-714).
- **CP**: ring (ring-flash-attn substitution) or ulysses (all-to-all head-sharding; works with any kernel incl. mamba/linear-attn) (`train.py:188-216`, trainer.py:161-162).
- Activation checkpointing full/selective per `freq`-th layer with fine-grained `targets`; activation *offloading* is a separate stream-overlapped CPU offload bounded by `max_inflight_activations` (`utils/act_offloading.py`, wrapped around forward `train.py:439`).

Doc summary table of the four FSDP knobs + tradeoffs: `docs/scaling.md:82-89`; AC table :116-137 (full AC ≈ −25% throughput; selective small; ac_offloading "−30-40% peak memory for ~3-5% throughput"); the layered "memory-tight recipe" (`docs/scaling.md:168-189`) is their canonical knob ordering: **FSDP+EP shard weights → CP shards activations → AC+offload shrink activations → fused-LM-head chunks the loss → compile reduces fragmentation → optimizer offload moves Adam state to CPU**.

---

## 6. Implicit sizing tables: example configs keyed to hardware

`examples/` is the canonical, maintained set (`docs/configuration.md:170`); `configs/` is internal/CI. The progression *is* their sizing table:

| Example | Model | seq_len | Deployment (H100/H200-class nodes) | Trainer parallelism / memory knobs | Orchestrator | Inference |
|---|---|---|---|---|---|---|
| `reverse_text/rl.toml` | Qwen3-0.6B | 2k | 1 train + 1 infer GPU (default) | none | bs 128, gs 16 | defaults |
| `alphabet_sort/rl.toml` | Qwen3-4B | 2k | 1 GPU each | LoRA r32, full AC | bs 512, gs 8 | defaults |
| `wiki_search/rl.toml` | Qwen3-4B | 4k | **2 train + 6 infer GPUs** | LoRA r8 | bs 512, gs 16, oversampling 2.0, zero-adv filter enforced | dp 6 (implied) |
| `configs/deepscaler/stage1.toml` | R1-Distill-1.5B | 8k | 2 train + 6 infer GPUs | full AC | bs 1024, gs 8 | dp = 6 |
| `examples/multinode/rl.toml` | Qwen3-30B-A3B | 2k | 1 train + 1 infer node | custom impl, fa3, **optim_cpu_offload**, AC freq 1 + offload(5) | bs 512, gs 16 | tp 4, dp 2; NCCL broadcast |
| `qwen30b_math/rl.toml` | Qwen3-30B-A3B | **32k** | 2 train + 2 infer nodes | **ep 8**, fa3, compile, AC freq 1 + offload(5) | bs 512, oversampling 2, off-policy 8 | **tp 8**; NCCL |
| `qwen30b_swe/rl.toml` | Qwen3-30B-A3B | **128k** | 2 train + 2 infer nodes | ep 8 + **cp 2**, fa3, compile, AC+offload | bs 512, oversampling 2, **off-policy 16** | tp 8; NCCL |
| `Intellect-3.1/rl.toml` | INTELLECT-3-Base (GLM-4.5-355B-A32B-based) | 128k | **4 train + 12 infer nodes** | ep 8 + cp 2, fa3, compile, AC+offload, **fused_lm_head 1024**, muon | **bs 2048**, oversampling 2, zero-adv enforced | tp 8; NCCL |
| `minimax_m2.5_swe/rl.toml` | MiniMax-M2.5-bf16 | 128k | 4 train + 12 infer nodes | ep 8 + **cp 4**, fa3, fused_lm_head 1024, compile, AC+offload | bs 2048, oversampling 2, off-policy 16 | tp 8; NCCL |
| `glm5_pd_disag/rl.toml` | GLM-5 (trainer bf16) / **GLM-5-FP8 (rollouts)** | 128k | **16 train + 2×8 infer nodes**, P/D disagg (4 prefill + 4 decode per replica) | ep 8, fused_lm_head **2048**, **optim_cpu_offload**, AC freq 1 + offload(**1**), compile, skip_optimizer ckpt, dist_timeout 7200 | **bs 4096, gs 16, oversampling 3, off-policy 16** | EP on, **gpu_memory_utilization 0.80** ("glm5 layers are too large for 0.85"), deep_gemm, **fp8 quantized NCCL weight transfer** |
| `configs/nemotron_4node/rl.toml` | Nemotron-3-Super-120B-A12B | 2k | 4 train + 4 infer nodes | ep 8, optim_cpu_offload, fused_lm_head 8192, AC+offload, compile | bs 128, gs 8 | tp 8, dp 1 |

Extracted pattern (the implicit auto-config policy):

- **Trainer:infer GPU ratio** rises with workload decode-heaviness: 1:1 (toy) → 1:3 (single-node 4B multi-turn) → 1:1 (30B math/SWE) → **1:3 (355B+ 128k agentic)**; GLM-5 P/D run is 1:1 because the FP8 rollout engine is cheap relative to the bf16 trainer.
- **Inference TP** scales with model size: ≤4B → tp1 (pure DP fill); 30B-A3B → tp4–8; 120B+ → tp8 (= full node), capacity then added as DP/replicas/EP-wide rather than deeper TP.
- **Trainer memory ladder** tracks model×context: dense small → nothing; ~30B MoE → custom impl + EP8 + full AC + activation offload (+compile); +long context → CP 2–4; ≥120B or 128k → + fused LM head (1024–8192 chunk; *smaller* chunk for bigger vocab/ctx pressure) and/or optimizer CPU offload; extreme (GLM-5) → everything, with `max_inflight_activations` dropped to 1.
- **Off-policy tolerance** 8 for single-turn/math, 16 for long agentic; `oversampling_factor` 2–3 on every serious run (over-provision rollouts so batch assembly never starves on stragglers; stragglers are cancelled by the off-policy cap).
- **Prefill-vs-decode shaping** appears only at the extreme end as P/D disaggregation with role-split all2all backends + optional Mooncake shared KV pool; below that, the knobs are prefix-cache stickiness + KV offload.
- **vLLM `gpu_memory_utilization`** is left at 0.9 everywhere except explicitly lowered for huge-layer models (0.80 for GLM-5).

### Benchmark baselines = empirical hardware table

`benchmarks/baselines/` holds ~60 JSON results named `benchmark-{1xa6000|1xh100|4xa6000|8xh100|8xh200|8xb200}-{model}-{rl|sft}-{lora-rank|full}-{ac-mode}-{attn}-{seq_len}-cp{N}-ep{N}.json`, each containing config (`num_gpus`, `lora_rank`, `seq_len`, `ac`, `micro_batches`, `device_name`) and measured `mfu`/`throughput`/`step_time`/`peak_memory{gib,pct}`, with OOM captured in `error_reason`. The harness (`benchmarks/scripts/run_single_benchmark.py`) sweeps {model × GPU type × seq_len 16k/64k × AC Recompute/Offload × LoRA/full}. This is the closest thing in the repo to a machine-readable (GPU, model, ctx) → (throughput, peak-mem) sizing table, and `--bench` (4 steps fake data, prints MFU/peak-mem table; `docs/scaling.md:255-275`, auto-setup trainer.py:597-605) is the prescribed manual fitting loop: "compare parallelism configs before committing a multi-day run."

Also relevant: a per-GPU **peak-BF16-FLOPS lookup table** used for MFU (A100 312T, H100 SXM 989T / PCIe 756T / NVL 835T, B200 2.25P, MI300X/MI325X 1307.4T; fallback A100 with warning) at `src/prime_rl/trainer/perf.py:74-106`, plus an architecture-aware active-params/FLOPs-per-token estimator handling GQA/MLA/MoE/LoRA (`perf.py:108-210`) — reusable building blocks for analytic auto-sizing.

---

## 7. Misc relevant details

- **Doc rules of thumb** (`docs/training.md:322-328`): batch_size ≥ 64 floor, 128–512 ablation range, "production RL often runs at 1024+"; group_size ≥ 8 floor, 16–32 common.
- **Renderer pool** `orchestrator.pool_size` for client-side tokenization serialization on long multi-turn prompts (orchestrator.py:522-525).
- **Router replay** (MoE): vLLM returns routed-expert ids per token; trainer replays them instead of re-routing, cutting trainer↔inference mismatch "by an order of magnitude" (`docs/inference.md:253-269`; trainer flag trainer.py:538, cross-auto-set rl.py:421-436); incompatible with KV offload (rl.py:438-449).
- **Elastic inference pools**: DNS-based discovery (`ClientConfig.elastic`, shared.py:91-99,136-141) — inference capacity can grow/shrink without config changes (this is their answer to "how many inference replicas": make it elastic rather than computed).
- **Transports**: rollouts orchestrator→trainer via filesystem (default) or ZMQ (`hwm=10` backpressure); micro-batches packer→ranks similarly (`src/prime_rl/transport/`).
- **Packer watchdog**: rank-0 packer kills the process after 30 min without progress to trigger external restart (`packer.py:23-24,63-71`).
- **Trainer step metrics** logged every step: throughput tokens/s, MFU, `peak_memory` (GiB), time breakdown incl. `wait_for_batch` (train.py:575-606,627-637) — the observability that substitutes for auto-tuning.
- For multi-tenant serving of many concurrent LoRA runs there is `max_concurrent_runs` + `MultiPacker` round-robin fair token scheduling across runs with per-run eviction (packer.py:118-335).

## 8. Takeaways for an auto-config design (AgileRL context)

1. prime-rl shows that an async stack can ship with **zero memory-model**: they replace it with (a) hard validator arithmetic for everything topological, (b) a cheap standardized `--bench` probe + persisted baselines per (GPU, model, seq_len, AC, LoRA) cell, and (c) a curated example ladder. An auto-sizer could literally consume their baselines JSON schema.
2. The strongest transferable mechanism is **token-budget packing replacing micro_batch_size/grad-accum entirely** (budget = `seq_len × dp_world`, FFD bins, global-token-count loss normalization with FSDP divide-factor correction). This makes trainer memory ~independent of rollout length distribution — the #1 thing that makes per-workload micro-batch auto-sizing unnecessary.
3. Staleness needs only two knobs: a hard pipeline lead (their fixed `TARGET_LAG=1`) and a per-rollout `max_off_policy_steps` (8/16 by workload). Both gate the *dispatcher*, not the trainer.
4. The GPU split (trainer:inference, TP depth, replica count) is the one thing they leave fully manual — their example ladder (§6 pattern) is the de-facto heuristic an auto-config tool would encode: ratio by decode-heaviness, TP by model size capped at one node, then DP/replicas, then EP-wide/P-D at the extreme.
5. Knob *ordering* for memory fitting is explicitly documented (memory-tight recipe) and is a ready-made search order for an OOM-driven descent: EP/FSDP → CP → AC → AC-offload → LM-head chunk → compile → optimizer offload → fsdp_cpu_offload.

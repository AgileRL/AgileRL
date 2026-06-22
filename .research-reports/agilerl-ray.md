# agilerl-integration (agilerl_ray): the Ray orchestration layer for ASYNC/DECOUPLED LLM RL

Research slice for the auto-config effort. Repo: `/Users/michaeldoherty/git/agilerl-integration` (read-only).

## 0. Which code is canonical (branch topology matters)

The async/decoupled architecture is **not on `dev`** yet. Branch state as of 2026-06-11:

| Checkout | Branch | Contents |
|---|---|---|
| `/Users/michaeldoherty/git/agilerl-integration` | `parallel-tests` (≈ `origin/dev` + CI test fixes) | NO async rollout. Only the legacy `LLMInferenceEngine` (offline eval) exists in `agilerl_ray/inference/`. |
| `/Users/michaeldoherty/git/agilerl-integration/.claude/worktrees/refactor-async-rl` | `feat/lora-only-weight-sync` | NCCL LoRA-sync work (merged forward). |
| **`/Users/michaeldoherty/git/agilerl-integration/.claude/worktrees/async-rollout-bnb-quant`** | **`claude/async-rollout-bnb-quant`** | **The bleeding edge: full async rollout (off-policy + on-policy modes), NCCL LoRA weight sync, bnb-4bit quant configs, configurable vLLM knobs. "Async rollout working and training with CISPO" validated on cluster.** |

All file:line references below are into the **async-rollout-bnb-quant worktree** (abbreviated `WT/`) unless noted. Lineage: `feature/async_rl` → `merge/async-rl-arch-change` → `feat/lora-only-weight-sync` (NCCL) → `claude/async-rollout-bnb-quant` (quant + configurable vLLM + on-policy mode). Auto-config design should target this branch's shape.

## 1. End-to-end architecture

### 1.1 Entry point and spec pipeline

```
ray job submit (ray_jobs/scripts/remote/submit-multi-turn-llm.sh)
  → python -m agilerl_ray.train --run-id N --manifest m.yaml --llm-model
        --resource-spec '{"computeResource":{"numCpus":23,"numGpus":2,"memoryBytes":"155"}}'
        --num-nodes 2
  → agilerl_ray/run.py:run() (CLI: WT/agilerl_ray/run.py:18-103)
  → EvoHPOTrainer (WT/agilerl_ray/training/trainer.py)
      → ConfigBuilder (WT/agilerl_ray/specs/builder.py)
          ├ AlgorithmSpecFactory → LLMAlgorithmSpec subclass (GRPO/GSPO/CISPO/DPO/SFT/LLMPPO/LLMREINFORCE)
          ├ EnvSpecFactory → LLMEnvSpec (env_type: reasoning|preference|sft|multiturn, WT/agilerl_ray/_typing.py:126-132)
          ├ ResourceSpecFactory → LLMResourceSpec.build (THE GPU split: WT/agilerl_ray/specs/resource.py:392-533)
          ├ TrainingSpec.build (async knobs: WT/agilerl_ray/specs/training.py:243-340)
          └ ReplayBufferSpecFactory (kind: rl|llm → LLMReplayBufferSpec, training.py:216-240)
      → DistributedPopulation (WT/agilerl_ray/agents/population.py) — one LLMAgentManager per pop member
```

Manifests are explicitly "Arena-like" (`WT/agilerl_ray/specs/builder.py:28`); the platform frontend hands the job a manifest plus the `--resource-spec` computeResource JSON.

### 1.2 The actor tree (async mode, per population member)

```
EvoHPOTrainer (driver, in job entrypoint)
└── DistributedPopulation (plain python, holds Ray handles)
    └── LLMAgentManager  [@ray.remote, llm_agent.py:2084]  — one per agent_idx
        │   creates ONE placement group = training_bundles + rollout_bundles (llm_agent.py:2237-2268)
        ├── DistributedLLMAgent ×N (trainer workers; @ray.remote via .options(), llm_agent.py:2292-2331)
        │     num_gpus=GPUS_PER_WORKER=1 (hardcoded, llm_agent.py:155-157), one per training bundle,
        │     torch.distributed/Accelerate+DeepSpeed group across them (master picked via socket bind, :2333-2344)
        ├── LLMReplayBuffer  [@ray.remote, llm_replay_buffer.py:682] — bundle_index=0, num_gpus=0
        │     ├── AsyncLLMRolloutBuffer (in-proc FIFO deques, one per learner)
        │     └── LLMExperienceReceiver (Pulsar consumer, mem_utils.py:123)
        └── LLMRolloutEngine  [@ray.remote, llm_rollout_engine.py:277] — one per rollout bundle
              num_cpus=0.01, num_gpus=0 (rollout_factory.py:32-33) — the GPU is consumed by its child:
              ├── AsyncEngine (vLLM AsyncLLMEngine subclass in its OWN Ray actor,
              │     llm_rollout_engine.py:219-273, scheduled onto the same PG bundle :891-901)
              └── AsyncRayMultiturnEnvWrapper ×(per agent) — Ray-actor vec env workers
                    (created in start(), :694-711; worker count auto-sized, see §4.3)
```

Key structural facts:

- **1 manager = 1 rollout engine = 1 agent** today. `engine_id` is set to `agent_idx` (llm_rollout_engine.py:312-322); `TrainingSpec` validator enforces `rollout_engines_per_agent == 1` (training.py:307-313) and `LLMResourceSpec.build` raises for any other value (resource.py:449-451). Round-robin multi-engine sharding code exists (`_agent_indices_for_engine`, llm_rollout_engine.py:203-216) but is currently bypassed.
- NCCL backend additionally requires exactly one engine per manager (llm_agent.py:2184-2190).
- The vLLM engine actor itself requests `num_cpus=0, num_gpus=0` and relies on `VLLM_RAY_BUNDLE_INDICES` + PG bundle scheduling for GPU affinity (llm_rollout_engine.py:230-233, 891-901), with `RAY_EXPERIMENTAL_NOSET_CUDA_ENV_VAR=1` set cluster-wide (multi-turn-llm-runtime-env.yaml).

### 1.3 Work/data flow

```
LLMRolloutEngine.run_loop (async, per batch):
  for each owned agent: spawn rollout_batch_size*group_size episode tasks
    each: vec_env.reset → loop { vLLM generate_async (LoRARequest per agent) → vec_env.step } until done/max_turns/prompt-budget
  as each GRPO group (group_size episodes) completes → pickle envelope → Pulsar producer.send_async
       topic: {AGILERL_ROLLOUT_TOPIC_ROUTE}/run_{run_id}/agent_{agent_idx}  (ZSTD, batching)
Pulsar broker
LLMExperienceReceiver (in LLMReplayBuffer actor) → validate → AsyncLLMRolloutBuffer.append_group
       (staleness fences: run_id, group_size, rollout_version lag, cycle_id)
DistributedLLMAgent (per-GPU learner) → memory.pop_batch_for_learner(learner_idx=rank, batch_num=data_batch_size_per_gpu)
  → _prepare_and_learn_multiturn → core_algo.learn()
  → every weight_sync_interval learns (off-policy) or every learn (on-policy): sync_weights() → buffer fence
```

Transport is **Pulsar pub/sub with pickled tensor envelopes** (llm_rollout_engine.py:1688-1723; envelope schema `RolloutGroupEnvelope`, llm_replay_buffer.py:47-96). The Ray object store is not used for experience; Ray is used for control-plane RPC + NCCL handles only. Envelope carries: run_id, engine_id, agent_idx, rollout_version, event_ts, correlation id, group_index, cycle_id, batch sizes, episode ids, completion ids/action masks/turn ids/rewards tensors, optional per-token vLLM sampling logps.

### 1.4 Trainer loops

`WT/agilerl_ray/training/train_llm_multi_turn.py`:

- `llm_multiturn_trainer_sync` (:470) — colocated/sync path (trainer drives `collect_rollouts_for_env` itself; HF generate or trainer-owned vLLM).
- `llm_multiturn_trainer_async` (:674) — async path. Two modes via `training.async_rollout_mode`:
  - **off_policy** (default): engines free-run (`start_rollout_engine` → `run_loop.remote()` unbounded, llm_agent.py:2560-2573); trainer pops from buffer; warmup gate `_wait_for_min_buffer_occupancy` requires `batch_size * buffer_occupancy_multiplier` groups (:403-445); weight sync every `algorithm.weight_sync_interval` learn steps (:829-841).
  - **on_policy**: per learn step the main process runs one bounded rollout cycle (`run_loop(max_iterations=1, cycle_id=k)`, :795-801), fences the buffer by cycle_id, pops only that cycle's groups, learns, then syncs weights immediately (:926-953). `rollout_batch_size` is auto-corrected to `data_batch_size_per_gpu * num_trainer_workers` (:711-727) and `weight_sync_interval` is overridden to 1 (:730-735).

`EvoHPOTrainer.train` starts/stops engines around the HPO loop: `population.start_rollout()` (off-policy) or `population.prepare_rollout()` (on-policy) at trainer.py:494-502, `stop_rollout()` at :706-708. `DistributedPopulation.mutations` pauses engines (soft quiesce), mutates, pushes new `AgentRuntimeConfig`, resumes (population.py:244-279).

### 1.5 Evaluation/fitness under async

`DistributedLLMAgent._evaluation_loop` (llm_agent.py:483-516): async mode computes fitness from the **newest buffered rollout group at the current rollout_version** (peek, no consumption, no extra GPU work) — falls back to most recent group. No dedicated eval rollouts in async mode.

## 2. Weight sync (the staleness machinery)

Two backends, selected by `training.weight_sync_backend: filesystem | nccl` (training.py:294; dispatcher `use_nccl_weight_sync`, weight_merge.py:34-36).

### 2.1 Filesystem (default)

Trainer writes the actor LoRA adapter to `AGILERL_ADAPTER_ROOT/agent_{idx}/actor` (`save_peft_model`, llm_agent.py:1087; root env var set in runtime env to `/opt/data`). Rollout engine `refresh_adapter` (llm_rollout_engine.py:1012-1101) re-reads from disk via vLLM `add_lora` with `load_inplace=True` after first load; content change detected via `compute_adapter_marker` (hash/stat marker, llm_utils.py:145). Requires shared filesystem between trainer and rollout nodes.

### 2.2 NCCL LoRA-only broadcast (the new path; bnb-quant compatible)

- Init: `LLMAgentManager._init_nccl_for_owned_engines` (llm_agent.py:2405-2529) — readiness pings, rendezvous (trainer node ip + open port), vLLM `WeightTransferConfig(backend="nccl")` on the engine (llm_rollout_engine.py:869-889) + trainer-side `NCCLWeightTransferEngine.trainer_init` (llm_agent.py:1141-1167), world_size=2 (trainer rank 0, vLLM worker rank 1). Retries: `AGILERL_NCCL_INIT_MAX_ATTEMPTS` (default 2), timeout 600s.
- Per sync (`sync_weights`, llm_utils.py:562-720): soft-quiesce engine → drain in-flight batch (unbounded wait, progress log every 60s) → pause vLLM generation + `start_weight_update(is_checkpoint_format=False)` → trainer broadcasts ONLY the actor-LoRA tensor state (`broadcast_weights_via_nccl`, llm_agent.py:1233-1313) → vLLM worker extension `AgileRLLoRAWorkerExtension.update_lora_via_nccl` (lora_worker_ext.py:55+) receives into a dict and hot-swaps the per-agent LoRA slot via `LoRAModel.from_lora_tensors` (no disk, base untouched → bnb-safe) → resume generation → `increment_agent_rollout_version`.
- Payload: KB–MB (e.g. 516 tensors for r=2 × 7 target modules). `packed=False` currently forced — vLLM's packed-buffer path corrupts small-r LoRA tensors (llm_agent.py:1294-1311). Buffers sized 16 MiB × 2 instead of vLLM's 1 GiB × 2 base-broadcast default (llm_agent.py:90-98).
- **Quiesce semantics**: soft — new batches blocked, in-flight episodes finish naturally (preserves up to ~50 episodes of GPU work per sync on long-episode envs like 50-turn Sudoku); drained groups land at the **old** rollout_version and are kept/fenced per buffer policy (llm_rollout_engine.py:406-501).

### 2.3 Staleness bounds (off-policyness controls)

Three independent fences in `AsyncLLMRolloutBuffer.append_group` (llm_replay_buffer.py:202-320):
1. **run_id**: receiver drops envelopes from stale runs (mem_utils.py:189-202) + per-run Pulsar topics make isolation broker-side.
2. **rollout_version lag**: `replay_buffer.max_rollout_version_lag` (manifest; training.py:156) prunes groups older than `latest - lag` versions. `max_rollout_version_lag: 1` + `weight_sync_interval: 1` ≈ near-on-policy (documented in cispo_multiturn_quant_nccl.yaml:32-36, 196-198).
3. **group_size / cycle_id**: HPO mutation of group_size clears the buffer (fence at :243-275); on_policy mode rejects/evicts non-current cycle_ids (:223-242).

Additionally `fence_agent_rollouts(rollout_version=...)` is called by the trainer right after each sync (train_llm_multi_turn.py:837-841, 931-936), dropping anything not matching the new version.

## 3. GPU/resource allocation model

### 3.1 Cluster discovery → LLMResourceSpec (resource.py)

`ResourceSpec.calculate_resources` (resource.py:121-224): cluster totals come either from the **user/platform `--resource-spec` JSON** (`computeResource.{numCpus,numGpus,memoryBytes}` × num_nodes via `get_nodespec_from_user`) or from live `ray.nodes()` enumeration of alive non-head workers (with `WORKERS_PER_JOB` env cap for parallel benchmarking).

`LLMResourceSpec.build` (worktree version, resource.py:392-533) — **the central GPU-split rule for async**:

```
async_rollout:
    assert total_gpus >= 2 * pop_size                 # hard floor (resource.py:453-455)
    training_gpus            = total_gpus - pop_size  # rollout takes exactly 1 GPU per agent
    training_gpus_per_agent  = training_gpus // pop_size
    rollout_gpus_per_agent   = rollout_engines_per_agent (=1)
else (colocated/sync):
    training_gpus_per_agent  = total_gpus // pop_size; rollout gets 0
cpus_per_agent   = total_cpus  / (pop_size + rollout_engines_per_agent*pop_size)   # even CPU split trainer vs rollout
memory_per_agent = total_mem   / (pop_size + rollout_engines_per_agent*pop_size)
training_cpus_per_agent = rollout_cpus_per_agent = cpus_per_agent - RESERVED_CPUS(0.5)
training_memory = rollout_memory = memory_per_agent * WORKER_MEMORY_RATIO(0.9)
actor (manager) overheads: ACTOR_CPUS=0.01, ACTOR_GPUS=0, ACTOR_MEMORY_RATIO=0.05  (resource.py:20-25)
```

So today the rollout:training GPU ratio is **fixed at 1 : (total/pop − 1)** — there is no knob for "2 rollout GPUs per trainer GPU" or fractional sharing; that's the most obvious auto-config gap. `env_resource_ratio` (manifest `ray_params`/`training`, default 0.5; e.g. 0.8 in async manifests) splits the CPU budget between env workers and the agent/engine process.

### 3.2 Placement groups

`LLMAgentManager.__init__` builds **one PG per agent** = `training_bundles + rollout_bundles` (llm_agent.py:2237-2268):
- training bundle i: `{CPU: training_cpus/N, GPU: training_gpus/N, memory: training_mem/N}` ×N trainer engines
- rollout bundle j: `{CPU: rollout_cpus*(R/N), GPU: rollout_gpus/R, memory: rollout_mem*(R/N)}` ×R engines
- PG `ready()` timeout 720s. Buffer pinned to bundle 0 (training.py:190-196). Strategy comes from `ray_params.ray_strategy` (manifests use `SPREAD`).
- vLLM child workers are pinned to the rollout bundle via `VLLM_RAY_BUNDLE_INDICES` (llm_rollout_engine.py:230-233) and a nested Ray runtime env (`build_vllm_nested_ray_runtime_env`, ray_utils.py:49).

### 3.3 Async env worker auto-sizing (existing mini-auto-config)

`_plan_async_multiturn_resources` (ray_utils.py:392-459) + `_resolve_async_env_num_workers` (:372-389): env worker count = `min(logical_num_envs, max(async_min_workers, (cpu_budget*env_resource_ratio − async_cpu_safety_margin)/async_cpu_per_worker))`, where `logical_num_envs = rollout_batch_size * group_size`. Knobs in `environment:` manifest block — `async_cpu_per_worker` (default 1.0; manifests use 0.25), `async_cpu_safety_margin` (1.0), `async_min_workers` (1), `async_require_parent_pg` (True) (env.py:331-341).

## 4. The complete knob inventory at this layer

### 4.1 Topology / mode switches
| Knob | Where | Default | Notes |
|---|---|---|---|
| `training.async_rollout` | training.py:291 | False | The COLOCATED↔DECOUPLED switch at this layer. False ⇒ trainer-driven rollouts (HF generate or trainer-owned vLLM per upstream AgileRL); True ⇒ rollout engines + Pulsar + buffer. |
| `training.async_rollout_mode` | training.py:292 | `off_policy` | `on_policy` = bounded cycles + sync-every-learn. |
| `training.rollout_engines_per_agent` | training.py:293, resource.py:449 | 0 (sync) / must be 1 (async) | Multi-engine support "soon"; sharding code already exists. |
| `training.weight_sync_backend` | training.py:294 | `filesystem` | `nccl` = LoRA-over-NCCL hot-swap. |
| `training.pop_size` | manifest | — | Drives GPU floor: async needs ≥ 2×pop GPUs. HPO population. |
| `training.hpo` | training.py:295 | True | Gates mutations (incl. rollout pause/resume cycle). |

### 4.2 Batching / queueing / staleness
| Knob | Where | Default | Notes |
|---|---|---|---|
| `training.rollout_batch_size` | training.py:289 | 1 | Groups per engine batch; logical env slots = this × group_size. Auto-overridden in on_policy mode. |
| `algorithm.group_size` | algo specs | — | GRPO group; also Pulsar message granularity. HPO-mutable (with buffer-thrash warning, manifest :151-160). |
| `algorithm.batch_size` → `data_batch_size_per_gpu` | algo.py:657-660 | — | Per-rank learn batch = batch_size // num_processes. |
| micro_batch_size_per_gpu | algo.py:661-664 | **hardcoded `min(per_rank,1)`** | "FIXME depends on model size and node size — needs to be dynamic". Prime auto-config target. |
| `algorithm.weight_sync_interval` | algo.py:473 | 2 | Learns between syncs (off-policy mode). |
| `replay_buffer.memory_size` | training.py:32 | env `DEFAULT_BUF_SIZE` or 10000 | Deque slots (groups), per learner shard. |
| `replay_buffer.buffer_occupancy_multiplier` | training.py:155 | 1 | Warmup gate: needs batch_size×mult groups before first learn. |
| `replay_buffer.max_rollout_version_lag` | training.py:156 | None (∞) | THE off-policyness bound. |
| learner routing | llm_replay_buffer.py:190-200 | crc32(episode_id) % num_learners (off-policy) / group_index % learners (on-policy) | Deterministic shard per trainer rank; `require_all_learners_ready=True` on pop. |

### 4.3 vLLM engine sizing (`algorithm.vllm_config`, consumed at llm_rollout_engine.py:773-901)
| Knob | Default | Notes |
|---|---|---|
| `gpu_memory_utilization` | **0.9 injected for async** when omitted (algo.py:65-74, 572-591; env-overridable `AGILERL_VLLM_GPU_MEMORY_UTILIZATION`); agilerl's colocated default is 0.3 | The codified colocated-vs-decoupled memory-split heuristic. |
| `max_num_seqs` | 16 (llm_rollout_engine.py:806-809) | Concurrency vs KV cache. Manifest doc: size num_slots ≈ 2× max_num_seqs so the KV cache stays saturated through the long-tail (cispo_multiturn_quant_nccl.yaml:218-227). |
| `max_num_batched_tokens` | unset → vLLM default (=max_model_len) | Prefill throughput knob (llm_rollout_engine.py:858-867). |
| `max_loras` | max(manifest, agents-on-engine) (:783-789) | One slot per agent. |
| `max_lora_rank` | 16, raised to lora_config.r (:790-795) | |
| `enable_lora` | True (:796-802) | False only for base-only smoke runs. |
| `quantization` / `dtype` / `kv_cache_dtype` / `kv_cache_memory_bytes` | None passthroughs (:841-857) | `quantization: bitsandbytes` pairs with trainer `quantization_config: nf4`. |
| `tensor_parallel_size` | 1 (NCCL path hardcodes 1, :884; factory reads it for CPU calc, rollout_factory.py:53-56) | Multi-GPU rollout engines not wired yet. |
| `enforce_eager` | False (FIXME at :827) | |
| sampling: `temperature`/`top_p`/`max_output_tokens` | spec, runtime-updatable via `AgentRuntimeConfig` (protocols.py:59-71) | Pushed at mutation/HPO boundaries; no engine restart. |

### 4.4 Context-length plumbing
`network.max_context_length` → `algorithm.max_model_len` (algo.py:524-525) → both trainer (HF prompt-budget validation) and rollout vLLM `max_model_len`. `algorithm.max_output_tokens` = per-turn generation cap (must be < max_model_len). Prompt-budget guard terminates episodes that would overflow (`_is_prompt_too_long`, llm_rollout_engine.py:1469-1481); `enable_sliding_window` opts into prompt truncation instead. `environment.max_turns` caps episode length (env.py:331).

### 4.5 Timeouts / env vars (all auto-config-relevant because they scale with workload)
- `AGILERL_ASYNC_POP_BATCH_TIMEOUT_S` (default 3000; cluster runtime-env sets 1000 with a rationale tied to 50-turn episodes × max_output_tokens through one bnb engine — train_llm_multi_turn.py:35, runtime-env yaml).
- `AGILERL_ASYNC_BOUNDED_CYCLE_TIMEOUT_S` (defaults to pop-batch timeout; :36-41).
- `AGILERL_GENERATION_TIMEOUT_S` 1200, `AGILERL_NCCL_SYNC_TIMEOUT_S` 1800, `VLLM_ENGINE_STARTUP_TIMEOUT_S` 900 (llm_rollout_engine.py:97-100); `AGILERL_ROLLOUT_STOP_TIMEOUT_S` 600, `NCCL_INIT_TIMEOUT_S` 600 (llm_agent.py:158-160).
- Required: `AGILERL_ROLLOUT_TOPIC_ROUTE` (Pulsar topic prefix; llm_rollout_engine.py:195-200), `APP_PULSAR_URL`, `AGILERL_ADAPTER_ROOT` (filesystem sync), `RAY_EXPERIMENTAL_NOSET_CUDA_ENV_VAR=1`.
- `AGILERL_ATTN_IMPLEMENTATION` default trainer attention backend when manifest omits (algo.py:539-544).

### 4.6 Trainer-side (consumed here, owned by agilerl lib but set via manifest)
`quantization_config` (nf4/int8/dict), `activation_offload`, `lora_target_scope`, `use_liger_loss`, `liger_token_chunk_size`, `cast_logprobs_to_fp32`, `gradient_checkpointing`, `attn_implementation`, `use_sequence_packing`, `fused_logprobs_chunk_rows`, `vllm_importance_sampling_{correction,apply,cap}` (Feature A — corrects rollout-vs-trainer numeric divergence, "matters more" under bnb-4bit + LoRA-NCCL; algo.py:447-510). DeepSpeed config is a hardcoded ZeRO-2 + CPU optimizer offload template (`DEFAULT_DEEPSPEED_CONFIG`, llm_agent.py:123-153); `gradient_accumulation_steps: 1` fixed; only `gradient_clipping` is wired from the manifest.

### 4.7 ConfigBuilder coercions (existing auto-config precedents)
- `use_vllm` force-coerced to False under async_rollout (builder.py:102-133) — the trainer must not spin an idle colocated vLLM.
- `use_vectorized_env=False` ignored under async (env.py:351-356).
- async gpu_memory_utilization default injection (algo.py:572-591).
- on_policy `rollout_batch_size` and `weight_sync_interval` auto-correction at runtime (train_llm_multi_turn.py:711-735).

## 5. Telemetry already present

- **MetricsHandler** (WT/agilerl_ray/utils/metrics_utils.py:92-374): OpenTelemetry counters/gauges/histograms (OTLP → OpenObserve per docs/openobserve.rst) + Pulsar `metrics_names` topic (Avro) announcing new series, keyed `run_{id}` / `agent_{idx}`. `get_system_resources()` reports CPU/mem/disk + **per-GPU load, used/free memory, utilization %, temperature** via GPUtil (:332-374; known FIXME: always reports as GPU 0 under Ray's device isolation).
- **Engine health endpoint** `LLMRolloutEngine.health()` (llm_rollout_engine.py:1890-1919): published/failed Pulsar counts, active_batches/active_generations, per-agent rollout_version, adapter path, sampling params, pending runtime updates — polled but not yet exported as metrics.
- **Trainer metric series** (train_llm_multi_turn.py:50-68, 644-660, 1022-1035): loss, kl, score, **completion_lengths (mean assistant tokens per episode)**, accuracy, ppo_updates, pg/vf loss, entropy, clipfrac, GRPO signal-health (`n_groups`, `surviving_group_frac`, `mean_group_reward_std`), vLLM IS-correction diagnostics (`vllm_is_ratio_mean/p95`, `frac_clamped`, `rows_skipped`); GPU allocated/reserved GB logged each report.
- Buffer occupancy is computable (`buffer_count_for_learner`) and logged on ingest; eviction causes are logged with per-cause breakdown (group_size change WARNING, version-lag prune INFO; llm_replay_buffer.py:248-318).
- **Missing for auto-config feedback loops**: no tokens/s, no vLLM KV-cache utilization, no queue-latency (event_ts is on envelopes but unused), no prefill/decode split, no NCCL sync duration metric (only print-logged stage markers `[NCCL_SYNC]`/`[NCCL_INIT]`).

## 6. Where auto-config naturally plugs in

1. **Manifest generation time (strongest fit)**: the Arena platform → manifest + `--resource-spec` JSON → `ConfigBuilder`. Everything auto-config wants to set is already a manifest field; an auto-config pass could consume (hardware = `--resource-spec`/`ray.nodes()`, model = `network.*`, workload = `environment.*` + `max_output_tokens`/`max_turns`) and emit the `training`/`vllm_config`/`replay_buffer` blocks. The benchmarking layer already programmatically mutates manifests (`benchmarking/manifest_builder.py` + `benchmarking/utils.py`).
2. **Spec-build time (`LLMResourceSpec.build` / `LLMAlgorithmSpec.from_dict`)**: precedents exist — the async 0.9 gpu_memory_utilization injection, the use_vllm coercion, the on_policy batch auto-correction, env-worker CPU auto-sizing. The trainer:rollout GPU split (resource.py:447-465) is the single highest-value formula to generalize (currently rigid 1-GPU-rollout-per-agent + integer floor-divide trainer GPUs, ≥2×pop assertion).
3. **Actor startup**: `LLMRolloutEngine._build_async_vllm` already resolves max_loras/max_lora_rank from runtime facts; could similarly derive max_num_seqs / max_num_batched_tokens / kv_cache budget from profiled model+VRAM instead of fixed defaults.
4. **Runtime**: `AgentRuntimeConfig` push path (update_agent_runtime → applied at batch boundaries) handles sampling/geometry only; engine-level vLLM settings need restart. A runtime auto-tuner would need new plumbing or engine recycling via the existing stop/start lifecycle.

## 7. Arena / platform-frontend hooks (for surfacing decisions in UI)

- Manifests are "Arena-like" throughout (builder.py:28, specs/__init__.py:118,208); platform submits via Ray Job API with per-job runtime envs (`ray_jobs/runtime-envs/*.yaml`) to clusters like `https://ray-dev-training-XXXX.arena-train.agilerl.rlops.ai`.
- The `--resource-spec '{"computeResource":{numCpus,numGpus,memoryBytes}}'` JSON is the platform's node-shape handshake (run.py:49-54, submit-multi-turn-llm.sh:29) — the natural place for the platform to also pass (or receive back) auto-config output.
- Metrics flow to the platform via Pulsar (`APP_TOPIC_PREFIX` e.g. `persistent://agilerl/ray-dev`, `metrics_names` topic registers series for the UI) and OTel; run_id keys everything, so auto-config decisions could be emitted as gauges/attributes on the same channel for arena UI display.
- A separate Streamlit benchmarking dashboard exists (`benchmarking/dashboard/`) reading S3 results — internal, not the arena UI.
- Datasets/rewards/checkpoints round-trip via S3 buckets under arena user paths (e.g. `s3://agrl-core-arena-dev/.../users/{id}/...`, llm_inference_engine.py:57).

## 8. Gotchas / constraints an auto-configurator must respect

- Hard assertion `total_gpus >= 2*pop_size` for async (resource.py:453-455); `pop_size > total_gpus` rejected (:437-442); 0-training-GPU resolution raises (:505-509).
- GPUS_PER_WORKER (trainer) hardcoded 1 (llm_agent.py:155); vLLM TP hardcoded 1 on NCCL path (llm_rollout_engine.py:884). Whole-GPU granularity everywhere in LLM async; fractional GPU only exists in the non-LLM RL specs (pack rule, main checkout resource.py:263-275).
- group_size mutation under HPO ⇒ buffer flush ⇒ trainer starvation risk — manifests pin min=max as a workaround (cispo_multiturn_quant_nccl.yaml:151-160).
- num_slots (= rollout_batch_size×group_size) vs max_num_seqs oversubscription ratio ~2× is the documented throughput heuristic (manifest :218-227).
- `pop_batch` timeouts must scale with max_turns × max_output_tokens × concurrency (runtime-env comment).
- vLLM packed NCCL broadcast is buggy for small-r LoRA (packed=False both sides or 300s NCCL watchdog deadlock; llm_utils.py:632-650, llm_agent.py:1294-1311).
- vLLM LoRA keep-resident patch (from agilerl lib) is mandatory on the engine or the adapter silently degrades to base (llm_rollout_engine.py:261-269, 903-913).
- Per-engine `DEFAULT_VLLM_GPU_MEMORY_UTILIZATION` env override exists but explicit manifest values always win.
- Trainer micro-batch is clamped to 1 with an explicit FIXME "needs to be dynamic" (algo.py:661-664) — gradient accumulation is implicitly batch_size/1.

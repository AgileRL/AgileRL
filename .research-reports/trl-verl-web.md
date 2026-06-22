# Auto-config research — TRL (local code) + verl & the broader field (web)

Research slice for AgileRL's "auto right-size LLM RL training configuration" effort.

- **Part A** is read from the local TRL checkout at `/Users/michaeldoherty/git/Misc/trl`
  (commit `56cbcda1dc83af81cbc1c45359449934620af207`, 2026-06-01, version `1.6.0.dev0`).
  All `file:line` references are against that commit.
- **Part B** is web research; every claim carries a URL. Web content was retrieved via
  search/fetch summarization on 2026-06-11, so treat paraphrased numbers as
  high-confidence-but-verify; direct quotes are marked.

---

# PART A — TRL GRPOTrainer's vLLM integration

## A.1 Topology modes: `colocate` (default) vs `server`

`GRPOConfig.vllm_mode` defaults to `"colocate"` (`trl/trainer/grpo_config.py:502-511`):

- **`colocate`** — a full vLLM `LLM` engine is constructed *inside every trainer rank's
  process* and shares the training GPUs. There is **one vLLM engine replica per training
  rank** (or per TP group, see A.3). No HTTP, no separate process.
- **`server`** — the trainer is an HTTP client (`VLLMClient`) to a standalone server
  launched with `trl vllm-serve` on **separate GPUs**. The docs are explicit that server
  and trainer **must not share CUDA devices** — TRL raises
  `RuntimeError: Attempting to use the same CUDA device for multiple distinct roles/ranks...`
  if they do (`docs/source/vllm_integration.md:190-195`).

There is additionally an experimental fully-async mode (`trl/experimental/async_grpo/`,
see A.9) that decouples rollout from training against a *stock* `vllm serve` instance.

### Colocate engine construction (`trl/generation/vllm_generation.py:348-365`)

```python
self.llm = LLM(
    model=model.name_or_path,
    tensor_parallel_size=self.tensor_parallel_size,
    gpu_memory_utilization=self.gpu_memory_utilization,
    max_model_len=self.max_model_length,
    max_num_seqs=self.max_num_seqs,
    enable_sleep_mode=self.enable_sleep_mode,
    model_impl=self.model_impl,
    distributed_executor_backend="external_launcher",
    # Feed identical seed for tp groups to ensure sampling results are the same across workers
    seed=accelerator.process_index // self.tensor_parallel_size,
    # Latest vLLM v1 memory profiler is misled by the high default value (i.e., 32768) - thinking there's not enough memory
    max_num_batched_tokens=4096,
    # Important so temperature scaling/logit tweaking affects the TIS log probs
    logprobs_mode="processed_logprobs",
    quantization=quantization,
)
```

Notable auto-set details:

- `distributed_executor_backend="external_launcher"` — vLLM piggybacks on the
  accelerate-launched process group instead of spawning its own workers; TRL sets
  `RANK/LOCAL_RANK/WORLD_SIZE` env vars itself (`vllm_generation.py:332-337`).
- **`max_num_batched_tokens` is hard-pinned to 4096** in colocate mode with the comment
  that vLLM v1's memory profiler is "misled" by the default 32768 when memory is shared
  with a trainer (`vllm_generation.py:361`). This is a knob AgileRL would want to derive
  rather than pin (verl's rule of thumb differs sharply — see B.1).
- `logprobs_mode="processed_logprobs"` so the returned sampling logprobs reflect
  temperature/top-p processing — required for the truncated-importance-sampling (TIS)
  correction (A.8).
- **bnb quantization auto-detection**: if any module is `bnb.nn.Linear4bit`, vLLM is
  initialized with `quantization="bitsandbytes"`; an 8-bit `Linear8bitLt` raises
  `ValueError("vLLM does not support in-flight 8-bit quantization.")`
  (`vllm_generation.py:339-346`).
- After construction, if sleep mode is on, the engine is immediately put to sleep:
  `self.llm.sleep(level=2)` (`vllm_generation.py:366-367`).

### Server mode plumbing

- Client connects to `http://{vllm_server_host}:{vllm_server_port}` (defaults
  `0.0.0.0:8000`, timeout 240 s, weight-sync group port 51216;
  `grpo_config.py:533-561`), only on the **main process**
  (`vllm_generation.py:302-311`).
- Weight sync joins the client as one extra rank into a `StatelessProcessGroup` of size
  `tensor_parallel_size * data_parallel_size + 1` and NCCL-broadcasts each named tensor
  from the client (`trl/generation/vllm_client.py:426-434, 489-511`;
  `trl/scripts/vllm_serve.py:1139`).
- The server CLI exposes the classic vLLM knobs with defaults:
  `--gpu_memory_utilization 0.9`, `--tensor_parallel_size 1`, `--data_parallel_size 1`,
  `--max_model_len None`, `--enable_prefix_caching None`, `--enforce_eager False`,
  `--kv_cache_dtype auto` (`trl/scripts/vllm_serve.py:223-331`). Note the server default
  GPU fraction is **0.9** (dedicated GPUs) vs the trainer-side colocate default **0.3**
  (shared GPUs) — TRL encodes the topology split directly in these two defaults.
- Generation requests are gathered to the main process, deduplicated
  (`all_prompts[::num_generations]`, then `n=num_generations` server-side), sent once,
  and the result is `broadcast_object_list`-scattered back so each rank receives exactly
  its slice (`vllm_generation.py:576-634`). Server-side, prompts are chunked evenly
  across DP workers (`vllm_serve.py:632-649`).
- The vllm-serve docstring warns that from vLLM 0.14.0 **offline DP scaling for dense
  models is unsupported** — "For dense models, keep this at 1"
  (`vllm_serve.py:234-241`, `docs/source/vllm_integration.md:260-261`).

## A.2 `vllm_gpu_memory_utilization` handling

- `GRPOConfig.vllm_gpu_memory_utilization` **defaults to 0.3** and *only applies in
  colocate mode*; in server mode you pass `--gpu_memory_utilization` (default 0.9) to
  `trl vllm-serve` separately (`grpo_config.py:564-571`; trainer stores it at
  `grpo_trainer.py:564` with the comment "only applies to colocation mode").
- The value is forwarded verbatim to vLLM's `LLM(gpu_memory_utilization=...)`
  (`vllm_generation.py:352`). **There is no auto-derivation, no probing, no validation**
  against trainer footprint — the 0.3 default is the entirety of TRL's "memory split"
  policy for colocated mode.
- `vllm_max_model_length` (default `None` → inferred from model config) is the second
  half of the KV budget story; the doc string tells users to set it to "at least the
  maximum prompt length in the dataset plus `max_completion_length`"
  (`grpo_config.py:572-578`). The async-GRPO doc repeats the same heuristic for
  `--max-model-len` ("A good starting point is the prompt length plus
  `max_completion_length`", `docs/source/async_grpo_trainer.md`).
- HF's co-located vLLM blog used `vllm_gpu_memory_utilization=0.5` for the
  Qwen2.5-Math-72B run on 8 GPUs with DeepSpeed ZeRO-3 + sleep mode
  (https://huggingface.co/blog/vllm-colocate).

## A.3 Batch arithmetic: `per_device_train_batch_size`, `num_generations`, `steps_per_generation`, `generation_batch_size`

This is TRL's only real "auto-derivation": closing the over-determined system in
`GRPOConfig.__post_init__` (`grpo_config.py:907-927`):

```python
num_processes = self.world_size
if self.generation_batch_size is None and self.steps_per_generation is None:
    self.steps_per_generation = self.gradient_accumulation_steps
    self.generation_batch_size = self.per_device_train_batch_size * num_processes * self.steps_per_generation
elif self.generation_batch_size is not None and self.steps_per_generation is None:
    if self.generation_batch_size % (self.per_device_train_batch_size * num_processes) != 0:
        raise ValueError(...)
    self.steps_per_generation = self.generation_batch_size // (self.per_device_train_batch_size * num_processes)
elif self.generation_batch_size is None and self.steps_per_generation is not None:
    self.generation_batch_size = self.per_device_train_batch_size * num_processes * self.steps_per_generation
else:
    raise ValueError("'generation_batch_size' and 'steps_per_generation' can not be both configured at the same time")
```

So the invariant is:

```
generation_batch_size == per_device_train_batch_size * world_size * steps_per_generation
steps_per_generation  defaults to gradient_accumulation_steps
```

Validation (all hard errors):

1. `generation_batch_size % (per_device_train_batch_size * world_size) == 0`
   (`grpo_config.py:914-918`).
2. `generation_batch_size % num_generations == 0` — "The generation batch must contain
   full prompt groups (no partials)" (`grpo_config.py:940-946`).
3. `num_generations >= 2` (advantages need a group) (`grpo_config.py:948-952`).
4. Eval: `(per_device_eval_batch_size * world_size) % num_generations_eval == 0`
   (`grpo_config.py:929-938`).
5. Sequence/context parallel is rejected outright: GRPO "builds model inputs after
   generation inside the trainer" so cp/sp sharding can't apply
   (`grpo_config.py:889-897`).

Defaults: `num_generations=8`, `max_completion_length=256`
(`grpo_config.py:392-398, 406-409`).

### How generation and optimization batches relate at runtime

- `get_train_dataloader` loads a **generation batch** of
  `per_device_train_batch_size * steps_per_generation` per rank instead of a per-step
  batch (`grpo_trainer.py:906-913`, the one-line change is `:910`).
- `RepeatSampler` (`grpo_trainer.py:942-949`) is built with
  `mini_repeat_count=num_generations`,
  `batch_size=generation_batch_size // num_generations` (number of *unique prompts* per
  generation round), and `repeat_count=num_iterations * steps_per_generation` so the
  same generations can be re-consumed across optimizer steps. The ASCII diagram at
  `grpo_trainer.py:924-939` documents the slicing.
- `_prepare_inputs` (`grpo_trainer.py:1139-1168`) regenerates only every
  `generate_every = steps_per_generation * num_iterations` steps; otherwise it slices
  `self._buffered_inputs[self._step % steps_per_generation]`. I.e. one big vLLM call
  amortized over `steps_per_generation` gradient-accumulation micro-steps ×
  `num_iterations` GRPO inner epochs.
- For loss-side forward passes (old logps / ref logps over the *whole* generation
  batch), computation is chunked back down to `per_device_train_batch_size`
  (`grpo_trainer.py:1934`, used at `:2050-2058, 2096-2104`) — "Chunk inputs into smaller
  batches to reduce memory peak" (`grpo_trainer.py:1064`).
- **`max_num_seqs` auto-derivation** for the colocate engine:
  `per_device_train_batch_size * vllm_tensor_parallel_size * steps_per_generation`
  (`grpo_trainer.py:784-786`) — i.e. exactly the number of sequences a rank (or TP
  group) will submit per generation round, capping vLLM's scheduler to the real
  concurrency. This is a good template for AgileRL: derive engine concurrency from the
  training batch geometry instead of leaving vLLM's default (usually 1024+).

### Colocate TP sub-groups

`vllm_tensor_parallel_size` (default 1) must divide world size
(`vllm_generation.py:316-320`). For TP>1, ranks are partitioned into contiguous
subgroups via `torch.distributed.new_subgroups_by_enumeration`
(`vllm_generation.py:322-330`); each rank `all_gather_object`s its prompts to the TP
group, every rank generates the full group batch (deterministic because each group
shares a seed, `:359`), then keeps only its own slice
(`vllm_generation.py:661-675, 701-709`). Generation uses `n=1` in colocate mode ("vLLM
on each GPU generates only 1", `:639`) because prompt duplication is done by the sampler.

## A.4 Sleep/wake usage

Controlled by `vllm_enable_sleep_mode` (default `False`, `grpo_config.py:520-526`).
Sequence in colocate mode:

1. **Init**: `llm.sleep(level=2)` right after engine build (`vllm_generation.py:366-367`).
2. **Weight sync** (`sync_weights`, `vllm_generation.py:447-449`): `empty_cache()` then
   `llm.wake_up(tags=["weights"])` — *weights only*, KV stays asleep, so weight-load and
   KV allocation never coexist with peak trainer memory. Comment cites Ascend NPU crash
   issue trl#5142 for why wake-before-load is mandatory.
3. **Generate** (`vllm_generation.py:564-573`): `empty_cache()`,
   `wake_up(tags=["weights"])` (idempotent), then a workaround for vLLM issue #29341 —
   `llm.collective_rpc("reload_weights")` (wrapped in try/except for non-CUDA backends).
   Then `wake_up(tags=["kv_cache"])` just before submitting prompts (`:680-681`).
4. **After generation**: `llm.sleep(level=2)` (`vllm_generation.py:716-717`) — so the
   optimizer step runs with vLLM fully offloaded.

Docs framing: sleep "offload[s] vLLM parameters and cache to CPU RAM during the
optimization step" at the cost of host–device transfer latency on wake
(`docs/source/reducing_memory_usage.md:320-345`). The HF colocate blog states they use
**level 2** for GRPO specifically because "the model is updated after every step" so
discarding weights costs nothing (https://huggingface.co/blog/vllm-colocate).

(NB for AgileRL: this two-tag staged wake — weights first, KV later — plus level-2 sleep
is precisely the pattern your bnb-4bit experiments showed to be broken for in-place
quantized reload; TRL avoids the bnb problem because `sync_weights()` rewrites every
weight via `load_weights` after each optimizer step anyway.)

## A.5 FSDP / DeepSpeed handling around generation & weight sync

`VLLMGeneration.sync_weights()` (`vllm_generation.py:439-526`) is called once per
optimizer step, gated by `self.state.global_step != self._last_loaded_step`
(`grpo_trainer.py:1343-1348`, `:799`), so grad-accum micro-steps never re-sync.

Per-backend strategy:

- **DeepSpeed ZeRO-3, non-PEFT**: per-parameter
  `deepspeed.zero.GatheredParameters([param])` context, then
  `vllm_client.update_named_param` (server) or
  `llm_engine.model_executor.driver_worker.model_runner.model.load_weights([(name, p)])`
  (colocate) (`vllm_generation.py:455-463, 512-520`). Memory-light: one param
  materialized at a time.
- **DeepSpeed ZeRO-3 + PEFT**: must gather **the full model at once**
  (`GatheredParameters(list(model.parameters()))`) because "merging adapters in a
  sharded manner is not supported"; then `model.merge_adapter()` → push merged base
  weights (skipping PEFT-prefixed params, stripping `base_model.model.`/`.base_layer`)
  → `model.unmerge_adapter()` inside the same context
  (`vllm_generation.py:465-501`). This is the peak-memory hot spot for PEFT+ZeRO-3.
- **FSDP1**: memory-efficient **post-order traversal** — recurse children first, then
  `FSDP.summon_full_params(module, recurse=False, writeback=False)` per FSDP module,
  pushing each full param and tracking `visited` to skip re-gathered subtrees
  (`vllm_generation.py:384-411`).
- **FSDP2**: iterate `module.state_dict()`, `param.full_tensor()` on each DTensor (CPU
  params are moved to CUDA first), with PEFT name fixups
  (`vllm_generation.py:413-437`).
- After any sync: `reset_prefix_cache()` on server or colocate engine
  (`vllm_generation.py:522-526`) so stale-cached prefixes from the old policy can't be
  reused.

Non-vLLM generation paths also reveal the gathering knobs:

- `ds3_gather_for_generation` (default `True`, `grpo_config.py:410-418`) — gathers
  ZeRO-3 weights for `model.generate`; disabling allows models larger than one GPU but
  is "not compatible with vLLM generation".
- `unwrap_model_for_generation(..., gather_deepspeed3_params=args.ds3_gather_for_generation)`
  plus `FSDP.summon_full_params(self.model_wrapped, recurse=False)` for FSDP
  (`grpo_trainer.py:1361-1411`; `trl/models/utils.py:196-231`).
- Reference model prep: `prepare_deepspeed` / `prepare_fsdp` / plain
  `accelerator.prepare_model(evaluation_mode=True)` (`grpo_trainer.py:828-834`).

## A.6 What TRL does NOT auto-configure (gaps relevant to AgileRL)

- No derivation of `vllm_gpu_memory_utilization` from model size/VRAM — fixed 0.3.
- No OOM-retry on the training step (TRL does not use accelerate's
  `find_executable_batch_size`; transformers' `auto_find_batch_size` exists in
  `TrainingArguments` but interacts badly with GRPO's generation-batch arithmetic and is
  not mentioned anywhere in the GRPO code/docs).
- No workload-profile awareness (prefill- vs decode-heavy): `max_num_batched_tokens`
  is pinned at 4096, `max_model_len` left to the user.
- No trainer-vs-inference GPU ratio guidance for server mode beyond "separate devices".
- TP choice is manual; the colocate blog's empirical finding: TP sharding *hurts* a
  1.5B model and gives up to 1.73× for 7B (https://huggingface.co/blog/vllm-colocate).

## A.7 The TIS correction (config-relevant side effect of vLLM)

Because vLLM logprobs ≠ trainer logprobs, TRL defaults
`vllm_importance_sampling_correction=True` with `vllm_importance_sampling_mode="sequence_mask"`
and cap `C=3.0` (`grpo_config.py:792-820`); ratios are computed from
`old_per_token_logps - sampling_per_token_logps` and clamped/masked
(`grpo_trainer.py:2044-2091`). Consequence for auto-config: when vLLM is enabled, the
trainer must *always* run an extra full forward pass over the generation batch to get
`old_per_token_logps` (`grpo_trainer.py:2047-2058`) — generation-batch-sized forward
memory must be budgeted even when `num_iterations=1`.

## A.8 Experimental async GRPO (`trl/experimental/async_grpo/`)

Decoupled topology against a stock `vllm serve` (requires
`VLLM_SERVER_DEV_MODE=1`, `--logprobs-mode processed_logprobs`,
`--weight-transfer-config '{"backend":"nccl"}'`; FSDP2 only, no DeepSpeed —
`docs/source/async_grpo_trainer.md`). Knobs:

- `max_staleness` (default 4): discard samples older than N weight updates
  (`async_grpo_config.py:175-176`).
- `weight_sync_steps` (default 1): NCCL weight push frequency (`:186`).
- **Auto-derivation**: `max_inflight_tasks` default `-1` →
  `max_staleness * per_device_train_batch_size * gradient_accumulation_steps * num_processes`
  — "Generating more than this is wasteful since the excess samples will be discarded"
  (`async_grpo_config.py:165-169`; doc).

---

# PART B — verl, the field, and OOM-retry semantics (web)

## B.1 verl performance tuning guide

Source: https://verl.readthedocs.io/en/latest/perf/perf_tuning.html (authors Guangming
Sheng, Jiali Zheng). Section order **is** the recommended tuning order:

1. **Rollout generation tuning**
2. **Enable remove padding (sequence packing)** — `use_remove_padding=True`
3. **Batch size tuning**
4. **Tuning for dynamic batch size** — `use_dynamic_bsz`
5. **Ulysses sequence parallel for long context**
6. **LigerKernel** (`use_liger: True`)
7. **FSDP forward prefetch** (`fsdp_config.forward_prefetch=True`)
8. **Migrate to FSDP2** (`actor_rollout_ref.*.strategy="fsdp2"`)
9. **Entropy-calc memory optimization** (`entropy_from_logits_with_chunking=True`,
   `entropy_checkpointing=True`)

### Rollout knobs (vLLM/SGLang)

- `gpu_memory_utilization`: for vLLM ≥0.7 it is a fraction of **total** GPU memory; for
  SGLang it is the fraction of **free** memory for *static* allocations (weights+KV) and
  the remainder is still used at runtime — semantics differ per backend. Recommended:
  **"A value between 0.5 and 0.7 often strikes a good balance between high throughput
  and avoiding OOM"** (colocated hybrid engine).
- "If the GPU cache utilization is relatively low in the log, increase `max_num_seqs`
  or `max_num_batched_tokens` [to] enlarge the effective batch size in the decoding
  stage"; set `max_num_batched_tokens > 2048` for throughput.
- **"Use a smaller `tensor_parallel_size`. When GPU resources allow, a smaller tensor
  parallel size spawns more vLLM replicas. Data parallelism (DP) can yield higher
  throughput than tensor parallelism (TP)."**
- CUDA graphs: `enforce_eager=False` required to use `cudagraph_capture_sizes`; smaller
  capture sizes reduce OOM risk at slight throughput cost (graph memory cannot be
  offloaded, so it lingers into the training phase).
- `free_cache_engine=True` (default): offload KV cache after the rollout stage
  (https://verl.readthedocs.io/en/latest/examples/config.html).

### Batch-size doctrine (the key conceptual split)

- **Algorithmic, global** knobs (affect convergence): `data.train_batch_size`,
  `actor.ppo_mini_batch_size`.
- **Performance, local (per-GPU)** knobs (throughput only):
  `*micro_batch_size_per_gpu`, and under dynamic batching `*_max_token_len_per_gpu`.
- Rule: "Increase the `*micro_batch_size_per_gpu` as much as possible till equals to
  normalized `mini_batch_size`" — i.e. grow micro-batch until grad accumulation
  disappears or OOM.

### Dynamic batching (token-budget batching)

`use_dynamic_bsz=True` replaces micro-batch counts with **token budgets per GPU**:

- `actor_rollout_ref.actor.ppo_max_token_len_per_gpu`: **at least
  `2 × (max_prompt_length + max_response_length)`** (example config: 16384, annotated
  `n * (max_prompt_length + max_response_length)`).
- Forward-only budgets (`log_prob_max_token_len_per_gpu`,
  `ref.log_prob_max_token_len_per_gpu`, critic `forward_max_token_len_per_gpu`, RM
  equivalents) can be set higher than fwd+bwd budgets; critic/RM ≈ 2× actor's.
- This is verl's answer to padding waste + variable-length rollouts and is the single
  most transferable mechanism for AgileRL: micro-batching by *token count* rather than
  *sequence count* makes the decode-heavy vs prefill-heavy distinction mostly moot on
  the training side.

### Long context & memory

- `ulysses_sequence_parallel_size > 1` for long context; for >32k also decrease
  micro-batch/token budgets.
- `enable_gradient_checkpointing=True`, `enable_activation_offload=True` (FSDP only);
  FSDP `param_offload`/`optimizer_offload` default False
  (https://verl.readthedocs.io/en/latest/examples/config.html).

### verl example-config defaults (config explanation page)

https://verl.readthedocs.io/en/latest/examples/config.html — `max_num_batched_tokens:
8192`, `max_num_seqs: 1024`, `enforce_eager: True` (CUDA graph disabled by default),
`free_cache_engine: True`, `log_prob_micro_batch_size_per_gpu: 16`,
`use_dynamic_bsz: False`, `ulysses_sequence_parallel_size: 1`.

### verl best-practices (DAPO + Qwen3-235B)

https://verl.readthedocs.io/en/latest/perf/best_practices.html — large-model addendum:

- Rollout: push `gpu_memory_utilization` to **0.8–0.9 when offload is enabled**; sizing
  rule **`(memory_per_gpu × gpu_memory_utilization × TP) > 2 × model_parameters`**
  (bf16 weights); `max_num_batched_tokens =
  max(8192, max_prompt_length + max_response_length, max_model_len)`;
  `enable_chunked_prefill` for utilization.
- Training (Megatron): "Prefer increasing TP first, add PP when necessary, extend
  sequence capacity with CP"; enable `param_offload`/`grad_offload`/`optimizer_offload`
  under memory pressure (`optimizer_offload_fraction: 1` for DeepSeek-scale); offload
  ref params whenever actor does.
- Monitor `clip_ratio` (>0.1 ⇒ `max_response_length` too small — a workload-profile
  signal AgileRL could consume automatically).

### verl "auto" settings

verl has **no shipped auto-tuner**. Feature request
https://github.com/volcengine/verl/issues/1845 ("Auto-tune batch size / max_token_len",
open) asks for exactly the AgileRL goal: derive `mini_batch_size`/`max_token_len` from
available GPU memory per Ray actor, "similar to transformers Trainer"
(`auto_find_batch_size`). No maintainer-shipped implementation as of fetch date. verl's
only auto-ish behaviors are dynamic token-budget batching (`use_dynamic_bsz`) and
`free_cache_engine`/offload toggles — thresholds remain manual.

## B.2 HF accelerate `find_executable_batch_size` (OOM-retry decorator)

Read directly from the installed accelerate 1.7.0:
`/Users/michaeldoherty/git/AgileRL/.venv/lib/python3.13/site-packages/accelerate/utils/memory.py:119-182`.
Semantics:

- Decorator wraps a function whose **first arg is `batch_size`**; starts at
  `starting_batch_size` (default 128).
- On exception, `should_reduce_batch_size(e)` (`memory.py:100-116`) string-matches
  RuntimeError messages: `" out of memory."` (CUDA/HIP/XPU), CUDNN_STATUS_NOT_SUPPORTED,
  CPU allocator failure, HPU devmem failure. Anything else re-raises.
- Default policy: **halve** (`batch_size // 2`), `clear_device_cache(garbage_collection=True)`
  between attempts, raise `RuntimeError("No executable batch size found, reached zero.")`
  at 0. A custom `reduce_batch_size_fn` can replace halving.
- Retry is whole-function: the wrapped body restarts from scratch each attempt — so in
  an RL trainer the candidate unit must be idempotent (e.g. a probe step), and partial
  state (optimizer, dataloader position) must be reset by the caller.
- transformers exposes this as `TrainingArguments.auto_find_batch_size`; caveat known in
  the field: in distributed settings every rank must shrink in lockstep or collectives
  desync — one reason verl issue #1845 frames it as a per-actor profiling problem
  instead.

## B.3 OpenRLHF

Sources: https://openrlhf.readthedocs.io/en/latest/performance.html,
https://openrlhf.readthedocs.io/en/latest/hybrid_engine.html,
https://blog.vllm.ai/2025/04/23/openrlhf-vllm.html.

- Topology flags: `--colocate_all_models` (hybrid engine: vLLM + actor + ref + critic +
  RM share GPUs) plus `--vllm_enable_sleep` and `--deepspeed_enable_sleep`; partial
  colocation via `--colocate_critic_reward`, `--colocate_actor_ref`. Recommendation:
  prefer hybrid engine + both sleep flags over distributed placement "when there are
  enough GPU memory".
- `--vllm_gpu_memory_utilization`: docs recommend **0.5 on 8×A100** for hybrid engine;
  OOM-troubleshooting ladder is "lower progressively 0.6 → 0.5 → 0.4".
- Batch heuristics: "Maximize `micro_batch_size` and minimize vLLM TP size" (prefer more
  DP engine replicas — same doctrine as verl); packing samples always on;
  `train.batch_size = rollout.batch_size × n_samples_per_prompt` is the common choice;
  gradient checkpointing first OOM mitigation, then ZeRO stage 2→3, then
  `adam_offload`; ring attention for >8k contexts.
- Ray fine-grained sharing: `VLLM_RAY_PER_WORKER_GPUS` / `VLLM_RAY_BUNDLE_INDICES`
  let multiple components share a GPU bundle (vLLM blog post). No auto-config; "start
  from a known-good recipe ... adjust one knob at a time".

## B.4 NeMo-RL

Sources: https://docs.nvidia.com/nemo/rl/latest/design-docs/generation.html,
https://docs.nvidia.com/nemo/rl/latest/guides/async-grpo.html,
https://github.com/NVIDIA-NeMo/RL/blob/main/examples/configs/grpo_math_1B.yaml.

- Topology is a first-class config: `policy.generation.colocated.enabled: true|false`
  with `colocated.resources.{num_nodes,gpus_per_node}` for the dedicated-inference case.
  Colocated mode uses sleep/wake ("wake up workers during prepare_for_generation");
  non-colocated only resets prefix cache after generation.
- Memory contract: when stages share a GPU, **the sum of their `gpu_memory_utilization`
  values must not exceed 1.0** (per their vllm config docs). Example colocated value:
  `vllm_cfg.gpu_memory_utilization: 0.6` (grpo_math_1B.yaml).
- Example batch geometry (grpo_math_1B.yaml): `train_global_batch_size: 512`,
  `train_micro_batch_size: 4`, `generation_batch_size: 32`,
  `logprob_batch_size: ${policy.train_micro_batch_size}`, `num_prompts_per_step: 32` ×
  `num_generations_per_prompt: 16`; `sequence_packing.enabled: True`
  (modified_first_fit_decreasing) with token budget
  `train_mb_tokens = max_total_sequence_length × train_micro_batch_size`;
  `dynamic_batching.enabled: False` when packing is on.
- Async GRPO: requires `colocated.enabled: false`; staleness knob
  `grpo.async_grpo.max_trajectory_age_steps` ("start with 1 and increase if needed");
  `in_flight_weight_updates: true` with the vLLM async engine; replay buffer auto-sized
  `buffer_size = num_prompts_per_step × max_trajectory_age_steps × 2`. **No recommended
  generation:training GPU ratio is published.**

## B.5 Async RL GPU-split heuristics (AReaL, SkyRL, slime)

### AReaL (decoupled, fully async)

https://arxiv.org/abs/2505.24298 (HTML: https://arxiv.org/html/2505.24298v1):

- "For AReaL, we maintain a fixed ratio between inference and training devices,
  allocating **three-quarters of the devices for inference**." Chosen "against an equal
  50-50 partition based on our early experiments, where the 75-25 partition demonstrated
  higher training throughput."
- Caveat from the authors: "the optimal partition may vary across different settings and
  could potentially benefit from dynamic adjustment during training."
- Staleness bound η=4; interruptible rollout workers + decoupled-PPO objective make the
  staleness tolerable; up to 2.57× throughput vs synchronous baselines (1.5B–32B models,
  H800 cluster, 64×8 GPUs).
- Follow-up AReaL-Hex (https://arxiv.org/abs/2511.00796) optimizes the *placement
  itself* over heterogeneous GPUs — evidence the ratio is workload-dependent and worth
  auto-deriving.

### SkyRL

https://docs.skyrl.ai/docs/configuration/config and
https://docs.skyrl.ai/docs/tutorials/one_step_off_async:

- Placement keys: `trainer.placement.colocate_all` (default true),
  `colocate_policy_ref`, `policy_num_nodes`, `policy_num_gpus_per_node`, etc.
- Hard constraint when colocated: `policy_num_gpus_per_node × policy_num_nodes ==
  num_inference_engines × inference_engine_tensor_parallel_size ×
  inference_engine_data_parallel_size` (validated in
  `utils/utils.py::validate_batch_sizes`-adjacent checks).
- Generator defaults: `gpu_memory_utilization: 0.8`, `max_num_batched_tokens: 8192`,
  `max_num_seqs: 1024`, `enable_prefix_caching: true`, `async_engine: true`.
- Batch defaults: `train_batch_size: 1024`, `policy_mini_batch_size: 256`,
  `micro_train_batch_size_per_gpu: 1`, `micro_forward_batch_size_per_gpu: 1`.
- One-step-off async example splits 8 GPUs **4:4** (trainer:generator) with
  `colocate_all=false`, generation N+1 overlapping training N via an asyncio queue —
  i.e. one-step staleness, the mildest async variant.

### slime (THUDM; GLM-4.5/4.6 post-training)

https://github.com/THUDM/slime,
https://github.com/THUDM/slime/blob/main/docs/en/get_started/usage.md,
https://lmsys.org/blog/2025-07-09-slime/:

- Single-flag topology: `--colocate` "ignores `--rollout-num-gpus` and makes the number
  of GPUs for training and inference equal" (time-sliced Megatron↔SGLang); otherwise
  decoupled with explicit budgets: total = `actor_num_nodes × actor_num_gpus_per_node +
  rollout_num_gpus`; `--rollout-num-gpus-per-engine` ≈ SGLang TP size.
- Colocated memory rule: "although Megatron and SGLang will offload sequentially, they
  still need to leave some memory for each other" — tune by **reducing
  `--sglang-mem-fraction-static`**.
- PPO adds a separate parallel critic pool (`--critic-num-nodes`,
  `--critic-num-gpus-per-node`).

## B.6 Cross-framework synthesis for AgileRL's auto-config

**Colocated GPU-fraction defaults observed** (all for the vLLM/SGLang share):
TRL 0.3 (trainer-heavy, full-model HF training in the remainder) · OpenRLHF 0.5 ·
verl 0.5–0.7 (0.8–0.9 only with training offload enabled) · NeMo-RL 0.6 with an
explicit "fractions must sum ≤1.0" contract · SkyRL 0.8 (but defaults to dedicated
engines). Dedicated/server-mode default is uniformly 0.9.

**Auto-derivations that actually exist in the wild** (candidates to copy):
1. TRL: close the batch system from
   {per_device_bs, world, grad_accum} → {steps_per_generation, generation_batch_size},
   with divisibility-by-`num_generations` validation (A.3).
2. TRL: engine `max_num_seqs` from training batch geometry (A.3).
3. TRL async: `max_inflight_tasks` from staleness × global batch (A.8).
4. NeMo-RL: replay buffer from `num_prompts_per_step × max_trajectory_age_steps × 2` (B.4).
5. verl best-practices inequality for minimum TP:
   `TP ≥ 2 × params_bytes / (mem_per_gpu × gpu_mem_util)` (B.1).
6. verl dynamic token budgets: `ppo_max_token_len_per_gpu ≥ 2×(max_prompt+max_response)`,
   forward-only budgets ~2× that (B.1).
7. accelerate: halving OOM-retry as last-resort safety net, with the distributed-lockstep
   caveat (B.2).
8. AReaL: start async splits at 75:25 inference:training; treat as tunable, not truth (B.5).

**Shared doctrine** across verl/OpenRLHF/TRL-blog: prefer more inference engine
replicas (DP) over deeper TP until the model stops fitting (with the vLLM ≥0.14 caveat
that offline DP for dense models is gone — replicas must be separate engines, which is
what TRL colocate does naturally, one engine per rank).

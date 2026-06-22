# vLLM engine knobs for RL rollout workloads: defaults, auto-sizing, and what to override

**Source repo:** `/Users/michaeldoherty/git/Misc/vllm` at commit `a10d69116cb25c8137eeb3f320add71d4e04fda9` (2026-05-20). This is the V1 engine era — V0 is gone; everything below is the V1 (`vllm/v1/`) code path. All file paths are relative to the repo root; line numbers are from this commit.

> **Caveat for AgileRL:** the vLLM pinned in AgileRL's `pyproject` may be older than this checkout. The big behavioral facts (V1 scheduler, profile-run memory accounting, sleep-mode tags, prefix caching default-on) have been stable since ~v0.8–0.10, but several *defaults* quoted here are newer (e.g. `gpu_memory_utilization=0.92` was 0.90 for a long time; CUDA-graph memory now charged against the KV budget since v0.21.0; sleep "level 0" and `mode=` are new). Verify against the pinned version before hard-coding numbers.

---

## 1. EngineArgs / VllmConfig: defaults and derivations

### 1.1 `gpu_memory_utilization` — default **0.92**, fraction of *TOTAL* GPU memory

`vllm/config/cache.py:66`:

```python
gpu_memory_utilization: float = Field(default=0.92, gt=0, le=1)
```

Semantics (docstring, cache.py:67-73): per-vLLM-instance budget; two instances on one GPU can each use 0.5. Critically, it is a fraction of **total** device memory, not free memory — `vllm/v1/worker/utils.py:403-423`:

```python
requested_memory = math.ceil(init_snapshot.total_memory * cache_config.gpu_memory_utilization)
if init_snapshot.free_memory < requested_memory:
    raise ValueError("Free memory on device ... is less than desired GPU memory utilization ...")
```

So in a **colocated** setup (trainer already holding memory), the constraint at vLLM init is `total * util <= free_at_init`, and the error above is the failure mode if the trainer's footprint pushes free below the request. The right auto-config formula is `util ≈ (free_after_trainer − headroom) / total`.

### 1.2 `kv_cache_memory_bytes` — the precise alternative (bypasses `gpu_memory_utilization`)

`vllm/config/cache.py:158-165`: when set, vLLM **skips memory profiling entirely** and reserves exactly this many bytes for KV cache — `vllm/v1/worker/gpu_worker.py:366-384` ("Note that kv_cache_memory_bytes (when not-None) **ignores gpu_memory_utilization**"). This is the most deterministic knob for a config auto-sizer: compute KV bytes yourself and hand them over. The profile run still executes (for compilation warm-up) but its measurement is unused.

Related: `num_gpu_blocks_override` (`cache.py:85-87`) pins the block count directly; `get_kv_cache_configs` adjusts the planning memory to `override * bytes_per_block` so admission checks stay consistent (`vllm/v1/core/kv_cache_utils.py:2007-2027`).

### 1.3 `max_model_len` — derived from HF config when unset; `-1` = auto-fit to memory

Default is `None` (`vllm/config/model.py:199`). Resolution in `_get_and_verify_max_len` (`model.py:2103-2239`):

- Derived from the HF config's max-position keys (`derived_max_model_len_and_key`, model.py:2114-2116), clamped by `tokenizer_config.model_max_length` (2129-2133), multiplied by RoPE scaling factor for non-su/longrope/llama3 rope types (2164-2188; Gemma3 explicitly skipped because its 128K is pre-scaled, 2162-2164), and for yarn it resets to `original_max_position_embeddings`.
- If nothing found: warns and falls back to **2048** (model.py:2147).
- `max_model_len = -1` means **auto-fit**: after profiling, `_auto_fit_max_model_len` (`vllm/v1/core/kv_cache_utils.py:1839-1900`) binary-searches the largest length whose worst-case single-request KV fits in available memory across all workers, and rewrites `model_config.max_model_len`. Triggered at `kv_cache_utils.py:2029-2032` when `original_max_model_len == -1`.
- User value > derived raises unless `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` (model.py:2233-2238).

**RL relevance:** `max_model_len` should be set to `max_prompt_len + max_completion_len` (or the multi-turn total). It is the single biggest lever on the "fits at all" check (§2.4) and on `estimate_max_model_len`. Auto-fit (`-1`) is a viable "don't think about it" mode but lets vLLM silently shrink context below what the env needs — an RL auto-configurer should instead compute the required length and validate.

### 1.4 `max_num_seqs` and `max_num_batched_tokens` — V1 defaults are usage-context + hardware dependent

The dataclass defaults (`vllm/config/scheduler.py:42-44`) are only test conveniences:

```python
DEFAULT_MAX_NUM_BATCHED_TOKENS: ClassVar[int] = 2048
DEFAULT_MAX_NUM_BATCHED_TOKENS_FOR_BATCHED_DP: ClassVar[int] = 256
DEFAULT_MAX_NUM_SEQS: ClassVar[int] = 128
```

Real defaults come from `EngineArgs.get_batch_defaults` (`vllm/engine/arg_utils.py:2263-2345`), keyed on usage context and device:

| Hardware | Context | `max_num_batched_tokens` | `max_num_seqs` |
|---|---|---|---|
| ≥70 GiB GPU, **not** A100 (H100/H200/MI300X) | `LLM` class (offline) | **16384** | **1024** |
| ≥70 GiB, not A100 | OpenAI API server | **8192** | **1024** |
| Everything else (**A100 included**, by explicit name check — arg_utils.py:2290-2293, "Setting large max_num_batched_tokens for A100 reduces throughput, see PR #17885") | `LLM` class | **8192** | **256** |
| Everything else | API server | **2048** | **256** |
| CPU | `LLM`/API | 4096·ws / 2048·ws | 256·ws / 128·ws |

Post-processing in `_set_default_max_num_seqs_and_batched_tokens_args` (`arg_utils.py:2424-2494`):
- `performance_mode == "throughput"` **doubles both** (when not user-set) (2456-2461). `performance_mode` is a top-level knob, default `"balanced"` (`vllm/config/vllm.py:86,358`); `"interactivity"` instead changes cudagraph capture granularity (§1.7).
- If chunked prefill is *disabled*, `max_num_batched_tokens = max(max_model_len, default)` (2467-2472), because without chunking a prompt must fit in one step (`config/scheduler.py:259-271` raises otherwise).
- Clamp `max_num_batched_tokens = min(max_num_seqs * max_model_len, …)` (2477-2480) and `max_num_seqs = min(max_num_seqs, max_num_batched_tokens)` (2488-2490).
- Validation: `max_num_batched_tokens >= max_num_seqs` required (`config/scheduler.py:273-278`).

**RL relevance:** GRPO-style rollouts via the `LLM` class on an A100 get 8192/256; on an H100, 16384/1024. These defaults are tuned for serving throughput, not for a colocated engine sharing VRAM with a trainer — `max_num_batched_tokens` directly sizes the profile-run activation peak (§2.2) and the LoRA static buffers (§5), so on a 40 GB card it is often worth *lowering* it to free KV/trainer memory.

### 1.5 `enable_chunked_prefill` — default **True** (generative models)

EngineArgs default `None` (`arg_utils.py:594`); resolved in `_set_default_chunked_prefill_and_prefix_caching_args` (`arg_utils.py:2347-2415`) from `model_config.is_chunked_prefill_supported` (`config/model.py:1759-1802`) — **True for all generative decoder models**, False for encoder-decoder and some poolers. Disabling it manually on a generative model logs "may cause the engine to crash or produce incorrect outputs" (arg_utils.py:2360-2369). Treat chunked prefill as always-on in V1.

### 1.6 `enable_prefix_caching` — default **True** (generative models)

Same resolution path: `model_config.is_prefix_caching_supported` (`config/model.py:1805+`) → True for generative decoders. `CacheConfig.enable_prefix_caching: bool = True` (`cache.py:91`). Hash algo default `"sha256"`; `"xxhash"` available for speed (`cache.py:93-108`).

Prefix-cache keys include **LoRA adapter name** (`vllm/v1/core/kv_cache_utils.py:462-475`, `_gen_lora_extra_hash_keys` returns `request.lora_request.lora_name`) — so in an async/decoupled RL setup where each weight-sync bumps the adapter, stale-adapter KV is never silently reused *if the LoRA name changes*; if the adapter is updated **in place under the same name**, cached prefixes computed with old weights WILL be reused — you must call `reset_prefix_cache()` after each weight update (`vllm/entrypoints/llm.py:987-992`). Sleep/wake also clears KV (§4), which handles this for the colocated path.

### 1.7 CUDA graphs + compilation

- Top-level `optimization_level` default **O2** (`vllm/config/vllm.py:352`), which maps to `cudagraph_mode = FULL_AND_PIECEWISE` + inductor compilation (`vllm.py:229-249`, `OPTIMIZATION_LEVEL_TO_CONFIG` at 272-277). O0 = no compile/no graphs, O1 = PIECEWISE.
- `enforce_eager: bool = False` (`config/model.py:226`). Setting it forces `CompilationMode.NONE` + `CUDAGraphMode.NONE` (`vllm.py:1020-1026`) and zeroes capture sizes (`vllm.py:1221-1226`). It is the "fast startup / low extra memory / slower decode" switch.
- **Capture sizes** (`VllmConfig._set_cudagraph_sizes`, `vllm.py:1585-1755`):
  ```python
  max_cudagraph_capture_size = min(max_num_seqs * decode_query_len * 2, 512)   # vllm.py:1638-1647
  # decode_query_len = 1 + num_speculative_tokens
  sizes = [1, 2, 4] + range(8, 256, 8) + range(256, max+1, 16)                 # vllm.py:1676-1688
  ```
  capped by `max_num_batched_tokens` (1648-1649). `performance_mode="interactivity"` instead captures every size 1..min(max,32) (1670-1674). User `cudagraph_capture_sizes` / `max_cudagraph_capture_size` override (compilation.py:631, 675-686).
- **Memory accounting:** since v0.21.0 the profile run *estimates CUDA-graph memory and subtracts it from the KV budget* by default (`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1`, `vllm/envs.py:272`; applied at `gpu_worker.py:398-447`). The worker logs equivalent-utilization suggestions (`gpu_worker.py:467-504`). So cudagraph cost is no longer a hidden OOM source — but it *does* shrink KV. Fewer capture sizes (or eager) = more KV blocks.

### 1.8 `swap_space` — **gone**; `cpu_offload_gb`; preemption is recompute

- `swap_space` is deprecated and **ignored** (`vllm/entrypoints/llm.py:242-248`). V1 has no CPU swap of preempted sequences.
- V1 **preemption = recompute**: when `allocate_slots` fails, the lowest-priority running request is preempted and its `num_computed_tokens` reset to 0 (`vllm/v1/core/sched/scheduler.py:929-941`); it re-prefills from scratch on resume (prefix cache can rescue most of that if enabled).
- `cpu_offload_gb: float = 0` (`vllm/config/offload.py:23-32`) — UVA zero-copy *weights* offload ("virtually increases GPU memory"; weights streamed from pinned CPU each forward). Heavy decode-throughput cost; not recommended for RL rollouts, but it's the lever of last resort to fit a model.
- New: KV offload to CPU via `kv_offloading_size` GiB + `kv_offloading_backend` ("native"/"lmcache") (`cache.py:167-176`) — extends effective KV space at PCIe cost; potentially interesting for very long multi-turn envs.

### 1.9 `kv_cache_dtype` — default `"auto"` (model dtype); fp8 halves KV

`vllm/config/cache.py:18-34,74-81`: options include `fp8` (=`fp8_e4m3`), `fp8_e5m2`, per-token-head int8/fp8, `nvfp4`, turboquant variants. `"auto"` uses model dtype (bf16 → 2 bytes/elem). `fp8` → 1 byte/elem → **2× KV capacity ≈ 2× max concurrency** for decode-heavy workloads; scales are loaded from checkpoint or default 1.0 (`calculate_kv_scales` deprecated, cache.py:109-113). Per-token-head modes additionally budget 2×4 bytes/token/head of scales out of the same allocation (`vllm/v1/kv_cache_interface.py:152-164`).

### 1.10 `block_size` — default **16**, backend may override

`CacheConfig.DEFAULT_BLOCK_SIZE = 16` (`cache.py:45`); applied only if the user didn't pass `--block-size` (`cache.py:223-236` sets `user_specified_block_size`). The platform then asks the chosen attention backend for a preferred size (`vllm/platforms/interface.py:484-500` → `get_preferred_block_size`); e.g. some MLA-sparse backends force 256 (`vllm/v1/attention/backends/mla/sparse_swa.py:106-107`), ROCm AITER unified forces 64. Block size matters mainly for prefix-cache hit granularity (smaller = finer reuse) and per-block page bytes; leave it to vLLM unless profiling says otherwise.

### 1.11 Scheduler policy & async scheduling

- `policy: "fcfs" | "priority"`, default **fcfs** (`config/scheduler.py:109-115`). RL rollout batches are homogeneous; fcfs is fine. `priority` could front-run eval/bootstrap requests if you ever mix.
- `async_scheduling` default `None` → **enabled** unless pooling model / incompatible spec-decode / unsupported executor (`vllm/config/vllm.py:899-963`). Overlaps CPU scheduling with GPU execution; keep on.
- `scheduler_reserve_full_isl: bool = True` (`config/scheduler.py:140-144`) — admission requires the **full input length** to fit in KV, not just the first chunk; prevents chunked-prefill thrashing with long prompts. Good default for prefill-heavy RL.
- `stream_interval` (`scheduler.py:151-155`) — batching of streamed tokens; irrelevant for offline `LLM.generate`-style rollouts.

---

## 2. The profile run: how `gpu_memory_utilization` becomes KV blocks

### 2.1 Sequence

`Worker.init_device` (`vllm/v1/worker/gpu_worker.py:239-309`): init NCCL **first** (so its buffers are in the baseline), snapshot memory, compute `requested_memory = ceil(total * gpu_memory_utilization)` and fail if `free < requested` (utils.py:403-423). Then `load_model` (weights, under the `"weights"` cumem tag when sleep mode is on, gpu_worker.py:338-345), then `determine_available_memory`.

### 2.2 `determine_available_memory` (`gpu_worker.py:353-506`)

```python
with memory_profiling(self.init_snapshot, weights_memory=...) as profile_result:
    self.model_runner.profile_run()                       # dummy forward
    profile_torch_peak = torch.accelerator.memory_stats(...)["allocated_bytes.all.peak"]
    cudagraph_memory_estimate = self.model_runner.profile_cudagraph_memory()  # if graphs enabled

profile_result.non_kv_cache_memory = (
    non_torch_increase + torch_peak_increase + weights_memory)

self.available_kv_cache_memory_bytes = (
    self.requested_memory
    - profile_result.non_kv_cache_memory
    - cudagraph_memory_estimate_applied)                  # gpu_worker.py:443-447
```

The **dummy batch**: `profile_run` (`vllm/v1/worker/gpu_model_runner.py:6073-6146`) runs `_dummy_run(self.max_num_tokens, is_profile=True)` where `max_num_tokens = scheduler_config.max_num_batched_tokens` (`gpu_model_runner.py:464`), with tokens spread **evenly across `min(num_tokens, max_num_seqs)` requests** (`gpu_model_runner.py:5599-5603`), plus a dummy sampler run, plus (for multimodal) a max-size encoder batch. So the activation peak reserved forever scales with `max_num_batched_tokens` — another reason that knob trades against KV space.

Memory taxonomy (excellent docstring at `vllm/utils/mem_utils.py:191-246`): (1) other processes' memory, (2) torch memory of this instance (weights + activation peak), (3) non-torch memory of this instance (NCCL, attention backend workspaces, CUDA context). `available_kv = requested − weights − activation_peak − non_torch − cudagraph_estimate`.

### 2.3 Blocks from bytes

`get_num_blocks` (`vllm/v1/core/kv_cache_utils.py:935-952`):

```python
num_blocks = int(available_memory // page_size // num_layers)
num_blocks = max(num_blocks, 0)
```

(uniform single-group case at kv_cache_utils.py:1268-1271 divides by `page_size_bytes` of the merged spec; hybrid case divides by per-group page size × group layer count, §3.3).

### 2.4 Why too-low `gpu_memory_utilization` → 0 blocks / errors

`available_kv` can go **negative** because `requested_memory` (util × total) can be smaller than the fixed costs (weights + activations + non-torch + graphs). Then `_check_enough_kv_cache_memory` (`kv_cache_utils.py:691-728`) raises:

```python
if available_memory <= 0:
    raise ValueError("No available memory for the cache blocks. Try increasing `gpu_memory_utilization` ...")
```

and even with positive memory, a second check requires room for **one full-`max_model_len` request**:

```python
needed_memory = max_memory_usage_bytes(...)   # one request at max_model_len, all layers
if needed_memory > available_memory:
    raise ValueError("To serve at least one request with the model's max seq len (...) ...")
```

with `estimate_max_model_len` (binary search, kv_cache_utils.py:740-791) included in the error message. Plus the early init check: `free < total*util` → ValueError at startup (utils.py:412-421). These three are the canonical failure modes a config auto-sizer must avoid: (a) util too high for colocated free memory, (b) util too low to cover fixed costs, (c) KV remainder too small for one max-length sequence.

After sizing, vLLM logs `"GPU KV cache size: N tokens"` and `"Maximum concurrency for L tokens per request: X.XXx"` (`kv_cache_utils.py:1733-1736`, computed by `get_max_concurrency_for_kv_cache_config` at 877-895) — `max_concurrency = num_blocks / blocks_per_max_len_request`. For RL you want this ≥ the number of concurrent rollout sequences (e.g. `num_envs × group_size`), or accept queueing.

---

## 3. KV cache math

### 3.1 Bytes per token per layer

`AttentionSpec.real_page_size_bytes` (`vllm/v1/kv_cache_interface.py:166-184`):

```python
2 * block_size * num_kv_heads * head_size * get_dtype_size(dtype)   # 2 = K + V
```

⇒ **bytes/token/layer = 2 × num_kv_heads × head_size × dtype_size**. Total = × num_layers × max_model_len per max-length request (`FullAttentionSpec.max_memory_usage_bytes`, kv_cache_interface.py:210-218: `cdiv(max_model_len, block_size) * page_size_bytes`, divided by dcp×pcp context-parallel size when used).

- **GQA**: `num_kv_heads` is the (small) KV head count, and it's the *per-worker* value — TP divides KV heads across ranks, so per-GPU KV cost scales 1/TP. Example: Qwen2.5-7B (28 layers, 4 KV heads, head 128, bf16) = 2·4·128·2 = 2 KiB/token/layer = 57344 B/token ≈ **18.3 GiB at 32K context for one sequence**... (×: 28 layers · 2048 B · 32768 tok = 1.75 GiB). Per 1000 tokens: 1.75 MiB × num_layers-equivalent — concretely 57344 B/token ≈ 56 KiB/token total.
- **dtype_size**: bf16/fp16 = 2, fp8 = 1 (via `get_dtype_size`), nvfp4 packs fp4 + per-16 fp8 scales (`nvfp4_kv_cache_full_dim`, kv_cache_interface.py:168-177).
- **Per-token-head quant** adds `2 * block_size * num_kv_heads * 4` bytes of scales per page (kv_cache_interface.py:157-160).

### 3.2 Sliding window (Gemma) — pool sized like full attention, but per-request usage is bounded

`SlidingWindowSpec` (`kv_cache_interface.py:434-494`): page bytes = `block_size * num_kv_heads * (head_size + head_size_v) * dtype_size` (identical to 2× when K/V head dims match). The per-request **admission bound** is what differs:

```python
num_tokens = min(sliding_window - 1 + max_num_batched_tokens, max_model_len)
blocks = cdiv(num_tokens, block_size) + 1                  # kv_cache_interface.py:463-483
```

so a Gemma-style SWA layer holds only ~window+chunk tokens per request regardless of context length; blocks behind the window are freed during decode. `ChunkedLocalAttentionSpec` (Llama-4 local) analogous with `attention_chunk_size` (407-431).

**Hybrid models (Gemma2/3: 5:1 SWA:full)**: layers are grouped by spec into KV-cache groups with equal layer counts (`_get_kv_cache_groups_uniform_page_size`, `kv_cache_utils.py:1057-1176` — the docstring walks the Gemma case). All groups must share a uniform page size; block tables are allocated per group; total blocks = `available // (page_size * group_size)` where group_size = layers per group (1304-1306). The hybrid manager means **full-attention layers dominate per-request KV cost** while SWA layers stay cheap — naive `2·heads·dim·layers·len` over-estimates Gemma KV by ~6×. `disable_hybrid_kv_cache_manager` (`config/scheduler.py:132-138`) collapses everything to full-attention accounting (wasteful; only for debugging/unsupported combos). `mamba_block_size`/`mamba_cache_mode` (`cache.py:120-140`) cover SSM-hybrid models similarly.

### 3.3 MLA (DeepSeek)

`MLAAttentionSpec` (`kv_cache_interface.py:337-396`): effectively `num_kv_heads=1` with a single latent `head_size` (576 = 512 kv_lora_rank + 64 rope for V3), and **no 2× K+V factor** — `real_page_size_bytes = block_size * num_kv_heads * head_size * dtype_size` (363-368). `fp8_ds_mla` uses fixed custom layouts: 656 B/token (V3.2) or 584 B/token (V4) (354-362). MLA KV is ~10-50× smaller per token than equivalent GQA — relevant if AgileRL ever fine-tunes DeepSeek-family models.

### 3.4 Auto-estimation utilities worth reusing

- `estimate_max_model_len(vllm_config, kv_cache_spec, available_memory)` — binary search, `kv_cache_utils.py:740-791`.
- `get_max_concurrency_for_kv_cache_config` — `kv_cache_utils.py:877-895`.
- `max_memory_usage_bytes` — `kv_cache_utils.py:731-737` (sums per-spec worst case).
These embody vLLM's own KV arithmetic including hybrid/SWA/MLA corner cases; an auto-configurer can import them rather than reimplementing.

---

## 4. Prefill vs decode tuning

### 4.1 How the V1 scheduler spends its token budget

`Scheduler.schedule()` (`vllm/v1/core/sched/scheduler.py:336-830`): one global budget `token_budget = max_num_scheduled_tokens` (= `max_num_batched_tokens` unless spec decode shrinks it, `config/scheduler.py:56-61`). **RUNNING requests are scheduled first** (line 364-498) — i.e. decodes (1 token each, + draft tokens) and in-flight chunked prefills — then WAITING requests consume the *remainder* (548-802). There is no separate prefill/decode phase: chunked prefill mixes both in every step. Consequences:

- **`max_num_batched_tokens` is the chunked-prefill chunk size.** A waiting prompt is admitted with `num_new_tokens = min(prompt_len_remaining, token_budget_left)` (scheduler.py:654-669). Bigger budget = faster prefill completion, but each step's forward is bigger, so **inter-token latency for concurrently decoding sequences degrades** — and in RL that latency is pure rollout wall-clock only if decodes dominate the tail.
- **`long_prefill_token_threshold`** (`config/scheduler.py:80-82`, default 0=off) caps tokens scheduled per request per step (applied at scheduler.py:390-391 for running and 655-657 for waiting). It defaults to `0.04 * max_model_len` only when `max_num_partial_prefills > 1` is set (`config/scheduler.py:244-246`). It is the knob to stop one giant prompt monopolizing steps.
- **`max_num_partial_prefills` / `max_long_partial_prefills` are vestigial in V1**: defined in config (`config/scheduler.py:70-78`) and CLI, but **nothing in `vllm/v1/` consumes them** (grep confirms only config + arg_utils references). Their only effect is triggering the long-threshold default above. Do not surface them as real knobs.
- `max_num_seqs` caps concurrent scheduled requests per step (`config/scheduler.py:63-68`); the real decode concurrency limit is usually KV space (§2.4 max-concurrency log), whichever is smaller.

### 4.2 Decode-heavy RL envs (short prompts, long multi-turn generations)

What matters, in order:
1. **KV capacity** — decode throughput saturates when `num_running ≈ max_concurrency`; everything in §2/§3 (util, kv_cache_dtype=fp8, max_model_len right-sizing) feeds this. Preemption (recompute, §1.8) is the failure mode when KV runs out mid-rollout — for grouped GRPO rollouts all sequences grow together, hitting KV pressure simultaneously; budget for `group_size × num_prompts × expected_len`, not average.
2. **`max_num_seqs`** — must be ≥ desired concurrent rollouts; default 256 (A100-class) is usually plenty; raising it costs cudagraph captures (max capture = `min(max_num_seqs·2, 512)`) and bigger persistent batch buffers.
3. **CUDA graphs** — decode steps are small-batch and Python-overhead-bound; `FULL_AND_PIECEWISE` (default O2) matters most here. `enforce_eager=True` for faster startup is a meaningful *decode* throughput sacrifice (commonly 10-30% at small batch).
4. **Prefix caching** — multi-turn envs resubmit the conversation each turn; with prefix caching on (default), turns t<n hit cache and only the new turn is prefilled. Shared system prompts across a GRPO group dedupe similarly (group of N samples from one prompt: 1 prefill + N-1 cache hits). Remember the LoRA-name key and `reset_prefix_cache()` on in-place weight updates (§1.6).
5. **`max_num_batched_tokens` can be modest** (4-8K): prompts are short, so prefill admission isn't the bottleneck.

### 4.3 Prefill-heavy RL envs (long prompt, short completion — e.g. reasoning-over-context)

1. **`max_num_batched_tokens` is the throughput knob** — it bounds prefill tokens/step. The H100 default (16384 offline) vs A100 (8192) reflects exactly this trade (arg_utils.py:2290-2312). Raising it raises the profile-run activation reservation (§2.2) and compile time; on colocated 40 GB cards measure before raising.
2. Chunked prefill is on by default; with mostly-prefill traffic the budget is spent almost entirely on prefill anyway.
3. `scheduler_reserve_full_isl=True` (default) prevents admitting prompts whose full ISL can't fit — keep it.
4. Prefix caching still pays if prompts share long templates/few-shot prefixes; otherwise it's near-free overhead (hashing).
5. `long_prefill_token_threshold` only if you need decode latency fairness while giant prompts stream in — for batch RL rollouts, leave 0 (max throughput).

---

## 5. Sleep mode and LoRA serving

### 5.1 Sleep levels (`vllm/entrypoints/llm.py:994-1032`, `vllm/v1/worker/gpu_worker.py:160-199`, `vllm/device_allocator/cumem.py:171-243`)

Requires `enable_sleep_mode=True` (`config/model.py:305`; implies/forces `enable_cumem_allocator`, model.py:530-538; CUDA-only). Weights are allocated under the cumem tag `"weights"` (gpu_worker.py:338-345), KV cache under `"kv_cache"` (gpu_worker.py:553).

- **Level 0** (new in this version): pause scheduling only; nothing freed (llm.py:1002-1003). Resume with `wake_up(tags=["scheduling"])`.
- **Level 1**: `allocator.sleep(offload_tags=("weights",))` — weights **copied to pinned CPU memory** and GPU pages unmapped; everything else (KV cache, cudagraph pools, workspaces in the pool) **discarded** (gpu_worker.py:173; cumem.py:192-207). Wake restores weights from CPU. Needs CPU RAM ≥ weights size.
- **Level 2**: `offload_tags=tuple()` — **everything discarded**, including weights; only module buffers (small) are saved/restored Python-side (gpu_worker.py:165-170, 190-196). Wake leaves weight memory mapped-but-uninitialized; caller must reload weights.
- `wake_up(tags=["weights"])` then later `wake_up(tags=["kv_cache"])` enables the RLHF two-phase pattern: map weights, copy in new weights, then re-create KV — avoids holding both at peak (llm.py:1019-1032). KV contents are **always lost** on sleep (both levels): prefix cache resets, in-flight requests must be drained (`mode="abort"|"wait"|"keep"` param, llm.py:994,1014-1016).

**AgileRL-specific caveat (from prior validation work, not this repo):** for bnb-4bit (QLoRA) bases, level 2 is unusable (weights can't be re-quantized in place on reload) and level 1 frees little of what the trainer needs; the working colocated pattern is the "standby" patch keeping `"weights"`-tagged allocations resident and freeing only KV — i.e. effectively a custom level between 1 and 0. The native tags machinery above is exactly the seam that patch hooks (`CuMemAllocator.sleep/wake_up`).

### 5.2 LoRA knobs (`vllm/config/lora.py`)

- `max_lora_rank: 16` default; allowed `{1,8,16,32,64,128,256,320,512}` (lora.py:26,34). Must be ≥ the training LoRA r — **GPU LoRA weight buffers are statically sized `max_loras × max_lora_rank × hidden` per target module**, so setting it above the actual rank wastes VRAM proportionally.
- `max_loras: 1` default (concurrently batched adapters, lora.py:36); RL with a single policy adapter needs exactly 1. `max_cpu_loras` defaults to `max_loras` (lora.py:43-45,110-116).
- Punica index buffers are sized by `max_num_batched_tokens` (`vllm/lora/punica_wrapper/punica_base.py:133-148`; also why `max_num_batched_tokens` is in the compile hash, `config/scheduler.py:204-213`).
- `fully_sharded_loras=False` (lora.py:38-42) — turn on for high TP/rank/seq-len; irrelevant at TP=1.
- `target_modules` (lora.py:48-51) can restrict LoRA application for speed.
- `specialize_active_lora` (lora.py:67-73) — extra cudagraphs per active-LoRA count; skip for single-adapter RL.

---

## 6. Speculative decoding for RL rollouts

Config: `vllm/config/speculative.py`. Methods: `ngram`, `ngram_gpu`, `suffix`, `draft_model`, `medusa`, `mlp_speculator`, EAGLE/MTP family, `custom_class` (speculative.py:59-68).

- **ngram** (`prompt_lookup`): drafts by matching the last n-gram against prior context. Defaults when unset: `prompt_lookup_min = prompt_lookup_max = 5` (speculative.py:587-590, "arbitrarily chosen"); `num_speculative_tokens` required. Zero extra GPU memory for a draft model; CPU-side numba matcher, currently capped at **1 thread** per rank (`vllm/v1/spec_decode/ngram_proposer.py:36-53`, threshold 8192 batch tokens for multithreading). Cost: each accepted-or-not draft token occupies scheduler budget (`num_lookahead_tokens` reserved in `allocate_slots`, scheduler.py:213-220,447) and KV blocks for lookahead; rejected tokens are wasted compute.
- **Benefit profile for RL:** ngram only pays when outputs **copy spans from the context** — code editing, retrieval-grounded answers, multi-turn envs that echo state. For free-form math/reasoning generation (typical GRPO), acceptance is low and it can net-lose. **Sampling caveat:** RL rollouts run temperature ~1.0; rejection sampling (`rejection_sample_method="standard"`, speculative.py:191-194) preserves the target distribution exactly, so logprob-correctness for importance ratios is maintained — but acceptance rates drop sharply at high temperature, further shrinking the win. The `"synthetic"` acceptance mode does NOT preserve the distribution — never use for RL.
- **suffix decoding** (Arctic Inference required, speculative.py:808-815) — adaptive ngram-on-steroids over a suffix tree of prior outputs; promising for repetitive RL envs but adds a pip dependency.
- EAGLE/draft-model: needs trained head / extra VRAM for the drafter; weights go stale as the policy trains — impractical for on-policy RL unless the head is also synced. Skip.
- Interaction: spec tokens inflate `max_cudagraph_capture_size` (`decode_query_len = 1 + num_speculative_tokens`, vllm.py:1639-1647); async scheduling stays enabled only for EAGLE/MTP/draft/ngram-GPU kinds (vllm.py:935-945) — plain CPU `ngram` **disables async scheduling**, an often-overlooked hidden cost.
- **Recommendation for the auto-configurer:** default OFF; expose `ngram` (k=3-5, lookup 2-5) as an opt-in for envs flagged as copy-heavy, and only with measured acceptance.

---

## 7. Override vs trust — summary for the AgileRL auto-configurer

**Trust vLLM (leave default):** chunked prefill (on), prefix caching (on, but call `reset_prefix_cache()` on in-place weight updates), block_size (16/backend), async scheduling (on), scheduler policy (fcfs), cudagraph capture-size list, hybrid KV grouping for Gemma/SWA/MLA, `scheduler_reserve_full_isl`.

**Always set explicitly:**
- `max_model_len` = prompt+completion budget from the env profile (never let it default to the model's 128K — KV "one full request" check will eat the budget).
- `gpu_memory_utilization` — computed from *free-after-trainer* memory in colocated mode (fraction-of-total semantics!), or generous (0.85-0.92) in decoupled mode. Or skip the guesswork with `kv_cache_memory_bytes`.
- `enable_sleep_mode=True` in colocated mode (and the standby patch for QLoRA).
- LoRA: `max_lora_rank` = training r exactly, `max_loras=1`.

**Tune from the workload profile:**
- `max_num_batched_tokens`: prefill-heavy ↑ (8-16K), decode-heavy ↓ (2-8K, freeing activation+LoRA-buffer memory for KV); remember the A100-vs-H100 default split.
- `max_num_seqs`: ≥ concurrent rollout count; check against logged "Maximum concurrency".
- `kv_cache_dtype="fp8"`: decode-heavy lever for 2× concurrency (validate logprob sensitivity for importance-ratio algorithms first).
- `enforce_eager`: only for debugging/startup-time; costs decode throughput.
- `long_prefill_token_threshold`: only if mixing live decodes with huge prompt streams.
- Speculative ngram: opt-in for copy-heavy envs only.

**Sanity formulas** (per GPU, TP-divided): `kv_bytes_per_token = 2 · (num_kv_heads/TP) · head_dim · dtype_size · num_full_attn_layers` (+ bounded SWA-layer term `≈ window` tokens for Gemma-like models); `available_kv ≈ total·util − weights/TP − activation_peak(max_num_batched_tokens) − NCCL/workspace (~1-2 GiB) − cudagraph_pool`; `max_concurrency = available_kv / (kv_bytes_per_token · max_model_len)`. vLLM's own implementations of these live in `vllm/v1/core/kv_cache_utils.py` (§3.4) and are importable.

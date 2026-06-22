# Auto-sizing mechanisms in the single-GPU FT lib + FT-zoo lib

**Slice:** the two single-GPU finetuning libraries (referred to throughout as **the FT lib** — the main training package with `models/`, `kernels/`, `studio/` — and **the FT-zoo lib** — its utility companion package). Both were read at local clones under `~/git/Misc/` (read-only). Naming note: per repo policy the project name is never written here; in file references below, `[zoo]/` means the FT-zoo lib's package root, `[ft]/` means the FT lib's package root, and `[studio]/` means the FT lib's `studio/backend/` tree. Env vars and identifiers that embed the project name are written with the placeholder `<FTLIB>` / `<ftlib>` (uppercase/lowercase project prefix).

These libraries are the state of the art at fitting LLM-RL training **and** a vLLM rollout engine on one GPU (colocated, time-sliced). Their key architectural premise, which everything below assumes: **vLLM is loaded first, then the HF training model is constructed from references to vLLM's own weight tensors** (zero-copy; `get_vllm_state_dict` slices vLLM's fused projections back into HF layout — `[zoo]/vllm_utils.py:945-1064`, consumed in `[ft]/models/llama.py:2569-2580`). So weight memory is paid once, and the auto-sizing problem reduces to splitting the *rest* of VRAM between (a) vLLM KV cache + CUDA graphs and (b) trainer-side LoRA/optimizer/activations.

---

## 1. The master memory formula: `approximate_vllm_memory_usage`

`[zoo]/vllm_utils.py:1487-1601`. Inputs: HF config, quantization flags, `max_seq_length`, requested `gpu_memory_utilization`, LoRA config, fp8-KV flag, `account_for_gradients` (= `training`), and two structural knobs (`parallel_sequences=64` is accepted but unused; `cuda_graph_overhead=True`). Returns `(max_num_batched_tokens, approx_max_num_seqs, actual_gpu_memory_utilization, memory_left_for_kv_cache_gb)`.

### 1.1 Budget basis: fraction of *free* memory, not total

```python
free_memory, total_memory = get_mem_info()          # torch.cuda.mem_get_info()
free_memory = gpu_memory_utilization * free_memory  # [zoo]/vllm_utils.py:1504-1506
...
actual_gpu_memory_utilization = free_memory / total_memory   # :1557
```

The user-facing `gpu_memory_utilization` (default 0.5 in `from_pretrained`) is interpreted as a fraction of **currently-free** VRAM, then converted into the fraction-of-total number vLLM actually wants (`actual_gpu_memory_utilization`), which is what gets passed as the engine arg (`[zoo]/vllm_utils.py:2199`). This makes the knob composable with whatever is already resident.

### 1.2 Model-weight element counts (per config)

`[zoo]/vllm_utils.py:1508-1524`:

```python
vocab_size = config.vocab_size
hd  = config.hidden_size
mlp_size = config.intermediate_size
n_layers = config.num_hidden_layers
n_kv_heads = getattr(config, "num_key_value_heads", 1)
n_heads    = getattr(config, "num_attention_heads", 1)
kv_size = hd // n_heads * n_kv_heads          # GQA-aware

qkvo = (hd + kv_size + kv_size + hd) * hd     # per-layer attn elements
mlp  = (hd * mlp_size) * 3                    # gate/up/down
layernorms = 2 * hd
embed_tokens = vocab_size * hd
lm_head = 0 if tie_word_embeddings else vocab_size * hd
```

### 1.3 LoRA adapter elements (vLLM-side LoRA slots)

`[zoo]/vllm_utils.py:1526-1535` — assumes LoRA on **all** of QKVO + MLP at `max_lora_rank`, times `max_loras`, **2 bytes (fp16)**:

```python
qkvo_A = hd * max_lora_rank * 4
qkvo_B = max_lora_rank * (hd + kv_size + kv_size + hd)
mlp_A  = hd * max_lora_rank * 2 + mlp_size * max_lora_rank
mlp_B  = max_lora_rank * (mlp_size + mlp_size) + max_lora_rank * hd
lora_elements = (qkvo_A + qkvo_B + mlp_A + mlp_B) * max_loras * n_layers * 2
```

(Two follow-on lines compute `gradient_lora_elements = 2×lora` and `parameter_lora_elements = 4×lora` for 8-bit-Adam state + fp32 master copy — `:1537-1541` — but they are **dead code**: never added to the budget. The training-side reservation is done purely via the activation term below.)

### 1.4 Trainer activation reservation (the colocated "tax")

`[zoo]/vllm_utils.py:1543-1556` — when `account_for_gradients=True` (i.e. `training=True`), a fixed-shape activation peak is estimated at **bsz=2** and `max_seq_length`:

```python
bsz = 2
activation_qkv  = max_seq_length * bsz * (hd + kv_size + kv_size)
residual_memory = (max_seq_length * bsz) * 2
activation_mlp  = max_seq_length * bsz * (mlp_size + mlp_size)
weights = mlp_size * hd                       # one layer's MLP weight as workspace proxy
maximum_activation = (activation_qkv + residual_memory + activation_mlp + weights) * 1.25 * 2   # +25% slack, fp16
if total_memory - free_memory < maximum_activation:
    free_memory = total_memory - maximum_activation
```

i.e. *if the currently-used headroom is smaller than the projected training-activation peak, shrink vLLM's budget so that the peak fits*. Context length enters linearly here. This is a **single-layer** activation model — it implicitly assumes their offloaded gradient checkpointing (Section 7) keeps only ~1 layer of activations live.

### 1.5 Quantization bytes-per-param factors

`[zoo]/vllm_utils.py:1559-1567`:

```python
total_quantizable_elements = (qkvo + mlp) * n_layers * 2          # fp16 bytes
total_float16_elements     = (layernorms + embed_tokens + lm_head) * 2
factor = 1
if load_in_4bit:   factor = 16/5    # = 3.2  (≈ 0.625 byte/param: nf4 + blockwise scales; "should be 4.5 but use 5")
elif load_in_8bit: factor = 8/5     # = 1.6  ("very vague approximation")
bytes_for_model = total_quantizable_elements / factor + total_float16_elements + lora_elements
```

Note `load_in_8bit` is overloaded: `load_vllm` passes `is_fp8` into it (`[zoo]/vllm_utils.py:1871`), so fp8 checkpoints are sized with the same 8/5 factor.

### 1.6 KV-cache sizing and derived limits

`[zoo]/vllm_utils.py:1569-1600`:

```python
float_bytes = 1.25 if float8_kv_cache else 2          # fp8 KV ≈ 1 byte + scale overhead
kv_elements = (kv_size * 2 * n_layers) * float_bytes  # bytes per token of KV
memory_left_for_kv_cache = free_memory - bytes_for_model

# CUDA-graph overhead reservation (FULL_AND_PIECEWISE + LoRA ≈ 172 graphs)
if cuda_graph_overhead:
    num_cuda_graphs = 172          # "51 sizes x 2 lora + 35 x 2 decode"
    per_graph_estimate = hd * n_layers * 4
    _cuda_graph_overhead = num_cuda_graphs * per_graph_estimate
    _cuda_graph_overhead = max(_cuda_graph_overhead, int(0.15 * 1024**3))   # floor 0.15 GiB
    _cuda_graph_overhead = min(_cuda_graph_overhead, int(1.0  * 1024**3))   # cap 1.0 GiB
    memory_left_for_kv_cache -= _cuda_graph_overhead

max_num_batched_tokens = int(0.95 * (memory_left_for_kv_cache / kv_elements))  # 5% safety
max_num_batched_tokens = (max_num_batched_tokens // 256) * 256                 # round down to 256
approx_max_num_seqs = int(max_num_batched_tokens / max_seq_length)             # worst case: every seq at full length
```

So: **KV token capacity = 0.95 × (budget − weights − LoRA − cudagraphs) / (2·kv_size·n_layers·bytes)**, and the theoretical concurrent-sequence count assumes each request occupies `max_seq_length` tokens — a deliberately pessimistic decode-heavy assumption.

### 1.7 How the analytical numbers are actually used

Important subtlety in `load_vllm` (`[zoo]/vllm_utils.py:1866-1910, 2048-2059`): the *analytical* `max_num_batched_tokens` / `approx_max_num_seqs` are used only to (a) compute `memory_left_for_kv_cache_gb`, (b) clamp `max_seq_length` down when the GPU can't hold even one full sequence:

```python
if max_num_batched_tokens <= 0:
    max_seq_length = 256; max_num_batched_tokens = 256
if max_num_batched_tokens <= max_seq_length:
    print("...Your GPU cannot handle sequence lengths of {max_seq_length}...")
    max_seq_length = max_num_batched_tokens          # [zoo]/vllm_utils.py:1900-1909
```

— and then both are **overwritten by an empirical tier table keyed on KV-cache GB headroom** (Section 2.2). The analytic formula picks the regime; lookup tables pick the engine knobs.

---

## 2. `load_vllm`: the full auto-configuration pipeline

`[zoo]/vllm_utils.py:1741-2373`. Signature defaults worth recording (`:1741-1767`): `gpu_memory_utilization=0.8`, `max_seq_length=8192`, `enable_lora=True`, `max_lora_rank=16`, `max_loras=1`, `enable_prefix_caching=True`, `compilation_config=3`, `conservativeness=1.0`, `max_logprobs=0` (logprob returns disabled to save memory), `max_num_seqs=256`, `enforce_eager=False`, `float8_kv_cache=False`. The FT lib's user-facing loaders default to `gpu_memory_utilization=0.5` and `max_lora_rank=64` (`[ft]/models/loader.py:262,265`) or `max_lora_rank=16` at the model-class level (`[ft]/models/llama.py:2224-2227`).

### 2.1 Standby-mode GPU-utilization targets (total-VRAM-keyed tiers)

When standby (sleep-mode time-slicing) is enabled, the lib **overrides the user's `gpu_memory_utilization` in both directions** to a per-GPU-size target (`[zoo]/vllm_utils.py:1778-1829`). Tiers are keyed on `ten_percent = 0.1 × total_GB`:

For vLLM ≥ 0.11 (`:1795-1815`):

```python
if   ten_percent >= 4.0: standby_target_gpu_util = 0.925   # ≥ 40 GB GPUs
elif ten_percent >= 2.5: standby_target_gpu_util = 0.9     # 25–40 GB
elif ten_percent >= 2.0: standby_target_gpu_util = 0.825   # 20–25 GB (L4/4090 class)
elif ten_percent >= 1.4: standby_target_gpu_util = 0.8     # 14–20 GB (T4 class)
elif ten_percent >= 1.0: standby_target_gpu_util = 0.775   # 10–14 GB
else:                    standby_target_gpu_util = 0.75
standby_target_gpu_util *= 0.95     # extra headroom: "vLLM ≥0.11 uses more"
```

(So an A100-40GB lands at 0.925×0.95 ≈ **0.879**; a 24 GB card at ≈ 0.784.) For vLLM < 0.11 a slightly lower table is used (0.9 / 0.875 / 0.85 / 0.825 / 0.8 / 0.75 / 0.7, `:1785-1794`). The override is bidirectional — too-low user values are raised ("Standby mode is enabled. Changing gpu_memory_utilization"), too-high values are lowered ("your setting ... will OOM") (`:1820-1829`) — escape hatch env `<FTLIB>_VLLM_STANDBY_UTIL_OVERRIDE` (`:1776`). The comment block at `:1796-1803` documents *why* the ≤24 GB tiers were lowered: vLLM reserves `util × 0.95 × total` for weights+KV; the remainder must fit the HF training side, and an 8B model in 4-bit needs ~4–5 GB for weights alone, otherwise `wake_up(tags=["kv_cache"]) → create_and_map` fails at the CUDA VMM level with `cudaErrorIllegalAddress`.

Pre-flight guard: if the analytic KV headroom is < 1 GB under standby, it warns before constructing `LLM()` ("may crash unrecoverably") (`:1881-1887`).

### 2.2 `max_num_seqs` / `max_num_batched_tokens` tier table (KV-GB-keyed)

`[zoo]/vllm_utils.py:2048-2059`, preceded by an in-source benchmark table (`:2029-2047`) showing non-KV memory vs `max_num_seqs` ("after max_num_seqs ≥ 64 we see linear increase in memory usage": 8.31 GiB at 64 → 9.14 at 256 → 18.80 GiB at 2048 for an 8B model):

```python
approx_max_num_seqs = max_num_seqs   # vLLM default 256
max_num_batched_tokens = 2048        # vLLM default
if   kv_gb <=  2: tokens, seqs = 2048, 8
elif kv_gb <=  4: tokens, seqs = 2048, 16
elif kv_gb <=  8: tokens, seqs = 4096, 32
elif kv_gb <= 12: tokens, seqs = 4096, 48
elif kv_gb <= 16: tokens, seqs = 6144, 64
elif kv_gb <= 24: tokens, seqs = 6144, 80
elif kv_gb <= 40: tokens, seqs = 8192, 96
elif kv_gb <= 48: tokens, seqs = 8192, 112
elif kv_gb <= 80: tokens, seqs = 8192, 128
else:             tokens, seqs = 8192, 256
```

In vLLM V1, `max_num_batched_tokens` *is* the chunked-prefill token budget per scheduler step, so this table is effectively the **prefill-chunk size schedule**. A second, finer `chunked_prefill_tokens` table exists (`:2076-2085`: 1024/1536/2048/3072/4096/4608/8192 over the same GB tiers) but is immediately overwritten with `chunked_prefill_tokens = max_seq_length` (`:2088`, "vLLM errors out from max_seq_length being bigger than chunked_prefill_tokens") and afterwards is only printed, never passed to the engine — the engine receives the 2048–8192 table value.

Modifiers:
- **Vision models**: `approx_max_num_seqs = 1`, `max_num_batched_tokens = max(8192, max_seq_length)` (each sequence may carry a multi-thousand-token image; `:2061-2070`), plus `limit_mm_per_prompt = {"image": 1, "video": 0}` (`:2229-2232`).
- **fp8 KV cache**: `approx_max_num_seqs ×= 1.05` (`:2072-2073`).
- **`conservativeness`** ∈ [0,1] linearly scales `approx_max_num_seqs` (floor 1) for low-VRAM devices (`:2090-2092`).

### 2.3 Swap space (host-RAM-keyed)

`[zoo]/vllm_utils.py:2094-2104`: `swap_space = 0` for ≤16 GB available host RAM, 2 GB for ≤24, 4 GB for ≤48, else 6 GB.

### 2.4 Engine args actually emitted

`[zoo]/vllm_utils.py:2197-2228`:

```python
engine_args = dict(
    model=..., gpu_memory_utilization=actual_gpu_memory_utilization,
    max_model_len=max_seq_length,
    quantization="bitsandbytes" if use_bitsandbytes else None,
    load_format="bitsandbytes" if use_bitsandbytes else "auto",
    kv_cache_dtype="fp8" if float8_kv_cache else "auto",
    dtype=dtype,                      # bf16 on cc≥8.0/HIP/XPU else fp16 (:1912-1928)
    max_num_batched_tokens=max_num_batched_tokens,
    max_num_seqs=approx_max_num_seqs,
    max_logprobs=max_logprobs,        # 0: disallow logprob returns
    seed=random_state,
    enable_lora=enable_lora, max_lora_rank=max_lora_rank, max_loras=max_loras,
    enable_prefix_caching=enable_prefix_caching,
    enable_chunked_prefill=enable_chunked_prefill,   # True except mllama
    compilation_config=compilation_config,
    enforce_eager=enforce_eager,
    swap_space=swap_space,
    enable_sleep_mode=<ftlib>_vllm_standby,
)
```

Unknown keys are stripped against `inspect.signature(EngineArgs)` for cross-version safety (`:2275-2282`).

Hardware-conditional extras:
- **Cascade attention disabled** on cc < 9 with vLLM < 0.11 — flagged `[[CRITICAL for RL on policy]]`; cascade attention on A100/L40 corrupted on-policy sampling (`:2234-2249`).
- **`block_size=32`** on Blackwell (cc ≥ 10) when `head_dim ≥ 256` to dodge a FlashInfer bug (`:2251-2260`).
- **LoRA rank rounded up** to vLLM's supported set (`{8,16,32,64,128,256,320,512}`, introspected from vLLM source at runtime) via `determine_max_lora_rank` (`[zoo]/vllm_utils.py:1604-1637`).
- **FlashInfer backend selection** with nvcc/ninja pre-flight, per-model support introspection (regex over the vLLM model source for attention-backend guards), `VLLM_USE_FLASHINFER_SAMPLER=1` on cc ≥ 8 (`:1941-1997`); `VLLM_USE_DEEP_GEMM=0` for fp8 on cc 10 (slower than triton, `:1856-1862`).
- **Prefix caching disabled** on cc < 7.5 (`:2000-2010`).
- On-the-fly **fp8 quantization** via torchao for vLLM ≥ 0.12 (`fp8_mode="row"|"block"` → `quantization="torchao"` + `hf_overrides` carrying a serialized `Float8DynamicActivationFloat8WeightConfig`, `:2263-2273`, config builder `:1717-1738`).

### 2.5 CUDA-graph / compile configuration

`compilation_config=3` expands into a `CompilationConfig` (`:2126-2195`): inductor backend, `cudagraph_num_of_warmups=1`, `use_cudagraph=True`, `cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE` when available; explicit `cudagraph_capture_sizes` / `max_capture_size` are deliberately left to vLLM defaults (commented out, `:2149-2151`). They do **not** shrink the capture-size list — instead they *budget* for it via the 172-graph reservation in the memory model (Section 1.6). `max_autotune`/`coordinate_descent_tuning` are off ("too slow"); combo kernels disabled below 80 GB GPUs (`:2135-2140`). GC is suppressed during graph capture for speed (`patch_vllm_graph_capture`, `:700-777`).

### 2.6 OOM retry/backoff loop

`[zoo]/vllm_utils.py:2287-2355`: engine construction retries once; on a memory-flavored error it backs off **`max_num_seqs ×= 0.75` and `gpu_memory_utilization ×= 0.85`** and retries. Under standby it never retries (the patched `CuMemAllocator` can't be re-instantiated in-process) and instead raises a `MemoryError` with ranked remediation advice: lower `gpu_memory_utilization` → disable standby → smaller model / 4-bit (`:2307-2320`).

### 2.7 Rollout batching by total VRAM

`generate_batches` (`[zoo]/vllm_utils.py:2632-2666`) splits generation requests so the engine never sees more than a VRAM-dependent slice of `llm.approx_max_num_seqs` (which `load_vllm` stamps on the engine object, `:2357`):

```python
if   total_memory_gb <=  8: n_batches = llm.approx_max_num_seqs // 10
elif total_memory_gb <= 16: n_batches = llm.approx_max_num_seqs // 5
elif total_memory_gb <= 24: n_batches = llm.approx_max_num_seqs // 2
else:                       n_batches = llm.approx_max_num_seqs
```

Override env: `<FTLIB>_VLLM_BATCHES` (`:2636`). `create_batches` (`:2376-2385`) is the plain splitter (default 64/req batch).

---

## 3. Standby / sleep-mode time-slicing (sizing-relevant parts)

Mechanism (`patch_vllm_enable_sleep_mode`, `[zoo]/vllm_utils.py:525-697`): `CuMemAllocator.sleep`/`wake_up` are replaced so any allocation tagged `"weights"` is **never** unmapped or remapped — base weights stay resident permanently because the trainer aliases them zero-copy. Everything else (notably the `"kv_cache"` tag and default-tag allocations) is `unmap_and_release`d on sleep; tags listed in `offload_tags` get a pinned-CPU backup first (`:589-614`). `wake_up` re-`create_and_map`s and restores CPU backups (`:617-645`). Net effect of `sleep(1)`: **weights stay on GPU; KV cache + scratch are freed back to the driver** — exactly the freed pool the trainer's optimizer/activations then use.

Enablement & gating: env `<FTLIB>_VLLM_STANDBY=1` (checked in `patch_vllm` `:793-807`; hard version gates: vLLM 0.10.x raises "std::bad_alloc ... insufficient memory headroom", 0.14.x raises "cudaErrorIllegalAddress ... during sleep/wake cycles" — `:794-805`). The model loaders refuse `<ftlib>_vllm_standby=True` without the env var (`[ft]/models/llama.py:2258-2264`, `[ft]/models/vision.py:613-615`).

### Triggers — when sleep/wake actually happens

1. **Auto-wake on generate**: `vllm.LLM.generate` and `AsyncLLMEngine.generate` are wrapped to call `self.wake_up()` first whenever `enable_sleep_mode` is set (vLLM internally no-ops if awake) — `[zoo]/vllm_utils.py:674-691`.
2. **GRPO trainer source rewrite** (for TRL versions without built-in sleep, i.e. < 0.23): every `self.llm.generate(...)` line in the trainer is wrapped with `wake_up()` before and `self.llm.sleep(os.environ.get('VLLM_SLEEP_MODE', 1))` after — `[ft]/models/rl_replacements.py:881-907`. So the duty cycle is: *wake → generate rollouts → sleep → grad steps → repeat*, with the sleep level controlled by env `VLLM_SLEEP_MODE` (default `1`).
3. **TRL ≥ 0.23 built-in sleep compatibility**: TRL's `wake_up(tags=["kv_cache"])` is regex-rewritten to bare `wake_up()` ("wakes everything to set is_sleeping=False"; safe because the patched allocator skips weights anyway), and TRL's duplicate `collective_rpc("reload_weights")` is stripped — `[ft]/models/rl_replacements.py:1928-1999`.
4. **Resume-from-checkpoint guard**: before `trainer.train(resume_from_checkpoint=...)`, the engine is put to sleep (`llm.sleep(1)`) so checkpoint loading has memory — `[ft]/models/rl.py:98-147`.
5. For TRL configs, when standby env is set the patcher injects `args.vllm_enable_sleep_mode=True` and forces `args.vllm_mode='colocate'` plus `self.llm = model.vllm_engine` (TRL's own `LLM(...)` construction is regex-replaced) — `[ft]/models/rl.py:1907-1931, 2004-2015`.

Sizing logic *around* standby is Section 2.1's tier table; the design contract restated: vLLM gets `util×0.95×total` for weights+KV, and `(1−util×0.95)×total` must cover LoRA params, optimizer states, activations and gradient-checkpoint buffers of the trainer. The allocator patch refuses `expandable_segments:True` in any `PYTORCH_*ALLOC_CONF` (`:536-544`).

---

## 4. Auto batch-size / chunking logic in training

### 4.1 GRPO logprob mini-batch autotune (`autotune_batch_and_chunks`)

`[zoo]/rl_replacements.py:281-330`. GRPO's expensive step is recomputing per-token logprobs over `(rows = bsz·num_generations, seq_len, vocab)`. The autotuner picks the **largest row-count B** whose hidden-state + chunked-logit memory fits in 80% of currently-free VRAM:

```python
final_m = max(4, seq_len // 4096) if multiplier is None else multiplier   # logit chunk multiplier
free_bytes, _ = torch.cuda.mem_get_info()
limit_gb = (free_bytes / 1024**3) * 0.80
b_vals = arange(total_input_rows, 0, -1)
hidden_gb = b_vals * seq_len * hidden_size * dtype_bytes / GB
logits_gb = ((b_vals/total_rows) * b_vals * seq_len * vocab_size * dtype_bytes / GB) / final_m
ok = (hidden_gb + logits_gb) <= limit_gb
# first (largest) feasible B wins; if none: return (4, final_m)  ← "your GPU will OOM"
```

Notes: `dtype_bytes` is passed as 16 for fp16/bf16 (32 for fp32) — `[zoo]/rl_replacements.py:775` — i.e. an ~8× safety factor over the raw element size, absorbing autograd temporaries; the logit term scales *quadratically* in B (the `(B/total)·B` factor) and is divided by the chunk multiplier because logits are materialized `final_m` chunks at a time. The number of logit chunks grows with context (`seq_len // 4096`).

Wiring (`[zoo]/rl_replacements.py:782-810`): runs once per accumulation window (`trainer._has_autotuned` flag), converts B to `args.<ftlib>_grpo_mini_batch = max(1, total_rows//B)` and stores the multiplier; user can pin both via config fields `<ftlib>_grpo_mini_batch` / `<ftlib>_logit_chunk_multiplier` (declared at `[ft]/models/rl.py:452-463`, validated ≤ generation batch at `:477-484`). A second, simpler chunking knob `<ftlib>_num_chunks` (default −1 → per-row chunking, snapped to a divisor of bsz via `factors`/`searchsorted` — `[zoo]/rl_replacements.py:763-766`) controls the fused-loss chunk count.

### 4.2 Selective-log-softmax chunking

All RL logprob computations run through fixed 4-way chunked log-softmax in fp32 (`chunked_selective_log_softmax`, `[zoo]/rl_replacements.py:44-67`) or hidden-states→lm_head chunked matmul (`chunked_hidden_states_selective_log_softmax`, `:70-114`) so the `(rows·seq, vocab)` logits never fully materialize.

### 4.3 Trainer-config batch defaults & constraints (GRPO)

The TRL-config patcher forcibly rewrites dataclass defaults (`[ft]/models/rl.py:1292-1330`): `per_device_train_batch_size=4`, `gradient_accumulation_steps=2`, `optim="adamw_8bit"`, `torch_empty_cache_steps=250`, `num_generations=8`, `vllm_mode="colocate"`, and **`auto_find_batch_size=False`** with the comment "Auto /2 batch size — too many people complained so removing" (`:1318`; re-asserted for GRPO "Cannot work on GRPO", `:1345`). Divisibility guard: `per_device_train_batch_size · grad_accum · world_size` must be a multiple of `num_generations`, else batch size is forcibly set to `num_generations` (`:1454-1486`). `dataset_num_proc` is auto-derived from CPU count and available host RAM (1 proc if <2 GB free; capped at `int(free_GB)`; `:1399-1412`).

### 4.4 Auto sample-packing / padding-free

The FT lib auto-enables padding-free batching for SFT when the user didn't set it (`_should_auto_padding_free`, `[ft]/trainer.py:72-79`), with a model blocklist (`gemma2`, `gpt_oss`; `:60-63`) and env kill-switch `<FTLIB>_DISABLE_AUTO_PADDING_FREE` (`:56-58`) — a token-throughput-per-VRAM optimization rather than a sizing formula, but it changes the effective tokens/step for a given batch size.

---

## 5. The "will it fit" estimator (studio backend)

`[studio]/utils/hardware/vram_estimation.py` (1308 lines) + spec `[studio]/utils/hardware/VRAM_ESTIMATION.md`. This is their most complete predictive model: **Total VRAM = weights + LoRA adapters + optimizer states + gradients + activations + CUDA overhead**, "all constants empirically calibrated against Llama-3.2-1B on B200" (`vram_estimation.py:4-11`).

### 5.1 Constants / bytes-per-param tables

`vram_estimation.py:18-70`:

```python
QUANT_4BIT_FACTOR = 16 / 5                    # 3.2 — nf4 + blockwise scales
DOUBLE_QUANT_4BIT_FACTOR = 3.6                # bnb double-quant repos (tighter)
CUDA_OVERHEAD_BYTES = int(1.4 * 1024**3)      # driver + torch runtime, calibrated RTX 5070 Ti
NON_FLASH_ATTENTION_FACTOR = 12.0             # eager attn score+workspace multiplier

OPTIMIZER_BYTES_PER_PARAM = {
    "adamw_8bit": 4,            # "BNB upcasts to fp32 during step"
    "paged_adamw_8bit": 4, "adamw_bnb_8bit": 4,
    "paged_adamw_32bit": 8,
    "adamw_torch": 6,           # "fused, no master copy"
    "adamw_torch_fused": 6,
    "sgd": 4,
}                                # unknown optimizers default to 4 (:1130-1133)

GC_LAYER_MULTIPLIERS = {         # effective live layers under gradient checkpointing
    "none":   (None, None),      # all L layers
    "true":   (2.0, 1.0),        # HF GC: full-FT 2.0 layers, LoRA 1.0 layer
    "<ftlib>": (1.5, 1.0),       # their offloaded GC: full-FT 1.5, LoRA 1.0
}
```

Gradients: `trainable_params × 2` bytes (fp16, accumulated in place; `:1136-1137`). LoRA adapters: `lora_params × 2` (`:1126-1127`).

### 5.2 Weights

`compute_model_weights_bytes` (`:913-934`): QLoRA → `quantized_elements × 2 / quant_4bit_factor + skipped_modules × 2 + non_quantizable × 2`; LoRA/full → everything × 2. Element counting handles GQA, MoE (routed + shared experts, dense-layer interleaving via `first_k_dense_replace` / `decoder_sparse_step` / `mlp_layer_types` / `moe_layers`, `:189-241`), MLA low-rank attention (DeepSeek-style `q_lora_rank/kv_lora_rank`, `:795-809`), KV-shared layers, per-layer-input embeddings, double-wide MLPs, tied embeddings, and `llm_int8_skip_modules` exclusions matched against a full alias map of module paths (`:610-772`).

### 5.3 LoRA parameter counting

`compute_lora_params` (`:995-1123`): per selected target module `in_dim·r + r·out_dim`, MoE experts × `num_experts`, with the policy detail that **`all-linear` excludes routed (nn.Parameter) experts** because peft only attaches to `nn.Linear` (`:1029-1043`). MLA has its own LoRA shape table (`_lora_attn_elements`, `:943-973`).

### 5.4 Activations (and the seq-len scaling law)

Per layer (same shape as the FT-zoo vLLM formula, `:1192-1209`):

```python
Per_layer = (S·B·(qkv_size) + S·B·2 + S·B·(2·mlp_size) [+ PLE]) × 2 × 1.25
```

For MoE layers, `mlp_size` is multiplied by `num_experts_per_tok` (+ shared experts + parallel dense MLP) since all routed intermediates are live (`:1156-1189`). With gradient checkpointing, activations = `max_layer_bytes × gc_multiplier` (1.0–2.0 layers); without, the sum over all layers (`:1212-1245`). Non-flash attention adds a quadratic term and the total is `max(linear, B·heads·S²·2·12.0·effective_layers)` (`:1146-1153`, doc §5) — i.e. **seq-len scaling is linear under flash/SDPA/flex and quadratic otherwise**, and the attention implementation is resolved through the FT lib's own resolver with a conservative "eager" fallback (`[studio]/utils/hardware/hardware.py:1059-1072`).

### 5.5 Floors (LoRA vs full-FT asymmetry)

`estimate_training_vram` (`:1260-1308`):

```python
gradient_floor = int(model_weights * 0.15)                  # full FT: autograd + fragmentation
if is_lora:
    gradient_floor = min(gradient_floor, max(activations, optimizer_bytes))
gradient_bytes = max(trainable_params * 2, gradient_floor)
```

— prevents the frozen 4-bit base from dominating overhead estimates in QLoRA (doc §6).

### 5.6 Multi-GPU pooling and auto GPU selection

`VramBreakdown.min_gpu_vram(n)` (`:145-158`): weights/LoRA/optimizer/gradients shard, **activations + CUDA overhead do not** — per-GPU need = `shardable/n + activations + overhead`. `auto_select_gpu_ids` (`[studio]/utils/hardware/hardware.py:1123-1272`) greedily adds GPUs ranked by free VRAM until both (a) pooled capacity fits, where each GPU after the first contributes only **85%** of its free VRAM (NCCL buffers/transfer/fragmentation discount, `:1204-1211`), and (b) the smallest selected GPU clears the precomputed `min_per_gpu_N` figure (`:1229-1235`). Inference-only fast path: 4-bit → `size/3.2 + max(30%, 2 GB)` buffer; 16-bit → `size × 1.3` (`hardware.py:1029-1036`). Config-less fallback ratios: full-FT `weights × 3.5 + 1.4 GB`; QLoRA `weights/3.2 + 4%·weights (LoRA) + 15%·weights·(B/4)·(S/2048) (activations) + 1.4 GB`; LoRA same but full-precision weights (`hardware.py:1104-1116`). Model size discovery order: HF safetensors metadata (`params × 2`) → config-derived param count → local weight-file bytes → a synthetic-1TB call into the FT-zoo's `approximate_vllm_memory_usage` to back out weight bytes (`hardware.py:948-986, 903-945`).

---

## 6. Activation-memory machinery that the sizing models assume

These are why the activation estimates can assume ~1 live layer:

- **Offloaded gradient checkpointing** (`[zoo]/gradient_checkpointing.py`): hidden states are checkpointed to pinned CPU buffers (`hidden_states.to("cpu", non_blocking=True)`, `:162`), with pre-allocated pools — `INITIAL_CPU_BUFFER_SIZE = 128·1024` elements × 200 buffers, `INITIAL_GPU_BUFFER_SIZE = 2·256·2048`, and **double-buffered H2D prefetch enabled only when free CUDA memory > `DOUBLE_BUFFER_HEADROOM = 512 MB`** (`:51-55`), with runtime self-disable on OOM ("Disabled double buffering due to insufficient VRAM", `:536-555`). Offload is skipped for tensors < 2 MB and for the last layer ("uses more VRAM and is slower", `:410-418`). Buffers grow on demand (`:515-517`).
- **Tiled MLP** (`[zoo]/tiled_mlp.py`): chunks MLP forward across the flattened `B·S` axis. The chunk size has a closed-form solve from a target activation budget (`get_max_flat_qlen`, `:45-63`):
  `max_flat_qlen = ceil((target_gb·2^30/nbytes − 3·hd·mlp) / (10·hd + 3·mlp + hd))`, padded to 64; when `target_gb` is unset it defaults to **50% of currently-free VRAM** (`:228-232`). The `10·hd` term models saved per-token tensors ("2 norms, 2 residual, 4 QKVO, 2 attention"). Patch selection by module-name suffix (`.mlp`, `.ffn`, `.feed_forward`, …, `:297-305`).

---

## 7. fp8 KV cache, quantized KV, decode-vs-prefill levers (consolidated)

- `float8_kv_cache=True` → engine `kv_cache_dtype="fp8"` (`[zoo]/vllm_utils.py:2203`), guarded to cc ≥ 8.0 (`:1835-1837`); sized at **1.25 bytes/element** vs 2 (scale overhead included, `:1570`); rewarded with +5% `max_num_seqs` (`:2072-2073`). Exposed top-level as `float8_kv_cache=False` in every `from_pretrained` (`[ft]/models/llama.py:2225`, `[ft]/models/loader.py:263`).
- Chunked prefill always on (except mllama) with the token budget from the GB-tier table (2048→8192) — this is their main prefill-vs-decode dial: small-KV GPUs take small prefill bites (2048) and few concurrent seqs (8), big GPUs take 8192-token bites with 96–256 seqs.
- `enable_prefix_caching=True` default — directly exploits GRPO's prompt-replication (num_generations× identical prompts) so prefill cost is paid ~once per prompt group.
- `max_num_seqs` is the decode-concurrency dial; their benchmark table (`:2029-2047`) justifies the 8–256 schedule by the measured ~linear non-KV memory growth past 64 seqs.
- `max_logprobs=0` default avoids logprob buffer allocation; GRPO importance-sampling logprobs default off (`vllm_importance_sampling_correction=False`, `[ft]/models/rl.py:1348`).
- fp8 *weights*: native fp8 checkpoints route through `is_fp8` (sized as 8/5 compression), and on-the-fly torchao fp8 (`load_in_fp8='row'|'block'`) for vLLM ≥ 0.12 (`:2263-2273`).
- No fp8/quantized **KV** on the training side — only the vLLM engine.

---

## 8. Transferable design lessons for AgileRL auto-config

1. **Hybrid analytic + lookup**: they compute an analytic weights/KV/activation budget (Section 1) but deliberately let *empirical tier tables* (keyed on KV-headroom GB and total-VRAM GB) choose the actual engine knobs (`max_num_seqs`, `max_num_batched_tokens`, util targets). The analytics decide feasibility and regime; benchmarks decide values.
2. **Budget from free memory, convert to fraction-of-total** for vLLM; reserve trainer activations *before* giving vLLM its share (Section 1.4) — this is the exact knob AgileRL needs for the colocated trainer/rollout split.
3. **Reserve explicitly for CUDA graphs** (~0.15–1.0 GiB, count×`hd·layers·4`) — an easily-missed term that otherwise eats the KV budget.
4. **Standby util must be GPU-size-tiered, not constant**: the (1−util) remainder must cover the *absolute* trainer footprint (4-bit weights ≈ params/3.2 + opt + activations), so bigger GPUs can run higher util (0.879 effective at 40 GB; 0.71–0.78 below 14 GB).
5. **Graceful degradation ladder**: clamp `max_seq_length` to what fits → retry with seqs×0.75/util×0.85 → raise actionable `MemoryError`; never silently retry under standby.
6. **Autotune the training mini-batch once per accumulation window from live `mem_get_info`** with a quadratic logit-memory model and a chunk multiplier that scales with context (`max(4, seq_len//4096)`) — cheap, dynamic, and avoids static worst-casing.
7. The studio's calibrated bytes-per-param tables (optimizer 4/6/8 B/param; 4-bit 3.2–3.6× compression; GC layer multipliers 1.0/1.5/2.0; 1.4 GB CUDA overhead; 15% gradient floor; 0.85 multi-GPU discount; activations-don't-shard rule) are directly liftable as priors for a "will it fit" pre-flight check.

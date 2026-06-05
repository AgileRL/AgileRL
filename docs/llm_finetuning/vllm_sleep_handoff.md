# Handoff: colocated vLLM sleep mode for QLoRA RL (standby sleep + weight sharing)

## TL;DR

Colocated training (HF/PEFT QLoRA trainer + vLLM rollout on one GPU) needs to free
GPU memory between rollout and the trainer's backward pass. vLLM's native
`sleep` is the wrong tool for bnb 4-bit:

- **level 1** (offload weights to CPU, copy back on wake): freed GPU pages are not
  reclaimed from PyTorch's allocator perspective in our setup, and the virtual
  address mapping is retained → trainer OOMs at learn time.
- **level 2** (discard weights, reload on wake): vLLM's bitsandbytes loader
  **cannot reload in place** — `reload_weights` raises `NotImplementedError` for
  the bnb format, and driving the load manually re-quantizes into garbage
  (confirmed: model emits deterministic multilingual word-soup after wake).

**The fix (validated): "standby" sleep — keep the base weights
physically resident on the GPU across sleep/wake and free only the KV cache.**
No reload, no offload, no garbage. In the repro this freed 11.29 GiB (→ 30.6 GiB
free for the trainer) while keeping ~11 GiB of weights resident, and generation
after wake was bit-identical to before sleep.

The **full win** (what to build) is the zero-copy design: the trainer and
vLLM **share the same base-weight tensors** (one copy on GPU), so standby's
"keep weights resident" costs nothing extra and the per-step LoRA sync is the
only data that moves. This handoff documents both.

---

## Reference implementation (local clones)

Local clones used during investigation (dev laptop, under `~/git/Misc/`):
- vLLM source: `~/git/Misc/vllm` (tag context: `v0.21.0` / `v0.21.1rc0`)
- A reference QLoRA + vLLM integration whose `vllm_utils.py` implements the
  standby patch and the zero-copy weight extraction. Browse it for the exact
  mechanism; the function names referenced below come from that file.

### 1. Standby sleep patch — `patch_vllm_enable_sleep_mode` (vllm_utils.py:525-697)

Monkeypatches `CuMemAllocator.sleep` / `wake_up` to **skip every allocation
tagged `"weights"`**:

```python
# sleep (vllm_utils.py:589-610)
for ptr, data in self.pointer_to_data.items():
    if data.tag == 'weights':
        continue                      # keep resident — never offload/discard
    if data.tag in offload_tags:
        ...cudaMemcpy to a pinned CPU backup...
    unmap_and_release(handle)         # only non-weight tags get freed
# wake_up (vllm_utils.py:628-644) symmetrically skips 'weights'
```

The comment in the reference code notes that the weights are managed
externally, so they are neither offloaded/deleted nor onloaded/created here.

They also wrap `LLM.generate` so it auto-`wake_up()`s if asleep
(vllm_utils.py:674-691).

### 2. Zero-copy weight sharing — `get_vllm_state_dict` (vllm_utils.py:849-1180)

Reaches into vLLM's live model
(`llm_engine.engine_core.engine_core.model_executor.driver_worker.model_runner.model`,
vllm_utils.py:881) and extracts every weight (and the bnb `quant_state`) as an
HF-equivalent `state_dict` **that points at the same tensor storage**. The PEFT
trainer is then built on top of those exact tensors, so there is only one base
copy on the GPU. `assert_same_state_dict` validates the aliasing.
`patch_vllm()` (vllm_utils.py:780) disables vLLM multiprocessing so the model is
reachable in-process, and patches the bnb linear method + LoRA tokenizer/loader.

---

## What to build in AgileRL

### Phase 1 (small, high value, already validated): standby sleep

Replace the current "sleep level 2 + reload" path with the standby patch. This
alone makes QLoRA + colocated sleep work today.

1. Add a module-level `patch_vllm_standby_sleep_mode()` to
   `agilerl/algorithms/core/base.py` (the validated code is in the appendix
   below — it is the same logic as the repro that worked).
2. Call it **once before** `self.llm = LLM(**llm_kwargs)` in the colocated init
   (search for `self.llm = LLM(` in base.py).
3. Delete the reload machinery (see "Cleanup" — it should already be gone if you
   reset to `8fdbe842`).
4. `_prepare_vllm_for_training` keeps calling `self.llm.sleep()`; with the patch
   this frees only KV. `_prepare_vllm_for_generation` calls `self.llm.wake_up()`
   and then the existing `_move_lora_to_vllm` (LoRA adapter sync). **No base
   reload.** The base never leaves the GPU.
5. Memory math (A100-40GB, this model): vLLM base resident ≈ 11 GiB, KV freed on
   sleep ≈ 11 GiB → ≈ 30 GiB free for the trainer. With nf4 trainer + activation
   offload this fits comfortably. Keep `--vllm-gpu-memory-utilization` modest so
   the KV pool (the only thing that frees) isn't oversized.

Caveat: standby helps **only** as much as the KV cache is large, because the
weights stay put. For separate-copy QLoRA that's fine on 40GB; for tighter
budgets you need Phase 2.

### Phase 2 (the redesign you actually want): shared base weights

Mirror `get_vllm_state_dict`: after vLLM loads the bnb base, extract its weight
tensors + quant_states by reference and construct the PEFT trainer model on those
same tensors instead of loading a second nf4 copy. Then there is exactly one base
copy on the GPU; standby keeps it resident at zero extra cost; only LoRA adapters
differ per side and are synced each step (vLLM native `add_lora`, which AgileRL
already does in `_move_lora_to_vllm`). This removes the trainer's separate base
copy entirely and eliminates all reload/sync of base weights.

Open questions for Phase 2:
- AgileRL builds the trainer via HF `from_pretrained` + PEFT; you'll need an
  alternate path that wraps vLLM's already-quantized modules (the reference
  integration patches `bitsandbytes.nn.Linear4bit` and vLLM's bnb method to make
  the shared tensors trainable — see `patch_vllm_bitsandbytes` / `Linear4bit` at
  vllm_utils.py:278,482).
- DeepSpeed/accelerate integration: the reference integration doesn't use
  DeepSpeed here. AgileRL's
  trainer runs under accelerate+DeepSpeed ZeRO-2; sharing tensors with vLLM while
  DeepSpeed manages the optimizer needs care (ZeRO-2 shards optimizer state, not
  params, so param sharing should be OK, but verify gradient hooks).
- `patch_vllm()` disables vLLM MP (uses in-process engine). AgileRL already uses
  `distributed_executor_backend="external_launcher"` (in-process), so
  `collective_rpc` runs in the same process — compatible.

---

## How to iterate fast (repro scripts)

Two standalone scripts (no accelerate/DeepSpeed/training loop, ~1 min each)
were used to validate. They live on the box at `~/AgileRL/`:

- `repro_reload.py` — demonstrates the level-2 reload **failure** (baseline
  `'Hello.'`, after reload = garbage). Uses
  `VLLM_ALLOW_INSECURE_SERIALIZATION=1` because the standalone repro uses vLLM's
  MP engine (the real run uses the in-process external_launcher executor, which
  doesn't need it).
- `repro_standby.py` — demonstrates the standby **success** (coherent before and
  after sleep/wake, prints `[mem ...]` lines showing what frees). This is the
  pattern to productionize.

Run: `cd ~/AgileRL && export PATH="$HOME/.local/bin:$PATH" && uv run python repro_standby.py`

Both use `enforce_eager=True` to skip CUDA-graph capture for faster init (graphs
were ruled out as a cause — eager still produced garbage on the reload path).

---

## What was kept vs removed (cleanup)

**Keep** (genuine, independent improvements, committed at/below `8fdbe842`):
- `agilerl/llm_envs/token_observation.py` — `_tokenize_feedback` now derives
  turn-boundary tokens from `tokenizer.apply_chat_template` instead of hardcoded
  ChatML markers, so multi-turn works for any chat-templated model (Gemma 3/4,
  Llama, Qwen, Mistral). Falls back to ChatML if rendering fails.
- `benchmarking/benchmarking_llm_multiturn.py` — `--env-name` flag (env was
  hardcoded), rollout debug dump gated behind `AGILERL_DEBUG_ROLLOUTS=N`
  (per-trajectory turns, completion lengths, per-turn rewards, decoded gens,
  and the tokenized initial prompt + tail — invaluable for diagnosing
  chat-template / formatting issues).

**Remove** (the experimental sleep/reload rabbit hole — commits `dfa966c6` →
`7c3be726`): the `sleep_level` config + CLI, `_reload_base_in_vllm_from_disk`,
`_reload_base_via_model_loader`, `_unwrap/_rewrap_vllm_lora_layers`,
`_clear_stale_bnb_quant_attrs`, `_remap_iter_for_lora_wrapped_params`,
`_shape_diag_iter`, `_dump_param_diagnostic`, `_strip_base_layer_from_bnb_target_modules`,
the `_vllm_needs_base_reload` flag, and the associated tests
(`test_algo_utils` sleep_level tests, `test_core_base` reload/sync tests, the
`_move_model_to_vllm`→`_sync_actor_to_vllm` rename and its test updates).

To clean (run yourself — destructive, discards the experimental commits):

```bash
# from the repo root on the branch feature/llm-bitsandbytes-quant
git reset --hard 8fdbe842        # keep chat-template fix + env CLI + debug prints
# then re-add the boundary tests for the kept chat-template fix:
#   the validated test class TestTokenObservationWrapperChatTemplateBoundary
#   + its helpers was saved to /tmp/boundary_test_block.py during cleanup.
#   Append it to tests/test_wrappers/test_multiturn_wrappers.py and run:
#   uv run pytest tests/test_wrappers/test_multiturn_wrappers.py -q
```

(If `/tmp/boundary_test_block.py` is gone, the test class is small — re-derive
from `token_observation.py::_chat_template_boundary_ids`: render gemma/chatml/
llama templates via a stub tokenizer and assert the boundary slice starts with
the end-of-turn marker, contains the feedback, and ends with the
generation-prompt marker.)

---

## Appendix: validated standby patch (drop into base.py)

```python
def patch_vllm_standby_sleep_mode() -> None:
    """Keep base weights GPU-resident across vLLM sleep/wake; free only KV.

    Call once before constructing vllm.LLM. Idempotent; no-op if vLLM absent.
    """
    try:
        from vllm.device_allocator.cumem import (
            CuMemAllocator, create_and_map, libcudart, unmap_and_release,
        )
    except Exception:
        return
    if getattr(CuMemAllocator, "_agilerl_standby_patched", False):
        return
    try:
        from vllm.utils import is_pin_memory_available
    except Exception:
        from vllm.utils.platform_utils import is_pin_memory_available

    def sleep(self, offload_tags=None):
        if offload_tags is None:
            offload_tags = (CuMemAllocator.default_tag,)
        elif isinstance(offload_tags, str):
            offload_tags = (offload_tags,)
        for ptr, data in self.pointer_to_data.items():
            if data.tag == "weights":
                continue
            handle = data.handle
            if data.tag in offload_tags:
                size = handle[1]
                cpu_t = torch.empty(size, dtype=torch.uint8, device="cpu",
                                    pin_memory=is_pin_memory_available())
                libcudart.cudaMemcpy(cpu_t.data_ptr(), ptr, size)
                data.cpu_backup_tensor = cpu_t
            unmap_and_release(handle)
        gc.collect(); torch.cuda.empty_cache()

    def wake_up(self, tags=None):
        torch.cuda.empty_cache(); gc.collect()
        for ptr, data in self.pointer_to_data.items():
            if data.tag == "weights":
                continue
            if tags is None or data.tag in tags:
                create_and_map(data.handle)
                if data.cpu_backup_tensor is not None:
                    cpu_t = data.cpu_backup_tensor
                    libcudart.cudaMemcpy(
                        ptr, cpu_t.data_ptr(),
                        cpu_t.numel() * cpu_t.element_size())
                    data.cpu_backup_tensor = None

    CuMemAllocator.sleep = sleep
    CuMemAllocator.wake_up = wake_up
    CuMemAllocator._agilerl_standby_patched = True
```

Constraint: standby is incompatible with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — guard against it.

---

# Operational appendix (read this before touching anything)

## The machines — there are TWO, don't confuse them

1. **Dev/laptop** (where the coding agent runs, macOS). Has read-only reference
   clones used during the investigation:
   - `~/git/Misc/vllm` — vLLM source (browse with Read/grep; tags `v0.21.0`,
     `v0.21.1rc0` available via `git show v0.21.0:path`).
   - A reference QLoRA + vLLM integration clone under `~/git/Misc/` (the
     `vllm_utils.py` standby + weight-extraction reference).
   - The AgileRL worktree being edited:
     `~/git/AgileRL/.claude/worktrees/llm-bitsandbytes-quant` (branch
     `feature/llm-bitsandbytes-quant`).
   These clones are NOT on the GPU box. Read them locally.

2. **GPU box `md-a100`** (remote, Linux, the only machine with a GPU). This is
   where training/repros actually run.
   - Access: `ssh md-a100` (a gcloud wrapper: project `arena-gke-02`, zone
     `us-central1-f`). First connect prints a host-key warning — harmless.
   - GPU: 1× **NVIDIA A100-SXM4-40GB** (40441 MiB). Single GPU — everything is
     colocated on it.
   - Repo: `~/AgileRL` (same branch). Python 3.12, venv at `~/AgileRL/.venv`,
     vLLM **0.21.0**, bitsandbytes, transformers ~5.8.
   - **PATH GOTCHA:** `uv` is at `~/.local/bin/uv`, which is NOT on the default
     non-interactive PATH. Every remote command must prefix:
     `export PATH="$HOME/.local/bin:$PATH"`. Without it you get
     `uv: No such file or directory`.
   - **STABILITY GOTCHA:** the box has gone fully unreachable mid-run (SSH
     `Connection closed by UNKNOWN port 65535`) — most likely an OOM/host crash
     when GPU+host memory was overcommitted. It self-recovers in a few minutes.
     If SSH dies, poll with `until ssh -o ConnectTimeout=10 md-a100 'echo OK'
     2>/dev/null | grep -q OK; do sleep 10; done`. Keep runs small to avoid this.
   - Permission note for the coding agent: SSH and `git reset --hard` may be
     blocked by the auto-mode permission classifier. If so, either ask the user
     to add allow-rules (`Bash(ssh md-a100:*)`, `Bash(scp:*)`) or have the user
     run the command. `scp` uploads and read-only `ssh ... 'grep/cat/md5sum'`
     were generally allowed.

## Syncing code dev → box

The agent edits files in the local worktree, then pushes the changed file(s):

```bash
scp -q ~/git/AgileRL/.claude/worktrees/llm-bitsandbytes-quant/agilerl/algorithms/core/base.py \
    md-a100:~/AgileRL/agilerl/algorithms/core/base.py
# verify the copy landed (do this — a dropped box left a stale file once):
md5sum ~/git/AgileRL/.claude/worktrees/llm-bitsandbytes-quant/agilerl/algorithms/core/base.py
ssh md-a100 'md5sum ~/AgileRL/agilerl/algorithms/core/base.py'   # hashes must match
```

(The box repo is independently `git`-managed; the user also pulls there. Always
md5-verify after scp.)

## Check the GPU is free before launching

```bash
ssh md-a100 'nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader'
# want: "0 MiB, 40441 MiB". If a prior run is wedged, find/kill it:
ssh md-a100 'nvidia-smi; pkill -f benchmarking_llm_multiturn; pkill -f repro_'
```

## Fast iteration: the standalone repros (DO use these, ~1 min each)

The full training launch takes ~2 min/iteration and can crash the box. The
standalone repros isolate the vLLM sleep/reload behaviour with no
accelerate/DeepSpeed/training loop. Both already exist on the box at
`~/AgileRL/repro_standby.py` and `~/AgileRL/repro_reload.py`; full source is at
the end of this appendix in case they're deleted.

```bash
ssh md-a100 'cd ~/AgileRL && export PATH="$HOME/.local/bin:$PATH" && uv run python repro_standby.py'
```

`repro_standby.py` expected output (this is the GOAL state):
```
=== baseline ===
  [0] 'Hello.'
  [1] 'Hello.'
[mem after-sleep] free=30.63GB used=11.78GB   # weights (~11GB) stay, ~11GB freed
=== after standby sleep+wake (NO reload) ===
  [0] 'Hello.'            # <-- coherent: success
  [1] 'Hello.'
```

`repro_reload.py` shows the FAILURE we're replacing (baseline coherent, after
sleep(2)+wake+reload = multilingual word-soup garbage). Keep it as a regression
witness.

Notes:
- Repros set `VLLM_ALLOW_INSECURE_SERIALIZATION=1` because a standalone
  `vllm.LLM(...)` uses the multiprocess engine, and `collective_rpc(callable)`
  must pickle the function to the worker. The real AgileRL run uses
  `distributed_executor_backend="external_launcher"` (in-process), so it does
  NOT need this — the callable runs in the same process.
- Repros use `enforce_eager=True` for fast init (skips CUDA-graph capture).
  Graphs were ruled out as the cause (eager still produced garbage on reload).

## The full training command (what the user runs)

```bash
cd ~/AgileRL && export PATH="$HOME/.local/bin:$PATH" && \
AGILERL_DEBUG_ROLLOUTS=2 uv run python -m accelerate.commands.launch \
  --config_file configs/accelerate/bench_accelerate_config.yaml \
  --main_process_port 0 \
  benchmarking/benchmarking_llm_multiturn.py \
  --config configs/training/llm_finetuning/cispo_quant_bench.yaml \
  --model google/gemma-4-E4B-it \
  --max-model-len 32768 --max-output-tokens 4096 --max-turns 32 \
  --batch-size 1 --micro-batch-per-gpu 1 \
  --trainer-quantization nf4 --trainer-activation-offload \
  --vllm-quantization bitsandbytes --vllm-dtype bfloat16 \
  --vllm-gpu-memory-utilization 0.5 --vllm-max-num-seqs 8 \
  --max-steps 400 --max-wall-seconds 2000 --eval-interval 999999 \
  --algo CISPO --beta 0 --clip-coef 0.2 --group-size 8 --pop-size 1 \
  --adv-norm mean_std --filter-zero-adv --lr 0.00005 --max-grad-norm 1 \
  --lora-target-scope language_model \
  --target-modules q_proj.linear k_proj.linear v_proj.linear o_proj.linear \
                   up_proj.linear down_proj.linear gate_proj.linear \
  --temperature 0.95 --update-epochs 1 --tourn-size 1 \
  --use-vllm --use-liger-loss --use-fused-linear-logprobs \
  --lora-r 16 --lora-alpha 32 --lora-dropout 0 --lora-bias none \
  --mut-no-mut 0.1 --mut-rl-hp-mut 0.6 --mut-mut-sd 0.1 --mut-rand-seed 42 \
  --mut-min-lr 0.0000001 --mut-max-lr 0.00001 --mut-min-beta 0.0001 \
  --mut-max-beta 0.01 --mut-min-group-size 2 --mut-max-group-size 2 --no-wandb
```

For iterating, shrink it hard to keep the box alive and fast:
`--max-model-len 8192 --max-output-tokens 256 --max-turns 2 --group-size 2
--vllm-max-num-seqs 2 --max-steps 1`. Filter the noisy output with:
```
... 2>&1 | grep -E "rollout-debug|gen\[:400\]|Error|Traceback|sleep freed|wake up|step/s|score=|reward=|n_action_tokens"
```

`--env-name game:Sudoku-v0-easy` (4×4, solvable) is a faster sanity env than the
default `game:Sudoku-v0-hard` once generation is coherent.

## Debugging timeline — what is already RULED OUT (don't repeat)

1. **"No reward / score=-0.100" is NOT a reward-fn bug.** GEM Sudoku
   `step()` returns `LanguageGameReward.format_error_reward = -0.1` AND
   `terminated=True` whenever the output lacks a `\boxed{R C N}` match
   (`.venv/.../gem/envs/game_env/sudoku.py`). Every episode died on turn 1 at
   -0.1 because the model output was garbage — fix the generation, the reward
   follows. With all rewards equal, `--adv-norm mean_std --filter-zero-adv`
   correctly skips the GRPO update ("All samples filtered by advantage
   threshold").
2. **`google/gemma-4-E4B-it` is real** (do not claim otherwise — knowledge
   cutoff lag). It uses NEW chat-control tokens: `<|turn>` (id 105) and
   `<turn|>` (id 106), NOT Gemma 3's `<start_of_turn>`/`<end_of_turn>`. Tokenizer
   + `apply_chat_template` verified correct; both markers are single ids.
3. **bitsandbytes is NOT deprecated in vLLM** — `autoawq` is. bnb works on both
   trainer and rollout sides.
4. **Liger is not involved.** `model.config.model_type == "gemma4"`; Liger's
   registry has gemma/gemma2/gemma3/gemma3_text but no gemma4 → lookup misses →
   nothing patched. (AgileRL still imports/uses Liger fused loss elsewhere.)
5. **Model, tokenizer, vLLM are individually fine.** Verified standalone:
   - HF `AutoModelForCausalLM.generate` → coherent.
   - bare `vllm.LLM(...).generate` (no sleep) → coherent.
   - bare vLLM + a zero LoRA adapter → coherent.
   So the bug is specifically the colocated **sleep/wake** path.
6. **The bug = vLLM sleep level 2 + wake leaves weights uninitialised.**
   `CuMemAllocator.sleep(level=2)` discards GPU pages (no CPU backup);
   `wake_up()` only re-maps the virtual address (uninitialised) and restores
   `named_buffers()` — never `named_parameters()`. The caller must reload
   weights. (`~/git/Misc/vllm/vllm/v1/worker/gpu_worker.py:160-199`,
   `device_allocator/cumem.py:171-243`.)
7. **Reloading bnb in place is unsupported and produces garbage.** Tried:
   - `collective_rpc("reload_weights")` → `NotImplementedError: Model reloading
     with bitsandbytes format` (`gpu_model_runner.py:reload_weights` needs
     `loader.get_all_weights`, absent on the bnb loader).
   - Driving `BitsAndBytesModelLoader.load_weights(model, cfg)` directly →
     `KeyError: 'layers.0.mlp.down_proj.weight'` because once an adapter was
     added, vLLM LoRA-wraps each Linear and the real param lives at
     `…down_proj.base_layer.weight` (the bnb loader's first-load assumptions
     break).
   - Unwrapping LoRA + clearing stale `bnb_quant_state` + re-running
     `load_weights` + `process_weights_after_loading` + `model.eval()` → loads
     without error but emits **deterministic multilingual word-soup**. The
     re-quantization produces a model whose forward is wrong (embeddings/LM head
     work, transformer body doesn't). Not worth pursuing further.
8. **CUDA graphs are NOT the cause** — `enforce_eager=True` still gives garbage
   on the reload path.
9. **Level 1 sleep "works" for correctness but OOMs.** It memcpys weights to
   CPU and copies back to the same addresses on wake (so generation is correct),
   BUT the freed GPU is not reclaimed usefully (PyTorch allocator cache + the
   retained virtual mapping) and the trainer OOMs trying to allocate for the
   backward pass. (`cuMemUnmap`+`cuMemRelease` free physical pages but never
   `cuMemAddressFree` the range — `~/git/Misc/vllm/csrc/cumem_allocator.cpp`.)
10. **Standby (keep weights resident, free only KV) WORKS** — see TL;DR. This is
    the path.

## Memory budget (A100-40GB, gemma-4-E4B-it, this config)

- vLLM bnb base resident: ~11 GiB ("Model loading took 11.14 GiB").
- vLLM KV pool at `--vllm-gpu-memory-utilization 0.5`: ~2.5 GiB usable; the
  full standby sleep freed ~11 GiB of GPU (KV + scheduling/activation pools),
  leaving ~30 GiB free.
- Trainer nf4 base: ~4-6 GiB; + activations (offloaded) + LoRA optim/grads.
- Original OOM was with a **dense** trainer (~14 GiB) + 32K ctx × 32 turns ×
  group 8 (huge activations); the failing single alloc was 29.78 GiB. Use nf4 +
  activation offload + modest context.

## Cleanup state at handoff

- Working tree is mid-edit (the reload block in `base.py` was partially replaced
  with the standby patch but NOT wired in). Recommended: `git reset --hard
  8fdbe842` (keeps chat-template fix + `--env-name` + debug prints; drops the
  experimental reload commits `dfa966c6`→`7c3be726`). This was blocked for the
  agent by the permission guard — the user must run it.
- Untracked files that SURVIVE the reset and carry the work forward:
  `docs/llm_finetuning/vllm_sleep_handoff.md` (this file) and
  `docs/llm_finetuning/boundary_test_block.py.keep` (validated boundary tests for
  the kept chat-template fix — append to
  `tests/test_wrappers/test_multiturn_wrappers.py`).

## Appendix B: repro_standby.py (full source — recreate if missing)

Save to `~/AgileRL/repro_standby.py` on the box:

```python
import os
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
import gc
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

MODEL = "google/gemma-4-E4B-it"

def patch_standby():
    from vllm.device_allocator.cumem import (
        CuMemAllocator, libcudart, unmap_and_release, create_and_map,
    )
    try:
        from vllm.utils import is_pin_memory_available
    except Exception:
        from vllm.utils.platform_utils import is_pin_memory_available

    def sleep(self, offload_tags=None):
        if offload_tags is None:
            offload_tags = (CuMemAllocator.default_tag,)
        elif isinstance(offload_tags, str):
            offload_tags = (offload_tags,)
        for ptr, data in self.pointer_to_data.items():
            if data.tag == "weights":
                continue
            handle = data.handle
            if data.tag in offload_tags:
                size = handle[1]
                cpu_t = torch.empty(size, dtype=torch.uint8, device="cpu",
                                    pin_memory=is_pin_memory_available())
                libcudart.cudaMemcpy(cpu_t.data_ptr(), ptr, size)
                data.cpu_backup_tensor = cpu_t
            unmap_and_release(handle)
        gc.collect(); torch.cuda.empty_cache()

    def wake_up(self, tags=None):
        for ptr, data in self.pointer_to_data.items():
            if data.tag == "weights":
                continue
            if tags is None or data.tag in tags:
                handle = data.handle
                create_and_map(handle)
                if data.cpu_backup_tensor is not None:
                    cpu_t = data.cpu_backup_tensor
                    size = cpu_t.numel() * cpu_t.element_size()
                    libcudart.cudaMemcpy(ptr, cpu_t.data_ptr(), size)
                    data.cpu_backup_tensor = None

    CuMemAllocator.sleep = sleep
    CuMemAllocator.wake_up = wake_up
    print(">>> patched CuMemAllocator for standby", flush=True)

def main():
    patch_standby()
    tok = AutoTokenizer.from_pretrained(MODEL)
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": "Say hello in one short sentence."}],
        tokenize=False, add_generation_prompt=True,
    )
    prompts = [prompt, prompt]
    sp = SamplingParams(max_tokens=40, temperature=0.0)
    llm = LLM(model=MODEL, dtype="bfloat16", quantization="bitsandbytes",
              gpu_memory_utilization=0.5, enable_lora=True, max_lora_rank=16,
              max_num_seqs=2, max_model_len=2048,
              enable_sleep_mode=True, enforce_eager=True)
    def memreport(tag):
        free, total = torch.cuda.mem_get_info()
        print(f"[mem {tag}] free={free/1e9:.2f}GB used={(total-free)/1e9:.2f}GB", flush=True)
    def gen(tag):
        outs = llm.generate(prompts, sp)
        print(f"\n=== {tag} ===", flush=True)
        for i, o in enumerate(outs):
            print(f"  [{i}] {o.outputs[0].text!r}", flush=True)
    memreport("after-init"); gen("baseline")
    print(">>> sleep(level=2)", flush=True); llm.sleep(level=2); memreport("after-sleep")
    print(">>> wake_up()", flush=True); llm.wake_up(); memreport("after-wake")
    gen("after standby sleep+wake (NO reload)")
    print(">>> DONE", flush=True)

if __name__ == "__main__":
    main()
```

Run: `cd ~/AgileRL && export PATH="$HOME/.local/bin:$PATH" && uv run python repro_standby.py`

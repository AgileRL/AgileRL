# ZeRO-3 distributed-training inventory

Merge base of `feature/zero3-support` and `main` (or `origin/main`):
`0692d4ae099bd7f74518aa51474d3aa67150045c`.

Tip inventoried: `feature/zero3-support` at
`b8c4712df6e191e99c1ea2228a22f148bb54bcfc`.

Scope: every change on the branch that affects distributed training behaviour
(parameter gather/release, offload, mixed precision, optimizer construction,
checkpointing, DP lockstep, weight sync to inference). Non-distributed-only
edits (pad-token resolution, chat-template, DQN tutorial vectorisation, Codecov
coverage-only tests, temporary adapter-load diagnostics that were later removed)
are omitted unless they touch a ZeRO/DP path.

| commit SHA | file:line | what it does | why (from commit message or code) | classification | notes |
| --- | --- | --- | --- | --- | --- |
| `5e1fa4d8` | `agilerl/utils/llm_utils.py:700-714` | `get_lora_params` returns only params whose names contain `"lora"` for scoped gathers | Gathering the full model OOMs at 30B; save/export/copy only need adapters | ALGORITHMIC | Memory strategy; carry “gather only touched params” into FSDP2 |
| `5e1fa4d8` | `agilerl/algorithms/core/base.py:2911` | LoRA-only `save_pretrained` gathers `get_lora_params` instead of all params | Same OOM / corruption concern | ALGORITHMIC | Mechanism still `gather_if_zero3` → `GatheredParameters` |
| `5e1fa4d8` | `agilerl/algorithms/core/base.py:3065-3075` | Checkpoint `load_state_dict` gathers only keys present in the saved dict, `modifier_rank=0` | Ungathered / multi-rank adapter writes silently corrupt under ZeRO-3 scatter | DS-API | API: `deepspeed.zero.GatheredParameters(..., modifier_rank=0)` — only rank 0’s writes persist |
| `5e1fa4d8` | `agilerl/algorithms/core/base.py:3489` | Tournament/export adapter save scoped to LoRA params | Avoid full-model gather | ALGORITHMIC | Same pattern as save path |
| `5e1fa4d8` | `agilerl/algorithms/core/base.py:3583-3586` | Gather `v_head` params before copying into a clone | Value-head params may be ZeRO-partitioned | DS-API | `GatheredParameters` around `load_state_dict` |
| `5e1fa4d8` | `agilerl/algorithms/core/base.py:3693-3694` | Same gather when cloning population member `v_head` | Same | DS-API | |
| `5e1fa4d8` | `agilerl/algorithms/core/base.py:5262` | vLLM rollout adapter export gathers LoRA params only | Full gather OOMs; PEFT filters base keys anyway | ALGORITHMIC | Still DS gather mechanism |
| `5e1fa4d8` | `agilerl/algorithms/core/base.py:5926-5935` | `_load_adapter_weights` gathers LoRA params with `modifier_rank=0` | Only modifier-rank writes survive scatter back to shards | DS-API | `GatheredParameters` + `modifier_rank` |
| `5e1fa4d8` | `agilerl/algorithms/core/base.py:5983-5985` | `_copy_adapter_weights` gathers source+target LoRA params with `modifier_rank=0` before `copy_` | Ungathered adapter copies silently corrupt the reference policy | DS-API | Critical for reference/critic sync under shard |
| `5e1fa4d8` | `agilerl/algorithms/core/base.py:6096-6105` | Another adapter-load gather scoped to LoRA + `modifier_rank=0` | Same scatter-write rule | DS-API | |
| `5e1fa4d8` | `agilerl/utils/algo_utils.py:2545-2630` | `clone_llm` no longer wraps construction in a full-model `gather_if_zero3` | Drop unnecessary gather from clone (commit message) | ALGORITHMIC | Clone builds a fresh module then loads a provided state dict; full gather was waste |
| `d23d9c5d` / `c9c23c2f` / `3929c9e7` | `agilerl/utils/llm_utils.py:658-697` | Add `gather_if_ds_param`: gather tensors that have `ds_id` for a block; default `modifier_rank=0` | External matmuls see empty partitioned shards; must wrap matmul/apply only, not `module.forward` (post-forward hooks re-partition) | DS-API | API: `deepspeed.zero.GatheredParameters` via `gather_if_zero3` |
| `293f6181` | `agilerl/utils/llm_utils.py:62-65`, `687-691` | Skip gather when `ds_status == ZeroParamStatus.AVAILABLE` (tied embeddings already owned by `embed_tokens`) | Harden ZeRO-3 lm_head gather; avoid double-gather / noop on already-available params | DS-API | API: `ZeroParamStatus` from `deepspeed.runtime.zero.partition_parameters` |
| `5dc87bd1` | `agilerl/utils/llm_utils.py:62` | Always initialize `ZeroParamStatus = None` sentinel when DeepSpeed absent | Codecov / import-safety | DS-API | Same API surface; coverage-motivated init only |
| `c9c23c2f` | `agilerl/algorithms/core/llm_ops/fused_logprobs.py:179-196` | Wrap fused chunked logprob matmul loop in `_gather_if_ds_param(lm_head_weight, lm_head_bias)` | Wrapping `actor.forward` left an empty shard for the fused path after post-forward re-partition | DS-API | Gather must outlive every chunk of the loop |
| `c9c23c2f` / `293f6181` | `agilerl/algorithms/core/llm_ops/fused_logprobs.py:265-275` | Backward of `FusedLinearLogProbsFunction` re-gathers lm_head; raises if `needs_weight_grad` and param has `ds_id` | Forward gather has exited and ZeRO-3 re-partitioned `.data`; full-finetune of lm_head through this path unsupported | DS-WORKAROUND | Limitation: ZeRO-3 param `.data` swap after gather exit; FSDP2 may allow a different rematerialisation story — needs verification |
| `3929c9e7` | `agilerl/algorithms/core/base.py:3979-3988` | `_liger_head_gather` returns `gather_if_ds_param(lm_head.weight, bias)` | Liger computes grads inside forward and only saves them; gathering for `apply` is enough. Drop stage-3 Liger disable | DS-API | Used by GRPO/PPO/REINFORCE/DPO/SFT Liger paths |
| `3929c9e7` | `agilerl/algorithms/grpo.py:1892`; also `ppo_llm.py:1371`, `reinforce_llm.py:1037`, `dpo.py:613`, `sft.py:351` | Wrap `LigerFusedLinear*Function.apply(...)` in `with self._liger_head_gather()` | Same | DS-API | |
| `d23d9c5d` / later | `agilerl/algorithms/core/base.py:5729`, `5784`; `grpo.py:1878`; `ppo_llm.py:1390`; `reinforce_llm.py:1056` | Vocab size for chunking uses `getattr(lm_head_weight, "ds_shape", shape)[0]` | Partitioned shard `.shape[0]` is not the real vocab under ZeRO-3 | DS-API | API: DeepSpeed param attribute `ds_shape` |
| `efb073f1` / `769c416c` | `agilerl/algorithms/core/llm_ops/fused_logprobs.py:22-45`, `179-196` | `fp32_lm_head_operands`: one hoisted fp32 `(V,H)` copy inside the gather; chunk loop upcasts hidden and matmuls in fp32 | Cast to fp32 before matmul to avoid NaNs; upcast once per loop (gigabyte copy must not repeat per chunk) | ALGORITHMIC | Numerical stability + memory; must stay inside gather under ZeRO-3 |
| `1282dec8` | `agilerl/algorithms/core/base.py:2721-2727` | Force-disable `use_memory_efficient_params` when `zero_stage == 3` | Naive `.to("cpu")` before vLLM wake breaks DeepSpeed parameter status and leaves embed weights on CPU for the next learn forward | DS-WORKAROUND | Limitation: ZeRO-3 in-place shard metadata incompatible with manual CPU offload of trainer params |
| `1282dec8` | `agilerl/algorithms/core/base.py:6384-6386`, `6411` | `_memory_efficient_params` / prepare-for-generation skip offload when stage 3 | Warning moved to init; runtime path still gated | DS-WORKAROUND | Same limitation |
| `be715bfe` / `d8d8b878` | `agilerl/algorithms/core/base.py:4405-4436`; `agilerl/utils/algo_utils.py:2598-2616` | Under ZeRO-3 (non-quantized), `autocast_adapter_dtype=False` and cast LoRA params to bf16 | PEFT’s default fp32 adapter upcast mixes dtypes in persistent-param allgather and fails on DeepSpeed 0.19.2 | DS-WORKAROUND | Limitation: DeepSpeed 0.19.2 coalesced allgather requires homogeneous dtype across gathered params |
| `311b3a50` | `agilerl/algorithms/core/base.py:4581-4615` | `_amp_ctx` pairs bf16 autocast with `_lora_input_cast_ctx` that disables PEFT `cast_input_dtype_enabled` | Stop paying for PEFT’s fp32 LoRA input cast under autocast (measured ~20% peak VRAM drop; grads bitwise identical) | ALGORITHMIC | Throughput/memory; fp32 *weights* remain load-bearing for Adam; only activation cast skipped while autocast is on |
| `311b3a50` | `agilerl/algorithms/core/llm_ops/fused_lora.py:80` (`_lora_delta` via `_cast_input_dtype`) | Fused LoRA delta respects the same cast switch | Open-coded cast previously ignored the switch | ALGORITHMIC | |
| `7b5e1d6a` | `agilerl/algorithms/core/base.py:3322-3329`, `3365`, `3374` | `_gradient_checkpointing_kwargs` → `{use_reentrant: zero_stage==3}` | Non-reentrant checkpointing rejects frozen LoRA base params after ZeRO-3 re-partitions them to empty shards during activation recompute | DS-WORKAROUND | Limitation: ZeRO-3 + non-reentrant HF checkpoint metadata check on empty shards |
| `a9407c1d` | `agilerl/algorithms/core/base.py:4938` | Fused reference pass uses `torch.no_grad()` instead of `torch.inference_mode()` | ZeRO-3 reallocates `param.data` during forward; tensors allocated under inference mode can never have `requires_grad` re-enabled, breaking adapter switching in the next update epoch | DS-WORKAROUND | Limitation: interaction of ZeRO-3 `.data` swap with inference-mode tensor bans |
| `efdcbecb` | `agilerl/algorithms/core/base.py:3160-3173` | After adapter load, `_refresh_deepspeed_master_weights` calls `optimizer.refresh_fp32_params()` | DeepSpeed snapshots fp32 master copies at engine build and writes them back over shards every step; live adapter writes are otherwise clobbered on first step | DS-API | API: DeepSpeed optimizer `refresh_fp32_params` |
| `f47d243e` | `agilerl/algorithms/core/base.py:207`, `1225-1226`, restore skip-list | `LOCAL_AGENT_ATTRIBUTES = {"device"}` skipped on checkpoint attribute restore | Checkpoint records writer’s device; population member resume was relocated onto that device while models stayed put | ALGORITHMIC | Multi-device / population correctness; not ZeRO-specific but on this branch’s checkpoint path |
| `fc04b4e3` | `agilerl/utils/llm_utils.py:406-407` | `needs_cross_rank_seq_padding` also true when `zero_stage == 3` | ZeRO-3 parameter gathers require identical per-rank `T` so every rank issues the same NCCL collectives | ALGORITHMIC | Carry lockstep requirement to FSDP2 full-shard / any collective-heavy path |
| `fc04b4e3` | `agilerl/algorithms/grpo.py:600-613`, `623-624`, `657-658` | Assert matching completion `T` across ranks before `learn()`; `wait_for_everyone` on early-return skip paths | Harden GRPO for ZeRO-3 DP lockstep | ALGORITHMIC | Early-return barriers prevent NCCL hang when peers still in collectives |
| `2221bab3` | `agilerl/algorithms/grpo.py:541-552`, `681` | `_raise_if_loss_not_finite_on_any_rank` via `allreduce_minmax_int` before backward | Local-only nonfinite raise desyncs peers into ZeRO-3 NCCL hangs | ALGORITHMIC | Same class of lockstep fix; needed under any sharded DP with collective backward |
| `f96f521c` | `agilerl/utils/llm_utils.py:512-561` | `align_completion_batch_shapes_across_ranks`: reduce `B`/`T` *before* local pad/stack; barrier after pad | Pad-then-gather left peers waiting in collectives under long multiturn | ALGORITHMIC | Ordering of metadata collective vs local work |
| `9c252382` | `agilerl/algorithms/core/base.py:4406-4423` | Gather expert `target_parameters` shards around `get_peft_model` attach under ZeRO-3 | PEFT reads targeted parameter shapes; under `zero.Init` they are placeholder shards | DS-API | `GatheredParameters` / `gather_if_zero3` around PEFT attach |
| `9c252382` | `agilerl/algorithms/core/base.py:4458-4465` | Re-gather expert params for `upgrade_moe_param_wrappers`; then `mark_expert_wrappers_as_zero3_leaves` | Convention checks need full shapes; leaf mark avoids per-parameter gather storms | DS-API | |
| `9c252382` | `agilerl/algorithms/core/llm_ops/moe_lora.py:381-395` | `mark_expert_wrappers_as_zero3_leaves` → `deepspeed.utils.set_z3_leaf_modules` | Whole subtree (adapters + packed expert weights) gathered at wrapper pre-forward | DS-API | API: `deepspeed.utils.set_z3_leaf_modules` |
| `9c252382` | `agilerl/algorithms/core/llm_ops/moe_lora.py:97-110`, `340-348` | `_is_partitioned` blocks only when `ds_status != AVAILABLE`; raise if split forward sees ungathered shards | Inside a leaf forward, raw reads see full tensors; guard is for misconfiguration | DS-API | Depends on ZeRO leaf gather semantics |
| `5d2e0d85` | `agilerl/algorithms/core/llm_ops/moe_lora.py` (module) | Split expert LoRA: per-expert rank-`r` slices instead of materializing full `B@A` delta | PEFT default allocates expert-weight-sized delta every forward | ALGORITHMIC | Real memory/throughput strategy for packed MoE LoRA; must carry under FSDP2 |
| `5d2e0d85` | `agilerl/utils/llm_utils.py:2493-2526`, `2575+` | `expert_lora_vllm_key_map` + filter/save path for vLLM fused-MoE LoRA layout | Rollout sync must export under nested-wrapper layout vLLM parses | ALGORITHMIC | Weight-sync contract with inference engine; independent of DeepSpeed once tensors are materialized |
| `5d2e0d85` / `9c252382` | `agilerl/algorithms/core/llm_ops/vllm_colocate.py:133-153` | `patch_vllm_3d_moe_lora_flag` sets `is_3d_moe_weight` on vLLM model class before engine construction | Without it `add_lora` rejects stacked-3D adapters | ALGORITHMIC | Inference-engine compatibility |
| `5d2e0d85` | `agilerl/algorithms/core/llm_ops/vllm_colocate.py:187` | Keep-resident patch also matches `w13_lora_b_stacked` / fused-MoE slots | Cover fused-MoE LoRA GPU slots | ALGORITHMIC | Colocated weight residency |
| `5d2e0d85` | `agilerl/algorithms/core/base.py:6200` | Call `patch_vllm_3d_moe_lora_flag` when `target_parameters` set | Wire MoE LoRA flag into `_configure_vllm` | ALGORITHMIC | |
| `5d2e0d85` | `agilerl/algorithms/core/base.py:4665-4671`; `fused_lora.py:193` `adapter_aligned_chunks` | Micro-batches never straddle an adapter-run boundary for packed-experts LoRA | Expert-sorted rows / one adapter per forward | ALGORITHMIC | Training-loop correctness with fused routing |
| `f78354e2` | `agilerl/utils/algo_utils.py:130-140` | `polyak_update` via `torch._foreach_lerp_` | Fuse per-parameter Polyak loop (~11× on small nets) | ALGORITHMIC | Throughput; not ZeRO-specific. Under Accelerate/DeepSpeed target nets may still be local modules |
| `f78354e2` | `agilerl/utils/algo_utils.py:143-165` | `adam_kwargs` sets `fused=True` when CUDA, not capturable, **and `accelerator is None`** | Enable fused CUDA Adam where safe | ALGORITHMIC | Explicitly *disabled* under Accelerate/DeepSpeed; relevant for post-Accelerate FSDP2 optimizer construction |
| pre-existing (`d6a159e7`) | `agilerl/utils/llm_utils.py:624-654` | `gather_if_zero3` → `deepspeed.zero.GatheredParameters` when `zero_stage==3` | Core gather/release helper all scoped paths call | DS-API | Not introduced on this branch (`DPO + zero3 distribution (#445)`), but every new gather call site depends on it |

## Gaps / not found on this branch

- **Gradient accumulation boundary handling**: no commits in
  `0692d4ae..feature/zero3-support` touch `is_gradient_accumulation_boundary` or
  equivalent DeepSpeed boundary APIs (`grep` over `agilerl/` finds none).
  Behaviour remains whatever Accelerate/DeepSpeed already did on `main`.
- **DeepSpeed config offload knobs** (`offload_optimizer` / `offload_param` in
  YAML): no config diff on this range; the only offload behaviour change is
  disabling AgileRL’s colocated trainer CPU offload under stage 3 (`1282dec8`).
- Temporary adapter-load diagnostic commits (`0353646c`, `dade84de`, `35a4f2ba`,
  `ce5f0faf`) left no lasting training behaviour beyond `efdcbecb`’s master-weight
  refresh.

## Highest risk items

1. **`efdcbecb` — `_refresh_deepspeed_master_weights` / `refresh_fp32_params`**
   (`base.py:3160-3173`). Least confident that the FSDP2 analogue is “do
   nothing”: FSDP2 + torch optimizers may keep a single param object, or may use
   a different master-weight / mixed-precision optimizer path. Need to know
   whether adapter loads into an already-wrapped FSDP2 module are visible to the
   next `optimizer.step` without an explicit refresh, and which optimizer (torch
   AdamW vs foreach/fused vs bitsandbytes) the migration will use.

2. **`c9c23c2f`/`293f6181` — fused logprob backward re-gather + hard error on
   trainable ZeRO-3 `lm_head`** (`fused_logprobs.py:265-275`). Unclear whether
   FSDP2’s all-gather / reshape semantics during `autograd.Function` backward
   match DeepSpeed’s “gather exits → `.data` is an empty shard” model. Need a
   parity test with LoRA-frozen vs trainable head under FSDP2 full shard before
   dropping or rewriting the gather.

3. **`9c252382` — `set_z3_leaf_modules` for MoE expert wrappers**
   (`moe_lora.py:381-395`). Leaf-module gather is a DeepSpeed-specific scheduling
   hook; FSDP2’s closest tools (e.g. custom `fully_shard` boundaries / ignored
   params / manual all-gather) are not verified here. Need confirmation of the
   intended FSDP2 unit of gather for packed 3D expert weights plus attached LoRA
   Linears, and whether raw weight reads inside the split forward remain valid
   without an explicit context manager.

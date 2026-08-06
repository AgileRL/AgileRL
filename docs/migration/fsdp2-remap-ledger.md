# FSDP2 migration remap ledger (Phases 5–7)

Worktree: `/home/mike/wt-fsdp` (`migration/fsdp`). DeepSpeed removed from package
deps and `agilerl/`; LLM tests retargeted to `dist_mode_factory` / `FSDPConfig`.

## ALGORITHMIC rows → survival locations

| Inventory theme | Current location (approx.) |
| --- | --- |
| Scoped LoRA gathers (`get_lora_params`) | `agilerl/utils/llm_utils.py:774+`, save/load in `base.py` via `gather_params` |
| LoRA-only `save_pretrained` scoped gather | `agilerl/algorithms/core/base.py` checkpoint save paths |
| Tournament/export adapter save scoped to LoRA | `base.py` export / tournament paths |
| `clone_llm` without full-model gather | `agilerl/utils/algo_utils.py:2545+` |
| Cross-rank seq padding / lockstep (`needs_cross_rank_seq_padding`, `align_completion_batch_shapes_across_ranks`) | `llm_utils.py:413+`, `519+`; `training/llm/reasoning.py`, `multiturn.py`; `grpo.py:600+` |
| Nonfinite loss all-rank check | `grpo.py:543+` |
| Split expert LoRA (memory strategy) | `moe_lora.py` (`SortedExpertsLoraWrapper`, `RoutedExpertsLoraWrapper`, `upgrade_moe_param_wrappers`) |
| MoE expert gather around PEFT attach | `base.py:4087–4120` (`gather_params(expert_params)`) |
| vLLM MoE LoRA key map / export | `llm_utils.py:2444+`; `vllm_colocate.py` |
| `patch_vllm_3d_moe_lora_flag` | `vllm_colocate.py:133+`; wired from `base.py` |
| `adapter_aligned_chunks` micro-batch boundaries | `fused_lora.py:193+`; used from `base.py` |
| `_lora_input_cast_ctx` / PEFT cast disable under autocast | `base.py` AMP helpers |
| `use_memory_efficient_params` colocated offload | `base.py:2715+`; algorithm kwargs in GRPO/PPO LLM |
| `LOCAL_AGENT_ATTRIBUTES` checkpoint skip | `base.py:219`, `1232` |
| `polyak_update` / fused foreach | `algo_utils.py:132+` (classic RL target nets) |
| Liger head gather wrapper | `base.py:3544+`; `grpo.py`, `dpo.py`, `sft.py`, `ppo_llm.py`, `reinforce_llm.py` |

## DS-API / DS-WORKAROUND remaps

| Original DS mechanism | FSDP2 / torch-native remap | Notes |
| --- | --- | --- |
| `GatheredParameters` / `gather_if_zero3` | `gather_params`, `gather_if_ds_param` (`llm_utils.py:637+`) | DTensor `full_tensor()` + module param swap; read-only during gather |
| `modifier_rank=0` writes | All ranks gather; writes during gather are discarded — use `load_full_state_dict` for writes | Documented in `gather_params` warning |
| `set_z3_leaf_modules` / `mark_expert_wrappers_as_zero3_leaves` | **Removed** — explicit `gather_params(expert_params)` at attach + upgrade | **Open risk:** `docs/migration/blockers.md` inventory risk #3 |
| `_refresh_deepspeed_master_weights` / `refresh_fp32_params` | **Removed** — torch optimizers on sharded params | Verify adapter load visibility on first step under FSDP2 |
| ZeRO-3 `ds_shape` vocab sizing | Plain `lm_head.weight.shape` / local tensor shape | No `ds_shape` attribute |
| Fused logprob ZeRO gather + lm_head finetune block | Fused logprob backward without DS re-gather (`fused_logprobs.py` Fable-aligned) | Full lm_head finetune under FSDP2 still needs verification |
| ZeRO-3 stage gates (`zero_stage==3`) | Removed; FSDP2 uses `is_distributed()` / `fsdp_config` | `needs_cross_rank_seq_padding` uses `world_size > 1` |
| DeepSpeed optimizer in config | Removed — `torch.optim.AdamW` (+ optional fused/foreach) | Classic RL still uses Accelerate `prepare` |
| Accelerate + DeepSpeed plugin on LLM path | Removed — `init_distributed`, `fully_shard`, `agilerl.utils.distributed` helpers | `torchrun` bootstrap |

## NEW-UNCOVERED LLM test handling

- **LLM algorithm tests** (`test_grpo.py`, `test_dpo.py`, `test_sft.py`, `test_ppo_llm.py`, `test_reinforce_llm.py`, `test_vllm.py`): aligned with Fable — `dist_mode_factory` (`None` / `"dist"` / `"fsdp2"`), no `Accelerator` / `DeepSpeedEngine` mocks; `pytest.importorskip("vllm")` where needed.
- **`conftest.py` (test_llms)**: vLLM + `torch.distributed` cleanup only (no `deepspeed.comm`).
- **`test_core_base.py`**: DeepSpeed checkpoint E2E/spy branches replaced with distributed checkpoint tests (`llm_distributed_checkpoint`, `TestLLMDistributedCheckpointSaveLoad`).
- **`test_train_llm.py`**: no `Accelerator` in finetune loops; mocks use `accelerator=None`.
- **`test_moe_lora.py`**: removed `test_mark_expert_wrappers_as_zero3_leaves`; `_is_partitioned` tests use DS status mocks for adapter path only.
- **`test_fused_logprobs.py`**: ZeRO-3 gather mocks removed (Fable-aligned).
- **HPO / utils**: `HAS_DEEPSPEED` gates → `HAS_LLM_DEPENDENCIES` (+ `HAS_VLLM` where needed).
- **`subprocess_runner.py`**: `setup_distributed_env` + `dist_mode_factory` fixture resolution.
- **Root `conftest.py`**: `deepspeed_env` → `distributed_env`; dropped `ACCELERATE_USE_DEEPSPEED`.

Tests still gated on optional extras (`agilerl[llm]`, vLLM) must import without those packages installed.

## Accelerate: what remains vs removed

| Still uses Accelerate (classic RL) | Removed from LLM path |
| --- | --- |
| `EvolvableAlgorithm` / `MultiAgentRLAlgorithm` in `base.py` | `LLMAlgorithm` spawn / training |
| `training/train_*.py`, `training/trainer.py` (`LocalTrainer` classic) | `training/llm/*` (reasoning, preference, sft, multiturn) |
| Population / tournament classic paths | LLM `finetune_llm_*` functions |
| Tests for classic agents (`test_core_base` Evolvable sections) | LLM algo tests, `test_train_llm.py` |

## Done-check grep summary (post Phase 5–7)

```text
# deepspeed imports in agilerl / demos / configs / pyproject.toml / tests
→ clean (migration markdown docs retain historical references)

# accelerate on LLM algorithm modules + llm_utils + distributed + demos/llm
→ clean on grpo/dpo/sft/ppo_llm/reinforce_llm/llm_utils/distributed/demos/llm
→ base.py still imports Accelerate for EvolvableAlgorithm (shared module)

# dist.get_rank / LOCAL_RANK outside bootstrap
→ llm_utils: clean
→ algorithms: vLLM colocate synthesises rendezvous env in base.py:5773+ (bootstrap)
→ distributed.py: bootstrap/logging only (expected)
```

## Known remaining test gaps

- GPU integration tests require `agilerl[llm]` + CUDA (+ vLLM for colocated tests); not run in this pass.
- MoE split-LoRA under real `fsdp_config` + `torchrun` not verified here — see `blockers.md` risk #3.
- `test_llm_utils.py` / `test_llm_envs.py` still use plain `Accelerator()` for dataset/env fixtures (non-DeepSpeed); not on the LLM training path.
- `uv.lock` still lists `deepspeed` until lockfile refresh (`uv lock`).

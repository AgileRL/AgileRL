# FSDP2 rebase onto main — drop/keep ledger

Rebase of `migration/fsdp` onto `origin/main` (`fec569af`, v2.9.1). The migration
deletes the DeepSpeed/ZeRO-3 backend. This ledger records what the rebase drops
deliberately, what it re-lands as follow-up commits, and what was verified already
present in FSDP2 form, so the rebase can be reviewed as "nothing silently lost".

## Dropped — DeepSpeed/ZeRO-3-only, no FSDP2 port

- [ ] `agilerl/utils/zero3_patches.py` — ds_persist / fetch-trace / `release_sub_module` patches (deepspeed 0.19.3 internals)
- [ ] ZeRO-3 leaf-module marking (`mark_expert_wrappers_as_zero3_leaves`)
- [ ] ZeRO-3 semantics of `gather_if_zero3` / `gather_if_ds_param` (FSDP2 equivalents exist: `gather_params` / `materialize_dtensors`)
- [ ] DeepSpeed fp32 master-weight refresh after adapter load (no FSDP2 master snapshot exists; covered by a new staleness test instead)
- [ ] `accelerator.*` call sites (replaced by `agilerl/utils/distributed.py` primitives)
- [ ] `deepspeed` dependency in `pyproject.toml`
- [ ] `agilerl/utils/patching.py` (ZeRO-3 hook resolution primitives; FSDP2 keeps the inline `_agilerl_kernel_opts_patched` flag)

## Keep-list — re-landed as individual commits after the rebase

| # | Item | Source | Mechanism | Status |
|---|------|--------|-----------|--------|
| 1 | Dist-env reset between tests (`WORLD_SIZE`/`RANK`/etc.) | `ae02ec5e` | hand-port into `tests/conftest.py` | pending |
| 2 | SM90+ flex-decode fix (`flex_decode_kernel_options`) | #643 via `001572e9` | hand-port into `patch_flex_attention_kernel_options` | pending |
| 3 | `aux_metric_name` registration fix (no `train/mean_kl = NaN`) | `001572e9` | hand-port in `grpo.py` | pending |
| 4 | Multi-process zero-advantage masking | #660 `8c30a100` | cherry-pick, adapt `accelerator` -> `get_world_size()`/`barrier()` | pending |
| 5 | `loss_norm="accumulation_window"` | `001572e9` | hand-port + lift `test_grpo_loss_norm.py` | pending |
| 6 | `mini_batch_size` optimizer-step sizing | #649 via `001572e9` | hand-port + lift `docs/llm_finetuning/batch_sizing.rst` | pending |
| 7 | Nemotron-H Liger/Mamba package | #651 via `001572e9` | lift `agilerl/architectures/` + tests; re-gate install hook on model identity | pending |
| 8 | Manifest keys `turn_advantage_trajectory_fallback` / `strict_chat_template_boundary` | `001572e9` | hand-port | pending |
| 9 | #620 hot-path fusion divergence (`f78354e2` vs `62d52677`) | both sides | manual reconcile | pending |

## Verified already present in FSDP2 form — no action

- `attention_mask_from_padded_ids` (learn-time masks cover trailing pad run)
- EOS append in SFT/preference collates (`llm_envs/sft.py`, `llm_envs/preference.py`)
- Pad-candidate aliasing `eos_token_id` rejection (`llm_utils._coerce_distinct_pad_id`)
- Non-finite-loss DP guard (`GRPO._raise_if_loss_not_finite_on_any_rank` via `allreduce_minmax_int`)
- Cross-rank sequence padding (`needs_cross_rank_seq_padding`, `align_completion_batch_shapes_across_ranks`)
- PEFT fp32 LoRA input-cast suppression under autocast (#623; `_lora_input_cast_ctx`)
- Colocated rollout overhead fixes (#638; `move_params_to_cpu` device-type fix, `_vllm_awake` offload guard)
- `compare_responses` pad-id-0 handling

## Verification gates (must pass before push)

- [ ] Full `uv run pytest` suite green, plus `-m "gpu or vllm"` subset
- [ ] Audit: `git diff origin/main...HEAD -- agilerl/ tests/` deletions minus DeepSpeed-symbol lines is empty or explained above
- [ ] New test: LoRA adapter load then optimizer step under FSDP2 + `CPUOffloadOptimizer` serves no stale state
- [ ] ep=1 expert-LoRA attach does not full-gather packed experts (#658 intent)
- [ ] ConstantTarget smokes re-run (`docs/migration/smoke/RESULTS-*.md`)

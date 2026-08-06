89

# Fable review: `feature/tool-masking-fsdp`

Refs used:
- `feature/tool-masking-fsdp` HEAD: `e4fa46446bb8ef25054e293f9a2f255804d988c0`
- `feature/zero3-support` HEAD: `b8c4712df6e191e99c1ea2228a22f148bb54bcfc`
- Merge base (MB): `ab7eb1fc948a33143eadf31a9acf38a0a2d6bfea` (`Enable Ruff linting on tests and fix violations (#564)`)
- NEW-UNCOVERED accelerate call sites on zero3: **89** (first line of this file).

---

## PART 1 — What `feature/tool-masking-fsdp` did

### Separate concerns first

This branch is a merge of two unrelated lines of work:

1. **Tool masking / OpenEnv llm_envs** (origin: `feature/tool-aware-token-masking`). Tool-schema injection into the chat template (`tools=`), generation-provenance masking via `turn_boundaries`, and the subsequent OpenEnv / RolloutEnv / DatasetEnv taxonomy. Unrelated to the FSDP2 / Accelerate migration.
2. **FSDP2 + Accelerate removal** (origin: `deepspeed-fsdp` / PR lineage around `1f357a25` → `a1492c13`). Native `torch.distributed`, optional FSDP2 sharding, complete removal of HF Accelerate.

They were joined by merge `c9f420a9` (*Merge deepspeed-fsdp into feature/tool-aware-token-masking*). Only FSDP2 / ACCELERATE-REMOVAL / MIXED commits feed the migration. **Do not carry tool-masking changes into the migration branch.**

### Tool-masking status on zero3 / main

- Core Phase 0 commits `36f612e1` and `165a11de` are **not** ancestors of `feature/zero3-support` or `origin/main` (`git merge-base --is-ancestor` exits 1).
- `tests/test_wrappers/test_token_observation_tools.py` exists on this branch HEAD but is absent from both `feature/zero3-support` and `origin/main`.
- On zero3, `agilerl/llm_envs/token_observation.py` still has `turn_boundaries` but **no** `tools=` chat-template injection (contrast Fable `agilerl/llm_envs/rollout_env.py:293-294,385-386`).

**Must not be lost:** the tool-masking / OpenEnv work is a separate product track. Land it on a dedicated branch off `feature/zero3-support` (or off `main` after zero3 merges) — never into the FSDP2 migration branch.

### Commit classification (`MB..HEAD`, 110 commits)

#### FSDP2 (1)

| SHA | subject |
| --- | --- |
| `feb1505d` | Convert LLM test suite to torch-native accelerator modes |

#### ACCELERATE-REMOVAL (6)

| SHA | subject |
| --- | --- |
| `60637ffd` | Cover remaining native-distributed branches flagged by codecov |
| `542f36c3` | Compare gather_tensor results on CPU in device-mocked tests |
| `96521255` | Use compound cpu:gloo,cuda:nccl backend for distributed init |
| `1ed1f8f0` | Retry distributed test rendezvous on EADDRINUSE port races |
| `ed8edc77` | Skip real gloo process-group tests on Windows |
| `844a5ccd` | Make consolidate_mutations test hermetic against leaked torch.distributed |

#### MIXED — flagged; hardest to lift cleanly (14)

| SHA | subject |
| --- | --- |
| `e4fa4644` | Merge feature/arg-fix: per-rank vLLM wake/sleep and LoRA staging for DP |
| `c9f420a9` | Merge deepspeed-fsdp into feature/tool-aware-token-masking |
| `fdf8dcb7` | train_llm: guarded teardown, rank-decorrelated seeding, per-epoch KL refresh, multi-episode eval |
| `b4329aae` | Tidy up now that loras are saved per-actor for DP |
| `ed09f3ae` | Update test_core_base.py |
| `67767192` | Wake and sleep colocated vLLM on every rank |
| `d82c09ce` | Add per rank lora saving |
| `192a1261` | Merge nightly (#564 ruff-on-tests + mutation compat fix) into deepspeed-fsdp |
| `b0fcc802` | Merge nightly (2.7.1, #522 squash + #562 coverage) into deepspeed-fsdp |
| `390dbd7f` | Merge nightly (bnb-quant + colocated-vLLM #522) into deepspeed-fsdp |
| `a1492c13` | Remove HF Accelerate and migrate to native torch.distributed |
| `b2eb2a21` | Restore adapter dtype on rebuild and isolate single-agent tests from leaked dist state |
| `5b6c66bd` | Fix DDP-wrapper fallout and finish core/test conversion |
| `1f357a25` | Remove DeepSpeed: torch-native DDP/FSDP2 LLM training via Accelerate |

#### TOOL-MASKING — do not carry into migration (89)

| SHA | subject |
| --- | --- |
| `c57d835e` | Address review comments: plain-language docstrings, no custom exception, cleaner server start |
| `62433bb3` | Remove redundant chunk_rows and lora-staging comments |
| `61db78f6` | Clarify ServedEnvClient ownership rationale in its docstring |
| `9f9479b9` | Cover VLLMConfig.sleep_mode_level validation |
| `002628b2` | Examples: repair entry points and default to in-process envs |
| `f566022b` | Algorithms: one LLMAlgorithm.test() with a turn cap, shared by GRPO/LLMPPO/LLMREINFORCE |
| `4ec01e00` | LLM env seam: backend-owned serving, strict /state, truncated parity, budget guards |
| `b27be91e` | Revert experimental config/benchmark tweaks to nightly baseline |
| `c546b71b` | Update test_core_base.py |
| `4adaad04` | [pre-commit.ci] auto fixes from pre-commit.com hooks |
| `be34c304` | Fix pre commit |
| `e81896ef` | Fix windows openenv tests |
| `ada95eea` | Create adapter path in build_vllm_rollout_lora_request test |
| `443ae1b5` | Update base.py |
| `2257c75d` | Increment package version |
| `8700a713` | Address protocol comments |
| `53cab2ba` | Fix CI |
| `f224a8aa` | Remove vllm lora diagnostics |
| `822e0bc0` | Remove vllm debugging statements |
| `6690fd1b` | Update comment in base.py |
| `fceedc06` | Default vllm sleep_mode_level back to 1 |
| `abf8247b` | Add some additional tests |
| `ee40c991` | Add final details to new dataset API |
| `6bf81533` | [pre-commit.ci] auto fixes from pre-commit.com hooks |
| `0bdbec1b` | Changed dataset api to return data on reset nd noop on step |
| `7be313cc` | Add docstrings and warning for deprecatde max_context_length arg in builder funtions |
| `3c24ee29` | Add more logging |
| `f60057df` | More comprehensive logging for lora copy |
| `28072b12` | [pre-commit.ci] auto fixes from pre-commit.com hooks |
| `7f14a97b` | Patch set lora to debug cuda error |
| `0aaafbd8` | Add debugging logs |
| `87a2b525` | [pre-commit.ci] auto fixes from pre-commit.com hooks |
| `9cd1a707` | Add option to set sleep mode |
| `ccd3ca5f` | [pre-commit.ci] auto fixes from pre-commit.com hooks |
| `7b1fa9de` | Add sleep level to vllm config |
| `5bd63829` | Add lora device guard to vllm |
| `74cb6ee7` | Hard deprecate old chunking args and unify under chunk_rows |
| `ae257700` | ci: add "ans" to codespell ignore-words-list |
| `e2f641ce` | LLM envs: pluggable RolloutEnv backend + local transport, BatchPointer, close-once |
| `a523bf7b` | Add AsyncOpenEnvClient (async httpx sibling) for the disaggregated rollout path |
| `33d2d144` | Clarify OpenEnvServer lifetime in docstring (per run, not per episode) |
| `7d342cba` | Remove the redundant tools= param from RolloutEnv |
| `22974224` | Drop the serve() one-liner; document why each OpenEnv wrapper exists |
| `869e864c` | One httpx OpenEnvClient; every env is a hosted server we hit |
| `1604a6fb` | [pre-commit.ci] auto fixes from pre-commit.com hooks |
| `35102b87` | Standardise the chunking args to just chunk_rows |
| `74f68c8c` | Rename GymEnvironment -> OpenEnvWrapper |
| `ff81340a` | OpenEnv: report the served env's real name, not a hardcoded label |
| `0433262a` | Merge origin/nightly into feature/tool-aware-token-masking |
| `a9187ed1` | Stop owned server directly on close (skip redundant /close round-trip) |
| `ec08ad08` | One OpenEnv server instance per rollout for batched training |
| `ccde4e11` | Remove public ReasoningEnv; source OpenEnv timeout from the manifest |
| `dc1a802b` | Fold ReasoningEnv into rollout_env.py; serve-and-train-by-URL demo |
| `17ffe0a7` | Fully adopt OpenEnv as the LLM env backend |
| `68076604` | Rework LLM env layer onto a unified OpenEnv HTTP interface |
| `37765f18` | Rename RolloutHarness->RolloutEnvWrapper; dissolve Trajectory into BatchRolloutEnv.envs |
| `f3b732b6` | Absorb the LLM rollout trajectory registry into BatchRolloutEnv |
| `8900f246` | Update rollout_env.py |
| `0391624d` | Cover train_llm.py reasoning-mode metric branches codecov flagged |
| `86c062af` | Fix RolloutHarness.reset: forward row_index only when set |
| `d01e6937` | Drop :ivar objective: from DatasetEnv (it's a constructor arg) |
| `accc89ab` | Rename DatasetEnv kind= -> objective= |
| `2ac09404` | Remove sliding-window / trajectory-stitching; simplify + document RolloutHarness |
| `dd957c77` | Use "loss" (not "mean_loss") as the preference metric key |
| `df7d6112` | Fix DPO loss display + add critic-LR mutation keys to ppo_llm.yaml |
| `99d583b4` | Fix SFT path of train_llm_dataset to consume SFT.learn's dict return |
| `0cebe2f9` | Test DatasetEnv's unknown-kind ValueError branch |
| `420b455c` | Fold DPO/SFT presets into DatasetEnv(kind=); finish llm_envs restructure |
| `4bc1cbb0` | Split LLM training entrypoints: train_llm_rollout + train_llm_dataset |
| `8617c2c5` | Cover the LLM-env taxonomy + RolloutHarness lines codecov flagged |
| `0a5b98c9` | Give RolloutHarness test doubles an __init__/_env so they instantiate |
| `be56a4e5` | Fix LLM test doubles for the RolloutHarness guard |
| `050149cc` | Rename TokenObservationWrapper -> RolloutHarness; compose, don't subclass |
| `01d1a42b` | Rename tmpl_kwargs -> chat_template_kwargs for clarity (review) |
| `d1d5aef4` | Cover the feedback-boundary tools path; drop internal-plan/private-repo refs |
| `8c48ceb9` | Fix stale LLM test env doubles for the RolloutEnv reset/step contract |
| `5e67720f` | Make LLM algo test() env doubles real RolloutEnv subclasses |
| `99bb802e` | Repoint LLM finetuning docs to RolloutEnv/finetune_llm_multiturn |
| `63811e11` | Fold BatchIterationState into BatchRolloutEnv |
| `0ee6529b` | llm_envs: collapse HuggingFaceGym/IterablePromptBatchGym into DatasetEnv; drop ReasoningGym + make_reasoning_rollout_env |
| `09b15a65` | llm_envs: rename ReasoningRolloutState->BatchIterationState, move cursor to BatchRolloutEnv |
| `33a52fab` | PR-3: make LLMEnv/RolloutEnv concrete base classes (reasoning = RolloutEnv(max_turns=1)) |
| `46156010` | [pre-commit.ci] auto fixes from pre-commit.com hooks |
| `9349d97d` | PR-3: fold ReasoningGym into a single-turn RolloutEnv (framework) |
| `cdb8b8b4` | llm_envs: introduce LLMEnv base + rename MultiTurnEnv->RolloutEnv, SyncMultiTurnVecEnv->BatchRolloutEnv |
| `9177153a` | llm_envs: collapse PreferenceGym + SFTGym into one descriptor-configured DatasetEnv |
| `22899b8e` | Fix: bare-wrapper test helper sets tools=None (matches __init__ default) |
| `165a11de` | Phase 0: tools= schema injection in TokenObservationWrapper + masking/turn_ids/tools tests |
| `36f612e1` | Phase 0 (tool-calling envs): P0-4 test scaffold for tool-aware TokenObservationWrapper |

### FSDP2 approach (assumptions explicit)

Introduced in two stages:

1. **`1f357a25`** — Remove DeepSpeed; keep Accelerate; add FSDP2 as an Accelerate plugin path (`fsdp_version=2`) alongside plain DDP. Still Accelerate-shaped.
2. **`a1492c13`** — Remove Accelerate entirely; add `agilerl/utils/distributed.py` with native helpers.

On HEAD, the FSDP2 path is:

- **`FSDPConfig`** dataclass (`agilerl/utils/distributed.py:185-204`): `reshard_after_forward`, `cpu_offload`, `param_dtype`, `reduce_dtype`.
- **`apply_fsdp2`** (`agilerl/utils/distributed.py:223-262`): walks HF `_no_split_modules` for per-block `fully_shard`, then `fully_shard` on the root. **No `DeviceMesh` / `mesh` argument is passed** — sharding uses the default process group.
- Wired from `LLMAlgorithm.wrap_models` (`agilerl/algorithms/core/base.py:2783-2798`): only when `self.distributed and self.fsdp_config is not None`; rebuilds optimizer after DTensor swap.
- Non-FSDP DP: **no DDP wrapper**; `sync_grads` averages LoRA grads at the accumulation boundary (`agilerl/algorithms/core/base.py:4176-4183`, `agilerl/utils/distributed.py:136-150`).
- Param gather seam: `gather_full_params` / `is_fsdp_sharded` / `load_full_state_dict` in `agilerl/utils/llm_utils.py:177-247` (FSDP2 `unshard`/`reshard`; rejects FSDP1).

**Assumption — single flat process group: YES.** Evidence:

- `init_distributed` calls `dist.init_process_group` once with no mesh (`agilerl/utils/distributed.py:72-75`).
- `get_rank` / `get_world_size` wrap global `dist.get_rank` / `dist.get_world_size` (`agilerl/utils/distributed.py:81-97`).
- `apply_fsdp2` never constructs or consumes a `DeviceMesh`.
- `shard_dataloader_kwargs` shards on global world size/rank (`agilerl/utils/distributed.py:278-288`).
- Batch sizing divides global batch by global world size (`agilerl/algorithms/core/base.py:4797-4804`).
- vLLM TP subgroups are carved from the flat world with `new_subgroups_by_enumeration` keyed by global process count (`agilerl/algorithms/core/base.py:5083-5107`).

There is no DP/TP/HSDP mesh abstraction. Global-rank topology is the only topology.

### Accelerate removal approach

- Delete `Accelerator` from algorithms, training loops, HPO, demos, tutorials, tests (`a1492c13` message and 120-file diff).
- Classic RL: strip distributed plumbing entirely (no accelerator params, no wrap/unwrap).
- LLM: `init_distributed` from launcher env (`RANK`/`LOCAL_RANK`/`WORLD_SIZE`/`MASTER_ADDR`/`MASTER_PORT`); helpers no-op without a process group (`agilerl/utils/distributed.py:1-9,41-78`).
- Replace `accelerator.prepare(dataloader)` with `shard_dataloader_kwargs` (`agilerl/llm_envs/dataset_env.py:131-144`).
- Replace `accelerator.wait_for_everyone` / `is_main_process` / gather with `barrier` / `is_main_process` / `gather_tensor` / `all_reduce_mean` (`agilerl/utils/utils.py:1370-1413`).
- Compound backend `cpu:gloo,cuda:nccl` (`96521255`, `agilerl/utils/distributed.py:71`).
- Follow-ups harden tests against leaked process groups and Windows gloo (`844a5ccd`, `ed8edc77`, `1ed1f8f0`, `542f36c3`, `60637ffd`).

---

## PART 2 — Drift analysis

Mechanical commands:

```
MB=$(git merge-base feature/tool-masking-fsdp feature/zero3-support)
# => ab7eb1fc948a33143eadf31a9acf38a0a2d6bfea
FABLE_FILES=$(git diff $MB feature/tool-masking-fsdp --name-only)  # 157 files
ZERO3_FILES=$(git diff $MB feature/zero3-support --name-only)      # 610 files
```

### CLEAN-LIFT (19 files) — in FABLE_FILES, not in ZERO3_FILES

Low risk to lift with a spot check. **Migration-relevant subset** (ignore OpenEnv/tool-masking CLEAN-LIFT for the migration branch):

- `agilerl/utils/distributed.py`
- `configs/accelerate/accelerate.yaml`
- `configs/accelerate/bench_accelerate_config.yaml`
- `configs/accelerate/grpo_accelerate_config.yaml`
- `docs/api/utils/distributed.rst`
- `tests/test_utils/test_distributed.py`
- `tests/test_algorithms/test_single_agent/conftest.py`

Tool-masking / OpenEnv CLEAN-LIFT (do **not** bring into migration):

- `agilerl/llm_envs/dataset_env.py`
- `agilerl/llm_envs/openenv.py`
- `agilerl/llm_envs/rollout_env.py`
- `docs/api/utils/llm_utils.rst`
- `docs/api/wrappers/llm_envs.rst`
- `docs/llm_finetuning/environments.rst`
- `docs/llm_finetuning/llm_checkpoints.rst`
- `docs/llm_finetuning/quantization.rst`
- `scripts/local/serve_openenv_reasoning.py`
- `tests/test_algorithms/test_llms/test_llm_algorithm.py`
- `tests/test_wrappers/test_openenv.py`
- `tests/test_wrappers/test_token_observation_tools.py`

### RE-DERIVE (138 files) — in both diffs

Replacement must be re-derived against current `feature/zero3-support` code:

- `.pre-commit-config.yaml`
- `README.md`
- `agilerl/__init__.py`
- `agilerl/algorithms/core/base.py`
- `agilerl/algorithms/core/llm_ops/fused_lora.py`
- `agilerl/algorithms/core/registry.py`
- `agilerl/algorithms/cqn.py`
- `agilerl/algorithms/ddpg.py`
- `agilerl/algorithms/dpo.py`
- `agilerl/algorithms/dqn.py`
- `agilerl/algorithms/dqn_rainbow.py`
- `agilerl/algorithms/grpo.py`
- `agilerl/algorithms/ippo.py`
- `agilerl/algorithms/maddpg.py`
- `agilerl/algorithms/matd3.py`
- `agilerl/algorithms/neural_ts_bandit.py`
- `agilerl/algorithms/neural_ucb_bandit.py`
- `agilerl/algorithms/ppo.py`
- `agilerl/algorithms/ppo_llm.py`
- `agilerl/algorithms/reinforce_llm.py`
- `agilerl/algorithms/sft.py`
- `agilerl/algorithms/td3.py`
- `agilerl/hpo/mutation.py`
- `agilerl/hpo/tournament.py`
- `agilerl/llm_envs/__init__.py`
- `agilerl/llm_envs/base.py`
- `agilerl/llm_envs/preference.py`
- `agilerl/llm_envs/reasoning.py`
- `agilerl/llm_envs/search.py`
- `agilerl/llm_envs/sft.py`
- `agilerl/llm_envs/sync_vec_env.py`
- `agilerl/llm_envs/token_observation.py`
- `agilerl/protocols.py`
- `agilerl/rollouts/on_policy.py`
- `agilerl/training/train_bandits.py`
- `agilerl/training/train_llm.py`
- `agilerl/training/train_multi_agent_off_policy.py`
- `agilerl/training/train_multi_agent_on_policy.py`
- `agilerl/training/train_off_policy.py`
- `agilerl/training/train_offline.py`
- `agilerl/training/train_on_policy.py`
- `agilerl/typing.py`
- `agilerl/utils/algo_utils.py`
- `agilerl/utils/ilql_utils.py`
- `agilerl/utils/llm_utils.py`
- `agilerl/utils/log_utils.py`
- `agilerl/utils/minari_utils.py`
- `agilerl/utils/utils.py`
- `agilerl/wrappers/agent.py`
- `agilerl/wrappers/llm_envs.py`
- `agilerl/wrappers/make_evolvable.py`
- `benchmarking/benchmarking_llm_multiturn.py`
- `benchmarking/benchmarking_llm_preference.py`
- `benchmarking/benchmarking_llm_reasoning.py`
- `benchmarking/benchmarking_multi_agent_off_policy.py`
- `benchmarking/benchmarking_multi_agent_on_policy.py`
- `benchmarking/benchmarking_off_policy_distributed.py`
- `benchmarking/benchmarking_offline_distributed.py`
- `benchmarking/benchmarking_sft.py`
- `configs/training/llm_finetuning/cispo_quant_bench.yaml`
- `configs/training/llm_finetuning/cispo_quant_bench_qwen.yaml`
- `configs/training/llm_finetuning/ppo_llm.yaml`
- `demos/llm/debugging/debugging_llm.py`
- `demos/llm/debugging/debugging_llm_stage_1.py`
- `demos/llm/debugging/debugging_llm_stage_2.py`
- `demos/llm/debugging/debugging_llm_stage_3.py`
- `demos/llm/debugging/debugging_llm_training_matrix.py`
- `demos/llm/debugging/debugging_value.py`
- `demos/llm/demo_llm_finetuning.py`
- `demos/single_agent/demo_off_policy_distributed.py`
- `demos/single_agent/demo_offline_distributed.py`
- `docs/api/algorithms/base.rst`
- `docs/api/algorithms/cispo.rst`
- `docs/api/algorithms/dpo.rst`
- `docs/api/algorithms/gspo.rst`
- `docs/api/algorithms/llmppo.rst`
- `docs/api/algorithms/llmreinforce.rst`
- `docs/api/algorithms/sft.rst`
- `docs/api/train.rst`
- `docs/api/utils/index.rst`
- `docs/distributed_training/index.rst`
- `docs/get_started/index.rst`
- `docs/llm_finetuning/index.rst`
- `docs/multi_agent_training/index.rst`
- `docs/tutorials/llm_finetuning/grpo_finetuning.rst`
- `docs/tutorials/llm_finetuning/grpo_hpo.rst`
- `docs/tutorials/llm_finetuning/multiturn_grpo_ppo.rst`
- `docs/tutorials/llm_finetuning/sft_dpo_finetuning.rst`
- `pyproject.toml`
- `tests/assets/build_tiny_llm_fixture.py`
- `tests/conftest.py`
- `tests/subprocess_runner.py`
- `tests/test_algorithms/conftest.py`
- `tests/test_algorithms/test_bandits/test_neural_ts.py`
- `tests/test_algorithms/test_bandits/test_neural_ucb.py`
- `tests/test_algorithms/test_base.py`
- `tests/test_algorithms/test_core_base.py`
- `tests/test_algorithms/test_llms/conftest.py`
- `tests/test_algorithms/test_llms/test_dpo.py`
- `tests/test_algorithms/test_llms/test_grpo.py`
- `tests/test_algorithms/test_llms/test_ppo_llm.py`
- `tests/test_algorithms/test_llms/test_quantization.py`
- `tests/test_algorithms/test_llms/test_reinforce_llm.py`
- `tests/test_algorithms/test_llms/test_sft.py`
- `tests/test_algorithms/test_llms/test_vllm.py`
- `tests/test_algorithms/test_multi_agent/conftest.py`
- `tests/test_algorithms/test_multi_agent/test_ippo.py`
- `tests/test_algorithms/test_multi_agent/test_maddpg.py`
- `tests/test_algorithms/test_multi_agent/test_matd3.py`
- `tests/test_algorithms/test_single_agent/test_cqn.py`
- `tests/test_algorithms/test_single_agent/test_ddpg.py`
- `tests/test_algorithms/test_single_agent/test_dqn.py`
- `tests/test_algorithms/test_single_agent/test_dqn_rainbow.py`
- `tests/test_algorithms/test_single_agent/test_ppo.py`
- `tests/test_algorithms/test_single_agent/test_td3.py`
- `tests/test_components/test_sampler.py`
- `tests/test_hpo/test_mutation.py`
- `tests/test_hpo/test_tournament.py`
- `tests/test_protocols.py`
- `tests/test_rollouts/test_on_policy.py`
- `tests/test_train/test_train.py`
- `tests/test_train/test_train_llm.py`
- `tests/test_utils/test_algo_utils.py`
- `tests/test_utils/test_ilql_utils.py`
- `tests/test_utils/test_llm_utils.py`
- `tests/test_utils/test_log_utils.py`
- `tests/test_utils/test_minari_utils.py`
- `tests/test_utils/test_utils.py`
- `tests/test_wrappers/test_agent.py`
- `tests/test_wrappers/test_llm_envs.py`
- `tests/test_wrappers/test_multiturn_wrappers.py`
- `tests/utils.py`
- `tutorials/language/train_bc_lm.py`
- `tutorials/language/train_ilql.py`
- `tutorials/llm_finetuning/grpo_reasoning.py`
- `tutorials/llm_finetuning/grpo_reasoning_hpo.py`
- `tutorials/llm_finetuning/multiturn_grpo_ppo.py`
- `uv.lock`

### NEW-UNCOVERED accelerate call sites on `feature/zero3-support`

Definition: every `accelerate` import or runtime call site on zero3 HEAD where either (a) the file is absent from `FABLE_FILES`, or (b) the file is in `FABLE_FILES` but the specific accelerate coupling did not exist at MB (added on the zero3 side after diverge). Type-annotation-only mentions are excluded; imports and method/ctor uses are included.

**Count: 89**

| file:line | kind | why | site |
| --- | --- | --- | --- |
| `agilerl/algorithms/cqn.py:11` | import | added-after-mb | `from accelerate import Accelerator` |
| `agilerl/algorithms/ddpg.py:13` | import | added-after-mb | `from accelerate import Accelerator` |
| `agilerl/algorithms/dpo.py:685` | call | added-after-mb | `self.accelerator.wait_for_everyone()` |
| `agilerl/algorithms/dqn.py:12` | import | added-after-mb | `from accelerate import Accelerator` |
| `agilerl/algorithms/dqn_rainbow.py:10` | import | added-after-mb | `from accelerate import Accelerator` |
| `agilerl/algorithms/grpo.py:543` | call | added-after-mb | `if self.accelerator is not None and self.accelerator.num_processes > 1:` |
| `agilerl/algorithms/grpo.py:598` | call | added-after-mb | `self.accelerator.num_processes if self.accelerator is not None else 1` |
| `agilerl/algorithms/grpo.py:603` | call | added-after-mb | `and self.accelerator.num_processes > 1` |
| `agilerl/algorithms/grpo.py:623` | call | added-after-mb | `if self.accelerator is not None and self.accelerator.num_processes > 1:` |
| `agilerl/algorithms/grpo.py:624` | call | added-after-mb | `self.accelerator.wait_for_everyone()` |
| `agilerl/algorithms/grpo.py:656` | call | added-after-mb | `if self.accelerator is not None and self.accelerator.num_processes > 1:` |
| `agilerl/algorithms/grpo.py:657` | call | added-after-mb | `self.accelerator.wait_for_everyone()` |
| `agilerl/algorithms/grpo.py:770` | call | added-after-mb | `self.accelerator.wait_for_everyone()` |
| `agilerl/algorithms/ippo.py:13` | import | added-after-mb | `from accelerate import Accelerator` |
| `agilerl/algorithms/maddpg.py:14` | import | added-after-mb | `from accelerate import Accelerator` |
| `agilerl/algorithms/matd3.py:14` | import | added-after-mb | `from accelerate import Accelerator` |
| `agilerl/algorithms/neural_ts_bandit.py:8` | import | added-after-mb | `from accelerate import Accelerator` |
| `agilerl/algorithms/neural_ucb_bandit.py:8` | import | added-after-mb | `from accelerate import Accelerator` |
| `agilerl/algorithms/ppo.py:13` | import | added-after-mb | `from accelerate import Accelerator` |
| `agilerl/algorithms/ppo_llm.py:901` | call | added-after-mb | `self.accelerator.wait_for_everyone()` |
| `agilerl/algorithms/reinforce_llm.py:783` | call | added-after-mb | `self.accelerator.wait_for_everyone()` |
| `agilerl/algorithms/sft.py:400` | call | added-after-mb | `self.accelerator.wait_for_everyone()` |
| `agilerl/algorithms/td3.py:13` | import | added-after-mb | `from accelerate import Accelerator` |
| `agilerl/logger.py:25` | import | file-not-in-fable-diff | `from accelerate import Accelerator` |
| `agilerl/logger.py:56` | call | file-not-in-fable-diff | `accelerator.wait_for_everyone()` |
| `agilerl/logger.py:58` | call | file-not-in-fable-diff | `yield accelerator is None or accelerator.is_main_process` |
| `agilerl/logger.py:61` | call | file-not-in-fable-diff | `accelerator.wait_for_everyone()` |
| `agilerl/models/algo.py:24` | import | file-not-in-fable-diff | `from accelerate import Accelerator` |
| `agilerl/models/algo.py:604` | call | file-not-in-fable-diff | `self.batch_size // accelerator.num_processes, 1` |
| `agilerl/models/env.py:37` | import | file-not-in-fable-diff | `from accelerate import Accelerator` |
| `agilerl/population.py:32` | import | file-not-in-fable-diff | `from accelerate import Accelerator` |
| `agilerl/population.py:643` | call | file-not-in-fable-diff | `if self.accelerator is not None and self.accelerator.is_main_process:` |
| `agilerl/population.py:644` | call | file-not-in-fable-diff | `steps = [step * self.accelerator.num_processes for step in steps]` |
| `agilerl/rollouts/on_policy.py:321` | call | added-after-mb | `accelerator.wait_for_everyone()` |
| `agilerl/train.py:20` | import | file-not-in-fable-diff | `from accelerate import Accelerator` |
| `agilerl/train.py:125` | call | file-not-in-fable-diff | `accelerator = Accelerator() if args.use_accelerator else None` |
| `agilerl/training/llm/multiturn.py:10` | import | file-not-in-fable-diff | `from accelerate import Accelerator` |
| `agilerl/training/llm/multiturn.py:146` | call | file-not-in-fable-diff | `data_increment = accelerator.num_processes if accelerator is not None else 1` |
| `agilerl/training/llm/multiturn.py:202` | call | file-not-in-fable-diff | `if accelerator is None or accelerator.is_main_process:` |
| `agilerl/training/llm/multiturn.py:270` | call | file-not-in-fable-diff | `world_size=accelerator.num_processes,` |
| `agilerl/training/llm/multiturn.py:306` | call | file-not-in-fable-diff | `if accelerator is None or accelerator.is_main_process:` |
| `agilerl/training/llm/multiturn.py:314` | call | file-not-in-fable-diff | `if accelerator is None or accelerator.is_main_process:` |
| `agilerl/training/llm/multiturn.py:322` | call | file-not-in-fable-diff | `if accelerator is None or accelerator.is_main_process:` |
| `agilerl/training/llm/multiturn.py:327` | call | file-not-in-fable-diff | `accelerator.wait_for_everyone()` |
| `agilerl/training/llm/multiturn.py:335` | call | file-not-in-fable-diff | `accelerator.wait_for_everyone()` |
| `agilerl/training/llm/multiturn.py:349` | call | file-not-in-fable-diff | `accelerator.wait_for_everyone()` |
| `agilerl/training/llm/preference.py:7` | import | file-not-in-fable-diff | `from accelerate import Accelerator` |
| `agilerl/training/llm/preference.py:129` | call | file-not-in-fable-diff | `data_increment = accelerator.num_processes if accelerator is not None else 1` |
| `agilerl/training/llm/preference.py:176` | call | file-not-in-fable-diff | `accelerator.wait_for_everyone()` |
| `agilerl/training/llm/preference.py:205` | call | file-not-in-fable-diff | `if accelerator is None or accelerator.is_main_process:` |
| `agilerl/training/llm/preference.py:220` | call | file-not-in-fable-diff | `accelerator.wait_for_everyone()` |
| `agilerl/training/llm/preference.py:234` | call | file-not-in-fable-diff | `accelerator.wait_for_everyone()` |
| `agilerl/training/llm/reasoning.py:7` | import | file-not-in-fable-diff | `from accelerate import Accelerator` |
| `agilerl/training/llm/reasoning.py:140` | call | file-not-in-fable-diff | `data_increment = accelerator.num_processes if accelerator is not None else 1` |
| `agilerl/training/llm/reasoning.py:188` | call | file-not-in-fable-diff | `accelerator.wait_for_everyone()` |
| `agilerl/training/llm/reasoning.py:209` | call | file-not-in-fable-diff | `world_size=accelerator.num_processes,` |
| `agilerl/training/llm/reasoning.py:235` | call | file-not-in-fable-diff | `if accelerator is None or accelerator.is_main_process:` |
| `agilerl/training/llm/reasoning.py:254` | call | file-not-in-fable-diff | `if accelerator is None or accelerator.is_main_process:` |
| `agilerl/training/llm/reasoning.py:269` | call | file-not-in-fable-diff | `accelerator.wait_for_everyone()` |
| `agilerl/training/llm/reasoning.py:283` | call | file-not-in-fable-diff | `accelerator.wait_for_everyone()` |
| `agilerl/training/llm/sft.py:6` | import | file-not-in-fable-diff | `from accelerate import Accelerator` |
| `agilerl/training/llm/sft.py:120` | call | file-not-in-fable-diff | `data_increment = accelerator.num_processes if accelerator is not None else 1` |
| `agilerl/training/llm/sft.py:162` | call | file-not-in-fable-diff | `accelerator.wait_for_everyone()` |
| `agilerl/training/llm/sft.py:185` | call | file-not-in-fable-diff | `if accelerator is None or accelerator.is_main_process:` |
| `agilerl/training/llm/sft.py:200` | call | file-not-in-fable-diff | `accelerator.wait_for_everyone()` |
| `agilerl/training/llm/sft.py:214` | call | file-not-in-fable-diff | `accelerator.wait_for_everyone()` |
| `agilerl/training/trainer.py:86` | import | file-not-in-fable-diff | `from accelerate import Accelerator` |
| `agilerl/training/trainer.py:332` | call | file-not-in-fable-diff | `self.accelerator = create_llm_accelerator()` |
| `agilerl/utils/algo_utils.py:29` | import | added-after-mb | `from accelerate import Accelerator` |
| `agilerl/utils/llm_utils.py:389` | call | added-after-mb | `tensor = torch.tensor(tensor, device=accelerator.device)` |
| `agilerl/utils/llm_utils.py:390` | call | added-after-mb | `tensor = tensor.to(accelerator.device)` |
| `agilerl/utils/llm_utils.py:391` | call | added-after-mb | `return accelerator.gather(tensor)` |
| `agilerl/utils/llm_utils.py:421` | call | added-after-mb | `t = torch.tensor([int(value)], device=accelerator.device, dtype=torch.long)` |
| `agilerl/utils/llm_utils.py:422` | call | added-after-mb | `gathered = accelerator.gather(t)` |
| `agilerl/utils/llm_utils.py:558` | call | added-after-mb | `accelerator.wait_for_everyone()` |
| `agilerl/utils/llm_utils.py:1857` | call | added-after-mb | `return f"cuda:{accelerator.process_index}"` |
| `agilerl/utils/trainer_utils.py:12` | import | file-not-in-fable-diff | `from accelerate import Accelerator` |
| `agilerl/utils/trainer_utils.py:273` | call | file-not-in-fable-diff | `agent_accelerator = Accelerator() if accelerator is not None else None` |
| `tests/test_models/test_pydantic_models.py:865` | call | file-not-in-fable-diff | `accelerator.num_processes = 2` |
| `tests/test_population.py:844` | call | file-not-in-fable-diff | `accelerator.is_main_process = True` |
| `tests/test_population.py:845` | call | file-not-in-fable-diff | `accelerator.num_processes = 4` |
| `tests/test_population.py:870` | call | file-not-in-fable-diff | `accelerator.is_main_process = False` |
| `tests/test_utils/test_llm_utils.py:1545` | call | added-after-mb | `accelerator.process_index = 3` |
| `tests/test_utils/test_llm_utils.py:1552` | call | added-after-mb | `accelerator.process_index = 2` |
| `tests/test_utils/test_llm_utils.py:3332` | call | added-after-mb | `accelerator.wait_for_everyone.side_effect = wait_for_everyone` |
| `tests/test_utils/test_llm_utils.py:3363` | call | added-after-mb | `assert accelerator.wait_for_everyone.call_count == 1` |
| `tests/test_utils/test_llm_utils.py:3399` | call | added-after-mb | `accelerator.wait_for_everyone.assert_called_once()` |
| `tests/test_utils/test_llm_utils.py:3422` | call | added-after-mb | `accelerator.wait_for_everyone.assert_called_once()` |
| `tests/test_utils/test_llm_utils.py:3485` | call | added-after-mb | `assert accelerator.wait_for_everyone.call_count == 1` |

Notable clusters with no Fable prior art:

- New package surface: `agilerl/logger.py`, `agilerl/population.py`, `agilerl/train.py`, `agilerl/training/trainer.py`, `agilerl/training/llm/{multiturn,preference,reasoning,sft}.py`, `agilerl/models/{algo,env}.py`, `agilerl/utils/trainer_utils.py`.
- ZeRO-3 lockstep barriers / world-size checks added after MB in `grpo.py`, `dpo.py`, `ppo_llm.py`, `reinforce_llm.py`, `sft.py`, `llm_utils.py` (`gather` / `allreduce_minmax_int` / `align_completion_batch_shapes_across_ranks`).
- Classic RL `from accelerate import Accelerator` re-introduced on zero3 after MB (Arena / v2.8 path) even though Fable stripped classic distributed plumbing.

---

## PART 3 — Flat-world audit (`feature/tool-masking-fsdp` HEAD)

Every hit below is a latent bug once the mesh has more than one dimension. Replacements use mesh-dim-local queries only (never recommend bare `dist.get_rank()` / `dist.get_world_size()` in model/training/data code). Global rank remains acceptable only in process-group bootstrap and logging (AGENTS.md rule 2).

### Hits

| location | pattern | mesh-dim-local replacement |
| --- | --- | --- |
| `agilerl/utils/distributed.py:83` | `dist.get_rank()` inside `get_rank()` | Bootstrap/logging only. Callers that mean DP identity must use `mesh["dp"].get_local_rank()`; do not re-export a global-rank helper into training/data. |
| `agilerl/utils/distributed.py:97` | `dist.get_world_size()` inside `get_world_size()` | Same: bootstrap/logging only. DP size → `mesh["dp"].size()`. |
| `agilerl/utils/distributed.py:88-89` | reads `LOCAL_RANK` env | Device placement at bootstrap: keep env read only inside `init_distributed` / `resolve_device`. Training code should take a resolved device, not re-read `LOCAL_RANK`. |
| `agilerl/utils/distributed.py:38` | `_LAUNCHER_ENV_VARS` includes `WORLD_SIZE` | Bootstrap-only rendezvous check; do not use `WORLD_SIZE` for batch/data decisions. |
| `agilerl/utils/distributed.py:91` | `get_rank() % device_count` fallback for local rank | Prefer launcher `LOCAL_RANK` at bootstrap; under a multi-dim mesh local device index must come from the host/device placement plan, not global rank arithmetic. |
| `agilerl/utils/distributed.py:102` | `is_main_process` via `get_rank()==0` | Logging/coordinator: `mesh.get_rank()==0` (global) is OK for logging; for DP-main within a replica group use `mesh["dp"].get_local_rank()==0`. |
| `agilerl/utils/distributed.py:174` | `resolve_device` → `cuda:{get_local_rank()}` | OK as bootstrap device bind if `get_local_rank` stays bootstrap-scoped; do not call from model forward/data. |
| `agilerl/utils/distributed.py:282-286` | **`DistributedSampler(..., num_replicas=get_world_size(), rank=get_rank())`** | See **Dataloader sharding** below. Use DP-dim size/rank; replicate across TP. |
| `agilerl/llm_envs/dataset_env.py:135,142` | calls `shard_dataloader_kwargs` | Inherits flat-world sampler; re-derive against mesh (DP shard, TP replicate). |
| `agilerl/utils/utils.py:1373` | `_collective_device` uses `get_local_rank()` | Collectives for metrics: device bind is bootstrap-like; rank list length must use the mesh dim that owns the collective (usually `mesh["dp"].size()`), not global world size. |
| `agilerl/utils/utils.py:1391` | `gather_tensor` allocates `get_world_size()` slots | `mesh["dp"].size()` (or the dim of the process group used for the gather). |
| `agilerl/utils/utils.py:1496-1503` | `_distributed_world_size` / `_distributed_rank` wrap globals | Replace call sites with `mesh["dp"].size()` / `mesh["dp"].get_local_rank()` for batch accounting and seed decorrelation. |
| `agilerl/utils/ilql_utils.py:32` | `num_processes = get_world_size()` | Config metadata: `mesh["dp"].size()` if it means data-parallel degree. |
| `agilerl/algorithms/core/base.py:2231-2233` | seed broadcast when `get_world_size()>1`; `seed += get_rank()` | Decorrelate on DP rank: `mesh["dp"].get_local_rank()` (TP peers must keep the **same** seed so they see identical data). |
| `agilerl/algorithms/core/base.py:4266-4267` | LoRA staging `rank_{get_rank()}` when `get_world_size()>1` | Per-engine staging: if one colocated engine per DP replica, key by `mesh["dp"].get_local_rank()` (or a dedicated host-local id). TP ranks that share an engine must share the staging dir. |
| `agilerl/algorithms/core/base.py:4529` | `torch.distributed.get_rank(group=self.tp_group)` | Prefer `mesh["tp"].get_local_rank()` once a TP mesh dim exists. Group-local rank is the right *idea*; the flat subgroup construction that feeds it still assumes a 1-D world (`5083-5107`). |
| `agilerl/algorithms/core/base.py:4797-4804` | `batch_size / get_world_size()` | `batch_size / mesh["dp"].size()` — TP ranks are not data-parallel replicas. |
| `agilerl/algorithms/core/base.py:5083-5085` | vLLM init uses `get_world_size` / `get_rank` / `get_local_rank` | Bootstrap/orchestration for the inference engine: global rank OK only while building the launcher mapping; TP degree must come from `mesh["tp"].size()`, DP degree from `mesh["dp"].size()`. |
| `agilerl/algorithms/core/base.py:5115-5124` | writes `RANK` / `LOCAL_RANK` / `WORLD_SIZE` for vLLM | Bootstrap env synthesis for an external launcher — allowed at the boundary; do not let training code read these back for mesh decisions. |
| `agilerl/algorithms/core/base.py:5333` | `device_index = get_local_rank()` | Device bind at engine setup; keep out of train/data loops. |
| `tests/test_algorithms/test_core_base.py:5250` | patches `torch.cuda.current_device` | Test seam only; production code on this branch does not call it. Under a mesh, tests should mock mesh-local device placement instead of a global current-device. |
| tests env cleanup (`tests/conftest.py:464-465`, `tests/subprocess_runner.py:47-48`, single/multi-agent conftests, `tests/test_utils/test_distributed.py`) | set/clear `LOCAL_RANK` / `WORLD_SIZE` | Test bootstrap fixtures — OK; production training must not mirror this pattern. |

### Dataloader sharding (critical under tensor parallelism)

**Current behaviour** (`agilerl/utils/distributed.py:265-288`, used at `agilerl/llm_envs/dataset_env.py:131-144`):

```python
DistributedSampler(dataset, num_replicas=get_world_size(), rank=get_rank(), shuffle=...)
```

This shards the dataset across **every** global rank.

Under tensor parallelism, all ranks inside a TP group must receive the **identical** batch. A sampler keyed on global world size gives each TP peer a different shard. Training then:

- consumes wrong (non-replicated) data inside the TP group,
- still produces a plausible loss curve,
- silently corrupts learning.

**Required replacement shape** (conceptual; do not invent APIs beyond mesh-dim queries):

- Shard only along the data-parallel dimension: `num_replicas=mesh["dp"].size()`, `rank=mesh["dp"].get_local_rank()`.
- Ranks that share a TP group must use the same DP rank (they are the same DP replica), so they draw the same samples.
- Do not use `dist.get_world_size()` / `dist.get_rank()` / `WORLD_SIZE` / `RANK` for sampler construction.

Until that exists, `shard_dataloader_kwargs` is flat-world-only and is unsafe to lift unchanged into a multi-dim mesh.

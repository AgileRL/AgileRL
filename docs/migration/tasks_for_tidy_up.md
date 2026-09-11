# Tasks for tidy-up

## De-duplicate methods in LLMAgent that do similar things

### Drop `update_existing_adapter`

`LLMAlgorithm.update_existing_adapter` is `_load_adapter_weights` plus a
weaker `requires_grad` loop. Delete it.

- Call `_restore_adapter_trainability(["actor", "critic"])` at the end of
  `_load_adapter_weights` (after the reference freeze). That is the real
  trainability restore; the `"actor" in name or "critic" in name` loop goes
  with the wrapper.
- Ray `load_peft_model` is gone. Remaining callers are OSS
  (`test_core_base.py`). Point those at `_load_adapter_weights` (public
  `load_adapter_weights` if another class needs it). Do not `getattr`
  around it.

## Drop post-`build_core_algo` shims in `DistributedLLMAgent`

Two leftover blocks after `LLMBuilder.build` in
`packages/agilerl-ray/agilerl_ray/agents/llm_agent.py` (`build_core_algo`,
~386–402). Neither fires on a real algorithm.

### `lr` / `lr_actor` copy

`if not hasattr(self._core_algo, "lr") and hasattr(..., "lr_actor")` then
`self._core_algo.lr = self._core_algo.lr_actor`.

`LLMAlgorithm` always sets `self.lr`. LLMPPO still has `lr_actor`, but it
already passes `lr=lr_actor` into `super().__init__`, so both exist after
construction. The branch only exists for the unit-test
`SimpleNamespace(lr_actor=...)`.

Keep the `lr` → `lr_actor` copy in `set_hyperparams` (LLMPPO HPO). Delete
the post-build shim and `test_build_core_algo_sets_lr_from_lr_actor`.

### `hasattr(..., "llm")` / `vllm_config` / `sleep_mode`

Async trainers set `use_vllm=False`, so they never get `core_algo.llm` and
the block never runs. Colocated OSS creates `self.llm` only after it has a
`vllm_config` (it default-constructs one if missing), so “vllm_config is
not set” cannot fire on a real algo.

OSS does not require `sleep_mode=True`. If Ray still wants that policy, put
it on the spec/builder, not a post-build `hasattr` probe.

Delete the block and
`test_build_core_algo_requires_vllm_config` /
`test_build_core_algo_requires_vllm_sleep_mode`.

## Drop `filesystem_adapter_weights_path`

One-line wrapper in
`packages/agilerl-ray/agilerl_ray/rollout/weight_sync/backends.py`:
`Path(publish_path) / ADAPTER_WEIGHTS_BASENAME`.

The constant already names the file. Sidecar uses the same join with no
helper (`Path(publish_path) / ADAPTER_SIDECAR_BASENAME`) in both
`DistributedLLMAgent.save_filesystem_adapter` and the filesystem backend.

Delete the helper. Inline at the two production call sites
(`save_filesystem_adapter` and the backend load). Keep
`filesystem_adapter_publish_path` — that one builds the directory layout.

## Drop TP replica collapse in `_make_env`

`DistributedLLMAgent._make_env` still calls `data_parallel_topology(
None, processes_per_replica=tensor_parallel_size)` like main
(`self.accelerator` there). That maps
process rank → replica index when colocated vLLM TP > 1.

This path never sees TP > 1: async returns before the call, colocated OSS
raises `tensor_parallel_size==1`, and async ResourceSpec rejects TP > 1.

Delete the `vllm_config` / `tensor_parallel_size` block and pass
`get_rank()` / `get_world_size()` into `LLMEnvBuildContext`. Then drop
the Ray import of `data_parallel_topology` if unused.

## Drop NCCL static wrappers on `DistributedLLMAgent`

`_infiniband_devices_present` and `_default_socket_ifname` in
`packages/agilerl-ray/agilerl_ray/agents/llm_agent.py` (~1632–1640)
are static methods that only call `infiniband_devices_present` and
`default_nccl_socket_ifname` from `agilerl_ray.utils.ray_utils`.

Delete both methods. Call the functions at the two sites in
`initialize_distributed` (~1701 / ~1705). Retarget the patches in
`tests/unit/agents/test_llm_agent.py` to the `ray_utils` names.

## Drop `_broadcast_object_list` on `DistributedLLMAgent`

Static method in `packages/agilerl-ray/agilerl_ray/agents/llm_agent.py`
(~1567). It re-checks `dist.is_available()` / `is_initialized()` /
`world_size == 1`, then calls `broadcast_object_list`. That is already
what `agilerl.distributed.broadcast_object_list` does.

The wrapper still passes `from_process=` (Accelerate). The OSS helper
takes `src=`.

Delete the method. Call `broadcast_object_list` at the three sites
(`run_learn_chunk` ~1010 / ~1019, `pop_metrics_buffer` ~1589) with
`src=`. Retarget
`tests/unit/agents/test_llm_agent_async_unit.py` from
`DistributedLLMAgent._broadcast_object_list` to
`agilerl_ray.agents.llm_agent.broadcast_object_list` (several
`test_llm_agent.py` patches already use that path).

## Drop `dist_ready`

`DistributedLLMAgent.dist_ready` is `is_distributed()` (`dist.is_available()
and dist.is_initialized()`) that raises instead of returning `False`.

The only production call is `LLMAgentManager`, immediately after
`initialize_distributed` has already returned on every worker
(`llm_agent.py` ~1982). That RPC is the one that calls
`dist.init_process_group`. A second round-trip cannot catch a failure the
first already would have raised.

Do not replace it with module-level `is_distributed()` from the manager —
that would check the manager process, not the actors. Delete the method,
the manager `ray.get([w.dist_ready.remote() ...])` line, the
`TestReadyAndGetNodeIp` dist tests, and the parametrized `dist_ready`
cases / mock `dist_ready.remote` stubs in
`tests/unit/agents/test_llm_agent.py`. Same leftover sequence as main.

## Drop dead `DistributedLLMAgent` attributes

`packages/agilerl-ray/agilerl_ray/agents/llm_agent.py` (`DistributedLLMAgent.__init__`
and `initialize_distributed`). Constructor args stay (`runtime`, `worker`,
`pulsar`); these fields are copies or never read.

### `rank`

Set from `worker.rank`, never read. Dist init uses
`global_rank = int(self.worker_id)`. The only spawn site
(`LLMAgentManager`) always passes `worker_id=i, rank=i`.

Delete `self.rank`. Delete `rank` from `LLMWorkerContext`
(`agilerl_ray/agents/context.py`) and from the manager's
`LLMWorkerContext(...)` call. Keep `worker_id`.

### `rollout_engine_ref`

Set to `None`, never read. Engines live on the manager
(`_rollout_engines`). Delete the attribute.

### `local_rank`

Set in `initialize_distributed` from Ray's GPU id, never read. The same
index already goes into `self.device` and `os.environ["LOCAL_RANK"]`.
Delete `self.local_rank`.

### `dataset_file_name` / `reward_file_name`

Class attributes on `DistributedLLMAgent`. No references in production or
tests. Delete both.

### Copies (optional follow-up)

- `_async_rollout` is `training_spec.async_rollout`. `_make_env` uses the
  private field; `_evaluation_loop` uses the `async_rollout` property.
  Collapse to one (`training_spec.async_rollout` or the property).
- `world_size` and `num_trainer_engines` are the same value at spawn
  (`world_size=self.num_trainer_engines`). Keep one name.
- `resource_spec` is stored by `BaseAgent` and never read again on this
  class after `training_gpus_per_agent` is copied into
  `num_trainer_engines`. Leave the constructor arg (needed for `super()`);
  do not add more copies.

`batch_size`, `data_batch_size_per_gpu`, `env_host_pool`, `memory`,
`_backend` (weight sync), `backend` (NCCL/Gloo string),
`score_array_dims`, and `unused_prompts` stay.

Class docstring still lists unpacked `agent_idx` / Pulsar fields and does
not mention `runtime` / `worker`. `__init__` docstring still says `worker`
carries DeepSpeed config; `LLMWorkerContext` no longer does. Fix those
when touching the class.

## Inline `COMPARABLE_HP_TYPES`

Module constant in `agilerl/models/algo.py` (~235). Used once, in
`_resume_and_warn_on_drift` (`isinstance(configured[name], COMPARABLE_HP_TYPES)`).

Delete the constant. Inline the tuple in that function as
`comparable_hp_types = (bool, int, float, str, type(None))`.
The comment stays with the local: skip types that have no meaningful
`!=` (a `LoraConfig`, a registry).

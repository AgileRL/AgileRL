# AgileRL

Deep reinforcement learning library focused on RLOps — evolutionary hyperparameter optimization, population-based training, and LLM alignment (SFT, DPO, GRPO, ILQL).

## Tooling

This project uses **uv** for dependency management. Always prefix commands with `uv run`:

```bash
uv run pytest tests/                          # run full test suite
uv run pytest tests/test_algorithms/ -x -q    # run algo tests, stop on first failure
uv run pytest -m "not llm"                    # skip LLM tests (slow)
uv run ruff check agilerl/                    # lint
uv run ruff format agilerl/                   # format
```

Install dependencies: `uv sync` (add `--extra llm` or `--extra all` for optional deps).

## Repo Structure

```
agilerl/
  algorithms/
    core/
      base.py              # Algorithm classes
      registry.py          # MutationRegistry, NetworkGroup, OptimizerConfig, HyperparameterConfig
      optimizer_wrapper.py # OptimizerWrapper — wraps torch optimizers with HPO metadata
    dqn.py, ddpg.py, ppo.py, td3.py, ...   # single-agent RL
    ippo.py, maddpg.py, matd3.py            # multi-agent RL
    sft.py, dpo.py, grpo.py, ilql.py        # LLM
  modules/             # EvolvableModule building blocks (MLP, CNN, LSTM, GPT, etc.)
  networks/            # EvolvableNetwork compositions (actors, Q-networks, value nets)
  components/          # ReplayBuffer, RolloutBuffer, MultiAgentReplayBuffer, Sampler
  training/            # High-level training loops per paradigm
    train_off_policy.py
    train_on_policy.py
    train_multi_agent_off_policy.py
    train_multi_agent_on_policy.py
    train_offline.py
    train_bandits.py
    train_llm.py
  hpo/
    mutation.py        # Mutations — applies architecture/parameter/HP mutations to a population
    tournament.py      # TournamentSelection — selects elite and new population by fitness
  wrappers/
    agent.py           # AgentWrapper (ABC), RSNorm, AsyncAgentsWrapper
    learning.py        # Skill (curriculum learning env wrapper), BanditEnv
    llm_envs.py        # ReasoningGym, SFTGym, PreferenceGym
    pettingzoo_wrappers.py
  protocols.py         # Structural typing interfaces (EvolvableAlgorithmProtocol, etc.)
  typing.py            # Shared type aliases
  utils/               # Misc utilities (algo_utils, llm_utils, evolvable_networks, ...)
tests/                 # Mirrors agilerl/ structure (test_algorithms/, test_modules/, etc.)
```

## Key Abstractions

### Algorithm hierarchy

Every algorithm ultimately subclasses `EvolvableAlgorithm` (`algorithms/core/base.py`):

```
EvolvableAlgorithm          (ABC, metaclass=RegistryMeta)
├── RLAlgorithm             single-agent; adds observation_space, action_space
│   └── DQN, DDPG, TD3, PPO, CQN, RainbowDQN, NeuralUCB, NeuralTS, ...
├── MultiAgentRLAlgorithm   multi-agent; adds per-agent spaces
│   └── MADDPG, MATD3, IPPO
└── LLMAlgorithm            language-model finetuning; manages LoRA adapters, DeepSpeed
    └── GRPO, DPO, SFT, ILQL
```

`EvolvableAlgorithm` is responsible for:
- Network registration and HPO bookkeeping via `MutationRegistry`.
- `clone()` — deep-copies the algorithm for tournament selection.
- `save_checkpoint()` / `load_checkpoint()` — serialisation.
- `wrap_models()` / `unwrap_models()` — Accelerate distributed training integration.
- `mutation_hook()` — user-extensible hook called after every mutation.

### Registry (`algorithms/core/registry.py`)

Each algorithm registers its evolvable components declaratively inside `__init__`:

| Call | Purpose |
|---|---|
| `self.register_network_group(NetworkGroup(...))` | Register a network (+ optional shared/target nets) for mutation and cloning |
| `self.registry.register_optimizer(OptimizerConfig(...))` | Record which optimizer maps to which networks and LR attribute |
| `self.registry.register_hyperparameter(name, RLParameter(...))` | Mark a scalar HP as mutable |

`NetworkGroup` stores: `eval_network` (optimised during training), optional `shared_networks` (e.g. target nets that shadow the eval net), and `policy=True` for the network used in `get_action()`.

`MutationRegistry` is read by `Mutations` to know what to mutate and by `EvolvableAlgorithm` to know what to clone/checkpoint.

### OptimizerWrapper (`algorithms/core/optimizer_wrapper.py`)

All optimizers in AgileRL algorithms must be wrapped with `OptimizerWrapper`. It:
- Instantiates the underlying `torch.optim.Optimizer`.
- Stores `network_names` and `lr_name` so `Mutations` can reinitialise the optimizer after an architecture mutation.
- Infers attribute names from the calling frame automatically — no manual naming needed.

### EvolvableModule / EvolvableNetwork

- **`EvolvableModule`** (`modules/base.py`): Base for neural network building blocks. Supports architecture mutations (`add_layer`, `remove_layer`, `add_nodes`, `remove_nodes`) and `clone()`.
- **`EvolvableNetwork`** (`networks/base.py`): Composes `EvolvableModule`s into a complete encoder–decoder network with a single `forward()`. Used as the `actor`, `critic`, `q_network`, etc. attributes on algorithms.

### AgentWrapper (`wrappers/agent.py`)

`AgentWrapper` monkey-patches an algorithm's `get_action()` and `learn()` with pre/post-processing steps without subclassing the algorithm. Concrete wrappers:

- **`RSNorm`** — online running-stats observation normalisation (off-policy only).
- **`AsyncAgentsWrapper`** — handles asynchronous agent stepping in multi-agent envs (IPPO only).

Wrappers are transparent to `Mutations`: `Mutations.mutation()` unwraps, mutates the inner agent, then re-wraps.

### HPO: Tournament selection + Mutations

The evolutionary HPO loop follows this order each `evo_steps` interval:

1. **Evaluate** — call `agent.test(env)` on every member of the population to get fitness scores.
2. **Tournament selection** (`TournamentSelection.select(pop)`) — ranks agents by recent fitness, clones the elite (best agent), then fills the rest of the new population by repeated tournament draws from the old population.
3. **Mutation** (`Mutations.mutation(pop)`) — for each member of the new population, randomly samples one of: no-op, parameter noise, HP mutation, architecture add, architecture remove. The elite is never mutated when `mutate_elite=False` (default).

A convenience wrapper `tournament_selection_and_mutation()` in `utils/utils.py` executes steps 2–3 together and is called by all built-in training loops.

## Training Loop Anatomy

All built-in training loops share the same outer structure:

```
while not all agents have reached max_steps:
    for agent in population:
        collect experience / rollouts   →  env.reset(), agent.get_action(), env.step()
        store in buffer (off-policy) or accumulate (on-policy)
        agent.learn(experiences)        →  gradient update(s)
    evaluate population                 →  agent.test(env)
    tournament_selection_and_mutation() →  evolve population
    (optional) save checkpoint
```

| Training loop | Algorithms | Buffer |
|---|---|---|
| `train_off_policy` | DQN, DDPG, TD3, RainbowDQN, CQN | `ReplayBuffer` / `PrioritizedReplayBuffer` / `MultiStepReplayBuffer` |
| `train_on_policy` | PPO | `RolloutBuffer` or explicit list accumulation |
| `train_multi_agent_off_policy` | MADDPG, MATD3 | `MultiAgentReplayBuffer` |
| `train_multi_agent_on_policy` | IPPO | per-agent rollout accumulation |
| `train_offline` | — | static dataset |
| `train_bandits` | NeuralUCB, NeuralTS | `BanditEnv` |
| `train_llm` | GRPO, DPO, SFT | environment-specific (ReasoningGym, SFTGym, PreferenceGym) |

## Adding a New Algorithm

1. Create `agilerl/algorithms/my_algo.py`.
2. Subclass the appropriate base (`RLAlgorithm`, `MultiAgentRLAlgorithm`, or `LLMAlgorithm`) and implement `learn()`, `get_action()`, and `preprocess_observation()`.
3. Inside `__init__`, build your networks as `EvolvableModule` / `EvolvableNetwork` instances and register them:
   ```python
   self.actor = QNetwork(...)
   self.actor_target = QNetwork(...)
   self.optimizer = OptimizerWrapper(optim.Adam, networks=self.actor, lr=self.lr)
   self.register_network_group(
       NetworkGroup(eval_network=self.actor, shared_networks=self.actor_target, policy=True)
   )
   ```
4. Add mutable HPs if desired:
   ```python
   self.registry.register_hyperparameter(
       "lr", RLParameter(min=1e-5, max=1e-2)
   )
   ```
5. Add a corresponding training loop in `agilerl/training/` or reuse an existing one.
6. Add tests in `tests/test_algorithms/` — follow the pattern of existing test files.

Look at `agilerl/algorithms/dqn.py` (off-policy, discrete) or `agilerl/algorithms/ppo.py` (on-policy, continuous) as reference implementations.

## Testing

Tests mirror the source layout. Run a specific test file:

```bash
uv run pytest tests/test_algorithms/test_dqn.py -v
```

LLM tests are marked with `@pytest.mark.llm` and excluded by default in most CI jobs. Run them explicitly with `-m llm`.

### Parallel vLLM testing — `kv_cache_memory_bytes` *and* `gpu_memory_utilization` are both load-bearing

`vllm`- and `gpu`-marked tests share a single pool of four `gputest0..gputest3` xdist groups (see `tests/conftest.py`). With the Linux CI's `-n auto --dist loadgroup` (8 workers on the gha-runner-scale-set node) this caps **total concurrent GPU-touching tests at 4**, while the remaining ~4 workers fan out across CPU-only tests in the **same single `pytest` invocation** — no two-phase split, no manual `coverage combine`. Two reasons for the 4-group cap:

1. **GPU memory.** Each Python-version matrix container has a dedicated ~14.6 GiB GPU. Measured peaks per test: ~4.4 GiB (`test_grpo_move_model_to_vllm`), ~3.0 GiB (`test_grpo_learn`), ~2.5 GiB (`test_grpo_clone_with_accelerator_vllm`); median ~1.5 GiB. With every `VLLMConfig` set to `gpu_memory_utilization≈0.2` + `kv_cache_memory_bytes=32 * 1024 * 1024` (see below), a vLLM worker reserves ~3.2 GiB and a `gpu` (DeepSpeed) test uses ~0.5 GiB; the worst-case "4 vLLM workers" case still fits in ~13 GiB.
2. **Port races.** Each worker's `deepspeed_env` fixture allocates a free `MASTER_PORT` via the standard bind-to-port-0 / close / return dance, which is TOCTOU. Above ~4 concurrent workers the collision rate starts producing `EADDRINUSE` during `torch.distributed.init_process_group`.

`vllm` tests run in `tests/subprocess_runner.py`-spawned subprocesses, so worker-process state is reset between them. `gpu` tests run in-process and can leak DeepSpeed groups / accelerator state to the next test sharing the same group; the per-fixture cleanup (`AcceleratorState._reset_state(True)`, etc.) handles this in practice for the test sets in this repo, but **don't add many more `gpu`-marked tests without re-checking** — DeepSpeed has no clean `destroy_process_group` path, so two `gpu` tests landing on the same worker can surface stale `deepspeed.utils.groups` caches (`Group <ProcessGroup ...> is not registered`) or `EADDRINUSE` on the previous test's still-bound `MASTER_PORT`.

**Every `VLLMConfig` constructed in tests sets two things** that together make intra-container vLLM parallelism safe:

- `kv_cache_memory_bytes=32 * 1024 * 1024` — short-circuits vLLM's `determine_available_memory` profile run, which would otherwise assert GPU free-memory is stable between pre- and post-profile snapshots and fail with `AssertionError: Error in memory profiling. Initial free memory X GiB, current free memory Y GiB` whenever a peer process releases memory mid-profile.
- `gpu_memory_utilization=0.2`-ish (small fraction) — vLLM's `gpu_worker.init_device` runs an upfront `free_memory >= total_memory * gpu_memory_utilization` check **before** the KV-cache path. With the vLLM default `0.9`, each worker would demand ~13 GiB of the 14.6 GiB GPU, so the second concurrent worker would always fail with `Free memory on device (X/14.58 GiB) on startup is less than desired GPU memory utilization (0.9, 13.12 GiB)`. `0.2` → ~2.9 GiB per worker, leaves room for 4 concurrent workers.

**When adding a new vLLM-using test**: route it through `generate_grpo` / `generate_reinforce` (which already set both knobs correctly) or, if you build a `VLLMConfig` directly, copy both the `kv_cache_memory_bytes` and small `gpu_memory_utilization` values from those factories. Without either, the new test will pass locally and flake under xdist.

### Test naming convention

Group tests by the source class method or free function they exercise:

- **Class methods** → `class Test<OwnerClass><MethodName>:` containing `def test_<behavior>(self): ...`.
  Example: tests for `DQN.learn` live in `class TestDQNLearn` as `test_returns_loss_dict`, `test_handles_one_dim_actions`, etc.
- **Free functions** → `class Test<FunctionName>:` when there are multiple behaviours; otherwise a single flat `def test_<funcname>_<behavior>(...)` is fine.
- **Integration tests** spanning multiple methods (e.g. init + learn + clone) → place under the class for the *focus* method (most often the last one called). Name for the scenario: `TestDQNClone.test_after_learning`.
- **Multiple source classes per test file** (e.g. `test_actors.py`) → use the actual class name in each test class; do not collapse to a single `Test<File>` umbrella.
- **Parametrized tests** behave identically on class methods (just add `self`); test IDs are unchanged. Stack `@pytest.mark.parametrize` directly on the method or the class — both work.
- **Pytest fixtures** with `scope="class"` are valid inside test classes when you need shared setup.

Reference examples in this repo: `tests/test_algorithms/test_core_base.py`, `tests/test_algorithms/test_registry.py`, `tests/test_algorithms/test_optimizer_wrapper.py`, and `tests/test_algorithms/test_single_agent/test_dqn.py`.

Apply the convention to **new test files immediately** and to **existing files when you touch them for any other reason**. Do not bulk-migrate untouched files.

### Test naming convention

Group tests by the source class method or free function they exercise:

- **Class methods** → `class Test<OwnerClass><MethodName>:` containing `def test_<behavior>(self): ...`.
  Example: tests for `DQN.learn` live in `class TestDQNLearn` as `test_returns_loss_dict`, `test_handles_one_dim_actions`, etc.
- **Free functions** → `class Test<FunctionName>:` when there are multiple behaviours; otherwise a single flat `def test_<funcname>_<behavior>(...)` is fine.
- **Integration tests** spanning multiple methods (e.g. init + learn + clone) → place under the class for the *focus* method (most often the last one called). Name for the scenario: `TestDQNClone.test_after_learning`.
- **Multiple source classes per test file** (e.g. `test_actors.py`) → use the actual class name in each test class; do not collapse to a single `Test<File>` umbrella.
- **Parametrized tests** behave identically on class methods (just add `self`); test IDs are unchanged. Stack `@pytest.mark.parametrize` directly on the method or the class — both work.
- **Pytest fixtures** with `scope="class"` are valid inside test classes when you need shared setup.

Reference examples in this repo: `tests/test_algorithms/test_core_base.py`, `tests/test_algorithms/test_registry.py`, `tests/test_algorithms/test_optimizer_wrapper.py`, and `tests/test_algorithms/test_single_agent/test_dqn.py`.

Apply the convention to **new test files immediately** and to **existing files when you touch them for any other reason**. Do not bulk-migrate untouched files.

## Git

Do NOT add `Co-Authored-By` lines to commit messages. Do not attribute commits to Claude or other coding agents.

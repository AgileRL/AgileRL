# AgileRL

Deep reinforcement learning library focused on RLOps — evolutionary hyperparameter optimization, population-based training, and LLM finetuning (SFT, DPO, GRPO, CISPO, GSPO, PPO, REINFORCE, ILQL).

## Tooling

This project uses **uv** for dependency management and **just** as the command runner. Always prefix ad-hoc commands with `uv run`:

```bash
just test [pytest args...]        # run pytest via scripts/run-tests.sh
just typecheck                    # static type checking (ty)
just build-docs                   # serve docs locally with auto-rebuild
uv run pytest tests/test_algorithms/test_dqn.py -v   # a single test file
uv run ruff check agilerl/        # lint
uv run ruff format agilerl/       # format
```

Install dependencies: `uv sync` (add `--extra llm`, `--extra arena`, or `--extra all` for optional deps). The repo is a uv workspace: the root `agilerl` package plus `agilerl-arena/` (Arena client + CLI), which ships the `agilerl.arena` namespace portion.

## Type checking

Run `just typecheck` (wraps `uv run ty check agilerl agilerl/arena`). It is fast, needs no GPU, and catches wrong attribute names, nonexistent methods, and bad argument types — run it after every edit, before reaching for the test suite.

- Config lives in `pyproject.toml` under `[tool.ty]`. The model is **strict by default with a legacy opt-out**: every module *not* listed in the legacy override block is fully checked and blocks CI (`.github/workflows/type-checks.yml`). Currently strict: `agilerl/arena`, `agilerl/models`, `metrics.py`, `typing.py`, `protocols.py`, `logger.py`, `train.py`, `agilerl/__init__.py`, and any new module you create outside the legacy list.
- **New files in legacy directories** inherit the opt-out; new modules elsewhere are strict automatically. Never add paths to the legacy list — it only shrinks.
- To migrate a module: delete it from the legacy `include` list in `[tool.ty]`, run `just typecheck`, fix what surfaces. Prefer real fixes (narrowing, correct annotations) over ignores; use `# ty: ignore[rule]` with a trailing reason only where the pattern is genuinely dynamic (registry metaprogramming, optional imports).
- `agilerl/arena` is a gitignored dev symlink to `agilerl-arena/agilerl/arena` (created by `just typecheck` on first run). Static checkers can't see pkgutil-style namespace merging, so the symlink is how `agilerl.arena` resolves; both paths are passed to `ty check` explicitly because ty does not traverse symlinked directories during discovery.
- Both wheels ship `py.typed` (PEP 561), so annotations on public API are part of the shipped product — keep them accurate.

## Repo Structure

```
agilerl/
  algorithms/
    core/
      base.py              # EvolvableAlgorithm / RLAlgorithm / MultiAgentRLAlgorithm / LLMAlgorithm
      registry.py          # MutationRegistry, NetworkGroup, OptimizerConfig, HyperparameterConfig, RLParameter
      optimizer_wrapper.py # OptimizerWrapper — wraps torch optimizers with HPO metadata
      llm_ops/             # fused loss paths and LLM-specific ops
    dqn.py, ddpg.py, ppo.py, td3.py, cqn.py, dqn_rainbow.py, ...   # single-agent RL
    ippo.py, maddpg.py, matd3.py                                   # multi-agent RL
    sft.py, dpo.py, grpo.py, cispo.py, gspo.py, ppo_llm.py,
    reinforce_llm.py, ilql.py, bc_lm.py                            # LLM finetuning
  llm_envs/            # LLM gyms: ReasoningGym, PreferenceGym, SFTGym, HuggingFaceGym,
                       # TokenObservationWrapper, SyncMultiTurnVecEnv, search tools
  models/              # Pydantic manifest specs (algorithms, networks, envs, HPO, TrainingManifest)
  modules/             # EvolvableModule building blocks (MLP, CNN, LSTM, GPT, etc.)
  networks/            # EvolvableNetwork compositions (actors, Q-networks, value nets)
  components/          # ReplayBuffer, RolloutBuffer, Sampler, segment trees, data utils
  rollouts/            # on-policy rollout collection
  training/
    trainer.py         # Trainer ABC -> LocalTrainer, ArenaTrainer (manifest-driven entry points)
    train_off_policy.py, train_on_policy.py, train_offline.py, train_bandits.py
    train_multi_agent_off_policy.py, train_multi_agent_on_policy.py
    train_llm.py, llm/ # LLM loops per paradigm (reasoning, preference, sft, multiturn)
  hpo/
    mutation.py        # Mutations — applies architecture/parameter/HP mutations to a population
    tournament.py      # TournamentSelection — selects elite and new population by fitness
  wrappers/
    agent.py           # AgentWrapper (ABC), RSNorm, AsyncAgentsWrapper
    learning.py        # Skill (curriculum learning env wrapper), BanditEnv
    llm_envs.py        # deprecated shim -> import from agilerl.llm_envs instead
  vector/              # vectorized env wrappers (incl. PettingZoo async vec env)
  metrics.py           # AgentMetrics / MultiAgentMetrics accumulators
  logger.py            # StdOutLogger, WandbLogger, CSVLogger, TensorboardLogger
  population.py        # Population container + MetricsReport
  protocols.py         # Structural typing interfaces (EvolvableAlgorithmProtocol, etc.)
  typing.py            # Shared type aliases
  utils/               # Misc utilities (algo_utils, llm_utils, evolvable_networks, ...)
agilerl-arena/agilerl/arena/   # Arena platform client, CLI, inference, on-prem (namespace portion)
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
└── LLMAlgorithm            language-model finetuning; manages LoRA adapters
    └── GRPO, CISPO, GSPO, DPO, SFT, PPO_LLM, REINFORCE_LLM, ILQL, BC_LM
```

`EvolvableAlgorithm` is responsible for network registration and HPO bookkeeping via `MutationRegistry`, `clone()` for tournament selection, checkpoint save/load, distributed-training wrapping, and the user-extensible `mutation_hook()`.

### Registry (`algorithms/core/registry.py`)

Each algorithm registers its evolvable components declaratively inside `__init__`:

| Call | Purpose |
|---|---|
| `self.register_network_group(NetworkGroup(...))` | Register a network (+ optional shared/target nets) for mutation and cloning |
| `self.registry.register_optimizer(OptimizerConfig(...))` | Record which optimizer maps to which networks and LR attribute |
| `self.registry.register_hyperparameter(name, RLParameter(...))` | Mark a scalar HP as mutable |

`NetworkGroup` stores an `eval_network` (optimised during training), optional `shared_networks` (e.g. target nets that shadow the eval net), and `policy=True` for the network used in `get_action()`. `MutationRegistry` is read by `Mutations` to know what to mutate and by `EvolvableAlgorithm` to know what to clone/checkpoint.

### OptimizerWrapper (`algorithms/core/optimizer_wrapper.py`)

All optimizers in AgileRL algorithms must be wrapped with `OptimizerWrapper`. It instantiates the underlying `torch.optim.Optimizer`, stores `network_names`/`lr_name` so `Mutations` can reinitialise the optimizer after an architecture mutation, and infers attribute names from the calling frame automatically.

### EvolvableModule / EvolvableNetwork

- **`EvolvableModule`** (`modules/base.py`): base for building blocks. Supports architecture mutations (`add_layer`, `remove_layer`, `add_nodes`, `remove_nodes`) and `clone()`.
- **`EvolvableNetwork`** (`networks/base.py`): composes `EvolvableModule`s into a complete encoder–decoder network with a single `forward()`. Used as the `actor`, `critic`, `q_network`, etc. attributes on algorithms.

### AgentWrapper (`wrappers/agent.py`)

`AgentWrapper` wraps an algorithm's `get_action()` and `learn()` with pre/post-processing without subclassing. Concrete wrappers: **`RSNorm`** (online observation normalisation, off-policy only) and **`AsyncAgentsWrapper`** (asynchronous agent stepping in multi-agent envs). Wrappers are transparent to `Mutations`: it unwraps, mutates the inner agent, then re-wraps.

### HPO: Tournament selection + Mutations

Each `evo_steps` interval: **evaluate** the population (`agent.test(env)`), **select** (`TournamentSelection.select(pop)` — clones the elite, fills the rest by tournament draws), **mutate** (`Mutations.mutation(pop)` — per member samples one of no-op, parameter noise, HP mutation, architecture add/remove; the elite is not mutated when `mutate_elite=False`). `tournament_selection_and_mutation()` in `utils/utils.py` executes selection + mutation together and is called by all built-in training loops.

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
```

| Training loop | Algorithms | Buffer / data source |
|---|---|---|
| `train_off_policy` | DQN, DDPG, TD3, RainbowDQN, CQN | `ReplayBuffer` variants |
| `train_on_policy` | PPO | `RolloutBuffer` / rollout accumulation |
| `train_multi_agent_off_policy` | MADDPG, MATD3 | `MultiAgentReplayBuffer` |
| `train_multi_agent_on_policy` | IPPO | per-agent rollout accumulation |
| `train_offline` | — | static dataset |
| `train_bandits` | NeuralUCB, NeuralTS | `BanditEnv` |
| `train_llm` + `training/llm/` | GRPO, DPO, SFT, ... | LLM gyms (`agilerl/llm_envs/`) |

`LocalTrainer` / `ArenaTrainer` (`training/trainer.py`) are the manifest-driven entry points: they parse a `TrainingManifest` (`agilerl/models/`) and dispatch to the loops above.

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
4. Add mutable HPs if desired: `self.registry.register_hyperparameter("lr", RLParameter(min=1e-5, max=1e-2))`.
5. Add a training loop in `agilerl/training/` or reuse an existing one; add a manifest spec in `agilerl/models/algorithms/` (and `agilerl-arena/.../models/algorithms/` if it should be submittable to Arena).
6. Add tests in `tests/test_algorithms/` — follow the pattern of existing test files.

Look at `agilerl/algorithms/dqn.py` (off-policy, discrete) or `agilerl/algorithms/ppo.py` (on-policy, continuous) as reference implementations.

## Testing

Tests mirror the source layout; run through `just test` (which wraps `scripts/run-tests.sh` — pytest with xdist defaults) or `uv run pytest` directly. Arena client tests live in `agilerl-arena/tests/`.

### GPU / vLLM test scheduling

`tests/conftest.py` routes `vllm`- and `gpu`-marked tests into a shared pool of `gputest0..3` xdist groups so GPU memory demand stays bounded; `vllm`-marked tests additionally run in subprocesses (`tests/subprocess_runner.py`). **Every `VLLMConfig` constructed in tests must set `kv_cache_memory_bytes`** (e.g. `32 * 1024 * 1024`): it triggers vLLM's early-return path in `determine_available_memory`, skipping a profiling assertion that flakes whenever a peer process on the same GPU frees memory mid-init. Route new vLLM tests through the existing helpers that already set it, or copy the flag and its comment — without it the test passes locally and flakes under xdist. See the docstrings at the top of `tests/conftest.py` for the full memory model.

### Test naming convention

Group tests by the source class method or free function they exercise:

- **Class methods** → `class Test<OwnerClass><MethodName>:` containing `def test_<behavior>(self): ...`.
  Example: tests for `DQN.learn` live in `class TestDQNLearn` as `test_returns_loss_dict`, `test_handles_one_dim_actions`, etc.
- **Free functions** → `class Test<FunctionName>:` when there are multiple behaviours; otherwise a single flat `def test_<funcname>_<behavior>(...)` is fine.
- **Integration tests** spanning multiple methods (e.g. init + learn + clone) → place under the class for the *focus* method (most often the last one called). Name for the scenario: `TestDQNClone.test_after_learning`.
- **Multiple source classes per test file** (e.g. `test_actors.py`) → use the actual class name in each test class; do not collapse to a single `Test<File>` umbrella.
- **Parametrized tests** behave identically on class methods (just add `self`); stack `@pytest.mark.parametrize` on the method or the class — both work.
- **Pytest fixtures** with `scope="class"` are valid inside test classes when you need shared setup.

Reference examples: `tests/test_algorithms/test_core_base.py`, `tests/test_algorithms/test_registry.py`, `tests/test_algorithms/test_optimizer_wrapper.py`.

Apply the convention to **new test files immediately** and to **existing files when you touch them for any other reason**. Do not bulk-migrate untouched files.

## Git

Do NOT add `Co-Authored-By` lines to commit messages. Do not attribute commits to Claude or other coding agents.

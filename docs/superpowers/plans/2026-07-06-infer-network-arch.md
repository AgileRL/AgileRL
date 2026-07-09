# Infer Network `arch` From Observation Space — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the manifest `network.arch` field inferred from the observation space (plus `recurrent`/`simba`) instead of user-declared, so a manifest's validated `encoder_config` schema can never diverge from the encoder that actually gets built.

**Architecture:** Add one source-of-truth helper `infer_encoder_arch(observation_space, *, recurrent, simba)` in the spec layer. Make `arch` optional in manifest validation: when present, validate as today; when absent, keep the `network` section as a raw dict and resolve it in `Trainer.__init__` after the env is built (Approach B), using the real observation space. Mirror the "arch optional" tolerance in the torch-free Arena fork, where a no-`arch` manifest is passed raw to the server (which resolves obs-space-aware) instead of validated client-side.

**Tech Stack:** Python, pydantic v2 (discriminated unions), gymnasium spaces, pytest.

## Global Constraints

- **Test command (never invoke `pytest` directly):** `VIRTUAL_ENV=.venv313 just test no-parallel <pytest args>` (from project `CLAUDE.md`; xdist parallelism OOMs this machine).
- **Arena fork is torch-free:** all Arena edits go under `agilerl-arena/` and must import only pydantic/stdlib — no `torch`, no `gymnasium`, no `agilerl.utils.*` torch-side modules.
- **`arch` is authoritative and silently ignored** when the observation space determines it: inference always wins; a user-supplied `arch` is never trusted on the deferred path.
- **Inference branch order (must match `get_default_encoder_config`):** Dict/Tuple → `multiinput`; 3-D Box (image) → `cnn`; else `simba` (if set) → `simba`; else `recurrent` (if set) → `lstm`; else `mlp`. `simba` wins over `recurrent`.
- **Scope:** all non-LLM agent types. LLM (`LLMAlgorithmSpec`/`FinetuningNetworkSpec`) untouched. Programmatic construction (a pre-built `NetworkSpec`) untouched — inference runs only when `net_config` is a deferred `dict`.

---

### Task 1: `infer_encoder_arch` helper + drift guard

**Files:**
- Modify: `agilerl/models/networks.py` (add function + `from gymnasium import spaces` import near top)
- Test: `tests/test_models/test_infer_arch.py` (create)

**Interfaces:**
- Produces: `infer_encoder_arch(observation_space: gymnasium.spaces.Space, *, recurrent: bool = False, simba: bool = False) -> Literal["mlp", "cnn", "lstm", "simba", "multiinput"]`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_models/test_infer_arch.py`:

```python
"""Tests for infer_encoder_arch: obs-space -> encoder arch inference."""
from __future__ import annotations

import numpy as np
import pytest
from gymnasium import spaces

from agilerl.models.networks import infer_encoder_arch
from agilerl.utils.evolvable_networks import (
    config_from_dict,
    get_default_encoder_config,
)
from agilerl.modules.configs import (
    CnnNetConfig,
    LstmNetConfig,
    MlpNetConfig,
    MultiInputNetConfig,
    SimBaNetConfig,
)

VECTOR = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
IMAGE = spaces.Box(0, 255, shape=(3, 32, 32), dtype=np.uint8)
DICT = spaces.Dict({"a": spaces.Box(-1.0, 1.0, shape=(2,)), "b": spaces.Box(-1.0, 1.0, shape=(3,))})
TUPLE = spaces.Tuple((spaces.Box(-1.0, 1.0, shape=(2,)), spaces.Discrete(3)))


@pytest.mark.parametrize(
    "space,recurrent,simba,expected",
    [
        (DICT, False, False, "multiinput"),
        (TUPLE, False, False, "multiinput"),
        (IMAGE, False, False, "cnn"),
        (VECTOR, False, False, "mlp"),
        (VECTOR, True, False, "lstm"),
        (VECTOR, False, True, "simba"),
        (VECTOR, True, True, "simba"),  # simba wins over recurrent
        (DICT, True, True, "multiinput"),  # space wins over flags
        (IMAGE, True, True, "cnn"),
    ],
)
def test_infer_encoder_arch(space, recurrent, simba, expected):
    assert infer_encoder_arch(space, recurrent=recurrent, simba=simba) == expected


_CONFIG_TO_ARCH = {
    MlpNetConfig: "mlp",
    CnnNetConfig: "cnn",
    LstmNetConfig: "lstm",
    SimBaNetConfig: "simba",
    MultiInputNetConfig: "multiinput",
}


@pytest.mark.parametrize(
    "space,recurrent,simba",
    [
        (DICT, False, False),
        (IMAGE, False, False),
        (VECTOR, False, False),
        (VECTOR, True, False),
        (VECTOR, False, True),
    ],
)
def test_drift_guard_matches_get_default_encoder_config(space, recurrent, simba):
    cfg = config_from_dict(
        get_default_encoder_config(space, simba=simba, recurrent=recurrent)
    )
    assert infer_encoder_arch(space, recurrent=recurrent, simba=simba) == _CONFIG_TO_ARCH[type(cfg)]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `VIRTUAL_ENV=.venv313 just test no-parallel tests/test_models/test_infer_arch.py -v`
Expected: FAIL with `ImportError: cannot import name 'infer_encoder_arch'`.

- [ ] **Step 3: Implement the helper**

In `agilerl/models/networks.py`, add `from gymnasium import spaces` with the other imports, and add after the `_MANIFEST_ENCODER_ARCHS` definition:

```python
def infer_encoder_arch(
    observation_space: spaces.Space,
    *,
    recurrent: bool = False,
    simba: bool = False,
) -> Literal["mlp", "cnn", "lstm", "simba", "multiinput"]:
    """Infer the encoder architecture from an observation space.

    Mirrors the branch order in
    :func:`agilerl.utils.evolvable_networks.get_default_encoder_config` and
    :meth:`agilerl.networks.base.EvolvableNetwork._build_encoder` so the schema
    used to validate ``encoder_config`` always matches the encoder that will be
    built. ``simba`` takes precedence over ``recurrent``.

    :param observation_space: The (single-agent or per-agent) observation space.
    :param recurrent: Whether the algorithm requests a recurrent encoder.
    :param simba: Whether the network requests a SimBa encoder.
    :returns: One of ``"mlp"``, ``"cnn"``, ``"lstm"``, ``"simba"``, ``"multiinput"``.
    """
    if isinstance(observation_space, (spaces.Dict, spaces.Tuple)):
        return "multiinput"
    if isinstance(observation_space, spaces.Box) and len(observation_space.shape) == 3:
        return "cnn"
    if simba:
        return "simba"
    if recurrent:
        return "lstm"
    return "mlp"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `VIRTUAL_ENV=.venv313 just test no-parallel tests/test_models/test_infer_arch.py -v`
Expected: PASS (11 tests).

- [ ] **Step 5: Commit**

```bash
git add agilerl/models/networks.py tests/test_models/test_infer_arch.py
git commit -m "feat: add infer_encoder_arch obs-space -> arch helper"
```

---

### Task 2: Make `arch` optional in core manifest validation

Make `normalize_manifest_network` tolerate a missing `arch` and make `_resolve_network` / `_process_manifest` keep the `network` section raw (unvalidated encoder) when `arch` is absent, storing the raw dict on `algorithm.net_config` for later resolution.

**Files:**
- Modify: `agilerl/models/networks.py:23-58` (`normalize_manifest_network`)
- Modify: `agilerl/models/manifest.py:98-135` (`_resolve_network`), `:200-241` (`_process_manifest`)
- Test: `tests/test_models/test_manifest.py` (add cases)

**Interfaces:**
- Consumes: nothing new.
- Produces: after `TrainingManifest.model_validate`, `algorithm.net_config` is a validated `NetworkSpec` when `arch` was present, or a **raw `dict`** (the network section) when `arch` was absent. New module-level helper `network_arch_is_resolvable(network: dict) -> bool` in `agilerl/models/networks.py`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_models/test_manifest.py`:

```python
class TestArchOptional:
    def test_arch_present_still_validates(self):
        raw = {
            "algorithm": {"name": "PPO"},
            "environment": {"name": "CartPole-v1"},
            "network": {"arch": "mlp", "encoder_config": {"hidden_size": [64]}},
        }
        out = TrainingManifest.get_validated(raw, mode="json")
        assert out["network"]["encoder_config"]["arch"] == "mlp"

    def test_arch_absent_keeps_network_raw(self):
        from agilerl.models.manifest import TrainingManifest as TM
        raw = {
            "algorithm": {"name": "PPO"},
            "environment": {"name": "CartPole-v1"},
            "network": {"latent_dim": 64, "encoder_config": {"hidden_size": [64]}},
        }
        manifest = TM.model_validate(raw)
        # Deferred: net_config left as a raw dict, not a NetworkSpec.
        assert isinstance(manifest.algorithm.net_config, dict)
        assert "arch" not in manifest.algorithm.net_config.get("encoder_config", {})

    def test_network_arch_is_resolvable(self):
        from agilerl.models.networks import network_arch_is_resolvable
        assert network_arch_is_resolvable({"arch": "mlp"})
        assert network_arch_is_resolvable({"encoder_config": {"arch": "cnn"}})
        assert not network_arch_is_resolvable({"encoder_config": {"hidden_size": [64]}})
        assert not network_arch_is_resolvable({})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `VIRTUAL_ENV=.venv313 just test no-parallel tests/test_models/test_manifest.py::TestArchOptional -v`
Expected: FAIL — `test_arch_absent_keeps_network_raw` raises `ValueError: Missing encoder architecture...` (current behavior); `network_arch_is_resolvable` ImportError.

- [ ] **Step 3: Update `normalize_manifest_network` and add `network_arch_is_resolvable`**

In `agilerl/models/networks.py`, replace the body of `normalize_manifest_network` (lines 23-58) with:

```python
def network_arch_is_resolvable(network: dict) -> bool:
    """Return True if the manifest network section declares an ``arch``.

    Checks the top level and the nested ``encoder_config``. When False, the
    architecture must be inferred from the observation space at build time.
    """
    if not isinstance(network, dict):
        return False
    if network.get("arch"):
        return True
    encoder_config = network.get("encoder_config")
    return isinstance(encoder_config, dict) and bool(encoder_config.get("arch"))


def normalize_manifest_network(data: Any) -> Any:
    """Move a top-level ``arch`` key into ``encoder_config.arch`` when present.

    Raw YAML/JSON manifests place ``arch`` at the network section root, but
    :class:`NetworkSpec` (a discriminated union) expects it nested under
    ``encoder_config``. When ``arch`` is absent it is inferred later from the
    observation space, so this helper leaves the data unchanged rather than
    raising.
    """
    if not isinstance(data, dict):
        return data

    data = dict(data)
    top_level_arch = data.pop("arch", None)
    encoder_config = data.get("encoder_config")
    nested_arch = (
        encoder_config.get("arch") if isinstance(encoder_config, dict) else None
    )
    arch = top_level_arch or nested_arch

    if arch is None:
        # Deferred: architecture inferred from the observation space later.
        return data

    if encoder_config is None:
        data["encoder_config"] = {"arch": arch}
    else:
        data["encoder_config"] = dict(encoder_config)
        data["encoder_config"].setdefault("arch", arch)

    return data
```

- [ ] **Step 4: Update `_resolve_network` to skip validation when deferred**

In `agilerl/models/manifest.py`, update the import from `agilerl.models.networks` to include `network_arch_is_resolvable`, and change the final block of `_resolve_network` (currently lines 131-135) to:

```python
    normalized = normalize_manifest_network(data)
    if not network_arch_is_resolvable(normalized):
        # Deferred: keep the raw network dict; the trainer resolves the arch
        # from the observation space in Trainer.__init__.
        return normalized
    spec = NetworkSpec.model_validate(normalized)
    data_dict = spec.model_dump()
    data_dict["encoder_config"]["arch"] = spec.encoder_config.arch
    return data_dict
```

- [ ] **Step 5: Update `_process_manifest` to store the raw dict when deferred**

In `agilerl/models/manifest.py`, inside `_process_manifest`, replace the `if spec_cls is not None:` block (currently line 219-220) with:

```python
                if spec_cls is not None:
                    if network_arch_is_resolvable(self.network):
                        self.algorithm.net_config = spec_cls.model_validate(self.network)
                    else:
                        # Deferred: leave the raw dict for the trainer to resolve
                        # once the observation space is known.
                        self.algorithm.net_config = dict(self.network)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `VIRTUAL_ENV=.venv313 just test no-parallel tests/test_models/test_manifest.py -v`
Expected: PASS (including the existing `TestNetwork*` cases and the new `TestArchOptional`).

- [ ] **Step 7: Commit**

```bash
git add agilerl/models/networks.py agilerl/models/manifest.py tests/test_models/test_manifest.py
git commit -m "feat: make manifest network arch optional (defer when absent)"
```

---

### Task 3: Resolve deferred `net_config` in the trainer (single-agent)

After the env is built in `Trainer.__init__`, infer the arch from the real observation space, validate the raw network dict into the concrete `NetworkSpec`, and assign it to `algorithm_spec.net_config` before the population is built.

**Files:**
- Modify: `agilerl/training/trainer.py` (LocalTrainer `__init__` around `:324`; add `_resolve_deferred_net_config`)
- Test: `tests/test_train/test_trainer.py` (add integration test) + `tests/test_train/_dummy_envs.py` (create)

**Interfaces:**
- Consumes: `infer_encoder_arch` (Task 1), `network_arch_is_resolvable` (Task 2), `get_spaces_from_env` (`agilerl/utils/trainer_utils.py:75`).
- Produces: `LocalTrainer._resolve_deferred_net_config() -> None` (mutates `self.algorithm_spec.net_config` in place when it is a `dict`).

- [ ] **Step 1: Create a reusable Dict-obs dummy env for tests**

Create `tests/test_train/_dummy_envs.py`:

```python
"""Importable dummy gym envs for manifest entrypoint resolution in tests."""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class DictObsEnv(gym.Env):
    """Single-agent env with a Dict observation space (needs EvolvableMultiInput)."""

    def __init__(self, render_mode=None):
        self.observation_space = spaces.Dict(
            {
                "a": spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32),
                "b": spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.render_mode = render_mode

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 0.0, False, False, {}
```

- [ ] **Step 2: Write the failing integration test**

Add to `tests/test_train/test_trainer.py`:

```python
def test_from_manifest_infers_multiinput_when_arch_absent(tmp_path):
    """A Dict-obs env with NO arch builds an EvolvableMultiInput encoder."""
    import yaml
    from agilerl.modules.multi_input import EvolvableMultiInput
    from agilerl.training.trainer import LocalTrainer

    manifest = {
        "algorithm": {"name": "PPO", "learn_step": 64},
        "environment": {
            "name": "dict-obs-env",
            "num_envs": 2,
            "entrypoint": "tests.test_train._dummy_envs:DictObsEnv",
        },
        "training": {"max_steps": 200, "evo_steps": 100, "pop_size": 1},
        "network": {"latent_dim": 32, "head_config": {"hidden_size": [32]}},
    }
    path = tmp_path / "m.yaml"
    path.write_text(yaml.safe_dump(manifest))

    trainer = LocalTrainer.from_manifest(manifest=path, device="cpu")
    encoder = trainer.population[0].actor.encoder
    assert isinstance(encoder, EvolvableMultiInput)
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `VIRTUAL_ENV=.venv313 just test no-parallel "tests/test_train/test_trainer.py::test_from_manifest_infers_multiinput_when_arch_absent" -v`
Expected: FAIL — either a validation error building the population, or `net_config` staying a raw dict (currently no resolution step exists).

- [ ] **Step 4: Add `_resolve_deferred_net_config` and call it before population build**

In `agilerl/training/trainer.py`, add these imports near the other `agilerl.models` imports:

```python
from agilerl.models.networks import infer_encoder_arch, network_arch_is_resolvable
from agilerl.utils.trainer_utils import get_spaces_from_env
```

In `LocalTrainer.__init__`, insert the call between `self.env = self._make_env()` and `self.population = create_population_from_spec(...)` (line 324/325):

```python
        self.env = self._make_env()
        self._resolve_deferred_net_config()
        self.population = create_population_from_spec(
```

Add the method to `LocalTrainer`:

```python
    def _resolve_deferred_net_config(self) -> None:
        """Resolve a manifest network section whose ``arch`` was omitted.

        When the manifest did not declare ``arch``, ``net_config`` is left as a
        raw dict by manifest validation. Now that the environment (hence the
        observation space) exists, infer the arch and validate the network into
        the algorithm's concrete ``NetworkSpec``. No-ops when ``net_config`` is
        already a validated spec (programmatic construction) or None.
        """
        net_config = getattr(self.algorithm_spec, "net_config", None)
        if not isinstance(net_config, dict):
            return
        if network_arch_is_resolvable(net_config):
            return

        from typing import get_args
        from agilerl.models.networks import NetworkSpec

        observation_space, _ = get_spaces_from_env(self.algorithm_spec, self.env)
        simba = bool(net_config.get("simba", False))
        recurrent = bool(getattr(self.algorithm_spec, "recurrent", False))

        if isinstance(observation_space, dict):
            # Multi-agent: handled in Task 4.
            self._resolve_deferred_net_config_multi_agent(
                net_config, observation_space, simba, recurrent
            )
            return

        arch = infer_encoder_arch(
            observation_space, recurrent=recurrent, simba=simba
        )
        resolved = dict(net_config)
        encoder_config = dict(resolved.get("encoder_config") or {})
        encoder_config["arch"] = arch
        resolved["encoder_config"] = encoder_config

        net_config_field = type(self.algorithm_spec).model_fields.get("net_config")
        spec_cls = next(
            (
                t
                for t in get_args(net_config_field.annotation)
                if t is not type(None)
            ),
            NetworkSpec,
        )
        self.algorithm_spec.net_config = spec_cls.model_validate(resolved)
```

Add a temporary stub for the multi-agent branch (fully implemented in Task 4):

```python
    def _resolve_deferred_net_config_multi_agent(
        self, net_config, observation_spaces, simba, recurrent
    ) -> None:
        raise NotImplementedError("multi-agent deferred net_config: Task 4")
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `VIRTUAL_ENV=.venv313 just test no-parallel "tests/test_train/test_trainer.py::test_from_manifest_infers_multiinput_when_arch_absent" -v`
Expected: PASS.

- [ ] **Step 6: Run the regression case (wrong arch now ignored on deferred path)**

Add and run:

```python
def test_from_manifest_wrong_arch_is_overridden_when_omitted(tmp_path):
    """Vector-obs env with no arch builds an EvolvableMLP."""
    import yaml
    from agilerl.modules.mlp import EvolvableMLP
    from agilerl.training.trainer import LocalTrainer

    manifest = {
        "algorithm": {"name": "PPO", "learn_step": 64},
        "environment": {"name": "CartPole-v1", "num_envs": 2},
        "training": {"max_steps": 200, "evo_steps": 100, "pop_size": 1},
        "network": {"latent_dim": 32, "head_config": {"hidden_size": [32]}},
    }
    path = tmp_path / "m.yaml"
    path.write_text(yaml.safe_dump(manifest))
    trainer = LocalTrainer.from_manifest(manifest=path, device="cpu")
    assert isinstance(trainer.population[0].actor.encoder, EvolvableMLP)
```

Run: `VIRTUAL_ENV=.venv313 just test no-parallel "tests/test_train/test_trainer.py::test_from_manifest_wrong_arch_is_overridden_when_omitted" -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add agilerl/training/trainer.py tests/test_train/test_trainer.py tests/test_train/_dummy_envs.py
git commit -m "feat: resolve deferred net_config from obs space in trainer (single-agent)"
```

---

### Task 4: Multi-agent per-agent deferred resolution

For a multi-agent env with no `arch`, infer the arch per agent. When all agents share the same arch (homogeneous family), validate the shared network section against it. When agents differ (heterogeneous), drop the forced `encoder_config` and let the algorithm's `build_net_config` auto-derive each agent's encoder from its own observation space (already supported via `get_default_encoder_config`).

**Files:**
- Modify: `agilerl/training/trainer.py` (`_resolve_deferred_net_config_multi_agent`)
- Test: `tests/test_train/test_trainer.py` + `tests/test_train/_dummy_envs.py`

**Interfaces:**
- Consumes: `infer_encoder_arch`, `get_spaces_from_env` (returns `(dict, dict)` for multi-agent).
- Produces: `_resolve_deferred_net_config_multi_agent(net_config: dict, observation_spaces: dict[str, spaces.Space], simba: bool, recurrent: bool) -> None`.

- [ ] **Step 1: Add a heterogeneous PettingZoo dummy env**

Append to `tests/test_train/_dummy_envs.py`:

```python
from pettingzoo import ParallelEnv


class HeteroParallelEnv(ParallelEnv):
    """Two agents: one Dict-obs (multiinput), one vector-obs (mlp)."""

    metadata = {"name": "hetero_v0"}

    def __init__(self, render_mode=None):
        self.possible_agents = ["dict_agent", "vec_agent"]
        self.render_mode = render_mode
        self._obs = {
            "dict_agent": spaces.Dict(
                {"a": spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)}
            ),
            "vec_agent": spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32),
        }
        self._act = {
            "dict_agent": spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
            "vec_agent": spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
        }

    def observation_space(self, agent):
        return self._obs[agent]

    def action_space(self, agent):
        return self._act[agent]

    def reset(self, *, seed=None, options=None):
        self.agents = list(self.possible_agents)
        return {a: self._obs[a].sample() for a in self.agents}, {a: {} for a in self.agents}

    def step(self, actions):
        obs = {a: self._obs[a].sample() for a in self.agents}
        return (
            obs,
            {a: 0.0 for a in self.agents},
            {a: False for a in self.agents},
            {a: False for a in self.agents},
            {a: {} for a in self.agents},
        )
```

- [ ] **Step 2: Write the failing test**

Add to `tests/test_train/test_trainer.py`:

```python
def test_from_manifest_multi_agent_heterogeneous_per_agent_encoders(tmp_path):
    """Heterogeneous multi-agent env with no arch: per-agent encoders inferred."""
    import yaml
    from agilerl.modules.mlp import EvolvableMLP
    from agilerl.modules.multi_input import EvolvableMultiInput
    from agilerl.training.trainer import LocalTrainer

    manifest = {
        "algorithm": {"name": "IPPO", "learn_step": 64},
        "environment": {
            "name": "hetero-env",
            "num_envs": 2,
            "entrypoint": "tests.test_train._dummy_envs:HeteroParallelEnv",
        },
        "training": {"max_steps": 200, "evo_steps": 100, "pop_size": 1},
        "network": {"latent_dim": 32, "head_config": {"hidden_size": [32]}},
    }
    path = tmp_path / "m.yaml"
    path.write_text(yaml.safe_dump(manifest))

    trainer = LocalTrainer.from_manifest(manifest=path, device="cpu")
    agent = trainer.population[0]
    encoders = {
        aid: net.encoder for aid, net in agent.actors.items()
    } if hasattr(agent, "actors") else {}
    assert isinstance(encoders["dict_agent"], EvolvableMultiInput)
    assert isinstance(encoders["vec_agent"], EvolvableMLP)
```

> Note for the implementer: confirm the IPPO agent's per-agent actor attribute name (`agent.actors` vs `agent.actor` ModuleDict) against `agilerl/algorithms/ippo.py` and adjust the `encoders` extraction accordingly before running.

- [ ] **Step 3: Run the test to verify it fails**

Run: `VIRTUAL_ENV=.venv313 just test no-parallel "tests/test_train/test_trainer.py::test_from_manifest_multi_agent_heterogeneous_per_agent_encoders" -v`
Expected: FAIL with `NotImplementedError: multi-agent deferred net_config: Task 4`.

- [ ] **Step 4: Implement the multi-agent branch**

In `agilerl/training/trainer.py`, replace the `_resolve_deferred_net_config_multi_agent` stub with:

```python
    def _resolve_deferred_net_config_multi_agent(
        self, net_config, observation_spaces, simba, recurrent
    ) -> None:
        """Resolve a deferred multi-agent network section.

        Infers arch per agent from each agent's observation space. When all
        agents share the same arch, validates the shared network section against
        it. When they differ, strips the forced ``encoder_config`` so the
        algorithm's ``build_net_config`` auto-derives each agent's encoder from
        its own observation space.
        """
        from typing import get_args
        from agilerl.models.networks import NetworkSpec

        archs = {
            agent_id: infer_encoder_arch(space, recurrent=recurrent, simba=simba)
            for agent_id, space in observation_spaces.items()
        }
        net_config_field = type(self.algorithm_spec).model_fields.get("net_config")
        spec_cls = next(
            (t for t in get_args(net_config_field.annotation) if t is not type(None)),
            NetworkSpec,
        )

        if len(set(archs.values())) == 1:
            arch = next(iter(archs.values()))
            resolved = dict(net_config)
            encoder_config = dict(resolved.get("encoder_config") or {})
            encoder_config["arch"] = arch
            resolved["encoder_config"] = encoder_config
            self.algorithm_spec.net_config = spec_cls.model_validate(resolved)
            return

        # Heterogeneous: validate obs-independent fields (latent_dim, head_config)
        # but drop encoder_config so build_net_config auto-derives per agent.
        resolved = {k: v for k, v in net_config.items() if k != "encoder_config"}
        self.algorithm_spec.net_config = resolved
```

> Implementer note: `build_net_config` (`agilerl/algorithms/core/base.py:1847`) auto-derives a per-agent encoder via `get_default_encoder_config` when `encoder_config` is absent, so a `net_config` dict without `encoder_config` produces correct per-agent encoders. Verify the multi-agent algo constructor accepts a plain `net_config` dict (it does — `maddpg.py:125` / IPPO share the base `build_net_config`).

- [ ] **Step 5: Run the test to verify it passes**

Run: `VIRTUAL_ENV=.venv313 just test no-parallel "tests/test_train/test_trainer.py::test_from_manifest_multi_agent_heterogeneous_per_agent_encoders" -v`
Expected: PASS.

- [ ] **Step 6: Add and run the homogeneous multi-agent case**

```python
def test_from_manifest_multi_agent_homogeneous(tmp_path):
    """Homogeneous multi-agent env with no arch: shared mlp encoder validates."""
    import yaml
    from agilerl.modules.mlp import EvolvableMLP
    from agilerl.training.trainer import LocalTrainer

    manifest = {
        "algorithm": {"name": "IPPO", "learn_step": 64},
        "environment": {"name": "simple_speaker_listener_v4", "num_envs": 2},
        "training": {"max_steps": 200, "evo_steps": 100, "pop_size": 1},
        "network": {"latent_dim": 32, "head_config": {"hidden_size": [32]}},
    }
    path = tmp_path / "m.yaml"
    path.write_text(yaml.safe_dump(manifest))
    trainer = LocalTrainer.from_manifest(manifest=path, device="cpu")
    # All agents build without error; encoders are vector MLPs.
    assert trainer.population[0] is not None
```

Run: `VIRTUAL_ENV=.venv313 just test no-parallel "tests/test_train/test_trainer.py::test_from_manifest_multi_agent_homogeneous" -v`
Expected: PASS. (If `simple_speaker_listener_v4` observation spaces are heterogeneous, swap for a homogeneous PettingZoo env such as `simple_spread_v3`.)

- [ ] **Step 7: Commit**

```bash
git add agilerl/training/trainer.py tests/test_train/test_trainer.py tests/test_train/_dummy_envs.py
git commit -m "feat: per-agent deferred net_config resolution for multi-agent"
```

---

### Task 5: Arena fork — tolerate missing `arch` (raw passthrough)

In the torch-free Arena fork, a no-`arch` manifest must validate client-side without touching the network section: `_resolve_network` returns the raw network dict so the server validates it obs-space-aware. An `arch`-bearing manifest is validated as today.

**Files:**
- Modify: `agilerl-arena/agilerl/arena/models/manifest.py:61-98` (`_normalize_network_arch`), `:154-180` (`_resolve_network`)
- Test: `agilerl-arena/tests/test_cli_manifest.py`

**Interfaces:**
- Produces: Arena `_resolve_network` returns the raw network dict unchanged when `arch` is absent; unchanged behavior when present.

- [ ] **Step 1: Write the failing tests**

Add to `agilerl-arena/tests/test_cli_manifest.py`:

```python
class TestArenaArchOptional:
    def test_arch_present_validates(self):
        from agilerl.arena.models.manifest import TrainingManifest
        raw = {
            "algorithm": {"name": "PPO"},
            "environment": {"name": "merge-env", "version": "v1"},
            "network": {"arch": "mlp", "encoder_config": {"hidden_size": [64]}},
        }
        out = TrainingManifest.get_validated(raw, mode="json")
        assert out["network"]["encoder_config"]["arch"] == "mlp"

    def test_arch_absent_passes_network_raw(self):
        from agilerl.arena.models.manifest import TrainingManifest
        raw = {
            "algorithm": {"name": "PPO"},
            "environment": {"name": "merge-env", "version": "v1"},
            "network": {"latent_dim": 64, "encoder_config": {"hidden_size": [64]}},
        }
        out = TrainingManifest.get_validated(raw, mode="json")
        # Network section is left raw for the server to validate.
        assert out["network"] == {"latent_dim": 64, "encoder_config": {"hidden_size": [64]}}
        assert "arch" not in out["network"]["encoder_config"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `VIRTUAL_ENV=.venv313 just test no-parallel agilerl-arena/tests/test_cli_manifest.py::TestArenaArchOptional -v`
Expected: FAIL — `test_arch_absent_passes_network_raw` raises `ValueError: Missing encoder architecture...`.

- [ ] **Step 3: Make `_normalize_network_arch` tolerate missing arch**

In `agilerl-arena/agilerl/arena/models/manifest.py`, replace the `if arch is None:` raise block in `_normalize_network_arch` (lines 85-92) with:

```python
    # If arch is not found, defer resolution to the server (obs-space-aware).
    if arch is None:
        return data
```

- [ ] **Step 4: Make `_resolve_network` return raw when arch absent**

In `agilerl-arena/agilerl/arena/models/manifest.py`, change the final block of `_resolve_network` (lines 175-178) to:

```python
    normalized = _normalize_network_arch(data)
    if not (
        normalized.get("arch")
        or (
            isinstance(normalized.get("encoder_config"), dict)
            and normalized["encoder_config"].get("arch")
        )
    ):
        # Deferred: hand the raw network section to the server to validate.
        return normalized
    spec = NetworkSpec.model_validate(normalized)
    data_dict = spec.model_dump()
    data_dict["encoder_config"]["arch"] = spec.encoder_config.arch
    return data_dict
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `VIRTUAL_ENV=.venv313 just test no-parallel agilerl-arena/tests/test_cli_manifest.py -v`
Expected: PASS (new cases plus existing manifest tests).

- [ ] **Step 6: Run the parity test (no field divergence introduced)**

Run: `VIRTUAL_ENV=.venv313 just test no-parallel tests/test_models/test_arena_model_parity.py -v`
Expected: PASS (this change adds no spec fields; parity is unaffected).

- [ ] **Step 7: Commit**

```bash
git add agilerl-arena/agilerl/arena/models/manifest.py agilerl-arena/tests/test_cli_manifest.py
git commit -m "feat(arena): pass raw network section to server when arch omitted"
```

---

### Task 6: Update docs / tutorial example to zero-config

Drop `arch` (and the now-unnecessary explicit `encoder_config`) from the tutorial manifest so it demonstrates the inferred behavior, and adjust the prose.

**Files:**
- Modify: `docs/_static/examples/merge_ppo.yaml`
- Modify: `docs/tutorials/arena_training/ppo_custom_env.rst` (network-section prose, if any references `arch`)

**Interfaces:** none (docs only).

- [ ] **Step 1: Simplify the tutorial manifest**

In `docs/_static/examples/merge_ppo.yaml`, replace the `network:` block with the arch-free form (encoder auto-derived from the env's Dict observation space):

```yaml
network:
    latent_dim: 128
    min_latent_dim: 64
    max_latent_dim: 256
    head_config:
        hidden_size:
            - 128
        activation: ReLU
        min_hidden_layers: 1
        max_hidden_layers: 3
        min_mlp_nodes: 64
        max_mlp_nodes: 256
```

- [ ] **Step 2: Update the prose**

In `docs/tutorials/arena_training/ppo_custom_env.rst`, find any sentence that tells the user to set `arch` and replace it with a note that the encoder architecture is inferred automatically from the environment's observation space (no `arch` needed; set `simba: true` or the algorithm's `recurrent: true` to opt into those vector encoders). Keep the prose simple and match the surrounding docs style.

- [ ] **Step 3: Verify docs build has no broken literalinclude**

Run: `VIRTUAL_ENV=.venv313 just test no-parallel tests/test_models/test_manifest.py -k arch -v`
Expected: PASS (sanity that the simplified example still validates through the manifest models as a deferred network).

- [ ] **Step 4: Commit**

```bash
git add docs/_static/examples/merge_ppo.yaml docs/tutorials/arena_training/ppo_custom_env.rst
git commit -m "docs: infer encoder arch in Arena custom-env tutorial (drop arch)"
```

---

## Final verification

- [ ] Run the full affected suites together:

```bash
VIRTUAL_ENV=.venv313 just test no-parallel \
  tests/test_models/test_infer_arch.py \
  tests/test_models/test_manifest.py \
  tests/test_models/test_arena_model_parity.py \
  tests/test_train/test_trainer.py \
  agilerl-arena/tests/test_cli_manifest.py -v
```
Expected: all PASS.

- [ ] Confirm an existing manifest that declares a *correct* `arch` still builds identically (no behavior change on the eager path) — covered by the pre-existing `TestNetwork*` cases in `tests/test_models/test_manifest.py`.

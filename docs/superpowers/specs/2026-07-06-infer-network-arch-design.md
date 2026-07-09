# Design: Infer network `arch` from the observation space

Date: 2026-07-06
Branch: `feature/infer-network-arch`

## Problem

A training manifest's `network` section requires users to declare the encoder
architecture explicitly:

```yaml
network:
    arch: mlp          # <-- user must pick this
    encoder_config:
        hidden_size: [64]
        ...
```

But the actual encoder is **not** chosen from `arch` at build time. In
`EvolvableNetwork._build_encoder` (`agilerl/networks/base.py:513-562`) the
encoder class is selected purely from the observation space plus two flags:

| Signal | Encoder built | implied arch |
|---|---|---|
| `Dict` / `Tuple` obs | `EvolvableMultiInput` | `multiinput` |
| image obs (`is_image_space`) | `EvolvableCNN` | `cnn` |
| vector obs + `simba=True` | `EvolvableSimBa` | `simba` |
| vector obs + `recurrent=True` | `EvolvableLSTM` | `lstm` |
| vector obs (default) | `EvolvableMLP` | `mlp` |

`arch` only selects **which pydantic schema validates `encoder_config`** during
manifest parsing (the discriminated union `EncoderType`, keyed on `arch`, in
`agilerl/models/networks.py`). When the declared `arch` disagrees with what
`_build_encoder` actually constructs, the validated config carries the wrong
fields and the build crashes. Concrete failure we hit: a `Dict`-observation env
with `arch: mlp` validates `encoder_config` as `MlpSpec` (with `hidden_size`),
then `_build_encoder` constructs `EvolvableMultiInput`, which rejects
`hidden_size` -> `TypeError: EvolvableMultiInput.__init__() got an unexpected
keyword argument 'hidden_size'`.

## Goal

Make `arch` an **inferred, authoritative** value derived from the same signals
`_build_encoder` uses (observation space + `recurrent` + `simba`), so users
never declare it and the validated schema can never diverge from the built
encoder. `arch` in a manifest becomes optional and is **silently ignored**.

## Key existing machinery to reuse

- `agilerl.utils.evolvable_networks.get_default_encoder_config(observation_space,
  simba=False, recurrent=False, layer_norm=True)` already maps an observation
  space to a default encoder config dataclass, with the exact branch order
  above (Dict/Tuple -> image -> simba -> recurrent -> mlp). This is the
  algorithm-layer equivalent of the inference we need.
- Multi-agent algorithms already resolve **per-agent** encoder configs from a
  single (or `None`) `net_config` via
  `RLAlgorithm.build_net_config` (`agilerl/algorithms/core/base.py:1847`),
  including auto-deriving per agent when `net_config is None`.
- `agilerl.utils.trainer_utils.get_spaces_from_env(algo_spec, env)`
  (`:75`) already dispatches by agent type and returns:
  - single-agent: `(single_observation_space, single_action_space)`
  - multi-agent: `({agent: obs_space}, {agent: action_space})`
- `is_image_space` exists in both `agilerl/utils/evolvable_networks.py:74` and
  `agilerl/utils/algo_utils.py:304`.

The obs->encoder inference therefore already exists at the algorithm layer. The
gap is entirely in the **manifest validation layer**, which hard-requires
`arch` before any environment (hence observation space) is available.

## Design

### 1. Single source of truth: `infer_encoder_arch`

Add to `agilerl/models/networks.py`:

```python
def infer_encoder_arch(
    observation_space: spaces.Space,
    *,
    recurrent: bool = False,
    simba: bool = False,
) -> Literal["mlp", "cnn", "lstm", "simba", "multiinput"]:
    ...
```

Branch order mirrors `get_default_encoder_config` **exactly** (simba wins over
recurrent):

1. `Dict` / `Tuple` -> `"multiinput"`
2. `is_image_space(space)` -> `"cnn"`
3. `simba` -> `"simba"`
4. `recurrent` -> `"lstm"`
5. else -> `"mlp"`

Uses only gymnasium space checks + `is_image_space`, keeping the spec layer
light. A drift-guard test (section 6) asserts it agrees with
`get_default_encoder_config` across all space types so the two cannot silently
diverge. (Refactoring both onto one shared classifier is a possible follow-up,
not in scope.)

### 2. `arch` becomes optional and ignored

`normalize_manifest_network` (`agilerl/models/networks.py:23`) no longer raises
when `arch` is absent, and never consumes a user-supplied `arch` as the
discriminator. Existing manifests (and docs) that still carry `arch` continue
to parse; the value is discarded. `EncoderType`'s `arch` discriminator is still
used **internally** to validate `encoder_config` once the arch is known
(section 3) - but that arch now comes from inference, not the user.

### 3. Defer encoder validation to post-env (Approach B)

Because `encoder_config`'s schema can't be discriminated without the
observation space, `TrainingManifest` stops eagerly validating it for non-LLM
algorithms:

- `_resolve_network` / `TrainingManifest._process_manifest`
  (`agilerl/models/manifest.py:200`): keep the `network` section as a raw dict
  and leave `algorithm.net_config` **unresolved** (a raw dict) when the
  algorithm is non-LLM. Obs-independent parts (`latent_dim`, `min/max_latent_dim`,
  `head_config`, `simba`, `random_seed`) may still be validated eagerly; the
  `encoder_config` sub-dict is carried raw.
- `Trainer.__init__` (`agilerl/training/trainer.py`, right after the existing
  `self.env = self._make_env()` and before `create_population_from_spec`):
  resolve the deferred network using the already-built env:
  1. `obs_space(s) = get_spaces_from_env(self.algorithm_spec, self.env)[0]`
     (no extra env instantiation; obs space is post-wrappers-authoritative).
  2. infer arch via `infer_encoder_arch(...)` using
     `getattr(self.algorithm_spec, "recurrent", False)` and the network's
     `simba` flag.
  3. inject arch, validate `encoder_config` into the concrete spec, and set
     `self.algorithm_spec.net_config` to the validated `NetworkSpec`.

If `net_config` is already a validated `NetworkSpec` instance (programmatic /
direct construction, not from a manifest), it is left untouched - inference is
**manifest-only**. The deferred vs resolved state is represented naturally by
`net_config` being a `dict` vs a `NetworkSpec` instance.

### 4. Multi-agent: per-agent schemas

For `MultiAgentRLAlgorithmSpec`, infer arch **per agent** from each agent's
observation space (from the `get_spaces_from_env` dict) and validate a
per-agent `encoder_config` mapping, leaning on the existing
`build_net_config`, which already accepts
`{agent_id: {encoder_config: ...}}` and auto-derives per agent when the config
is `None`.

- Homogeneous env + single shared `encoder_config` in the manifest: allowed
  (matches the existing `MultiAgentSetup.HOMOGENEOUS` assertion in
  `build_net_config`); the shared config is validated against the (common)
  inferred arch.
- Heterogeneous agents (e.g. one `Dict`, one vector): each agent's network is
  validated/built against its own inferred arch. This is expressible now that
  the manifest carries per-agent encoder configs, which a single-schema
  manifest could not express before.
- Manifest omits `encoder_config` entirely: pass through to `build_net_config`
  as `None` and let the algorithm auto-derive per agent.

### 5. Arena fork (client-side manifest models)

The Arena backend **already performs its own obs-space-aware resolution** when
it validates/profiles an uploaded environment (server-side, via the
`/api/cli/v1/environments/profile` and validation endpoints) - so it can fill
`arch` from the real observation space itself. The gap is purely **client-side**:
the Arena fork's pydantic models still hard-require `arch`
(`_normalize_network_arch` raises when it is missing), so a new zero-config
(no-`arch`) manifest would be rejected locally before it is ever submitted.

Changes (all under `agilerl-arena/`, per project convention that Arena edits
live in that directory - it is a deliberately **torch-free** fork, pydantic-only,
with no gymnasium/observation-space access):

- `agilerl-arena/agilerl/arena/models/manifest.py`: split the network path on
  whether `arch` is present.
  - **`arch` provided:** validate the network section client-side exactly as
    today (`_normalize_network_arch` + discriminated `NetworkSpec`).
  - **`arch` absent:** do **no** client-side validation of the network section -
    `_resolve_network` returns the raw network dict unchanged, and it is
    submitted as-is for the **server** to validate (obs-space-aware).
    `_normalize_network_arch` no longer raises on a missing `arch`.
- `agilerl-arena/agilerl/arena/models/networks.py`: no schema change needed for
  the deferred path (the raw dict bypasses `NetworkSpec` entirely); the
  discriminated `NetworkSpec` continues to require `arch` on the eager,
  arch-provided path.

Critically, the fork does **not** run `infer_encoder_arch` - it has no local
observation space. When `arch` is absent it validates **nothing** about the
network section and hands the raw dict to the backend, which resolves and
validates it obs-space-aware. When `arch` *is* present, client-side validation
is unchanged.

### 6. Scope & compatibility

- **In scope:** all non-LLM agent types - single-agent (`RLAlgorithmSpec`),
  multi-agent (`MultiAgentRLAlgorithmSpec`), offline (GymEnvSpec subclass), and
  bandit (`BanditEnvSpec`). Bandit/offline reuse the single-agent obs-space
  path via `get_spaces_from_env`.
- **Untouched:** LLM algorithms (`LLMAlgorithmSpec` / `FinetuningNetworkSpec`) -
  no encoder inference; that path already bypasses the encoder discriminator.
- **Programmatic construction** (passing a built `NetworkSpec`/algorithm to the
  trainer directly) is unaffected - inference only runs when `net_config` is a
  deferred dict.
- **Docs:** update `docs/_static/examples/merge_ppo.yaml` to drop `arch` (and
  the now-unnecessary explicit `encoder_config` where auto-derivation
  suffices), demonstrating the new zero-config behavior in the Arena tutorial.

### 7. Testing

- **Unit** (`infer_encoder_arch`): every space type (`Box` 1D, `Box` image,
  `Dict`, `Tuple`, `Discrete`) x `simba`/`recurrent` combinations, asserting the
  simba-over-recurrent precedence.
- **Drift guard:** for each space type, assert `infer_encoder_arch` agrees with
  the arch implied by `get_default_encoder_config`.
- **Manifest -> trainer integration:** a manifest with **no** `arch`:
  - `Dict`-obs env (MergeEnv-style) builds `EvolvableMultiInput`
  - vector-obs env builds `EvolvableMLP`
  - image-obs env builds `EvolvableCNN`
  - recurrent PPO builds `EvolvableLSTM`
  - `simba: true` builds `EvolvableSimBa`
  - multi-agent heterogeneous env builds the correct per-agent encoders
- **Regression:** a manifest with a *wrong* `arch: mlp` on a `Dict`-obs env now
  builds `EvolvableMultiInput` successfully instead of crashing.
- **No-regression:** existing manifests that declare a *correct* `arch` still
  build identically.
- **Arena fork:** a no-`arch` manifest passes through the Arena
  `TrainingManifest` (client-side) with the network section left
  **raw/unvalidated** and serialized as-is for the server; a manifest **with**
  `arch` still gets full client-side validation and serializes identically to
  today.

## Non-goals

- Refactoring `_build_encoder` / `get_default_encoder_config` onto a single
  shared classifier (possible follow-up).
- Any change to LLM network handling.
- Changing programmatic (non-manifest) construction behavior.

## Risk / open considerations

- `infer_encoder_arch` and `get_default_encoder_config` must stay in lockstep;
  the drift-guard test enforces this.
- Deferring `encoder_config` validation moves some manifest errors from
  parse-time to trainer-construction-time (after the env is built). Acceptable:
  the env must be built to know the space anyway.
- The core `infer_encoder_arch` and the Arena backend's server-side resolution
  must agree. The Arena fork in this repo does not infer (it only tolerates a
  missing `arch`), so there is no duplicated inference to drift here; the
  backend's own resolution (out of repo) should mirror `get_default_encoder_config`.

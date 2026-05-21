"""Gymnasium wrappers for MuJoCo Playground (MJX) environments.

Provides two wrappers:

* ``PlaygroundGymWrapper`` — wraps a single MJX env as ``gymnasium.Env``.
* ``BatchedPlaygroundVectorEnv`` — wraps a single MJX env and uses
  ``jax.vmap`` to step *all* environments in one GPU call, exposing a
  ``gymnasium.vector.VectorEnv`` interface.

When a ``torch_device`` is provided, the batched variant returns PyTorch
CUDA tensors for observations via DLPack zero-copy transfer, eliminating
the GPU→CPU→GPU round-trip that would otherwise occur when feeding
observations into a PyTorch policy network.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import jax
import jax.experimental.layout as _layout
import jax.numpy as jnp
import numpy as np
import torch
from gymnasium import spaces
from gymnasium.vector.utils import batch_space

# Flax 0.12.x references ``jax.experimental.layout.Format`` which was
# removed in JAX 0.4.38+.  Patch the symbol so the import chain
# (mujoco_playground -> flax) doesn't crash.
if not hasattr(_layout, "Format"):
    _layout.Format = _layout.DeviceLocalLayout

from mujoco_playground import registry  # noqa: E402


# ---------------------------------------------------------------------------
# Environment-specific training defaults (matching official configs)
# ---------------------------------------------------------------------------

_ENV_DEFAULTS: dict[str, dict[str, Any]] = {
    "WalkerWalk": {
        "action_repeat": 1,
        "reward_scaling": 10.0,
        "normalize_obs": True,
    },
}


# ---------------------------------------------------------------------------
# Running observation normalizer (Welford online algorithm)
# ---------------------------------------------------------------------------

class _RunningMeanStd:
    """Track running mean/variance of a data stream for normalization.

    Uses Welford's parallel/batch update for numerical stability.
    """

    def __init__(self, shape: tuple[int, ...], eps: float = 1e-4) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = eps

    def update(self, batch: np.ndarray) -> None:
        batch = np.asarray(batch, dtype=np.float64)
        batch_mean = np.mean(batch, axis=0)
        batch_var = np.var(batch, axis=0)
        batch_count = batch.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        self.mean += delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var = (
            m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        ) / tot_count
        self.count = tot_count


# ---------------------------------------------------------------------------
# Single-instance wrapper (kept for simple testing / debugging)
# ---------------------------------------------------------------------------

class PlaygroundGymWrapper(gym.Env):
    """Wraps a MuJoCo Playground JAX environment as a ``gymnasium.Env``."""

    metadata: dict[str, Any] = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        env_name: str = "WalkerWalk",
        seed: int = 0,
        impl: str = "jax",
    ) -> None:
        super().__init__()

        cfg = registry.get_default_config(env_name)
        cfg.impl = impl
        self._env = registry.load(env_name, config=cfg)
        self._key = jax.random.PRNGKey(seed)
        self._state = None

        obs_size = self._env.observation_size
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        ctrl_range = np.array(self._env.mj_model.actuator_ctrlrange, dtype=np.float32)
        self.action_space = spaces.Box(
            low=ctrl_range[:, 0],
            high=ctrl_range[:, 1],
            dtype=np.float32,
        )

        self._jit_reset = jax.jit(self._env.reset)
        self._jit_step = jax.jit(self._env.step)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self._key = jax.random.PRNGKey(seed)
        self._key, reset_key = jax.random.split(self._key)
        self._state = self._jit_reset(reset_key)
        return np.asarray(self._state.obs, dtype=np.float32), {}

    def step(self, action):
        self._state = self._jit_step(self._state, jnp.asarray(action, dtype=jnp.float32))
        obs = np.asarray(self._state.obs, dtype=np.float32)
        reward = float(self._state.reward)
        done = bool(self._state.done)
        info = {k: np.asarray(v) for k, v in self._state.metrics.items()}
        return obs, reward, False, done, info

    def render(self):
        return None


# ---------------------------------------------------------------------------
# Batched vector wrapper — the fast path
# ---------------------------------------------------------------------------

class BatchedPlaygroundVectorEnv(gym.vector.VectorEnv):
    """Batched MuJoCo Playground environment using ``jax.vmap``.

    Steps *all* sub-environments in a single JAX call on the accelerator
    (GPU/TPU).  Includes auto-reset: environments that reach the episode
    length or encounter NaN physics are transparently reset to cached
    initial states (matching MuJoCo Playground's ``BraxAutoResetWrapper``
    with ``full_reset=False``).

    For known environments (e.g. ``WalkerWalk``), recommended training
    defaults (``action_repeat``, ``reward_scaling``, ``normalize_obs``)
    are applied automatically unless overridden explicitly.

    When ``torch_device`` is provided (e.g. ``"cuda"``), observations are
    returned as PyTorch tensors on that device via DLPack **zero-copy**
    transfer, avoiding the GPU→CPU→GPU round-trip that would otherwise
    happen when the RL algorithm preprocesses numpy observations into
    CUDA tensors.  Rewards, termination, and truncation flags are still
    returned as numpy arrays (they are tiny and consumed by scalar Python
    bookkeeping in the training loop).

    :param env_name: Registered MuJoCo Playground environment name.
    :param num_envs: Number of parallel environments.
    :param seed: Initial PRNG seed.
    :param impl: MJX backend (``"jax"`` or ``"warp"``).
    :param action_repeat: Control steps per agent action.  ``None`` uses
        the environment default (1 for WalkerWalk, 4 otherwise).
    :param reward_scaling: Multiply raw rewards by this factor.  ``None``
        uses the environment default (10.0 for WalkerWalk, 1.0 otherwise).
    :param normalize_obs: If ``True``, apply running mean/std
        normalization to observations (clipped to [-10, 10]).
    :param torch_device: When set, observations are returned as PyTorch
        tensors on this device via DLPack.  Pass ``None`` to return raw
        JAX arrays (the old behaviour).
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        env_name: str = "WalkerWalk",
        num_envs: int = 64,
        seed: int = 0,
        impl: str = "jax",
        *,
        sim_dt: float | None = None,
        action_repeat: int | None = None,
        reward_scaling: float | None = None,
        normalize_obs: bool | None = None,
        torch_device: str | torch.device | None = None,
    ) -> None:
        cfg = registry.get_default_config(env_name)
        cfg.impl = impl
        if sim_dt is not None:
            cfg.sim_dt = sim_dt
        self._episode_length = cfg.episode_length
        self._env = registry.load(env_name, config=cfg)
        self._key = jax.random.PRNGKey(seed)
        self._states = None
        self._first_data = None
        self._first_obs = None

        defaults = _ENV_DEFAULTS.get(env_name, {})
        action_repeat = (
            action_repeat if action_repeat is not None
            else defaults.get("action_repeat", 4)
        )
        self._action_repeat = action_repeat
        self._reward_scaling = (
            reward_scaling if reward_scaling is not None
            else defaults.get("reward_scaling", 1.0)
        )
        self._normalize_obs = (
            normalize_obs if normalize_obs is not None
            else defaults.get("normalize_obs", False)
        )

        self._torch_device = torch.device(torch_device) if torch_device is not None else None
        self._init_step_counts = np.linspace(
            0, cfg.episode_length, num_envs, endpoint=False, dtype=np.int32,
        )
        self._step_counts = self._init_step_counts.copy()

        obs_size = self._env.observation_size
        ctrl_range = np.array(self._env.mj_model.actuator_ctrlrange, dtype=np.float32)

        single_obs = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        single_act = spaces.Box(
            low=ctrl_range[:, 0], high=ctrl_range[:, 1], dtype=np.float32
        )

        self.num_envs = num_envs
        self.single_observation_space = single_obs
        self.single_action_space = single_act
        self.observation_space = batch_space(single_obs, num_envs)
        self.action_space = batch_space(single_act, num_envs)
        self.env_steps_per_step = action_repeat

        if self._normalize_obs:
            self._obs_rms = _RunningMeanStd(shape=(obs_size,))

        self._vmap_reset = jax.jit(jax.vmap(self._env.reset))

        if action_repeat > 1:
            env_step_vmap = jax.vmap(self._env.step)

            def _multi_step(states, actions):
                init_reward = jnp.zeros_like(states.reward)
                init_done = jnp.zeros(states.done.shape, dtype=jnp.bool_)

                def body(carry, _):
                    s, r_acc, any_d = carry
                    s_next = env_step_vmap(s, actions)
                    safe_r = jnp.nan_to_num(s_next.reward, nan=0.0)
                    r_acc = r_acc + jnp.where(any_d, 0.0, safe_r)
                    any_d = any_d | (s_next.done > 0.5)
                    return (s_next, r_acc, any_d), None

                (final_states, total_reward, any_done), _ = jax.lax.scan(
                    body, (states, init_reward, init_done), None,
                    length=action_repeat,
                )
                return final_states, total_reward, any_done

            self._step_fn = jax.jit(_multi_step)
        else:
            env_step_vmap = jax.vmap(self._env.step)

            def _single_step(states, actions):
                s_next = env_step_vmap(states, actions)
                safe_r = jnp.nan_to_num(s_next.reward, nan=0.0)
                return s_next.replace(reward=safe_r), safe_r, s_next.done > 0.5

            self._step_fn = jax.jit(_single_step)

        def _apply_auto_reset(states, first_data, first_obs, done_mask):
            """Replace data/obs of done envs with cached initial states."""
            def where_done(reset_val, current_val):
                mask = done_mask.reshape(-1, *([1] * (reset_val.ndim - 1)))
                return jnp.where(mask, reset_val, current_val)

            new_data = jax.tree.map(where_done, first_data, states.data)
            new_obs = jax.tree.map(where_done, first_obs, states.obs)
            new_done = jnp.where(done_mask, jnp.zeros_like(states.done), states.done)
            return states.replace(data=new_data, obs=new_obs, done=new_done)

        self._auto_reset_fn = jax.jit(_apply_auto_reset)

    # -- observation helpers -------------------------------------------------

    def _maybe_normalize_obs(self, jax_obs: jnp.ndarray) -> jnp.ndarray:
        """Optionally normalize observations using running mean/std.

        Returns the input unchanged when ``normalize_obs`` is disabled.
        """
        if not self._normalize_obs:
            return jax_obs
        self._obs_rms.update(np.asarray(jax_obs, dtype=np.float32))
        mean = jnp.asarray(self._obs_rms.mean, dtype=jnp.float32)
        std = jnp.sqrt(jnp.asarray(self._obs_rms.var, dtype=jnp.float32) + 1e-8)
        return jnp.clip((jax_obs - mean) / std, -10.0, 10.0)

    # -- JAX ↔ PyTorch helpers -----------------------------------------------

    def _obs_to_torch(self, jax_obs: jnp.ndarray) -> torch.Tensor:
        """Convert a JAX observation array to a PyTorch CUDA tensor.

        Uses DLPack for a zero-copy view, then ``.clone()`` so the tensor
        owns its memory independently of JAX's buffer-donation lifecycle.
        The GPU→GPU clone is orders of magnitude cheaper than a GPU→CPU→GPU
        round-trip through numpy.
        """
        return torch.from_dlpack(jax_obs).clone()

    @staticmethod
    def _actions_to_jax(actions: np.ndarray | torch.Tensor) -> jnp.ndarray:
        """Accept numpy *or* PyTorch actions and return a JAX array.

        When the caller supplies a CUDA tensor the conversion goes through
        DLPack (zero-copy).  Numpy arrays are passed straight to JAX which
        handles the host→device copy internally.
        """
        if isinstance(actions, torch.Tensor):
            return jnp.from_dlpack(actions.detach().contiguous())
        return actions

    # -- gymnasium.vector.VectorEnv interface --------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor | jnp.ndarray, dict[str, Any]]:
        if seed is not None:
            self._key = jax.random.PRNGKey(seed)

        self._key, split_key = jax.random.split(self._key)
        reset_keys = jax.random.split(split_key, self.num_envs)
        self._states = self._vmap_reset(reset_keys)

        self._first_data = self._states.data
        self._first_obs = self._states.obs

        self._step_counts = self._init_step_counts.copy()

        normalized_obs = self._maybe_normalize_obs(self._states.obs)
        obs = (
            self._obs_to_torch(normalized_obs)
            if self._torch_device is not None
            else normalized_obs
        )
        return obs, {}

    def step(
        self, actions: np.ndarray | torch.Tensor,
    ) -> tuple[
        torch.Tensor | jnp.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        dict[str, Any],
    ]:
        jax_actions = self._actions_to_jax(actions)

        self._states, total_reward, any_done = self._step_fn(
            self._states, jax_actions,
        )
        reward = np.array(total_reward, dtype=np.float32)
        env_done = np.array(any_done, dtype=np.bool_)

        self._step_counts += self._action_repeat
        time_limit = self._step_counts >= self._episode_length
        truncated = env_done | time_limit
        self._step_counts[truncated] = 0

        if np.any(truncated):
            done_jax = jnp.array(truncated)
            self._states = self._auto_reset_fn(
                self._states, self._first_data, self._first_obs, done_jax,
            )

        if self._reward_scaling != 1.0:
            reward *= self._reward_scaling

        normalized_obs = self._maybe_normalize_obs(self._states.obs)
        obs = (
            self._obs_to_torch(normalized_obs)
            if self._torch_device is not None
            else normalized_obs
        )

        terminated = np.zeros(self.num_envs, dtype=np.bool_)
        return obs, reward, terminated, truncated, {"_episode_done": truncated}

    def render(self):
        return None

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Factory for the old entrypoint-based manifest path (single env)
# ---------------------------------------------------------------------------

def make_env(
    env_name: str = "WalkerWalk",
    seed: int = 0,
    impl: str = "jax",
    **kwargs: Any,
) -> PlaygroundGymWrapper:
    """Factory function used as the manifest entrypoint."""
    return PlaygroundGymWrapper(env_name=env_name, seed=seed, impl=impl, **kwargs)

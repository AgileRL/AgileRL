"""Benchmark raw env stepping throughput for EnvPool PandaRobotiqPushCube (no agent).

Measures the number of environment steps per second achieved by a vectorized
EnvPool environment over a fixed wall-clock window.  No RL agent or training
loop is involved — actions are sampled uniformly at random from the action space.

Usage::

    python hpo_improvements/panda_robotiq_push_cube/benchmark_steps.py
    python hpo_improvements/panda_robotiq_push_cube/benchmark_steps.py --num-envs 512
    python hpo_improvements/panda_robotiq_push_cube/benchmark_steps.py --num-envs 256 --duration 60
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

# ---------------------------------------------------------------------------
# Ensure the repo root is on sys.path so local imports work when the script is
# invoked from any working directory.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

ENV_ID = "PandaRobotiqPushCube-v1"


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------


def _build_envpool_env(
    *,
    env_id: str,
    num_envs: int,
    seed: int,
    flatten_obs: bool,
) -> Any:
    """Create an EnvPool vectorised Gymnasium environment."""
    try:
        import envpool  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        if exc.name != "envpool":
            raise
        msg = (
            "EnvPool is not installed. Install it before running this benchmark, "
            "e.g. `uv add envpool`."
        )
        raise ImportError(msg) from exc

    try:
        if hasattr(envpool, "make_gymnasium"):
            env = envpool.make_gymnasium(env_id, num_envs=num_envs, seed=seed)
        else:
            env = envpool.make(env_id, num_envs=num_envs, seed=seed)
    except AssertionError as exc:
        if "is not supported" in str(exc):
            msg = (
                f"EnvPool task '{env_id}' is not available in this build.\n"
                'Use `uv run python -c "import envpool; print(envpool.list_all_envs())"` '
                "to inspect supported task IDs."
            )
            raise ValueError(msg) from exc
        raise

    if flatten_obs:
        env = gym.wrappers.vector.FlattenObservation(env)

    return env


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for env-step benchmarking."""
    p = argparse.ArgumentParser(
        description=f"Benchmark EnvPool {ENV_ID} steps/s (no agent).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--num-envs",
        type=int,
        default=256,
        help="Number of parallel environments (matches the training YAML default).",
    )
    p.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Benchmark duration in seconds.",
    )
    p.add_argument(
        "--warmup-steps",
        type=int,
        default=20,
        help="Number of warm-up steps executed before the timed window begins.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed passed to EnvPool.",
    )
    p.add_argument(
        "--no-flatten",
        action="store_true",
        help="Disable the FlattenObservation wrapper.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------


def main() -> None:
    """Run a fixed-duration environment stepping benchmark."""
    args = parse_args()
    flatten_obs = not args.no_flatten

    print(f"Building env: {ENV_ID}  x{args.num_envs}  (flatten_obs={flatten_obs})")
    env = _build_envpool_env(
        env_id=ENV_ID,
        num_envs=args.num_envs,
        seed=args.seed,
        flatten_obs=flatten_obs,
    )

    obs_shape = (
        env.single_observation_space.shape
        if hasattr(env, "single_observation_space")
        else env.observation_space.shape
    )
    _act_shape_display = (
        env.single_action_space.shape
        if hasattr(env, "single_action_space")
        else env.action_space.shape
    )
    print(f"  obs_space={obs_shape}  act_space={_act_shape_display}")

    env.reset(seed=args.seed)

    # ------------------------------------------------------------------
    # Resolve the single-env action space so we can batch-sample correctly.
    # EnvPool expects actions shaped (num_envs, *action_shape).
    # ------------------------------------------------------------------
    single_act_space = (
        env.single_action_space
        if hasattr(env, "single_action_space")
        else env.action_space
    )
    act_low = single_act_space.low
    act_high = single_act_space.high
    act_shape = single_act_space.shape  # e.g. (7,)

    rng = np.random.default_rng(args.seed)

    def _sample_actions() -> np.ndarray:
        """Sample a batch of random actions shaped (num_envs, *act_shape)."""
        return rng.uniform(act_low, act_high, size=(args.num_envs, *act_shape)).astype(
            np.float32
        )

    # ------------------------------------------------------------------
    # Warm-up: let any lazy initialisation / JIT compilation finish
    # outside the timed window.
    # ------------------------------------------------------------------
    print(f"Warming up ({args.warmup_steps} steps)...")
    for _ in range(args.warmup_steps):
        env.step(_sample_actions())

    # ------------------------------------------------------------------
    # Timed benchmark loop
    # ------------------------------------------------------------------
    steps_per_call = args.num_envs  # each env.step() advances all envs by 1
    print(
        f"Benchmarking for {args.duration:.0f}s "
        f"({steps_per_call} env-steps per call)..."
    )

    total_steps = 0
    t_start = time.perf_counter()
    deadline = t_start + args.duration

    while True:
        env.step(_sample_actions())
        total_steps += steps_per_call
        if time.perf_counter() >= deadline:
            break

    elapsed = time.perf_counter() - t_start
    steps_per_sec = total_steps / elapsed

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print()
    print("=" * 52)
    print(f"  Env:          {ENV_ID}")
    print(f"  Num envs:     {args.num_envs}")
    print(f"  Duration:     {elapsed:.2f} s")
    print(f"  Total steps:  {total_steps:,}")
    print(f"  Steps/s:      {steps_per_sec:,.0f}")
    print("=" * 52)

    env.close()


if __name__ == "__main__":
    main()

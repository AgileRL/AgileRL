"""Benchmark raw env stepping throughput (no RL loop).

Measures the number of environment steps per second achieved by
``BatchedPlaygroundVectorEnv`` over a fixed wall-clock window.

Usage::

    python hpo_improvements/mujoco_playground_locomotion/benchmark_steps.py
    python hpo_improvements/mujoco_playground_locomotion/benchmark_steps.py --env HumanoidWalk --num-envs 128
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from env_wrapper import BatchedPlaygroundVectorEnv  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for env-step benchmarking."""
    p = argparse.ArgumentParser(description="Benchmark env steps/s")
    p.add_argument("--env", type=str, default="WalkerWalk")
    p.add_argument("--num-envs", type=int, default=1024)
    p.add_argument("--impl", type=str, default="jax", choices=["jax", "warp"])
    p.add_argument(
        "--duration", type=float, default=30.0, help="Benchmark duration in seconds."
    )
    p.add_argument(
        "--warmup-steps",
        type=int,
        default=10,
        help="Number of warm-up steps (excluded from timing).",
    )
    return p.parse_args()


def main() -> None:
    """Run a fixed-duration environment stepping benchmark."""
    args = parse_args()

    print(f"Building env: {args.env} x {args.num_envs} (impl={args.impl})")
    env = BatchedPlaygroundVectorEnv(
        env_name=args.env,
        num_envs=args.num_envs,
        impl=args.impl,
    )
    print(
        f"  obs_space={env.single_observation_space.shape}  "
        f"act_space={env.single_action_space.shape}"
    )

    _obs, _ = env.reset(seed=42)

    # Warm-up: let JIT compilation happen outside the timed window
    print(f"Warming up ({args.warmup_steps} steps)...")
    for _ in range(args.warmup_steps):
        actions = np.array(env.action_space.sample(), dtype=np.float32)
        _obs, *_ = env.step(actions)

    # Timed benchmark
    steps_per_call = env.num_envs
    print(
        f"Benchmarking for {args.duration:.0f}s  (steps_per_call={steps_per_call}) ..."
    )
    total_steps = 0
    t_start = time.perf_counter()
    deadline = t_start + args.duration

    while True:
        actions = np.array(env.action_space.sample(), dtype=np.float32)
        _obs, *_ = env.step(actions)
        total_steps += steps_per_call
        if time.perf_counter() >= deadline:
            break

    elapsed = time.perf_counter() - t_start
    steps_per_sec = total_steps / elapsed

    print()
    print("=" * 50)
    print(f"  Env:          {args.env}")
    print(f"  Num envs:     {args.num_envs}")
    print(f"  Duration:     {elapsed:.2f}s")
    print(f"  Total steps:  {total_steps:,}")
    print(f"  Steps/s:      {steps_per_sec:,.0f}")
    print("=" * 50)

    env.close()


if __name__ == "__main__":
    main()

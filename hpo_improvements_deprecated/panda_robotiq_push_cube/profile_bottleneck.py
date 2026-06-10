"""Profile where time goes in the PandaRobotiqPushCube PPO training loop.

Breaks the loop into four phases and times each independently:
  1. env.step()        — EnvPool CPU physics
  2. obs → GPU        — CPU-to-GPU transfer (numpy → torch)
  3. network forward  — GPU actor inference (get actions)
  4. action → CPU     — GPU-to-CPU transfer (torch → numpy)

No gradient updates are timed here; this isolates the per-step bottleneck
that runs every single environment step during data collection.

Usage::

    uv run python hpo_improvements/panda_robotiq_push_cube/profile_bottleneck.py
    uv run python hpo_improvements/panda_robotiq_push_cube/profile_bottleneck.py --num-envs 256 --steps 2000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import gymnasium as gym  # noqa: E402

ENV_ID = "PandaRobotiqPushCube-v1"


# ---------------------------------------------------------------------------
# Minimal MLP that mirrors the YAML architecture
# encoder: [256, 256] Sigmoid  +  head: [128] Sigmoid  →  7-dim output
# ---------------------------------------------------------------------------


class ActorMLP(nn.Module):
    """Minimal MLP actor mirroring the YAML architecture for profiling."""

    def __init__(self, obs_dim: int = 48, act_dim: int = 7) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.Sigmoid(),
            nn.Linear(256, 256),
            nn.Sigmoid(),
            nn.Linear(256, 128),
            nn.Sigmoid(),
            nn.Linear(128, act_dim),
            nn.Tanh(),  # action in [-1, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass."""
        return self.net(x)


# ---------------------------------------------------------------------------
# EnvPool factory (same as train.py)
# ---------------------------------------------------------------------------


def _build_env(num_envs: int, seed: int = 42) -> gym.Env:
    try:
        import envpool  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        msg = "EnvPool not installed."
        raise ImportError(msg) from exc

    if hasattr(envpool, "make_gymnasium"):
        env = envpool.make_gymnasium(ENV_ID, num_envs=num_envs, seed=seed)
    else:
        env = envpool.make(ENV_ID, num_envs=num_envs, seed=seed)

    return gym.wrappers.vector.FlattenObservation(env)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for per-step bottleneck profiling."""
    p = argparse.ArgumentParser(
        description="Profile per-step bottleneck for PandaRobotiqPushCube PPO.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument(
        "--steps", type=int, default=2000, help="Timed steps (after warm-up)."
    )
    p.add_argument(
        "--warmup", type=int, default=50, help="Warm-up steps (excluded from timing)."
    )
    p.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sync(device: str) -> None:
    """Block until all CUDA kernels finish (no-op on CPU)."""
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _timer() -> float:
    return time.perf_counter()


# ---------------------------------------------------------------------------
# Main profiling loop
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the per-step bottleneck profiler."""
    args = parse_args()
    device = args.device
    print(f"Device: {device}  |  Num envs: {args.num_envs}  |  Steps: {args.steps}")

    # ---- build env & actor ------------------------------------------------
    env = _build_env(args.num_envs)
    actor = ActorMLP().to(device)
    actor.eval()

    obs_np, _ = env.reset()

    # ---- warm-up ----------------------------------------------------------
    print(f"Warming up ({args.warmup} steps)...")
    with torch.no_grad():
        for _ in range(args.warmup):
            obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
            act_t = actor(obs_t)
            act_np = act_t.cpu().numpy()
            obs_np, _, _, _, _ = env.step(act_np)
    _sync(device)

    # ---- timed loop -------------------------------------------------------
    t_env = 0.0  # env.step  (CPU physics)
    t_h2d = 0.0  # obs  CPU → GPU
    t_net = 0.0  # actor forward pass on GPU
    t_d2h = 0.0  # action GPU → CPU
    total_env_steps = 0

    print(f"Profiling {args.steps} steps...")
    with torch.no_grad():
        for _ in range(args.steps):
            # 1. obs  CPU → GPU
            _sync(device)
            t0 = _timer()
            obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
            _sync(device)
            t_h2d += _timer() - t0

            # 2. network forward
            t0 = _timer()
            act_t = actor(obs_t)
            _sync(device)
            t_net += _timer() - t0

            # 3. action GPU → CPU
            t0 = _timer()
            act_np = act_t.cpu().numpy()
            _sync(device)
            t_d2h += _timer() - t0

            # 4. env.step (CPU physics)
            t0 = _timer()
            obs_np, _, _, _, _ = env.step(act_np)
            t_env += _timer() - t0

            total_env_steps += args.num_envs

    env.close()

    # ---- report -----------------------------------------------------------
    t_total = t_env + t_h2d + t_net + t_d2h

    def pct(t: float) -> str:
        return f"{100 * t / t_total:5.1f}%"

    def ms_per_step(t: float) -> str:
        return f"{1000 * t / args.steps:.3f} ms"

    print()
    print("=" * 60)
    print(f"  Device:              {device}")
    print(f"  Num envs:            {args.num_envs}")
    print(f"  Timed steps:         {args.steps:,}")
    print(f"  Total env-steps:     {total_env_steps:,}")
    print()
    print(f"  {'Phase':<28}  {'Total (s)':>9}  {'ms/step':>10}  {'Share':>6}")
    print(f"  {'-' * 28}  {'-' * 9}  {'-' * 10}  {'-' * 6}")
    print(
        f"  {'env.step  (CPU physics)':<28}  {t_env:9.3f}  {ms_per_step(t_env):>10}  {pct(t_env):>6}"
    )
    print(
        f"  {'obs  CPU→GPU  (H2D)':<28}  {t_h2d:9.3f}  {ms_per_step(t_h2d):>10}  {pct(t_h2d):>6}"
    )
    print(
        f"  {'actor forward  (GPU)':<28}  {t_net:9.3f}  {ms_per_step(t_net):>10}  {pct(t_net):>6}"
    )
    print(
        f"  {'action GPU→CPU  (D2H)':<28}  {t_d2h:9.3f}  {ms_per_step(t_d2h):>10}  {pct(t_d2h):>6}"
    )
    print(f"  {'-' * 28}  {'-' * 9}  {'-' * 10}  {'-' * 6}")
    print(
        f"  {'TOTAL (data-collection)':<28}  {t_total:9.3f}  {ms_per_step(t_total):>10}  {'100.0%':>6}"
    )
    print()
    steps_per_sec = total_env_steps / t_total
    print(f"  End-to-end steps/s (with agent): {steps_per_sec:,.0f}")
    print(f"  Env-only steps/s  (no agent):    {total_env_steps / t_env:,.0f}")
    print(f"  Agent overhead factor:            {t_total / t_env:.2f}x")
    print("=" * 60)

    # verdict
    shares = {
        "CPU physics (env.step)": t_env,
        "CPU→GPU transfer (H2D)": t_h2d,
        "GPU forward pass (actor)": t_net,
        "GPU→CPU transfer (D2H)": t_d2h,
    }
    dominant = max(shares, key=shares.__getitem__)
    print(f"\n  ► Dominant bottleneck: {dominant}  ({pct(shares[dominant])})")
    print()


if __name__ == "__main__":
    main()

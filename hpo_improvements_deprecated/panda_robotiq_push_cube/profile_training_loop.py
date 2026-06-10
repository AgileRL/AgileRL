"""Profile the full PPO training cycle for PandaRobotiqPushCube.

Unlike profile_bottleneck.py (which only times inference), this script
replicates one complete PPO "tick" as AgileRL would execute it:

  Phase A  — Data collection  (env.step + get_action, no gradients)
  Phase B  — GAE / advantage  (CPU rollout-buffer post-processing)
  Phase C  — PPO update       (multiple epochs of mini-batch gradient steps)
  Phase D  — Evaluation       (env.step + inference, eval_loop full episodes)

Each phase is timed independently so the dominant bottleneck is obvious.

Usage::

    uv run python hpo_improvements/panda_robotiq_push_cube/profile_training_loop.py
    uv run python hpo_improvements/panda_robotiq_push_cube/profile_training_loop.py --device cpu
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import gymnasium as gym  # noqa: E402

ENV_ID = "PandaRobotiqPushCube-v1"
OBS_DIM = 48
ACT_DIM = 7
MAX_EP_STEPS = 3000  # from YAML


# ---------------------------------------------------------------------------
# Actor + Critic MLPs  (mirrors the YAML architecture)
# ---------------------------------------------------------------------------


class ActorMLP(nn.Module):
    """Stochastic actor MLP mirroring the YAML architecture for profiling."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, 256),
            nn.Sigmoid(),
            nn.Linear(256, 256),
            nn.Sigmoid(),
            nn.Linear(256, 128),
            nn.Sigmoid(),
            nn.Linear(128, ACT_DIM),
        )
        self.log_std = nn.Parameter(torch.zeros(ACT_DIM))

    def forward(self, x: torch.Tensor):
        """Sample action and log-prob via reparameterised Normal."""
        mean = self.net(x)
        std = self.log_std.exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1)
        return action.tanh(), log_prob


class CriticMLP(nn.Module):
    """Critic MLP mirroring the YAML architecture for profiling."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, 256),
            nn.Sigmoid(),
            nn.Linear(256, 256),
            nn.Sigmoid(),
            nn.Linear(256, 128),
            nn.Sigmoid(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return scalar state-value estimate."""
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# EnvPool factory
# ---------------------------------------------------------------------------


def _build_env(num_envs: int, seed: int = 42) -> gym.Env:
    try:
        import envpool  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        msg = "EnvPool not installed."
        raise ImportError(msg) from exc
    env = (
        envpool.make_gymnasium if hasattr(envpool, "make_gymnasium") else envpool.make
    )(ENV_ID, num_envs=num_envs, seed=seed)
    return gym.wrappers.vector.FlattenObservation(env)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for full PPO cycle profiling."""
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Profile full PPO training cycle for PandaRobotiqPushCube.",
    )
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument(
        "--rollout-steps",
        type=int,
        default=300,
        help="Steps collected per rollout (learn_step / num_envs = 76800/256).",
    )
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument(
        "--eval-loops",
        type=int,
        default=8,
        help="Full episodes per evaluation (matches eval_loop in YAML).",
    )
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _t() -> float:
    return time.perf_counter()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the full PPO training cycle profiler."""
    args = parse_args()
    dev = args.device
    n_envs = args.num_envs
    n_steps = args.rollout_steps
    total_collection_steps = n_steps * n_envs  # = learn_step

    print(
        f"Device: {dev}  |  num_envs: {n_envs}  |  rollout_steps: {n_steps}"
        f"  |  learn_step: {total_collection_steps:,}"
    )
    print(
        f"update_epochs: {args.update_epochs}  |  batch_size: {args.batch_size}"
        f"  |  eval_loops: {args.eval_loops}"
    )

    env = _build_env(n_envs)
    actor = ActorMLP().to(dev)
    critic = CriticMLP().to(dev)
    opt_a = optim.Adam(actor.parameters(), lr=1e-4)
    opt_c = optim.Adam(critic.parameters(), lr=1e-4)

    obs_np, _ = env.reset()

    # -----------------------------------------------------------------------
    # Warm-up
    # -----------------------------------------------------------------------
    print(f"\nWarming up ({args.warmup} steps)...")
    actor.eval()
    critic.eval()
    with torch.no_grad():
        for _ in range(args.warmup):
            obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=dev)
            act_t, _ = actor(obs_t)
            obs_np, *_ = env.step(act_t.cpu().numpy())
    _sync(dev)

    # -----------------------------------------------------------------------
    # PHASE A — Data collection
    # -----------------------------------------------------------------------
    print(f"\nPhase A: collecting {n_steps} rollout steps x {n_envs} envs ...")
    actor.eval()
    critic.eval()

    obs_buf = np.zeros((n_steps, n_envs, OBS_DIM), dtype=np.float32)
    act_buf = np.zeros((n_steps, n_envs, ACT_DIM), dtype=np.float32)
    rew_buf = np.zeros((n_steps, n_envs), dtype=np.float32)
    done_buf = np.zeros((n_steps, n_envs), dtype=np.float32)
    val_buf = np.zeros((n_steps, n_envs), dtype=np.float32)
    logp_buf = np.zeros((n_steps, n_envs), dtype=np.float32)

    t_h2d = t_actor = t_critic = t_d2h = t_env = 0.0

    obs_np, _ = env.reset()
    with torch.no_grad():
        for step in range(n_steps):
            obs_buf[step] = obs_np

            _sync(dev)
            t0 = _t()
            obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=dev)
            _sync(dev)
            t_h2d += _t() - t0

            t0 = _t()
            act_t, logp_t = actor(obs_t)
            _sync(dev)
            t_actor += _t() - t0

            t0 = _t()
            val_t = critic(obs_t)
            _sync(dev)
            t_critic += _t() - t0

            t0 = _t()
            act_np = act_t.cpu().numpy()
            logp_np = logp_t.cpu().numpy()
            val_np = val_t.cpu().numpy()
            _sync(dev)
            t_d2h += _t() - t0

            act_buf[step] = act_np
            logp_buf[step] = logp_np
            val_buf[step] = val_np

            t0 = _t()
            obs_np, rew_np, done_np, trunc_np, _ = env.step(act_np)
            t_env += _t() - t0

            rew_buf[step] = rew_np
            done_buf[step] = np.logical_or(done_np, trunc_np).astype(np.float32)

    t_collect = t_h2d + t_actor + t_critic + t_d2h + t_env
    print(
        f"  Done in {t_collect:.2f}s  →  {total_collection_steps / t_collect:,.0f} steps/s"
    )

    # -----------------------------------------------------------------------
    # PHASE B — GAE advantage computation  (CPU)
    # -----------------------------------------------------------------------
    print("Phase B: computing GAE advantages …")
    _sync(dev)
    t0 = _t()

    gamma, lam = 0.994, 0.95
    adv_buf = np.zeros_like(rew_buf)
    last_gae = np.zeros(n_envs, dtype=np.float32)
    with torch.no_grad():
        next_val = (
            critic(torch.as_tensor(obs_np, dtype=torch.float32, device=dev))
            .cpu()
            .numpy()
        )
    _sync(dev)

    for t in reversed(range(n_steps)):
        next_v = next_val if t == n_steps - 1 else val_buf[t + 1]
        next_nd = 1.0 - done_buf[t]
        delta = rew_buf[t] + gamma * next_v * next_nd - val_buf[t]
        last_gae = delta + gamma * lam * next_nd * last_gae
        adv_buf[t] = last_gae
    ret_buf = adv_buf + val_buf

    t_gae = _t() - t0
    print(f"  Done in {t_gae * 1000:.1f}ms")

    # -----------------------------------------------------------------------
    # PHASE C — PPO gradient updates
    # -----------------------------------------------------------------------
    print(
        f"Phase C: PPO update  ({args.update_epochs} epochs, "
        f"batch_size={args.batch_size}) …"
    )

    # Flatten rollout: (T*N, ...)
    obs_flat = torch.as_tensor(obs_buf.reshape(-1, OBS_DIM), dtype=torch.float32)
    act_flat = torch.as_tensor(act_buf.reshape(-1, ACT_DIM), dtype=torch.float32)
    logp_flat = torch.as_tensor(logp_buf.reshape(-1), dtype=torch.float32)
    adv_flat = torch.as_tensor(adv_buf.reshape(-1), dtype=torch.float32)
    ret_flat = torch.as_tensor(ret_buf.reshape(-1), dtype=torch.float32)
    val_flat = torch.as_tensor(val_buf.reshape(-1), dtype=torch.float32)

    # Normalise advantages
    adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

    n_samples = obs_flat.shape[0]
    clip_coef = 0.15
    vf_coef = 0.5
    ent_coef = 0.005

    actor.train()
    critic.train()
    _sync(dev)
    t0 = _t()

    n_minibatches = 0
    for _epoch in range(args.update_epochs):
        idx = torch.randperm(n_samples)
        for start in range(0, n_samples, args.batch_size):
            mb_idx = idx[start : start + args.batch_size]
            mb_obs = obs_flat[mb_idx].to(dev)
            mb_act = act_flat[mb_idx].to(dev)
            mb_logp = logp_flat[mb_idx].to(dev)
            mb_adv = adv_flat[mb_idx].to(dev)
            mb_ret = ret_flat[mb_idx].to(dev)
            mb_val = val_flat[mb_idx].to(dev)

            # Actor
            mean = actor.net(mb_obs)
            std = actor.log_std.exp().expand_as(mean)
            dist = torch.distributions.Normal(mean, std)
            new_logp = dist.log_prob(mb_act).sum(-1)
            entropy = dist.entropy().sum(-1).mean()

            log_ratio = new_logp - mb_logp
            ratio = log_ratio.exp()
            pg_loss1 = -mb_adv * ratio
            pg_loss2 = -mb_adv * ratio.clamp(1 - clip_coef, 1 + clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Critic
            new_val = critic(mb_obs)
            v_clipped = mb_val + (new_val - mb_val).clamp(-clip_coef, clip_coef)
            vf_loss1 = (new_val - mb_ret).pow(2)
            vf_loss2 = (v_clipped - mb_ret).pow(2)
            vf_loss = 0.5 * torch.max(vf_loss1, vf_loss2).mean()

            loss = pg_loss - ent_coef * entropy + vf_coef * vf_loss

            opt_a.zero_grad()
            opt_c.zero_grad()
            loss.backward()
            opt_a.step()
            opt_c.step()
            n_minibatches += 1

    _sync(dev)
    t_update = _t() - t0
    print(
        f"  Done in {t_update:.2f}s  ({n_minibatches} mini-batches, "
        f"{t_update / n_minibatches * 1000:.2f}ms each)"
    )

    # -----------------------------------------------------------------------
    # PHASE D — Evaluation  (eval_loop full episodes, vectorized)
    # -----------------------------------------------------------------------
    print(f"Phase D: evaluation  ({args.eval_loops} full episodes x {n_envs} envs) ...")
    actor.eval()
    critic.eval()
    _sync(dev)
    t0 = _t()
    t_eval_env = t_eval_net = 0.0
    eval_env_steps = 0

    with torch.no_grad():
        for _loop in range(args.eval_loops):
            obs_np, _ = env.reset()
            finished = np.zeros(n_envs, dtype=bool)
            ep_step = 0

            while not np.all(finished) and ep_step < MAX_EP_STEPS:
                _sync(dev)
                ta = _t()
                obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=dev)
                act_t, _ = actor(obs_t)
                act_np = act_t.cpu().numpy()
                _sync(dev)
                t_eval_net += _t() - ta

                ta = _t()
                obs_np, _, done_np, trunc_np, _ = env.step(act_np)
                t_eval_env += _t() - ta

                finished |= np.logical_or(done_np, trunc_np)
                ep_step += 1
                eval_env_steps += n_envs

    _sync(dev)
    t_eval = _t() - t0
    print(
        f"  Done in {t_eval:.2f}s  ({eval_env_steps:,} env-steps, "
        f"{eval_env_steps / t_eval:,.0f} steps/s)"
    )

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    t_loop = t_collect + t_gae + t_update + t_eval

    def pct(t: float) -> str:
        return f"{100 * t / t_loop:5.1f}%"

    def sps(steps: int, t: float) -> str:
        return f"{steps / t:,.0f} stp/s"

    print()
    print("=" * 72)
    print(
        f"  Device: {dev}  |  num_envs: {n_envs}  |  learn_step: {total_collection_steps:,}"
    )
    print()
    print(f"  {'Phase':<40}  {'Time (s)':>8}  {'Share':>6}")
    print(f"  {'-' * 40}  {'-' * 8}  {'-' * 6}")
    print(f"  {'A  collect: env.step (CPU physics)':<40}  {t_env:8.3f}  {pct(t_env)}")
    print(f"  {'A  collect: obs H2D':<40}  {t_h2d:8.3f}  {pct(t_h2d)}")
    print(f"  {'A  collect: actor forward (GPU)':<40}  {t_actor:8.3f}  {pct(t_actor)}")
    print(
        f"  {'A  collect: critic forward (GPU)':<40}  {t_critic:8.3f}  {pct(t_critic)}"
    )
    print(f"  {'A  collect: act+val D2H':<40}  {t_d2h:8.3f}  {pct(t_d2h)}")
    print(f"  {'-' * 40}  {'-' * 8}  {'-' * 6}")
    print(f"  {'A  TOTAL collection':<40}  {t_collect:8.3f}  {pct(t_collect)}")
    print(f"  {'B  GAE advantage (CPU)':<40}  {t_gae:8.3f}  {pct(t_gae)}")
    print(f"  {'C  PPO gradient update (GPU)':<40}  {t_update:8.3f}  {pct(t_update)}")
    print(f"  {'D  Evaluation (env+inference)':<40}  {t_eval:8.3f}  {pct(t_eval)}")
    print(f"  {'-' * 40}  {'-' * 8}  {'-' * 6}")
    print(f"  {'TOTAL per training tick':<40}  {t_loop:8.3f}  {'100.0%':>6}")
    print()
    print(
        f"  Effective steps/s (collection only):  {total_collection_steps / t_collect:,.0f}"
    )
    print(
        f"  Effective steps/s (whole tick):        {total_collection_steps / t_loop:,.0f}"
    )
    print()

    # sub-breakdown of collection
    print("  Collection sub-breakdown:")
    for label, t in [
        ("env.step", t_env),
        ("H2D", t_h2d),
        ("actor fwd", t_actor),
        ("critic fwd", t_critic),
        ("D2H", t_d2h),
    ]:
        print(f"    {label:<14}  {pct(t)}  of collection")

    # dominant phase
    phases = {
        "A: data collection": t_collect,
        "B: GAE computation": t_gae,
        "C: PPO update": t_update,
        "D: evaluation": t_eval,
    }
    dominant = max(phases, key=phases.__getitem__)
    print()
    print(f"  ► Dominant bottleneck: {dominant}  ({pct(phases[dominant])})")
    print("=" * 72)


if __name__ == "__main__":
    main()

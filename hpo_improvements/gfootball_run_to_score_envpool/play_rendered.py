"""Render a trained AgileRL DQN agent in native Google Research Football.

This script loads the same DQN architecture from a training manifest, restores
checkpoint weights, and runs rendered episodes in upstream ``gfootball``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import spaces

from agilerl.models.manifest import TrainingManifest

_RESET_TUPLE_LEN = 2
_STEP_5_LEN = 5
_STEP_4_LEN = 4


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Render a trained AgileRL DQN checkpoint in native GFootball.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parent / "dqn_run_to_score.yaml",
        help="Path to the training manifest used for the checkpoint architecture.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(__file__).resolve().parent
        / "checkpoints"
        / "gfootball_academy_run_to_score_elite_DQN.pt",
        help="Path to an AgileRL DQN checkpoint.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of rendered episodes to run.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device for inference (e.g. cpu, cuda, mps).",
    )
    parser.add_argument(
        "--env-name",
        type=str,
        default="academy_run_to_score",
        help="Native Google Research Football scenario name.",
    )
    return parser.parse_args()


def _flatten_obs(obs: Any) -> np.ndarray:
    """Convert nested GFootball observations to a flat float32 vector."""
    if isinstance(obs, dict):
        flat_parts = [_flatten_obs(obs[k]) for k in sorted(obs.keys())]
        return (
            np.concatenate(flat_parts, axis=0)
            if flat_parts
            else np.empty((0,), dtype=np.float32)
        )
    if isinstance(obs, (tuple, list)):
        flat_parts = [_flatten_obs(x) for x in obs]
        return (
            np.concatenate(flat_parts, axis=0)
            if flat_parts
            else np.empty((0,), dtype=np.float32)
        )
    arr = np.asarray(obs, dtype=np.float32)
    return arr.reshape(-1)


def _extract_obs(reset_output: Any) -> Any:
    """Handle Gym and Gymnasium reset signatures."""
    if isinstance(reset_output, tuple) and len(reset_output) == _RESET_TUPLE_LEN:
        return reset_output[0]
    return reset_output


def main() -> None:
    """Load a DQN checkpoint and render native GFootball rollouts."""
    try:
        import gfootball.env as football_env  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        msg = (
            "gfootball is not installed. Install native Google Research Football "
            "to render rollouts, e.g. `uv add gfootball`."
        )
        raise ImportError(msg) from exc

    args = parse_args()
    if args.episodes < 1:
        msg = "--episodes must be >= 1"
        raise ValueError(msg)
    if not args.checkpoint.exists():
        msg = f"Checkpoint not found: {args.checkpoint}"
        raise FileNotFoundError(msg)

    env = football_env.create_environment(
        env_name=args.env_name,
        representation="raw",
        stacked=False,
        render=True,
    )

    try:
        sample_obs = _extract_obs(env.reset())
        flat_sample_obs = _flatten_obs(sample_obs)
        if flat_sample_obs.size == 0:
            msg = "Could not infer observation size from reset output."
            raise ValueError(msg)

        obs_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(int(flat_sample_obs.size),),
            dtype=np.float32,
        )
        action_space = spaces.Discrete(int(env.action_space.n))

        validated = TrainingManifest.get_validated(args.manifest, mode="python")
        if str(validated.algorithm.name).upper() != "DQN":
            msg = f"Manifest algorithm must be DQN for this script, got {validated.algorithm.name}."
            raise ValueError(msg)

        agent = validated.algorithm.build_algorithm(
            observation_space=obs_space,
            action_space=action_space,
            index=0,
            device=args.device,
        )
        agent.load_checkpoint(str(args.checkpoint))
        agent.set_training_mode(False)

        for episode_idx in range(args.episodes):
            obs = _extract_obs(env.reset())
            done = False
            trunc = False
            total_reward = 0.0
            step_count = 0

            while not (done or trunc):
                flat_obs = _flatten_obs(obs)
                batched_obs = flat_obs[np.newaxis, :]
                action = agent.get_action(batched_obs, epsilon=0.0)
                action_int = int(np.asarray(action).reshape(-1)[0])

                step_out = env.step(action_int)
                if len(step_out) == _STEP_5_LEN:
                    obs, reward, done, trunc, _info = step_out
                elif len(step_out) == _STEP_4_LEN:
                    obs, reward, done, _info = step_out
                    trunc = False
                else:
                    msg = f"Unexpected step() output length: {len(step_out)}"
                    raise RuntimeError(msg)

                total_reward += float(reward)
                step_count += 1

            print(
                f"Episode {episode_idx + 1}/{args.episodes} "
                f"| reward={total_reward:.3f} | steps={step_count}"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()

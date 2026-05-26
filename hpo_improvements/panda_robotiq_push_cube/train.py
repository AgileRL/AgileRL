"""Train PPO on EnvPool MuJoCo Playground PandaRobotiqPushCube.

Example usage::

    uv run python hpo_improvements/panda_robotiq_push_cube/train.py
    uv run python hpo_improvements/panda_robotiq_push_cube/train.py --device cpu --wb
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import gymnasium as gym
import torch
from accelerate import Accelerator

from agilerl.models.env import GymEnvSpec
from agilerl.models.manifest import TrainingManifest
from agilerl.training.trainer import LocalTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


class EnvPoolLocalTrainer(LocalTrainer):
    """Local trainer that uses a pre-built EnvPool vector environment."""

    _prebuilt_env: Any = None

    def _make_env(self) -> Any:
        if self._prebuilt_env is not None:
            return self._prebuilt_env
        return super()._make_env()

    @classmethod
    def from_manifest_with_env(
        cls,
        manifest: str | Path | TrainingManifest,
        vector_env: Any,
        *,
        resume_from_checkpoint: str | None = None,
        device: str | torch.device = "cpu",
        accelerator: Accelerator | None = None,
    ) -> EnvPoolLocalTrainer:
        """Build a trainer from a manifest and injected EnvPool vector env."""
        validated = (
            TrainingManifest.get_validated(manifest, mode="python")
            if not isinstance(manifest, TrainingManifest)
            else manifest
        )
        env_data = {
            k: v for k, v in dict(validated.environment).items() if v is not None
        }
        env_spec = GymEnvSpec(**env_data)
        cls._prebuilt_env = vector_env
        try:
            return cls(
                algorithm=validated.algorithm,
                environment=env_spec,
                training=validated.training,
                mutation=validated.mutation,
                tournament=validated.tournament_selection,
                replay_buffer=validated.replay_buffer,
                resume_from_checkpoint=resume_from_checkpoint,
                device=device,
                accelerator=accelerator,
            )
        finally:
            cls._prebuilt_env = None


def _build_envpool_env(
    *,
    env_id: str,
    num_envs: int,
    seed: int,
    env_kwargs: dict[str, Any],
) -> Any:
    """Create an EnvPool vectorized Gymnasium environment."""
    try:
        import envpool  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        if exc.name != "envpool":
            raise
        msg = (
            "EnvPool is not installed. Install it before running this benchmark, e.g. "
            "`uv add envpool`."
        )
        raise ImportError(msg) from exc

    kwargs = dict(env_kwargs)
    flatten_obs = bool(kwargs.pop("flatten_obs", True))
    if "env_type" in kwargs:
        msg = (
            "`env_kwargs.env_type` is reserved by EnvPool's factory API and conflicts "
            "with `make_gymnasium`. Remove it from the manifest."
        )
        raise ValueError(msg)
    kwargs.update({"num_envs": num_envs, "seed": seed})

    try:
        if hasattr(envpool, "make_gymnasium"):
            env = envpool.make_gymnasium(env_id, **kwargs)
        else:
            env = envpool.make(env_id, **kwargs)

        if flatten_obs:
            env = gym.wrappers.vector.FlattenObservation(env)
    except AssertionError as exc:
        if "is not supported" in str(exc):
            msg = (
                f"EnvPool task '{env_id}' is not available in this build.\n"
                'Use `uv run python -c "import envpool; print(envpool.list_all_envs())"` '
                "to inspect supported task IDs."
            )
            raise ValueError(msg) from exc
        raise
    else:
        return env


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for local training."""
    parser = argparse.ArgumentParser(
        description="Train PPO on EnvPool MuJoCo Playground PandaRobotiqPushCube.",
    )
    parser.add_argument(
        "manifest",
        type=Path,
        nargs="?",
        default=Path(__file__).resolve().parent / "ppo_panda_robotiq_push_cube.yaml",
        help="Path to a YAML/JSON training manifest.",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device (default: cuda if available, else cpu).",
    )
    parser.add_argument(
        "--use-accelerator",
        action="store_true",
        help="Use Accelerator for training.",
    )
    parser.add_argument(
        "--wb",
        action="store_true",
        help="Enable Weights & Biases logging.",
    )
    parser.add_argument(
        "--wandb-api-key",
        type=str,
        default=None,
        help="Weights & Biases API key.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="AgileRL",
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Custom Weights & Biases run name.",
    )
    parser.add_argument(
        "--env-seed",
        type=int,
        default=0,
        help="Seed passed to the EnvPool environment.",
    )
    parser.add_argument(
        "--elite-path",
        type=Path,
        default=Path(__file__).resolve().parent
        / "checkpoints"
        / "panda_robotiq_push_cube_elite_PPO.pt",
        help=(
            "Path where the best agent checkpoint is saved at the end of training "
            "(default: hpo_improvements/panda_robotiq_push_cube/checkpoints/"
            "panda_robotiq_push_cube_elite_PPO.pt)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run local PPO training on EnvPool PandaRobotiqPushCube."""
    args = parse_args()
    logger.info("Loading manifest: %s", args.manifest)
    validated = TrainingManifest.get_validated(args.manifest, mode="python")

    env_cfg = dict(validated.environment)
    cfg = dict(env_cfg.get("config") or {})
    env_id = str(cfg.pop("env_id", env_cfg["name"]))
    env_kwargs = dict(cfg.pop("env_kwargs", {}))

    if not env_id:
        msg = "Environment id cannot be empty. Set environment.name or environment.config.env_id."
        raise ValueError(msg)

    batched_env = _build_envpool_env(
        env_id=env_id,
        num_envs=int(env_cfg["num_envs"]),
        seed=args.env_seed,
        env_kwargs=env_kwargs,
    )

    accelerator = Accelerator() if args.use_accelerator else None
    trainer = EnvPoolLocalTrainer.from_manifest_with_env(
        validated,
        batched_env,
        device=args.device,
        accelerator=accelerator,
    )

    logger.info(
        "Algorithm: %s | Env: %s | Pop size: %d | Steps: %d | Device: %s",
        trainer.algorithm_spec.name,
        env_id,
        trainer.training_spec.pop_size,
        trainer.training_spec.max_steps,
        args.device,
    )

    wandb_kwargs: dict[str, Any] = {"project": args.wandb_project}
    if args.wandb_run_name:
        wandb_kwargs["addl_args"] = {"name": args.wandb_run_name}

    args.elite_path.parent.mkdir(parents=True, exist_ok=True)
    _population, last_fitnesses = trainer.train(
        wb=args.wb,
        wandb_api_key=args.wandb_api_key,
        wandb_kwargs=wandb_kwargs if args.wb else None,
        save_elite=True,
        elite_path=str(args.elite_path),
        verbose=True,
    )
    logger.info("Saved best agent checkpoint to: %s", args.elite_path)
    logger.info("Training complete. Best fitness: %.4f", max(last_fitnesses))


if __name__ == "__main__":
    main()

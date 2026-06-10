"""Train on MuJoCo Playground locomotion envs from a manifest.

Example usage::

    python -m hpo_improvements.mujoco_playground_locomotion.train ppo_walker_walk.yaml
    python -m hpo_improvements.mujoco_playground_locomotion.train ppo_walker_walk.yaml --device cpu
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from hpo_improvements.mujoco_playground_locomotion.env_wrapper import (
    BatchedPlaygroundVectorEnv,
)

from agilerl.models.env import GymEnvSpec
from agilerl.models.manifest import TrainingManifest
from agilerl.training.trainer import LocalTrainer

logging.basicConfig(
    level=getattr(
        logging, os.getenv("AGILERL_LOG_LEVEL", "INFO").upper(), logging.INFO
    ),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


class BatchedLocalTrainer(LocalTrainer):
    """LocalTrainer that uses a pre-built batched MuJoCo Playground vector env."""

    _prebuilt_env: Any = None

    def _make_env(self) -> Any:
        if self._prebuilt_env is not None:
            return self._prebuilt_env
        return super()._make_env()

    @classmethod
    def from_manifest_with_env(
        cls,
        manifest: str | Path | TrainingManifest,
        batched_env: Any,
        *,
        resume_from_checkpoint: str | None = None,
        device: str | torch.device = "cpu",
        accelerator: Accelerator | None = None,
    ) -> BatchedLocalTrainer:
        """Build a trainer from a manifest and injected batched environment."""
        validated = (
            TrainingManifest.get_validated(manifest, mode="python")
            if not isinstance(manifest, TrainingManifest)
            else manifest
        )
        env_data = {
            k: v for k, v in dict(validated.environment).items() if v is not None
        }
        env_spec = GymEnvSpec(**env_data)
        cls._prebuilt_env = batched_env
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


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for local training."""
    parser = argparse.ArgumentParser(
        description="Run local evolutionary RL training from a manifest.",
    )
    parser.add_argument(
        "manifest",
        type=Path,
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
        "--checkpoint-steps",
        type=int,
        default=None,
        help="Save a checkpoint every N episodes.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Directory for checkpoint files.",
    )
    parser.add_argument(
        "--overwrite-checkpoints",
        action="store_true",
        help="Overwrite previous checkpoints during training.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Resume training from checkpoint.",
    )
    parser.add_argument(
        "--save-elite",
        action="store_true",
        help="Persist the elite agent after training.",
    )
    parser.add_argument(
        "--elite-path",
        type=str,
        default=None,
        help="Path for the saved elite agent.",
    )
    parser.add_argument(
        "--tensorboard",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable TensorBoard logging.",
    )
    parser.add_argument(
        "--tensorboard-log-dir",
        type=str,
        default="tensorboard_logs",
        help="Directory for TensorBoard logs.",
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print training metrics output.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Custom Weights & Biases project name.",
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
        help="Seed passed to the Playground vector env wrapper.",
    )
    return parser.parse_args()


def _build_batched_env(
    *,
    env_name: str,
    num_envs: int,
    impl: str,
    staggered_resets: bool,
    seed: int,
    torch_device: str,
) -> Any:
    return BatchedPlaygroundVectorEnv(
        env_name=env_name,
        num_envs=num_envs,
        impl=impl,
        staggered_resets=staggered_resets,
        seed=seed,
        torch_device=torch_device,
    )


def main() -> None:
    """Run local evolutionary RL training from a manifest."""
    args = parse_args()

    logger.info("Loading manifest: %s", args.manifest)

    validated = TrainingManifest.get_validated(args.manifest, mode="python")
    env_cfg = dict(validated.environment)

    batched_env = _build_batched_env(
        env_name=env_cfg["name"],
        num_envs=env_cfg["num_envs"],
        impl=env_cfg.get("impl", "jax"),
        staggered_resets=env_cfg.get("staggered_resets", False),
        seed=args.env_seed,
        torch_device=args.device,
    )

    accelerator = Accelerator() if args.use_accelerator else None

    trainer = BatchedLocalTrainer.from_manifest_with_env(
        validated,
        batched_env,
        resume_from_checkpoint=args.resume_from_checkpoint,
        device=args.device,
        accelerator=accelerator,
    )

    logger.info(
        "Algorithm: %s | Env: %s | Pop size: %d | Steps: %d | Device: %s",
        trainer.algorithm_spec.name,
        trainer.env_spec.name,
        trainer.training_spec.pop_size,
        trainer.training_spec.max_steps,
        args.device,
    )

    wandb_kwargs: dict[str, Any] = {}
    if args.wandb_project:
        wandb_kwargs["project"] = args.wandb_project
    if args.wandb_run_name:
        wandb_kwargs.setdefault("addl_args", {})["name"] = args.wandb_run_name

    _population, last_fitnesses = trainer.train(
        wb=args.wb,
        wandb_api_key=args.wandb_api_key,
        wandb_kwargs=wandb_kwargs or None,
        tensorboard=args.tensorboard,
        tensorboard_log_dir=args.tensorboard_log_dir,
        checkpoint_steps=args.checkpoint_steps,
        checkpoint_path=args.checkpoint_path,
        overwrite_checkpoints=args.overwrite_checkpoints,
        save_elite=args.save_elite,
        elite_path=args.elite_path,
        verbose=args.verbose,
    )

    logger.info("Training complete. Best fitness: %.4f", max(last_fitnesses))


if __name__ == "__main__":
    main()

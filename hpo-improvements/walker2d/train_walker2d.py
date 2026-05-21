"""Train PPO on Gymnasium MuJoCo Walker2d with AgileRL and Weights & Biases.

Uses ``LocalTrainer`` and a YAML training manifest. The default manifest
(``ppo_walker2d.yaml``) sets hyperparameters suited to continuous MuJoCo
locomotion: Tanh MLP policy, vectorized envs, and standard PPO rollout sizes.

Prerequisites::

    uv pip install 'gymnasium[mujoco]' wandb

Usage::

    uv run python hpo-improvements/walker2d/train_walker2d.py
    uv run python hpo-improvements/walker2d/train_walker2d.py --no-wb
    uv run python hpo-improvements/walker2d/train_walker2d.py --device cuda --num-envs 32
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Union

import torch

from agilerl.models.manifest import TrainingManifest
from agilerl.training.trainer import LocalTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = SCRIPT_DIR / "ppo_walker2d.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PPO on Gymnasium Walker2d (AgileRL + W&B)",
    )
    parser.add_argument(
        "-m",
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Path to training manifest YAML.",
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="Override environment id (default: Walker2d-v5 from manifest).",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help="Override parallel environment count.",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device (default: cuda if available).",
    )
    parser.add_argument(
        "--wb",
        action="store_true",
        help="Enable Weights & Biases logging.",
    )
    parser.add_argument("--no-wb", dest="wb", action="store_false")
    parser.set_defaults(wb=True)
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="agilerl-walker2d",
        help="W&B project name.",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="W&B run name (default: ppo-walker2d-<num_envs>env).",
    )
    parser.add_argument(
        "--wandb-api-key",
        type=str,
        default=None,
        help="W&B API key (optional if already logged in).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override training max_steps from manifest.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Resume training from a checkpoint path.",
    )
    return parser.parse_args()


def _apply_cli_overrides(
    manifest_path: Path, args: argparse.Namespace,
) -> Union[Path, dict[str, Any]]:
    """Return manifest path or an overridden dict for ``LocalTrainer.from_manifest``."""
    if args.env is None and args.num_envs is None and args.max_steps is None:
        return manifest_path

    data = TrainingManifest._load_yaml(manifest_path)
    env_section = data.setdefault("environment", {})
    if args.env is not None:
        env_section["name"] = args.env
    if args.num_envs is not None:
        env_section["num_envs"] = args.num_envs
    if args.max_steps is not None:
        data.setdefault("training", {})["max_steps"] = args.max_steps

    return data


def main() -> None:
    args = parse_args()
    manifest = _apply_cli_overrides(args.manifest, args)

    trainer = LocalTrainer.from_manifest(
        manifest=manifest,
        resume_from_checkpoint=args.resume_from_checkpoint,
        device=args.device,
    )

    env_name = trainer.env_spec.name
    num_envs = trainer.env_spec.num_envs
    logger.info(
        "Algorithm: %s | Env: %s | Pop size: %d | Max steps: %d | Device: %s",
        trainer.algorithm_spec.name,
        env_name,
        trainer.training_spec.pop_size,
        trainer.training_spec.max_steps,
        args.device,
    )

    wandb_kwargs: dict[str, Any] = {}
    if args.wandb_project:
        wandb_kwargs["project"] = args.wandb_project
    run_name = args.wandb_run_name or f"ppo-{env_name.lower()}-{num_envs}env"
    wandb_kwargs.setdefault("addl_args", {})["name"] = run_name
    wandb_kwargs.setdefault("addl_args", {})["config"] = {
        "env": env_name,
        "num_envs": num_envs,
        "manifest": str(args.manifest),
    }

    _population, last_fitnesses = trainer.train(
        wb=args.wb,
        wandb_api_key=args.wandb_api_key,
        wandb_kwargs=wandb_kwargs or None,
        verbose=True,
    )

    logger.info("Training complete. Best fitness: %.4f", max(last_fitnesses))


if __name__ == "__main__":
    main()

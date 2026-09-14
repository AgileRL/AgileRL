# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Train locally using AgileRL evolutionary HPO from a manifest specifying the training configuration.

Example usage::

    agilerl train configs/training/ppo/ppo.yaml
    agilerl train configs/training/dqn/dqn.yaml --device cuda
    agilerl train configs/training/ddpg/ddpg.yaml --wb --checkpoint-steps 50
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import torch

from agilerl.training.trainer import LocalTrainer
from agilerl.utils.trainer_utils import started_by_accelerate_launch

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


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
        help="Deprecated; accelerate launch is detected automatically.",
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
    return parser.parse_args()


def _warn_unused_use_accelerator_flag() -> None:
    """Warn that ``--use-accelerator`` does not construct Accelerator."""
    warnings.warn(
        "--use-accelerator is unused; accelerate launch is detected automatically.",
        DeprecationWarning,
        stacklevel=3,
    )
    if not started_by_accelerate_launch():
        warnings.warn(
            "No accelerate launch detected; not using Accelerator. "
            "Launch with: accelerate launch --config_file "
            "configs/accelerate/accelerate.yaml -m agilerl.train "
            "<manifest>",
            UserWarning,
            stacklevel=3,
        )


def main() -> None:
    """Run local evolutionary RL training from a manifest."""
    args = parse_args()

    if args.use_accelerator:
        _warn_unused_use_accelerator_flag()

    logger.info("Loading manifest: %s", args.manifest)

    # Load the Trainer from the manifest
    trainer = LocalTrainer.from_manifest(
        manifest=args.manifest,
        resume_from_checkpoint=args.resume_from_checkpoint,
        device=args.device,
    )

    logger.info(
        "Algorithm: %s | Env: %s | Pop size: %d | Steps: %d | Device: %s",
        trainer.algorithm_spec.name,
        trainer.env_spec.name,
        trainer.training_spec.pop_size,
        trainer.training_spec.max_steps,
        args.device,
    )

    # Train the population of agents
    _population, last_fitnesses = trainer.train(
        wb=args.wb,
        wandb_api_key=args.wandb_api_key,
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


if __name__ == "__main__":  # pragma: no cover
    main()  # pragma: no cover

"""Train locally using AgileRL evolutionary HPO from a manifest specifying the training configuration.

Example usage::

    python train.py configs/training/ppo/ppo.yaml
    python train.py configs/training/dqn/dqn.yaml --device cuda
    python train.py configs/training/ddpg/ddpg.yaml --wb --checkpoint 50
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from accelerate import Accelerator

from agilerl.models.algo import LLMAlgorithmSpec
from agilerl.training.trainer import LocalTrainer, Trainer

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
        "-m",
        "--manifest",
        type=Path,
        required=True,
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
    return parser.parse_args()


def main() -> None:
    """Run local evolutionary RL training from a manifest."""
    args = parse_args()

    logger.info("Loading manifest: %s", args.manifest)

    validated_manifest = Trainer.get_validated_manifest(args.manifest)
    is_llm = isinstance(validated_manifest.algorithm, LLMAlgorithmSpec)

    accelerator = Accelerator() if args.use_accelerator or is_llm else None

    # Load the Trainer from the manifest
    trainer = LocalTrainer.from_manifest(
        manifest=args.manifest,
        resume_from_checkpoint=args.resume_from_checkpoint,
        device=args.device,
        accelerator=accelerator,
    )

    logger.info(
        "Algorithm: %s | Env: %s | Pop size: %d | Steps: %d | Device: %s",
        trainer.algorithm_spec.name,
        trainer.env_spec.name,
        trainer.training_spec.population_size,
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


if __name__ == "__main__":
    main()

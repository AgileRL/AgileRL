"""Single-agent benchmarking script.

Loads a training manifest and runs evolutionary RL training via
``LocalTrainer.from_manifest()``.

Example usage::

    python benchmarks/single_agent.py configs/training/ppo/ppo.yaml
    python benchmarks/single_agent.py configs/training/dqn/dqn.yaml --device cuda
    python benchmarks/single_agent.py configs/training/ddpg/ddpg.yaml --wb --checkpoint 50
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from agilerl.training.trainer import LocalTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for single-agent training."""
    parser = argparse.ArgumentParser(
        description="Run single-agent evolutionary RL training from a manifest.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to a YAML/JSON training manifest.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device (default: cuda if available, else cpu).",
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
        "--checkpoint",
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
        action="store_true",
        help="Enable TensorBoard logging.",
    )
    parser.add_argument(
        "--tensorboard-log-dir",
        type=str,
        default="tensorboard_logs",
        help="Directory for TensorBoard logs.",
    )
    return parser.parse_args()


def main() -> None:
    """Run single-agent evolutionary RL training from a manifest."""
    args = parse_args()

    logger.info("Loading manifest: %s", args.manifest)

    # Load the Trainer from the manifest
    trainer = LocalTrainer.from_manifest(args.manifest, device=args.device)

    logger.info(
        "Algorithm: %s | Env: %s | Pop: %d | Steps: %d",
        trainer.algorithm.name,
        getattr(trainer.environment, "name", trainer.environment),
        trainer.training.population_size,
        trainer.training.max_steps,
    )

    # Train the population of agents
    _population, fitness_history = trainer.train(
        wb=args.wb,
        wandb_api_key=args.wandb_api_key,
        tensorboard=args.tensorboard,
        tensorboard_log_dir=args.tensorboard_log_dir,
        checkpoint=args.checkpoint,
        checkpoint_path=args.checkpoint_path,
        save_elite=args.save_elite,
        elite_path=args.elite_path,
    )

    best_fitness = max(f for gen in fitness_history for f in gen)
    logger.info("Training complete. Best fitness: %.4f", best_fitness)


if __name__ == "__main__":
    main()

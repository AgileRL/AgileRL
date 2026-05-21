"""Train PPO on a MuJoCo Playground locomotion env with AgileRL evolutionary HPO.

Uses ``BatchedPlaygroundVectorEnv`` to step all environments in a single
``jax.vmap`` call, then feeds NumPy observations into AgileRL's PyTorch
training loop.  This bypasses ``GymEnvSpec.make_env()`` (which would
create N independent ``SyncVectorEnv`` instances) by injecting the
pre-built batched env into the trainer.

Usage::

    uv run python hpo-improvements/mujoco_playground_locomotion/run.py
    uv run python hpo-improvements/mujoco_playground_locomotion/run.py --no-wb
    uv run python hpo-improvements/mujoco_playground_locomotion/run.py --env HumanoidWalk --num-envs 128
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import torch

from agilerl.models.env import GymEnvSpec
from agilerl.models.manifest import TrainingManifest
from agilerl.training.trainer import LocalTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = SCRIPT_DIR / "ppo_walker_walk.yaml"


# ---------------------------------------------------------------------------
# A thin LocalTrainer subclass that accepts a pre-built environment
# ---------------------------------------------------------------------------

class _BatchedLocalTrainer(LocalTrainer):
    """LocalTrainer that uses a pre-built batched VectorEnv."""

    _prebuilt_env = None

    def _make_env(self):
        if self._prebuilt_env is not None:
            return self._prebuilt_env
        return super()._make_env()

    @classmethod
    def from_manifest_with_env(
        cls,
        manifest_path: str | Path,
        batched_env: Any,
        *,
        device: str = "cpu",
    ) -> _BatchedLocalTrainer:
        validated = TrainingManifest.get_validated(manifest_path, mode="python")
        env_data = {k: v for k, v in dict(validated.environment).items() if v is not None}
        env_spec = GymEnvSpec(**env_data)

        cls._prebuilt_env = batched_env
        try:
            trainer = cls(
                algorithm=validated.algorithm,
                environment=env_spec,
                training=validated.training,
                mutation=validated.mutation,
                tournament=validated.tournament_selection,
                replay_buffer=validated.replay_buffer,
                device=device,
            )
        finally:
            cls._prebuilt_env = None

        return trainer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PPO + MuJoCo Playground (batched)")
    parser.add_argument(
        "-m", "--manifest", type=Path, default=DEFAULT_MANIFEST,
        help="Path to training manifest YAML.",
    )
    parser.add_argument(
        "--env", type=str, default=None,
        help="Override environment name (e.g. HumanoidWalk, WalkerWalk).",
    )
    parser.add_argument(
        "--num-envs", type=int, default=None,
        help="Override number of parallel environments.",
    )
    parser.add_argument(
        "--impl", type=str, default="jax", choices=["jax", "warp"],
        help="MJX backend implementation.",
    )
    parser.add_argument(
        "-d", "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--wb", action="store_true", help="Enable W&B logging.")
    parser.add_argument("--no-wb", dest="wb", action="store_false")
    parser.set_defaults(wb=True)
    parser.add_argument("--wandb-project", type=str, default="agilerl-mujoco-playground")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Lazy import so the flax compat patch runs first
    sys.path.insert(0, str(SCRIPT_DIR))
    from env_wrapper import BatchedPlaygroundVectorEnv

    # Read manifest to pick up defaults, then apply CLI overrides
    manifest_data = TrainingManifest._load_yaml(args.manifest)
    env_name = args.env or manifest_data.get("environment", {}).get("name", "WalkerWalk")
    num_envs = args.num_envs or manifest_data.get("environment", {}).get("num_envs", 64)

    logger.info("Building batched env: %s x %d (impl=%s, device=%s)", env_name, num_envs, args.impl, args.device)
    batched_env = BatchedPlaygroundVectorEnv(
        env_name=env_name,
        num_envs=num_envs,
        impl=args.impl,
        torch_device=args.device,
    )
    logger.info(
        "Env ready: obs=%s  act=%s",
        batched_env.single_observation_space.shape,
        batched_env.single_action_space.shape,
    )

    trainer = _BatchedLocalTrainer.from_manifest_with_env(
        args.manifest,
        batched_env,
        device=args.device,
    )

    logger.info(
        "Algorithm: %s | Env: %s | Pop size: %d | Steps: %d | Device: %s",
        trainer.algorithm_spec.name,
        env_name,
        trainer.training_spec.pop_size,
        trainer.training_spec.max_steps,
        args.device,
    )

    wandb_kwargs: dict[str, Any] = {}
    if args.wandb_project:
        wandb_kwargs["project"] = args.wandb_project
    run_name = args.wandb_run_name or f"ppo-{env_name.lower()}-batched"
    wandb_kwargs.setdefault("addl_args", {})["name"] = run_name

    _population, last_fitnesses = trainer.train(
        wb=args.wb,
        wandb_kwargs=wandb_kwargs or None,
        verbose=True,
    )

    logger.info("Training complete.  Best fitness: %.4f", max(last_fitnesses))


if __name__ == "__main__":
    main()

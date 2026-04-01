r"""LLM fine-tuning benchmarking script.

Loads a training manifest for an LLM algorithm (GRPO, DPO) and runs
evolutionary fine-tuning via ``LocalTrainer.from_manifest()``.

Because LLM manifests omit the ``environment`` section, the caller must
supply the dataset, tokenizer, and (for reasoning) a reward function
and conversation template via CLI flags.  These are merged into the
manifest dict before constructing the trainer.

Example usage::

    python benchmarks/llm.py configs/training/grpo.yaml \
        --model Qwen/Qwen2.5-1.5B \
        --dataset gsm8k --dataset-config main \
        --reward-fn my_rewards:gsm8k_reward \
        --conversation-template templates/gsm8k.yaml \
        --max-model-len 1024

    python benchmarks/llm.py configs/training/dpo.yaml \
        --model Qwen/Qwen2.5-1.5B \
        --dataset argilla/dpo-mix-7k \
        --max-model-len 512
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import torch
import yaml

from agilerl.training.trainer import LocalTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for LLM fine-tuning."""
    parser = argparse.ArgumentParser(
        description="Run LLM evolutionary fine-tuning from a manifest.",
    )
    parser.add_argument(
        "manifest",
        type=Path,
        help="Path to a YAML/JSON training manifest.",
    )

    model_group = parser.add_argument_group("model")
    model_group.add_argument(
        "--model",
        type=str,
        required=True,
        dest="pretrained_model_name_or_path",
        help="HuggingFace model identifier (e.g. Qwen/Qwen2.5-1.5B).",
    )
    model_group.add_argument(
        "--max-model-len",
        type=int,
        default=1024,
        help="Maximum model context length.",
    )

    lora_group = parser.add_argument_group("lora")
    lora_group.add_argument("--lora-r", type=int, default=16, help="LoRA rank.")
    lora_group.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha.")
    lora_group.add_argument(
        "--lora-dropout", type=float, default=0.05, help="LoRA dropout."
    )

    env_group = parser.add_argument_group("environment")
    env_group.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="HuggingFace dataset identifier.",
    )
    env_group.add_argument(
        "--dataset-config",
        type=str,
        default=None,
        help="Dataset config name (e.g. 'main').",
    )
    env_group.add_argument(
        "--reward-fn",
        type=str,
        default=None,
        help="Entrypoint for the reward function (module:callable). Required for reasoning.",
    )
    env_group.add_argument(
        "--conversation-template",
        type=Path,
        default=None,
        help="Path to a YAML file containing the conversation template list. Required for reasoning.",
    )
    env_group.add_argument(
        "--train-split",
        type=str,
        default="train",
        help="Dataset split for training.",
    )
    env_group.add_argument(
        "--test-split",
        type=str,
        default="test",
        help="Dataset split for evaluation.",
    )
    env_group.add_argument(
        "--data-batch-size",
        type=int,
        default=8,
        help="Per-GPU data batch size.",
    )
    env_group.add_argument(
        "--max-context-length",
        type=int,
        default=None,
        help="Maximum tokenized context length.",
    )

    runtime_group = parser.add_argument_group("runtime")
    runtime_group.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device.",
    )
    runtime_group.add_argument(
        "--wb",
        action="store_true",
        help="Enable Weights & Biases logging.",
    )
    runtime_group.add_argument(
        "--wandb-api-key",
        type=str,
        default=None,
        help="Weights & Biases API key.",
    )
    return parser.parse_args()


def _load_conversation_template(path: Path | None) -> list[dict[str, str]] | None:
    if path is None:
        return None
    with Path(path).open() as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, list):
        msg = f"Conversation template must be a YAML list, got {type(data).__name__}"
        raise TypeError(msg)
    return data


def _build_manifest_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Build the algorithm and environment overrides from CLI flags."""
    algo_overrides: dict[str, Any] = {
        "pretrained_model_name_or_path": args.pretrained_model_name_or_path,
        "max_model_len": args.max_model_len,
        "lora_config": {
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "task_type": "CAUSAL_LM",
        },
    }

    env_section: dict[str, Any] = {
        "name": args.dataset,
        "dataset_name": args.dataset,
        "tokenizer_name": args.pretrained_model_name_or_path,
        "train_split": args.train_split,
        "test_split": args.test_split,
        "data_batch_size_per_gpu": args.data_batch_size,
    }

    if args.dataset_config is not None:
        env_section["dataset_config"] = args.dataset_config
    if args.reward_fn is not None:
        env_section["reward_fn"] = args.reward_fn
    if args.max_context_length is not None:
        env_section["max_context_length"] = args.max_context_length

    conversation_template = _load_conversation_template(args.conversation_template)
    if conversation_template is not None:
        env_section["conversation_template"] = conversation_template

    return {"algorithm": algo_overrides, "environment": env_section}


def main() -> None:
    """Run LLM evolutionary fine-tuning from a manifest."""
    args = parse_args()

    logger.info("Loading manifest: %s", args.manifest)
    with Path(args.manifest).open() as fh:
        manifest_data = yaml.safe_load(fh)

    overrides = _build_manifest_overrides(args)
    manifest_data.setdefault("algorithm", {}).update(overrides["algorithm"])
    manifest_data["environment"] = overrides["environment"]

    trainer = LocalTrainer.from_manifest(manifest_data, device=args.device)
    logger.info(
        "Algorithm: %s | Dataset: %s | Pop: %d",
        trainer.algorithm.name,
        args.dataset,
        trainer.training.population_size,
    )

    _population, fitness_history = trainer.train(
        wb=args.wb,
        wandb_api_key=args.wandb_api_key,
    )

    best_fitness = max(f for gen in fitness_history for f in gen)
    logger.info("Training complete. Best fitness: %.4f", best_fitness)


if __name__ == "__main__":
    main()

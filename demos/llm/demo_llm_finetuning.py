# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""LLM fine-tuning demo -- SFT and DPO with full CLI support.

Train, warm-start, or interactively evaluate LoRA-adapted language models.
Everything -- model, dataset, LoRA, population size, evolution -- is read from a
training manifest under ``configs/training/llm_finetuning/`` and handed to
:class:`~agilerl.training.trainer.LocalTrainer`.

Examples
--------
Train SFT::

    python demos/llm/demo_llm_finetuning.py sft

Train DPO from the base model::

    python demos/llm/demo_llm_finetuning.py dpo

Warm-start DPO from a prior SFT checkpoint (its actor becomes the DPO reference;
hyperparameters still come from the manifest)::

    python demos/llm/demo_llm_finetuning.py dpo --load-path outputs/20260101_120000_SFT

Evaluate a saved checkpoint interactively::

    python demos/llm/demo_llm_finetuning.py sft --eval --load-path outputs/20260101_120000_SFT

Multi-GPU / DeepSpeed via accelerate::

    accelerate launch demos/llm/demo_llm_finetuning.py sft
"""

from agilerl import HAS_LLM_DEPENDENCIES

if not HAS_LLM_DEPENDENCIES:
    msg = (
        "LLM dependencies are not installed. "
        "Install them with `pip install agilerl[llm]`."
    )
    raise ImportError(msg)

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from accelerate import Accelerator
from peft import LoraConfig
from transformers import AutoTokenizer

from agilerl.algorithms.dpo import DPO
from agilerl.algorithms.sft import SFT
from agilerl.training.trainer import LocalTrainer
from agilerl.utils.llm_utils import compare_responses, sample_eval_prompts

CONFIG_DIR = "configs/training/llm_finetuning"


def _make_accelerator() -> Accelerator | None:
    """Return an ``Accelerator`` only when launched under DeepSpeed."""
    try:
        accelerator = Accelerator()
    except Exception:
        return None
    return accelerator if accelerator.state.deepspeed_plugin is not None else None


def build_manifest(config_path: str) -> dict[str, Any]:
    """Load the training manifest.

    :param config_path: Path to the YAML manifest.
    :type config_path: str
    :return: The manifest dict, ready for ``LocalTrainer.from_manifest``.
    :rtype: dict[str, Any]
    """
    with open(config_path) as f:
        return yaml.safe_load(f)


def main(
    config_path: str,
    save_path: str = "outputs",
    load_path: str | None = None,
    wb: bool = False,
    eval_samples: int = 5,
) -> None:
    """Run an SFT or DPO fine-tuning loop from a training manifest.

    :param config_path: Path to the YAML manifest describing the run.
    :type config_path: str
    :param save_path: Directory to save the elite LoRA checkpoint.
    :type save_path: str
    :param load_path: Optional checkpoint directory to warm-start from (e.g. an SFT
        run when training DPO). Only its LoRA adapters are taken -- the optimizer
        starts fresh and the manifest keeps every hyperparameter.
    :type load_path: str | None
    :param wb: Whether to log the run to Weights & Biases.
    :type wb: bool
    :param eval_samples: Number of prompts used for the qualitative comparison.
    :type eval_samples: int
    """
    accelerator = _make_accelerator()
    manifest = build_manifest(config_path)

    print(f"Building trainer from {config_path} ...")
    if load_path is not None:
        print(f"Warm-starting from LoRA checkpoint {load_path} ...")
    trainer = LocalTrainer.from_manifest(
        manifest,
        load_weights_from=load_path,
        accelerator=accelerator,
    )

    print(f"Fine-tuning {trainer.algorithm_spec.name} agents...")
    pop, _fitnesses = trainer.train(
        save_elite=True,
        elite_path=save_path,
        wb=wb,
    )

    print("\nQualitative response comparison (elite agent):")
    elite = max(pop, key=lambda a: a.fitness[-1] if a.fitness else float("-inf"))
    prompts = sample_eval_prompts(trainer.env, n=eval_samples)
    compare_responses(elite, trainer.tokenizer, prompts)


def eval_mode(mode: str, load_path: str, max_new_tokens: int = 200) -> None:
    """Load a saved LoRA checkpoint and enter an interactive prompt loop.

    :param mode: ``"sft"`` or ``"dpo"`` — selects the agent class.
    :type mode: str
    :param load_path: Checkpoint directory written by training, holding an
        ``actor/`` LoRA adapter.
    :type load_path: str
    :param max_new_tokens: Maximum tokens to generate per response.
    :type max_new_tokens: int
    """
    # The adapter records the base it was trained against and its own LoRA config,
    # which is enough to rebuild the agent and load the adapter onto it.
    adapter_path = str(Path(load_path) / "actor")
    lora_config = LoraConfig.from_pretrained(adapter_path)
    base_model_name = lora_config.base_model_name_or_path

    print(f"Loading tokenizer from {base_model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Building agent on {base_model_name}, loading adapter {adapter_path} ...")
    agent_cls = SFT if mode == "sft" else DPO
    agent = agent_cls(
        model_name=base_model_name,
        lora_config=lora_config,
        pad_token_id=tokenizer.pad_token_id,
        pad_token=tokenizer.eos_token,
    )
    agent.load_weights(load_path)

    print(f"\nEval mode ready  |  base: {base_model_name}  |  adapter: {adapter_path}")
    print("Enter a prompt and press Enter to generate a response.")
    print("Type 'quit', 'q', 'exit', or press Ctrl+C to quit.\n")

    while True:
        try:
            prompt = input("Prompt> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting eval mode.")
            break

        if not prompt or prompt.lower() in {"quit", "q", "exit"}:
            print("Exiting eval mode.")
            break

        compare_responses(
            agent,
            tokenizer,
            [(prompt, None, None)],
            max_new_tokens=max_new_tokens,
            show_base_model=False,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM fine-tuning demo (SFT / DPO)")
    parser.add_argument("mode", choices=["sft", "dpo"], help="Fine-tuning algorithm")
    parser.add_argument(
        "--save-path",
        default="outputs",
        help="Base directory to save the elite LoRA checkpoint (default: outputs)",
    )
    parser.add_argument(
        "--no-timestamp",
        action="store_true",
        help="Disable the timestamp sub-directory, overwriting any existing checkpoint",
    )
    parser.add_argument(
        "--load-path",
        default=None,
        help=(
            "Checkpoint directory from a previous run (containing actor/ and "
            "attributes.pt). In training mode, warm-starts from its LoRA adapter "
            "(e.g. an SFT run when training DPO); hyperparameters still come from "
            "the manifest. In eval mode (--eval), loads it for inference."
        ),
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Enter interactive eval mode instead of training (requires --load-path)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
        help="Maximum tokens to generate per response in eval mode (default: 200)",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=5,
        help="Prompts used for the post-training comparison (default: 5)",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log the run to Weights & Biases",
    )
    parser.add_argument(
        "--config",
        default=None,
        help=f"Path to a training manifest (default: {CONFIG_DIR}/{{mode}}.yaml)",
    )
    args = parser.parse_args()

    if args.eval:
        if not args.load_path:
            parser.error("--load-path is required when --eval is set")
        eval_mode(
            mode=args.mode,
            load_path=args.load_path,
            max_new_tokens=args.max_new_tokens,
        )
    else:
        save_path = (
            args.save_path
            if args.no_timestamp
            else f"{args.save_path}/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.mode.upper()}"
        )
        main(
            config_path=args.config or f"{CONFIG_DIR}/{args.mode}.yaml",
            save_path=save_path,
            load_path=args.load_path,
            wb=args.wandb,
            eval_samples=args.eval_samples,
        )

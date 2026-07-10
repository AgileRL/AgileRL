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

Warm-start DPO from a prior SFT checkpoint::

    python demos/llm/demo_llm_finetuning.py dpo --load-path outputs/sft/actor

Evaluate a saved checkpoint interactively::

    python demos/llm/demo_llm_finetuning.py sft --eval --load-path outputs/sft/actor

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
import json
import shutil
import tempfile
from datetime import datetime
from typing import Any

import yaml
from accelerate import Accelerator
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

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


def _merge_adapter_into_base(load_path: str, dest: str) -> str:
    """Fold a saved LoRA adapter into its base model and save the dense result.

    AgileRL manages its own adapters on an immutable base and rejects
    ``PeftModel`` inputs, so a warm start is expressed as a dense model on disk
    that the manifest can point at.

    :param load_path: Directory holding ``adapter_config.json`` and the adapter weights.
    :type load_path: str
    :param dest: Directory to write the merged model and tokenizer to.
    :type dest: str
    :return: ``dest``, ready to use as ``pretrained_model_name_or_path``.
    :rtype: str
    """
    with open(f"{load_path}/adapter_config.json") as f:
        base_model_name = json.load(f)["base_model_name_or_path"]

    print(f"Merging LoRA adapter {load_path} into base {base_model_name} ...")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    merged = PeftModel.from_pretrained(
        base_model, load_path, adapter_name="actor"
    ).merge_and_unload()

    merged.save_pretrained(dest)
    AutoTokenizer.from_pretrained(base_model_name).save_pretrained(dest)
    return dest


def build_manifest(
    config_path: str, load_path: str | None, dest: str
) -> dict[str, Any]:
    """Load the training manifest, optionally repointing it at merged warm-start weights.

    :param config_path: Path to the YAML manifest.
    :type config_path: str
    :param load_path: Optional LoRA adapter directory to warm-start from.
    :type load_path: str | None
    :param dest: Directory to write merged warm-start weights to.
    :type dest: str
    :return: The manifest dict, ready for ``LocalTrainer.from_manifest``.
    :rtype: dict[str, Any]
    """
    with open(config_path) as f:
        manifest = yaml.safe_load(f)

    if load_path is not None:
        manifest["network"]["pretrained_model_name_or_path"] = _merge_adapter_into_base(
            load_path, dest
        )
    return manifest


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
    :param load_path: Optional path to a pre-trained LoRA checkpoint to warm-start from
        (e.g. an SFT adapter when running DPO).
    :type load_path: str | None
    :param wb: Whether to log the run to Weights & Biases.
    :type wb: bool
    :param eval_samples: Number of prompts used for the qualitative comparison.
    :type eval_samples: int
    """
    accelerator = _make_accelerator()
    warm_start_dir = tempfile.mkdtemp(prefix="agilerl_warm_start_")

    try:
        manifest = build_manifest(config_path, load_path, warm_start_dir)

        print(f"Building trainer from {config_path} ...")
        trainer = LocalTrainer.from_manifest(manifest, accelerator=accelerator)

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
    finally:
        shutil.rmtree(warm_start_dir, ignore_errors=True)


def eval_mode(mode: str, load_path: str, max_new_tokens: int = 200) -> None:
    """Load a saved LoRA checkpoint and enter an interactive prompt loop.

    :param mode: ``"sft"`` or ``"dpo"`` — selects the agent class.
    :type mode: str
    :param load_path: Path to a directory containing ``adapter_config.json``
        and the LoRA adapter weights.
    :type load_path: str
    :param max_new_tokens: Maximum tokens to generate per response.
    :type max_new_tokens: int
    """
    with open(f"{load_path}/adapter_config.json") as f:
        base_model_name = json.load(f)["base_model_name_or_path"]

    print(f"Loading tokenizer from {base_model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model from {base_model_name} ...")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)

    print(f"Applying LoRA adapter from {load_path} ...")
    # Fold the saved adapter into the base — AgileRL rejects PeftModel inputs.
    actor_network = PeftModel.from_pretrained(base_model, load_path).merge_and_unload()

    agent_cls = SFT if mode == "sft" else DPO
    agent = agent_cls(
        actor_network=actor_network,
        pad_token_id=tokenizer.eos_token_id,
        pad_token=tokenizer.eos_token,
    )

    print(f"\nEval mode ready  |  base: {base_model_name}  |  adapter: {load_path}")
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
            "Path to a LoRA checkpoint. In training mode, warm-starts from these weights "
            "(e.g. an SFT adapter for DPO). In eval mode (--eval), loads for interactive "
            "inference. Must contain adapter_model.safetensors / adapter_config.json."
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

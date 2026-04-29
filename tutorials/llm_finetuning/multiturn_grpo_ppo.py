"""Train LLMPPO, LLMREINFORCE, or GRPO on multi-turn GuessTheNumber.

This script is used by the multi-turn GRPO vs LLMPPO tutorial and keeps the
setup identical between runs so only the optimization algorithm changes.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import gem
import yaml
from transformers import AutoTokenizer

from agilerl import HAS_LLM_DEPENDENCIES
from agilerl.training.train_llm import finetune_llm_multiturn
from agilerl.utils.algo_utils import VLLMConfig
from agilerl.utils.llm_utils import create_llm_accelerator
from agilerl.utils.utils import create_population, _normalize_algo_name
from agilerl.llm_envs import TokenObservationWrapper

if not HAS_LLM_DEPENDENCIES:
    msg = (
        "LLM dependencies are not installed. "
        "Install them with `pip install agilerl[llm]`."
    )
    raise ImportError(msg)


DEFAULT_MODEL_PATH = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_ENV_NAME = "game:GuessTheNumber-v0-easy"
DEFAULT_PPO_CONFIG = "configs/training/llm_finetuning/ppo_llm.yaml"
DEFAULT_GRPO_CONFIG = "configs/training/llm_finetuning/grpo_multiturn.yaml"
DEFAULT_REINFORCE_CONFIG = "configs/training/llm_finetuning/reinforce_llm.yaml"


def _load_init_hp(config_path: str) -> dict[str, Any]:
    """Load and return INIT_HP from a YAML training config.

    :param config_path: Path to the YAML config file.
    :type config_path: str
    :return: Initial hyperparameter dictionary.
    :rtype: dict[str, Any]
    """
    with Path(config_path).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    init_hp = config.get("INIT_HP")
    if not isinstance(init_hp, dict):
        msg = f"Missing or invalid INIT_HP in config: {config_path}"
        raise ValueError(msg)
    return dict(init_hp)


def _default_config_for_algo(algo: str) -> str:
    """Return tutorial default config path for the selected algorithm."""
    algo_name = _normalize_algo_name(algo)
    if algo_name == "LLMPPO":
        return DEFAULT_PPO_CONFIG
    if algo_name == "GRPO":
        return DEFAULT_GRPO_CONFIG
    if algo_name == "LLMREINFORCE":
        return DEFAULT_REINFORCE_CONFIG
    msg = f"Unsupported algorithm '{algo}'. Use LLMPPO, LLMREINFORCE, or GRPO."
    raise ValueError(msg)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the tutorial training script."""
    parser = argparse.ArgumentParser(
        description="Multi-turn LLMPPO/GRPO tutorial on GuessTheNumber."
    )
    parser.add_argument(
        "--algo",
        type=str,
        choices=["LLMPPO", "LLMREINFORCE", "GRPO"],
        default="LLMPPO",
        help="Algorithm to train.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to a YAML config. Defaults by --algo.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Hugging Face model path.",
    )
    parser.add_argument(
        "--env-name",
        type=str,
        default=DEFAULT_ENV_NAME,
        help="GEM environment id.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=2048,
        help="Total sample steps for tutorial runs.",
    )
    parser.add_argument(
        "--evaluation-interval",
        type=int,
        default=10,
        help="Training iterations between evaluation logs.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="saved_llms/multiturn_tutorial",
        help="Directory to save checkpoints.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging.",
    )
    return parser.parse_args()


def main() -> None:
    """Run multi-turn training with LLMPPO, LLMREINFORCE, or GRPO."""
    args = parse_args()
    config_path = args.config or _default_config_for_algo(args.algo)
    init_hp = _load_init_hp(config_path)
    init_hp["ALGO"] = args.algo

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    env_probe = gem.make(args.env_name)
    max_turns = env_probe.max_turns
    if hasattr(env_probe, "close"):
        env_probe.close()

    def env_factory() -> TokenObservationWrapper:
        """Create one wrapped multi-turn environment instance."""
        env = gem.make(args.env_name)
        return TokenObservationWrapper(
            env=env,
            tokenizer=tokenizer,
            max_turns=max_turns,
            pad_id=tokenizer.pad_token_id,
            apply_chat_template=True,
            max_model_len=init_hp.get("MAX_MODEL_LEN"),
            max_output_tokens=init_hp.get("MAX_OUTPUT_TOKENS"),
        )

    accelerator = create_llm_accelerator()
    use_vllm = bool(init_hp.get("USE_VLLM", True))
    vllm_config = (
        VLLMConfig(
            tensor_parallel_size=1,
            gpu_memory_utilization=0.85,
            max_num_seqs=16,
            sleep_mode=True,
        )
        if use_vllm
        else None
    )

    pop = create_population(
        algo=args.algo,
        net_config=None,
        INIT_HP=init_hp,
        population_size=1,
        accelerator=accelerator,
        tokenizer=tokenizer,
        model_name=args.model_path,
        vllm_config=vllm_config,
    )
    agent = pop[0]

    try:
        finetune_llm_multiturn(
            pop=[agent],
            max_turns=max_turns,
            env_factory=env_factory,
            init_hp=init_hp,
            max_steps=args.max_steps,
            save_elite=True,
            elite_path=args.output_dir,
            wb=args.wandb,
            evo_steps=None,
            tournament=None,
            mutation=None,
            evaluation_interval=args.evaluation_interval,
            max_reward=1.0,
            verbose=True,
            accelerator=accelerator,
        )
    finally:
        if accelerator is not None:
            accelerator.end_training()


if __name__ == "__main__":
    main()

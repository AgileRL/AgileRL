from agilerl import HAS_LLM_DEPENDENCIES

if not HAS_LLM_DEPENDENCIES:
    msg = "LLM dependencies are not installed. Please install them using `pip install agilerl[llm]`."
    raise ImportError(
        msg,
    )

import argparse
import os

import gem
import yaml
from transformers import AutoTokenizer
from agilerl.algorithms import LLMPPO, LLMREINFORCE, GRPO, CISPO, GSPO
from agilerl.training.train_llm import finetune_llm_multiturn
from agilerl.utils.algo_utils import VLLMConfig
from agilerl.utils.llm_utils import create_llm_accelerator
from agilerl.utils.utils import create_population
from agilerl.llm_envs import (
    TokenObservationWrapper,
)

MODEL_PATH = "Qwen/Qwen2.5-0.5B-Instruct"
ENV_NAME = "game:GuessTheNumber-v0-easy"

ALGO_REGISTRY = {
    "LLMPPO": LLMPPO,
    "LLMREINFORCE": LLMREINFORCE,
    "GRPO": GRPO,
    "CISPO": CISPO,
    "GSPO": GSPO,
}


def main(
    init_hp,
    mut_p,
    *,
    wb: bool,
    wandb_api_key: str | None,
    wandb_project: str,
    wandb_entity: str | None,
    wandb_run_name: str | None,
):
    algo_name = init_hp["ALGO"]
    algo_cls = ALGO_REGISTRY.get(algo_name)
    if algo_cls is None:
        msg = f"Unknown algorithm '{algo_name}'. Supported: {', '.join(ALGO_REGISTRY)}"
        raise ValueError(msg)

    model_name = MODEL_PATH
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    base_env = gem.make(ENV_NAME)
    max_turns = base_env.max_turns

    def env_factory():
        env = gem.make(ENV_NAME)
        return TokenObservationWrapper(
            env,
            tokenizer,
            max_turns,
            tokenizer.pad_token_id,
            max_model_len=init_hp.get("MAX_MODEL_LEN", None),
            max_output_tokens=init_hp.get("MAX_OUTPUT_TOKENS", None),
        )

    accelerator = create_llm_accelerator()

    pop_size = init_hp.get("POP_SIZE", 1)
    vllm_sleep = pop_size == 1

    vllm_config = (
        VLLMConfig(
            tensor_parallel_size=1,
            gpu_memory_utilization=0.45,
            max_num_seqs=16,
            sleep_mode=vllm_sleep,
        )
        if init_hp.get("USE_VLLM", False)
        else None
    )
    pop = create_population(
        algo=algo_name,
        net_config=None,
        INIT_HP=init_hp,
        population_size=pop_size,
        accelerator=accelerator,
        tokenizer=tokenizer,
        model_name=model_name,
        vllm_config=vllm_config,
    )

    finetune_llm_multiturn(
        pop=pop,
        max_turns=max_turns,
        init_hp=init_hp,
        wb=wb,
        wandb_api_key=wandb_api_key,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_run_name=wandb_run_name,
        save_elite=True,
        elite_path="saved_llms",
        evo_steps=None,
        mutation=None,
        tournament=None,
        evaluation_interval=10,
        max_reward=1.0,
        verbose=True,
        max_steps=300_000,
        accelerator=accelerator,
        env_factory=env_factory,
    )
    if accelerator is not None:
        accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-turn LLM benchmarking")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training/llm_finetuning/ppo_llm.yaml",
        help="Path to the YAML config file",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=os.environ.get("WANDB_PROJECT", "AgileRL"),
        help="W&B project (default: AgileRL or $WANDB_PROJECT)",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=os.environ.get("WANDB_ENTITY"),
        help="W&B entity / team (optional; fallback $WANDB_ENTITY)",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=os.environ.get("WANDB_RUN_NAME"),
        help="W&B run name (optional; fallback $WANDB_RUN_NAME)",
    )
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)
    init_hp = config["INIT_HP"]
    mut_p = config["MUTATION_PARAMS"]
    wandb_key = os.environ.get("WANDB_API_KEY")
    main(
        init_hp,
        mut_p,
        wb=not args.no_wandb,
        wandb_api_key=wandb_key,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
    )

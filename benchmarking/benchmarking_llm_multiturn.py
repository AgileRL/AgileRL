from agilerl import HAS_LLM_DEPENDENCIES

if not HAS_LLM_DEPENDENCIES:
    msg = "LLM dependencies are not installed. Please install them using `pip install agilerl[llm]`."
    raise ImportError(
        msg,
    )

import argparse

import gem
from huggingface_hub import snapshot_download
import yaml
from transformers import AutoTokenizer
from agilerl.algorithms import LLMPPO, LLMREINFORCE, GRPO
from agilerl.training.train_llm import finetune_llm_multiturn
from agilerl.utils.algo_utils import VLLMConfig
from agilerl.utils.llm_utils import create_llm_accelerator
from agilerl.utils.utils import create_population
from agilerl.llm_envs import (
    FormatRewardWrapper,
    SearchTool,
    TokenObservationWrapper,
)

MODEL_PATH = "Qwen/Qwen2.5-0.5B-Instruct"
ENV_NAME = "game:GuessTheNumber-v0-easy"

ALGO_REGISTRY = {
    "LLMPPO": LLMPPO,
    "LLMREINFORCE": LLMREINFORCE,
    "GRPO": GRPO,
}


def main(init_hp, mut_p):
    algo_name = init_hp["ALGO"]
    algo_cls = ALGO_REGISTRY.get(algo_name)
    if algo_cls is None:
        msg = f"Unknown algorithm '{algo_name}'. Supported: {', '.join(ALGO_REGISTRY)}"
        raise ValueError(msg)

    actor_network = None
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

    vllm_config = (
        VLLMConfig(
            tensor_parallel_size=1,
            gpu_memory_utilization=0.85,
            max_num_seqs=16,
            sleep_mode=True,
        )
        if init_hp.get("USE_VLLM", False)
        else None
    )
    pop = create_population(
        algo=algo_name,
        net_config=None,
        INIT_HP=init_hp,
        population_size=1,
        accelerator=accelerator,
        tokenizer=tokenizer,
        model_name=model_name,
        actor_network=actor_network,
        vllm_config=vllm_config,
    )
    agent = pop[0]

    finetune_llm_multiturn(
        pop=[agent],
        max_turns=max_turns,
        init_hp=init_hp,
        wb=True,
        save_elite=True,
        elite_path="saved_llms",
        evo_steps=None,
        mutation=None,
        tournament=None,
        evaluation_interval=10,
        max_reward=1.0,
        verbose=True,
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
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)
    init_hp = config["INIT_HP"]
    mut_p = config["MUTATION_PARAMS"]
    main(init_hp, mut_p)

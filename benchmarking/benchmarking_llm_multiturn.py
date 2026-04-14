from agilerl import HAS_LLM_DEPENDENCIES

if not HAS_LLM_DEPENDENCIES:
    msg = "LLM dependencies are not installed. Please install them using `pip install agilerl[llm]`."
    raise ImportError(
        msg,
    )

import argparse

import gem
from huggingface_hub import snapshot_download
from gem.tools.tool_env_wrapper import ToolEnvWrapper
import yaml
from transformers import AutoTokenizer
from agilerl.algorithms import LLMPPO, LLMReinforce, GRPO
from agilerl.training.train_llm import finetune_llm_multiturn
from agilerl.utils.algo_utils import VLLMConfig
from agilerl.utils.llm_utils import create_llm_accelerator
from agilerl.utils.utils import create_population
from agilerl.wrappers.gem_wrappers import (
    FormatRewardWrapper,
    SearchTool,
    TokenObservationWrapper,
)

MODEL_PATH = "Qwen/Qwen2.5-0.5B-Instruct"
ENV_NAME = "game:GuessTheNumber-v0-easy"
USE_TINY_DEBUG_MODEL = False
USE_VLLM = not USE_TINY_DEBUG_MODEL
PRELOAD_MODEL = True

ALGO_REGISTRY = {
    "LLMPPO": LLMPPO,
    "LLMReinforce": LLMReinforce,
    "GRPO": GRPO,
}


def _download_model_to_cache(model_name: str) -> str:
    """Pre-download model artifacts into local HF cache and return cache path."""
    print(f"Pre-downloading model to HF cache: {model_name}")
    local_path = snapshot_download(
        repo_id=model_name,
        resume_download=True,
    )
    print(f"Model cached at: {local_path}")
    return local_path


def main(init_hp, mut_p):
    algo_name = init_hp["ALGO"]
    algo_cls = ALGO_REGISTRY.get(algo_name)
    if algo_cls is None:
        msg = f"Unknown algorithm '{algo_name}'. Supported: {', '.join(ALGO_REGISTRY)}"
        raise ValueError(msg)

    if USE_TINY_DEBUG_MODEL:
        from benchmarking.tiny_model import TinyDigitTokenizer, build_tiny_actor_network

        actor_network = build_tiny_actor_network()
        tokenizer = TinyDigitTokenizer()
        model_name = None
        target_modules = ["c_attn", "c_proj", "c_fc"]
        apply_chat_template = False
    else:
        actor_network = None
        model_name = MODEL_PATH
        if PRELOAD_MODEL:
            _download_model_to_cache(model_name)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "down_proj",
            "gate_proj",
        ]
        apply_chat_template = True
    search_tool = SearchTool(search_url="http://localhost:8888/search")
    base_env = gem.make(ENV_NAME)
    # tool_env = ToolEnvWrapper(
    #     env=sample_env,
    #     tools=[search_tool],
    #     tool_success_reward=0.1,
    #     max_tool_uses=2,
    # )
    max_turns = base_env.max_turns

    # fmt_env = FormatRewardWrapper(tool_env)
    def env_factory():
        env = gem.make(ENV_NAME)
        return TokenObservationWrapper(
            env,
            tokenizer,
            max_turns,
            tokenizer.pad_token_id,
            apply_chat_template=apply_chat_template,
            max_model_len=init_hp.get("MAX_MODEL_LEN", None),
            max_output_tokens=init_hp.get("MAX_OUTPUT_TOKENS", None),
        )

    accelerator = create_llm_accelerator() if not USE_TINY_DEBUG_MODEL else None

    vllm_config = (
        VLLMConfig(
            tensor_parallel_size=1,
            gpu_memory_utilization=0.85,
            max_num_seqs=16,
            sleep_mode=True,
        )
        if USE_VLLM
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

from agilerl import HAS_LLM_DEPENDENCIES

if not HAS_LLM_DEPENDENCIES:
    msg = "LLM dependencies are not installed. Please install them using `pip install agilerl[llm]`."
    raise ImportError(
        msg,
    )

import os

import gem
import yaml
from transformers import AutoTokenizer

from agilerl.algorithms import CISPO, GRPO, GSPO, LLMPPO, LLMREINFORCE
from agilerl.llm_envs import RolloutHarness
from agilerl.training.train_llm import finetune_llm_multiturn
from agilerl.utils.algo_utils import VLLMConfig
from agilerl.utils.llm_utils import (
    build_bnb_quantization_config,
    create_llm_accelerator,
)
from agilerl.utils.utils import create_population

CONFIG_PATH = "configs/training/llm_finetuning/cispo_quant_bench.yaml"
MODEL_PATH = "google/gemma-4-E4B-it"
ENV_NAME = "game:Sudoku-v0-hard"

ALGO_REGISTRY = {
    "LLMPPO": LLMPPO,
    "LLMREINFORCE": LLMREINFORCE,
    "GRPO": GRPO,
    "CISPO": CISPO,
    "GSPO": GSPO,
}


def _patch_sudoku_prompt(reset_method):
    r"""Wrap a gem Sudoku env ``reset`` to fix errors in its instruction prompt.

    gem's ``SudokuEnv._get_instructions`` renders rule 3 as
    ``"nine {scale}x{scale} subgrids"``, which is wrong: a 9x9 board (hard) has
    nine **3x3** boxes and a 4x4 board (easy) has **four 2x2** boxes. The worked
    example ``\boxed{1 1 5}`` is also out of range on the 4x4 (easy) board, whose
    digits only go up to 4. Each replacement matches only its own variant, so the
    wrapper is safe to apply to either difficulty.
    """

    def wrapper(*args, **kwargs):
        obs_text, info = reset_method(*args, **kwargs)
        obs_text = obs_text.replace("nine 9x9 subgrids", "nine 3x3 subgrids")
        obs_text = obs_text.replace("nine 4x4 subgrids", "four 2x2 subgrids")
        obs_text = obs_text.replace(r"\boxed{1 1 5}", r"\boxed{1 1 4}")
        return obs_text, info

    return wrapper


def main(init_hp, mut_p):
    algo_name = init_hp["ALGO"]
    if algo_name not in ALGO_REGISTRY:
        msg = f"Unknown algorithm '{algo_name}'. Supported: {', '.join(ALGO_REGISTRY)}"
        raise ValueError(msg)

    quantization_config = build_bnb_quantization_config(
        init_hp.get("QUANTIZATION", "none")
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    base_env = gem.make(ENV_NAME)
    rollout_max_turns = init_hp.get("MAX_TURNS", getattr(base_env, "max_turns", 32))

    def env_factory():
        env = gem.make(ENV_NAME)
        if "sudoku" in ENV_NAME.lower():
            # Fix gem's Sudoku instruction-prompt bugs (subgrid wording + an
            # out-of-range worked example on the easy/4x4 board).
            env.reset = _patch_sudoku_prompt(env.reset)
        return RolloutHarness(
            env,
            tokenizer,
            rollout_max_turns,
            tokenizer.pad_token_id,
            max_model_len=init_hp.get("MAX_MODEL_LEN", None),
            max_output_tokens=init_hp.get("MAX_OUTPUT_TOKENS", None),
            enable_sliding_window=init_hp.get("ENABLE_SLIDING_WINDOW", False),
        )

    accelerator = create_llm_accelerator()

    # Colocated vLLM rollout: vLLM and the trainer each hold their own base and
    # the GPU is shared via vLLM native sleep/wake (the engine's base is backed
    # up to host RAM on sleep and restored on wake; the trainer's base is
    # offloaded to CPU during rollout). Only LoRA adapters are synced per
    # rollout. vLLM mirrors the trainer's precision — bitsandbytes when the
    # trainer quantizes, dense bf16 otherwise.
    use_vllm = bool(init_hp.get("USE_VLLM", True))
    vllm_config = (
        VLLMConfig(
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            max_num_seqs=10,
            sleep_mode=init_hp.get("POP_SIZE", 1) == 1,
            quantization=("bitsandbytes" if quantization_config is not None else None),
            dtype="bfloat16",
            strip_multimodal_towers=True,
        )
        if use_vllm
        else None
    )

    algo_kwargs = {}
    if quantization_config is not None:
        algo_kwargs["quantization_config"] = quantization_config

    pop = create_population(
        algo=algo_name,
        net_config=None,
        INIT_HP=init_hp,
        population_size=init_hp.get("POP_SIZE", 1),
        accelerator=accelerator,
        tokenizer=tokenizer,
        model_name=MODEL_PATH,
        vllm_config=vllm_config,
        algo_kwargs=algo_kwargs or None,
    )

    finetune_llm_multiturn(
        pop=pop,
        max_turns=rollout_max_turns,
        env_factory=env_factory,
        init_hp=init_hp,
        wb=True,
        # WANDB_API_KEY / WANDB_PROJECT / WANDB_ENTITY are the env vars wandb
        # itself documents; pass them through rather than invent config keys.
        wandb_api_key=os.environ.get("WANDB_API_KEY"),
        wandb_project=os.environ.get("WANDB_PROJECT", "AgileRL"),
        wandb_entity=os.environ.get("WANDB_ENTITY"),
        wandb_run_name=os.environ.get("WANDB_RUN_NAME"),
        save_elite=True,
        elite_path="saved_llms",
        evaluation_interval=10,
        max_reward=1.0,
        verbose=True,
        accelerator=accelerator,
    )
    if accelerator is not None:
        accelerator.end_training()


if __name__ == "__main__":
    with open(CONFIG_PATH) as file:
        config = yaml.safe_load(file)
    init_hp = config["INIT_HP"]
    mut_p = config["MUTATION_PARAMS"]
    main(init_hp, mut_p)

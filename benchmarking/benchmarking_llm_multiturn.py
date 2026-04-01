from agilerl import HAS_LLM_DEPENDENCIES

if not HAS_LLM_DEPENDENCIES:
    msg = "LLM dependencies are not installed. Please install them using `pip install agilerl[llm]`."
    raise ImportError(
        msg,
    )

import argparse

import gem
from gem.tools.tool_env_wrapper import ToolEnvWrapper
import yaml
from peft import LoraConfig
from transformers import AutoTokenizer
from agilerl.algorithms import LLMPPO, LLMReinforce
from agilerl.training.train_llm import finetune_llm_multiturn
from agilerl.utils.algo_utils import VLLMConfig
from agilerl.utils.llm_utils import create_llm_accelerator
from agilerl.wrappers.gem_wrappers import (
    FormatRewardWrapper,
    SearchTool,
    TokenObservationWrapper,
)

MODEL_PATH = "Qwen/Qwen2.5-0.5B-Instruct"
ENV_NAME = "qa:NaturalQuestions"
MAX_CONTEXT_LENGTH = 4096
MAX_OUTPUT_TOKENS = 256
USE_TINY_DEBUG_MODEL = False
USE_VLLM = not USE_TINY_DEBUG_MODEL

ALGO_REGISTRY = {
    "LLMPPO": LLMPPO,
    "LLMReinforce": LLMReinforce,
}


def _build_agent(
    algo_cls,
    init_hp,
    *,
    model_name,
    actor_network,
    target_modules,
    tokenizer,
    accelerator,
):
    """Construct the agent from the YAML config, passing only the kwargs each
    algorithm class accepts."""
    lora_config = LoraConfig(
        r=16,
        lora_alpha=64,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    vllm_config = (
        VLLMConfig(
            tensor_parallel_size=1,
            gpu_memory_utilization=0.3,
            max_num_seqs=4,
            sleep_mode=True,
        )
        if USE_VLLM
        else None
    )

    common_kwargs = dict(
        model_name=model_name,
        actor_network=actor_network,
        lora_config=lora_config,
        micro_batch_size_per_gpu=1,
        use_vllm=USE_VLLM,
        pad_token_id=tokenizer.pad_token_id,
        pad_token=tokenizer.pad_token,
        use_separate_reference_adapter=True,
        batch_size=init_hp["BATCH_SIZE"],
        beta=init_hp["BETA"],
        clip_coef=init_hp["CLIP_COEF"],
        max_grad_norm=init_hp["MAX_GRAD_NORM"],
        update_epochs=init_hp["UPDATE_EPOCHS"],
        temperature=init_hp["TEMPERATURE"],
        max_output_tokens=MAX_OUTPUT_TOKENS,
        max_model_len=MAX_CONTEXT_LENGTH,
        accelerator=accelerator,
        vllm_config=vllm_config,
        gradient_checkpointing=True,
    )

    if algo_cls is LLMPPO:
        common_kwargs |= dict(
            lr_actor=init_hp["LR_ACTOR"],
            lr_critic=init_hp.get("LR_CRITIC"),
            vf_coef=init_hp.get("VF_COEF", 0.5),
            gamma=init_hp.get("GAMMA", 1.0),
            gae_lambda=init_hp.get("GAE_LAMBDA", 1.0),
        )
    elif algo_cls is LLMReinforce:
        common_kwargs |= dict(
            lr=init_hp["LR"],
            gamma=init_hp.get("GAMMA", 1.0),
        )

    return algo_cls(**common_kwargs)


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
    sample_env = gem.make(ENV_NAME)
    tool_env = ToolEnvWrapper(
        env=sample_env,
        tools=[search_tool],
        tool_success_reward=0.1,
        max_tool_uses=2,
    )
    max_turns = tool_env.max_tool_uses + 1
    fmt_env = FormatRewardWrapper(tool_env)
    env = TokenObservationWrapper(
        fmt_env,
        tokenizer,
        max_turns,
        tokenizer.pad_token_id,
        apply_chat_template=apply_chat_template,
        max_model_len=None if USE_TINY_DEBUG_MODEL else MAX_CONTEXT_LENGTH,
        max_output_tokens=None if USE_TINY_DEBUG_MODEL else MAX_OUTPUT_TOKENS,
    )
    accelerator = create_llm_accelerator() if not USE_TINY_DEBUG_MODEL else None

    init_hp["MAX_MODEL_LEN"] = MAX_CONTEXT_LENGTH
    assert tokenizer.pad_token_id != tokenizer.eos_token_id, (
        "Pad token and eos token are the same"
    )

    agent = _build_agent(
        algo_cls,
        init_hp,
        model_name=model_name,
        actor_network=actor_network,
        target_modules=target_modules,
        tokenizer=tokenizer,
        accelerator=accelerator,
    )

    finetune_llm_multiturn(
        pop=[agent],
        env=env,
        tokenizer=tokenizer,
        max_turns=max_turns,
        init_hp=init_hp,
        wb=False,
        save_elite=True,
        elite_path="saved_llms",
        evo_steps=None,
        mutation=None,
        tournament=None,
        evaluation_interval=10,
        max_reward=1.0,
        verbose=True,
        accelerator=accelerator,
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

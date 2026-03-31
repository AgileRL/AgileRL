from agilerl import HAS_LLM_DEPENDENCIES

if not HAS_LLM_DEPENDENCIES:
    msg = "LLM dependencies are not installed. Please install them using `pip install agilerl[llm]`."
    raise ImportError(
        msg,
    )

import gem
import yaml
from peft import LoraConfig
from transformers import AutoTokenizer
from gem.tools.search_tool import SearchTool
from gem.tools.tool_env_wrapper import ToolEnvWrapper

from agilerl.algorithms import LLMPPO
from agilerl.training.train_llm import finetune_llm_multiturn
from agilerl.utils.algo_utils import VLLMConfig
from agilerl.utils.llm_utils import create_llm_accelerator
from agilerl.wrappers.token_observation import TokenObservationWrapper

MODEL_PATH = "Qwen/Qwen2.5-0.5B-Instruct"
ENV_NAME = "qa:NaturalQuestions"
MAX_CONTEXT_LENGTH = 4096
MAX_OUTPUT_TOKENS = 512
USE_TINY_DEBUG_MODEL = False
USE_VLLM = not USE_TINY_DEBUG_MODEL


def main(init_hp, mut_p):
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
    search_tool = SearchTool(search_url="https://www.google.com")
    sample_env = gem.make(ENV_NAME)
    max_turns = 10 # sample_env.max_turns
    tool_env = ToolEnvWrapper(
        env=sample_env,
        tools=[search_tool],    
    )
    env = TokenObservationWrapper(
        tool_env,
        tokenizer,
        max_turns,
        tokenizer.pad_token_id,
        apply_chat_template=apply_chat_template,
    )
    accelerator = create_llm_accelerator() if not USE_TINY_DEBUG_MODEL else None

    init_hp["ALGO"] = "LLMPPO"
    init_hp["MAX_MODEL_LEN"] = MAX_CONTEXT_LENGTH
    assert tokenizer.pad_token_id != tokenizer.eos_token_id, (
        "Pad token and eos token are the same"
    )

    llm_ppo = LLMPPO(
        model_name=model_name,
        actor_network=actor_network,
        lora_config=LoraConfig(
            r=16,
            lora_alpha=64,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        ),
        micro_batch_size_per_gpu=2,
        use_vllm=USE_VLLM,
        pad_token_id=tokenizer.pad_token_id,
        pad_token=tokenizer.pad_token,
        use_separate_reference_adapter=True,
        batch_size=init_hp["BATCH_SIZE"],
        beta=init_hp["BETA"],
        lr_actor=init_hp["LR_ACTOR"],
        lr_critic=init_hp["LR_CRITIC"],
        clip_coef=init_hp["CLIP_COEF"],
        max_grad_norm=init_hp["MAX_GRAD_NORM"],
        update_epochs=init_hp["UPDATE_EPOCHS"],
        temperature=init_hp["TEMPERATURE"],
        max_output_tokens=MAX_OUTPUT_TOKENS,
        max_model_len=MAX_CONTEXT_LENGTH,
        accelerator=accelerator,
        vf_coef=init_hp["VF_COEF"],
        gamma=init_hp["GAMMA"],
        gae_lambda=init_hp["GAE_LAMBDA"],
        vllm_config=VLLMConfig(
            tensor_parallel_size=1,
            gpu_memory_utilization=0.4,
            max_num_seqs=4,
            sleep_mode=True,
        )
        if USE_VLLM
        else None,
        gradient_checkpointing=True,
    )

    finetune_llm_multiturn(
        pop=[llm_ppo],
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
        apply_chat_template=apply_chat_template,
    )
    if accelerator is not None:
        accelerator.end_training()


if __name__ == "__main__":
    with open("configs/training/llm_finetuning/ppo_llm.yaml") as file:
        config = yaml.safe_load(file)
    init_hp = config["INIT_HP"]
    mut_p = config["MUTATION_PARAMS"]
    main(init_hp, mut_p)

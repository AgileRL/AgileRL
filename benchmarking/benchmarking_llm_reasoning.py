from agilerl import HAS_LLM_DEPENDENCIES

if not HAS_LLM_DEPENDENCIES:
    msg = "LLM dependencies are not installed. Please install them using `pip install agilerl[llm]`."
    raise ImportError(
        msg,
    )

import re

import yaml
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from agilerl.algorithms import LLMPPO
from agilerl.training.train_llm import finetune_llm_reasoning
from agilerl.utils.llm_utils import ReasoningGym
from agilerl.utils.algo_utils import VLLMConfig
from agilerl.utils.utils import create_population

MODEL_PATH = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET = "Jiayi-Pan/Countdown-Tasks-3to4"
MAX_CONTEXT_LENGTH = 756
USE_TINY_DEBUG_MODEL = False
USE_VLLM = not USE_TINY_DEBUG_MODEL

def make_dataset(dataset_name: str) -> tuple[Dataset, Dataset]:
    raw_dataset = (
        load_dataset(dataset_name, split="train").shuffle(seed=42).select(range(50000))
    )
    raw_dataset = raw_dataset.rename_column("target", "answer")
    raw_dataset = raw_dataset.rename_column("nums", "question")
    train_test_split = raw_dataset.train_test_split(test_size=0.2)
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]
    return train_dataset, test_dataset


def format_reward_func(completions, target, **kwargs):
    rewards = []
    for completion, _gt in zip(completions, target, strict=False):
        try:
            # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
            completion = "<think>" + completion
            regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"
            matches = re.search(regex, completion, re.DOTALL)
            if matches is None or len(matches.groups()) != 2:
                rewards.append(0.0)
            else:
                rewards.append(1.0)
        except Exception:  # noqa: PERF203
            rewards.append(0.0)
    return rewards


def equation_reward_func(completions, target, nums, **kwargs):
    rewards = []

    for completion, gt, numbers in zip(completions, target, nums, strict=False):
        try:
            # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
            completion = "<think>" + completion
            answer_tags = re.findall(r"<answer>([\s\S]*?)<\/answer>", completion)

            if len(answer_tags) != 1:
                rewards.append(0.0)
                continue

            equation = answer_tags[0].strip()
            used_numbers = [int(n) for n in re.findall(r"\d+", equation)]

            if sorted(used_numbers) != sorted(numbers):
                rewards.append(0.0)
                continue

            allowed_pattern = r"^[\d+\-*/().\s]+$"
            if not re.match(allowed_pattern, equation):
                rewards.append(0.0)
                continue

            result = eval(equation, {"__builtins__": None}, {})

            if abs(float(result) - float(gt)) < 1e-5:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        except Exception:
            rewards.append(0.0)
    return rewards


def combined_rewards(completion, solution, prompt):


    return (
        equation_reward_func([completion], [solution], [prompt])[0]
        + format_reward_func([completion], [solution])[0]
    )


def main(init_hp, mut_p):

    if USE_TINY_DEBUG_MODEL:
        from benchmarking.tiny_model import build_tiny_actor_network, TinyDigitTokenizer
        actor_network = build_tiny_actor_network()
        tokenizer = TinyDigitTokenizer()
        model_name = None
        target_modules=["c_attn", "c_proj", "c_fc"]
    else:
        actor_network = None
        model_name = MODEL_PATH
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        target_modules = ["q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj"]

    # tokenizer.pad_token = tokenizer.eos_token
    train_dataset, test_dataset = make_dataset(DATASET)

    # Define a conversation template for the reasoning task, refer to questions and answers as q and a respectively
    conversation_template = [
        {
            "role": "system",
            "content": "You are a helpful assistant. You first think about the reasoning process in your mind and then provide the user with the answer.",
        },
        {
            "role": "user",
            "content": "Using each number in this list only once {question}, create an equation that equals {answer}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final equation and answer in <answer> </answer> tags, for example <answer>(1 + 2) / 3</answer>.",
        },
        {"role": "assistant", "content": "Let me solve this step by step.\n<think>"},
    ]

    # Convert the HuggingFace dataset into a Gymnasium environment
    accelerator = Accelerator() if not USE_TINY_DEBUG_MODEL else None
    env = ReasoningGym(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        reward_fn=combined_rewards,
        conversation_template=conversation_template,
        data_batch_size_per_gpu=init_hp["BATCH_SIZE"], # FIXME this needs fixing
        accelerator=accelerator,
        max_context_length=MAX_CONTEXT_LENGTH,
        return_raw_completions=USE_VLLM,
    )

    # Add the zero stage to the initialization hyperparameters
    init_hp["ALGO"] = "LLMPPO"
    # init_hp["ZERO_STAGE"] = accelerator.state.deepspeed_plugin.deepspeed_config[
    #     "zero_optimization"
    # ]["stage"]
    init_hp["MAX_MODEL_LEN"] = MAX_CONTEXT_LENGTH


    llm_ppo = LLMPPO(
        model_name=model_name,
        actor_network=actor_network,
        lora_config=LoraConfig(
            r=16,
            lora_alpha=64,
            target_modules=target_modules,
            bias="none",
            modules_to_save=["summary"],
            task_type="CAUSAL_LM",
        ),
        micro_batch_size_per_gpu=8,
        use_vllm=USE_VLLM,
        pad_token_id=tokenizer.pad_token_id,
        pad_token=tokenizer.pad_token,
        use_separate_reference_adapter=False,
        batch_size=init_hp["BATCH_SIZE"],
        beta=init_hp["BETA"],
        lr=init_hp["LR"],
        clip_coef=init_hp["CLIP_COEF"],
        max_grad_norm=init_hp["MAX_GRAD_NORM"],
        update_epochs=init_hp["UPDATE_EPOCHS"],
        temperature=init_hp["TEMPERATURE"],
        max_model_len=init_hp["MAX_MODEL_LEN"],
        accelerator=accelerator,
        vf_coef=init_hp["VF_COEF"],
        gamma=init_hp["GAMMA"],
        gae_lambda=init_hp["GAE_LAMBDA"],
        vllm_config=VLLMConfig(
            tensor_parallel_size=1,
            gpu_memory_utilization=0.5,
            max_num_seqs=2,
            sleep_mode=True,
        )
    )

    print("llm_ppo.lr", llm_ppo.lr)

    algo_kwargs = {
        "model_name": MODEL_PATH,
        "lora_config": LoraConfig(
            r=16,
            lora_alpha=64,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
        ),
        "use_vllm": USE_VLLM,
        "pad_token_id": tokenizer.pad_token_id,
        "pad_token": tokenizer.pad_token,
    }

    # pop = create_population(
    #     algo=init_hp["ALGO"],
    #     net_config=None,
    #     INIT_HP=init_hp,
    #     hp_config=None,
    #     population_size=init_hp["POP_SIZE"],
    #     accelerator=accelerator,
    #     algo_kwargs=algo_kwargs,
    # )


    finetune_llm_reasoning(
        pop=[llm_ppo],
        env=env,
        init_hp=init_hp,
        evaluation_interval=10,
        wb=True,
        save_elite=True,
        elite_path="saved_llms",    
        max_reward=2.0,
        evo_steps=None,
        mutation=None,
        tournament=None,
        accelerator=accelerator,
        verbose=True,
    )
    accelerator.end_training()


if __name__ == "__main__":
    with open("configs/training/llm_finetuning/ppo_llm.yaml") as file:
        config = yaml.safe_load(file)
        print(config)
    init_hp = config["INIT_HP"]
    mut_p = config["MUTATION_PARAMS"]
    main(init_hp, mut_p)

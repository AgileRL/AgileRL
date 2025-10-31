import re

from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from agilerl.algorithms import GRPO
from agilerl.training.train_llm import finetune_llm_reasoning
from agilerl.utils.algo_utils import VLLMConfig
from agilerl.utils.llm_utils import ReasoningGym

MODEL_PATH = "Qwen/Qwen2.5-0.5B"
DATASET = "Jiayi-Pan/Countdown-Tasks-3to4"
USE_VLLM = True
MAX_CONTEXT_LENGTH = 1024


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

    for completion, gt in zip(completions, target):

        try:
            # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
            completion = "<think>" + completion
            regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"
            match = re.search(regex, completion, re.DOTALL)
            if match is None or len(match.groups()) != 2:
                rewards.append(0.0)
            else:
                rewards.append(1.0)
        except Exception:
            rewards.append(0.0)
    return rewards


def equation_reward_func(completions, target, nums, **kwargs):
    rewards = []

    for completion, gt, numbers in zip(completions, target, nums):
        try:
            # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
            completion = "<think>" + completion
            answer_tags = re.findall(r"<answer>([\s\S]*?)<\/answer>", completion)

            if len(answer_tags) != 1:
                rewards.append(0.0)
                continue

            equation = answer_tags[0].strip()
            used_numbers = [int(n) for n in re.findall(r"\d+", equation)]

            if sorted(used_numbers) != sorted(numbers.flatten().tolist()):
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
    reward = (
        equation_reward_func([completion], [solution], [prompt])[0]
        + format_reward_func([completion], [solution])[0]
    )

    if reward == 2.0:
        with open("countdown_completions.txt", "a") as text_file:
            text_file.write(
                f"Prompt {prompt}" + "\n" + completion + "\n" + "=" * 50 + "\n"
            )

    return reward


def main():
    # Instantiate the model and the associated tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    train_dataset, test_dataset = make_dataset(DATASET)

    # Convert the HuggingFace dataset into a Gymnasium environment
    accelerator = Accelerator()

    # Define the conversation template
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
    env = ReasoningGym(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        reward_fn=combined_rewards,
        conversation_template=conversation_template,
        data_batch_size_per_gpu=10,
        accelerator=accelerator,
        return_raw_completions=USE_VLLM,  # This is necessary for vLLM to work
        max_context_length=MAX_CONTEXT_LENGTH,
    )

    # Define the LoRA configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=64,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "down_proj",
            "gate_proj",
        ],
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
    )

    # Instantiate the grpo agent
    agent = GRPO(
        model_name=MODEL_PATH,
        pad_token_id=tokenizer.eos_token_id,
        pad_token=tokenizer.eos_token,
        lora_config=lora_config,
        batch_size=16,
        max_model_len=MAX_CONTEXT_LENGTH,
        group_size=8,
        accelerator=accelerator,
        use_vllm=USE_VLLM,
        vllm_config=VLLMConfig(sleep_mode=True, max_num_seqs=4),
    )
    finetune_llm_reasoning(
        pop=[agent],
        env=env,
        evaluation_interval=10,
        wb=True,
        save_elite=True,
        elite_path="checkpoints",
        max_reward=2.0,
        accelerator=accelerator,
        num_epochs=1,
    )


if __name__ == "__main__":
    main()

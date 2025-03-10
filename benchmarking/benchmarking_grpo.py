import re
from typing import Tuple

import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from agilerl.algorithms import GRPO
from agilerl.training.train_llm import finetune_llm
from agilerl.utils.llm_utils import HuggingFaceGym

MODEL_PATH = "Qwen/Qwen2.5-3B"
DATASET = "Jiayi-Pan/Countdown-Tasks-3to4"


def create_model(pretrained_model_name_or_path):
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    peft_config = LoraConfig(
        r=32,
        lora_alpha=32,
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
    model = get_peft_model(model, peft_config)
    return model


def countdown_chat_template(q, a, tokenizer):
    conversation = [
        {
            "role": "system",
            "content": "You are a helpful assistant. You first think about the reasoning process in your mind and then provide the user with the answer.",
        },
        {
            "role": "user",
            "content": f"Using each number in this tensor only once {tuple(i.item() for i in q)}, create an equation that equals {a.item()}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final equation and answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 = 1 </answer>.",
        },
        {"role": "assistant", "content": "Let me solve this step by step.\n<think>"},
    ]
    updated_prompt = tokenizer.apply_chat_template(
        conversation, tokenize=False, continue_final_message=True
    )
    tokenized_prompt = tokenizer(
        [updated_prompt],
        return_tensors="pt",
        padding=True,
        padding_side="left",
        return_attention_mask=True,
    )
    return tokenized_prompt


def make_dataset(dataset_name: str) -> Tuple[Dataset, Dataset]:
    raw_dataset = (
        load_dataset(DATASET, split="train").shuffle(seed=42).select(range(50000))
    )
    raw_dataset = raw_dataset.rename_column("target", "answer")
    raw_dataset = raw_dataset.rename_column("nums", "question")
    train_test_split = raw_dataset.train_test_split(test_size=0.1)
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]
    return train_dataset, test_dataset


def format_reward_func(completions, target, **kwargs):
    rewards = []

    for completion, gt in zip(completions, target):

        try:
            # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
            completion = "<think>" + completion
            # Check if the format is correct
            regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"

            match = re.search(regex, completion, re.DOTALL)
            # if the format is not correct, reward is 0
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
            # Check if the format is correct
            match = re.search(r"<answer>(.*?)<\/answer>", completion)
            if match is None:
                rewards.append(0.0)
                continue
            # Extract the "answer" part from the completion
            equation = match.group(1).strip()
            # Extract all numbers from the equation
            used_numbers = [int(n) for n in re.findall(r"\d+", equation)]

            # Check if all numbers are used exactly once
            if sorted(used_numbers) != sorted(numbers):
                rewards.append(0.0)
                continue
            # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
            allowed_pattern = r"^[\d+\-*/().\s]+$"
            if not re.match(allowed_pattern, equation):
                rewards.append(0.0)
                continue

            # Evaluate the equation with restricted globals and locals
            result = eval(equation, {"__builtins__": None}, {})
            # Check if the equation is correct and matches the ground truth
            if abs(float(result) - float(gt)) < 1e-5:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        except Exception:
            # If evaluation fails, reward is 0
            rewards.append(0.0)
    return rewards


def combined_rewards(completion, solution, prompt):
    reward = (
        equation_reward_func([completion], [solution], [prompt])[0]
        + format_reward_func([completion], [solution])[0]
    )

    print(
        f"""
    ============================================, \n
    Completion: {completion}, \n
    Numbers: {prompt}, \n
    Correct Answer: {solution.item()} \n
    Reward: {reward}
    """
    )

    return reward


def custom_collate_fn(batch):
    # Extract answers and questions
    answers = torch.tensor([item["answer"] for item in batch])

    # For questions of variable length, we need to pad them
    # First, find the maximum length
    max_len = max(len(item["question"]) for item in batch)

    # Create padded tensor
    questions = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, item in enumerate(batch):
        q_len = len(item["question"])
        questions[i, :q_len] = torch.tensor(item["question"])

    return {"answer": answers, "question": questions}


def main():
    # Instantiate the model and the associated tokenizer
    model = create_model(**{"pretrained_model_name_or_path": MODEL_PATH})
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    train_dataset, test_dataset = make_dataset(DATASET)
    # Convert the HuggingFace dataset into a Gymnasium environment
    env = HuggingFaceGym(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        reward_fn=combined_rewards,
        apply_chat_template_fn=countdown_chat_template,
        max_answer_tokens=1024,
        data_batch_size=1,
        custom_collate_fn=custom_collate_fn,
    )
    # Instantiate the grpo agent
    agent = GRPO(
        env.observation_space,
        env.action_space,
        actor_network=model,
        pad_token_id=tokenizer.eos_token_id,
        batch_size=1,
        group_size=12,
        reduce_memory_peak=False,
        accelerator=Accelerator(),
    )
    finetune_llm(
        agent=agent,
        env=env,
        evaluation_interval=5,
        wb=False,
        checkpoint_interval=1,
        checkpoint_path="saved_llms",
    )


if __name__ == "__main__":
    main()

import re

import gymnasium as gym
from transformers import AutoTokenizer

REASONING_SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def format_reward(completions):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


def make_conversation(example):
    return {
        "prompt": [
            {"role": "system", "content": REASONING_SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]},
        ],
    }


def tokenize_function(examples, tokenizer):
    results = {}

    # Handle the prompt field (which is a list of dictionaries)
    formatted_prompts = []
    for prompt_list in examples["prompt"]:
        formatted_prompt = ""
        for message in prompt_list:
            if message["role"] == "system":
                formatted_prompt += (
                    f"<|im_start|>system\n{message['content']}<|im_end|>\n"
                )
            elif message["role"] == "user":
                formatted_prompt += (
                    f"<|im_start|>user\n{message['content']}<|im_end|>\n"
                )
        formatted_prompts.append(formatted_prompt)

    # Tokenize the formatted prompts
    prompt_tokenized = tokenizer(
        formatted_prompts,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    # Tokenize the answers separately
    answer_tokenized = tokenizer(
        examples["answer"],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    # Add tokenized data to results
    results["input_ids"] = prompt_tokenized["input_ids"]
    results["attention_mask"] = prompt_tokenized["attention_mask"]
    results["labels"] = answer_tokenized["input_ids"]

    return results


def prepare_dataset(): ...


class HuggingFaceGym(gym.Env):
    def __init__(
        self,
        dataset_name: str,
        tokenizer: AutoTokenizer,
    ):
        # raw_dataset = load_dataset(dataset_name, "main")
        # train_dataset = raw_dataset["train"].map(make_conversation)
        # test_dataset = raw_dataset["test"].map(make_conversation)
        # tokenized_dataset.set_format("torch")
        # data_collator = DataCollatorWithPadding(tokenizer)
        # self.train_dataloader = DataLoader(
        #     tokenized_dataset["train"], batch_size=8, collate_fn=data_collator
        # )
        # self.test_dataloader = DataLoader(
        #     tokenized_dataset["test"], batch_size=8, collate_fn=data_collator
        # )
        # self.train = True
        pass

    def step(self):
        pass

    def reset(self):
        pass

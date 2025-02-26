import re
from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, List, Tuple

import gymnasium as gym
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding

REASONING_SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def format_reward(completion: str) -> float:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    pattern_match = re.match(pattern, completion)
    return 1.0 if pattern_match else -1.0


def accuracy_reward(completion: str, solution: str) -> float:
    """Reward function that checks if the completion is the same as the ground truth."""

    # Obtain numerical answer
    pattern = re.compile(r"#### (\-?[0-9\.\,]+)")
    correct_answer = pattern.search(solution)
    correct_answer = correct_answer.group(1).strip()

    # Obtain our models answer
    pattern = r"\d+\.\d+|\d+/\d+|\d+"
    nums = re.findall(pattern, completion)
    if len(nums) == 0:
        return -1.0
    answer = nums[-1]
    return 3 if (answer == correct_answer) else -3


def reward_function(completion: str, solution: str) -> float:
    return accuracy_reward(completion, solution) + format_reward(completion)


class HuggingFaceGym(gym.Env):
    def __init__(
        self,
        dataset_name: str,
        tokenizer: AutoTokenizer,
        reward_fn: Callable[..., float],
        system_prompt: str = REASONING_SYSTEM_PROMPT,
        max_answer_tokens: int = 512,
        batch_size: int = 8,
    ) -> None:
        self.name = dataset_name
        self.reward_fn = reward_fn
        self.system_prompt = system_prompt
        self.tokenizer = tokenizer
        raw_dataset = load_dataset(dataset_name, "main")
        self.train_dataset = raw_dataset["train"]
        self.test_dataset = raw_dataset["test"]
        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True
        )
        self.test_dataloader = DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False
        )
        self.dataloader = iter(self.train_dataloader)
        self.reset_called = False

        self.observation_space = gym.spaces.Box(low=0, high=tokenizer.vocab_size - 1)
        self.action_space = gym.spaces.Box(
            low=0, high=tokenizer.vocab_size - 1, shape=(max_answer_tokens,)
        )
        self.eval_mode = False

    def step(
        self, completions: torch.Tensor
    ) -> Tuple[List[BatchEncoding], torch.Tensor]:
        self.reset_called = False
        rewards = self._decode_and_evaluate(completions)
        new_tokenized_prompts = self._get_next_batch()
        self.last_tokenized_prompts = new_tokenized_prompts
        return new_tokenized_prompts, rewards

    def reset(self) -> Tuple[List[BatchEncoding], Dict[str, Any]]:
        if self.reset_called:
            raise RuntimeError(
                "env.reset() cannot be called more than once sequentially, it must follow with env.step()."
            )
        self.reset_called = True
        self.info = {}
        new_tokenized_prompts = self._get_next_batch()
        self.last_tokenized_prompts = new_tokenized_prompts
        return new_tokenized_prompts, self.info

    def _decode_and_evaluate(self, completions: List[torch.Tensor]) -> torch.Tensor:
        # This is for a batch of completions (prompt_batch x group_size), List of tensors of length batch size, each tensor is a group of answers
        total_rewards = []
        if self.eval_mode:
            decoded_completions = []
        for idx, (group_completion, answer) in enumerate(
            zip(completions, self.answers)
        ):  # Vectorize this in the future
            decoded_group_completion = self.tokenizer.batch_decode(
                group_completion[
                    :, self.last_tokenized_prompts[idx]["input_ids"].shape[1] :
                ],
                skip_special_tokens=True,
            )
            if self.eval_mode:
                decoded_completions.append(decoded_group_completion)
            rewards = [
                self.reward_fn(completion, answer)
                for completion in decoded_group_completion
            ]
            total_rewards.append(rewards)
        # Shape of the returned tensor is (batch_size X group_size)
        if self.eval_mode:
            for idx, answer in enumerate(decoded_completions):
                print(f"Question: {self.questions[idx]}")
                print(f"Answer: {answer}")
                print(f"Correct answer: {self.answers[idx]}")
                print(f"Rewards: {total_rewards[idx]}")
                print("\n")
        return torch.tensor(total_rewards)

    def _get_next_batch(self) -> List[BatchEncoding]:
        batch = next(iter(self.dataloader))
        self.questions = batch["question"]
        self.answers = batch["answer"]
        tokenized_prompts = [
            apply_chat_template(question, self.system_prompt, self.tokenizer)
            for question in self.questions
        ]
        return tokenized_prompts

    @contextmanager
    def eval(self) -> Generator[None, None, None]:
        self.dataloader = self.test_dataloader
        self.eval_mode = True
        yield
        self.dataloader = self.train_dataloader
        self.eval_mode = False

    def __len__(self):
        return len(self.train_dataloader)


def apply_chat_template(
    question: str, system_prompt: str, tokenizer: AutoTokenizer
) -> BatchEncoding:
    conversation = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": question,
        },
    ]
    updated_prompt = tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    tokenized_prompt = tokenizer(
        [updated_prompt],
        return_tensors="pt",
        padding=True,
        padding_side="left",
        return_attention_mask=True,
    )
    return tokenized_prompt

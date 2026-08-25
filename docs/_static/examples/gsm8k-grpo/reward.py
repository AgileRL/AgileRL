# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

import re

from openenv.core.rubrics.base import Rubric

from agilerl.llm_envs.rubrics import reward_fn_to_rubric


def extract_gold_answer(answer: str) -> str | None:
    """Return the final numeric answer from a GSM8K answer (the value after ``####``)."""
    match = re.search(r"####\s*(-?[\d,]+)", answer)
    return match.group(1).replace(",", "").strip() if match else None


def extract_model_answer(completion: str) -> str | None:
    """Return the final number inside the model's ``<answer>...</answer>`` block."""
    block = re.search(r"<answer>([\s\S]*?)</answer>", completion)
    if block is None:
        return None
    number = re.search(r"-?[\d,]+", block.group(1))
    return number.group(0).replace(",", "").strip() if number else None


def correctness_reward(answer: str, completion: str) -> float:
    gold = extract_gold_answer(answer)
    predicted = extract_model_answer(completion)
    if gold is None or predicted is None:
        return 0.0
    return 1.0 if predicted == gold else 0.0


def format_reward(completion: str) -> float:
    text = "<think>" + completion
    regex = r"^<think>[\s\S]*?</think>\n<answer>[\s\S]*?</answer>$"
    return 1.0 if re.match(regex, text.strip()) else 0.0


def reward(question: str, answer: str, completion: str) -> float:
    """Combined GSM8K reward: answer correctness plus output format."""
    del question
    return correctness_reward(answer, completion) + format_reward(completion)


def _combined(completion: str, answer, question) -> float:
    del question
    return correctness_reward(answer, completion) + format_reward(completion)


RUBRIC = reward_fn_to_rubric(_combined, name="gsm8k")


class Correctness(Rubric):
    """Example leaf rubric for component-level reporting."""

    def forward(self, action, observation) -> float:
        return correctness_reward(str(observation.answer), action.message)


class Format(Rubric):
    """Example leaf rubric for component-level reporting."""

    def forward(self, action, observation) -> float:
        del observation
        return format_reward(action.message)


class Combined(Rubric):
    """Example composition: OpenEnv auto-registers ``correct`` / ``fmt`` children."""

    def __init__(self) -> None:
        super().__init__()
        self.correct = Correctness()
        self.fmt = Format()

    def forward(self, action, observation) -> float:
        return self.correct(action, observation) + self.fmt(action, observation)

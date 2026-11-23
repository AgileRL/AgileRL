# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""GSM8K reward: the right final number, in the ``<answer>`` tags the prompt asks for."""

import re

ANSWER_BLOCK = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def final_text(completion: str) -> str:
    """Return the text after the last ``</think>``, or the whole completion when there is none."""
    return completion.rpartition("</think>")[2].strip()


def to_number(text: str) -> float | None:
    """Return the first number in ``text``, ignoring thousands separators."""
    match = NUMBER.search(text)
    return float(match.group(0).replace(",", "")) if match else None


def gold_answer(answer: str) -> float | None:
    """Return a GSM8K row's final answer: the number after ``####``."""
    _, marker, final = answer.rpartition("####")
    return to_number(final) if marker else None


def correctness_reward(text: str, answer: str) -> float:
    """Return 1.0 when the last ``<answer>`` block holds the gold number."""
    blocks = ANSWER_BLOCK.findall(text)
    if not blocks:
        return 0.0
    predicted = to_number(blocks[-1])
    return 1.0 if predicted is not None and predicted == gold_answer(answer) else 0.0


def format_reward(text: str) -> float:
    """Return 1.0 when the text ends with one ``<answer>`` block holding only a number."""
    blocks = ANSWER_BLOCK.findall(text)
    if len(blocks) != 1 or not text.endswith("</answer>"):
        return 0.0
    return 1.0 if NUMBER.fullmatch(blocks[0].strip()) else 0.0


def reward_fn(completion: str, answer: str, question: str) -> float:
    """Score one completion: correctness plus format, at most 2.0.

    Both parts read only the text after the reasoning, so a number the model
    tries while thinking does not count.

    :param completion: The model's turn, reasoning included when it has any.
    :param answer: The row's GSM8K answer, ending ``#### <number>``.
    :param question: The row's question.
    :return: The reward.
    """
    text = final_text(completion)
    return correctness_reward(text, answer) + format_reward(text)

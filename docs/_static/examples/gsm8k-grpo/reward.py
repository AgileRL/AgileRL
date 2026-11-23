# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""GSM8K reward: the right final number, in the ``<answer>`` tags the prompt asks for."""

import re

ANSWER_BLOCK = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def to_number(text: str) -> float | None:
    """Return the first number in ``text``, ignoring thousands separators."""
    match = NUMBER.search(text)
    return float(match.group(0).replace(",", "")) if match else None


def gold_answer(answer: str) -> float | None:
    """Return a GSM8K row's final answer: the number after ``####``."""
    _, marker, final = answer.rpartition("####")
    return to_number(final) if marker else None


def correctness_reward(response: str, answer: str) -> float:
    """Return 1.0 when the last ``<answer>`` block holds the gold number."""
    blocks = ANSWER_BLOCK.findall(response)
    if not blocks:
        return 0.0
    predicted = to_number(blocks[-1])
    return 1.0 if predicted is not None and predicted == gold_answer(answer) else 0.0


def format_reward(response: str) -> float:
    """Return 1.0 when the response ends with one ``<answer>`` block holding only a number."""
    blocks = ANSWER_BLOCK.findall(response)
    if len(blocks) != 1 or not response.endswith("</answer>"):
        return 0.0
    return 1.0 if NUMBER.fullmatch(blocks[0].strip()) else 0.0


def reward_fn(
    completion: str, answer: str, question: str, response: str | None = None
) -> float:
    """Score one completion: correctness plus format, at most 2.0.

    :param completion: The whole turn, reasoning included.
    :param answer: The row's GSM8K answer, ending ``#### <number>``.
    :param question: The row's question.
    :param response: The text after the reasoning ends.
    :return: The reward.
    """
    if response is None:
        # Arena releases that do not pass ``response`` pass the whole turn.
        response = completion.rpartition("</think>")[2].strip()
    return correctness_reward(response, answer) + format_reward(response)

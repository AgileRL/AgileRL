"""Reward functions for the Countdown arithmetic reasoning task.

Used with the Jiayi-Pan/Countdown-Tasks-3to4 dataset. The model must produce
an arithmetic equation inside <answer> tags that evaluates to the target number,
using each provided number exactly once.
"""

from __future__ import annotations

import re

EXPECTED_GROUPS = 2
ANSWER_TAG_COUNT = 1
EQUATION_TOLERANCE = 1e-5


def format_reward_func(completions, target, **kwargs):
    """Reward correct use of <think>...</think> and <answer>...</answer> tags."""
    rewards = []
    regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"
    for completion, _gt in zip(completions, target, strict=False):
        candidate = "<think>" + completion
        match = re.search(regex, candidate, re.DOTALL)
        if match is None or len(match.groups()) != EXPECTED_GROUPS:
            rewards.append(0.0)
        else:
            rewards.append(1.0)
    return rewards


def equation_reward_func(completions, target, nums, **kwargs):
    """Reward correct arithmetic equations that use all numbers and hit the target."""
    rewards = []
    for completion, gt, numbers in zip(completions, target, nums, strict=False):
        try:
            candidate = "<think>" + completion
            answer_tags = re.findall(r"<answer>([\s\S]*?)<\/answer>", candidate)

            if len(answer_tags) != ANSWER_TAG_COUNT:
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

            result = eval(equation, {"__builtins__": None}, {})  # noqa: S307

            if abs(float(result) - float(gt)) < EQUATION_TOLERANCE:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        except (NameError, SyntaxError, TypeError, ValueError, ZeroDivisionError):
            rewards.append(0.0)
    return rewards


def combined_rewards(completion, solution, prompt):
    """Combine format and equation rewards (max 2.0)."""
    return (
        equation_reward_func([completion], [solution], [prompt])[0]
        + format_reward_func([completion], [solution])[0]
    )

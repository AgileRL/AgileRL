"""Reward functions for the Countdown arithmetic reasoning task.

Used with the Jiayi-Pan/Countdown-Tasks-3to4 dataset. The model must produce
an arithmetic equation inside <answer> tags that evaluates to the target number,
using each provided number exactly once.
"""

from __future__ import annotations

import re


def format_reward_func(completions, target, **kwargs):
    """Reward correct use of <think>...</think> and <answer>...</answer> tags."""
    rewards = []
    for completion, gt in zip(completions, target):
        try:
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
    """Reward correct arithmetic equations that use all numbers and hit the target."""
    rewards = []
    for completion, gt, numbers in zip(completions, target, nums):
        try:
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

            result = eval(equation, {"__builtins__": None}, {})  # noqa: S307

            if abs(float(result) - float(gt)) < 1e-5:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        except Exception:
            rewards.append(0.0)
    return rewards


def combined_rewards(completion, solution, prompt):
    """Combined format + equation reward (max 2.0)."""
    reward = (
        equation_reward_func([completion], [solution], [prompt])[0]
        + format_reward_func([completion], [solution])[0]
    )
    return reward

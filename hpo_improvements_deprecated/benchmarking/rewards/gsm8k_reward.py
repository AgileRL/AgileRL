"""Reward functions for the GSM8K grade-school math reasoning task.

Used with the openai/gsm8k dataset. The model must produce step-by-step
reasoning inside <think> tags and a final numerical answer inside <answer> tags.
"""

from __future__ import annotations

import re

EXPECTED_GROUPS = 2
ANSWER_TAG_COUNT = 1
NUMERIC_TOLERANCE = 1e-5


def _extract_gsm8k_answer(text: str) -> str | None:
    """Extract the final number after '####' from a GSM8K ground-truth answer."""
    match = re.search(r"####\s*(.+)", text)
    if match:
        return match.group(1).strip().replace(",", "")
    return None


def _normalize_number(s: str) -> float | None:
    """Parse a string into a float, stripping common formatting."""
    s = s.strip().replace(",", "").replace("$", "").replace("%", "")
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def format_reward_func(completions, target, **kwargs):
    """Reward correct use of <think>...</think> and <answer>...</answer> tags."""
    rewards = []
    regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"
    for completion in completions:
        candidate = "<think>" + completion
        match = re.search(regex, candidate, re.DOTALL)
        if match is None or len(match.groups()) != EXPECTED_GROUPS:
            rewards.append(0.0)
        else:
            rewards.append(1.0)
    return rewards


def answer_reward_func(completions, target, **kwargs):
    """Reward correct numerical answers matching the GSM8K ground truth."""
    rewards = []
    for completion, gt in zip(completions, target, strict=False):
        try:
            candidate = "<think>" + completion
            answer_tags = re.findall(r"<answer>([\s\S]*?)<\/answer>", candidate)

            if len(answer_tags) != ANSWER_TAG_COUNT:
                rewards.append(0.0)
                continue

            predicted = _normalize_number(answer_tags[0])
            expected_str = _extract_gsm8k_answer(gt) if "####" in str(gt) else str(gt)
            expected = _normalize_number(expected_str)

            if (
                predicted is not None
                and expected is not None
                and abs(predicted - expected) < NUMERIC_TOLERANCE
            ):
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        except (AttributeError, TypeError, ValueError):
            rewards.append(0.0)
    return rewards


def combined_rewards(completion, solution, prompt):
    """Combine format and answer rewards (max 2.0)."""
    return (
        answer_reward_func([completion], [solution])[0]
        + format_reward_func([completion], [solution])[0]
    )

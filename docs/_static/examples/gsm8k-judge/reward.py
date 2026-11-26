# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

import re

from openenv.core.rubrics.base import Rubric
from openenv.core.rubrics.containers import WeightedSum

from agilerl.llm_envs.rubrics import LLMGroupJudgeRubric, LLMJudgeRubric

CRITERIA = """\
The reply solves the math word problem. Each step of the working follows from
the problem or the step before it, and a student could follow it. The reply
ends with one final number inside <answer></answer> tags, and that number
matches the reference answer. Penalise wrong arithmetic, skipped steps,
repetition and padding."""


def extract_gold_answer(answer: str) -> str | None:
    """Return the final numeric answer from a GSM8K answer (the value after ``####``)."""
    match = re.search(r"####\s*(-?[\d,]+)", answer)
    return match.group(1).replace(",", "").strip() if match else None


def extract_model_answer(completion: str) -> str | None:
    """Return the final number inside the model's last ``<answer>...</answer>`` block."""
    blocks = re.findall(r"<answer>([\s\S]*?)</answer>", completion)
    if not blocks:
        return None
    number = re.search(r"-?[\d,]+", blocks[-1])
    return number.group(0).replace(",", "").strip() if number else None


class Correctness(Rubric):
    """1.0 when the number in ``<answer>`` matches the GSM8K gold answer."""

    component_name = "correct"

    def forward(self, action, observation) -> float:
        gold = extract_gold_answer(str(observation.answer))
        predicted = extract_model_answer(action.message)
        if gold is None or predicted is None:
            return 0.0
        return 1.0 if predicted == gold else 0.0


class RUBRIC(WeightedSum):
    """Correct final answer, plus a judge's score for the working."""

    def __init__(self, judge) -> None:
        super().__init__(
            [Correctness(), LLMJudgeRubric(judge, CRITERIA)],
            weights=[0.6, 0.4],
        )


class RULER(LLMGroupJudgeRubric):
    """One judge call ranks every completion in a GRPO group against the criteria."""

    def __init__(self, judge) -> None:
        super().__init__(judge, CRITERIA)

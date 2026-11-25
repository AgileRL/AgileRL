# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Rubrics for the built-in rewards a manifest picks in ``environment.reward``.

Each scores the turn's final answer: the assistant's text after any reasoning,
as the rubric transcript holds it.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Sequence
from typing import Any

from jsonschema.validators import validator_for
from openenv.core.rubrics.base import Rubric

from agilerl.arena.models.rewards import (
    BuiltinReward,
    ExactMatchReward,
    JsonSchemaReward,
    NumericReward,
    RegexReward,
    ToolCallReward,
)
from agilerl.llm_envs.rubrics import TurnAction, TurnObservation, first_json_object

NUMBER = re.compile(r"-?(?:\d[\d,]*(?:\.\d+)?|\.\d+)")
BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
TOOL_CALL_BLOCK = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

Scorer = Callable[[str, object], float]


def final_answer(transcript: Sequence[dict[str, str]]) -> str:
    """The last assistant message's text after its reasoning.

    :param transcript: Rubric transcript messages.
    :return: That message's ``content``.
    """
    return [m for m in transcript if m["role"] == "assistant"][-1]["content"]


def extract_answer(pattern: str | None, text: str) -> str | None:
    """The first group of ``pattern``'s last match in ``text``, or all of ``text``.

    :param pattern: Regular expression with a capturing group, or ``None``.
    :param text: The reply.
    :return: The answer, or ``None`` when the pattern does not match.
    """
    if pattern is None:
        return text
    matches = list(re.finditer(pattern, text, re.DOTALL))
    return matches[-1].group(1) if matches else None


def normalized(text: str) -> str:
    """Casefold, collapse whitespace, and drop a trailing full stop."""
    return " ".join(text.split()).casefold().removesuffix(".")


def last_number(text: str) -> float | None:
    """The last number in ``text``, thousands separators ignored, or ``None``."""
    numbers = NUMBER.findall(text)
    return float(numbers[-1].replace(",", "")) if numbers else None


def reference_number(answer: object) -> float | None:
    """The row's ``answer`` as a number, or ``None`` when it holds none."""
    if isinstance(answer, (int, float)) and not isinstance(answer, bool):
        return float(answer)
    return None if answer is None else last_number(str(answer))


def exact_match_scorer(reward: ExactMatchReward) -> Scorer:
    """Score 1 when the answer equals the row's ``answer``."""

    def score(text: str, answer: object) -> float:
        found = extract_answer(reward.answer_pattern, text)
        if found is None or answer is None:
            return 0.0
        expected = str(answer)
        if reward.normalize:
            return float(normalized(found) == normalized(expected))
        return float(found == expected)

    return score


def numeric_scorer(reward: NumericReward) -> Scorer:
    """Score 1 when the answer's number is within ``tolerance`` of the row's."""

    def score(text: str, answer: object) -> float:
        found = extract_answer(reward.answer_pattern, text)
        if found is not None and reward.answer_pattern is None:
            boxed = BOXED.findall(found)
            found = boxed[-1] if boxed else found
        number = None if found is None else last_number(found)
        expected = reference_number(answer)
        if number is None or expected is None:
            return 0.0
        return float(
            math.isclose(number, expected, rel_tol=0.0, abs_tol=reward.tolerance)
        )

    return score


def regex_scorer(reward: RegexReward) -> Scorer:
    """Score 1 when the reply matches the pattern."""
    pattern = re.compile(reward.pattern, re.DOTALL)
    match = pattern.fullmatch if reward.full_match else pattern.search
    return lambda text, _answer: float(match(text) is not None)


def json_schema_scorer(reward: JsonSchemaReward) -> Scorer:
    """Score 1 when the reply's first JSON object is valid against the schema.

    :raises jsonschema.SchemaError: The schema itself is invalid.
    """
    validator_class = validator_for(reward.json_schema)
    validator_class.check_schema(reward.json_schema)
    validator = validator_class(reward.json_schema)

    def score(text: str, _answer: object) -> float:
        found = first_json_object(text)
        return float(found is not None and validator.is_valid(found))

    return score


def parsed_tool_call(text: str) -> tuple[str, dict[str, Any]] | None:
    """The first ``{"name": ..., "arguments": {...}}`` object in ``text``.

    A ``<tool_call>`` block is read first when the text has one.

    :param text: The reply, or the row's answer.
    :return: The name and arguments, or ``None`` when there is no such object.
    """
    block = TOOL_CALL_BLOCK.search(text)
    found = first_json_object(block.group(1) if block else text)
    if found is None:
        return None
    name, arguments = found.get("name"), found.get("arguments")
    if not isinstance(name, str) or not isinstance(arguments, dict):
        return None
    return name, arguments


def tool_call_scorer(reward: ToolCallReward) -> Scorer:
    """Score 1 when the reply's first tool call is the expected one."""

    def score(text: str, answer: object) -> float:
        call = parsed_tool_call(text)
        if call is None:
            return 0.0
        name, arguments = call
        if reward.name is not None:
            return float(
                name == reward.name
                and all(key in arguments for key in reward.required_arguments)
            )
        expected = answer if isinstance(answer, str) else json.dumps(answer)
        return float(call == parsed_tool_call(expected))

    return score


SCORERS: dict[type[BuiltinReward], Callable[[Any], Scorer]] = {
    ExactMatchReward: exact_match_scorer,
    NumericReward: numeric_scorer,
    RegexReward: regex_scorer,
    JsonSchemaReward: json_schema_scorer,
    ToolCallReward: tool_call_scorer,
}


class BuiltinRewardRubric(Rubric):
    """Scores every turn with one built-in reward, reported under its ``type``.

    :param reward: The manifest's ``environment.reward``.
    """

    def __init__(self, reward: BuiltinReward) -> None:
        super().__init__()
        self._score = SCORERS[type(reward)](reward)
        self.component_name = reward.type

    def forward(self, action: TurnAction, observation: TurnObservation) -> float:
        return self._score(final_answer(observation.transcript), observation.answer)

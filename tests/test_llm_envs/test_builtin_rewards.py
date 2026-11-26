# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Built-in rewards score a turn's final answer against the row's answer."""

from __future__ import annotations

from typing import Any, ClassVar

import pytest
from jsonschema.exceptions import SchemaError
from pydantic import TypeAdapter

from agilerl.arena.models.rewards import RewardSpec
from agilerl.llm_envs.builtin_rewards import BuiltinRewardRubric, final_answer
from agilerl.llm_envs.rubrics import TurnAction, TurnObservation

REWARD = TypeAdapter(RewardSpec)


def score(reward: dict[str, Any], reply: str, answer: Any = None) -> float:
    """Score one turn whose final answer is ``reply``.

    The turn's whole text carries reasoning, so a reward that read it instead of
    the final answer would fail these cases.
    """
    rubric = BuiltinRewardRubric(REWARD.validate_python(reward))
    observation = TurnObservation(
        env_reward=0.0,
        env_ran=True,
        done=True,
        turn_index=0,
        prompt="question",
        answer=answer,
        transcript=[
            {"role": "user", "content": "question"},
            {
                "role": "assistant",
                "content": reply,
                "completion": f"<think>99</think>{reply}",
            },
        ],
    )
    return rubric(
        TurnAction(message=f"<think>99</think>{reply}", completion=reply), observation
    )


class TestFinalAnswer:
    def test_reads_the_last_assistant_message(self) -> None:
        transcript = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "first"},
            {"role": "user", "content": "again"},
            {"role": "assistant", "content": "second"},
            {"role": "user", "content": "env reply"},
        ]

        assert final_answer(transcript) == "second"


class TestExactMatch:
    @pytest.mark.parametrize(
        ("reply", "answer", "expected"),
        [
            ("Paris", "Paris", 1.0),
            ("  paris. ", "Paris", 1.0),
            ("New   York", "new york", 1.0),
            ("Lyon", "Paris", 0.0),
            ("Paris", None, 0.0),
            ("42", 42, 1.0),
        ],
    )
    def test_normalized(self, reply: str, answer: Any, expected: float) -> None:
        assert score({"type": "exact_match"}, reply, answer) == expected

    def test_without_normalize_case_counts(self) -> None:
        reward = {"type": "exact_match", "normalize": False}

        assert score(reward, "paris", "Paris") == 0.0
        assert score(reward, "Paris", "Paris") == 1.0

    def test_answer_pattern_reads_the_last_match(self) -> None:
        reward = {"type": "exact_match", "answer_pattern": r"Answer: (\w+)"}

        assert score(reward, "Answer: Lyon\nNo wait. Answer: Paris", "Paris") == 1.0
        assert score(reward, "Paris", "Paris") == 0.0


class TestNumeric:
    @pytest.mark.parametrize(
        ("reply", "answer", "expected"),
        [
            ("So the total is 1,234.", "1234", 1.0),
            ("We get 3 apples, then 7.", 7, 1.0),
            ("It is -2.5", "-2.5", 1.0),
            ("Ignore 9, it is \\boxed{12} and 5 more", "12", 1.0),
            ("It is .5", 0.5, 1.0),
            ("It is 8", "#### 8", 1.0),
            ("It is 8", "9", 0.0),
            ("No number here", "9", 0.0),
            ("It is 8", "eight", 0.0),
            ("It is 8", None, 0.0),
            ("It is 1", True, 0.0),
        ],
    )
    def test_reads_the_boxed_or_last_number(
        self, reply: str, answer: Any, expected: float
    ) -> None:
        assert score({"type": "numeric"}, reply, answer) == expected

    def test_tolerance_is_absolute_and_inclusive(self) -> None:
        reward = {"type": "numeric", "tolerance": 0.5}

        assert score(reward, "3.5", "3") == 1.0
        assert score(reward, "3.6", "3") == 0.0

    def test_answer_pattern_replaces_the_boxed_rule(self) -> None:
        reward = {"type": "numeric", "answer_pattern": r"Answer: (.*)"}

        assert score(reward, "\\boxed{1} Answer: 2", "2") == 1.0
        assert score(reward, "\\boxed{2}", "2") == 0.0


class TestRegex:
    def test_search_finds_the_pattern_anywhere(self) -> None:
        reward = {"type": "regex", "pattern": r"\d{3}-\d{4}"}

        assert score(reward, "Call 555-1234 today") == 1.0
        assert score(reward, "Call me") == 0.0

    def test_full_match_needs_the_whole_reply(self) -> None:
        reward = {"type": "regex", "pattern": r"(yes|no)", "full_match": True}

        assert score(reward, "yes") == 1.0
        assert score(reward, "yes, I think") == 0.0

    def test_spans_lines(self) -> None:
        reward = {"type": "regex", "pattern": r"start.*end", "full_match": True}

        assert score(reward, "start\nmiddle\nend") == 1.0


class TestJsonSchema:
    REWARD: ClassVar[dict[str, Any]] = {
        "type": "json_schema",
        "schema": {
            "type": "object",
            "required": ["city", "population"],
            "properties": {"population": {"type": "integer"}},
        },
    }

    @pytest.mark.parametrize(
        ("reply", "expected"),
        [
            ('{"city": "Paris", "population": 2100000}', 1.0),
            ('Here you go:\n{"city": "Paris", "population": 2100000} done', 1.0),
            ('{"city": "Paris"}', 0.0),
            ('{"city": "Paris", "population": "many"}', 0.0),
            ('{"city": "Paris", "population": ', 0.0),
            ("no json", 0.0),
        ],
    )
    def test_first_object_must_satisfy_the_schema(
        self, reply: str, expected: float
    ) -> None:
        assert score(self.REWARD, reply) == expected

    def test_an_invalid_schema_fails_when_the_rubric_is_built(self) -> None:
        with pytest.raises(SchemaError):
            score({"type": "json_schema", "schema": {"type": 5}}, "{}")


class TestToolCall:
    CALL = '<tool_call>{"name": "search", "arguments": {"q": "weather", "n": 3}}</tool_call>'

    def test_named_tool_with_required_arguments(self) -> None:
        reward = {"type": "tool_call", "name": "search", "required_arguments": ["q"]}

        assert score(reward, self.CALL) == 1.0
        assert score({**reward, "required_arguments": ["page"]}, self.CALL) == 0.0
        assert score({**reward, "name": "browse"}, self.CALL) == 0.0

    def test_a_bare_object_counts_as_a_call(self) -> None:
        reward = {"type": "tool_call", "name": "search"}

        assert score(reward, 'Calling {"name": "search", "arguments": {}}') == 1.0

    @pytest.mark.parametrize(
        "reply",
        [
            "I will search for it.",
            '<tool_call>{"name": "search"}</tool_call>',
            '<tool_call>{"name": 3, "arguments": {}}</tool_call>',
            '<tool_call>{"name": "search", "arguments": "q=weather"}</tool_call>',
        ],
    )
    def test_a_reply_without_a_valid_call_scores_0(self, reply: str) -> None:
        assert score({"type": "tool_call", "name": "search"}, reply) == 0.0

    @pytest.mark.parametrize(
        ("answer", "expected"),
        [
            ('{"name": "search", "arguments": {"n": 3, "q": "weather"}}', 1.0),
            ({"name": "search", "arguments": {"q": "weather", "n": 3}}, 1.0),
            ({"name": "search", "arguments": {"q": "weather"}}, 0.0),
            ({"name": "browse", "arguments": {"q": "weather", "n": 3}}, 0.0),
            (None, 0.0),
        ],
    )
    def test_without_a_name_the_answer_is_the_expected_call(
        self, answer: Any, expected: float
    ) -> None:
        assert score({"type": "tool_call"}, self.CALL, answer) == expected


def test_the_rubric_reports_under_the_reward_type() -> None:
    rubric = BuiltinRewardRubric(
        REWARD.validate_python({"type": "regex", "pattern": "x"})
    )

    assert rubric.component_name == "regex"

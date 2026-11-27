# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Built-in rewards a manifest picks by name in ``environment.reward``.

Each one scores a turn's final answer, the text after any reasoning, against
the row's ``answer`` column. Map another column onto ``answer`` with
``environment.columns``. ``snowflake_function`` scores each finished group of
completions with a function the buyer binds to the Snowflake app.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Self

if TYPE_CHECKING:
    # env.py imports this module for RewardSpec.
    from agilerl.models.env import LLMEnvSpec


def compiled(pattern: str) -> re.Pattern[str]:
    """Compile a manifest regular expression.

    :param pattern: The expression.
    :return: The compiled pattern.
    :raises ValueError: It is not a valid Python regular expression.
    """
    try:
        return re.compile(pattern)
    except re.error as exc:
        msg = f"{pattern!r} is not a valid regular expression: {exc}"
        raise ValueError(msg) from exc


class RewardBase(BaseModel):
    """Fields every built-in reward shares."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class AnswerPatternReward(RewardBase):
    """A reward that can read the answer out of a longer reply."""

    answer_pattern: str | None = Field(
        default=None,
        description=(
            "Regular expression whose first group is the answer, for example "
            "'Answer: (.*)'. The last match counts. A reply with no match scores 0. "
            "Unset reads the whole reply."
        ),
    )

    @field_validator("answer_pattern")
    @classmethod
    def _check_answer_pattern(cls, pattern: str | None) -> str | None:
        if pattern is not None and compiled(pattern).groups < 1:
            msg = "answer_pattern needs a capturing group, for example 'Answer: (.*)'."
            raise ValueError(msg)
        return pattern


class ExactMatchReward(AnswerPatternReward):
    """1 when the answer equals the row's ``answer``, else 0."""

    type: Literal["exact_match"] = Field(
        description="Compare the final answer with the row's answer."
    )
    normalize: bool = Field(
        default=True,
        description=(
            "Ignore case, repeated whitespace, and a trailing full stop before comparing."
        ),
    )


class NumericReward(AnswerPatternReward):
    r"""1 when the answer is a number within ``tolerance`` of the row's ``answer``, else 0.

    Without ``answer_pattern`` the answer is the last ``\boxed{...}`` in the
    reply, or else its last number.
    """

    type: Literal["numeric"] = Field(
        description="Compare the final answer's number with the row's answer."
    )
    tolerance: float = Field(
        default=1e-6,
        ge=0.0,
        description="Largest absolute difference that still counts as correct.",
    )


class RegexReward(RewardBase):
    """1 when the reply matches ``pattern``, else 0."""

    type: Literal["regex"] = Field(
        description="Check the final answer against a regular expression."
    )
    pattern: str = Field(min_length=1, description="Python regular expression.")
    full_match: bool = Field(
        default=False,
        description="Match the whole reply instead of finding the pattern anywhere in it.",
    )

    @field_validator("pattern")
    @classmethod
    def _check_pattern(cls, pattern: str) -> str:
        compiled(pattern)
        return pattern


class JsonSchemaReward(RewardBase):
    """1 when the reply's first JSON object is valid against ``schema``, else 0."""

    type: Literal["json_schema"] = Field(
        description="Check the final answer's first JSON object against a JSON Schema."
    )
    json_schema: dict[str, Any] = Field(
        alias="schema",
        description="JSON Schema the object must satisfy.",
    )


class ToolCallReward(RewardBase):
    """1 when the reply's first tool call is the expected one, else 0.

    A tool call is a JSON object with ``name`` and ``arguments``, bare or inside
    ``<tool_call>`` tags. With ``name`` set, the call must use that tool and pass
    every key in ``required_arguments``. Without it, the row's ``answer`` is the
    expected call, as such an object or its JSON text, and both the name and the
    arguments must equal it.
    """

    type: Literal["tool_call"] = Field(
        description="Check the final answer's first tool call."
    )
    name: str | None = Field(
        default=None,
        min_length=1,
        description="Tool the call must use. Unset reads the expected call from answer.",
    )
    required_arguments: list[str] = Field(
        default_factory=list,
        description="Argument keys the call must pass. Needs name.",
    )

    @field_validator("required_arguments")
    @classmethod
    def _check_required_arguments(cls, keys: list[str]) -> list[str]:
        if len(set(keys)) != len(keys):
            msg = "required_arguments lists a key twice."
            raise ValueError(msg)
        return keys

    @model_validator(mode="after")
    def _check_name(self) -> Self:
        if self.required_arguments and self.name is None:
            msg = (
                "required_arguments needs name; without it answer is the expected call."
            )
            raise ValueError(msg)
        return self


class SnowflakeFunctionReward(RewardBase):
    """Scores with the function bound to the Snowflake app's ``reward_function`` reference.

    Only the Snowflake Native App runs it. The app sends finished groups of
    completions to the function in batched queries, and its score is each
    completion's reward.
    """

    type: Literal["snowflake_function"] = Field(
        description="Score with the function bound to the Snowflake app's reward_function reference."
    )


BuiltinReward = (
    ExactMatchReward | NumericReward | RegexReward | JsonSchemaReward | ToolCallReward
)

RewardSpec = Annotated[
    BuiltinReward | SnowflakeFunctionReward, Field(discriminator="type")
]


def check_reward_fields(spec: LLMEnvSpec) -> None:
    """Check an LLM env's reward against its rubric file fields.

    :param spec: The env spec.
    :raises ValueError: ``reward`` is set on a dataset env or with rubric file
        fields, or a rubric file field is set without ``rubric_file_path``.
    """
    if spec.reward is not None:
        if spec.env_type != "rollout":
            msg = "reward scores rollouts; a dataset environment trains on its rows."
            raise ValueError(msg)
        clashing = [
            field
            for field, value in (
                ("rubric_file_path", spec.rubric_file_path),
                ("group_rubric_name", spec.group_rubric_name),
                ("judge", spec.judge),
            )
            if value is not None
        ]
        if clashing:
            msg = f"reward replaces the rubric file; drop {', '.join(clashing)}."
            raise ValueError(msg)
    if spec.rubric_file_path is not None:
        return
    if spec.judge is not None:
        msg = "judge is handed to a rubric; set rubric_file_path to a Rubric that takes it."
        raise ValueError(msg)
    if spec.group_rubric_name is not None:
        msg = "group_rubric_name names a symbol in rubric_file_path; set it too."
        raise ValueError(msg)
    if spec.rubric_name is not None:
        msg = "rubric_name names a symbol in rubric_file_path; set it too."
        raise ValueError(msg)


__all__ = [
    "BuiltinReward",
    "ExactMatchReward",
    "JsonSchemaReward",
    "NumericReward",
    "RegexReward",
    "RewardSpec",
    "SnowflakeFunctionReward",
    "ToolCallReward",
    "check_reward_fields",
]

# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Built-in rewards in ``environment.reward``: their options and where a manifest may use them."""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from agilerl.models.env import LLMEnvSpec
from agilerl.models.manifest import IN_SNOWFLAKE_APP, TrainingManifest
from agilerl.models.rewards import (
    ExactMatchReward,
    JsonSchemaReward,
    NumericReward,
    RegexReward,
    RewardSpec,
    SnowflakeFunctionReward,
    ToolCallReward,
)

REWARD = TypeAdapter(RewardSpec)
NETWORK = {
    "pretrained_model_name_or_path": "Qwen/Qwen3.5-0.8B",
    "max_context_length": 128,
}


def rollout(**fields: object) -> dict[str, object]:
    """A rollout environment over dataset rows, with ``fields`` added."""
    return {
        "env_type": "rollout",
        "dataset": "openai/gsm8k",
        "prompt_template": {"user_0": "{question}"},
        **fields,
    }


class TestRewardSpec:
    @pytest.mark.parametrize(
        ("document", "kind"),
        [
            ({"type": "exact_match"}, ExactMatchReward),
            ({"type": "numeric", "tolerance": 0.5}, NumericReward),
            ({"type": "regex", "pattern": r"^\d+$"}, RegexReward),
            ({"type": "json_schema", "schema": {"type": "object"}}, JsonSchemaReward),
            ({"type": "tool_call", "name": "search"}, ToolCallReward),
            ({"type": "snowflake_function"}, SnowflakeFunctionReward),
        ],
    )
    def test_type_picks_the_reward(self, document: dict, kind: type) -> None:
        assert isinstance(REWARD.validate_python(document), kind)

    def test_defaults(self) -> None:
        exact = REWARD.validate_python({"type": "exact_match"})
        numeric = REWARD.validate_python({"type": "numeric"})
        regex = REWARD.validate_python({"type": "regex", "pattern": "x"})

        assert (exact.normalize, exact.answer_pattern) == (True, None)
        assert numeric.tolerance == 1e-6
        assert regex.full_match is False

    def test_answer_pattern_keeps_a_pattern_with_a_group(self) -> None:
        reward = REWARD.validate_python(
            {"type": "numeric", "answer_pattern": "Answer: (.*)"}
        )

        assert reward.answer_pattern == "Answer: (.*)"

    def test_json_schema_reads_the_schema_key(self) -> None:
        reward = REWARD.validate_python(
            {"type": "json_schema", "schema": {"required": ["a"]}}
        )

        assert reward.json_schema == {"required": ["a"]}

    @pytest.mark.parametrize(
        ("document", "message"),
        [
            ({"type": "llm_judge"}, "does not match any of the expected tags"),
            ({"type": "regex", "pattern": "("}, "missing \\)"),
            ({"type": "regex", "pattern": ""}, "at least 1 character"),
            (
                {"type": "exact_match", "answer_pattern": "Answer: .*"},
                "answer_pattern needs a capturing group",
            ),
            ({"type": "numeric", "tolerance": -1}, "greater than or equal to 0"),
            ({"type": "json_schema"}, "schema"),
            (
                {"type": "tool_call", "required_arguments": ["query"]},
                "required_arguments needs name",
            ),
            (
                {
                    "type": "tool_call",
                    "name": "search",
                    "required_arguments": ["q", "q"],
                },
                "lists a key twice",
            ),
            ({"type": "exact_match", "field": "answer"}, "Extra inputs"),
        ],
    )
    def test_rejects_bad_options(self, document: dict, message: str) -> None:
        with pytest.raises(ValidationError, match=message):
            REWARD.validate_python(document)


class TestEnvironmentReward:
    def test_a_rollout_takes_a_reward(self) -> None:
        spec = LLMEnvSpec.model_validate(rollout(reward={"type": "numeric"}))

        assert isinstance(spec.reward, NumericReward)
        assert spec.rubric_file_path is None

    def test_a_dataset_environment_refuses_a_reward(self) -> None:
        with pytest.raises(ValidationError, match="reward scores rollouts"):
            LLMEnvSpec.model_validate(
                {
                    "env_type": "dataset",
                    "objective": "sft",
                    "dataset": "openai/gsm8k",
                    "reward": {"type": "exact_match"},
                }
            )

    def test_a_reward_refuses_a_rubric_file_and_its_options(self) -> None:
        with pytest.raises(
            ValidationError, match="drop rubric_file_path, group_rubric_name, judge"
        ):
            LLMEnvSpec.model_validate(
                rollout(
                    reward={"type": "exact_match"},
                    rubric_file_path="reward.py",
                    group_rubric_name="Ranker",
                    judge={"model": "judge-1", "base_url": "http://judge"},
                )
            )

    def test_a_reward_with_a_judge_is_told_to_drop_the_judge(self) -> None:
        with pytest.raises(ValidationError, match="drop judge"):
            LLMEnvSpec.model_validate(
                rollout(
                    reward={"type": "exact_match"},
                    judge={"model": "judge-1", "base_url": "http://judge"},
                )
            )

    def test_a_dataset_rollout_with_a_reward_needs_no_rubric_file(self) -> None:
        manifest = TrainingManifest.model_validate(
            {
                "algorithm": {"name": "GRPO", "group_size": 4, "batch_size": 2},
                "environment": rollout(reward={"type": "exact_match"}),
                "network": NETWORK,
                "training": {"max_steps": 2},
            }
        )

        assert isinstance(manifest.environment.reward, ExactMatchReward)

    def test_a_dataset_rollout_with_no_reward_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="needs reward or rubric_file_path"):
            TrainingManifest.model_validate(
                {
                    "algorithm": {"name": "GRPO", "group_size": 4, "batch_size": 2},
                    "environment": rollout(),
                    "network": NETWORK,
                    "training": {"max_steps": 2},
                }
            )

    def test_the_snowflake_app_asks_for_a_reward_not_a_rubric_file(self) -> None:
        with pytest.raises(
            ValidationError,
            match=r"needs environment\.reward; the Snowflake app takes no rubric file",
        ):
            TrainingManifest.model_validate(
                {
                    "algorithm": {"name": "GRPO", "group_size": 4, "batch_size": 2},
                    "environment": rollout(),
                    "network": NETWORK,
                    "training": {"max_steps": 2},
                },
                context={IN_SNOWFLAKE_APP: True},
            )

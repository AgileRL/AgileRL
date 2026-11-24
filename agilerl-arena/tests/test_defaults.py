# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for Arena's default LLM manifest."""

from __future__ import annotations

from typing import Any

import pytest

from agilerl.arena.models import SUPPORTED_MODEL_INFO
from agilerl.arena.models.defaults import default_verdict
from agilerl.arena.models.manifest import REWARD_FILE_AT_SUBMIT
from agilerl.arena.models.verdict import verdict

MODEL = "Qwen/Qwen3.5-0.8B"


def dataset_rollout() -> dict[str, Any]:
    return {
        "name": "Reasoning test",
        "env_type": "rollout",
        "dataset": "Reasoning test",
        "columns": {"answer": "answer", "question": "question"},
        "prompt_template": {"system_0": "Solve it.", "user_1": "{question}"},
    }


def document(algorithm: dict[str, Any], environment: dict[str, Any]) -> dict[str, Any]:
    return {
        "algorithm": algorithm,
        "environment": environment,
        "network": {"pretrained_model_name_or_path": MODEL},
    }


class TestDefaultVerdict:
    @pytest.mark.parametrize(
        "name", ["GRPO", "CISPO", "GSPO", "LLMPPO", "LLMREINFORCE"]
    )
    def test_dataset_rollout_trains_one_epoch(self, name: str) -> None:
        # Act
        record = default_verdict(document({"name": name}, dataset_rollout()))

        # Assert
        assert record["ok"] is True
        training = record["payload"]["training"]
        assert training["num_epochs"] == 1
        assert "max_steps" not in training
        assert training["reporting_interval"] == 1
        assert training["evaluation_interval"] == 10
        assert training["checkpoint_export"] == {
            "format": "adapter",
            "trigger": "final",
        }
        assert training["rollout_version_stamp"] == "oldest_turn"

    def test_rollout_without_reward_file_names_no_rubric(self) -> None:
        # Act
        record = default_verdict(document({"name": "GRPO"}, dataset_rollout()))

        # Assert
        environment = record["payload"]["environment"]
        assert "rubric_file_path" not in environment
        assert "rubric_name" not in environment

    def test_payload_passes_verdict_once_submit_adds_the_reward_file(self) -> None:
        # Arrange
        payload = default_verdict(document({"name": "GRPO"}, dataset_rollout()))[
            "payload"
        ]
        payload["environment"]["rubric_file_path"] = "s3://bucket/reward.py"

        # Act
        record = verdict(payload)

        # Assert
        assert record["ok"] is True
        assert record["payload"]["environment"]["rubric_name"] == "reward_fn"

    def test_keeps_an_authored_reward_file_and_rubric(self) -> None:
        # Arrange
        environment = {
            **dataset_rollout(),
            "rubric_file_path": "s3://bucket/reward.py",
            "rubric_name": "RUBRIC",
        }

        # Act
        record = default_verdict(document({"name": "GRPO"}, environment))

        # Assert
        assert record["payload"]["environment"]["rubric_file_path"] == (
            "s3://bucket/reward.py"
        )
        assert record["payload"]["environment"]["rubric_name"] == "RUBRIC"

    def test_fills_context_length_from_the_catalog(self) -> None:
        # Act
        record = default_verdict(document({"name": "GRPO"}, dataset_rollout()))

        # Assert
        assert record["payload"]["network"]["max_context_length"] == (
            SUPPORTED_MODEL_INFO[MODEL].max_context_length
        )

    def test_keeps_authored_training_values(self) -> None:
        # Arrange
        partial = document({"name": "GRPO"}, dataset_rollout())
        partial["training"] = {"reporting_interval": 5}

        # Act
        record = default_verdict(partial)

        # Assert
        assert record["payload"]["training"]["reporting_interval"] == 5

    def test_sft_dataset_trains_one_epoch(self) -> None:
        # Arrange
        environment = {
            "name": "rows",
            "env_type": "dataset",
            "dataset": "rows",
            "objective": "sft",
        }

        # Act
        record = default_verdict(document({"name": "SFT"}, environment))

        # Assert
        assert record["ok"] is True
        assert record["payload"]["training"]["num_epochs"] == 1
        assert "rubric_file_path" not in record["payload"]["environment"]

    def test_hosted_env_keeps_its_own_reward(self) -> None:
        # Arrange
        environment = {"name": "env", "env_type": "rollout", "env_image": "env:1"}

        # Act
        record = default_verdict(document({"name": "GRPO"}, environment))

        # Assert
        assert record["ok"] is True
        assert "rubric_file_path" not in record["payload"]["environment"]
        assert "rubric_name" not in record["payload"]["environment"]
        assert record["payload"]["training"]["max_steps"] == 200

    def test_gym_env_gets_the_rollout_step_budget(self) -> None:
        # Arrange
        environment = {"name": "Acrobot", "env_type": "gym", "version": 1}

        # Act
        record = default_verdict(document({"name": "GRPO"}, environment))

        # Assert
        assert record["ok"] is True, record
        assert record["payload"]["training"]["max_steps"] == 200
        assert "num_epochs" not in record["payload"]["training"]

    def test_deferring_the_reward_file_leaves_submit_checks_off(self) -> None:
        # Arrange
        partial = document({"name": "GRPO", "batch_size": 6}, dataset_rollout())

        # Act
        record = default_verdict(partial)
        payload = record["payload"]
        deferred = verdict(payload, context={REWARD_FILE_AT_SUBMIT: True})
        submitted = verdict(payload, submit=True, context={REWARD_FILE_AT_SUBMIT: True})

        # Assert
        assert record["ok"] is True
        assert payload["algorithm"]["batch_size"] == 6
        assert deferred["ok"] is True
        assert submitted["ok"] is False
        assert submitted["errors"][0]["path"] == "algorithm.batch_size"

    def test_reports_an_unknown_algorithm(self) -> None:
        # Act
        record = default_verdict(document({"name": "NOPE"}, dataset_rollout()))

        # Assert
        assert record["ok"] is False
        assert "NOPE" in record["errors"][0]["message"]

    def test_rejects_a_classic_algorithm(self) -> None:
        # Act
        record = default_verdict(
            {"algorithm": {"name": "DQN"}, "environment": {"name": "CartPole-v1"}}
        )

        # Assert
        assert record["ok"] is False
        assert record["errors"][0]["path"] == "algorithm.name"

    def test_reports_an_invalid_document(self) -> None:
        # Arrange
        environment = {**dataset_rollout(), "num_envs": 0}

        # Act
        record = default_verdict(document({"name": "GRPO"}, environment))

        # Assert
        assert record["ok"] is False
        assert record["errors"][0]["path"] == "environment.rollout.num_envs"

    def test_keeps_an_authored_step_budget(self) -> None:
        # Arrange
        environment = {
            "name": "rows",
            "env_type": "dataset",
            "dataset": "rows",
            "objective": "sft",
        }
        partial = document({"name": "SFT"}, environment)
        partial["training"] = {"max_steps": 50}

        # Act
        record = default_verdict(partial)

        # Assert
        assert record["ok"] is True
        assert record["payload"]["training"]["max_steps"] == 50
        assert "num_epochs" not in record["payload"]["training"]

    def test_rejects_a_group_rubric_without_its_file(self) -> None:
        # Arrange
        environment = {
            "name": "env",
            "env_type": "rollout",
            "env_image": "env:1",
            "group_rubric_name": "RULER",
        }

        # Act
        record = default_verdict(document({"name": "GRPO"}, environment))

        # Assert
        assert record["ok"] is False
        assert "group_rubric_name" in record["errors"][0]["message"]

    def test_reports_a_section_that_is_not_a_mapping(self) -> None:
        # Act
        record = default_verdict({"algorithm": None, "environment": {}})

        # Assert
        assert record["ok"] is False

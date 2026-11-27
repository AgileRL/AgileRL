# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for Arena's default LLM manifest."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from agilerl.models import SFTSpec
from agilerl.models.defaults import default_verdict
from agilerl.models.prompt_template import DEFAULT_PROMPT_TEMPLATE
from agilerl.models.verdict import verdict

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
    def test_dataset_rollout_leaves_the_length_to_one_pass(self, name: str) -> None:
        # Act
        record = default_verdict(document({"name": name}, dataset_rollout()))

        # Assert
        assert record["ok"] is True
        training = record["payload"]["training"]
        assert "num_epochs" not in training
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

    def test_fills_the_default_context_length(self) -> None:
        # Act
        record = default_verdict(document({"name": "GRPO"}, dataset_rollout()))

        # Assert
        assert record["payload"]["network"]["max_context_length"] == 2048

    def test_keeps_authored_training_values(self) -> None:
        # Arrange
        partial = document({"name": "GRPO"}, dataset_rollout())
        partial["training"] = {"reporting_interval": 5}

        # Act
        record = default_verdict(partial)

        # Assert
        assert record["payload"]["training"]["reporting_interval"] == 5

    def test_sft_dataset_leaves_the_length_to_one_pass(self) -> None:
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
        assert "num_epochs" not in record["payload"]["training"]
        assert "max_steps" not in record["payload"]["training"]
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

    def test_rejects_a_batch_size_outside_the_options(self) -> None:
        # Act
        record = default_verdict(
            document({"name": "GRPO", "batch_size": 6}, dataset_rollout())
        )

        # Assert
        assert record["ok"] is False
        assert record["errors"][0]["path"] == "algorithm.batch_size"

    def test_warns_that_the_context_is_the_default(self) -> None:
        # Act
        record = default_verdict(document({"name": "GRPO"}, dataset_rollout()))

        # Assert
        assert record["ok"] is True, record
        assert record["payload"]["network"]["max_context_length"] == 2048
        assert any(
            "default context of 2048 tokens" in warning
            for warning in record["warnings"]
        )

    def test_fills_a_bundled_environment_context_and_settings(self) -> None:
        # Act
        record = default_verdict(
            document(
                {"name": "GRPO"},
                {"name": "AgentWorldModel", "env_type": "rollout", "num_envs": 1},
            )
        )

        # Assert
        assert record["ok"] is True, record
        assert record["payload"]["network"]["max_context_length"] == 32768
        assert record["payload"]["algorithm"]["max_output_tokens"] == 2048
        assert not any("default context" in w for w in record["warnings"])

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
        assert record["errors"][0]["path"] == "environment.num_envs"

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

    def test_sets_the_vllm_memory_share(self) -> None:
        # Act
        record = default_verdict(document({"name": "GRPO"}, dataset_rollout()))

        # Assert
        assert record["ok"] is True
        assert (
            record["payload"]["algorithm"]["vllm_config"]["gpu_memory_utilization"]
            == 0.9
        )

    def test_names_no_lora_modules(self) -> None:
        # Act
        record = default_verdict(document({"name": "GRPO"}, dataset_rollout()))

        # Assert
        assert "target_modules" not in record["payload"]["network"]["lora_config"]

    def test_default_manifest_passes_submit(self) -> None:
        # Arrange
        payload = default_verdict(document({"name": "GRPO"}, dataset_rollout()))[
            "payload"
        ]
        payload["environment"]["rubric_file_path"] = "s3://bucket/reward.py"

        # Act
        record = verdict(payload, submit=True)

        # Assert
        assert record["ok"] is True, record["errors"]
        assert record["warnings"] == []

    def test_user_view_leaves_out_what_arena_chooses_and_passes_submit(self) -> None:
        # Act
        record = default_verdict(
            document({"name": "GRPO"}, dataset_rollout()), user_view=True
        )

        # Assert
        payload = record["payload"]
        payload["environment"]["rubric_file_path"] = "s3://bucket/reward.py"
        assert {"mutation", "tournament_selection"}.isdisjoint(payload)
        assert {"pop_size", "hpo", "rollout_version_stamp"}.isdisjoint(
            payload["training"]
        )
        assert "batch_size" not in payload["algorithm"]
        assert "gpu_memory_utilization" not in payload["algorithm"]["vllm_config"]
        assert "task_type" not in payload["network"]["lora_config"]
        assert verdict(payload, submit=True)["ok"] is True

    def test_user_view_leaves_out_evo_steps_so_a_short_run_validates(self) -> None:
        # Arrange
        record = default_verdict(
            document({"name": "GRPO"}, dataset_rollout()), user_view=True
        )
        payload = record["payload"]
        payload["environment"]["rubric_file_path"] = "s3://bucket/reward.py"
        payload["training"]["max_steps"] = 2

        # Act
        result = verdict(payload)

        # Assert
        assert "evo_steps" not in record["payload"]["training"]
        assert result["ok"] is True, result["errors"]
        assert result["payload"]["training"]["evo_steps"] == 2

    def test_user_view_keeps_an_image_environment_submittable(self) -> None:
        # Arrange
        environment = {
            "name": "MyEnv",
            "env_type": "rollout",
            "env_image": "registry/my-env:1",
            "env_port": 9000,
            "cpus_per_env_host": 2,
        }

        # Act
        record = default_verdict(
            document({"name": "GRPO"}, environment), user_view=True
        )

        # Assert
        payload = record["payload"]
        assert payload["environment"]["env_image"] == "registry/my-env:1"
        assert {"env_port", "cpus_per_env_host"}.isdisjoint(payload["environment"])
        assert verdict(payload, submit=True)["ok"] is True

    def test_uploaded_entrypoint_with_host_resources_is_hosted(self) -> None:
        # Arrange
        environment = {
            "name": "env",
            "env_type": "rollout",
            "entrypoint": "mod:Env",
            "version": 3,
            "cpus_per_env_host": 1.5,
            "env_host_memory_bytes": 4 * 1024**3,
        }

        # Act
        record = default_verdict(document({"name": "GRPO"}, environment))

        # Assert
        assert record["ok"] is True, record
        assert record["payload"]["environment"]["env_hosts"] == 1

    @pytest.mark.parametrize(
        ("name", "objective", "unused", "kept"),
        [
            ("SFT", "sft", {"beta", "answer_continuation"}, set()),
            ("DPO", "preference", {"answer_continuation"}, {"beta"}),
        ],
    )
    def test_dataset_algorithm_leaves_out_fields_it_does_not_use(
        self, name: str, objective: str, unused: set[str], kept: set[str]
    ) -> None:
        # Arrange
        environment = {
            "name": "rows",
            "env_type": "dataset",
            "dataset": "rows",
            "objective": objective,
        }

        # Act
        record = default_verdict(document({"name": name}, environment))

        # Assert
        assert record["ok"] is True, record
        algorithm = record["payload"]["algorithm"]
        assert unused.isdisjoint(algorithm)
        assert kept <= algorithm.keys()

    def test_rollout_algorithm_keeps_answer_continuation(self) -> None:
        # Act
        record = default_verdict(document({"name": "GRPO"}, dataset_rollout()))

        # Assert
        assert record["payload"]["algorithm"]["answer_continuation"] is False

    def test_rejects_a_model_outside_the_catalog(self) -> None:
        # Arrange
        partial = document({"name": "GRPO"}, dataset_rollout())
        partial["network"]["pretrained_model_name_or_path"] = "acme/not-a-model"

        # Act
        record = default_verdict(partial)

        # Assert
        assert record["ok"] is False
        assert record["errors"] == [
            {
                "path": "network.pretrained_model_name_or_path",
                "message": "Arena does not support the model 'acme/not-a-model'.",
                "kind": "value_error",
            }
        ]

    @pytest.mark.parametrize(
        "model",
        ["/models/my-model", "./checkpoints/mine", "~/models/mine"],
    )
    def test_accepts_a_path_when_the_context_length_is_set(self, model: str) -> None:
        # Arrange
        partial = document({"name": "GRPO"}, dataset_rollout())
        partial["network"] = {
            "pretrained_model_name_or_path": model,
            "max_context_length": 1024,
        }

        # Act
        record = default_verdict(partial)

        # Assert
        assert record["ok"] is True, record
        network = record["payload"]["network"]
        assert network["pretrained_model_name_or_path"] == model
        assert network["max_context_length"] == 1024

    def test_accepts_an_on_disk_checkout_when_the_context_length_is_set(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        checkout = tmp_path / "models" / "mine"
        checkout.mkdir(parents=True)
        (checkout / "config.json").write_text("{}", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        partial = document({"name": "GRPO"}, dataset_rollout())
        partial["network"] = {
            "pretrained_model_name_or_path": "models/mine",
            "max_context_length": 1024,
        }

        # Act
        record = default_verdict(partial)

        # Assert
        assert record["ok"] is True, record
        assert (
            record["payload"]["network"]["pretrained_model_name_or_path"]
            == "models/mine"
        )

    def test_rejects_a_relative_name_with_no_checkout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        monkeypatch.chdir(tmp_path)
        partial = document({"name": "GRPO"}, dataset_rollout())
        partial["network"] = {
            "pretrained_model_name_or_path": "models/mine",
            "max_context_length": 1024,
        }

        # Act
        record = default_verdict(partial)

        # Assert
        assert record["ok"] is False
        assert record["errors"] == [
            {
                "path": "network.pretrained_model_name_or_path",
                "message": "Arena does not support the model 'models/mine'.",
                "kind": "value_error",
            }
        ]

    def test_a_local_path_without_context_length_is_a_missing_field(self) -> None:
        # Arrange
        partial = document({"name": "GRPO"}, dataset_rollout())
        partial["network"]["pretrained_model_name_or_path"] = "/models/my-model"

        # Act
        record = default_verdict(partial)

        # Assert
        assert record["ok"] is False
        assert record["errors"] == [
            {
                "path": "network.max_context_length",
                "message": "Field required",
                "kind": "missing",
            }
        ]

    def test_rejects_sft_beta_outside_zero_to_one(self) -> None:
        # Arrange
        environment = {
            "name": "rows",
            "env_type": "dataset",
            "dataset": "rows",
            "objective": "sft",
        }
        partial = document({"name": "SFT", "beta": -1}, environment)

        # Act
        record = default_verdict(partial)

        # Assert
        assert record["ok"] is False
        assert record["errors"] == [
            {
                "path": "algorithm.beta",
                "message": "Input should be greater than or equal to 0",
                "kind": "greater_than_equal",
            }
        ]

    def test_rejects_an_sft_beta_mutation_outside_zero_to_one(self) -> None:
        # Arrange
        environment = {
            "name": "rows",
            "env_type": "dataset",
            "dataset": "rows",
            "objective": "sft",
        }
        partial = document({"name": "SFT"}, environment)
        partial["mutation"] = {"rl_hp_selection": {"beta": {"min": -1, "max": 2}}}

        # Act
        record = default_verdict(partial)

        # Assert
        assert record["ok"] is False
        assert record["errors"] == [
            {
                "path": "",
                "message": (
                    "Value error, mutation.rl_hp_selection cannot mutate 'beta': "
                    "range [-1.0, 2.0] leaves the field limits (>= 0.0 and <= 1.0): "
                    "min -1.0 is below 0.0; max 2.0 is above 1.0"
                ),
                "kind": "value_error",
            }
        ]


def untemplated_rollout() -> dict[str, Any]:
    environment = dataset_rollout()
    del environment["prompt_template"]
    environment["reward"] = {"type": "exact_match"}
    return environment


class TestDefaultVerdictStartingValues:
    def test_fills_the_prompt_template_and_max_reward_over_dataset_rows(
        self,
    ) -> None:
        # Act
        record = default_verdict(document({"name": "GRPO"}, untemplated_rollout()))

        # Assert
        assert record["ok"] is True, record
        environment = record["payload"]["environment"]
        assert environment["prompt_template"] == {
            "system_0": (
                "A conversation between User and Assistant. The user asks a "
                "question, and the Assistant solves it. The assistant first "
                "thinks about the reasoning process in the mind and then gives "
                "the final answer inside <answer> </answer> tags, i.e., "
                "<answer> answer here </answer>."
            ),
            "user_1": (
                "Answer the following question as accurately, clearly, and "
                "concisely as possible.\n\nQuestion: {question}\n"
            ),
        }
        assert environment["max_reward"] == 1.0
        assert environment["max_turns"] == 1
        assert verdict(record["payload"], submit=True)["ok"] is True

    def test_fills_the_default_model_when_the_request_names_none(self) -> None:
        # Arrange
        partial = document({"name": "GRPO"}, untemplated_rollout())
        del partial["network"]

        # Act
        record = default_verdict(partial)

        # Assert
        assert record["ok"] is True, record
        assert record["payload"]["network"]["pretrained_model_name_or_path"] == (
            "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
        )

    def test_keeps_a_model_named_only_on_the_algorithm(self) -> None:
        # Arrange
        partial = document(
            {"name": "GRPO", "pretrained_model_name_or_path": MODEL},
            untemplated_rollout(),
        )
        del partial["network"]

        # Act
        record = default_verdict(partial)

        # Assert
        assert record["ok"] is True, record
        payload = record["payload"]
        assert payload["algorithm"]["pretrained_model_name_or_path"] == MODEL
        assert "network" not in payload
        assert payload["environment"]["prompt_template"] == DEFAULT_PROMPT_TEMPLATE
        assert payload["environment"]["max_reward"] == 1.0
        assert verdict(payload, submit=True)["ok"] is True

    def test_keeps_an_authored_template_max_reward_and_model(self) -> None:
        # Arrange
        template = {"system_prompt": "Be brief.", "user_prompt": "Q: {question}"}
        environment = {
            **untemplated_rollout(),
            "prompt_template": template,
            "max_reward": 5.0,
        }

        # Act
        record = default_verdict(document({"name": "GRPO"}, environment))

        # Assert
        assert record["ok"] is True, record
        assert record["payload"]["environment"]["prompt_template"] == template
        assert record["payload"]["environment"]["max_reward"] == 5.0
        assert record["payload"]["network"]["pretrained_model_name_or_path"] == MODEL

    def test_leaves_a_hosted_env_without_template_or_max_reward(self) -> None:
        # Arrange
        environment = {"name": "env", "env_type": "rollout", "env_image": "env:1"}

        # Act
        record = default_verdict(document({"name": "GRPO"}, environment))

        # Assert
        assert record["ok"] is True, record
        assert "prompt_template" not in record["payload"]["environment"]
        assert "max_reward" not in record["payload"]["environment"]

    def test_leaves_a_traces_rollout_without_template_or_max_reward(self) -> None:
        # Arrange
        environment = {
            "name": "traces",
            "env_type": "rollout",
            "dataset": "traces",
            "dataset_format": "traces",
        }

        # Act
        record = default_verdict(document({"name": "OPD"}, environment))

        # Assert
        assert record["ok"] is True, record
        assert "prompt_template" not in record["payload"]["environment"]
        assert "max_reward" not in record["payload"]["environment"]

    def test_validation_fills_no_template(self) -> None:
        # Act
        record = verdict(document({"name": "GRPO"}, untemplated_rollout()))

        # Assert
        assert record["ok"] is False
        assert "Prompt template is required" in record["errors"][0]["message"]


class TestSFTSpecBeta:
    def test_keeps_the_bounds_and_leaves_beta_out_of_the_dump(self) -> None:
        # Act
        low = SFTSpec(beta=0.0)
        high = SFTSpec(beta=1.0)

        # Assert
        assert low.beta == 0.0
        assert high.beta == 1.0
        assert "beta" not in low.model_dump()
        assert "beta" not in high.model_dump()

    @pytest.mark.parametrize(
        ("beta", "match"),
        [(-1, "greater than or equal to 0"), (1.5, "less than or equal to 1")],
    )
    def test_rejects_a_value_outside_zero_to_one(self, beta: float, match: str) -> None:
        with pytest.raises(ValidationError, match=match):
            SFTSpec(beta=beta)


class TestDefaultVerdictOPDReference:
    def test_dataset_rollout_teacher_reads_the_answer_column(self) -> None:
        # Act
        record = default_verdict(
            document({"name": "OPD"}, dataset_rollout()), user_view=True
        )

        # Assert
        assert record["ok"] is True
        assert record["payload"]["algorithm"]["teacher"]["context_key"] == "answer"
        assert verdict(record["payload"], submit=True)["ok"] is True

    def test_keeps_an_authored_reference_column(self) -> None:
        # Arrange
        algorithm = {"name": "OPD", "teacher": {"context_key": "solution"}}

        # Act
        record = default_verdict(document(algorithm, dataset_rollout()))

        # Assert
        assert record["payload"]["algorithm"]["teacher"]["context_key"] == "solution"

    def test_base_weights_teacher_gets_no_reference(self) -> None:
        # Arrange
        algorithm = {"name": "OPD", "teacher": {"base": True}}

        # Act
        record = default_verdict(document(algorithm, dataset_rollout()))

        # Assert
        assert "context_key" not in record["payload"]["algorithm"]["teacher"]

    def test_environment_without_an_answer_column_keeps_the_trace_reference(
        self,
    ) -> None:
        # Arrange
        environment = {**dataset_rollout(), "columns": {"question": "question"}}

        # Act
        record = default_verdict(document({"name": "OPD"}, environment))

        # Assert
        assert record["payload"]["algorithm"]["teacher"]["context_key"] == (
            "reference_trace"
        )

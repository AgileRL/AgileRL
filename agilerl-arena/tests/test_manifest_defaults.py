# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for arena.models.defaults and `arena manifest defaults`."""

from __future__ import annotations

import json
from typing import Any

import pytest
from click.testing import CliRunner
from pydantic import ValidationError

from agilerl.arena.cli import main
from agilerl.arena.models import TrainingManifest, default_manifest, verdict


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


CARTPOLE: dict[str, Any] = {"name": "CartPole-v1"}
MPE: dict[str, Any] = {"name": "mpe2.simple_speaker_listener_v4"}
ROLLOUT_ENV: dict[str, Any] = {"env_type": "rollout", "entrypoint": "pkg:Env"}
REASONING_ENV: dict[str, Any] = {
    "env_type": "rollout",
    "dataset": "org/reasoning",
    "prompt_template": {"user_0": "{question}"},
    "rubric_file_path": "/tmp/rubric.py",
}
PREFERENCE_ENV: dict[str, Any] = {
    "env_type": "dataset",
    "dataset": "org/prefs",
    "objective": "preference",
}
SFT_ENV: dict[str, Any] = {
    "env_type": "dataset",
    "dataset": "org/sft",
    "objective": "sft",
}
TABULAR_ENV: dict[str, Any] = {"env_type": "tabular", "dataset": "mnist"}
MODEL = "Qwen/Qwen3-4B"


def _revalidates(payload: dict[str, Any]) -> dict[str, Any]:
    """The payload passes `manifest validate` and round-trips unchanged."""
    result = verdict.verdict(payload)
    assert result["ok"], result.get("errors")
    return result["payload"]


class TestDefaultManifest:
    def test_classic_off_policy(self) -> None:
        payload = default_manifest("DQN", environment=CARTPOLE)

        assert payload["algorithm"]["name"] == "DQN"
        assert payload["environment"]["name"] == "CartPole-v1"
        _revalidates(payload)

    def test_classic_on_policy(self) -> None:
        payload = default_manifest("PPO", environment=CARTPOLE)

        assert payload["algorithm"]["name"] == "PPO"
        _revalidates(payload)

    def test_multi_agent(self) -> None:
        payload = default_manifest("IPPO", environment=MPE)

        assert payload["algorithm"]["name"] == "IPPO"
        _revalidates(payload)

    def test_grpo_rollout(self) -> None:
        payload = default_manifest("GRPO", model=MODEL, environment=ROLLOUT_ENV)

        assert payload["network"]["pretrained_model_name_or_path"] == MODEL
        assert payload["network"]["max_context_length"] == 40960
        _revalidates(payload)

    def test_grpo_dataset_backed_rollout(self) -> None:
        payload = default_manifest("GRPO", model=MODEL, environment=REASONING_ENV)

        assert payload["environment"]["max_turns"] == 1
        _revalidates(payload)

    def test_dpo(self) -> None:
        payload = default_manifest("DPO", model=MODEL, environment=PREFERENCE_ENV)

        assert payload["algorithm"]["name"] == "DPO"
        _revalidates(payload)

    def test_sft(self) -> None:
        payload = default_manifest("SFT", model=MODEL, environment=SFT_ENV)

        assert payload["algorithm"]["name"] == "SFT"
        _revalidates(payload)

    def test_llmppo(self) -> None:
        payload = default_manifest("LLMPPO", model=MODEL, environment=ROLLOUT_ENV)

        assert payload["algorithm"]["name"] == "LLMPPO"
        _revalidates(payload)

    def test_supervised(self) -> None:
        payload = default_manifest("Supervised", environment=TABULAR_ENV)

        assert payload["algorithm"]["name"] == "Supervised"
        assert payload["algorithm"]["lr"] == 0.001
        assert payload["algorithm"]["fitness_metric"] == "accuracy"
        assert payload["training"]["num_epochs"] == 100
        assert payload["training"]["evo_steps"] == 128
        _revalidates(payload)

    def test_latent_ppo(self) -> None:
        payload = default_manifest(
            "LatentPPO", environment={"env_type": "non_tabular", "dataset": "seg"}
        )

        assert payload["algorithm"]["name"] == "LatentPPO"
        assert payload["algorithm"]["batch_size"] == 32
        assert payload["training"]["max_steps"] == 2000
        assert payload["training"]["evo_steps"] == 8
        _revalidates(payload)

    def test_unknown_algorithm(self) -> None:
        with pytest.raises(ValueError, match="Unknown algorithm 'Nope'"):
            default_manifest("Nope", environment=CARTPOLE)

    def test_llm_requires_model(self) -> None:
        with pytest.raises(ValueError, match="pass one with --model"):
            default_manifest("GRPO", environment=ROLLOUT_ENV)

    def test_classic_rejects_model(self) -> None:
        with pytest.raises(ValueError, match="--model is LLM-only"):
            default_manifest("DQN", model=MODEL, environment=CARTPOLE)

    def test_missing_environment(self) -> None:
        with pytest.raises(ValueError, match="needs an environment"):
            default_manifest("DQN")

    def test_unknown_model_needs_explicit_context(self) -> None:
        with pytest.raises(ValueError, match="pass max_context_length in --network"):
            default_manifest("GRPO", model="org/custom", environment=ROLLOUT_ENV)

    def test_unknown_model_with_explicit_context(self) -> None:
        payload = default_manifest(
            "GRPO",
            model="org/custom",
            environment=ROLLOUT_ENV,
            network={
                "max_context_length": 8192,
                "lora_config": {"target_modules": ["q_proj", "v_proj"]},
            },
        )

        assert payload["network"]["max_context_length"] == 8192
        _revalidates(payload)

    def test_unknown_model_without_targets_raises(self) -> None:
        with pytest.raises(ValidationError, match="target_modules is required"):
            default_manifest(
                "GRPO",
                model="org/custom",
                environment=ROLLOUT_ENV,
                network={"max_context_length": 8192},
            )

    def test_network_model_conflict(self) -> None:
        with pytest.raises(ValueError, match="disagrees with network"):
            default_manifest(
                "GRPO",
                model=MODEL,
                environment=ROLLOUT_ENV,
                network={
                    "pretrained_model_name_or_path": "other/model",
                    "max_context_length": 1024,
                },
            )

    def test_network_override_wins(self) -> None:
        payload = default_manifest(
            "GRPO",
            model=MODEL,
            environment=ROLLOUT_ENV,
            network={"max_context_length": 8192},
        )

        assert payload["network"]["max_context_length"] == 8192


class TestSupervisedSpec:
    def test_object_detection_batch_and_loss(self) -> None:
        payload = default_manifest(
            "Supervised",
            environment=TABULAR_ENV,
        )
        assert payload["algorithm"]["batch_size"] == 64

        manifest = TrainingManifest.model_validate(
            {
                "algorithm": {"name": "Supervised", "task_type": "object_detection"},
                "environment": TABULAR_ENV,
            }
        )

        assert manifest.algorithm.batch_size == 32
        assert manifest.training.num_epochs == 8
        assert manifest.training.evo_steps == 16

    def test_regression_rejects_cross_entropy(self) -> None:
        with pytest.raises(ValidationError, match="incompatible with loss"):
            TrainingManifest.model_validate(
                {
                    "algorithm": {
                        "name": "Supervised",
                        "task_type": "regression",
                        "loss_config": {"name": "CrossEntropyLoss"},
                    },
                    "environment": TABULAR_ENV,
                }
            )

    def test_regression_defaults_to_mse(self) -> None:
        manifest = TrainingManifest.model_validate(
            {
                "algorithm": {
                    "name": "Supervised",
                    "task_type": "regression",
                    "loss_config": {"name": "MSELoss"},
                },
                "environment": TABULAR_ENV,
            }
        )

        assert manifest.algorithm.fitness_metric == "mse"


class TestLatentPPOSpec:
    def test_reward_weights_must_match_components(self) -> None:
        with pytest.raises(ValidationError, match="must match reward_components"):
            TrainingManifest.model_validate(
                {
                    "algorithm": {
                        "name": "LatentPPO",
                        "reward_weights": [1.0],
                    },
                    "environment": {"env_type": "non_tabular", "dataset": "seg"},
                }
            )

    def test_latent_network_validates(self) -> None:
        manifest = TrainingManifest.model_validate(
            {
                "algorithm": {"name": "LatentPPO"},
                "environment": {"env_type": "non_tabular", "dataset": "seg"},
                "network": {"training_mode": "frozen_decoder"},
            }
        )

        assert manifest.network.training_mode == "frozen_decoder"
        assert manifest.network.num_conv_blocks == 4

    def test_latent_rejects_encoder_network(self) -> None:
        with pytest.raises(ValidationError, match="must be a latent network"):
            TrainingManifest.model_validate(
                {
                    "algorithm": {"name": "LatentPPO"},
                    "environment": {"env_type": "non_tabular", "dataset": "seg"},
                    "network": {
                        "encoder_config": {"arch": "mlp", "hidden_size": [64]},
                        "head_config": {"hidden_size": [64]},
                    },
                }
            )

    def test_classic_rejects_latent_network(self) -> None:
        with pytest.raises(ValidationError, match="must not declare"):
            TrainingManifest.model_validate(
                {
                    "algorithm": {"name": "DQN"},
                    "environment": CARTPOLE,
                    "network": {"training_mode": "joint"},
                }
            )


class TestManifestDefaultsCommand:
    def test_prints_defaulted_spec(self, runner: CliRunner) -> None:
        result = runner.invoke(
            main,
            [
                "manifest",
                "defaults",
                "--algorithm",
                "DQN",
                "--env",
                json.dumps(CARTPOLE),
            ],
        )

        assert result.exit_code == 0, result.output
        assert json.loads(result.output)["algorithm"]["name"] == "DQN"

    def test_llm_defaults(self, runner: CliRunner) -> None:
        result = runner.invoke(
            main,
            [
                "manifest",
                "defaults",
                "--algorithm",
                "GRPO",
                "--model",
                MODEL,
                "--env",
                json.dumps(ROLLOUT_ENV),
            ],
        )

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["network"]["max_context_length"] == 40960

    def test_missing_model_exits_invalid(self, runner: CliRunner) -> None:
        result = runner.invoke(
            main,
            [
                "manifest",
                "defaults",
                "--algorithm",
                "GRPO",
                "--env",
                json.dumps(ROLLOUT_ENV),
            ],
        )

        assert result.exit_code == 1
        assert "--model" in result.output

    def test_bad_json_exits_failed(self, runner: CliRunner) -> None:
        result = runner.invoke(
            main, ["manifest", "defaults", "--algorithm", "DQN", "--env", "{nope"]
        )

        assert result.exit_code == 2
        assert "could not parse JSON" in result.output

    def test_unknown_algorithm_exits_invalid(self, runner: CliRunner) -> None:
        result = runner.invoke(
            main,
            [
                "manifest",
                "defaults",
                "--algorithm",
                "Nope",
                "--env",
                json.dumps(CARTPOLE),
            ],
        )

        assert result.exit_code == 1
        assert "Unknown algorithm" in result.output

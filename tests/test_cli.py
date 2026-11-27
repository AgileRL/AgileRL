# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for arena.cli — Click CLI commands and config."""

from __future__ import annotations

import json
import logging
import uuid
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import click
import pytest
import yaml
from click.testing import CliRunner

from agilerl.cli import (
    CLI_PLUGIN_GROUP,
    _redact_agent_rows_for_display,
    arena_client,
    load_cli_plugins,
    main,
)
from agilerl.config import CommandConfig, build_client
from agilerl.exceptions import ArenaAPIError, ArenaTrainingError
from agilerl.inference import SessionDetail, SessionInfo, SessionMessage
from agilerl.inference.cache import load_active_session, save_active_session
from agilerl.models.model_info import STATUS_PREVIEW, SUPPORTED_MODEL_INFO
from agilerl.output import supports_interactive_selection


def _llm_manifest(model: str) -> str:
    return (
        "algorithm: {name: GRPO}\n"
        "environment: {env_type: rollout, entrypoint: 'envs:MyEnv'}\n"
        f"network: {{pretrained_model_name_or_path: {model}, "
        "max_context_length: 1024}\n"
    )


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def mock_agent(mock_client) -> MagicMock:
    """A deployed-agent mock returned by ``client.open_inference_agent()``."""
    agent = MagicMock()
    agent.__enter__ = MagicMock(return_value=agent)
    agent.__exit__ = MagicMock(return_value=False)
    mock_client.open_inference_agent.return_value = agent
    return agent


@pytest.fixture
def mock_client() -> MagicMock:
    return MagicMock(
        spec_set=[
            "get_default_run_spec",
            "close",
            "login",
            "logout",
            "get_current_user",
            "get_user_credits",
            "get_credit_balance",
            "get_credit_ledger",
            "list_resources",
            "list_environments",
            "environment_exists",
            "list_environment_entrypoints",
            "validate_environment",
            "delete_environment",
            "duplicate_environment_version",
            "submit_experiment",
            "quote_experiment",
            "list_experiments",
            "resume_experiment",
            "quote_resume",
            "stop_experiment",
            "list_checkpoints",
            "get_experiment_status",
            "iter_experiment_logs",
            "download_experiment_metrics",
            "preview_experiment_metrics_csv",
            "deploy_agent",
            "update_inference_deployment",
            "delete_inference_deployment",
            "list_inference_deployments",
            "_ensure_inference_binding",
            "open_inference_agent",
            "list_projects",
            "create_project",
            "delete_project",
            "set_default_project",
            "get_default_project",
            "list_datasets",
            "dataset_exists",
            "create_dataset",
            "delete_dataset",
            "list_models",
            "get_model_info",
        ]
    )


@contextmanager
def _patched_arena_client(mock_client: MagicMock):
    """Patch build_client so arena_client yields our mock."""
    with patch("agilerl.config.build_client", return_value=mock_client):
        yield


class TestCommandConfig:
    def test_creates_dataclass(self):
        cfg = CommandConfig(
            api_key="key",
            base_url="http://localhost",
            keycloak_url="http://kc",
            realm="test",
            client_id="cli",
            request_timeout=10,
            upload_timeout=60,
        )
        assert cfg.api_key == "key"
        assert cfg.base_url == "http://localhost"
        assert cfg.keycloak_url == "http://kc"
        assert cfg.realm == "test"
        assert cfg.client_id == "cli"
        assert cfg.request_timeout == 10
        assert cfg.upload_timeout == 60

    def test_slots(self):
        cfg = CommandConfig(
            api_key=None,
            base_url=None,
            keycloak_url=None,
            realm=None,
            client_id=None,
            request_timeout=30,
            upload_timeout=300,
        )
        with pytest.raises(AttributeError):
            cfg.nonexistent = "x"  # type: ignore[attr-defined]

    @patch("agilerl.config.build_client")
    def test_build_client_calls_configure_and_init(self, mock_build):
        with patch("agilerl.config.ArenaClient") as MockArenaClient:
            MockArenaClient.configure = MagicMock()
            cfg = CommandConfig(
                api_key="pat_123",
                base_url="http://api",
                keycloak_url="http://kc",
                realm="myrealm",
                client_id="mycli",
                request_timeout=5,
                upload_timeout=120,
            )
            build_client(cfg, MockArenaClient)
            MockArenaClient.configure.assert_called_once_with(
                base_url="http://api",
                keycloak_url="http://kc",
                realm="myrealm",
                client_id="mycli",
            )
            MockArenaClient.assert_called_once_with(
                api_key="pat_123",
                org_key=None,
                external_user_id=None,
                request_timeout=5,
                upload_timeout=120,
            )


class TestResolveRootCommandConfig:
    def test_returns_existing_command_config_unchanged(self):
        from agilerl.config import _resolve_root_command_config

        cfg = CommandConfig(
            api_key="key",
            base_url="http://api",
            keycloak_url="http://kc",
            realm="realm",
            client_id="cli",
            request_timeout=10,
            upload_timeout=60,
        )
        ctx = click.Context(click.Command("root"))
        ctx.obj = cfg
        assert _resolve_root_command_config(ctx) is cfg

    def test_non_dict_params_fall_back_to_defaults(self):
        """When ctx.obj is unset and ctx.params is not a dict, defaults are used."""
        from agilerl.config import _resolve_root_command_config

        ctx = click.Context(click.Command("root"))
        ctx.obj = None
        ctx.params = None  # type: ignore[assignment]

        cfg = _resolve_root_command_config(ctx)
        assert isinstance(cfg, CommandConfig)
        assert cfg.api_key is None
        assert cfg.org_key is None
        assert cfg.external_user_id is None
        assert cfg.base_url is None
        assert cfg.request_timeout == 30
        assert cfg.upload_timeout == 300


class TestArenaClientContextManager:
    def test_yields_client_and_closes(self, mock_client):
        cfg = CommandConfig(
            api_key=None,
            base_url=None,
            keycloak_url=None,
            realm=None,
            client_id=None,
            request_timeout=30,
            upload_timeout=300,
        )
        with patch("agilerl.config.build_client", return_value=mock_client):
            with arena_client(cfg) as client:
                assert client is mock_client
        mock_client.close.assert_called_once()

    def test_handles_arena_error(self, mock_client):
        cfg = CommandConfig(
            api_key=None,
            base_url=None,
            keycloak_url=None,
            realm=None,
            client_id=None,
            request_timeout=30,
            upload_timeout=300,
        )
        mock_client.get_current_user.side_effect = ArenaAPIError(
            "internal server error", status_code=500
        )
        with patch("agilerl.config.build_client", return_value=mock_client):
            with patch("agilerl.config.handle_error") as mock_handle:
                with arena_client(cfg) as client:
                    client.get_current_user()
                mock_handle.assert_called_once()
        mock_client.close.assert_called_once()

    def test_reraises_non_arena_error(self, mock_client):
        cfg = CommandConfig(
            api_key=None,
            base_url=None,
            keycloak_url=None,
            realm=None,
            client_id=None,
            request_timeout=30,
            upload_timeout=300,
        )
        with patch("agilerl.config.build_client", return_value=mock_client):

            def _raise_unexpected():
                msg = "unexpected"
                with arena_client(cfg):
                    raise RuntimeError(msg)

            with pytest.raises(RuntimeError, match="unexpected"):
                _raise_unexpected()
        mock_client.close.assert_called_once()


class TestMainGroup:
    def test_help(self, runner):
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Arena CLI" in result.output

    def test_sets_config_on_context(self, runner):
        @main.command("_test_ctx")
        @click.pass_obj
        def _test_ctx(config):
            click.echo(f"timeout={config.request_timeout}")

        with patch("agilerl.config.build_client"):
            result = runner.invoke(main, ["--request-timeout", "42", "_test_ctx"])
        assert "timeout=42" in result.output


class FakeEntryPoint:
    def __init__(self, register):
        self._register = register

    def load(self):
        return self._register


class TestLoadCliPlugins:
    def test_registers_plugin_commands_on_the_group(self, runner, monkeypatch):
        # Arrange
        def register(group: click.Group) -> None:
            @group.command("hello")
            def hello() -> None:
                click.echo("hello from plugin")

        requested = []

        def fake_entry_points(*, group):
            requested.append(group)
            return [FakeEntryPoint(register)]

        monkeypatch.setattr("agilerl.cli.entry_points", fake_entry_points)
        group = click.Group("arena")

        # Act
        load_cli_plugins(group)
        result = runner.invoke(group, ["hello"])

        # Assert
        assert requested == [CLI_PLUGIN_GROUP]
        assert result.exit_code == 0
        assert result.output == "hello from plugin\n"

    def test_no_plugins_leaves_the_group_unchanged(self, monkeypatch):
        monkeypatch.setattr("agilerl.cli.entry_points", lambda *, group: [])
        group = click.Group("arena")

        load_cli_plugins(group)

        assert group.commands == {}

    def test_plugin_error_propagates(self, monkeypatch):
        def broken(group: click.Group) -> None:
            msg = "plugin failed to register"
            raise RuntimeError(msg)

        monkeypatch.setattr(
            "agilerl.cli.entry_points",
            lambda *, group: [FakeEntryPoint(broken)],
        )

        with pytest.raises(RuntimeError, match="plugin failed to register"):
            load_cli_plugins(click.Group("arena"))


class TestManifestCommands:
    def test_validate_reports_a_valid_file(self, runner, tmp_path):
        path = tmp_path / "grpo.yaml"
        path.write_text(_llm_manifest("Qwen/Qwen3.5-0.8B"), encoding="utf-8")

        result = runner.invoke(main, ["manifest", "validate", str(path)])

        assert result.exit_code == 0
        assert "valid (GRPO)" in result.output

    def test_validate_rejects_classic_rl(self, runner):
        result = runner.invoke(
            main,
            ["manifest", "validate", "-"],
            input="algorithm: {name: DQN}\nenvironment: {name: CartPole-v1}\n",
        )

        assert result.exit_code == 1
        assert (
            "algorithm.name: Classic RL is no longer supported in Arena"
            in result.output
        )

    def test_validate_json_reports_an_invalid_manifest(self, runner):
        result = runner.invoke(
            main,
            ["manifest", "validate", "-", "--json"],
            input="algorithm: {name: NOPE}\nenvironment: {env_type: rollout}\n",
        )

        assert result.exit_code == 0
        assert '"ok": false' in result.output

    def test_validate_prints_errors_without_json(self, runner):
        result = runner.invoke(
            main,
            ["manifest", "validate", "-"],
            input="algorithm: {name: NOPE}\nenvironment: {env_type: rollout}\n",
        )

        assert result.exit_code == 1
        assert "not a registered algorithm" in result.output

    def test_validate_rejects_hpo(self, runner):
        result = runner.invoke(
            main,
            ["manifest", "validate", "-"],
            input=_llm_manifest("Qwen/Qwen3.5-0.8B") + "training: {hpo: true}\n",
        )

        assert result.exit_code == 1
        assert (
            "training.hpo: HPO and populations aren't available for LLM training"
            in result.output
        )

    def test_validate_rejects_an_llm_population(self, runner):
        result = runner.invoke(
            main,
            ["manifest", "validate", "-"],
            input=(
                "algorithm: {name: GRPO, group_size: 4, batch_size: 2}\n"
                "environment: {env_type: rollout, dataset: openai/gsm8k, "
                "reward_file_path: reward.py, prompt_template: {user_0: '{question}'}}\n"
                "network: {pretrained_model_name_or_path: Qwen/Qwen2.5-0.5B-Instruct, "
                "max_context_length: 512}\n"
                "training: {max_steps: 100, pop_size: 2}\n"
            ),
        )

        assert result.exit_code == 1
        assert (
            "training.pop_size: HPO and populations aren't available for LLM training"
            in result.output
        )

    @pytest.mark.parametrize(
        ("extra", "error"),
        [
            (
                "algorithm: {name: GRPO, batch_size: 6}\n",
                "algorithm.batch_size: must be a power of 2 between 1 and 1024.",
            ),
            (
                "algorithm: {name: GRPO}\nreplay_buffer: {kind: classic}\n",
                "replay_buffer.kind must be 'llm', got 'classic'",
            ),
            (
                "algorithm: {name: GRPO}\ntraining: {rollout_mode: colocated}\n",
                "training.rollout_mode: colocated runs only locally",
            ),
        ],
        ids=["batch-size", "classic-buffer", "colocated"],
    )
    def test_validate_rejects_what_submit_rejects(self, runner, extra, error):
        manifest = extra + (
            "environment: {env_type: rollout, entrypoint: 'envs:MyEnv'}\n"
            "network: {pretrained_model_name_or_path: Qwen/Qwen3.5-0.8B, "
            "max_context_length: 1024}\n"
        )

        result = runner.invoke(main, ["manifest", "validate", "-"], input=manifest)

        assert result.exit_code == 1
        assert error in " ".join(result.output.split())

    def test_validate_lists_each_registered_algorithm(self, runner):
        result = runner.invoke(
            main,
            ["manifest", "validate", "-"],
            input="algorithm: {name: NOPE}\nenvironment: {env_type: rollout}\n",
        )

        assert (
            "Registered: CISPO, DPO, GRPO, GSPO, LLMPPO, LLMREINFORCE, OPD, SFT."
            in " ".join(result.output.split())
        )

    def test_validate_suggests_the_closest_algorithm(self, runner):
        result = runner.invoke(
            main,
            ["manifest", "validate", "-"],
            input="algorithm: {name: grp0}\nenvironment: {env_type: rollout}\n",
        )

        assert result.exit_code == 1
        assert "Did you mean GRPO?" in result.output

    def test_validate_refuses_a_deprecated_model(self, runner):
        result = runner.invoke(
            main, ["manifest", "validate", "-"], input=_llm_manifest("Qwen/Qwen3-4B")
        )

        assert result.exit_code == 1
        assert (
            "network.pretrained_model_name_or_path: Model Qwen/Qwen3-4B is "
            "deprecated and can no longer be submitted."
            in " ".join(result.output.split())
        )
        assert "valid (GRPO)" not in result.output

    def test_validate_refuses_a_hub_id_outside_the_catalog(self, runner):
        result = runner.invoke(
            main,
            ["manifest", "validate", "-"],
            input=_llm_manifest("Qwen/Qwen3.5-0.9B"),
        )

        assert result.exit_code == 1
        assert (
            "network.pretrained_model_name_or_path: Arena does not support the "
            "model 'Qwen/Qwen3.5-0.9B'. Did you mean Qwen/Qwen3.5-0.8B?"
            in " ".join(result.output.split())
        )

    def test_validate_refuses_a_preview_model(self, runner, monkeypatch):
        hub_id = "Qwen/Qwen3.5-0.8B"
        monkeypatch.setitem(
            SUPPORTED_MODEL_INFO,
            hub_id,
            SUPPORTED_MODEL_INFO[hub_id].model_copy(update={"status": STATUS_PREVIEW}),
        )

        result = runner.invoke(
            main, ["manifest", "validate", "-"], input=_llm_manifest(hub_id)
        )

        assert result.exit_code == 1
        assert f"Model {hub_id} is in preview" in result.output

    @pytest.mark.parametrize(
        "manifest",
        [
            _llm_manifest("Qwen/Qwen3.5-0.8B"),
            _llm_manifest("./checkpoints/mine"),
            _llm_manifest("/mnt/checkpoints/mine"),
        ],
        ids=["live", "relative-path", "absolute-path"],
    )
    def test_validate_json_has_no_warning_to_report(self, runner, manifest):
        result = runner.invoke(
            main, ["manifest", "validate", "-", "--json"], input=manifest
        )

        assert result.exit_code == 0
        verdict = json.loads(result.output)
        assert verdict["ok"] is True
        assert verdict["warnings"] == []

    def test_validate_json_accepts_a_relative_checkout(
        self, runner, tmp_path, monkeypatch
    ):
        checkout = tmp_path / "ckpt"
        checkout.mkdir()
        (checkout / "config.json").write_text("{}", encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(
            main,
            ["manifest", "validate", "-", "--json"],
            input=_llm_manifest("ckpt"),
        )

        assert result.exit_code == 0
        verdict = json.loads(result.output)
        assert verdict["ok"] is True
        assert verdict["warnings"] == []

    def test_validate_does_not_suggest_a_hub_id_for_a_local_checkout(
        self, runner, tmp_path, monkeypatch
    ):
        checkout = tmp_path / "models" / "Qwen" / "Qwen3.5-0.8B"
        checkout.mkdir(parents=True)
        (checkout / "config.json").write_text("{}", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        model = "models/Qwen/Qwen3.5-0.8B"

        result = runner.invoke(
            main, ["manifest", "validate", "-"], input=_llm_manifest(model)
        )

        assert result.exit_code == 0
        assert "not in the Arena supported model catalog" not in result.output
        assert "Did you mean" not in result.output
        assert "valid (GRPO)" in result.output

    def test_validate_warns_when_a_relative_checkout_is_missing(
        self, runner, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(
            main, ["manifest", "validate", "-"], input=_llm_manifest("ckpt")
        )

        assert result.exit_code == 0
        assert "Model ckpt is not in the Arena supported model catalog" in result.output

    def test_validate_json_reports_a_deprecated_model_as_an_error(self, runner):
        result = runner.invoke(
            main,
            ["manifest", "validate", "-", "--json"],
            input=_llm_manifest("Qwen/Qwen3-4B"),
        )

        assert json.loads(result.output)["errors"] == [
            {
                "path": "network.pretrained_model_name_or_path",
                "message": (
                    "Model Qwen/Qwen3-4B is deprecated and can no longer be submitted."
                ),
                "kind": "model_status",
            }
        ]

    def test_validate_warns_that_lora_target_modules_are_ignored(self, runner):
        manifest = (
            "algorithm: {name: GRPO}\n"
            "environment: {env_type: rollout, entrypoint: 'envs:MyEnv'}\n"
            "network:\n"
            "  pretrained_model_name_or_path: Qwen/Qwen3.5-0.8B\n"
            "  max_context_length: 1024\n"
            "  lora_config: {target_modules: [q_proj, k_proj, v_proj, o_proj]}\n"
        )

        result = runner.invoke(main, ["manifest", "validate", "-"], input=manifest)

        assert result.exit_code == 0
        assert "warning: lora_config.target_modules is ignored" in result.output
        assert result.output.count("target_modules is ignored") == 1

    def test_validate_json_invalid_verdict_has_the_same_keys(self, runner):
        result = runner.invoke(
            main,
            ["manifest", "validate", "-", "--json"],
            input="algorithm: {name: NOPE}\nenvironment: {env_type: rollout}\n",
        )

        verdict = json.loads(result.output)
        assert verdict["ok"] is False
        assert verdict["warnings"] == []

    def test_root_json_flag_applies_to_validate(self, runner):
        result = runner.invoke(
            main,
            ["--json", "manifest", "validate", "-"],
            input=_llm_manifest("Qwen/Qwen3.5-0.8B"),
        )

        assert json.loads(result.output)["ok"] is True

    def test_validate_payload_prints_resolved_manifest(self, runner, tmp_path):
        path = tmp_path / "grpo.yaml"
        path.write_text(_llm_manifest("Qwen/Qwen3.5-0.8B"), encoding="utf-8")

        result = runner.invoke(main, ["manifest", "validate", str(path), "--payload"])

        assert result.exit_code == 0
        assert '"name": "GRPO"' in result.output

    def test_validate_unreadable_file_exits_failed(self, runner):
        result = runner.invoke(main, ["manifest", "validate", "/does/not/exist.yaml"])

        assert result.exit_code == 2
        assert "could not read" in result.output

    def test_schema_prints_json(self, runner):
        result = runner.invoke(main, ["manifest", "schema"])

        assert result.exit_code == 0
        assert '"title": "AgileRL training manifest"' in result.output

    def test_module_entrypoint_prints_help(self, monkeypatch, capsys):
        import runpy
        import sys

        from agilerl import cli as cli_mod

        monkeypatch.setattr(sys, "argv", ["agilerl.cli", "--help"])
        with pytest.raises(SystemExit) as exc_info:
            runpy.run_path(cli_mod.__file__, run_name="__main__")

        assert exc_info.value.code == 0
        assert "Arena CLI" in capsys.readouterr().out


class TestLoginCommand:
    def test_login(self, runner, mock_client):
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["login"])
        assert result.exit_code == 0
        mock_client.login.assert_called_once_with(timeout=300)
        mock_client.close.assert_called_once()

    def test_login_rejects_force_flag(self, runner, mock_client):
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["login", "--force"])
        assert result.exit_code == 2
        assert "No such option" in result.output
        assert "--force" in result.output
        mock_client.login.assert_not_called()

    def test_login_custom_timeout(self, runner, mock_client):
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["login", "--timeout", "60"])
        assert result.exit_code == 0
        mock_client.login.assert_called_once_with(timeout=60)


class TestLogoutCommand:
    def test_logout(self, runner, mock_client):
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["logout"])
        assert result.exit_code == 0
        mock_client.logout.assert_called_once()
        mock_client.close.assert_called_once()


class TestUserCommands:
    def test_user_profile(self, runner, mock_client):
        mock_client.get_current_user.return_value = {
            "first_name": "Jane",
            "last_name": "Doe",
            "email": "jane@example.com",
        }
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["user", "profile"])
        assert result.exit_code == 0
        assert "Jane Doe" in result.output
        assert "jane@example.com" in result.output

    def test_user_profile_json(self, runner, mock_client):
        user = {
            "first_name": "Jane",
            "last_name": "Doe",
            "email": "jane@example.com",
        }
        mock_client.get_current_user.return_value = user

        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["--json", "user", "profile"])

        assert result.exit_code == 0
        assert json.loads(result.output) == user
        assert "Jane Doe" not in result.output


CREDIT_BALANCE = {
    "balance": 423.81,
    "accrued_cost": 16.8,
    "balance_after_accrued": 407.01,
    "accrued": [
        {
            "kind": "job",
            "id": 1507,
            "experiment_id": 1699,
            "name": "gsm8k-grpo-v5",
            "since": "2026-11-19T21:00:00",
            "hourly_cost": 252.03,
            "accrued_cost": 16.8,
        }
    ],
}

LEDGER = [
    {
        "time": "2026-11-19T21:18:55.617",
        "kind": "Job",
        "job_id": 1507,
        "deployment_id": None,
        "experiment_id": 1699,
        "item": "gsm8k-grpo-v5",
        "description": "job-1507 cost at 2026-11-19 20:00:00",
        "amount": -16.8,
        "balance": 423.81,
    }
]


def _assert_table_columns(output: str, columns: list[str]) -> None:
    header = next(
        line for line in output.splitlines() if all(name in line for name in columns)
    )
    positions = [header.index(name) for name in columns]
    assert positions == sorted(positions)


class TestUserCreditsCommand:
    def test_prints_balance_and_accrued_cost(self, runner, mock_client):
        # Arrange
        mock_client.get_credit_balance.return_value = CREDIT_BALANCE

        # Act
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["user", "credits"])

        # Assert
        assert result.exit_code == 0, result.output
        assert "$423.81" in result.output
        assert "Accrued, not charged:   $16.80" in result.output
        assert "$407.01" in result.output
        assert "gsm8k-grpo-v5" in result.output
        assert "2026-11-19 21:00 UTC" in result.output
        _assert_table_columns(
            result.output,
            ["kind", "id", "name", "since", "hourly_cost", "accrued_cost"],
        )
        assert "experiment_id" not in result.output
        assert "1699" not in result.output

    def test_json_keeps_the_bare_balance(self, runner, mock_client):
        mock_client.get_credit_balance.return_value = CREDIT_BALANCE

        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["--json", "user", "credits"])

        assert result.exit_code == 0, result.output
        assert json.loads(result.output) == 423.81

    def test_older_server_prints_the_plain_balance(self, runner, mock_client):
        # Arrange
        mock_client.get_credit_balance.side_effect = ArenaAPIError(
            "Not Found", status_code=404
        )
        mock_client.get_user_credits.return_value = 1000

        # Act
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["--json", "user", "credits"])

        # Assert
        assert result.exit_code == 0, result.output
        assert json.loads(result.output) == 1000

    def test_other_errors_are_not_hidden(self, runner, mock_client):
        mock_client.get_credit_balance.side_effect = ArenaAPIError(
            "boom", status_code=500
        )

        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["user", "credits"])

        assert result.exit_code != 0
        mock_client.get_user_credits.assert_not_called()

    def test_ledger_table(self, runner, mock_client):
        mock_client.get_credit_ledger.return_value = LEDGER

        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["user", "credits", "--ledger"])

        assert result.exit_code == 0, result.output
        assert "gsm8k-grpo-v5" in result.output
        assert "-$16.80" in result.output
        assert "$423.81" in result.output
        _assert_table_columns(
            result.output,
            ["time", "item", "kind", "amount", "balance", "description"],
        )
        assert "job_id" not in result.output
        assert "deployment_id" not in result.output
        assert "experiment_id" not in result.output
        assert "1699" not in result.output
        assert "None" not in result.output
        mock_client.get_credit_ledger.assert_called_once_with(
            job_id=None, deployment_id=None, limit=None
        )

    def test_ledger_filters_are_passed_through(self, runner, mock_client):
        mock_client.get_credit_ledger.return_value = LEDGER

        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main,
                [
                    "--json",
                    "user",
                    "credits",
                    "--ledger",
                    "--job",
                    "1507",
                    "--limit",
                    "5",
                ],
            )

        assert result.exit_code == 0, result.output
        assert json.loads(result.output) == LEDGER
        mock_client.get_credit_ledger.assert_called_once_with(
            job_id=1507, deployment_id=None, limit=5
        )

    def test_ledger_other_errors_are_not_hidden(self, runner, mock_client):
        mock_client.get_credit_ledger.side_effect = ArenaAPIError(
            "boom", status_code=500
        )

        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["user", "credits", "--ledger"])

        assert result.exit_code != 0
        assert "no credit ledger" not in result.output

    def test_ledger_on_older_server_says_so(self, runner, mock_client):
        mock_client.get_credit_ledger.side_effect = ArenaAPIError(
            "Not Found", status_code=404
        )

        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["user", "credits", "--ledger"])

        assert result.exit_code != 0
        assert "no credit ledger" in result.output

    def test_ledger_filters_need_ledger(self, runner, mock_client):
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["user", "credits", "--job", "1"])

        assert result.exit_code == 2
        assert "need --ledger" in result.output


class TestResourcesListCommand:
    def test_prints_one_row_per_gpu_type(self, runner, mock_client):
        # Arrange
        mock_client.list_resources.return_value = {
            "gpuTypes": [
                {
                    "gpuType": "h100-80",
                    "gpusPerNode": [8],
                    "numGpus": [8, 16, 32],
                    "devOnly": False,
                },
                {
                    "gpuType": "b200",
                    "gpusPerNode": [1, 8],
                    "numGpus": [1, 2, 4, 8],
                    "devOnly": False,
                },
            ]
        }

        # Act
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["resources", "list"])

        # Assert
        assert result.exit_code == 0
        assert "GPU type" in result.output
        assert "--num-gpus" in result.output
        assert "GPUs per node" in result.output
        h100_line = next(
            line for line in result.output.splitlines() if "h100-80" in line
        )
        b200_line = next(line for line in result.output.splitlines() if "b200" in line)
        assert "8, 16, 32" in h100_line
        assert "1, 2, 4, 8" in b200_line
        assert "1, 8" in b200_line
        assert "yes" not in result.output
        mock_client.list_resources.assert_called_once_with(algorithm=None)

    def test_passes_the_algorithm(self, runner, mock_client):
        mock_client.list_resources.return_value = {"gpuTypes": []}

        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main, ["resources", "list", "--algorithm", "SUPERVISED"]
            )

        assert result.exit_code == 0
        mock_client.list_resources.assert_called_once_with(algorithm="SUPERVISED")

    def test_marks_dev_only_gpu_types(self, runner, mock_client):
        mock_client.list_resources.return_value = {
            "gpuTypes": [
                {"gpuType": "l4", "gpusPerNode": [1], "devOnly": False},
                {"gpuType": "b200", "gpusPerNode": [8], "devOnly": True},
            ]
        }

        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["resources", "list"])

        b200_line = next(line for line in result.output.splitlines() if "b200" in line)
        l4_line = next(line for line in result.output.splitlines() if "l4" in line)
        assert "Dev only" in result.output
        assert "yes" in b200_line
        assert "yes" not in l4_line

    def test_json_emits_the_gpu_types(self, runner, mock_client):
        gpu_types = [
            {"gpuType": "h100-80", "gpusPerNode": [1, 8], "devOnly": False},
        ]
        mock_client.list_resources.return_value = {"gpuTypes": gpu_types}

        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["--json", "resources", "list"])

        assert result.exit_code == 0
        assert json.loads(result.output) == gpu_types

    def test_derives_gpu_types_from_tiers_on_an_older_platform(
        self, runner, mock_client
    ):
        # Arrange
        mock_client.list_resources.return_value = {
            "tiers": {
                "h100-8": {
                    "name": "a3-highgpu-8g",
                    "num_gpus": 8,
                    "gpu_type": "nvidia-h100-80gb",
                },
                "h100-1": {
                    "name": "a3-highgpu-1g",
                    "num_gpus": 1,
                    "gpu_type": "nvidia-h100-80gb",
                },
                "l4-1": {
                    "name": "g2-standard-8",
                    "num_gpus": 1,
                    "gpu_type": "nvidia-l4",
                },
                "cpu": {"name": "n2-standard-8", "num_gpus": 0, "gpu_type": None},
            }
        }

        # Act
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["--json", "resources", "list"])

        # Assert
        assert result.exit_code == 0
        assert json.loads(result.output) == [
            {"gpuType": "h100-80", "gpusPerNode": [1, 8], "devOnly": False},
            {"gpuType": "l4", "gpusPerNode": [1], "devOnly": False},
        ]

    def test_never_prints_tier_names(self, runner, mock_client):
        mock_client.list_resources.return_value = {
            "tiers": {
                "h100-8": {
                    "name": "a3-highgpu-8g",
                    "num_gpus": 8,
                    "gpu_type": "nvidia-h100-80gb",
                },
            }
        }

        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["resources", "list"])

        assert result.exit_code == 0
        assert "h100-80" in result.output
        assert "a3-highgpu-8g" not in result.output


class TestEnvListCommand:
    def test_env_list_default(self, runner, mock_client):
        mock_client.list_environments.return_value = {
            "my-env": {"v1": {"validated": True, "profiled": False}}
        }
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["env", "list"])
        assert result.exit_code == 0
        mock_client.list_environments.assert_called_once_with(
            name=None, include_arena=False
        )

    def test_env_list_with_name(self, runner, mock_client):
        mock_client.list_environments.return_value = {}
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["env", "list", "--name", "foo"])
        assert result.exit_code == 0
        mock_client.list_environments.assert_called_once_with(
            name="foo", include_arena=False
        )

    def test_env_list_include_arena(self, runner, mock_client):
        mock_client.list_environments.return_value = {}
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["env", "list", "--include-arena"])
        assert result.exit_code == 0
        mock_client.list_environments.assert_called_once_with(
            name=None, include_arena=True
        )


class TestEnvExistsCommand:
    def test_env_exists(self, runner, mock_client):
        mock_client.environment_exists.return_value = {"exists": True}
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["env", "exists", "my-env"])
        assert result.exit_code == 0
        mock_client.environment_exists.assert_called_once_with(
            name="my-env", version=None
        )

    def test_env_exists_with_version(self, runner, mock_client):
        mock_client.environment_exists.return_value = {"exists": True}
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["env", "exists", "my-env", "--version", "v2"])
        assert result.exit_code == 0
        mock_client.environment_exists.assert_called_once_with(
            name="my-env", version="v2"
        )


class TestEnvEntrypointsCommand:
    def test_entrypoints(self, runner, mock_client):
        mock_client.list_environment_entrypoints.return_value = [
            "main:MyEnv",
            "alt:AltEnv",
        ]
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["env", "entrypoints", "my-env"])
        assert result.exit_code == 0
        mock_client.list_environment_entrypoints.assert_called_once_with(
            name="my-env", version=None
        )


class TestEnvValidateCommand:
    def test_validate_by_name(self, runner, mock_client):
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["env", "validate", "my-env"])
        assert result.exit_code == 0
        mock_client.validate_environment.assert_called_once_with(
            name="my-env",
            version=None,
            source=None,
            env_config=None,
            requirements=None,
            entrypoint=None,
            description=None,
            do_rollouts=False,
        )

    def test_validate_with_source(self, runner, mock_client, tmp_path):
        src_dir = tmp_path / "envdir"
        src_dir.mkdir()
        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main, ["env", "validate", "my-env", "--source", str(src_dir)]
            )
        assert result.exit_code == 0
        mock_client.validate_environment.assert_called_once()
        call_kwargs = mock_client.validate_environment.call_args.kwargs
        assert call_kwargs["name"] == "my-env"
        assert call_kwargs["source"] == src_dir

    def test_validate_source_without_name_errors(self, runner, mock_client, tmp_path):
        # The name is never inferred from --source; it must be passed explicitly.
        src_dir = tmp_path / "envdir"
        src_dir.mkdir()
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["env", "validate", "--source", str(src_dir)])
        assert result.exit_code != 0
        assert "Missing argument" in result.output
        mock_client.validate_environment.assert_not_called()

    def test_validate_no_name_errors(self, runner, mock_client):
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["env", "validate"])
        assert result.exit_code != 0
        assert "Missing argument" in result.output
        mock_client.validate_environment.assert_not_called()

    def test_validate_with_rollouts(self, runner, mock_client):
        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main,
                ["env", "validate", "my-env", "--do-rollouts"],
            )
        assert result.exit_code == 0
        call_kwargs = mock_client.validate_environment.call_args.kwargs
        assert call_kwargs["do_rollouts"] is True

    def test_passed_checks_exit_zero(self, runner, mock_client):
        # Arrange
        mock_client.validate_environment.return_value = {"cpu_per_env": 0.1}

        # Act
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["--json", "env", "validate", "my-env"])

        # Assert
        assert result.exit_code == 0
        assert json.loads(result.output) == {"cpu_per_env": 0.1}

    @pytest.mark.parametrize("error_code", ["VALIDATION_FAILED", "VALIDATOR_ERROR"])
    def test_failed_checks_exit_non_zero(self, runner, mock_client, error_code):
        # Arrange
        failed = {"error_code": error_code}
        mock_client.validate_environment.return_value = failed

        # Act
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["--json", "env", "validate", "my-env"])

        # Assert
        assert result.exit_code == 1
        assert json.loads(result.output) == failed

    @pytest.mark.parametrize("flag", ["--multi-agent", "--language-based", "--classic"])
    def test_rejects_removed_classic_flags(self, runner, mock_client, flag):
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["env", "validate", "my-env", flag])

        assert result.exit_code == 2
        assert f"No such option '{flag}'" in result.output
        mock_client.validate_environment.assert_not_called()


class TestEnvDeleteCommand:
    def test_delete_confirmed(self, runner, mock_client):
        mock_client.delete_environment.return_value = ""
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["env", "delete", "my-env", "--yes"])
        assert result.exit_code == 0
        mock_client.delete_environment.assert_called_once_with(
            name="my-env", version=None, confirm=True
        )

    def test_delete_aborted(self, runner, mock_client):
        mock_client.delete_environment.return_value = None
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["env", "delete", "my-env"])
        assert result.exit_code == 0
        assert "deleted successfully" not in result.output
        mock_client.delete_environment.assert_called_once_with(
            name="my-env", version=None, confirm=False
        )

    def test_delete_with_version(self, runner, mock_client):
        mock_client.delete_environment.return_value = None
        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main, ["env", "delete", "my-env", "--version", "v2", "--yes"]
            )
        assert result.exit_code == 0
        mock_client.delete_environment.assert_called_once_with(
            name="my-env", version="v2", confirm=True
        )

    def test_delete_non_empty_result(self, runner, mock_client):
        mock_client.delete_environment.return_value = {"status": "pending"}
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["env", "delete", "my-env", "--yes"])
        assert result.exit_code == 0


class TestEnvDuplicateCommand:
    def test_duplicate(self, runner, mock_client):
        mock_client.duplicate_environment_version.return_value = {"ok": True}
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["env", "duplicate", "my-env", "v2-copy"])
        assert result.exit_code == 0
        mock_client.duplicate_environment_version.assert_called_once_with(
            name="my-env", new_version="v2-copy", version=None
        )

    def test_duplicate_with_version(self, runner, mock_client):
        mock_client.duplicate_environment_version.return_value = {"ok": True}
        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main,
                ["env", "duplicate", "my-env", "v2-copy", "--version", "v1"],
            )
        assert result.exit_code == 0
        mock_client.duplicate_environment_version.assert_called_once_with(
            name="my-env", new_version="v2-copy", version="v1"
        )


SUBMIT_QUOTE = {
    "resource_id": "h100-80-8x",
    "resource_name": "NVIDIA H100 80GB x8",
    "gpu_type": "h100-80",
    "num_gpus": 8,
    "num_nodes": 1,
    "hourly_cost": 25.6,
    "max_steps": 1000,
    "step_seconds": 36.0,
    "estimated_hours": 11.25,
    "estimated_cost": 288.0,
    "startup_minutes": 13.0,
    "startup_cost": 5.5466,
}


class TestExperimentSubmitCommand:
    @pytest.fixture(autouse=True)
    def quoted(self, mock_client):
        mock_client.quote_experiment.return_value = dict(SUBMIT_QUOTE)

    def test_submit_help_still_requires_a_reward_file(self, runner):
        result = runner.invoke(main, ["experiments", "submit", "--help"])
        help_text = " ".join(result.output.split())

        assert result.exit_code == 0
        assert "Required when the manifest targets a reasoning dataset." in help_text
        assert "OPD" not in help_text

    def test_submit(self, runner, mock_client, tmp_path):
        manifest = tmp_path / "manifest.yaml"
        manifest.write_text("experiment: test")
        mock_client.submit_experiment.return_value = {"id": "exp-123"}
        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main, ["experiments", "submit", "--yes", str(manifest)]
            )
        assert result.exit_code == 0
        mock_client.submit_experiment.assert_called_once_with(
            manifest=manifest,
            num_gpus=None,
            gpu_type=None,
            project=None,
            experiment_name=None,
            reward_file=None,
            completion=None,
            idempotency_key=None,
        )

    def test_submit_all_options(self, runner, mock_client, tmp_path):
        manifest = tmp_path / "m.yaml"
        manifest.write_text("x: 1")
        reward = tmp_path / "reward.py"
        reward.write_text("def reward(q, a, c):\n    return 0.0\n")
        mock_client.submit_experiment.return_value = {"id": "exp-456"}
        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main,
                [
                    "experiments",
                    "submit",
                    "--yes",
                    str(manifest),
                    "--num-gpus",
                    "4",
                    "--gpu-type",
                    "l4",
                    "--project",
                    "proj1",
                    "--experiment-name",
                    "my-exp",
                    "--reward-file",
                    str(reward),
                    "--completion",
                    "test output",
                ],
            )
        assert result.exit_code == 0
        mock_client.submit_experiment.assert_called_once_with(
            manifest=manifest,
            num_gpus=4,
            gpu_type="l4",
            project="proj1",
            experiment_name="my-exp",
            reward_file=reward,
            completion="test output",
            idempotency_key=None,
        )

    def test_submit_idempotency_key(self, runner, mock_client, tmp_path):
        manifest = tmp_path / "manifest.yaml"
        manifest.write_text("experiment: test")
        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main,
                [
                    "experiments",
                    "submit",
                    "--yes",
                    str(manifest),
                    "--idempotency-key",
                    "run-1",
                ],
            )
        assert result.exit_code == 0
        assert (
            mock_client.submit_experiment.call_args.kwargs["idempotency_key"] == "run-1"
        )

    def test_submit_gpu_flags(self, runner, mock_client, tmp_path):
        manifest = tmp_path / "m.yaml"
        manifest.write_text("x: 1")
        mock_client.submit_experiment.return_value = {"id": "exp-789"}
        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main,
                [
                    "experiments",
                    "submit",
                    "--yes",
                    str(manifest),
                    "--num-gpus",
                    "8",
                    "--gpu-type",
                    "h100-80",
                ],
            )
        assert result.exit_code == 0
        mock_client.submit_experiment.assert_called_once_with(
            manifest=manifest,
            num_gpus=8,
            gpu_type="h100-80",
            project=None,
            experiment_name=None,
            reward_file=None,
            completion=None,
            idempotency_key=None,
        )

    def test_submit_rejects_zero_gpus(self, runner, mock_client, tmp_path):
        manifest = tmp_path / "m.yaml"
        manifest.write_text("x: 1")
        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main, ["experiments", "submit", str(manifest), "--num-gpus", "0"]
            )
        assert result.exit_code == 2
        assert "Invalid value" in result.output
        mock_client.submit_experiment.assert_not_called()

    def test_submit_missing_manifest(self, runner, mock_client):
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["experiments", "submit"])
        assert result.exit_code != 0

    def test_help_offers_only_gpu_flags(self, runner):
        result = runner.invoke(main, ["experiments", "submit", "--help"])

        help_text = " ".join(result.output.split())
        assert result.exit_code == 0
        assert "--resource-id" not in help_text
        assert "--num-nodes" not in help_text
        assert "--num-gpus" in help_text
        assert "--gpu-type" in help_text
        assert "Omit both GPU flags to let Arena pick the cheapest fit." in help_text

    @pytest.mark.parametrize("flag", ["--resource-id", "--num-nodes"])
    def test_submit_rejects_a_resource_class_flag(
        self, runner, mock_client, tmp_path, flag
    ):
        manifest = tmp_path / "m.yaml"
        manifest.write_text("x: 1")

        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main, ["experiments", "submit", str(manifest), flag, "2"]
            )

        assert result.exit_code == 2
        assert f"No such option '{flag}'" in result.output
        mock_client.submit_experiment.assert_not_called()

    def test_root_json_prints_the_submit_result(self, runner, mock_client, tmp_path):
        manifest = tmp_path / "m.yaml"
        manifest.write_text("algorithm: {name: GRPO}\n")
        mock_client.submit_experiment.return_value = {"job_id": 3}

        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main, ["--json", "experiments", "submit", "--yes", str(manifest)]
            )

        assert result.exit_code == 0
        assert json.loads(result.stdout) == {"job_id": 3}

    def test_arena_json_env_var_turns_json_on(self, runner, mock_client, tmp_path):
        manifest = tmp_path / "m.yaml"
        manifest.write_text("algorithm: {name: GRPO}\n")
        mock_client.submit_experiment.return_value = {"job_id": 3}

        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main,
                ["experiments", "submit", "--yes", str(manifest)],
                env={"ARENA_JSON": "1"},
            )

        assert json.loads(result.stdout) == {"job_id": 3}

    def test_without_json_submit_prints_no_result_document(
        self, runner, mock_client, tmp_path
    ):
        manifest = tmp_path / "m.yaml"
        manifest.write_text("algorithm: {name: GRPO}\n")
        mock_client.submit_experiment.return_value = {"job_id": 3}

        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["experiments", "submit", str(manifest)])

        assert "job_id" not in result.output


REWARD_ERROR = (
    "Reward function validation did not pass: ValueError: boom (line 2, in reward_fn)"
)


class TestExperimentSubmitPricePreview:
    @pytest.fixture
    def manifest(self, tmp_path):
        path = tmp_path / "m.yaml"
        path.write_text("algorithm: {name: GRPO}\n")
        return path

    @pytest.fixture(autouse=True)
    def quoted(self, mock_client):
        mock_client.quote_experiment.return_value = dict(SUBMIT_QUOTE)
        mock_client.submit_experiment.return_value = {"job_id": 3}

    @staticmethod
    def interactive(answer: bool = True):
        return patch("agilerl.submit_quote.stdin_is_a_terminal", return_value=answer)

    def test_prints_hardware_rate_and_total_before_submitting(
        self, runner, mock_client, manifest
    ):
        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main,
                [
                    "experiments",
                    "submit",
                    "--yes",
                    str(manifest),
                    "--num-gpus",
                    "8",
                    "--gpu-type",
                    "h100-80",
                ],
            )

        assert result.exit_code == 0
        assert (
            "Hardware:        8x h100-80 on 1 node (NVIDIA H100 80GB x8, h100-80-8x)"
            in result.stderr
        )
        assert "Hourly rate:     $25.60 per hour" in result.stderr
        assert (
            "Startup:         ~13 min ($5.55) before training starts" in result.stderr
        )
        assert (
            "Estimated total: $288.00 for about 11.2 h (1,000 steps)" in result.stderr
        )
        mock_client.quote_experiment.assert_called_once_with(
            manifest=manifest,
            num_gpus=8,
            gpu_type="h100-80",
            project=None,
            experiment_name=None,
            reward_file=None,
            completion=None,
        )
        mock_client.submit_experiment.assert_called_once()

    def test_says_when_there_is_no_duration_estimate(
        self, runner, mock_client, manifest
    ):
        mock_client.quote_experiment.return_value = {
            **SUBMIT_QUOTE,
            "max_steps": None,
            "estimated_hours": None,
            "estimated_cost": None,
        }

        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main, ["experiments", "submit", "--yes", str(manifest)]
            )

        assert result.exit_code == 0
        assert "Hourly rate:     $25.60 per hour" in result.stderr
        assert "Estimated total: no duration estimate" in result.stderr

    def test_declining_stops_before_anything_is_billed(
        self, runner, mock_client, manifest
    ):
        # Arrange
        # cli.click.confirm is click.confirm; keep the real function.
        real_confirm = click.confirm

        def confirm_with_stdout_captured(*args, **kwargs):
            assert supports_interactive_selection() is False
            return real_confirm(*args, **kwargs)

        # Act
        with (
            _patched_arena_client(mock_client),
            self.interactive(),
            patch(
                "agilerl.cli.click.confirm",
                side_effect=confirm_with_stdout_captured,
            ),
        ):
            result = runner.invoke(
                main, ["experiments", "submit", str(manifest)], input="n\n"
            )

        # Assert
        assert result.exit_code == 1
        assert "Submit and start billing?" in result.stderr
        mock_client.submit_experiment.assert_not_called()

    def test_accepting_submits(self, runner, mock_client, manifest):
        with _patched_arena_client(mock_client), self.interactive():
            result = runner.invoke(
                main, ["experiments", "submit", str(manifest)], input="y\n"
            )

        assert result.exit_code == 0
        mock_client.submit_experiment.assert_called_once()

    @pytest.mark.parametrize("flag", ["--yes", "-y"])
    def test_yes_submits_without_asking(self, runner, mock_client, manifest, flag):
        with _patched_arena_client(mock_client), self.interactive():
            result = runner.invoke(main, ["experiments", "submit", str(manifest), flag])

        assert result.exit_code == 0
        assert "Hourly rate:" in result.stderr
        assert "Submit and start billing?" not in result.stderr
        mock_client.submit_experiment.assert_called_once()

    def test_non_interactive_without_yes_refuses(self, runner, mock_client, manifest):
        with _patched_arena_client(mock_client), self.interactive(answer=False):
            result = runner.invoke(main, ["experiments", "submit", str(manifest)])

        assert result.exit_code == 2
        assert "Pass --yes to submit anyway." in result.stderr
        mock_client.submit_experiment.assert_not_called()

    def test_dry_run_prints_the_price_and_submits_nothing(
        self, runner, mock_client, manifest
    ):
        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main, ["experiments", "submit", str(manifest), "--dry-run"]
            )

        assert result.exit_code == 0
        assert "Hourly rate:" in result.output
        assert "Startup:         ~13 min ($5.55) before training starts" in (
            result.output
        )
        assert "Estimated total:" in result.output
        mock_client.submit_experiment.assert_not_called()

    def test_a_server_without_a_startup_estimate_prints_no_startup_line(
        self, runner, mock_client, manifest
    ):
        quote = {
            key: value
            for key, value in SUBMIT_QUOTE.items()
            if key not in ("startup_minutes", "startup_cost")
        }
        mock_client.quote_experiment.return_value = quote

        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main, ["experiments", "submit", "--yes", str(manifest)]
            )

        assert result.exit_code == 0
        assert "Hourly rate:     $25.60 per hour" in result.stderr
        assert "Startup:" not in result.stderr

    @pytest.mark.parametrize("flag", ["--yes", "--dry-run"])
    def test_1x_training_keys_print_one_warning_line(
        self, runner, mock_client, tmp_path, flag
    ):
        manifest = tmp_path / "m.yaml"
        manifest.write_text(
            "algorithm: {name: GRPO}\ntraining: {learning_delay: 0, max_steps: 5}\n"
        )

        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["experiments", "submit", str(manifest), flag])

        assert result.exit_code == 0, result.output
        warnings = [
            line for line in result.stderr.splitlines() if line.startswith("warning:")
        ]
        assert warnings == [
            (
                "warning: training.learning_delay ignored; removed in a future "
                "agilerl release."
            )
        ]

    def test_dry_run_with_json_prints_the_quote(self, runner, mock_client, manifest):
        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main, ["--json", "experiments", "submit", str(manifest), "--dry-run"]
            )

        assert result.exit_code == 0
        assert json.loads(result.stdout) == SUBMIT_QUOTE
        mock_client.submit_experiment.assert_not_called()

    def test_server_without_quotes_submits_as_before(
        self, runner, mock_client, manifest
    ):
        mock_client.quote_experiment.side_effect = ArenaAPIError(
            "Not Found", status_code=404
        )

        with _patched_arena_client(mock_client), self.interactive():
            result = runner.invoke(main, ["experiments", "submit", str(manifest)])

        assert result.exit_code == 0
        assert "cannot price a run before it starts" in result.stderr
        mock_client.submit_experiment.assert_called_once()

    def test_a_rejected_quote_stops_the_submit(self, runner, mock_client, manifest):
        mock_client.quote_experiment.side_effect = ArenaTrainingError(
            "No resource class has enough GPU memory for this run.", status_code=400
        )

        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["experiments", "submit", str(manifest)])

        assert result.exit_code == 1
        assert "enough GPU memory" in result.output
        mock_client.submit_experiment.assert_not_called()

    @pytest.mark.parametrize(
        "args",
        [["--dry-run"], ["--yes"], []],
        ids=["dry-run", "yes", "terminal"],
    )
    def test_refuses_a_failing_reward_without_pricing_or_submitting(
        self, runner, mock_client, manifest, args
    ):
        mock_client.quote_experiment.return_value["reward_error"] = REWARD_ERROR

        with _patched_arena_client(mock_client), self.interactive():
            result = runner.invoke(
                main, ["experiments", "submit", str(manifest), *args], input="y\n"
            )

        output = " ".join(result.output.split())
        assert result.exit_code == 1
        assert f"{REWARD_ERROR}. Fix the reward before submitting." in output
        assert "Hourly rate:" not in output
        assert "start billing?" not in output
        mock_client.submit_experiment.assert_not_called()

    def test_dry_run_with_json_refuses_a_failing_reward(
        self, runner, mock_client, manifest
    ):
        mock_client.quote_experiment.return_value["reward_error"] = REWARD_ERROR

        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main, ["--json", "experiments", "submit", str(manifest), "--dry-run"]
            )

        assert result.exit_code == 1
        assert result.stdout == ""
        error = json.loads(result.stderr)
        assert error["type"] == "ArenaValidationError"
        assert REWARD_ERROR in error["error"]


DATASETS = [
    {
        "name": "my-ds",
        "hf_dataset_id": None,
        "category": "sft",
        "training_row_count": 24,
    }
]


class TestDatasetsListCommand:
    def test_list_registered_shows_rows(self, runner, mock_client):
        mock_client.list_datasets.return_value = DATASETS
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["datasets", "list"])
        assert result.exit_code == 0, result.output
        mock_client.list_datasets.assert_called_once_with(search=None)
        _assert_table_columns(
            result.output, ["name", "hf_dataset_id", "category", "rows"]
        )
        assert "training_row_count" not in result.output
        assert "24" in result.output

    @pytest.mark.parametrize(
        ("dataset", "cell"),
        [
            ({"training_row_count": 0}, "0"),
            ({"training_row_count": None}, ""),
            ({}, ""),
        ],
        ids=["zero", "not-counted", "older-server"],
    )
    def test_list_rows_cell(self, runner, mock_client, dataset, cell):
        mock_client.list_datasets.return_value = [
            {"name": "my-ds", "hf_dataset_id": None, "category": "sft"} | dataset
        ]
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["datasets", "list"])
        assert result.exit_code == 0, result.output
        _assert_table_columns(
            result.output, ["name", "hf_dataset_id", "category", "rows"]
        )
        data_row = next(line for line in result.output.splitlines() if "my-ds" in line)
        assert data_row.split("│")[4].strip() == cell

    @pytest.mark.parametrize("hf_dataset_id", [None, ""], ids=["null", "empty"])
    def test_list_shows_a_missing_hf_dataset_id_as_an_empty_cell(
        self, runner, mock_client, hf_dataset_id
    ):
        mock_client.list_datasets.return_value = [
            {
                "name": "my-ds",
                "hf_dataset_id": hf_dataset_id,
                "category": "sft",
                "training_row_count": 24,
            }
        ]
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["datasets", "list"])
        assert result.exit_code == 0, result.output
        data_row = next(line for line in result.output.splitlines() if "my-ds" in line)
        assert data_row.split("│")[2].strip() == ""
        assert "None" not in result.output

    def test_list_json_keeps_training_row_count(self, runner, mock_client):
        mock_client.list_datasets.return_value = DATASETS
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["--json", "datasets", "list"])
        assert result.exit_code == 0, result.output
        assert json.loads(result.output) == DATASETS

    def test_search_strips_descriptions_and_sorts(self, runner, mock_client):
        mock_client.list_datasets.return_value = [
            {
                "name": "low",
                "hf_dataset_id": "org/low",
                "description": "secret",
                "downloads": 10,
            },
            {
                "name": "high",
                "hf_dataset_id": "org/high",
                "description": "secret",
                "downloads": 100,
            },
        ]
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["datasets", "list", "--search", "countdown"])
        assert result.exit_code == 0
        mock_client.list_datasets.assert_called_once_with(search="countdown")
        assert "description" not in result.output
        assert result.output.index("high") < result.output.index("low")


class TestDatasetsExistsCommand:
    def test_exists(self, runner, mock_client):
        mock_client.dataset_exists.return_value = {
            "exists": True,
            "datasetType": "reasoning",
        }
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["datasets", "exists", "my-ds"])
        assert result.exit_code == 0
        mock_client.dataset_exists.assert_called_once_with(name="my-ds")


class TestModelsListCommand:
    def test_list(self, runner, mock_client):
        mock_client.list_models.return_value = [
            {
                "model_name": "ibm-granite/granite-3.3-2b-instruct",
                "num_params": 2_000_000_000,
                "max_context_length": 8192,
            }
        ]

        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["models", "list"])

        assert result.exit_code == 0
        mock_client.list_models.assert_called_once_with()
        assert "ibm-granite/granite-3.3-2b-instruct" in result.output

    def test_list_shows_summary_columns_not_lora_detail(self, runner, mock_client):
        mock_client.list_models.return_value = [
            {
                "model_name": "Qwen/Qwen3.5-0.8B",
                "architecture": "hybrid",
                "num_params": 852_985_920,
                "lora_info": {"q_proj": [{"in_features": 1024}]},
                "lora_ranks": [1, 8, 16],
                "max_context_length": 262144,
                "status": "live",
                "modalities": ["text", "vision"],
            }
        ]

        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["models", "list"])

        assert result.exit_code == 0
        assert "Qwen/Qwen3.5-0.8B" in result.output
        assert "0.85B" in result.output
        assert "lora_info" not in result.output
        assert "in_features" not in result.output

    def test_list_marks_disabled_and_unknown_values(self, runner, mock_client):
        mock_client.list_models.return_value = [
            {"model_name": "org/off", "enabled": False, "status": "live"},
            {"model_name": "org/bare"},
        ]

        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["models", "list"])

        assert result.exit_code == 0
        assert "disabled" in result.output
        assert "None" not in result.output

    def test_root_json_keeps_every_field(self, runner, mock_client):
        rows = [{"model_name": "org/a", "lora_info": {"q_proj": []}, "enabled": True}]
        mock_client.list_models.return_value = rows

        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["--json", "models", "list"])

        assert json.loads(result.output) == rows


class TestManifestTemplateCommand:
    def test_prints_yaml_that_round_trips(self, runner, mock_client):
        spec = {"algorithm": {"name": "GRPO"}, "training": {"max_steps": 100}}
        mock_client.get_default_run_spec.return_value = spec

        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main,
                ["manifest", "template", "GRPO", "--model", "Qwen/Qwen3.5-0.8B"],
            )

        assert result.exit_code == 0
        assert yaml.safe_load(result.output) == spec
        assert list(yaml.safe_load(result.output)) == ["algorithm", "training"]

    def test_root_json_prints_json(self, runner, mock_client):
        spec = {"name": "GRPO"}
        mock_client.get_default_run_spec.return_value = spec

        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main,
                ["--json", "manifest", "template", "GRPO", "--section", "algorithm"],
            )

        assert json.loads(result.output) == spec

    def test_forwards_the_env(self, runner, mock_client):
        mock_client.get_default_run_spec.return_value = {"algorithm": {}}

        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main,
                [
                    "manifest",
                    "template",
                    "LLMPPO",
                    "--env",
                    "ticket-sim",
                    "--env-version",
                    "v1",
                    "--env-entrypoint",
                    "ticket_sim:TicketEnv",
                ],
            )

        assert result.exit_code == 0
        kwargs = mock_client.get_default_run_spec.call_args.kwargs
        assert kwargs["env_name"] == "ticket-sim"
        assert kwargs["env_version"] == "v1"
        assert kwargs["env_entrypoint"] == "ticket_sim:TicketEnv"

    def test_classic_rl_says_arena_no_longer_runs_it(self, runner):
        result = runner.invoke(
            main, ["manifest", "template", "PPO", "--env", "my-gym-env"]
        )

        assert result.exit_code != 0
        assert "Classic RL is no longer supported in Arena" in result.output

    def test_leaves_out_what_arena_chooses(self, runner, mock_client):
        mock_client.get_default_run_spec.return_value = {
            "algorithm": {"name": "GRPO", "batch_size": 16},
            "training": {"max_steps": 100, "pop_size": 1, "hpo": False},
            "mutation": {"rl_hp_selection": {}},
            "tournament_selection": {"tournament_size": 2},
        }

        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["manifest", "template", "GRPO"])

        assert result.exit_code == 0
        assert yaml.safe_load(result.output) == {
            "algorithm": {"name": "GRPO"},
            "training": {"max_steps": 100},
        }

    def test_section_leaves_out_what_arena_chooses(self, runner, mock_client):
        mock_client.get_default_run_spec.return_value = {
            "lora_config": {"lora_r": 16, "task_type": "CAUSAL_LM"},
        }

        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main, ["manifest", "template", "GRPO", "--section", "network"]
            )

        assert result.exit_code == 0
        assert yaml.safe_load(result.output) == {"lora_config": {"lora_r": 16}}

    def test_rejects_an_unknown_section(self, runner):
        result = runner.invoke(
            main, ["manifest", "template", "GRPO", "--section", "nope"]
        )

        assert result.exit_code == 2
        assert "Invalid value for '--section'" in result.output


class TestModelsSupportedCommand:
    def test_prints_every_supported_model_without_a_server(self, runner) -> None:
        result = runner.invoke(main, ["models", "supported"])

        assert result.exit_code == 0
        models = {row["hub_id"]: row for row in json.loads(result.output)}
        assert set(models) == set(SUPPORTED_MODEL_INFO)
        qwen = models["Qwen/Qwen3-4B"]
        assert qwen["architecture"] == "dense"
        assert qwen["status"] == "deprecated"
        assert models["Qwen/Qwen2.5-0.5B-Instruct"]["status"] == "deprecated"
        assert (
            models["nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16"]["status"]
            == "live"
        )
        assert qwen["max_context_length"] == 40960
        assert qwen["num_params"] == 4_022_468_096
        assert models["openai/gpt-oss-20b"]["num_params"] == 20_914_757_184
        assert qwen["lora_ranks"] == [1, 8, 16, 32, 64, 128, 256, 320, 512]
        assert "_gram_estimate" in qwen["lora_info"]
        assert qwen["supports_expert_parallel"] is False
        assert qwen["num_routed_experts"] is None
        lightning = models["nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"]
        assert lightning["supports_expert_parallel"] is True
        assert lightning["num_routed_experts"] == 128


class TestModelsInfoCommand:
    def test_info(self, runner, mock_client):
        mock_client.get_model_info.return_value = {
            "modules": ["q_proj", "v_proj"],
            "max_context_length": 8192,
        }

        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main,
                ["models", "info", "ibm-granite/granite-3.3-2b-instruct"],
            )

        assert result.exit_code == 0
        mock_client.get_model_info.assert_called_once_with(
            "ibm-granite/granite-3.3-2b-instruct"
        )
        assert "q_proj" in result.output

    def test_info_requires_model_name(self, runner, mock_client):
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["models", "info"])

        assert result.exit_code != 0
        mock_client.get_model_info.assert_not_called()


class TestDatasetsCreateCommand:
    def test_create_with_column_mapping_file(self, runner, mock_client, tmp_path):
        mapping_file = tmp_path / "mapping.json"
        mapping_file.write_text('{"prompt": "question"}', encoding="utf-8")
        csv_file = tmp_path / "data.csv"
        csv_file.write_bytes(b"question,answer\nhi,bye\n")
        mock_client.create_dataset.return_value = {"name": "new-ds", "id": 1}
        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main,
                [
                    "datasets",
                    "create",
                    "new-ds",
                    "--category",
                    "reasoning",
                    "--column-mapping-file",
                    str(mapping_file),
                    "--file",
                    str(csv_file),
                    "--description",
                    "test set",
                ],
            )
        assert result.exit_code == 0
        mock_client.create_dataset.assert_called_once_with(
            name="new-ds",
            category="reasoning",
            column_mapping='{"prompt": "question"}',
            description="test set",
            file=csv_file,
            config=None,
            hf_dataset_name=None,
            hf_config=None,
            hf_split=None,
        )

    def test_create_without_mapping_raises_usage_error(self, runner, mock_client):
        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main,
                ["datasets", "create", "new-ds", "--category", "reasoning"],
            )
        assert result.exit_code != 0
        assert "Provide --column-mapping or --column-mapping-file." in result.output
        mock_client.create_dataset.assert_not_called()

    def test_create_with_parquet_folder_and_config(self, runner, mock_client, tmp_path):
        mapping_file = tmp_path / "mapping.json"
        mapping_file.write_text('{"question": "q"}', encoding="utf-8")
        shard = tmp_path / "main" / "train.parquet"
        shard.parent.mkdir()
        shard.write_bytes(b"PAR1")
        mock_client.create_dataset.return_value = {"name": "gsm8k", "id": 2}
        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main,
                [
                    "datasets",
                    "create",
                    "gsm8k",
                    "--category",
                    "reasoning",
                    "--column-mapping-file",
                    str(mapping_file),
                    "--file",
                    str(tmp_path),
                    "--config",
                    "main",
                ],
            )
        assert result.exit_code == 0
        mock_client.create_dataset.assert_called_once_with(
            name="gsm8k",
            category="reasoning",
            column_mapping='{"question": "q"}',
            description=None,
            file=tmp_path,
            config="main",
            hf_dataset_name=None,
            hf_config=None,
            hf_split=None,
        )


class TestDatasetsDeleteCommand:
    def test_delete_with_yes(self, runner, mock_client):
        mock_client.delete_dataset.return_value = {"name": "old-ds", "archived": True}
        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main,
                ["datasets", "delete", "old-ds", "--yes"],
            )
        assert result.exit_code == 0
        mock_client.delete_dataset.assert_called_once_with(name="old-ds", confirm=True)


class TestExperimentListCommand:
    def test_list(self, runner, mock_client):
        mock_client.list_experiments.return_value = [{"name": "exp1"}]
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["experiments", "list", "--project", "proj1"])
        assert result.exit_code == 0
        mock_client.list_experiments.assert_called_once_with(project="proj1")

    def test_empty_project_points_to_submit(self, runner, mock_client):
        mock_client.list_experiments.return_value = []

        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["experiments", "list", "--project", "proj1"])

        assert result.exit_code == 0
        assert "arena experiments submit" in result.output

    def test_list_missing_project(self, runner, mock_client):
        from agilerl.exceptions import ArenaConfigError

        mock_client.list_experiments.side_effect = ArenaConfigError(
            "No project specified.",
            sdk_hint="Pass a project name or set a default.",
            cli_hint="Use --project or set a default with 'arena projects set-default <name>'.",
        )
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["experiments", "list"])
        assert result.exit_code != 0

    def test_table_shows_the_status_label_and_json_keeps_both(
        self, runner, mock_client
    ):
        mock_client.list_experiments.return_value = [
            {
                "experiment_name": "grpo-v1",
                "status": "Queued",
                "status_label": "Waiting for capacity",
            }
        ]
        with _patched_arena_client(mock_client):
            table = runner.invoke(main, ["experiments", "list", "--project", "proj1"])
            as_json = runner.invoke(
                main, ["--json", "experiments", "list", "--project", "proj1"]
            )

        assert table.exit_code == 0
        assert "Waiting for capacity" in table.output
        assert "Queued" not in table.output
        assert "status_label" not in table.output
        assert json.loads(as_json.output) == mock_client.list_experiments.return_value

    def test_table_keeps_status_when_there_is_no_label(self, runner, mock_client):
        mock_client.list_experiments.return_value = [
            {"experiment_name": "grpo-v2", "status": "Running"},
            {
                "experiment_name": "grpo-v3",
                "status": "Pending",
                "status_label": "Scheduled",
            },
        ]
        with _patched_arena_client(mock_client):
            table = runner.invoke(main, ["experiments", "list"])

        assert table.exit_code == 0
        assert "Running" in table.output
        assert "Scheduled" in table.output
        assert "Pending" not in table.output


class TestExperimentResumeCommand:
    @pytest.fixture(autouse=True)
    def quoted(self, mock_client):
        mock_client.quote_resume.return_value = dict(SUBMIT_QUOTE)

    def test_resume(self, runner, mock_client):
        mock_client.resume_experiment.return_value = {"resumed": True}
        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main,
                ["experiments", "resume", "--yes", "my-exp", "--max-steps", "1000"],
            )
        assert result.exit_code == 0
        mock_client.resume_experiment.assert_called_once_with(
            experiment_name="my-exp",
            max_steps=1000,
            checkpoint=None,
            new_name=None,
            project=None,
        )

    def test_resume_optional_fields(self, runner, mock_client):
        mock_client.resume_experiment.return_value = {"resumed": True}
        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main,
                [
                    "experiments",
                    "resume",
                    "--yes",
                    "my-exp",
                    "--max-steps",
                    "1000",
                    "--checkpoint",
                    "500",
                    "--new-name",
                    "my-exp-2",
                    "--project",
                    "proj",
                ],
            )
        assert result.exit_code == 0
        mock_client.resume_experiment.assert_called_once_with(
            experiment_name="my-exp",
            max_steps=1000,
            checkpoint="500",
            new_name="my-exp-2",
            project="proj",
        )

    @pytest.mark.parametrize(
        ("new_name_args", "line"),
        [
            ([], "Resume of 'my-exp' started.\n"),
            (
                ["--new-name", "my-exp-2"],
                "Resume of 'my-exp' started as 'my-exp-2'.\n",
            ),
        ],
        ids=["default-name", "new-name"],
    )
    def test_resume_names_the_run_instead_of_an_empty_table(
        self, runner, mock_client, new_name_args, line
    ):
        mock_client.resume_experiment.return_value = [{"submissionId": None}]
        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main,
                [
                    "experiments",
                    "resume",
                    "--yes",
                    "my-exp",
                    "--max-steps",
                    "1000",
                    *new_name_args,
                ],
            )

        assert result.exit_code == 0
        assert result.stdout.endswith(line)
        assert "submissionId" not in result.stdout

    def test_resume_json_prints_the_server_response(self, runner, mock_client):
        mock_client.resume_experiment.return_value = [{"submissionId": None}]
        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main,
                [
                    "--json",
                    "experiments",
                    "resume",
                    "--yes",
                    "my-exp",
                    "--max-steps",
                    "1000",
                ],
            )

        assert result.exit_code == 0
        assert json.loads(result.stdout.splitlines()[-1]) == [{"submissionId": None}]

    def test_resume_missing_max_steps(self, runner, mock_client):
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["experiments", "resume", "my-exp"])
        assert result.exit_code != 0


class TestExperimentResumePricePreview:
    @pytest.fixture(autouse=True)
    def quoted(self, mock_client):
        mock_client.quote_resume.return_value = dict(SUBMIT_QUOTE)
        mock_client.resume_experiment.return_value = {"resumed": True}

    @staticmethod
    def interactive(answer: bool = True):
        return patch("agilerl.submit_quote.stdin_is_a_terminal", return_value=answer)

    def test_prints_hardware_rate_and_total_before_resuming(self, runner, mock_client):
        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main,
                ["experiments", "resume", "--yes", "my-exp", "--max-steps", "1000"],
            )

        assert result.exit_code == 0
        assert "Hourly rate:     $25.60 per hour" in result.stderr
        assert (
            "Estimated total: $288.00 for about 11.2 h (1,000 steps)" in result.stderr
        )
        mock_client.quote_resume.assert_called_once_with(
            experiment_name="my-exp",
            max_steps=1000,
            checkpoint=None,
            new_name=None,
            project=None,
        )
        mock_client.resume_experiment.assert_called_once()

    def test_declining_stops_before_anything_is_billed(self, runner, mock_client):
        with _patched_arena_client(mock_client), self.interactive():
            result = runner.invoke(
                main,
                ["experiments", "resume", "my-exp", "--max-steps", "1000"],
                input="n\n",
            )

        assert result.exit_code == 1
        assert "Resume and start billing?" in result.stderr
        mock_client.resume_experiment.assert_not_called()

    def test_accepting_resumes(self, runner, mock_client):
        with _patched_arena_client(mock_client), self.interactive():
            result = runner.invoke(
                main,
                ["experiments", "resume", "my-exp", "--max-steps", "1000"],
                input="y\n",
            )

        assert result.exit_code == 0
        mock_client.resume_experiment.assert_called_once()

    @pytest.mark.parametrize("flag", ["--yes", "-y"])
    def test_yes_resumes_without_asking(self, runner, mock_client, flag):
        with _patched_arena_client(mock_client), self.interactive():
            result = runner.invoke(
                main, ["experiments", "resume", "my-exp", "--max-steps", "1000", flag]
            )

        assert result.exit_code == 0
        assert "Hourly rate:" in result.stderr
        assert "Resume and start billing?" not in result.stderr
        mock_client.resume_experiment.assert_called_once()

    def test_dry_run_prints_the_price_and_resumes_nothing(self, runner, mock_client):
        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main,
                ["experiments", "resume", "my-exp", "--max-steps", "1000", "--dry-run"],
            )

        assert result.exit_code == 0
        assert "Hourly rate:     $25.60 per hour" in result.output
        mock_client.quote_resume.assert_called_once_with(
            experiment_name="my-exp",
            max_steps=1000,
            checkpoint=None,
            new_name=None,
            project=None,
        )
        mock_client.resume_experiment.assert_not_called()

    def test_dry_run_with_json_prints_the_quote(self, runner, mock_client):
        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main,
                [
                    "--json",
                    "experiments",
                    "resume",
                    "my-exp",
                    "--max-steps",
                    "1000",
                    "--dry-run",
                ],
            )

        assert result.exit_code == 0
        assert json.loads(result.stdout) == SUBMIT_QUOTE
        mock_client.resume_experiment.assert_not_called()

    def test_non_interactive_without_yes_refuses(self, runner, mock_client):
        with _patched_arena_client(mock_client), self.interactive(answer=False):
            result = runner.invoke(
                main, ["experiments", "resume", "my-exp", "--max-steps", "1000"]
            )

        assert result.exit_code == 2
        assert "Pass --yes to resume anyway." in result.stderr
        mock_client.resume_experiment.assert_not_called()

    def test_server_without_resume_quotes_resumes_as_before(self, runner, mock_client):
        mock_client.quote_resume.side_effect = ArenaAPIError(
            "Not Found", status_code=404
        )

        with _patched_arena_client(mock_client), self.interactive():
            result = runner.invoke(
                main, ["experiments", "resume", "my-exp", "--max-steps", "1000"]
            )

        assert result.exit_code == 0
        assert "cannot price a run before it starts" in result.stderr
        mock_client.resume_experiment.assert_called_once()

    def test_a_rejected_quote_stops_the_resume(self, runner, mock_client):
        mock_client.quote_resume.side_effect = ArenaAPIError(
            "Halt the experiment before resuming", status_code=400
        )

        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main, ["experiments", "resume", "my-exp", "--max-steps", "1000"]
            )

        assert result.exit_code == 1
        assert "Halt the experiment before resuming" in result.output
        mock_client.resume_experiment.assert_not_called()


class TestExperimentStopCommand:
    def test_stop(self, runner, mock_client):
        mock_client.stop_experiment.return_value = {"stopped": True}
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["experiments", "stop", "my-exp"])
        assert result.exit_code == 0
        mock_client.stop_experiment.assert_called_once_with("my-exp", project=None)

    def test_stop_with_project(self, runner, mock_client):
        mock_client.stop_experiment.return_value = {"stopped": True}
        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main, ["experiments", "stop", "my-exp", "--project", "proj1"]
            )
        assert result.exit_code == 0
        mock_client.stop_experiment.assert_called_once_with("my-exp", project="proj1")


class TestExperimentStatusCommand:
    def test_prints_the_failure_reason(self, runner, mock_client):
        mock_client.get_experiment_status.return_value = {
            "experiment_name": "my-exp",
            "status": "Failed",
            "failure_reason": "Installing the run's Python packages failed.",
            "last_modified": "2026-11-23T10:00:00+00:00",
        }

        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["experiments", "status", "my-exp"])

        assert result.exit_code == 0, result.output
        assert "Installing the run's Python packages failed." in result.output
        mock_client.get_experiment_status.assert_called_once_with(
            "my-exp", project=None
        )

    def test_json_keeps_every_field(self, runner, mock_client):
        status = {
            "experiment_name": "my-exp",
            "status": "Running",
            "failure_reason": None,
        }
        mock_client.get_experiment_status.return_value = status

        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["--json", "experiments", "status", "my-exp"])

        assert json.loads(result.output) == status


def _log_page(lines, *, status="Failed", failure_reason=None):
    return {
        "experiment_name": "my-exp",
        "status": status,
        "failure_reason": failure_reason,
        "lines": lines,
        "next_after": None,
        "has_more": False,
    }


class TestExperimentLogsCommand:
    def test_prints_lines_then_the_failure_reason(self, runner, mock_client):
        mock_client.iter_experiment_logs.return_value = iter(
            [
                _log_page(
                    [
                        {
                            "timestamp": 1_763_892_000_000_000,
                            "severity": "ERROR",
                            "message": "boom",
                        }
                    ],
                    failure_reason="Training ran out of GPU memory.",
                )
            ]
        )

        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main, ["experiments", "logs", "my-exp", "--tail", "20", "--follow"]
            )

        assert result.exit_code == 0, result.output
        assert "2025-11-23 10:00:00 ERROR   boom" in result.output
        assert "Failed: Training ran out of GPU memory." in result.output
        mock_client.iter_experiment_logs.assert_called_once_with(
            "my-exp", tail=20, follow=True, level=None, project=None
        )

    def test_a_run_that_failed_before_training_says_why(self, runner, mock_client):
        mock_client.iter_experiment_logs.return_value = iter(
            [
                _log_page(
                    [], failure_reason="Installing the run's Python packages failed."
                )
            ]
        )

        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["experiments", "logs", "my-exp"])

        assert "No log lines for my-exp yet." in result.output
        assert "Failed: Installing the run's Python packages failed." in result.output

    def test_json_is_one_object_per_line_then_a_summary(self, runner, mock_client):
        line = {"timestamp": 1, "severity": "INFO", "message": "step 1"}
        mock_client.iter_experiment_logs.return_value = iter(
            [_log_page([line], status="Succeeded")]
        )

        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["--json", "experiments", "logs", "my-exp"])

        objects = [json.loads(text) for text in result.output.splitlines()]
        assert objects == [
            line,
            {
                "experiment_name": "my-exp",
                "status": "Succeeded",
                "failure_reason": None,
            },
        ]

    def test_tail_is_bounded(self, runner, mock_client):
        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main, ["experiments", "logs", "my-exp", "--tail", "0"]
            )

        assert result.exit_code != 0
        mock_client.iter_experiment_logs.assert_not_called()


class TestExperimentCheckpointsCommand:
    def test_checkpoints(self, runner, mock_client):
        mock_client.list_checkpoints.return_value = [{"step": 100}]
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["experiments", "checkpoints", "my-exp"])
        assert result.exit_code == 0
        mock_client.list_checkpoints.assert_called_once_with("my-exp", project=None)

    def test_checkpoints_passes_the_project(self, runner, mock_client):
        mock_client.list_checkpoints.return_value = [{"step": 100}]

        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main, ["experiments", "checkpoints", "my-exp", "--project", "proj"]
            )

        assert result.exit_code == 0
        mock_client.list_checkpoints.assert_called_once_with("my-exp", project="proj")

    def test_no_checkpoints_says_so(self, runner, mock_client):
        mock_client.list_checkpoints.return_value = []

        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["experiments", "checkpoints", "my-exp"])

        assert result.exit_code == 0
        assert "No checkpoints for my-exp." in result.output
        assert "Value" not in result.output

    def test_no_checkpoints_keeps_brackets_in_the_name(self, runner, mock_client):
        mock_client.list_checkpoints.return_value = []

        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["experiments", "checkpoints", "run[abc]"])

        assert "No checkpoints for run[abc]." in result.output

    def test_root_json_empty_is_an_empty_list(self, runner, mock_client):
        mock_client.list_checkpoints.return_value = []

        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main, ["--json", "experiments", "checkpoints", "my-exp"]
            )

        assert json.loads(result.output) == []

    def test_checkpoints_orders_steps_first_and_rounds_size(self, runner, mock_client):
        mock_client.list_checkpoints.return_value = [
            {
                "evaluation_score": None,
                "size_mb": 0.8336706161499023,
                "steps": 2080000,
                "training_score": None,
            }
        ]
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["experiments", "checkpoints", "my-exp"])
        assert result.exit_code == 0
        # steps header precedes the other columns, size_mb is two decimals.
        output = result.output.replace("\n", " ")
        assert output.index("steps") < output.index("evaluation_score")
        assert output.index("steps") < output.index("size_mb")
        assert "0.83" in output
        assert "0.8336706161499023" not in output


class TestExperimentMetricsCommand:
    def test_metrics_basic(self, runner, mock_client, tmp_path):
        target = tmp_path / "metrics.csv"
        target.write_text("step,reward\n1,0.5\n")
        mock_client.download_experiment_metrics.return_value = target
        mock_client.preview_experiment_metrics_csv.return_value = (
            b"step,reward\n1,0.5\n",
            "text/csv",
            None,
        )
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["experiments", "metrics", "my-exp"])
        assert result.exit_code == 0
        mock_client.download_experiment_metrics.assert_called_once_with(
            experiment_name="my-exp",
            output_path=None,
            metrics=None,
            project=None,
            overwrite=False,
        )

    def test_metrics_no_preview(self, runner, mock_client, tmp_path):
        target = tmp_path / "metrics.json"
        target.write_text("{}")
        mock_client.download_experiment_metrics.return_value = target
        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main, ["experiments", "metrics", "my-exp", "--preview-rows", "0"]
            )
        assert result.exit_code == 0

    def test_metrics_with_specific_metrics(self, runner, mock_client, tmp_path):
        target = tmp_path / "metrics.csv"
        target.write_text("step,loss\n1,0.1\n")
        mock_client.download_experiment_metrics.return_value = target
        mock_client.preview_experiment_metrics_csv.return_value = (
            b"step,loss\n1,0.1\n",
            "text/csv",
            None,
        )
        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main,
                [
                    "experiments",
                    "metrics",
                    "my-exp",
                    "--metric",
                    "loss",
                    "--metric",
                    "reward",
                ],
            )
        assert result.exit_code == 0
        mock_client.download_experiment_metrics.assert_called_once_with(
            experiment_name="my-exp",
            output_path=None,
            metrics=["loss", "reward"],
            project=None,
            overwrite=False,
        )

    def test_metrics_forwards_the_project(self, runner, mock_client, tmp_path):
        target = tmp_path / "metrics.csv"
        target.write_text("step,loss\n1,0.1\n")
        mock_client.download_experiment_metrics.return_value = target

        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main,
                [
                    "experiments",
                    "metrics",
                    "my-exp",
                    "--project",
                    "llm-work",
                    "--preview-rows",
                    "0",
                ],
            )

        assert result.exit_code == 0
        mock_client.download_experiment_metrics.assert_called_once_with(
            experiment_name="my-exp",
            output_path=None,
            metrics=None,
            project="llm-work",
            overwrite=False,
        )

    def test_metrics_non_csv_content_type(self, runner, mock_client, tmp_path):
        target = tmp_path / "metrics.json"
        target.write_text("{}")
        mock_client.download_experiment_metrics.return_value = target
        mock_client.preview_experiment_metrics_csv.return_value = (
            b'{"data": []}',
            "application/json",
            None,
        )
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["experiments", "metrics", "my-exp"])
        assert result.exit_code == 0

    def test_metrics_falls_back_to_file_read_for_csv(
        self, runner, mock_client, tmp_path
    ):
        """When preview content-type is not text/csv but file is .csv, read from disk."""
        target = tmp_path / "metrics.csv"
        target.write_bytes(b"step,reward\n1,0.5\n2,0.7\n")
        mock_client.download_experiment_metrics.return_value = target
        mock_client.preview_experiment_metrics_csv.return_value = (
            b"",
            "application/octet-stream",
            None,
        )
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["experiments", "metrics", "my-exp"])
        assert result.exit_code == 0

    def test_metrics_overwrite_flag_is_forwarded(self, runner, mock_client, tmp_path):
        target = tmp_path / "metrics.csv"
        target.write_text("step,loss\n1,0.1\n")
        mock_client.download_experiment_metrics.return_value = target

        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main,
                [
                    "experiments",
                    "metrics",
                    "my-exp",
                    "--output-file",
                    str(target),
                    "--overwrite",
                    "--preview-rows",
                    "0",
                ],
            )

        assert result.exit_code == 0
        mock_client.download_experiment_metrics.assert_called_once_with(
            experiment_name="my-exp",
            output_path=target,
            metrics=None,
            project=None,
            overwrite=True,
        )

    def test_metrics_existing_file_error_names_the_flag(
        self, runner, mock_client, tmp_path
    ):
        target = tmp_path / "metrics.csv"
        mock_client.download_experiment_metrics.side_effect = FileExistsError(
            f"Cannot write to {target}: a file of that name already exists. "
            "Remove it, choose a different output path, or pass --overwrite "
            "(overwrite=True in the SDK) to replace it."
        )

        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main,
                ["experiments", "metrics", "my-exp", "--output-file", str(target)],
            )

        assert result.exit_code == 1
        assert "--overwrite" in result.output
        assert not isinstance(result.exception, FileExistsError)


class TestAgentListCommand:
    def test_list_default(self, runner, mock_client):
        mock_client.list_inference_deployments.return_value = [
            {"name": "dep1", "api_key": "secret"}
        ]
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["agent", "list"])
        assert result.exit_code == 0
        mock_client.list_inference_deployments.assert_called_once_with(
            name=None, experiment_name=None, project_name=None
        )

    def test_no_agents_points_to_deploy(self, runner, mock_client):
        mock_client.list_inference_deployments.return_value = []

        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["agent", "list"])

        assert result.exit_code == 0
        assert "arena agent deploy" in result.output
        assert "Value" not in result.output

    def test_no_match_for_a_filter_does_not_suggest_deploying(
        self, runner, mock_client
    ):
        mock_client.list_inference_deployments.return_value = []

        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["agent", "list", "--name", "typo"])

        assert "No deployed agents match those filters." in result.output
        assert "arena agent deploy" not in result.output

    def test_list_with_filters(self, runner, mock_client):
        mock_client.list_inference_deployments.return_value = []
        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main,
                [
                    "agent",
                    "list",
                    "--name",
                    "dep1",
                    "--experiment-name",
                    "exp1",
                    "--project-name",
                    "proj1",
                ],
            )
        assert result.exit_code == 0
        mock_client.list_inference_deployments.assert_called_once_with(
            name="dep1", experiment_name="exp1", project_name="proj1"
        )

    def test_list_never_prints_a_credential(self, runner, mock_client):
        mock_client.list_inference_deployments.return_value = [
            {"name": "dep1", "api_key": "secret123"}
        ]
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["agent", "list"])
        assert result.exit_code == 0
        assert "secret123" not in result.output

    def test_list_surfaces_memory_scope(self, runner, mock_client):
        mock_client.list_inference_deployments.return_value = [
            {"name": "dep1", "memory_scope": "organization"}
        ]
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["agent", "list"])
        assert result.exit_code == 0
        assert "organization" in result.output


class TestAgentDeployCommand:
    def test_deploy(self, runner, mock_client):
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["agent", "deploy", "my-exp"])
        assert result.exit_code == 0
        mock_client.deploy_agent.assert_called_once_with(
            experiment_name="my-exp",
            checkpoint=None,
            memory_scope=None,
            num_gpus=None,
            gpu_type=None,
        )

    def test_deploy_with_checkpoint(self, runner, mock_client):
        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main,
                ["agent", "deploy", "my-exp", "--checkpoint", "step-500"],
            )
        assert result.exit_code == 0
        mock_client.deploy_agent.assert_called_once_with(
            experiment_name="my-exp",
            checkpoint="step-500",
            memory_scope=None,
            num_gpus=None,
            gpu_type=None,
        )

    def test_deploy_with_gpu_flags(self, runner, mock_client):
        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main,
                [
                    "agent",
                    "deploy",
                    "my-exp",
                    "--num-gpus",
                    "8",
                    "--gpu-type",
                    "h100-80",
                ],
            )
        assert result.exit_code == 0
        mock_client.deploy_agent.assert_called_once_with(
            experiment_name="my-exp",
            checkpoint=None,
            memory_scope=None,
            num_gpus=8,
            gpu_type="h100-80",
        )

    def test_deploy_rejects_zero_gpus(self, runner, mock_client):
        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main, ["agent", "deploy", "my-exp", "--num-gpus", "0"]
            )
        assert result.exit_code == 2
        assert "Invalid value" in result.output
        mock_client.deploy_agent.assert_not_called()

    def test_help_shows_gpu_flags(self, runner):
        result = runner.invoke(main, ["agent", "deploy", "--help"])

        help_text = " ".join(result.output.split())
        assert result.exit_code == 0
        assert "--num-gpus" in help_text
        assert "--gpu-type" in help_text
        assert (
            "Without ``--num-gpus`` and ``--gpu-type``, a redeploy keeps the "
            "agent's resources and a new deployment uses Arena's default."
        ) in help_text

    @pytest.mark.parametrize("scope", ["user", "organization"])
    def test_deploy_with_memory_scope(self, runner, mock_client, scope):
        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main,
                ["agent", "deploy", "my-exp", "--memory-scope", scope],
            )
        assert result.exit_code == 0
        mock_client.deploy_agent.assert_called_once_with(
            experiment_name="my-exp",
            checkpoint=None,
            memory_scope=scope,
            num_gpus=None,
            gpu_type=None,
        )

    def test_deploy_rejects_unknown_memory_scope(self, runner, mock_client):
        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main,
                ["agent", "deploy", "my-exp", "--memory-scope", "team"],
            )
        assert result.exit_code != 0
        mock_client.deploy_agent.assert_not_called()


class TestAgentUpdateCommand:
    def test_update(self, runner, mock_client):
        mock_client.update_inference_deployment.return_value = {"name": "agent-2"}
        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main,
                [
                    "agent",
                    "update",
                    "agent-1",
                    "--new-name",
                    "agent-2",
                    "--disabled",
                    "--experiment-name",
                    "exp1",
                ],
            )
        assert result.exit_code == 0
        mock_client.update_inference_deployment.assert_called_once_with(
            "agent-1",
            new_name="agent-2",
            description=None,
            enabled=False,
            experiment_name="exp1",
            project_name=None,
        )


class TestAgentDeleteCommand:
    def test_delete_confirmed(self, runner, mock_client):
        mock_client.delete_inference_deployment.return_value = {"success": True}
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["agent", "delete", "agent-1", "--yes"])
        assert result.exit_code == 0
        mock_client.delete_inference_deployment.assert_called_once_with(
            "agent-1", experiment_name=None, project_name=None
        )

    def test_delete_aborted(self, runner, mock_client):
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["agent", "delete", "agent-1"], input="n\n")
        assert result.exit_code == 0
        mock_client.delete_inference_deployment.assert_not_called()


class TestAgentRunCommand:
    def test_run_sets_active_agent(self, runner, mock_client):
        mock_client._ensure_inference_binding.return_value = ("http://x", "key")
        with (
            _patched_arena_client(mock_client),
            patch("agilerl.cli.save_active_agent") as mock_save,
        ):
            result = runner.invoke(
                main,
                [
                    "agent",
                    "run",
                    "my-dep",
                    "--experiment-name",
                    "exp1",
                    "--project-name",
                    "proj1",
                ],
            )
        assert result.exit_code == 0
        mock_client._ensure_inference_binding.assert_called_once_with(
            "my-dep",
            refresh=False,
            experiment_name="exp1",
            project_name="proj1",
        )
        mock_save.assert_called_once_with(
            "my-dep",
            experiment_name="exp1",
            project_name="proj1",
        )


def _sent_session_id(mock_agent) -> str | None:
    return mock_agent.generate_stream.call_args.kwargs["session_id"]


class TestAgentGenerateCommand:
    def test_generate_streams(self, runner, mock_client, mock_agent):
        mock_agent.generate_stream.return_value = iter(["foo", "bar"])
        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main,
                ["agent", "generate", "my-dep", "--prompt", "hi"],
            )
        assert result.exit_code == 0
        assert "foobar" in result.output

    def test_generate_asks_for_512_tokens_by_default(
        self, runner, mock_client, mock_agent
    ):
        mock_agent.generate_stream.return_value = iter(["ok"])
        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main, ["agent", "generate", "my-dep", "--prompt", "hi"]
            )
        assert result.exit_code == 0
        params = mock_agent.generate_stream.call_args.kwargs["params"]
        assert params == {"max_new_tokens": 512}

    def test_generate_forwards_max_tokens(self, runner, mock_client, mock_agent):
        mock_agent.generate_stream.return_value = iter(["ok"])
        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main,
                [
                    "agent",
                    "generate",
                    "my-dep",
                    "--prompt",
                    "hi",
                    "--max-tokens",
                    "2048",
                ],
            )
        assert result.exit_code == 0
        params = mock_agent.generate_stream.call_args.kwargs["params"]
        assert params == {"max_new_tokens": 2048}

    def test_generate_refuses_zero_max_tokens(self, runner, mock_client, mock_agent):
        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main,
                ["agent", "generate", "my-dep", "--prompt", "hi", "--max-tokens", "0"],
            )
        assert result.exit_code == 2
        assert "--max-tokens" in result.output
        mock_agent.generate_stream.assert_not_called()

    def test_first_prompt_starts_and_records_a_session(
        self, runner, mock_client, mock_agent
    ):
        mock_agent.generate_stream.return_value = iter(["ok"])
        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main, ["agent", "generate", "my-dep", "--prompt", "hi"]
            )
        assert result.exit_code == 0
        started = _sent_session_id(mock_agent)
        # A real UUID, so the deployment can key history off it.
        assert uuid.UUID(started).version == 4
        assert load_active_session("my-dep") == started

    def test_the_next_prompt_continues_the_session_it_started(
        self, runner, mock_client, mock_agent
    ):
        mock_agent.generate_stream.side_effect = [iter(["one"]), iter(["two"])]
        with _patched_arena_client(mock_client):
            runner.invoke(main, ["agent", "generate", "my-dep", "--prompt", "hi"])
            first = _sent_session_id(mock_agent)
            runner.invoke(main, ["agent", "generate", "my-dep", "--prompt", "again"])
            second = _sent_session_id(mock_agent)
        assert first == second

    def test_each_deployment_starts_its_own_session(
        self, runner, mock_client, mock_agent
    ):
        mock_agent.generate_stream.side_effect = [iter(["a"]), iter(["b"])]
        with _patched_arena_client(mock_client):
            runner.invoke(main, ["agent", "generate", "dep-a", "--prompt", "hi"])
            first = _sent_session_id(mock_agent)
            runner.invoke(main, ["agent", "generate", "dep-b", "--prompt", "hi"])
            second = _sent_session_id(mock_agent)
        assert first != second
        assert load_active_session("dep-a") == first
        assert load_active_session("dep-b") == second

    def test_starting_a_session_is_logged_at_info(
        self, runner, mock_client, mock_agent, caplog
    ):
        mock_agent.generate_stream.return_value = iter(["reply"])
        with _patched_arena_client(mock_client), caplog.at_level(logging.INFO):
            result = runner.invoke(
                main, ["agent", "generate", "my-dep", "--prompt", "hi"]
            )
        assert result.exit_code == 0
        records = [r for r in caplog.records if "Started session" in r.getMessage()]
        assert len(records) == 1
        assert records[0].levelno == logging.INFO
        assert (
            records[0].getMessage()
            == f"Started session {load_active_session('my-dep')}."
        )

    def test_continuing_a_session_is_not_logged(
        self, runner, mock_client, mock_agent, caplog
    ):
        save_active_session("my-dep", "s-resumed")
        mock_agent.generate_stream.return_value = iter(["ok"])
        with _patched_arena_client(mock_client), caplog.at_level(logging.INFO):
            runner.invoke(main, ["agent", "generate", "my-dep", "--prompt", "hi"])
        assert not [r for r in caplog.records if "Started session" in r.getMessage()]

    def test_generate_forwards_session_id(self, runner, mock_client, mock_agent):
        mock_agent.generate_stream.return_value = iter(["ok"])
        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main,
                ["agent", "generate", "my-dep", "--prompt", "hi", "--session-id", "s1"],
            )
        assert result.exit_code == 0
        mock_agent.generate_stream.assert_called_once_with(
            "hi", params={"max_new_tokens": 512}, session_id="s1"
        )

    def test_generate_continues_the_resumed_session(
        self, runner, mock_client, mock_agent
    ):
        save_active_session("my-dep", "s-resumed")
        mock_agent.generate_stream.return_value = iter(["ok"])
        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main, ["agent", "generate", "my-dep", "--prompt", "hi"]
            )
        assert result.exit_code == 0
        mock_agent.generate_stream.assert_called_once_with(
            "hi", params={"max_new_tokens": 512}, session_id="s-resumed"
        )

    def test_generate_only_resumes_its_own_deployment(
        self, runner, mock_client, mock_agent
    ):
        save_active_session("other-dep", "s-resumed")
        mock_agent.generate_stream.return_value = iter(["ok"])
        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main, ["agent", "generate", "my-dep", "--prompt", "hi"]
            )
        assert result.exit_code == 0
        assert _sent_session_id(mock_agent) != "s-resumed"
        assert load_active_session("other-dep") == "s-resumed"

    def test_session_id_overrides_the_resumed_session(
        self, runner, mock_client, mock_agent
    ):
        save_active_session("my-dep", "s-resumed")
        mock_agent.generate_stream.return_value = iter(["ok"])
        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main,
                ["agent", "generate", "my-dep", "--prompt", "hi", "--session-id", "s1"],
            )
        assert result.exit_code == 0
        mock_agent.generate_stream.assert_called_once_with(
            "hi", params={"max_new_tokens": 512}, session_id="s1"
        )

    def test_new_session_abandons_the_resumed_session(
        self, runner, mock_client, mock_agent
    ):
        save_active_session("my-dep", "s-resumed")
        mock_agent.generate_stream.return_value = iter(["ok"])
        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main,
                ["agent", "generate", "my-dep", "--prompt", "hi", "--new-session"],
            )
        assert result.exit_code == 0
        started = _sent_session_id(mock_agent)
        assert started != "s-resumed"
        assert uuid.UUID(started).version == 4

    def test_new_session_becomes_the_one_later_prompts_continue(
        self, runner, mock_client, mock_agent
    ):
        save_active_session("my-dep", "s-resumed")
        mock_agent.generate_stream.return_value = iter(["ok"])
        with _patched_arena_client(mock_client):
            runner.invoke(
                main,
                ["agent", "generate", "my-dep", "--prompt", "hi", "--new-session"],
            )
        assert load_active_session("my-dep") == _sent_session_id(mock_agent)

    def test_session_id_does_not_change_what_is_resumed(
        self, runner, mock_client, mock_agent
    ):
        save_active_session("my-dep", "s-resumed")
        mock_agent.generate_stream.return_value = iter(["ok"])
        with _patched_arena_client(mock_client):
            runner.invoke(
                main,
                ["agent", "generate", "my-dep", "--prompt", "hi", "--session-id", "s1"],
            )
        assert _sent_session_id(mock_agent) == "s1"
        assert load_active_session("my-dep") == "s-resumed"

    def test_session_id_and_new_session_conflict(self, runner, mock_client, mock_agent):
        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main,
                [
                    "agent",
                    "generate",
                    "my-dep",
                    "--prompt",
                    "hi",
                    "--session-id",
                    "s1",
                    "--new-session",
                ],
            )
        assert result.exit_code != 0
        assert "not both" in result.output
        mock_agent.generate_stream.assert_not_called()
        assert load_active_session("my-dep") is None

    def test_generate_uses_active_agent(self, runner, mock_client, mock_agent):
        mock_agent.generate_stream.return_value = iter(["ok"])
        active = MagicMock(
            deployment_name="cached-dep",
            experiment_name="exp1",
            project_name=None,
        )
        with (
            _patched_arena_client(mock_client),
            patch("agilerl.cli.load_active_agent", return_value=active),
        ):
            result = runner.invoke(main, ["agent", "generate", "--prompt", "hi"])
        assert result.exit_code == 0
        mock_client.open_inference_agent.assert_called_once_with(
            "cached-dep",
            refresh=False,
            experiment_name="exp1",
            project_name=None,
        )

    def test_generate_without_active_agent_fails(self, runner, mock_client):
        with (
            _patched_arena_client(mock_client),
            patch("agilerl.cli.load_active_agent", return_value=None),
        ):
            result = runner.invoke(main, ["agent", "generate", "--prompt", "hi"])
        assert result.exit_code != 0
        assert "No active agent" in result.output


class TestAgentSessionsCommands:
    def test_sessions_list(self, runner, mock_client, mock_agent):
        mock_agent.list_sessions.return_value = [
            SessionInfo(
                session_id="s1",
                created_at="2026-07-30T00:00:00Z",
                last_updated="2026-08-01T00:00:00Z",
            )
        ]
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["agent", "sessions", "list", "my-dep"])
        assert result.exit_code == 0
        mock_agent.list_sessions.assert_called_once_with()
        assert "s1" in result.output
        assert "2026-07-30" in result.output
        assert "Created At" in result.output
        assert "Last Updated" in result.output

    def test_sessions_list_uses_active_agent(self, runner, mock_client, mock_agent):
        mock_agent.list_sessions.return_value = []
        active = MagicMock(
            deployment_name="cached-dep",
            experiment_name=None,
            project_name=None,
        )
        with (
            _patched_arena_client(mock_client),
            patch("agilerl.cli.load_active_agent", return_value=active),
        ):
            result = runner.invoke(main, ["agent", "sessions", "list"])
        assert result.exit_code == 0
        mock_client.open_inference_agent.assert_called_once_with(
            "cached-dep",
            refresh=False,
            experiment_name=None,
            project_name=None,
        )

    def test_sessions_get(self, runner, mock_client, mock_agent):
        mock_agent.get_session.return_value = SessionDetail(
            session_id="s1",
            messages=[
                SessionMessage(role="user", content="what is my name?"),
                SessionMessage(role="assistant", content="Samuel."),
            ],
        )
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["agent", "sessions", "get", "s1", "my-dep"])
        assert result.exit_code == 0
        mock_agent.get_session.assert_called_once_with("s1")
        assert "Session s1" in result.output
        assert "2 messages" in result.output
        assert "what is my name?" in result.output
        assert "Samuel." in result.output

    def test_sessions_get_omits_timestamps_the_route_does_not_return(
        self, runner, mock_client, mock_agent
    ):
        mock_agent.get_session.return_value = SessionDetail(session_id="s1")
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["agent", "sessions", "get", "s1", "my-dep"])
        assert result.exit_code == 0
        assert "None" not in result.output
        assert "created" not in result.output

    def test_sessions_get_shows_timestamps_when_present(
        self, runner, mock_client, mock_agent
    ):
        mock_agent.get_session.return_value = SessionDetail(
            session_id="s1", created_at="2026-08-01T00:00:00Z"
        )
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["agent", "sessions", "get", "s1", "my-dep"])
        assert "created 2026-08-01T00:00:00Z" in result.output

    def test_sessions_get_on_an_empty_session(self, runner, mock_client, mock_agent):
        mock_agent.get_session.return_value = SessionDetail(session_id="s1")
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["agent", "sessions", "get", "s1", "my-dep"])
        assert "No messages in this session." in result.output

    def test_sessions_list_marks_the_resumed_session(
        self, runner, mock_client, mock_agent
    ):
        save_active_session("my-dep", "s2")
        mock_agent.list_sessions.return_value = [
            SessionInfo(session_id="s1"),
            SessionInfo(session_id="s2"),
        ]
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["agent", "sessions", "list", "my-dep"])
        assert result.exit_code == 0
        assert "Current" in result.output
        marked = [line for line in result.output.splitlines() if "●" in line]
        assert len(marked) == 1
        assert "s2" in marked[0]

    def test_sessions_list_with_no_sessions_says_so(
        self, runner, mock_client, mock_agent
    ):
        mock_agent.list_sessions.return_value = []
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["agent", "sessions", "list", "my-dep"])
        assert result.exit_code == 0
        assert "No chat sessions on my-dep yet." in result.output
        assert "Session Id" not in result.output
        assert "Current" not in result.output

    def test_sessions_list_marks_nothing_when_none_is_resumed(
        self, runner, mock_client, mock_agent
    ):
        mock_agent.list_sessions.return_value = [SessionInfo(session_id="s1")]
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["agent", "sessions", "list", "my-dep"])
        assert result.exit_code == 0
        assert "●" not in result.output

    def test_sessions_get_without_active_agent_fails(self, runner, mock_client):
        with (
            _patched_arena_client(mock_client),
            patch("agilerl.cli.load_active_agent", return_value=None),
        ):
            result = runner.invoke(main, ["agent", "sessions", "get", "s1"])
        assert result.exit_code != 0
        assert "No active agent" in result.output


class TestProjectsListCommand:
    def test_list(self, runner, mock_client):
        mock_client.list_projects.return_value = [{"name": "proj1"}]
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["projects", "list"])
        assert result.exit_code == 0
        mock_client.list_projects.assert_called_once()


class TestProjectsCreateCommand:
    def test_create(self, runner, mock_client):
        mock_client.create_project.return_value = {"name": "new-proj"}
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["projects", "create", "new-proj"])
        assert result.exit_code == 0
        mock_client.create_project.assert_called_once_with(
            name="new-proj", description=None
        )

    def test_create_with_options(self, runner, mock_client):
        mock_client.create_project.return_value = {"name": "llm-proj"}
        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main,
                [
                    "projects",
                    "create",
                    "llm-proj",
                    "--description",
                    "An LLM project",
                ],
            )
        assert result.exit_code == 0
        mock_client.create_project.assert_called_once_with(
            name="llm-proj", description="An LLM project"
        )

    def test_llm_based_flag_is_accepted_and_ignored(self, runner, mock_client):
        mock_client.create_project.return_value = {"name": "llm-proj"}
        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main, ["projects", "create", "llm-proj", "--llm-based"]
            )

        assert result.exit_code == 0, result.output
        mock_client.create_project.assert_called_once_with(
            name="llm-proj", description=None
        )

    def test_simulation_based_flag_fails_with_the_reason(self, runner, mock_client):
        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main, ["projects", "create", "sim-proj", "--simulation-based"]
            )

        assert result.exit_code == 2
        assert "agilerl 3 creates LLM projects only" in result.output
        mock_client.create_project.assert_not_called()

    def test_1x_flags_are_hidden_from_help(self, runner):
        result = runner.invoke(main, ["projects", "create", "--help"])

        assert result.exit_code == 0
        assert "--llm-based" not in result.output
        assert "--simulation-based" not in result.output


class TestProjectsDeleteCommand:
    def test_delete_confirmed(self, runner, mock_client):
        mock_client.delete_project.return_value = {"deleted": True}
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["projects", "delete", "my-proj", "--yes"])
        assert result.exit_code == 0
        mock_client.delete_project.assert_called_once_with(name="my-proj")

    def test_delete_aborted(self, runner, mock_client):
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["projects", "delete", "my-proj"], input="n\n")
        assert result.exit_code == 0
        assert "Aborted" in result.output
        mock_client.delete_project.assert_not_called()


class TestProjectsDefaultCommand:
    def test_set_default(self, runner, mock_client):
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["projects", "set-default", "my-proj"])
        assert result.exit_code == 0
        # The command delegates persistence (and the confirmation log) to the
        # client, which is mocked here, so behavior is verified via the call.
        mock_client.set_default_project.assert_called_once_with("my-proj")

    def test_get_default_when_set(self, runner, mock_client):
        mock_client.get_default_project.return_value = "my-proj"
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["projects", "get-default"])
        assert result.exit_code == 0
        assert result.output.strip() == "my-proj"

    def test_get_default_when_unset(self, runner, mock_client):
        mock_client.get_default_project.return_value = None
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["projects", "get-default"])
        assert result.exit_code == 0
        assert "No default project set" in result.output

    def test_get_default_json(self, runner, mock_client):
        mock_client.get_default_project.return_value = "my-proj"

        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["--json", "projects", "get-default"])

        assert result.exit_code == 0
        assert json.loads(result.output) == {"project": "my-proj"}


class TestRedactAgentRows:
    def test_redacts_api_key(self):
        rows = [{"name": "dep1", "api_key": "secret", "url": "http://x"}]
        result = _redact_agent_rows_for_display(rows)
        assert len(result) == 1
        assert "api_key" not in result[0]
        assert result[0]["name"] == "dep1"
        assert result[0]["url"] == "http://x"

    def test_redacts_nested_spec_api_key(self):
        rows = [
            {
                "name": "dep1",
                "spec": {"api_key": "nested_secret", "url": "http://y"},
            }
        ]
        result = _redact_agent_rows_for_display(rows)
        assert "api_key" not in result[0].get("spec", {})
        assert result[0]["spec"]["url"] == "http://y"

    def test_leaves_the_caller_rows_untouched(self):
        rows = [{"name": "dep1", "api_key": "secret"}]
        _redact_agent_rows_for_display(rows)
        assert rows[0]["api_key"] == "secret"

    def test_keeps_non_credential_fields(self):
        rows = [{"name": "dep1", "memory_scope": "organization", "url": "http://x"}]
        result = _redact_agent_rows_for_display(rows)
        assert result[0]["memory_scope"] == "organization"
        assert result[0]["url"] == "http://x"

    def test_empty_rows(self):
        assert _redact_agent_rows_for_display([]) == []


class TestCliErrorHandling:
    def test_arena_error_exits_nonzero(self, runner, mock_client):
        mock_client.login.side_effect = ArenaAPIError("auth failed", status_code=401)
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["login"])
        assert result.exit_code != 0

    def test_runtime_error_propagates(self, runner, mock_client):
        mock_client.logout.side_effect = RuntimeError("unexpected crash")
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["logout"])
        assert result.exit_code != 0


class TestGlobalOptions:
    def test_api_key_passed_to_config(self, runner, mock_client):
        with patch("agilerl.config.build_client", return_value=mock_client) as m:
            result = runner.invoke(main, ["--api-key", "pat_abc", "logout"])
        assert result.exit_code == 0
        config_arg = m.call_args[0][0]
        assert config_arg.api_key == "pat_abc"

    def test_org_key_passed_to_config(self, runner, mock_client):
        with patch("agilerl.config.build_client", return_value=mock_client) as m:
            result = runner.invoke(
                main,
                [
                    "--org-key",
                    "arena_org_secret",
                    "--external-user-id",
                    "partner-user-1",
                    "logout",
                ],
            )
        assert result.exit_code == 0
        config_arg = m.call_args[0][0]
        assert config_arg.org_key == "arena_org_secret"
        assert config_arg.external_user_id == "partner-user-1"

    def test_keycloak_url_passed_to_config(self, runner, mock_client):
        with patch("agilerl.config.build_client", return_value=mock_client) as m:
            result = runner.invoke(
                main, ["--keycloak-url", "http://kc.example", "logout"]
            )
        assert result.exit_code == 0
        config_arg = m.call_args[0][0]
        assert config_arg.keycloak_url == "http://kc.example"

    def test_base_url_passed_to_config(self, runner, mock_client):
        with patch("agilerl.config.build_client", return_value=mock_client) as m:
            result = runner.invoke(main, ["--base-url", "http://local:8000", "logout"])
        assert result.exit_code == 0
        config_arg = m.call_args[0][0]
        assert config_arg.base_url == "http://local:8000"

    def test_upload_timeout_passed_to_config(self, runner, mock_client):
        with patch("agilerl.config.build_client", return_value=mock_client) as m:
            result = runner.invoke(main, ["--upload-timeout", "600", "logout"])
        assert result.exit_code == 0
        config_arg = m.call_args[0][0]
        assert config_arg.upload_timeout == 600

    def test_invalid_timeout_rejected(self, runner):
        result = runner.invoke(main, ["--request-timeout", "0", "login"])
        assert result.exit_code != 0


class TestAgentSessionsResume:
    @pytest.fixture
    def sessions(self, mock_agent):
        mock_agent.list_sessions.return_value = [
            SessionInfo(session_id="s1", last_updated="2026-08-05"),
            SessionInfo(session_id="s2", last_updated="2026-08-04"),
            SessionInfo(session_id="s3", last_updated="2026-07-30"),
        ]
        return mock_agent

    def test_explicit_session_id_is_persisted(self, runner, mock_client, sessions):
        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main,
                ["agent", "sessions", "resume", "my-dep", "--session-id", "s2"],
            )
        assert result.exit_code == 0
        assert load_active_session("my-dep") == "s2"
        assert "s2" in result.output

    def test_unknown_session_id_is_rejected(self, runner, mock_client, sessions):
        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main,
                ["agent", "sessions", "resume", "my-dep", "--session-id", "nope"],
            )
        assert result.exit_code != 0
        assert "No session" in result.output
        assert load_active_session("my-dep") is None

    def test_picker_selection_is_persisted(self, runner, mock_client, sessions):
        with (
            _patched_arena_client(mock_client),
            patch("agilerl.cli.supports_interactive_selection", return_value=True),
            patch(
                "agilerl.cli.select_row",
                return_value={"session_id": "s3"},
            ) as mock_select,
        ):
            result = runner.invoke(main, ["agent", "sessions", "resume", "my-dep"])
        assert result.exit_code == 0
        assert load_active_session("my-dep") == "s3"
        assert mock_select.call_args.kwargs["selected"] == 0

    def test_picker_opens_on_the_session_already_resumed(
        self, runner, mock_client, sessions
    ):
        save_active_session("my-dep", "s2")
        with (
            _patched_arena_client(mock_client),
            patch("agilerl.cli.supports_interactive_selection", return_value=True),
            patch(
                "agilerl.cli.select_row", return_value={"session_id": "s2"}
            ) as mock_select,
        ):
            result = runner.invoke(main, ["agent", "sessions", "resume", "my-dep"])
        assert result.exit_code == 0
        assert mock_select.call_args.kwargs["selected"] == 1

    def test_cancelling_the_picker_changes_nothing(self, runner, mock_client, sessions):
        save_active_session("my-dep", "s1")
        with (
            _patched_arena_client(mock_client),
            patch("agilerl.cli.supports_interactive_selection", return_value=True),
            patch("agilerl.cli.select_row", return_value=None),
        ):
            result = runner.invoke(main, ["agent", "sessions", "resume", "my-dep"])
        assert result.exit_code != 0
        assert load_active_session("my-dep") == "s1"

    def test_non_interactive_terminal_asks_for_session_id(
        self, runner, mock_client, sessions
    ):
        with (
            _patched_arena_client(mock_client),
            patch("agilerl.cli.supports_interactive_selection", return_value=False),
            patch("agilerl.cli.select_row") as mock_select,
        ):
            result = runner.invoke(main, ["agent", "sessions", "resume", "my-dep"])
        assert result.exit_code != 0
        assert "--session-id" in result.output
        mock_select.assert_not_called()

    def test_deployment_with_no_sessions_says_so(self, runner, mock_client, mock_agent):
        mock_agent.list_sessions.return_value = []
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["agent", "sessions", "resume", "my-dep"])
        assert result.exit_code != 0
        assert "No chat sessions" in result.output

    def test_resume_uses_the_active_agent(self, runner, mock_client, sessions):
        active = MagicMock(
            deployment_name="cached-dep", experiment_name=None, project_name=None
        )
        with (
            _patched_arena_client(mock_client),
            patch("agilerl.cli.load_active_agent", return_value=active),
        ):
            result = runner.invoke(
                main, ["agent", "sessions", "resume", "--session-id", "s1"]
            )
        assert result.exit_code == 0
        assert load_active_session("cached-dep") == "s1"


class TestAgentSessionsDelete:
    def test_delete_with_yes_skips_the_prompt(self, runner, mock_client, mock_agent):
        mock_agent.delete_session.return_value = True
        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main, ["agent", "sessions", "delete", "s1", "my-dep", "--yes"]
            )
        assert result.exit_code == 0
        mock_agent.delete_session.assert_called_once_with("s1")
        assert "Deleted session s1" in result.output

    def test_delete_confirms_before_deleting(self, runner, mock_client, mock_agent):
        mock_agent.delete_session.return_value = True
        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main, ["agent", "sessions", "delete", "s1", "my-dep"], input="y\n"
            )
        assert result.exit_code == 0
        assert "Delete session 's1' and its messages?" in result.output
        mock_agent.delete_session.assert_called_once_with("s1")

    def test_declining_the_prompt_deletes_nothing(
        self, runner, mock_client, mock_agent
    ):
        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main, ["agent", "sessions", "delete", "s1", "my-dep"], input="n\n"
            )
        assert result.exit_code == 0
        assert "Aborted." in result.output
        mock_agent.delete_session.assert_not_called()

    def test_declining_keeps_the_current_session(self, runner, mock_client, mock_agent):
        save_active_session("my-dep", "s1")
        with _patched_arena_client(mock_client):
            runner.invoke(
                main, ["agent", "sessions", "delete", "s1", "my-dep"], input="n\n"
            )
        assert load_active_session("my-dep") == "s1"

    def test_deleting_the_current_session_clears_it(
        self, runner, mock_client, mock_agent
    ):
        save_active_session("my-dep", "s1")
        mock_agent.delete_session.return_value = True
        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main, ["agent", "sessions", "delete", "s1", "my-dep", "--yes"]
            )
        assert result.exit_code == 0
        assert load_active_session("my-dep") is None

    def test_deleting_another_session_keeps_the_current_one(
        self, runner, mock_client, mock_agent
    ):
        save_active_session("my-dep", "s2")
        mock_agent.delete_session.return_value = True
        with _patched_arena_client(mock_client):
            runner.invoke(
                main, ["agent", "sessions", "delete", "s1", "my-dep", "--yes"]
            )
        assert load_active_session("my-dep") == "s2"

    def test_a_session_already_gone_says_so(self, runner, mock_client, mock_agent):
        mock_agent.delete_session.return_value = False
        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main, ["agent", "sessions", "delete", "s1", "my-dep", "--yes"]
            )
        assert result.exit_code == 0
        assert "No session s1" in result.output

    def test_a_session_already_gone_is_still_cleared_locally(
        self, runner, mock_client, mock_agent
    ):
        save_active_session("my-dep", "s1")
        mock_agent.delete_session.return_value = False
        with _patched_arena_client(mock_client):
            runner.invoke(
                main, ["agent", "sessions", "delete", "s1", "my-dep", "--yes"]
            )
        assert load_active_session("my-dep") is None

    def test_delete_uses_the_active_agent(self, runner, mock_client, mock_agent):
        mock_agent.delete_session.return_value = True
        active = MagicMock(
            deployment_name="cached-dep", experiment_name=None, project_name=None
        )
        with (
            _patched_arena_client(mock_client),
            patch("agilerl.cli.load_active_agent", return_value=active),
        ):
            result = runner.invoke(main, ["agent", "sessions", "delete", "s1", "--yes"])
        assert result.exit_code == 0
        mock_client.open_inference_agent.assert_called_once_with(
            "cached-dep", refresh=False, experiment_name=None, project_name=None
        )

    def test_delete_requires_a_session_id(self, runner, mock_client, mock_agent):
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["agent", "sessions", "delete"])
        assert result.exit_code != 0
        mock_agent.delete_session.assert_not_called()


class TestAgentSessionsClear:
    def test_clear_removes_the_current_session(self, runner):
        save_active_session("my-dep", "s1")
        result = runner.invoke(main, ["agent", "sessions", "clear", "my-dep"])
        assert result.exit_code == 0
        assert load_active_session("my-dep") is None
        assert "Cleared the session" in result.output

    def test_clear_with_nothing_to_clear_says_so(self, runner):
        result = runner.invoke(main, ["agent", "sessions", "clear", "my-dep"])
        assert result.exit_code == 0
        assert "No session to clear" in result.output

    def test_clear_leaves_other_deployments_alone(self, runner):
        save_active_session("my-dep", "s1")
        save_active_session("other-dep", "s2")
        runner.invoke(main, ["agent", "sessions", "clear", "my-dep"])
        assert load_active_session("other-dep") == "s2"

    def test_clear_uses_the_active_agent(self, runner):
        save_active_session("cached-dep", "s1")
        active = MagicMock(
            deployment_name="cached-dep", experiment_name=None, project_name=None
        )
        with patch("agilerl.cli.load_active_agent", return_value=active):
            result = runner.invoke(main, ["agent", "sessions", "clear"])
        assert result.exit_code == 0
        assert load_active_session("cached-dep") is None

    def test_clear_without_active_agent_fails(self, runner):
        with patch("agilerl.cli.load_active_agent", return_value=None):
            result = runner.invoke(main, ["agent", "sessions", "clear"])
        assert result.exit_code != 0
        assert "No active agent" in result.output

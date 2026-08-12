# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for arena.cli — Click CLI commands and config."""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import click
import pytest
from click.testing import CliRunner

from agilerl.arena.cli import (
    _redact_agent_rows_for_display,
    arena_client,
    main,
)
from agilerl.arena.config import CommandConfig, build_client
from agilerl.arena.exceptions import ArenaAPIError


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def mock_client() -> MagicMock:
    return MagicMock(
        spec_set=[
            "close",
            "login",
            "logout",
            "get_current_user",
            "get_user_credits",
            "list_resources",
            "list_environments",
            "environment_exists",
            "list_environment_entrypoints",
            "validate_environment",
            "profile_environment",
            "delete_environment",
            "duplicate_environment_version",
            "submit_experiment",
            "list_experiments",
            "resume_experiment",
            "stop_experiment",
            "list_checkpoints",
            "download_experiment_metrics",
            "preview_experiment_metrics_csv",
            "deploy_agent",
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
        ]
    )


@contextmanager
def _patched_arena_client(mock_client: MagicMock):
    """Patch build_client so arena_client yields our mock."""
    with patch("agilerl.arena.config.build_client", return_value=mock_client):
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

    @patch("agilerl.arena.config.build_client")
    def test_build_client_calls_configure_and_init(self, mock_build):
        with patch("agilerl.arena.config.ArenaClient") as MockArenaClient:
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
            build_client(cfg)
            MockArenaClient.configure.assert_called_once_with(
                base_url="http://api",
                keycloak_url="http://kc",
                realm="myrealm",
                client_id="mycli",
            )
            MockArenaClient.assert_called_once_with(
                api_key="pat_123",
                request_timeout=5,
                upload_timeout=120,
            )


class TestResolveRootCommandConfig:
    def test_returns_existing_command_config_unchanged(self):
        from agilerl.arena.config import _resolve_root_command_config

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
        from agilerl.arena.config import _resolve_root_command_config

        ctx = click.Context(click.Command("root"))
        ctx.obj = None
        ctx.params = None  # type: ignore[assignment]

        cfg = _resolve_root_command_config(ctx)
        assert isinstance(cfg, CommandConfig)
        assert cfg.api_key is None
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
        with patch("agilerl.arena.config.build_client", return_value=mock_client):
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
        with patch("agilerl.arena.config.build_client", return_value=mock_client):
            with patch("agilerl.arena.config.handle_error") as mock_handle:
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
        with patch("agilerl.arena.config.build_client", return_value=mock_client):

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

        with patch("agilerl.arena.config.build_client"):
            result = runner.invoke(main, ["--request-timeout", "42", "_test_ctx"])
        assert "timeout=42" in result.output


class TestLoginCommand:
    def test_login(self, runner, mock_client):
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["login"])
        assert result.exit_code == 0
        mock_client.login.assert_called_once_with(timeout=300, force=False)
        mock_client.close.assert_called_once()

    def test_login_force(self, runner, mock_client):
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["login", "--force"])
        assert result.exit_code == 0
        mock_client.login.assert_called_once_with(timeout=300, force=True)

    def test_login_custom_timeout(self, runner, mock_client):
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["login", "--timeout", "60"])
        assert result.exit_code == 0
        mock_client.login.assert_called_once_with(timeout=60, force=False)


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

    def test_user_credits(self, runner, mock_client):
        mock_client.get_user_credits.return_value = {"credits": 1000}
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["user", "credits"])
        assert result.exit_code == 0


class TestResourcesCommands:
    def test_resources_list(self, runner, mock_client):
        mock_client.list_resources.return_value = {
            "tiers": {
                "medium": {
                    "name": "arena-medium",
                    "num_gpus": 4,
                    "gpu_type": "A100",
                    "num_cpus": 16,
                    "ram_gb": 64,
                    "price_per_node_hour": 10.0,
                },
                "small": {
                    "name": "arena-small",
                    "num_gpus": 1,
                    "gpu_type": "T4",
                    "num_cpus": 4,
                    "ram_gb": 16,
                    "price_per_node_hour": 2.5,
                },
            }
        }
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["resources", "list"])
        assert result.exit_code == 0

    def test_resources_list_no_gpu(self, runner, mock_client):
        mock_client.list_resources.return_value = {
            "tiers": {
                "cpu": {
                    "name": "arena-cpu",
                    "num_gpus": 0,
                    "gpu_type": None,
                    "num_cpus": 8,
                    "ram_gb": 32,
                    "price_per_node_hour": 1.0,
                },
            }
        }
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["resources", "list"])
        assert result.exit_code == 0


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
            multi_agent=False,
            language_based=False,
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

    def test_validate_multi_agent_with_rollouts(self, runner, mock_client):
        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main,
                ["env", "validate", "my-env", "--multi-agent", "--do-rollouts"],
            )
        assert result.exit_code == 0
        call_kwargs = mock_client.validate_environment.call_args.kwargs
        assert call_kwargs["multi_agent"] is True
        assert call_kwargs["language_based"] is False
        assert call_kwargs["do_rollouts"] is True

    def test_validate_language_based(self, runner, mock_client):
        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main,
                ["env", "validate", "my-gem-env", "--language-based"],
            )
        assert result.exit_code == 0
        call_kwargs = mock_client.validate_environment.call_args.kwargs
        assert call_kwargs["language_based"] is True


class TestEnvProfileCommand:
    def test_profile(self, runner, mock_client):
        mock_client.profile_environment.return_value = {
            "avg_cpu_per_env": 0.312,
            "memory_per_env_gb": 0.126,
            "steps_per_second": 2750.545,
        }
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["env", "profile", "my-env"])
        assert result.exit_code == 0
        mock_client.profile_environment.assert_called_once_with(
            name="my-env", version=None
        )

    def test_profile_none_result(self, runner, mock_client):
        mock_client.profile_environment.return_value = None
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["env", "profile", "my-env"])
        assert result.exit_code == 0


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


class TestExperimentSubmitCommand:
    def test_submit(self, runner, mock_client, tmp_path):
        manifest = tmp_path / "manifest.yaml"
        manifest.write_text("experiment: test")
        mock_client.submit_experiment.return_value = {"id": "exp-123"}
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["experiments", "submit", str(manifest)])
        assert result.exit_code == 0
        mock_client.submit_experiment.assert_called_once_with(
            manifest=manifest,
            resource_id="arena-medium",
            num_nodes=2,
            project=None,
            experiment_name=None,
            reward_file=None,
            completion=None,
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
                    str(manifest),
                    "--resource-id",
                    "arena-large",
                    "--num-nodes",
                    "4",
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
            resource_id="arena-large",
            num_nodes=4,
            project="proj1",
            experiment_name="my-exp",
            reward_file=reward,
            completion="test output",
        )

    def test_submit_missing_manifest(self, runner, mock_client):
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["experiments", "submit"])
        assert result.exit_code != 0


class TestDatasetsListCommand:
    def test_list_registered(self, runner, mock_client):
        mock_client.list_datasets.return_value = [{"name": "my-ds", "category": "sft"}]
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["datasets", "list"])
        assert result.exit_code == 0
        mock_client.list_datasets.assert_called_once_with(search=None)

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
        mock_client.dataset_exists.return_value = {"exists": True, "id": 3}
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["datasets", "exists", "my-ds"])
        assert result.exit_code == 0
        mock_client.dataset_exists.assert_called_once_with(name="my-ds")


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

    def test_list_missing_project(self, runner, mock_client):
        from agilerl.arena.exceptions import ArenaConfigError

        mock_client.list_experiments.side_effect = ArenaConfigError(
            "No project specified.",
            sdk_hint="Pass a project name or set a default.",
            cli_hint="Use --project or set a default with 'arena projects set-default <name>'.",
        )
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["experiments", "list"])
        assert result.exit_code != 0


class TestExperimentResumeCommand:
    def test_resume(self, runner, mock_client):
        mock_client.resume_experiment.return_value = {"resumed": True}
        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main, ["experiments", "resume", "my-exp", "--max-steps", "1000"]
            )
        assert result.exit_code == 0
        mock_client.resume_experiment.assert_called_once_with(
            experiment_name="my-exp", max_steps=1000
        )

    def test_resume_missing_max_steps(self, runner, mock_client):
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["experiments", "resume", "my-exp"])
        assert result.exit_code != 0


class TestExperimentStopCommand:
    def test_stop(self, runner, mock_client):
        mock_client.stop_experiment.return_value = {"stopped": True}
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["experiments", "stop", "my-exp"])
        assert result.exit_code == 0
        mock_client.stop_experiment.assert_called_once_with("my-exp")


class TestExperimentCheckpointsCommand:
    def test_checkpoints(self, runner, mock_client):
        mock_client.list_checkpoints.return_value = [{"step": 100}]
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["experiments", "checkpoints", "my-exp"])
        assert result.exit_code == 0
        mock_client.list_checkpoints.assert_called_once_with(experiment_name="my-exp")

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

    def test_list_show_api_keys(self, runner, mock_client):
        mock_client.list_inference_deployments.return_value = [
            {"name": "dep1", "api_key": "secret123"}
        ]
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["agent", "list", "--show-api-keys"])
        assert result.exit_code == 0


class TestAgentDeployCommand:
    def test_deploy(self, runner, mock_client):
        with _patched_arena_client(mock_client):
            result = runner.invoke(main, ["agent", "deploy", "my-exp"])
        assert result.exit_code == 0
        mock_client.deploy_agent.assert_called_once_with(
            experiment_name="my-exp", checkpoint=None
        )

    def test_deploy_with_checkpoint(self, runner, mock_client):
        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main,
                ["agent", "deploy", "my-exp", "--checkpoint", "step-500"],
            )
        assert result.exit_code == 0
        mock_client.deploy_agent.assert_called_once_with(
            experiment_name="my-exp", checkpoint="step-500"
        )


class TestAgentRunCommand:
    def test_run_sets_active_agent(self, runner, mock_client):
        mock_client._ensure_inference_binding.return_value = ("http://x", "key")
        with (
            _patched_arena_client(mock_client),
            patch("agilerl.arena.cli.save_active_agent") as mock_save,
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


class TestAgentGenerateCommand:
    def test_generate_streams(self, runner, mock_client):
        mock_agent = MagicMock()
        mock_agent.generate_stream.return_value = iter(["foo", "bar"])
        mock_agent.__enter__ = MagicMock(return_value=mock_agent)
        mock_agent.__exit__ = MagicMock(return_value=False)
        mock_client.open_inference_agent.return_value = mock_agent
        with _patched_arena_client(mock_client):
            result = runner.invoke(
                main,
                ["agent", "generate", "my-dep", "--prompt", "hi"],
            )
        assert result.exit_code == 0
        mock_agent.generate_stream.assert_called_once_with("hi")
        assert result.output.strip() == "foobar"

    def test_generate_uses_active_agent(self, runner, mock_client):
        mock_agent = MagicMock()
        mock_agent.generate_stream.return_value = iter(["ok"])
        mock_agent.__enter__ = MagicMock(return_value=mock_agent)
        mock_agent.__exit__ = MagicMock(return_value=False)
        mock_client.open_inference_agent.return_value = mock_agent
        active = MagicMock(
            deployment_name="cached-dep",
            experiment_name="exp1",
            project_name=None,
        )
        with (
            _patched_arena_client(mock_client),
            patch("agilerl.arena.cli.load_active_agent", return_value=active),
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
            patch("agilerl.arena.cli.load_active_agent", return_value=None),
        ):
            result = runner.invoke(main, ["agent", "generate", "--prompt", "hi"])
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
            name="new-proj", description=None, llm_based=False
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
                    "--llm-based",
                ],
            )
        assert result.exit_code == 0
        mock_client.create_project.assert_called_once_with(
            name="llm-proj", description="An LLM project", llm_based=True
        )


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


class TestRedactAgentRows:
    def test_show_api_keys_true(self):
        rows = [{"name": "dep1", "api_key": "secret"}]
        result = _redact_agent_rows_for_display(rows, show_api_keys=True)
        assert result is rows

    def test_redacts_api_key(self):
        rows = [{"name": "dep1", "api_key": "secret", "url": "http://x"}]
        result = _redact_agent_rows_for_display(rows, show_api_keys=False)
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
        result = _redact_agent_rows_for_display(rows, show_api_keys=False)
        assert "api_key" not in result[0].get("spec", {})
        assert result[0]["spec"]["url"] == "http://y"

    def test_empty_rows(self):
        result = _redact_agent_rows_for_display([], show_api_keys=False)
        assert result == []


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
        with patch("agilerl.arena.config.build_client", return_value=mock_client) as m:
            result = runner.invoke(main, ["--api-key", "pat_abc", "logout"])
        assert result.exit_code == 0
        config_arg = m.call_args[0][0]
        assert config_arg.api_key == "pat_abc"

    def test_base_url_passed_to_config(self, runner, mock_client):
        with patch("agilerl.arena.config.build_client", return_value=mock_client) as m:
            result = runner.invoke(main, ["--base-url", "http://local:8000", "logout"])
        assert result.exit_code == 0
        config_arg = m.call_args[0][0]
        assert config_arg.base_url == "http://local:8000"

    def test_upload_timeout_passed_to_config(self, runner, mock_client):
        with patch("agilerl.arena.config.build_client", return_value=mock_client) as m:
            result = runner.invoke(main, ["--upload-timeout", "600", "logout"])
        assert result.exit_code == 0
        config_arg = m.call_args[0][0]
        assert config_arg.upload_timeout == 600

    def test_invalid_timeout_rejected(self, runner):
        result = runner.invoke(main, ["--request-timeout", "0", "login"])
        assert result.exit_code != 0


class TestFormatProfileMetrics:
    def test_known_key_gets_human_readable_label(self):
        from agilerl.arena.cli import _format_profile_metrics

        result = _format_profile_metrics({"avg_cpu_per_env": 45.123456})
        assert "Avg CPU per Env (%)" in result

    def test_unknown_key_falls_back_to_title_case(self):
        from agilerl.arena.cli import _format_profile_metrics

        result = _format_profile_metrics({"foo_bar": 1.5})
        assert "Foo Bar" in result

    def test_float_values_formatted_to_3_decimal_places(self):
        from agilerl.arena.cli import _format_profile_metrics

        result = _format_profile_metrics({"steps_per_second": 123.456789})
        assert result["Steps per Second"] == "123.457"

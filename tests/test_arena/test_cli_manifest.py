"""Tests for manifest-backed Arena CLI plumbing."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import click
import pytest

from agilerl.arena.cli_manifest import (
    build_manifest_click_command,
    pythonize_manifest_param_name,
    register_on_prem_manifest_group,
    write_binary_atomic,
)
from agilerl.arena.client import ArenaClient
from agilerl.arena.config import CommandConfig
from agilerl.arena.exceptions import ArenaValidationError


def _command_config() -> CommandConfig:
    return CommandConfig(
        api_key="test-key",
        base_url=None,
        keycloak_url=None,
        realm=None,
        client_id=None,
        request_timeout=30,
        upload_timeout=300,
    )


@pytest.fixture
def api_key_client() -> ArenaClient:
    with patch("agilerl.arena.auth.KeycloakOpenID"):
        return ArenaClient(api_key="test-key")


class TestPythonizeManifestParamName:
    def test_camel_case(self) -> None:
        assert pythonize_manifest_param_name("archivedType") == "archived_type"

    def test_snake_case(self) -> None:
        assert pythonize_manifest_param_name("num_nodes") == "num_nodes"


class TestGetCliCapabilities:
    def test_404_returns_none(self, api_key_client: ArenaClient) -> None:
        resp = MagicMock()
        resp.status_code = 404
        with patch.object(api_key_client._http, "request", return_value=resp):
            assert api_key_client.get_cli_capabilities(force_refresh=True) is None

    def test_invalid_json_body_returns_none(self, api_key_client: ArenaClient) -> None:
        resp = MagicMock()
        resp.status_code = 200
        resp.json.side_effect = ValueError("no JSON object")
        with patch.object(api_key_client._http, "request", return_value=resp):
            assert api_key_client.get_cli_capabilities(force_refresh=True) is None

    def test_wrong_schema_returns_none(self, api_key_client: ArenaClient) -> None:
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "ok": True,
            "data": {"schemaVersion": 999},
        }
        with patch.object(api_key_client._http, "request", return_value=resp):
            assert api_key_client.get_cli_capabilities(force_refresh=True) is None


class TestInvokeManifestCommand:
    def test_rejects_non_allowlisted_path(self, api_key_client: ArenaClient) -> None:
        invoke = {
            "method": "GET",
            "path": "/api/evil",
            "responseKind": "json",
            "params": [],
        }
        with pytest.raises(ArenaValidationError):
            api_key_client.invoke_manifest_command(invoke, {})

    def test_accepts_on_prem_prefix_json(self, api_key_client: ArenaClient) -> None:
        invoke = {
            "method": "POST",
            "path": "/api/cli/v1/on-prem/enable",
            "responseKind": "json",
            "params": [],
        }
        with patch.object(api_key_client, "_request", return_value={"ok": True}) as mocked:
            api_key_client.invoke_manifest_command(invoke, {})
        mocked.assert_called_once_with("POST", "/api/cli/v1/on-prem/enable")

    def test_binary_branch_uses_raw(self, api_key_client: ArenaClient) -> None:
        invoke = {
            "method": "GET",
            "path": "/api/cli/v1/on-prem/classes/deployment-setup",
            "responseKind": "binary",
            "params": [
                {
                    "name": "id",
                    "click": {"option": ["--id"], "metavar": "INT"},
                    "in": "query",
                    "type": "int",
                    "required": True,
                    "help": "",
                }
            ],
        }
        fake = (b"hello", "application/octet-stream", None)
        with patch.object(api_key_client, "_request_raw", return_value=fake) as mocked:
            out = api_key_client.invoke_manifest_command(invoke, {"id": 3})
        assert out == fake
        mocked.assert_called_once()
        _args, kwargs = mocked.call_args
        assert kwargs["params"] == {"id": 3}


class TestBuildManifestClickCommand:
    def test_help_registers_options(self) -> None:
        invoke = {
            "method": "GET",
            "path": "/api/cli/v1/on-prem/classes/get",
            "responseKind": "json",
            "params": [
                {
                    "name": "id",
                    "click": {"option": ["--id"], "metavar": "INT"},
                    "in": "query",
                    "type": "int",
                    "required": True,
                    "help": "Class id",
                }
            ],
        }
        cmd = build_manifest_click_command("get", "Fetch class", invoke)
        runner = click.testing.CliRunner()
        result = runner.invoke(cmd, ["--help"], obj=_command_config())
        assert result.exit_code == 0
        assert "--id" in result.output


class TestWriteBinaryAtomic:
    def test_writes_via_replace(self, tmp_path: Path) -> None:
        dest = tmp_path / "out.bin"
        write_binary_atomic(dest, b"abc", force=False)
        assert dest.read_bytes() == b"abc"


class TestRegisterOnPremManifestGroup:
    def test_registers_named_subgroup(self) -> None:
        @click.group()
        def root() -> None:
            pass

        register_on_prem_manifest_group(root)
        names = [n for n, _c in root.commands.items()]
        assert "on-prem" in names


CAP_FIXTURE = {
    "schemaVersion": 1,
    "features": {"onPremCli": True},
    "cli": {
        "manifestSchemaVersion": 1,
        "root": {
            "type": "group",
            "name": "on-prem",
            "help": "root",
            "children": [
                {
                    "type": "command",
                    "name": "ping",
                    "help": "Ping",
                    "invoke": {
                        "method": "GET",
                        "path": "/api/cli/v1/on-prem/provider",
                        "responseKind": "json",
                        "params": [],
                    },
                }
            ],
        },
    },
}


class TestOnPremDynamicIntegration:
    def test_lazy_group_loads_fixture_manifest(self) -> None:
        @click.group()
        def root() -> None:
            """root"""

        register_on_prem_manifest_group(root)

        client_mock = MagicMock(spec=ArenaClient)
        client_mock.get_cli_capabilities.return_value = CAP_FIXTURE

        build_patch = patch(
            "agilerl.arena.cli_manifest.build_client",
            return_value=client_mock,
        )

        runner = click.testing.CliRunner()
        with build_patch:
            res = runner.invoke(
                root,
                ["on-prem", "ping", "--help"],
                obj=_command_config(),
            )
        assert res.exit_code == 0
        client_mock.close.assert_called_once()

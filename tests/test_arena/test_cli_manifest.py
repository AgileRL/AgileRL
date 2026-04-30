"""Tests for manifest-backed Arena CLI plumbing."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import click
import pytest
from click.testing import CliRunner

from agilerl.arena.cli_manifest import (
    ArenaRootGroup,
    build_manifest_click_command,
    caps_allow_on_prem_at_root,
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


class TestCapsAllowOnPremAtRoot:
    def test_enterprise_true(self) -> None:
        assert caps_allow_on_prem_at_root({"enterprise": True})

    def test_onprem_cli_feature_without_enterprise(self) -> None:
        assert caps_allow_on_prem_at_root(
            {"enterprise": False, "features": {"onPremCli": True}},
        )

    def test_neither(self) -> None:
        assert not caps_allow_on_prem_at_root(
            {"enterprise": False, "features": {"onPremCli": False}},
        )

    def test_missing_features(self) -> None:
        assert not caps_allow_on_prem_at_root({"enterprise": False})

    def test_cli_manifest_schema_without_flags(self) -> None:
        assert not caps_allow_on_prem_at_root(
            {
                "enterprise": False,
                "features": {"onPremCli": False},
                "cli": {"manifestSchemaVersion": 1, "root": {}},
            },
        )


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

    def test_schema_version_string_one_accepted(self, api_key_client: ArenaClient) -> None:
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "ok": True,
            "data": {"schemaVersion": "1", "enterprise": False},
        }
        with patch.object(api_key_client._http, "request", return_value=resp):
            caps = api_key_client.get_cli_capabilities(force_refresh=True)
        assert caps == {"schemaVersion": "1", "enterprise": False}


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
        runner = CliRunner()
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


class TestArenaRootGroupVisibility:
    @staticmethod
    def _root_with_on_prem() -> click.Group:
        @click.group(cls=ArenaRootGroup)
        def root() -> None:
            """Arena root."""

        register_on_prem_manifest_group(root)
        return root

    def test_help_lists_on_prem_when_enterprise(self) -> None:
        root = self._root_with_on_prem()
        with patch(
            "agilerl.arena.cli_manifest.capabilities_show_on_prem_root",
            return_value=True,
        ):
            r = CliRunner().invoke(
                root,
                ["--help"],
                obj=_command_config(),
            )
        assert r.exit_code == 0
        assert "on-prem" in r.output

    def test_help_hides_on_prem_when_not_enterprise(self) -> None:
        root = self._root_with_on_prem()
        with patch(
            "agilerl.arena.cli_manifest.capabilities_show_on_prem_root",
            return_value=False,
        ):
            r = CliRunner().invoke(
                root,
                ["--help"],
                obj=_command_config(),
            )
        assert r.exit_code == 0
        assert "on-prem" not in r.output

    def test_help_hides_on_prem_when_caps_unavailable(self) -> None:
        root = self._root_with_on_prem()
        with patch(
            "agilerl.arena.cli_manifest.capabilities_show_on_prem_root",
            return_value=None,
        ):
            r = CliRunner().invoke(
                root,
                ["--help"],
                obj=_command_config(),
            )
        assert r.exit_code == 0
        assert "on-prem" not in r.output

    def test_main_help_uses_argv_before_callback_for_capabilities(self) -> None:
        """Eager ``--help`` runs before ``main`` sets ``ctx.obj``; config must come from params."""
        from agilerl.arena.cli import main

        captured: dict[str, object] = {}

        def capture(cfg: CommandConfig) -> bool:
            captured["cfg"] = cfg
            return True

        with patch(
            "agilerl.arena.cli_manifest.capabilities_show_on_prem_root",
            side_effect=capture,
        ):
            r = CliRunner().invoke(
                main,
                [
                    "--base-url",
                    "http://localhost:3001",
                    "--keycloak-url",
                    "http://localhost:8023",
                    "--api-key",
                    "arena_pat_testtoken",
                    "--help",
                ],
            )
        assert r.exit_code == 0
        cfg = captured["cfg"]
        assert isinstance(cfg, CommandConfig)
        assert cfg.api_key == "arena_pat_testtoken"
        assert cfg.base_url == "http://localhost:3001"
        assert cfg.keycloak_url == "http://localhost:8023"
        assert "on-prem" in r.output


CAP_FIXTURE_V2 = {
    "schemaVersion": 1,
    "enterprise": True,
    "features": {"onPremCli": True},
    "cli": {
        "manifestSchemaVersion": 2,
        "root": {
            "type": "group",
            "name": "on-prem",
            "help": "root",
            "children": [
                {
                    "type": "group",
                    "name": "providers",
                    "help": "providers",
                    "children": [
                        {
                            "type": "command",
                            "name": "get",
                            "help": "Get provider",
                            "invoke": {
                                "method": "GET",
                                "path": "/api/cli/v1/on-prem/provider",
                                "responseKind": "json",
                                "params": [],
                            },
                        }
                    ],
                },
                {
                    "type": "group",
                    "name": "classes",
                    "help": "classes",
                    "children": [],
                },
                {
                    "type": "group",
                    "name": "install",
                    "help": "install",
                    "children": [],
                },
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
        client_mock.get_cli_capabilities.return_value = CAP_FIXTURE_V2

        build_patch = patch(
            "agilerl.arena.cli_manifest.build_client",
            return_value=client_mock,
        )

        runner = CliRunner()
        with build_patch:
            res = runner.invoke(
                root,
                ["on-prem", "providers", "get", "--help"],
                obj=_command_config(),
            )
        assert res.exit_code == 0
        assert client_mock.get_cli_capabilities.call_count >= 1
        assert client_mock.close.call_count >= 1

    def test_install_group_includes_bootstrap(self) -> None:
        @click.group()
        def root() -> None:
            """root"""

        register_on_prem_manifest_group(root)
        client_mock = MagicMock(spec=ArenaClient)
        client_mock.get_cli_capabilities.return_value = CAP_FIXTURE_V2

        with patch(
            "agilerl.arena.cli_manifest.build_client",
            return_value=client_mock,
        ):
            res = CliRunner().invoke(
                root,
                ["on-prem", "install", "bootstrap", "--help"],
                obj=_command_config(),
            )
        assert res.exit_code == 0
        assert "--num-nodes" in res.output


class TestInstallBootstrap:
    def test_bootstrap_invokes_enable_create_bundle(self) -> None:
        from agilerl.arena.cli_on_prem_bootstrap import run_install_bootstrap

        client = MagicMock(spec=ArenaClient)
        client.invoke_manifest_command.side_effect = [
            {},
            {"id": 42},
            (b"zip-bytes", "application/zip", None),
        ]

        out = Path("/tmp/arena-bootstrap-test.zip")
        with patch(
            "agilerl.arena.cli_on_prem_bootstrap.write_binary_atomic",
        ) as write_mock:
            run_install_bootstrap(
                client,
                name="pool",
                num_nodes=2,
                output=out,
                setup_type="helm",
                archived_type="zip",
                cpus=4,
                gpus=0,
                memory="32 GB",
                description=None,
                enabled=True,
                force=False,
            )

        assert client.invoke_manifest_command.call_count == 3
        enable_call = client.invoke_manifest_command.call_args_list[0]
        assert enable_call[0][0]["path"] == "/api/cli/v1/on-prem/enable"

        create_call = client.invoke_manifest_command.call_args_list[1]
        create_body = create_call[0][1]
        assert create_body["name"] == "pool"
        assert create_body["num_nodes"] == 2
        assert create_body["metadata"]["computeResource"]["numCpus"] == 4

        bundle_call = client.invoke_manifest_command.call_args_list[2]
        assert bundle_call[0][1]["id"] == 42
        assert bundle_call[0][1]["setupType"] == "helm"
        write_mock.assert_called_once()

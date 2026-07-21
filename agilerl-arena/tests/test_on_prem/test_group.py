"""Tests for on-prem capability gating and the lazy dynamic group."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import click
import pytest
from click.testing import CliRunner

from agilerl.arena.client import ArenaClient
from agilerl.arena.config import CommandConfig
from agilerl.arena.on_prem import (
    ArenaRootGroup,
    capabilities_show_on_prem_root,
    caps_allow_on_prem_at_root,
    register_on_prem_manifest_group,
)
from agilerl.arena.on_prem.group import (
    _ON_PREM_ENSURED_META_KEY,
    OnPremDynamicGroup,
)


class TestCapsAllowOnPremAtRoot:
    def test_enterprise_true(self) -> None:
        assert caps_allow_on_prem_at_root({"enterprise": True})

    def test_onprem_cli_feature_without_enterprise(self) -> None:
        assert caps_allow_on_prem_at_root(
            {"enterprise": False, "features": {"onPremCli": True}}
        )

    def test_neither(self) -> None:
        assert not caps_allow_on_prem_at_root(
            {"enterprise": False, "features": {"onPremCli": False}}
        )

    def test_missing_features(self) -> None:
        assert not caps_allow_on_prem_at_root({"enterprise": False})


class TestCapabilitiesShowOnPremRoot:
    @pytest.mark.parametrize(
        ("caps", "expected"),
        [
            (None, None),  # capabilities unavailable
            ({"enterprise": True}, True),
            ({"enterprise": False, "features": {"onPremCli": False}}, False),
        ],
    )
    def test_resolves_visibility_and_closes_client(
        self,
        command_config: CommandConfig,
        caps: dict[str, object] | None,
        expected: bool | None,
    ) -> None:
        client_mock = MagicMock(spec=ArenaClient)
        client_mock._get_cli_capabilities.return_value = caps
        with patch(
            "agilerl.arena.on_prem.group.build_client", return_value=client_mock
        ):
            result = capabilities_show_on_prem_root(command_config)
        assert result is expected
        client_mock.close.assert_called_once()


class TestRegisterOnPremManifestGroup:
    def test_registers_named_subgroup(self) -> None:
        @click.group()
        def root() -> None:
            pass

        register_on_prem_manifest_group(root)
        assert "on-prem" in root.commands


class TestArenaRootGroupVisibility:
    @staticmethod
    def _root_with_on_prem() -> click.Group:
        @click.group(cls=ArenaRootGroup)
        def root() -> None:
            """Arena root."""

        register_on_prem_manifest_group(root)
        return root

    @pytest.mark.parametrize(
        ("visibility", "should_show"),
        [(True, True), (False, False), (None, False)],
    )
    def test_help_visibility_follows_capabilities(
        self,
        command_config: CommandConfig,
        visibility: bool | None,
        should_show: bool,
    ) -> None:
        root = self._root_with_on_prem()
        with patch(
            "agilerl.arena.on_prem.group.capabilities_show_on_prem_root",
            return_value=visibility,
        ):
            r = CliRunner().invoke(root, ["--help"], obj=command_config)
        assert r.exit_code == 0
        assert ("on-prem" in r.output) is should_show

    def test_hidden_on_prem_command_not_resolvable(
        self, command_config: CommandConfig
    ) -> None:
        # When capabilities hide on-prem, get_command must return None so the
        # command is genuinely unreachable (not just absent from --help).
        root = self._root_with_on_prem()
        with patch(
            "agilerl.arena.on_prem.group.capabilities_show_on_prem_root",
            return_value=False,
        ):
            r = CliRunner().invoke(root, ["on-prem", "--help"], obj=command_config)
        assert r.exit_code != 0
        assert "No such command" in r.output

    def test_main_help_uses_argv_before_callback_for_capabilities(self) -> None:
        """Eager ``--help`` runs before ``main`` sets ``ctx.obj``; config comes from params."""
        from agilerl.arena.cli import main

        captured: dict[str, object] = {}

        def capture(cfg: CommandConfig) -> bool:
            captured["cfg"] = cfg
            return True

        with patch(
            "agilerl.arena.on_prem.group.capabilities_show_on_prem_root",
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
                {"type": "group", "name": "install", "help": "install", "children": []},
            ],
        },
    },
}


class TestOnPremDynamicGroup:
    @staticmethod
    def _root_with_caps(
        caps: dict[str, object] | None,
    ) -> tuple[click.Group, MagicMock]:
        @click.group()
        def root() -> None:
            """root"""

        register_on_prem_manifest_group(root)
        client_mock = MagicMock(spec=ArenaClient)
        client_mock._get_cli_capabilities.return_value = caps
        return root, client_mock

    def test_lazy_group_loads_fixture_manifest(
        self, command_config: CommandConfig
    ) -> None:
        root, client_mock = self._root_with_caps(CAP_FIXTURE_V2)
        with patch(
            "agilerl.arena.on_prem.group.build_client", return_value=client_mock
        ):
            res = CliRunner().invoke(
                root, ["on-prem", "providers", "get", "--help"], obj=command_config
            )
        assert res.exit_code == 0
        assert client_mock._get_cli_capabilities.call_count >= 1
        assert client_mock.close.call_count >= 1

    def test_install_command_replaces_manifest_install(
        self, command_config: CommandConfig
    ) -> None:
        root, client_mock = self._root_with_caps(CAP_FIXTURE_V2)
        with patch(
            "agilerl.arena.on_prem.group.build_client", return_value=client_mock
        ):
            res = CliRunner().invoke(
                root, ["on-prem", "install", "--help"], obj=command_config
            )
        assert res.exit_code == 0
        assert "--manager" in res.output
        assert "helm" in res.output.lower()
        assert "install-docker" in res.output

    @pytest.mark.parametrize(
        ("caps", "expected"),
        [
            (None, "not available from this Arena server"),
            ({"schemaVersion": 999}, "does not support on-prem CLI"),
            (
                {
                    "schemaVersion": 1,
                    "enterprise": False,
                    "features": {"onPremCli": False},
                },
                "not enabled for your account",
            ),
            (
                {"schemaVersion": 1, "enterprise": True, "cli": None},
                "temporarily unavailable",
            ),
            (
                {
                    "schemaVersion": 1,
                    "enterprise": True,
                    "cli": {"manifestSchemaVersion": 1},
                },
                "too old",
            ),
            (
                {
                    "schemaVersion": 1,
                    "enterprise": True,
                    "cli": {"manifestSchemaVersion": 2},
                },
                "configuration from Arena is invalid",
            ),
        ],
    )
    def test_unavailable_capabilities_show_friendly_notice(
        self,
        command_config: CommandConfig,
        caps: dict[str, object] | None,
        expected: str,
    ) -> None:
        root, client_mock = self._root_with_caps(caps)
        with patch(
            "agilerl.arena.on_prem.group.build_client", return_value=client_mock
        ):
            res = CliRunner().invoke(root, ["on-prem", "help"], obj=command_config)
        assert res.exit_code == 0
        assert expected in res.output
        assert "/api/" not in res.output  # no backend endpoints leak

    def test_on_prem_group_help_lists_loaded_commands(
        self, command_config: CommandConfig
    ) -> None:
        # Rendering ``on-prem --help`` exercises list_commands, which lazily
        # ensures the manifest tree is loaded before listing subcommands.
        root, client_mock = self._root_with_caps(CAP_FIXTURE_V2)
        with patch(
            "agilerl.arena.on_prem.group.build_client", return_value=client_mock
        ):
            res = CliRunner().invoke(root, ["on-prem", "--help"], obj=command_config)
        assert res.exit_code == 0
        assert "providers" in res.output
        assert "install" in res.output


class TestOnPremDynamicGroupEnsure:
    """Direct unit tests for the lazy ``_ensure`` loader's guard branches."""

    def test_ensure_is_noop_when_already_ensured(
        self, command_config: CommandConfig
    ) -> None:
        group = OnPremDynamicGroup()
        ctx = click.Context(group, obj=command_config)
        ctx.meta[_ON_PREM_ENSURED_META_KEY] = True
        with patch("agilerl.arena.on_prem.group.build_client") as build_client:
            group._ensure(ctx)
        build_client.assert_not_called()

    def test_ensure_raises_without_command_config_on_root(self) -> None:
        group = OnPremDynamicGroup()
        ctx = click.Context(group, obj=None)
        with pytest.raises(click.ClickException, match="missing CommandConfig"):
            group._ensure(ctx)

    def test_ensure_skips_rebuild_when_fingerprint_unchanged(
        self, command_config: CommandConfig
    ) -> None:
        group = OnPremDynamicGroup()
        client_mock = MagicMock(spec=ArenaClient)
        client_mock._get_cli_capabilities.return_value = CAP_FIXTURE_V2
        with patch(
            "agilerl.arena.on_prem.group.build_client", return_value=client_mock
        ):
            # First call builds the tree and records the capabilities fingerprint.
            group._ensure(click.Context(group, obj=command_config))
            assert "providers" in group.commands
            # A fresh context with identical caps must short-circuit on the
            # unchanged fingerprint rather than rebuilding the command tree.
            group._ensure(click.Context(group, obj=command_config))
        assert "providers" in group.commands
        assert client_mock._get_cli_capabilities.call_count == 2

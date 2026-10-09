# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for BYOC capability gating and the lazy ``cluster`` group."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import click
import pytest
from click.testing import CliRunner

from agilerl.arena.byoc import (
    caps_allow_byoc_at_root,
    register_byoc_manifest_group,
    resolve_byoc_root_access,
)
from agilerl.arena.byoc.group import (
    BYOC_ENSURED_META_KEY,
    ArenaRootGroup,
    ByocRootAccess,
    ClusterDynamicGroup,
    _manifest_providers_node,
)
from agilerl.arena.client import ArenaClient
from agilerl.arena.config import CommandConfig


class TestCapsAllowByocAtRoot:
    def test_enterprise_true(self) -> None:
        assert caps_allow_byoc_at_root({"enterprise": True})

    def test_onprem_cli_feature_without_enterprise(self) -> None:
        assert caps_allow_byoc_at_root(
            {"enterprise": False, "features": {"onPremCli": True}}
        )

    def test_byoc_cli_feature_without_onprem_cli(self) -> None:
        assert caps_allow_byoc_at_root(
            {"enterprise": False, "features": {"byocCli": True}}
        )

    def test_neither(self) -> None:
        assert not caps_allow_byoc_at_root(
            {
                "enterprise": False,
                "features": {"onPremCli": False, "byocCli": False},
            }
        )

    def test_missing_features(self) -> None:
        assert not caps_allow_byoc_at_root({"enterprise": False})


class TestResolveByocRootAccess:
    @pytest.mark.parametrize(
        ("caps", "expected"),
        [
            ({"enterprise": True}, ByocRootAccess.ALLOWED),
            (
                {"enterprise": False, "features": {"byocCli": True}},
                ByocRootAccess.ALLOWED,
            ),
            (None, ByocRootAccess.DENIED),  # capabilities unavailable
            (
                {"enterprise": False, "features": {"onPremCli": False}},
                ByocRootAccess.DENIED,
            ),
        ],
    )
    def test_resolves_access_and_closes_client(
        self,
        command_config: CommandConfig,
        caps: dict[str, object] | None,
        expected: ByocRootAccess,
    ) -> None:
        client_mock = MagicMock(spec=ArenaClient)
        client_mock.is_authenticated = True
        client_mock._get_cli_capabilities.return_value = caps

        with patch("agilerl.arena.byoc.group.build_client", return_value=client_mock):
            result = resolve_byoc_root_access(command_config)

        assert result is expected
        client_mock.close.assert_called_once()

    def test_unauthenticated_skips_capabilities_fetch(
        self, command_config: CommandConfig
    ) -> None:
        # Arrange
        client_mock = MagicMock(spec=ArenaClient)
        client_mock.is_authenticated = False

        # Act
        with patch("agilerl.arena.byoc.group.build_client", return_value=client_mock):
            result = resolve_byoc_root_access(command_config)

        # Assert
        assert result is ByocRootAccess.UNAUTHENTICATED
        client_mock._get_cli_capabilities.assert_not_called()
        client_mock.close.assert_called_once()


class TestRegisterByocManifestGroup:
    def test_registers_only_cluster(self) -> None:
        @click.group()
        def root() -> None:
            pass

        register_byoc_manifest_group(root)
        assert "cluster" in root.commands
        assert "on-prem" not in root.commands


class TestArenaRootGroupVisibility:
    @staticmethod
    def _root_with_cluster() -> click.Group:
        @click.group(cls=ArenaRootGroup)
        def root() -> None:
            """Arena root."""

        register_byoc_manifest_group(root)
        return root

    @pytest.mark.parametrize(
        ("access", "should_show"),
        [
            (ByocRootAccess.ALLOWED, True),
            (ByocRootAccess.DENIED, False),
            (ByocRootAccess.UNAUTHENTICATED, False),
        ],
    )
    def test_help_visibility_follows_capabilities(
        self,
        command_config: CommandConfig,
        access: ByocRootAccess,
        should_show: bool,
    ) -> None:
        root = self._root_with_cluster()
        with patch(
            "agilerl.arena.byoc.group.resolve_byoc_root_access",
            return_value=access,
        ):
            r = CliRunner().invoke(root, ["--help"], obj=command_config)
        assert r.exit_code == 0
        assert ("cluster" in r.output) is should_show

    def test_byoc_cli_only_shows_cluster(self, command_config: CommandConfig) -> None:
        root = self._root_with_cluster()
        client_mock = MagicMock(spec=ArenaClient)
        client_mock.is_authenticated = True
        client_mock._get_cli_capabilities.return_value = {
            "enterprise": False,
            "features": {"byocCli": True},
        }

        with patch("agilerl.arena.byoc.group.build_client", return_value=client_mock):
            r = CliRunner().invoke(root, ["--help"], obj=command_config)

        assert r.exit_code == 0
        assert "cluster" in r.output

    def test_denied_gated_command_not_resolvable(
        self, command_config: CommandConfig
    ) -> None:
        # When capabilities hide gated groups, the command must be genuinely
        # unreachable (not just absent from --help), and Click must not offer
        # the hidden name as a correction for itself.
        root = self._root_with_cluster()

        with patch(
            "agilerl.arena.byoc.group.resolve_byoc_root_access",
            return_value=ByocRootAccess.DENIED,
        ):
            r = CliRunner().invoke(root, ["cluster", "--help"], obj=command_config)

        assert r.exit_code != 0
        assert "No such command 'cluster'." in r.output
        assert "Did you mean" not in r.output

    def test_unauthenticated_gated_command_suggests_login(
        self, command_config: CommandConfig
    ) -> None:
        # Arrange
        root = self._root_with_cluster()

        # Act
        with patch(
            "agilerl.arena.byoc.group.resolve_byoc_root_access",
            return_value=ByocRootAccess.UNAUTHENTICATED,
        ):
            r = CliRunner().invoke(root, ["cluster"], obj=command_config)

        # Assert
        assert r.exit_code != 0
        assert "arena login" in r.output
        assert "ARENA_API_KEY" in r.output
        assert "No such command" not in r.output

    def test_main_help_uses_argv_before_callback_for_capabilities(self) -> None:
        """Eager ``--help`` runs before ``main`` sets ``ctx.obj``; config comes from params."""
        from agilerl.arena.cli import main

        captured: dict[str, object] = {}

        def capture(cfg: CommandConfig) -> ByocRootAccess:
            captured["cfg"] = cfg
            return ByocRootAccess.ALLOWED

        with patch(
            "agilerl.arena.byoc.group.resolve_byoc_root_access",
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
        assert "cluster" in r.output


CAP_FIXTURE_V2 = {
    "schemaVersion": 1,
    "enterprise": False,
    "features": {"byocCli": True},
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
                                "path": "/api/cli/v1/byoc/provider",
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


class TestClusterDynamicGroup:
    @staticmethod
    def _root_with_caps(
        caps: dict[str, object] | None,
    ) -> tuple[click.Group, MagicMock]:
        @click.group()
        def root() -> None:
            """root"""

        register_byoc_manifest_group(root)
        client_mock = MagicMock(spec=ArenaClient)
        client_mock._get_cli_capabilities.return_value = caps
        return root, client_mock

    def test_lazy_group_loads_fixture_manifest(
        self, command_config: CommandConfig
    ) -> None:
        root, client_mock = self._root_with_caps(CAP_FIXTURE_V2)
        with patch("agilerl.arena.byoc.group.build_client", return_value=client_mock):
            res = CliRunner().invoke(
                root, ["cluster", "providers", "get", "--help"], obj=command_config
            )
        assert res.exit_code == 0
        assert client_mock._get_cli_capabilities.call_count >= 1
        assert client_mock.close.call_count >= 1

    @pytest.mark.parametrize(
        "caps",
        [
            None,
            {"schemaVersion": 999},
            {
                "schemaVersion": 1,
                "enterprise": False,
                "features": {"onPremCli": False},
            },
            {"schemaVersion": 1, "enterprise": True, "cli": None},
            {
                "schemaVersion": 1,
                "enterprise": True,
                "cli": {"manifestSchemaVersion": 1},
            },
            {
                "schemaVersion": 1,
                "enterprise": True,
                "cli": {"manifestSchemaVersion": 2},
            },
        ],
    )
    def test_unusable_capabilities_keep_hardcoded_commands(
        self,
        command_config: CommandConfig,
        caps: dict[str, object] | None,
    ) -> None:
        # Arrange
        root, client_mock = self._root_with_caps(caps)

        # Act
        with patch("agilerl.arena.byoc.group.build_client", return_value=client_mock):
            res = CliRunner().invoke(root, ["cluster", "--help"], obj=command_config)

        # Assert
        assert res.exit_code == 0
        cluster = root.commands["cluster"]
        assert "providers" not in cluster.commands
        assert "register" in cluster.commands
        assert "/api/" not in res.output  # no backend endpoints leak

    def test_cluster_group_help_lists_loaded_commands(
        self, command_config: CommandConfig
    ) -> None:
        # Rendering ``cluster --help`` exercises list_commands, which lazily
        # ensures the manifest tree is loaded before listing subcommands.
        root, client_mock = self._root_with_caps(CAP_FIXTURE_V2)
        with patch("agilerl.arena.byoc.group.build_client", return_value=client_mock):
            res = CliRunner().invoke(root, ["cluster", "--help"], obj=command_config)
        assert res.exit_code == 0
        assert "providers" in res.output
        cluster = root.commands["cluster"]
        assert "register" in cluster.commands
        assert "install" not in cluster.commands
        assert "classes" not in cluster.commands
        assert "clusters" not in cluster.commands


class TestClusterDynamicGroupEnsure:
    """Direct unit tests for the lazy ``_ensure`` loader's guard branches."""

    def test_ensure_is_noop_when_already_ensured(
        self, command_config: CommandConfig
    ) -> None:
        group = ClusterDynamicGroup()
        ctx = click.Context(group, obj=command_config)
        ctx.meta[BYOC_ENSURED_META_KEY] = True
        with patch("agilerl.arena.byoc.group.build_client") as build_client:
            group._ensure(ctx)
        build_client.assert_not_called()

    def test_ensure_raises_without_command_config_on_root(self) -> None:
        group = ClusterDynamicGroup()
        ctx = click.Context(group, obj=None)
        with pytest.raises(click.ClickException, match="missing CommandConfig"):
            group._ensure(ctx)

    def test_ensure_skips_rebuild_when_fingerprint_unchanged(
        self, command_config: CommandConfig
    ) -> None:
        group = ClusterDynamicGroup()
        client_mock = MagicMock(spec=ArenaClient)
        client_mock._get_cli_capabilities.return_value = CAP_FIXTURE_V2
        with patch("agilerl.arena.byoc.group.build_client", return_value=client_mock):
            # First call attaches providers and records the capabilities fingerprint.
            group._ensure(click.Context(group, obj=command_config))
            assert "providers" in group.commands
            assert "register" in group.commands
            # A fresh context with identical caps must short-circuit on the
            # unchanged fingerprint rather than rebuilding the command tree.
            group._ensure(click.Context(group, obj=command_config))
        assert "providers" in group.commands
        assert "register" in group.commands
        assert client_mock._get_cli_capabilities.call_count == 2

    def test_ensure_drops_providers_when_capabilities_become_unusable(
        self, command_config: CommandConfig
    ) -> None:
        # Arrange
        group = ClusterDynamicGroup()
        client_mock = MagicMock(spec=ArenaClient)
        client_mock._get_cli_capabilities.side_effect = [CAP_FIXTURE_V2, None]

        # Act
        with patch("agilerl.arena.byoc.group.build_client", return_value=client_mock):
            group._ensure(click.Context(group, obj=command_config))
            group._ensure(click.Context(group, obj=command_config))

        # Assert
        assert "providers" not in group.commands
        assert "register" in group.commands


class TestArenaRootGroupGetCommand:
    def test_hides_cluster_when_access_is_denied(
        self, command_config: CommandConfig
    ) -> None:
        group = ArenaRootGroup()

        @group.command("cluster")
        def cluster() -> None:
            """cluster"""

        @group.command("login")
        def login() -> None:
            """login"""

        with patch(
            "agilerl.arena.byoc.group.resolve_byoc_root_access",
            return_value=ByocRootAccess.DENIED,
        ):
            ctx = click.Context(group, obj=command_config)

            assert group.get_command(ctx, "cluster") is None
            assert group.get_command(ctx, "login") is not None


class TestManifestProvidersNode:
    def test_returns_none_when_children_omit_providers(self) -> None:
        caps = {
            "schemaVersion": 1,
            "enterprise": True,
            "cli": {
                "manifestSchemaVersion": 2,
                "root": {
                    "children": [
                        {"type": "group", "name": "install", "help": "install"},
                    ]
                },
            },
        }

        assert _manifest_providers_node(caps) is None


class TestArenaRootGroupResolveCommand:
    def test_denied_access_hides_cluster(self, command_config: CommandConfig) -> None:
        group = ArenaRootGroup()

        @group.command("cluster")
        def cluster() -> None:
            """cluster"""

        @group.command("login")
        def login() -> None:
            """login"""

        with patch(
            "agilerl.arena.byoc.group.resolve_byoc_root_access",
            return_value=ByocRootAccess.DENIED,
        ):
            result = CliRunner().invoke(group, ["cluster"], obj=command_config)

        assert result.exit_code != 0
        assert "No such command" in result.output

    def test_allowed_access_resolves_cluster(
        self, command_config: CommandConfig
    ) -> None:
        group = ArenaRootGroup()

        @group.command("cluster")
        def cluster() -> None:
            """cluster"""

        with patch(
            "agilerl.arena.byoc.group.resolve_byoc_root_access",
            return_value=ByocRootAccess.ALLOWED,
        ):
            ctx = click.Context(group, obj=command_config)
            name, command, rest = group.resolve_command(ctx, ["cluster"])

        assert name == "cluster"
        assert command is not None
        assert rest == []

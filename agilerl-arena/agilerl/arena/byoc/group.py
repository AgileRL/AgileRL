# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Capability-gated Click groups for enterprise cluster commands.

``ArenaRootGroup`` hides ``cluster`` at the CLI root unless capabilities grant
access, and points an unauthenticated caller at ``arena login``;
``ClusterDynamicGroup`` carries the hardcoded cluster commands and lazily adds
``providers`` from the server's capabilities manifest after authentication.
"""

from __future__ import annotations

import json
import logging
from enum import Enum
from typing import Any

import click

from agilerl.arena.byoc.commands import (
    build_cluster_destroy_command,
    build_cluster_generate_spec_command,
    build_cluster_plan_command,
    build_cluster_provision_command,
    build_cluster_register_command,
    build_cluster_rotate_token_command,
    build_cluster_status_command,
    build_cluster_unregister_command,
)
from agilerl.arena.cli_manifest import attach_manifest_tree
from agilerl.arena.config import (
    CommandConfig,
    _resolve_root_command_config,
    build_client,
)

logger = logging.getLogger("agilerl.arena.byoc")

CAPABILITIES_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 2
BYOC_ACCESS_META_KEY = "agilerl.arena.byoc_access"
BYOC_ENSURED_META_KEY = "agilerl.arena.byoc_ensured"


class ByocRootAccess(Enum):
    """Whether ``arena cluster`` is reachable at the CLI root, and why not."""

    ALLOWED = "allowed"
    UNAUTHENTICATED = "unauthenticated"
    DENIED = "denied"


def _capabilities_fingerprint(caps: dict[str, Any] | None) -> str:
    """Stable string for comparing capability payloads (detect upgrades / entitlement changes).

    :param caps: The capabilities document, or ``None`` if it could not be loaded.
    :type caps: dict[str, Any] | None
    :returns: A canonical string fingerprint of *caps*.
    :rtype: str
    """
    if caps is None:
        return "__missing__"
    return json.dumps(caps, sort_keys=True, separators=(",", ":"), default=str)


def caps_allow_byoc_at_root(caps: dict[str, Any]) -> bool:
    """Whether capabilities warrant exposing ``arena cluster`` at the CLI root.

    Uses strict ``is True`` checks so stray truthy JSON values do not unlock the group.
    Access is granted when ``enterprise`` is true, or when ``features.byocCli`` or
    ``features.onPremCli`` is true.

    :param caps: The capabilities document.
    :type caps: dict[str, Any]
    :returns: ``True`` if the enterprise cluster group should be exposed at the root.
    :rtype: bool
    """
    if caps.get("enterprise") is True:
        return True
    features = caps.get("features")
    if not isinstance(features, dict):
        return False
    return features.get("byocCli") is True or features.get("onPremCli") is True


def resolve_byoc_root_access(config: CommandConfig) -> ByocRootAccess:
    """Resolve whether ``arena cluster`` is reachable for the current credentials.

    :param config: The command configuration used to build a client.
    :type config: CommandConfig
    :returns: ``UNAUTHENTICATED`` without a credential, ``ALLOWED`` when
        capabilities grant BYOC CLI access, otherwise ``DENIED`` (not
        entitled, **404**, bad JSON).
    :rtype: ByocRootAccess
    """
    client = build_client(config)
    try:
        if not client.is_authenticated:
            return ByocRootAccess.UNAUTHENTICATED
        caps = client._get_cli_capabilities(force_refresh=True)
    finally:
        client.close()
    if caps is not None and caps_allow_byoc_at_root(caps):
        return ByocRootAccess.ALLOWED
    return ByocRootAccess.DENIED


class ArenaRootGroup(click.Group):
    """Arena CLI root: omit enterprise groups unless capabilities grant access."""

    GATED_ROOT_COMMANDS = frozenset({"cluster"})

    def list_commands(self, ctx: click.Context) -> list[str]:
        """List subcommands, omitting gated groups when capabilities forbid them.

        :param ctx: The current Click context.
        :type ctx: click.Context
        :returns: The sorted, visibility-filtered command names.
        :rtype: list[str]
        """
        cmds = super().list_commands(ctx)
        if self._root_access(ctx) is not ByocRootAccess.ALLOWED:
            cmds = [c for c in cmds if c not in self.GATED_ROOT_COMMANDS]
        return sorted(cmds)

    def get_command(
        self,
        ctx: click.Context,
        cmd_name: str,
    ) -> click.Command | click.Group | None:
        """Resolve a subcommand, hiding gated groups when capabilities forbid them.

        :param ctx: The current Click context.
        :type ctx: click.Context
        :param cmd_name: The requested command name.
        :type cmd_name: str
        :returns: The command, or ``None`` if absent or hidden.
        :rtype: click.Command | click.Group | None
        """
        if (
            cmd_name in self.GATED_ROOT_COMMANDS
            and self._root_access(ctx) is not ByocRootAccess.ALLOWED
        ):
            return None
        return super().get_command(ctx, cmd_name)

    def resolve_command(
        self,
        ctx: click.Context,
        args: list[str],
    ) -> tuple[str | None, click.Command | None, list[str]]:
        """Resolve the invoked subcommand, explaining a hidden gated group.

        :param ctx: The current Click context.
        :type ctx: click.Context
        :param args: The remaining command-line arguments.
        :type args: list[str]
        :returns: The resolved name, command, and remaining arguments.
        :rtype: tuple[str | None, click.Command | None, list[str]]
        :raises click.UsageError: If a gated group is hidden for these credentials.
        """
        if args and args[0] in self.GATED_ROOT_COMMANDS and not ctx.resilient_parsing:
            access = self._root_access(ctx)
            if access is ByocRootAccess.UNAUTHENTICATED:
                msg = (
                    f"'arena {args[0]}' needs a signed-in account. "
                    "Run 'arena login', or set ARENA_API_KEY, then retry."
                )
                raise click.UsageError(msg, ctx=ctx)
            if access is ByocRootAccess.DENIED:
                # Click builds its "did you mean" list from the unfiltered
                # command dict, which would offer the hidden name as its own fix.
                msg = f"No such command {args[0]!r}."
                raise click.UsageError(msg, ctx=ctx)
        return super().resolve_command(ctx, args)

    @staticmethod
    def _root_access(ctx: click.Context) -> ByocRootAccess:
        """Return BYOC root access for this invocation, caching the decision.

        :param ctx: The current Click context (used for its ``meta`` cache).
        :type ctx: click.Context
        :returns: The resolved access state.
        :rtype: ByocRootAccess
        """
        cached = ctx.meta.get(BYOC_ACCESS_META_KEY)
        if cached is None:
            cached = resolve_byoc_root_access(_resolve_root_command_config(ctx))
            ctx.meta[BYOC_ACCESS_META_KEY] = cached
        return cached


def _manifest_providers_node(caps: dict[str, Any] | None) -> dict[str, Any] | None:
    """Find the ``providers`` group in a capabilities manifest this version can use.

    :param caps: The capabilities document, or ``None`` if it could not be loaded.
    :type caps: dict[str, Any] | None
    :returns: The ``providers`` manifest node, or ``None`` when the manifest is
        missing, unsupported, not entitled, or has no ``providers`` group.
    :rtype: dict[str, Any] | None
    """
    if caps is None or caps.get("schemaVersion") != CAPABILITIES_SCHEMA_VERSION:
        return None
    if not caps_allow_byoc_at_root(caps):
        return None
    cli = caps.get("cli")
    if not isinstance(cli, dict):
        return None
    if cli.get("manifestSchemaVersion") != MANIFEST_SCHEMA_VERSION:
        return None
    root = cli.get("root")
    if not isinstance(root, dict):
        return None
    for child in root.get("children") or []:
        if child.get("type") == "group" and child.get("name") == "providers":
            return child
    return None


class ClusterDynamicGroup(click.Group):
    """``arena cluster``: hardcoded cluster commands plus a lazy ``providers`` group.

    Refetches capabilities when you use this group so entitlement changes (e.g. enterprise
    promotion) are reflected without restarting the CLI process.
    """

    def __init__(self) -> None:
        """Create the ``cluster`` group (``providers`` loads on first access)."""
        super().__init__(
            name="cluster",
            help="Manage Arena enterprise clusters.",
        )
        self.add_command(build_cluster_register_command())
        self.add_command(build_cluster_unregister_command())
        self.add_command(build_cluster_rotate_token_command())
        self.add_command(build_cluster_generate_spec_command())
        self.add_command(build_cluster_provision_command())
        self.add_command(build_cluster_plan_command())
        self.add_command(build_cluster_status_command())
        self.add_command(build_cluster_destroy_command())
        self._caps_fingerprint: str | None = None

    def _ensure(self, ctx: click.Context) -> None:
        """Load the ``providers`` subgroup from capabilities (once per context).

        Refetches capabilities and rebuilds ``providers`` only when the capabilities
        fingerprint changes; otherwise this is a no-op. The hardcoded cluster
        commands are never touched.

        :param ctx: The current Click context.
        :type ctx: click.Context
        :returns: None
        :rtype: None
        :raises click.ClickException: If the root context is missing its config.
        """
        if ctx.meta.get(BYOC_ENSURED_META_KEY):
            return
        ctx.meta[BYOC_ENSURED_META_KEY] = True

        config = ctx.find_root().obj
        if not isinstance(config, CommandConfig):
            msg = "Arena CLI internal error: missing CommandConfig on root context."
            raise click.ClickException(msg)

        client = build_client(config)
        try:
            caps = client._get_cli_capabilities(force_refresh=True)
        finally:
            client.close()

        fp = _capabilities_fingerprint(caps)
        if fp == self._caps_fingerprint:
            return
        self._caps_fingerprint = fp

        self.commands.pop("providers", None)
        providers = _manifest_providers_node(caps)
        if providers is None:
            return
        attach_manifest_tree(self, {"children": [providers]})

    def list_commands(self, ctx: click.Context) -> list[str]:
        """Ensure ``providers`` is loaded, then list subcommands.

        :param ctx: The current Click context.
        :type ctx: click.Context
        :returns: The sorted cluster subcommand names.
        :rtype: list[str]
        """
        self._ensure(ctx)
        return sorted(self.commands.keys())

    def get_command(
        self,
        ctx: click.Context,
        cmd_name: str,
    ) -> click.Command | click.Group | None:
        """Ensure ``providers`` is loaded, then resolve a subcommand by name.

        :param ctx: The current Click context.
        :type ctx: click.Context
        :param cmd_name: The requested command name.
        :type cmd_name: str
        :returns: The command, or ``None`` if absent.
        :rtype: click.Command | click.Group | None
        """
        self._ensure(ctx)
        return super().get_command(ctx, cmd_name)


def register_byoc_manifest_group(app: click.Group) -> None:
    """Attach the top-level ``cluster`` group to the Arena CLI root.

    :param app: The Arena CLI root group to attach the group to.
    :type app: click.Group
    :returns: None
    :rtype: None
    """
    app.add_command(ClusterDynamicGroup())

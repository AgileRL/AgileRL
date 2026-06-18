"""Capability-gated Click groups for the ``arena on-prem`` command tree.

``ArenaRootGroup`` hides ``on-prem`` at the CLI root unless capabilities grant
access; ``OnPremDynamicGroup`` lazily builds the on-prem subcommands from the
server's capabilities manifest after authentication.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import click

from agilerl.arena.cli_manifest import attach_manifest_tree
from agilerl.arena.config import (
    CommandConfig,
    _resolve_root_command_config,
    build_client,
)
from agilerl.arena.on_prem.commands import register_on_prem_install

logger = logging.getLogger("agilerl.arena.on_prem")

CAPABILITIES_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 2
_ON_PREM_HIDDEN_META_KEY = "agilerl.arena.on_prem_hidden"
_ON_PREM_ENSURED_META_KEY = "agilerl.arena.on_prem_ensured"


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


def caps_allow_on_prem_at_root(caps: dict[str, Any]) -> bool:
    """Whether capabilities warrant exposing ``arena on-prem`` at the CLI root.

    Uses strict ``is True`` checks so stray truthy JSON values do not unlock the group.

    :param caps: The capabilities document.
    :type caps: dict[str, Any]
    :returns: ``True`` if on-prem should be exposed at the root.
    :rtype: bool
    """
    if caps.get("enterprise") is True:
        return True
    features = caps.get("features")
    if isinstance(features, dict) and features.get("onPremCli") is True:
        return True
    return False


def capabilities_show_on_prem_root(config: CommandConfig) -> bool | None:
    """Return whether ``arena on-prem`` should appear after fetching capabilities.

    :param config: The command configuration used to build a client.
    :type config: CommandConfig
    :returns: ``True``/``False`` for visibility, or ``None`` if capabilities could
        not be loaded (no auth, **404**, bad JSON).
    :rtype: bool | None
    """
    client = build_client(config)
    try:
        caps = client._get_cli_capabilities(force_refresh=True)
    finally:
        client.close()
    if caps is None:
        return None
    return caps_allow_on_prem_at_root(caps)


class ArenaRootGroup(click.Group):
    """Arena CLI root: omit ``on-prem`` unless capabilities grant on-prem CLI access."""

    _ON_PREM = "on-prem"

    def list_commands(self, ctx: click.Context) -> list[str]:
        """List subcommands, omitting ``on-prem`` when capabilities forbid it.

        :param ctx: The current Click context.
        :type ctx: click.Context
        :returns: The sorted, visibility-filtered command names.
        :rtype: list[str]
        """
        cmds = super().list_commands(ctx)
        if self._hide_on_prem(ctx):
            cmds = [c for c in cmds if c != self._ON_PREM]
        return sorted(cmds)

    def get_command(
        self,
        ctx: click.Context,
        cmd_name: str,
    ) -> click.Command | click.Group | None:
        """Resolve a subcommand, hiding ``on-prem`` when capabilities forbid it.

        :param ctx: The current Click context.
        :type ctx: click.Context
        :param cmd_name: The requested command name.
        :type cmd_name: str
        :returns: The command, or ``None`` if absent or hidden.
        :rtype: click.Command | click.Group | None
        """
        if cmd_name == self._ON_PREM and self._hide_on_prem(ctx):
            return None
        return super().get_command(ctx, cmd_name)

    @staticmethod
    def _hide_on_prem(ctx: click.Context) -> bool:
        """Return whether ``on-prem`` should be hidden, caching the decision.

        :param ctx: The current Click context (used for its ``meta`` cache).
        :type ctx: click.Context
        :returns: ``True`` if the ``on-prem`` group should be hidden.
        :rtype: bool
        """
        cached = ctx.meta.get(_ON_PREM_HIDDEN_META_KEY)
        if cached is None:
            cfg = _resolve_root_command_config(ctx)
            cached = capabilities_show_on_prem_root(cfg) is not True
            ctx.meta[_ON_PREM_HIDDEN_META_KEY] = cached
        return cached


class OnPremDynamicGroup(click.Group):
    """Loads on-prem subcommands from capabilities after auth (lazy).

    Refetches capabilities when you use this group so entitlement changes (e.g. enterprise
    promotion) are reflected without restarting the CLI process.
    """

    def __init__(self) -> None:
        """Create the lazy ``on-prem`` group (commands load on first access)."""
        super().__init__(
            name="on-prem",
            help=(
                "Enterprise on-prem worker clusters (from Arena capabilities). "
                "Quick start: install / down / teardown. See arena on-prem install --help."
            ),
        )
        self._caps_fingerprint: str | None = None

    def _register_notice(self, message: str) -> None:
        """Register a ``help`` command that prints *message* (used when unavailable).

        :param message: The user-facing notice to display.
        :type message: str
        :returns: None
        :rtype: None
        """

        @click.command("help")
        @click.pass_context
        def help_cmd(ctx: click.Context) -> None:
            click.echo(message)
            ctx.exit(0)

        self.add_command(help_cmd, name="help")

    def _ensure(self, ctx: click.Context) -> None:
        """Load on-prem subcommands from capabilities (once per context).

        Refetches capabilities and rebuilds the command tree only when the
        capabilities fingerprint changes; otherwise this is a no-op.

        :param ctx: The current Click context.
        :type ctx: click.Context
        :returns: None
        :rtype: None
        :raises click.ClickException: If the root context is missing its config.
        """
        if ctx.meta.get(_ON_PREM_ENSURED_META_KEY):
            return
        ctx.meta[_ON_PREM_ENSURED_META_KEY] = True

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

        self.commands.clear()
        self._caps_fingerprint = fp

        if caps is None:
            self._register_notice(
                "On-prem commands are not available from this Arena server. "
                "Contact your administrator or upgrade Arena.",
            )
            return

        if caps.get("schemaVersion") != CAPABILITIES_SCHEMA_VERSION:
            self._register_notice(
                "This agilerl version does not support on-prem CLI from your "
                "Arena server. Upgrade agilerl.",
            )
            return

        if not caps_allow_on_prem_at_root(caps):
            self._register_notice(
                "On-prem CLI is not enabled for your account. "
                "Run ``arena user profile`` to check your account, or contact "
                "your administrator.",
            )
            return

        cli = caps.get("cli")
        if not isinstance(cli, dict):
            self._register_notice(
                "On-prem CLI is temporarily unavailable. "
                "Try again later or contact your administrator.",
            )
            return

        if cli.get("manifestSchemaVersion") != MANIFEST_SCHEMA_VERSION:
            self._register_notice(
                "This agilerl version is too old for on-prem CLI. Upgrade agilerl.",
            )
            return

        root = cli.get("root")
        if not isinstance(root, dict):
            self._register_notice(
                "On-prem CLI configuration from Arena is invalid. "
                "Contact your administrator.",
            )
            return

        attach_manifest_tree(self, root)
        register_on_prem_install(self)

    def list_commands(self, ctx: click.Context) -> list[str]:
        """Ensure commands are loaded, then list them.

        :param ctx: The current Click context.
        :type ctx: click.Context
        :returns: The sorted on-prem subcommand names.
        :rtype: list[str]
        """
        self._ensure(ctx)
        return sorted(self.commands.keys())

    def get_command(
        self,
        ctx: click.Context,
        cmd_name: str,
    ) -> click.Command | click.Group | None:
        """Ensure commands are loaded, then resolve one by name.

        :param ctx: The current Click context.
        :type ctx: click.Context
        :param cmd_name: The requested command name.
        :type cmd_name: str
        :returns: The command, or ``None`` if absent.
        :rtype: click.Command | click.Group | None
        """
        self._ensure(ctx)
        return super().get_command(ctx, cmd_name)


def register_on_prem_manifest_group(app: click.Group) -> None:
    """Attach the lazy ``on-prem`` command group to the Arena CLI root.

    :param app: The Arena CLI root group to attach the on-prem group to.
    :type app: click.Group
    :returns: None
    :rtype: None
    """
    app.add_command(OnPremDynamicGroup(), name="on-prem")

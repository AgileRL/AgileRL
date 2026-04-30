"""Dynamic Click commands driven by ``GET /api/cli/v1/capabilities`` manifest."""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import click

from agilerl.arena.config import (
    CommandConfig,
    build_client,
    resolve_root_command_config,
)
from agilerl.arena.exceptions import ArenaValidationError
from agilerl.arena.output import emit_result

logger = logging.getLogger(__name__)

CAPABILITIES_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 2


def handle_help_option(
    ctx: click.Context,
    param: click.Parameter,
    value: bool,
) -> None:
    """For use with ``is_eager=False`` so connection flags parse before ``--help``."""
    if not value or ctx.resilient_parsing:
        return
    click.echo(ctx.get_help(), color=ctx.color)
    ctx.exit()


def _capabilities_fingerprint(caps: dict[str, Any] | None) -> str:
    """Stable string for comparing capability payloads (detect upgrades / entitlement changes)."""
    if caps is None:
        return "__missing__"
    return json.dumps(caps, sort_keys=True, separators=(",", ":"), default=str)


def caps_allow_on_prem_at_root(caps: dict[str, Any]) -> bool:
    """Whether capabilities warrant exposing ``arena on-prem`` at the CLI root.

    Uses strict ``is True`` checks so stray truthy JSON values do not unlock the group.
    """
    if caps.get("enterprise") is True:
        return True
    features = caps.get("features")
    if isinstance(features, dict) and features.get("onPremCli") is True:
        return True
    return False


def capabilities_show_on_prem_root(config: CommandConfig) -> bool | None:
    """Return whether ``arena on-prem`` should appear after fetching capabilities.

    ``None`` means the capabilities document could not be loaded (no auth, **404**, bad JSON).
    """
    client = build_client(config)
    try:
        caps = client.get_cli_capabilities(force_refresh=True)
    finally:
        client.close()
    if caps is None:
        return None
    return caps_allow_on_prem_at_root(caps)


class ArenaRootGroup(click.Group):
    """Arena CLI root: omit ``on-prem`` unless capabilities grant on-prem CLI access."""

    _ON_PREM = "on-prem"

    def list_commands(self, ctx: click.Context) -> list[str]:
        cmds = super().list_commands(ctx)
        if self._hide_on_prem(ctx):
            cmds = [c for c in cmds if c != self._ON_PREM]
        return sorted(cmds)

    def get_command(
        self,
        ctx: click.Context,
        cmd_name: str,
    ) -> click.Command | click.Group | None:
        if cmd_name == self._ON_PREM and self._hide_on_prem(ctx):
            return None
        return super().get_command(ctx, cmd_name)

    @staticmethod
    def _hide_on_prem(ctx: click.Context) -> bool:
        cfg = resolve_root_command_config(ctx)
        return capabilities_show_on_prem_root(cfg) is not True


def pythonize_manifest_param_name(name: str) -> str:
    """Map manifest ``name`` (camelCase or snake_case) to a valid Python identifier."""
    if name == "id":
        return "id"
    if "_" in name and not any(c.isupper() for c in name):
        return name
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1)
    return s2.lower()


def write_binary_atomic(dest: Path, data: bytes, *, force: bool = False) -> None:
    """Write *data* to *dest* via a temp file and ``os.replace``."""
    dest = dest.expanduser().resolve()
    if dest.exists() and not force:
        msg = f"Refusing to overwrite existing file {dest} (use --force)."
        raise click.ClickException(msg)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, dest)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _manifest_spec_to_click_option(spec: dict[str, Any]) -> Callable[[Any], Any]:
    py_name = pythonize_manifest_param_name(spec["name"])
    opts = tuple(spec["click"]["option"])
    help_txt = spec.get("help") or ""
    required = bool(spec["required"])
    in_ = spec["in"]
    typ = spec["type"]

    if typ not in {"string", "int", "bool", "json"}:
        msg = f"Unsupported manifest param type {typ!r}"
        raise ArenaValidationError(msg, cli_hint="Upgrade the agilerl package.")

    if in_ == "client" and typ == "bool":
        return click.option(
            *opts,
            py_name,
            is_flag=True,
            default=False,
            show_default=True,
            help=help_txt,
        )

    if typ == "json":
        opt_type = str
    elif typ == "int":
        opt_type = int
    elif typ == "bool":
        opt_type = click.BOOL
    else:
        opt_type = str

    if typ == "bool" and in_ == "body" and not required:
        flag = opts[0]
        suffix = flag.lstrip("-")
        pair = f"{flag}/--no-{suffix}"
        return click.option(pair, py_name, default=None, help=help_txt)

    return click.option(
        *opts,
        py_name,
        type=opt_type,
        required=required,
        default=None,
        show_default=False,
        metavar=spec["click"].get("metavar"),
        help=help_txt,
    )


def _parse_json_cli_value(raw: str) -> Any:
    import json

    path_raw = raw.strip()
    if path_raw.startswith("@"):
        p = Path(path_raw[1:]).expanduser().resolve()
        blob = p.read_text(encoding="utf-8")
        return json.loads(blob)
    return json.loads(raw)


def build_manifest_click_command(
    name: str,
    help_txt: str | None,
    invoke: dict[str, Any],
) -> click.Command:
    param_specs = list(invoke.get("params") or [])

    def callback(config: CommandConfig, **kw: Any) -> None:
        parsed: dict[str, Any] = {}
        for spec in param_specs:
            py_key = pythonize_manifest_param_name(spec["name"])
            val = kw.get(py_key)
            if val is None and not spec["required"]:
                continue
            if val is None and spec["required"]:
                msg = f"Missing required option for {spec['name']!r}."
                raise click.UsageError(msg)
            if spec["type"] == "json" and isinstance(val, str):
                val = _parse_json_cli_value(val)
            parsed[spec["name"]] = val

        from agilerl.arena.cli import arena_client

        with arena_client(config) as client:
            result = client.invoke_manifest_command(invoke, parsed)

        response_kind = invoke.get("responseKind")
        if response_kind == "binary":
            raw_b, _ctype, _disp = result  # type: ignore[misc]
            out_p = parsed.get("outputPath")
            force = bool(parsed.get("force"))
            if out_p:
                write_binary_atomic(Path(os.fspath(out_p)), raw_b, force=force)
                click.echo(f"Wrote {out_p}")
            else:
                click.echo(raw_b.decode("utf-8", errors="replace"))
            return

        emit_result(result)

    wrapped = click.pass_obj(callback)
    for spec in reversed(param_specs):
        wrapped = _manifest_spec_to_click_option(spec)(wrapped)

    decorator_params = getattr(wrapped, "__click_params__", [])
    params: list[click.Parameter] = list(reversed(decorator_params))
    if hasattr(wrapped, "__click_params__"):
        del wrapped.__click_params__

    params.append(
        click.Option(
            ["-h", "--help"],
            is_flag=True,
            expose_value=False,
            is_eager=False,
            help="Show this message and exit.",
            callback=handle_help_option,
        ),
    )

    return click.Command(
        name=name,
        help=help_txt or "",
        callback=wrapped,
        params=params,
        add_help_option=False,
    )


def attach_manifest_tree(group: click.Group, node: dict[str, Any]) -> None:
    for child in node.get("children") or []:
        typ = child.get("type")
        if typ == "group":
            sub = click.Group(
                name=child["name"],
                help=child.get("help") or "",
            )
            attach_manifest_tree(sub, child)
            group.add_command(sub, name=child["name"])
        elif typ == "command":
            invoke = child.get("invoke") or {}
            cmd = build_manifest_click_command(
                child["name"],
                child.get("help"),
                invoke,
            )
            group.add_command(cmd, name=child["name"])
        else:
            logger.warning("Skipping unknown manifest node type %r", typ)


class OnPremDynamicGroup(click.Group):
    """Loads on-prem subcommands from capabilities after auth (lazy).

    Refetches capabilities when you use this group so entitlement changes (e.g. enterprise
    promotion) are reflected without restarting the CLI process.
    """

    def __init__(self) -> None:
        super().__init__(
            name="on-prem",
            help=(
                "Enterprise on-prem worker clusters (from Arena capabilities). "
                "Quick start: install / teardown. See arena on-prem install --help."
            ),
        )
        self._caps_fingerprint: str | None = None

    def _register_notice(self, message: str) -> None:
        @click.command("help")
        @click.pass_context
        def help_cmd(ctx: click.Context) -> None:
            click.echo(message)
            ctx.exit(0)

        self.add_command(help_cmd, name="help")

    def _ensure(self, ctx: click.Context) -> None:
        config = ctx.find_root().obj
        if not isinstance(config, CommandConfig):
            msg = "Arena CLI internal error: missing CommandConfig on root context."
            raise click.ClickException(msg)

        client = build_client(config)
        try:
            caps = client.get_cli_capabilities(force_refresh=True)
        finally:
            client.close()

        fp = _capabilities_fingerprint(caps)
        if fp == self._caps_fingerprint:
            return

        self.commands.clear()
        self._caps_fingerprint = fp

        if caps is None:
            self._register_notice(
                "Arena server has no CLI capabilities document at "
                "/api/cli/v1/capabilities (upgrade the platform). "
                "See REST docs for on-prem endpoints.",
            )
            return

        if caps.get("schemaVersion") != CAPABILITIES_SCHEMA_VERSION:
            self._register_notice(
                "Capabilities schemaVersion is not supported by this CLI — upgrade agilerl.",
            )
            return

        if not caps_allow_on_prem_at_root(caps):
            self._register_notice(
                "On-prem CLI is not enabled for this account "
                "(need ``enterprise: true`` or ``features.onPremCli: true``). "
                "Confirm ``GET /api/cli/v1/capabilities`` for your token.",
            )
            return

        cli = caps.get("cli")
        if not isinstance(cli, dict):
            self._register_notice("Capabilities response missing cli manifest.")
            return

        if cli.get("manifestSchemaVersion") != MANIFEST_SCHEMA_VERSION:
            self._register_notice(
                "CLI manifestSchemaVersion is not supported — upgrade agilerl.",
            )
            return

        root = cli.get("root")
        if not isinstance(root, dict):
            self._register_notice("Malformed CLI manifest (missing root).")
            return

        attach_manifest_tree(self, root)
        from agilerl.arena.cli_on_prem_install import register_on_prem_install

        register_on_prem_install(self)

    def list_commands(self, ctx: click.Context) -> list[str]:
        self._ensure(ctx)
        return sorted(self.commands.keys())

    def get_command(
        self,
        ctx: click.Context,
        cmd_name: str,
    ) -> click.Command | click.Group | None:
        self._ensure(ctx)
        return super().get_command(ctx, cmd_name)


def register_on_prem_manifest_group(app: click.Group) -> None:
    """Attach the lazy ``on-prem`` command group to the Arena CLI root."""
    app.add_command(OnPremDynamicGroup(), name="on-prem")

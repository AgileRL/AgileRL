"""Dynamic Click commands driven by ``GET /api/cli/v1/capabilities`` manifest."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Callable

import click

from agilerl.arena.config import CommandConfig
from agilerl.arena.exceptions import ArenaValidationError
from agilerl.arena.output import emit_result

logger = logging.getLogger(__name__)

CAPABILITIES_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1


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
                raise click.UsageError(f"Missing required option for {spec['name']!r}.")
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

    return click.Command(name=name, help=help_txt or "", callback=wrapped)


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
    """Loads on-prem subcommands from capabilities after auth (lazy)."""

    def __init__(self) -> None:
        super().__init__(
            name="on-prem",
            help="Enterprise on-prem cluster helpers (loaded from Arena capabilities).",
        )
        self._lazy_ready = False

    def _register_notice(self, message: str) -> None:
        @click.command("help")
        @click.pass_context
        def help_cmd(ctx: click.Context) -> None:
            click.echo(message)
            ctx.exit(0)

        self.add_command(help_cmd, name="help")

    def _ensure(self, ctx: click.Context) -> None:
        if self._lazy_ready:
            return
        self._lazy_ready = True

        from agilerl.arena.config import build_client

        config = ctx.find_root().obj
        if not isinstance(config, CommandConfig):
            raise click.ClickException(
                "Arena CLI internal error: missing CommandConfig on root context.",
            )

        client = build_client(config)
        try:
            caps = client.get_cli_capabilities()
        finally:
            client.close()

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

        if not caps.get("features", {}).get("onPremCli"):
            self._register_notice(
                "On-prem CLI commands are hidden by the server (features.onPremCli is false). "
                "Typical causes: your org has no Stripe billing sync row, "
                "stripe_billing_sync.enterprise is false, and there is no on-prem provider yet. "
                "Ask your admin to set enterprise on your org’s billing sync row (or deploy a "
                "server build that publishes the manifest when billing sync is absent)."
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

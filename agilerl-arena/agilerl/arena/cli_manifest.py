# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Generic machinery for building Click commands from a server manifest node.

Turns a manifest command/group tree (from ``GET /api/cli/v1/capabilities``) into
runnable :class:`click.Command` objects. The on-prem capability gating that drives
*which* manifest gets loaded lives in :mod:`agilerl.arena.on_prem.group`.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click

from agilerl.arena.client import ManifestInvoke, ManifestParamSpec
from agilerl.arena.config import CommandConfig, arena_client
from agilerl.arena.exceptions import ArenaValidationError
from agilerl.arena.output import emit_result
from agilerl.arena.typing import JSONValue

logger = logging.getLogger(__name__)

ALLOWED_OPTION_TYPES = {"string", "int", "bool", "json"}
# Manifest option type -> the Click parameter type to parse it with.
CLICK_OPTION_TYPES: dict[str, type[str | int] | click.ParamType] = {
    "json": str,
    "int": int,
    "bool": click.BOOL,
    "string": str,
}


def handle_help_option(
    ctx: click.Context,
    param: click.Parameter,
    value: bool,
) -> None:
    """For use with ``is_eager=False`` so connection flags parse before ``--help``.

    :param ctx: The current Click context.
    :type ctx: click.Context
    :param param: The Click parameter this callback is attached to.
    :type param: click.Parameter
    :param value: Whether ``--help`` was passed.
    :type value: bool
    :returns: None
    :rtype: None
    """
    if not value or ctx.resilient_parsing:
        return
    click.echo(ctx.get_help(), color=ctx.color)
    ctx.exit()


def pythonize_manifest_param_name(name: str) -> str:
    """Map manifest ``name`` (camelCase or snake_case) to a valid Python identifier.

    :param name: The manifest parameter name.
    :type name: str
    :returns: The snake_case Python identifier.
    :rtype: str
    """
    if name == "id":
        return "id"
    if "_" in name and not any(c.isupper() for c in name):
        return name
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1)
    return s2.lower()


def write_binary_atomic(dest: Path, data: bytes, *, force: bool = False) -> None:
    """Write *data* to *dest* via a temp file and ``os.replace``.

    :param dest: The destination file path (``~`` is expanded, path resolved).
    :type dest: Path
    :param data: The bytes to write.
    :type data: bytes
    :param force: If ``True``, overwrite an existing file.
    :type force: bool
    :returns: None
    :rtype: None
    :raises click.ClickException: If *dest* exists and *force* is ``False``.
    """
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


def _option_help(spec: ManifestParamSpec) -> str:
    """Return the help text for a param spec (empty string if unset).

    :param spec: The manifest parameter spec.
    :type spec: ManifestParamSpec
    :returns: The help text, or ``""``.
    :rtype: str
    """
    return spec.get("help") or ""


def _client_flag_option(spec: ManifestParamSpec) -> Callable[[Any], Any]:
    """Client-side boolean → a plain ``is_flag`` switch.

    :param spec: The manifest parameter spec.
    :type spec: ManifestParamSpec
    :returns: A ``click.option`` decorator.
    :rtype: Callable[[Any], Any]
    """
    return click.option(
        *tuple(spec["click"]["option"]),
        pythonize_manifest_param_name(spec["name"]),
        is_flag=True,
        default=False,
        show_default=True,
        help=_option_help(spec),
    )


def _body_bool_pair_option(spec: ManifestParamSpec) -> Callable[[Any], Any]:
    """Optional body boolean → a ``--x/--no-x`` toggle defaulting to unset.

    :param spec: The manifest parameter spec.
    :type spec: ManifestParamSpec
    :returns: A ``click.option`` decorator.
    :rtype: Callable[[Any], Any]
    """
    flag = next(iter(spec["click"]["option"]))
    pair = f"{flag}/--no-{flag.lstrip('-')}"
    return click.option(
        pair,
        pythonize_manifest_param_name(spec["name"]),
        default=None,
        help=_option_help(spec),
    )


def _typed_option(spec: ManifestParamSpec) -> Callable[[Any], Any]:
    """Any other param → a typed option (string/int/bool/json).

    :param spec: The manifest parameter spec.
    :type spec: ManifestParamSpec
    :returns: A ``click.option`` decorator.
    :rtype: Callable[[Any], Any]
    """
    return click.option(
        *tuple(spec["click"]["option"]),
        pythonize_manifest_param_name(spec["name"]),
        type=CLICK_OPTION_TYPES[spec["type"]],
        required=bool(spec["required"]),
        default=None,
        show_default=False,
        metavar=spec["click"].get("metavar"),
        help=_option_help(spec),
    )


@dataclass(frozen=True)
class OptionRule:
    """A predicate over a param spec and the Click-option builder it selects."""

    match: Callable[[ManifestParamSpec], bool]
    build: Callable[[ManifestParamSpec], Callable[[Any], Any]]


OPTION_RULES: tuple[OptionRule, ...] = (
    OptionRule(
        lambda spec: spec["in"] == "client" and spec["type"] == "bool",
        _client_flag_option,
    ),
    OptionRule(
        lambda spec: (
            spec["type"] == "bool" and spec["in"] == "body" and not spec["required"]
        ),
        _body_bool_pair_option,
    ),
)


def _manifest_spec_to_click_option(spec: ManifestParamSpec) -> Callable[[Any], Any]:
    """Build the ``click.option`` decorator for a single manifest param spec.

    Validates the declared type, then dispatches to the first matching rule in
    :data:`OPTION_RULES`, falling back to a plain typed option.

    :param spec: The manifest parameter spec.
    :type spec: ManifestParamSpec
    :returns: A ``click.option`` decorator for the parameter.
    :rtype: Callable[[Any], Any]
    :raises ArenaValidationError: If the declared option type is unsupported.
    """
    if spec["type"] not in ALLOWED_OPTION_TYPES:
        msg = f"Unsupported on-prem option type {spec['type']!r}"
        raise ArenaValidationError(
            msg,
            cli_hint="Upgrade agilerl — the server sent an on-prem "
            "configuration this version can't use.",
        )
    for rule in OPTION_RULES:
        if rule.match(spec):
            return rule.build(spec)
    return _typed_option(spec)


def _parse_json_cli_value(raw: str) -> JSONValue:
    """Parse a JSON CLI value, or load JSON from a file when prefixed with ``@``.

    :param raw: The raw option string; a leading ``@`` reads JSON from that path.
    :type raw: str
    :returns: The decoded JSON value.
    :rtype: JSONValue
    """
    path_raw = raw.strip()
    if path_raw.startswith("@"):
        p = Path(path_raw[1:]).expanduser().resolve()
        blob = p.read_text(encoding="utf-8")
        return json.loads(blob)
    return json.loads(raw)


def build_manifest_click_command(
    name: str,
    help_txt: str | None,
    invoke: ManifestInvoke,
) -> click.Command:
    """Turn one manifest command node into a runnable :class:`click.Command`.

    Adds a Click option per declared param, wires a callback that calls the
    server via :meth:`ArenaClient._invoke_manifest_command` (writing binary
    responses to ``--output-path`` or echoing JSON), and attaches a
    non-eager ``--help`` so connection flags parse first.

    :param name: The command name.
    :type name: str
    :param help_txt: The command help text, or ``None``.
    :type help_txt: str | None
    :param invoke: The call descriptor (method, path, responseKind, params).
    :type invoke: ManifestInvoke
    :returns: The runnable Click command.
    :rtype: click.Command
    """
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

        with arena_client(config) as client:
            result = client._invoke_manifest_command(invoke, parsed)

        response_kind = invoke.get("responseKind")
        if response_kind == "binary":
            raw_b, _ctype, _disp = result  # type: ignore[misc]
            out_p = parsed.get("outputPath")
            force = bool(parsed.get("force"))
            if out_p:
                write_binary_atomic(Path(os.fspath(out_p)), raw_b, force=force)
                logger.info("Wrote %s", out_p)
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
    """Recursively attach manifest ``group``/``command`` children onto *group*.

    Nested groups recurse; command nodes become commands via
    :func:`build_manifest_click_command`; unknown node types are skipped.

    :param group: The Click group to attach children onto.
    :type group: click.Group
    :param node: The manifest node whose ``children`` are attached.
    :type node: dict[str, Any]
    :returns: None
    :rtype: None
    """
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
            invoke: ManifestInvoke = child.get("invoke") or {}
            cmd = build_manifest_click_command(
                child["name"],
                child.get("help"),
                invoke,
            )
            group.add_command(cmd, name=child["name"])
        else:
            logger.warning("Skipping unknown manifest node type %r", typ)

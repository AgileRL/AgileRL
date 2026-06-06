"""Shared CLI machinery for AgileRL benchmarking scripts.

This module centralises the argparse plumbing common to *every* benchmarking
script so each one can expose a standardised set of command-line flags — a
config path, ``INIT_HP`` / ``MUTATION_PARAMS`` / ``NET_CONFIG`` overrides, and
Weights & Biases settings — without re-implementing the boilerplate.

Two tiers build on this:

* **Classic RL benchmarks** (``benchmarking_on_policy`` etc.) use this module
  directly via :func:`build_classic_parser` / :func:`resolve_classic`.
* **LLM benchmarks** layer the richer dataclass machinery from
  :mod:`benchmark_cli_llm` on top, reusing the W&B / override helpers and the
  dataclass-to-argparse bridge (:func:`add_dataclass_arguments`) here.

It is kept deliberately dependency-light (standard library + PyYAML only) so it
imports without ``torch`` / ``transformers`` and can be unit-tested in
isolation. The heavyweight objects (``VLLMConfig``, quantization configs, the
agent population) are built later inside each script's ``main()``.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import typing
from dataclasses import dataclass
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

import yaml

# Default config sections every benchmark YAML is expected to expose. Classic RL
# benchmarks additionally carry a ``NET_CONFIG`` section (see
# ``CLASSIC_SECTIONS``); LLM benchmarks only use the first two.
INIT_HP_SECTION = "INIT_HP"
MUTATION_SECTION = "MUTATION_PARAMS"
NET_CONFIG_SECTION = "NET_CONFIG"

CORE_SECTIONS: tuple[str, ...] = (INIT_HP_SECTION, MUTATION_SECTION)
CLASSIC_SECTIONS: tuple[str, ...] = (
    INIT_HP_SECTION,
    MUTATION_SECTION,
    NET_CONFIG_SECTION,
)

# Map a config-section name to its repeatable ``KEY=VALUE`` override flag. Kept
# explicit (rather than derived) so the established ``--init-hp-override`` /
# ``--mutation-override`` spellings stay stable.
_SECTION_OVERRIDE_FLAG: dict[str, str] = {
    INIT_HP_SECTION: "--init-hp-override",
    MUTATION_SECTION: "--mutation-override",
    NET_CONFIG_SECTION: "--net-config-override",
}


def _section_override_flag(section: str) -> str:
    return _SECTION_OVERRIDE_FLAG.get(
        section, f"--{section.lower().replace('_', '-')}-override"
    )


def _section_override_dest(section: str) -> str:
    # Strip the leading "--" and convert kebab to snake for the argparse dest.
    return _section_override_flag(section)[2:].replace("-", "_")


def parse_key_value(raw: str) -> tuple[str, object]:
    """Parse a ``KEY=VALUE`` override token.

    The key is upper-cased to match the convention of the YAML config sections;
    the value is parsed as YAML, so ``GAE_LAMBDA=1`` yields an int and
    ``TARGET_MODULES=[q_proj]`` yields a list. Raises
    :class:`argparse.ArgumentTypeError` on malformed input so argparse surfaces
    a clean error message.
    """
    if "=" not in raw:
        msg = f"expected KEY=VALUE, got {raw!r}"
        raise argparse.ArgumentTypeError(msg)
    key, _, value_raw = raw.partition("=")
    key = key.strip().upper()
    if not key:
        msg = f"empty config key in {raw!r}"
        raise argparse.ArgumentTypeError(msg)
    value = yaml.safe_load(value_raw.strip())
    return key, value


def load_config(path: str) -> dict[str, Any]:
    """Load a benchmark YAML config file into a plain dict."""
    with open(path) as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        msg = (
            f"Config {path!r} did not parse to a mapping (got {type(config).__name__})."
        )
        raise ValueError(msg)
    return config


# --------------------------------------------------------------------------- #
# Weights & Biases
# --------------------------------------------------------------------------- #
@dataclass
class WandbSettings:
    """Resolved W&B options, decoupled from argparse so callers stay tidy."""

    enabled: bool
    project: str
    entity: str | None
    run_name: str | None
    api_key: str | None = None


def add_wandb_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the standard W&B flags shared by all benchmarks.

    ``--no-wandb`` disables logging; project / entity / run-name fall back to
    the ``WANDB_PROJECT`` / ``WANDB_ENTITY`` / ``WANDB_RUN_NAME`` environment
    variables so CI and sweeps can set them without touching the command line.
    """
    group = parser.add_argument_group("Weights & Biases")
    group.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging.",
    )
    group.add_argument(
        "--wandb-project",
        type=str,
        default=os.environ.get("WANDB_PROJECT", "AgileRL"),
        help="W&B project (default: AgileRL or $WANDB_PROJECT).",
    )
    group.add_argument(
        "--wandb-entity",
        type=str,
        default=os.environ.get("WANDB_ENTITY"),
        help="W&B entity / team (optional; fallback $WANDB_ENTITY).",
    )
    group.add_argument(
        "--wandb-run-name",
        type=str,
        default=os.environ.get("WANDB_RUN_NAME"),
        help="W&B run name (optional; fallback $WANDB_RUN_NAME).",
    )


def build_wandb_settings(args: argparse.Namespace) -> WandbSettings:
    """Collapse the W&B flags (plus ``$WANDB_API_KEY``) into a settings object."""
    return WandbSettings(
        enabled=not getattr(args, "no_wandb", False),
        project=getattr(args, "wandb_project", "AgileRL"),
        entity=getattr(args, "wandb_entity", None),
        run_name=getattr(args, "wandb_run_name", None),
        api_key=os.environ.get("WANDB_API_KEY"),
    )


# --------------------------------------------------------------------------- #
# Config + section overrides
# --------------------------------------------------------------------------- #
def add_config_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_config: str,
    sections: tuple[str, ...] = CORE_SECTIONS,
) -> None:
    """Add ``--config``, ``--print-config`` and a ``KEY=VALUE`` override flag
    per config section.

    :param default_config: Default YAML path used when ``--config`` is omitted.
    :param sections: Config sections to expose override flags for. ``INIT_HP``
        and ``MUTATION_PARAMS`` are standard; classic RL benchmarks add
        ``NET_CONFIG``.
    """
    config_help = (
        f"Path to the YAML config file (default: {default_config})."
        if default_config is not None
        else "Path to a YAML config file (optional; defaults to the script's built-in config)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=default_config,
        help=config_help,
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the fully-resolved config (after overrides) as YAML and exit.",
    )
    group = parser.add_argument_group(
        "Config overrides (override values loaded from the YAML config)"
    )
    for section in sections:
        flag = _section_override_flag(section)
        group.add_argument(
            flag,
            dest=_section_override_dest(section),
            action="append",
            default=[],
            metavar="KEY=VALUE",
            type=parse_key_value,
            help=(
                f"Override a {section} entry; VALUE is parsed as YAML "
                f"(e.g. {flag} LR=1e-4). Repeatable."
            ),
        )


def apply_section_overrides(
    config: dict[str, Any],
    args: argparse.Namespace,
    *,
    sections: tuple[str, ...] = CORE_SECTIONS,
) -> None:
    """Apply the repeatable ``KEY=VALUE`` override flags onto a loaded config.

    Missing sections are created on demand so a benchmark whose YAML omits, say,
    ``NET_CONFIG`` can still receive ``--net-config-override`` values.
    """
    for section in sections:
        overrides = getattr(args, _section_override_dest(section), None) or []
        if not overrides:
            continue
        target = config.setdefault(section, {})
        for key, value in overrides:
            target[key] = value


def maybe_print_config_and_exit(
    config: dict[str, Any],
    args: argparse.Namespace,
    *,
    sections: tuple[str, ...] | None = None,
) -> None:
    """If ``--print-config`` was passed, dump the (sub)config as YAML and exit.

    :param sections: When given, only these sections are printed; otherwise the
        whole resolved config is dumped.
    """
    if not getattr(args, "print_config", False):
        return
    if sections is None:
        payload: dict[str, Any] = config
    else:
        payload = {s: config[s] for s in sections if s in config}
    yaml.safe_dump(payload, sys.stdout, sort_keys=False, default_flow_style=False)
    raise SystemExit(0)


# --------------------------------------------------------------------------- #
# Classic (non-LLM) RL benchmark convenience entry point
# --------------------------------------------------------------------------- #
def build_classic_parser(
    *,
    description: str,
    default_config: str | None,
    sections: tuple[str, ...] = CLASSIC_SECTIONS,
) -> argparse.ArgumentParser:
    """Build a standard parser for a classic (non-LLM) RL benchmark.

    Wires the config + override flags and the W&B group. Scripts may add their
    own flags to the returned parser before calling :func:`resolve_classic`.
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_config_arguments(parser, default_config=default_config, sections=sections)
    add_wandb_arguments(parser)
    return parser


def resolve_classic(
    parser: argparse.ArgumentParser,
    *,
    sections: tuple[str, ...] = CLASSIC_SECTIONS,
    argv: list[str] | None = None,
    wandb_section: str = INIT_HP_SECTION,
    wandb_key: str = "WANDB",
    base_config: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], argparse.Namespace]:
    """Parse args, load the config, apply overrides and resolve W&B.

    The ``--no-wandb`` flag is folded into the config (``INIT_HP['WANDB']`` by
    default) so classic training loops that read ``wb=INIT_HP['WANDB']`` pick it
    up with no further wiring. ``--print-config`` is honoured here (prints and
    exits). Returns the resolved config dict and the parsed namespace.

    :param base_config: Fallback config used when ``--config`` is not given (for
        scripts that define their config inline rather than in a YAML file). A
        shallow copy is taken so the caller's literal is not mutated.
    """
    args = parser.parse_args(argv)
    if getattr(args, "config", None):
        config = load_config(args.config)
    elif base_config is not None:
        config = {
            key: dict(value) if isinstance(value, dict) else value
            for key, value in base_config.items()
        }
    else:
        parser.error("no --config given and the script has no built-in config")
    apply_section_overrides(config, args, sections=sections)
    if getattr(args, "no_wandb", False):
        config.setdefault(wandb_section, {})[wandb_key] = False
    maybe_print_config_and_exit(config, args, sections=sections)
    return config, args


# --------------------------------------------------------------------------- #
# Dataclass <-> argparse bridge
# --------------------------------------------------------------------------- #
# Generic helpers that turn a (config) dataclass into a set of flat argparse
# flags and back. The dataclass is the single source of truth: a field's type
# drives the flag's type / choices, and ``<FIELD>.upper()`` is the config key it
# maps to. LLM benchmarks use this so the hyperparameter flags (``--lr``,
# ``--group-size`` …) are generated from the dataclass instead of a
# hand-maintained spec table.


def _flag_name(field_name: str, *, flag_prefix: str) -> str:
    return f"--{flag_prefix}{field_name.replace('_', '-')}"


def _yaml_value(raw: str) -> Any:
    """Parse a CLI token as YAML (so ``[0.8, 2.0]`` becomes a list, ``0.3`` a
    float). Used for union-typed fields such as ``clip_coef``."""
    return yaml.safe_load(raw)


def _unwrap_optional(annotation: Any) -> tuple[Any, bool]:
    """Return ``(inner, is_optional)`` for ``Optional[X]`` / ``X | None``."""
    if get_origin(annotation) is Union:
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return args[0], True
    return annotation, False


def _add_field_argument(
    group: argparse._ArgumentGroup | argparse.ArgumentParser,
    field: dataclasses.Field,
    annotation: Any,
    *,
    flag: str,
    dest: str,
    help_text: str,
) -> None:
    """Add a single argparse flag derived from a dataclass field's type.

    All generated flags default to ``None`` so a caller can distinguish
    "user passed this" from "left at the config / dataclass default" — the
    field's own default is only applied when neither config nor CLI sets it.
    """
    inner, _optional = _unwrap_optional(annotation)
    origin = get_origin(inner)

    common = {"dest": dest, "default": None, "help": help_text}

    if inner is bool:
        group.add_argument(flag, action=argparse.BooleanOptionalAction, **common)
    elif origin is Literal:
        group.add_argument(
            flag, type=str, choices=[str(c) for c in get_args(inner)], **common
        )
    elif origin in (list, typing.List):
        group.add_argument(flag, nargs="+", metavar="VALUE", **common)
    elif origin is Union:
        # e.g. Union[float, list[float]] (clip_coef): accept a YAML scalar or list.
        group.add_argument(flag, type=_yaml_value, metavar="YAML", **common)
    elif inner is int:
        group.add_argument(flag, type=int, **common)
    elif inner is float:
        group.add_argument(flag, type=float, **common)
    else:
        group.add_argument(flag, type=str, **common)


def add_dataclass_arguments(
    parser: argparse.ArgumentParser,
    dc_type: type,
    *,
    title: str,
    flag_prefix: str = "",
    dest_prefix: str = "",
    skip: frozenset[str] = frozenset(),
    flag_overrides: dict[str, str] | None = None,
) -> None:
    """Register a flat argparse flag for each field of ``dc_type``.

    :param title: Argument-group title shown in ``--help``.
    :param flag_prefix: Prepended to every flag (e.g. ``"mut-"`` →
        ``--mut-no-mut``).
    :param dest_prefix: Prepended to every argparse ``dest`` to namespace the
        values (e.g. ``"hp_"`` → ``args.hp_lr``), avoiding collisions between
        sections / runtime flags.
    :param skip: Field names to omit (e.g. fields driven by a dedicated runtime
        flag such as ``quantization``).
    :param flag_overrides: ``{field_name: "--custom-flag"}`` for the rare field
        whose desired flag differs from the auto-derived kebab name.
    """
    flag_overrides = flag_overrides or {}
    group = parser.add_argument_group(title)
    hints = get_type_hints(dc_type)
    for field in dataclasses.fields(dc_type):
        if field.name in skip:
            continue
        flag = flag_overrides.get(
            field.name, _flag_name(field.name, flag_prefix=flag_prefix)
        )
        dest = f"{dest_prefix}{field.name}"
        annotation = hints.get(field.name, field.type)
        _add_field_argument(
            group,
            field,
            annotation,
            flag=flag,
            dest=dest,
            help_text=f"Override {field.name.upper()}.",
        )


def apply_dataclass_overrides(
    instance: Any,
    args: argparse.Namespace,
    *,
    dest_prefix: str = "",
    skip: frozenset[str] = frozenset(),
) -> None:
    """Apply CLI flag values (from :func:`add_dataclass_arguments`) onto a
    dataclass instance, leaving fields the user did not set untouched."""
    for field in dataclasses.fields(instance):
        if field.name in skip:
            continue
        value = getattr(args, f"{dest_prefix}{field.name}", None)
        if value is not None:
            setattr(instance, field.name, value)


def dataclass_from_mapping(
    dc_type: type,
    mapping: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Build a ``dc_type`` instance from a config mapping (case-insensitive).

    Keys matching a dataclass field (compared lower-case) populate the instance;
    every other key is returned verbatim in the ``unknown`` dict so callers can
    pass it through to the final config without losing it.
    """
    field_names = {f.name for f in dataclasses.fields(dc_type)}
    known: dict[str, Any] = {}
    unknown: dict[str, Any] = {}
    for key, value in (mapping or {}).items():
        lowered = key.lower()
        if lowered in field_names:
            known[lowered] = value
        else:
            unknown[key] = value
    return dc_type(**known), unknown


def dataclass_to_upper_dict(instance: Any) -> dict[str, Any]:
    """Serialise a dataclass instance to an ``UPPER_SNAKE``-keyed config dict,
    dropping ``None`` values so unset optional fields fall back to the
    algorithm's own defaults downstream."""
    out: dict[str, Any] = {}
    for field in dataclasses.fields(instance):
        value = getattr(instance, field.name)
        if value is not None:
            out[field.name.upper()] = value
    return out

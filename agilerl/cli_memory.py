# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
r"""``arena memory`` commands: size a training manifest against a GPU on Arena.

Exit 0 if both phases fit (or a value is solved), 3 if over budget (or no
value fits), 2 on a usage error.
"""

from __future__ import annotations

import json
import logging
import math
import sys
from dataclasses import dataclass
from typing import Any

import click
import yaml
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from agilerl.config import CommandConfig, arena_client
from agilerl.estimator_api import EXIT_OK, EXIT_USAGE
from agilerl.models.verdict import read_manifest
from agilerl.output import JSON_OUTPUT_META_KEY, json_output_enabled

logger = logging.getLogger(__name__)

GiB = 1024**3
# Cells a phase bar spans at the device's usable capacity.
BAR_WIDTH = 56
# Phase and advice panels stay inside this many columns.
PANEL_WIDTH = 78
# One colour per component, cycled.
BAR_COLOURS = (
    "#4e79a7",
    "#f28e2b",
    "#59a14f",
    "#b07aa1",
    "#edc948",
    "#76b7b2",
    "#e15759",
    "#9c755f",
)
# Per-component glyphs without colour, so segments stay distinguishable.
BAR_GLYPHS = ("#", "=", "+", "*", "o", ":", "~", ".")


@dataclass(frozen=True)
class ReportGlyphs:
    """Characters the report draws for one console."""

    # One glyph per component, not colour.
    per_component: bool
    free: str
    separator: str


def report_glyphs(console: Console) -> ReportGlyphs:
    """Solid blocks for a colour UTF-8 console; ASCII when it cannot encode Unicode."""
    unicode = console.encoding.lower().startswith("utf")
    colour = console.color_system is not None and not console.no_color
    if not unicode:
        return ReportGlyphs(per_component=True, free="-", separator=" | ")
    if not colour:
        return ReportGlyphs(per_component=True, free="·", separator=" · ")
    return ReportGlyphs(per_component=False, free="░", separator=" · ")


def phase_total(components: list[dict[str, Any]]) -> int:
    """Bytes of a phase's components."""
    return sum(component["bytes"] for component in components)


def manifest_model_id(manifest: dict[str, Any]) -> str | None:
    """The model a manifest trains: ``algorithm`` wins over ``network``, as validation lifts it."""
    algorithm = manifest.get("algorithm") or {}
    network = manifest.get("network") or {}
    return algorithm.get("pretrained_model_name_or_path") or network.get(
        "pretrained_model_name_or_path"
    )


def render_phase(phase: dict[str, Any], glyphs: ReportGlyphs) -> Panel:
    """One phase: a stacked bar scaled to usable capacity, totals, and a row per component.

    A run over capacity is truncated at the bar's end and flagged with ``>>``.
    """
    usable = phase["device_usable_bytes"]
    total = phase_total(phase["components"])
    parts = [(c["label"], c["bytes"]) for c in phase["components"] if c["bytes"] > 0]
    bar = Text()
    rows = Table.grid(padding=(0, 2))
    rows.add_column()
    rows.add_column()
    rows.add_column(justify="right")
    rows.add_column(justify="right", style="dim")
    drawn = 0
    for index, (label, size) in enumerate(parts):
        colour = BAR_COLOURS[index % len(BAR_COLOURS)]
        glyph = BAR_GLYPHS[index % len(BAR_GLYPHS)] if glyphs.per_component else "█"
        cells = max(0, min(max(1, round(BAR_WIDTH * size / usable)), BAR_WIDTH - drawn))
        drawn += cells
        bar.append(glyph * cells, style=colour)
        rows.add_row(
            Text(glyph if glyphs.per_component else "■", style=colour),
            label,
            f"{size / GiB:.2f} GiB",
            f"{size / usable:.0%}",
        )
    fits = phase["fits"]
    bar.append(glyphs.free * (BAR_WIDTH - drawn), style="grey30")
    if not fits:
        bar.append(" >>", style="bold red")

    # Training's budget keeps the estimator's measured underprediction free.
    # Generation's budget is the usable capacity.
    headroom = phase["budget_bytes"] - total
    if phase["phase"] == "training":
        buffer = 1 - phase["budget_bytes"] / usable
        room = (
            (
                f"{headroom / GiB:.2f} GiB under the {buffer:.1%} buffer",
                "green",
            )
            if headroom >= 0
            else (
                (f"SHORTFALL {-headroom / GiB:.2f} GiB past the {buffer:.1%} buffer"),
                "bold red",
            )
        )
    else:
        room = (
            (f"{headroom / GiB:.2f} GiB headroom", "green")
            if headroom >= 0
            else (f"SHORTFALL {-headroom / GiB:.2f} GiB", "bold red")
        )
    summary = Text.assemble(
        (f"{total / GiB:.2f}", "bold"),
        f" / {usable / GiB:.2f} GiB usable{glyphs.separator}"
        f"{phase['device_total_bytes'] / GiB:.0f} GiB card{glyphs.separator}",
        room,
    )
    warnings = [Text(f"! {w}", style="yellow") for w in phase["warnings"]]
    return Panel(
        Group(bar, summary, Text(), rows, *warnings),
        title=Text(phase["phase"].capitalize(), style="bold"),
        title_align="left",
        subtitle=(
            Text("FITS", style="bold green")
            if fits
            else Text("OVER BUDGET", style="bold red")
        ),
        subtitle_align="right",
        border_style="green" if fits else "red",
        width=PANEL_WIDTH,
    )


def render_fixes(advice: list[dict[str, Any]]) -> Panel:
    """The ranked setting changes for an over-budget run."""
    fixes = Table.grid(padding=(0, 1))
    for rank, suggestion in enumerate(advice, start=1):
        fixes.add_row(
            Text(f"{rank}.", style="dim"),
            Text(suggestion["phase"], style="cyan"),
            f"{suggestion['action']} (saves ~{suggestion['saves_bytes'] / GiB:.1f} GiB)",
        )
    return Panel(
        fixes,
        title=Text("Fixes by memory saved", style="bold"),
        title_align="left",
        border_style="cyan",
        width=PANEL_WIDTH,
    )


def print_report(model_id: str | None, report: dict[str, Any], no_color: bool) -> None:
    """Print the estimate as phase bars, and the fixes when it is over budget."""
    # Rich drops colour off a terminal and honours NO_COLOR.
    console = Console(highlight=False, color_system=None if no_color else "auto")
    glyphs = report_glyphs(console)
    title = Text("Memory estimate", style="bold")
    if model_id is not None:
        title.append(f"{glyphs.separator}{model_id}")
    console.print(title)
    parallelism = report.get("parallelism")
    if parallelism is not None:
        console.print(
            Text(
                "Training parallelism: "
                f"tp={parallelism['tp']}, ep={parallelism['ep']}, cp={parallelism['cp']}, "
                f"shard_group_size={parallelism['shard_group_size']}",
                style="dim",
            )
        )
    engine = report["engine"]
    if engine is not None:
        console.print(
            Text(
                f"Rollout engine: tensor_parallel_size={engine['tensor_parallel_size']}"
                f"{glyphs.separator}max_num_seqs={engine['max_num_seqs']}",
                style="dim",
            )
        )
    console.print(
        Text("Training and generation are separate peaks, one bar each.", style="dim")
    )
    console.print(render_phase(report["training"], glyphs))
    console.print(render_phase(report["generation"], glyphs))
    if not report["fits"]:
        console.print(render_fixes(report["advice"]))
        logger.info("Blocked: apply a fix above or use a larger GPU.")


def use_json_output(as_json: bool) -> bool:
    """Turn on JSON output for this command when its own ``--json`` is set.

    :param as_json: The command's ``--json`` flag.
    :return: Whether this command prints JSON, from either ``--json``.
    """
    if as_json:
        click.get_current_context().meta[JSON_OUTPUT_META_KEY] = True
    return json_output_enabled()


def exit_usage(message: str, as_json: bool) -> None:
    """Exit 2 with ``message`` on stderr: ``{"error": ..., "type": "UsageError"}`` under ``--json``."""
    if as_json:
        click.echo(json.dumps({"error": message, "type": "UsageError"}), err=True)
    else:
        print(message, file=sys.stderr)
    sys.exit(EXIT_USAGE)


def load_manifest(manifest_path: str, as_json: bool) -> dict[str, Any]:
    """The manifest document; exits 2 when it cannot be read as a mapping."""
    try:
        manifest = read_manifest(manifest_path)
        if not isinstance(manifest, dict):
            msg = "manifest must be a mapping"
            raise ValueError(msg)
    except (OSError, ValueError, yaml.YAMLError) as err:
        exit_usage(f"could not validate {manifest_path}: {err}", as_json)
    return manifest


def check_local_options(
    config_path: str | None, device_gb: float | None, as_json: bool
) -> None:
    """Exit 2 on options Arena cannot take: a local config, or a capacity JSON cannot carry."""
    if config_path is not None:
        exit_usage(
            "--config is not supported: Arena sizes models in its catalog "
            "from their bundled config.",
            as_json,
        )
    if device_gb is not None and not (device_gb > 0 and math.isfinite(device_gb)):
        exit_usage(
            f"--device-gb: capacity must be a positive number of GiB, got {device_gb}.",
            as_json,
        )


def print_solve_report(report: dict[str, Any], no_color: bool) -> None:
    """Print the solved value and both phases at it."""
    console = Console(highlight=False, color_system=None if no_color else "auto")
    glyphs = report_glyphs(console)
    reason = (
        "checkpoint / --hi cap" if report["limited_by"] == "bound" else "GPU memory"
    )
    console.print(
        Text.assemble(
            ("Solved ", "bold"),
            (f"{report['field']} = {report['value']}", "bold cyan"),
            f"  ({reason})",
        )
    )
    engine = report["engine"]
    gmu = engine.get("gpu_memory_utilization")
    console.print(
        Text(
            f"{report['model']}{glyphs.separator}"
            f"TP={engine['tensor_parallel_size']}{glyphs.separator}"
            f"{engine['max_num_seqs']} seqs"
            + ("" if gmu is None else f"{glyphs.separator}gmu={gmu:g}"),
            style="dim",
        )
    )
    console.print(render_phase(report["training"], glyphs))
    console.print(render_phase(report["generation"], glyphs))


@click.group("memory")
def memory_group() -> None:
    """Estimate GPU memory, or solve one setting against a card."""


@memory_group.command("estimate")
@click.argument("manifest_path", metavar="MANIFEST")
@click.option(
    "--gpu",
    default=None,
    help=(
        "Training GPU: a GPU type from `arena resources list` such as 'h100-80', "
        "or a name such as 'NVIDIA L4'."
    ),
)
@click.option(
    "--device-gb",
    type=float,
    default=None,
    help="Training GPU memory in GiB, for a GPU the catalogue does not know.",
)
@click.option(
    "--num-gpus",
    type=click.IntRange(min=1),
    default=None,
    help=(
        "GPUs on the node the run gets. Needed when the run picks "
        "tensor_parallel_size or max_num_seqs from it."
    ),
)
@click.option("--config", "config_path", default=None, hidden=True)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Emit the full estimate as JSON instead of the report.",
)
@click.option(
    "--no-color",
    is_flag=True,
    help="Plain glyphs instead of colour (also honours NO_COLOR).",
)
@click.pass_obj
def memory_estimate(
    config: CommandConfig,
    manifest_path: str,
    gpu: str | None,
    device_gb: float | None,
    num_gpus: int | None,
    config_path: str | None,
    as_json: bool,
    no_color: bool,
) -> None:
    """Size MANIFEST (a training manifest path) against a GPU.

    Run settings come from the manifest; name the device with --gpu (a GPU
    type from `arena resources list`) or --device-gb. MANIFEST - reads stdin.
    Arena runs the estimate, so this needs a login.
    """
    as_json = use_json_output(as_json)
    check_local_options(config_path, device_gb, as_json)
    manifest = load_manifest(manifest_path, as_json)
    with arena_client(config) as client:
        estimate = client.estimate_memory(manifest, gpu, device_gb, num_gpus)
    if estimate.exit_code == EXIT_USAGE:
        exit_usage(estimate.usage_error, as_json)
    if as_json:
        print(json.dumps(estimate.report, indent=2))
    else:
        print_report(manifest_model_id(manifest), estimate.report, no_color)
    sys.exit(estimate.exit_code)


@memory_group.command("solve")
@click.argument("field", type=click.Choice(["max_model_len"], case_sensitive=True))
@click.argument("manifest_path", metavar="MANIFEST")
@click.option(
    "--gpu",
    default=None,
    help=(
        "GPU: a GPU type from `arena resources list` such as 'h100-80', "
        "or a name such as 'NVIDIA L4'."
    ),
)
@click.option(
    "--device-gb",
    type=float,
    default=None,
    help="GPU memory in GiB, for a GPU the catalogue does not know.",
)
@click.option(
    "--num-gpus",
    type=click.IntRange(min=1),
    default=None,
    help=(
        "GPUs on the node the run gets. Needed when the run picks "
        "tensor_parallel_size or max_num_seqs from it."
    ),
)
@click.option(
    "--hi",
    type=click.IntRange(min=0),
    default=None,
    help="Search ceiling; also capped at the checkpoint's RoPE limit.",
)
@click.option("--config", "config_path", default=None, hidden=True)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Emit the solve result as JSON instead of the report.",
)
@click.option(
    "--no-color",
    is_flag=True,
    help="Plain glyphs instead of colour (also honours NO_COLOR).",
)
@click.pass_obj
def memory_solve(
    config: CommandConfig,
    field: str,
    manifest_path: str,
    gpu: str | None,
    device_gb: float | None,
    num_gpus: int | None,
    hi: int | None,
    config_path: str | None,
    as_json: bool,
    no_color: bool,
) -> None:
    """Solve max_model_len for MANIFEST: the largest context that still fits, given everything else.

    Arena runs the solve for models in its catalog, so this needs a login.
    """
    as_json = use_json_output(as_json)
    check_local_options(config_path, device_gb, as_json)
    manifest = load_manifest(manifest_path, as_json)
    with arena_client(config) as client:
        solved = client.solve_memory(manifest, field, gpu, device_gb, num_gpus, hi)
    if solved.exit_code == EXIT_USAGE:
        exit_usage(solved.report["error"], as_json)
    if as_json:
        print(json.dumps(solved.report, indent=2))
    elif solved.exit_code != EXIT_OK:
        print(solved.report["error"], file=sys.stderr)
    else:
        print_solve_report(solved.report, no_color)
    sys.exit(solved.exit_code)

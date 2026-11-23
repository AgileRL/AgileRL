# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
r"""``arena memory`` commands: size a training manifest against a GPU on Arena.

Exit 0 if both phases fit, 3 if either is over budget, 2 on a usage error.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from typing import Any

import click
import yaml
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from agilerl.arena.config import CommandConfig, arena_client
from agilerl.arena.estimator_api import EXIT_USAGE
from agilerl.arena.models.verdict import read_manifest
from agilerl.arena.output import json_output_enabled

logger = logging.getLogger(__name__)

GiB = 1024**3
# Training must leave this share of the card free: the estimator's measured
# underprediction.
MAX_UNDERPREDICTION = 0.089
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


def within_buffer(predicted_bytes: int, usable_bytes: int) -> bool:
    """Whether a predicted peak still fits after the underprediction buffer."""
    return predicted_bytes <= int(usable_bytes * (1 - MAX_UNDERPREDICTION))


def phase_total(components: list[dict[str, Any]]) -> int:
    """Bytes of a phase's components."""
    return sum(component["bytes"] for component in components)


def phase_fits(phase: dict[str, Any]) -> bool:
    """Whether one phase of the report fits, as the estimator decides it.

    Training keeps the underprediction buffer free on the card and in host
    RAM. Generation must fit both, and vLLM's start-up pool must hold one
    full sequence.
    """
    total = phase_total(phase["components"])
    usable = phase["device_usable_bytes"]
    host_total = phase_total(phase["host"])
    host_usable = phase["host_usable_bytes"]
    if phase["phase"] == "training":
        return within_buffer(total, usable) and (
            host_usable is None or within_buffer(host_total, host_usable)
        )
    if total > usable or (host_usable is not None and host_total > host_usable):
        return False
    kv = next((c for c in phase["components"] if c["key"] == "kv_cache"), None)
    return (
        kv is None
        or kv["detail"]["startup_pool"] >= kv["detail"]["full_sequence_bytes"]
    )


def manifest_model_id(manifest: dict[str, Any]) -> str:
    """The model a manifest trains: ``algorithm`` wins over ``network``, as validation lifts it."""
    algorithm = manifest.get("algorithm") or {}
    network = manifest.get("network") or {}
    return (
        algorithm.get("pretrained_model_name_or_path")
        or network["pretrained_model_name_or_path"]
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
    fits = phase_fits(phase)
    bar.append(glyphs.free * (BAR_WIDTH - drawn), style="grey30")
    if not fits:
        bar.append(" >>", style="bold red")

    # Training keeps the measured underprediction free. Generation is the
    # engine's own budget and stays on the point estimate.
    if phase["phase"] == "training":
        headroom = int(usable * (1 - MAX_UNDERPREDICTION)) - total
        room = (
            (
                f"{headroom / GiB:.2f} GiB under the {MAX_UNDERPREDICTION:.1%} buffer",
                "green",
            )
            if headroom >= 0
            else (
                (
                    f"SHORTFALL {-headroom / GiB:.2f} GiB past the "
                    f"{MAX_UNDERPREDICTION:.1%} buffer"
                ),
                "bold red",
            )
        )
    else:
        headroom = usable - total
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


def print_report(model_id: str, report: dict[str, Any], no_color: bool) -> None:
    """Print the estimate as phase bars, and the fixes when it is over budget."""
    # Rich drops colour off a terminal and honours NO_COLOR.
    console = Console(highlight=False, color_system=None if no_color else "auto")
    glyphs = report_glyphs(console)
    console.print(
        Text.assemble(("Memory estimate", "bold"), f"{glyphs.separator}{model_id}")
    )
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


@click.group("memory")
def memory_group() -> None:
    """Estimate a training manifest's GPU memory on Arena."""


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
    as_json: bool,
    no_color: bool,
) -> None:
    """Size MANIFEST (a training manifest path, or - for stdin) against a GPU.

    Run settings come from the manifest; name the device with --gpu (a GPU
    type from `arena resources list`) or --device-gb. Arena runs the estimate,
    so this needs a login.
    """
    try:
        manifest = read_manifest(manifest_path)
        if not isinstance(manifest, dict):
            msg = "manifest must be a mapping"
            raise ValueError(msg)
    except (OSError, ValueError, yaml.YAMLError) as err:
        print(f"could not validate {manifest_path}: {err}", file=sys.stderr)
        sys.exit(EXIT_USAGE)
    with arena_client(config) as client:
        estimate = client.estimate_memory(manifest, gpu, device_gb, num_gpus)
    if estimate.exit_code == EXIT_USAGE:
        print(estimate.usage_error, file=sys.stderr)
        sys.exit(EXIT_USAGE)
    if as_json or json_output_enabled():
        print(json.dumps(estimate.report, indent=2))
    else:
        print_report(manifest_model_id(manifest), estimate.report, no_color)
    sys.exit(estimate.exit_code)

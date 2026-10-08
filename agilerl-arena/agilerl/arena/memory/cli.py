# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Size a training manifest against a GPU.

Settings come from the manifest; the device from ``--gpu`` / ``--device-gb``.
Pass ``--config`` to stay offline.

Exit 0 if both phases fit, 3 if either is over budget, 2 on a usage error.
"""

from __future__ import annotations

import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click
import yaml
from pydantic import ValidationError
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from agilerl.arena.memory.advice import advise
from agilerl.arena.memory.estimator import PhaseBreakdown, estimate_run
from agilerl.arena.memory.manifest import (
    llm_spec,
    lookup_gpu,
    run_config_from_manifest,
)
from agilerl.arena.memory.specs import (
    DeviceSpec,
    GiB,
    ModelArch,
    RunConfig,
)
from agilerl.arena.models.manifest import TrainingManifest

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_OVER_BUDGET = 3
EXIT_USAGE = 2

# Multi-token-prediction heads some checkpoints ship (Nemotron 3.5); the
# trainer does not load them.
MTP_TENSOR_PREFIX = "mtp."


def checkpoint_param_count(model_id: str, arch: ModelArch) -> int | None:
    """Resident parameters from the Hub's safetensors index (metadata only).

    A tied-embedding checkpoint may still ship ``lm_head.weight`` as a second
    copy of the table; ``from_pretrained`` re-ties them, so only one copy is
    resident. Multi-token-prediction tensors are not loaded. ``None`` when the
    repo publishes no index.
    """
    try:
        # Optional hub extra.
        from huggingface_hub import get_safetensors_metadata
        from huggingface_hub.errors import (
            NotASafetensorsRepoError,
            SafetensorsParsingError,
        )
    except ImportError:
        return None

    try:
        metadata = get_safetensors_metadata(model_id)
    except (OSError, NotASafetensorsRepoError, SafetensorsParsingError) as err:
        # An expired HF token or a gated repo also lands here.
        logger.warning(
            "Could not read %s's safetensors metadata (%s); using the analytic "
            "parameter count. Check HF_TOKEN if the repo is gated.",
            model_id,
            err,
        )
        return None
    total = sum(
        tensor.parameter_count
        for file in metadata.files_metadata.values()
        for name, tensor in file.tensors.items()
        if not name.startswith(MTP_TENSOR_PREFIX)
    )
    if not total:
        return None
    if arch.tied_embeddings and "lm_head.weight" in metadata.weight_map:
        total -= arch.vocab_size * arch.hidden_size
    return int(total)


def _load_model_config(model_id: str, config_path: str | None) -> dict[str, Any]:
    """The checkpoint's raw ``config.json``: ``--config``, a local checkpoint, or the Hub.

    The estimator reads raw keys; transformers' ``NemotronHConfig.to_dict()``
    omits ``num_hidden_layers``, which it derives from ``layers_block_type``.
    """
    if config_path is not None:
        return json.loads(Path(config_path).read_text())
    local = Path(model_id) / "config.json"
    if local.is_file():
        return json.loads(local.read_text())
    try:
        # Optional ``hub`` extra.
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError
    except ImportError as err:
        msg = (
            "huggingface_hub is not installed, so the config for "
            f"{model_id!r} cannot be fetched. Pass "
            "--config path/to/config.json instead."
        )
        raise ValueError(msg) from err
    try:
        return json.loads(Path(hf_hub_download(model_id, "config.json")).read_text())
    except (OSError, EntryNotFoundError, ValueError) as err:
        msg = (
            f"could not fetch the config for {model_id!r} ({err}); pass "
            "--config path/to/config.json instead."
        )
        raise ValueError(msg) from err


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


def _report_glyphs(console: Console) -> ReportGlyphs:
    """Solid blocks for a colour UTF-8 console; ASCII when it cannot encode Unicode."""
    unicode = console.encoding.lower().startswith("utf")
    colour = console.color_system is not None and not console.no_color
    if not unicode:
        return ReportGlyphs(per_component=True, free="-", separator=" | ")
    if not colour:
        return ReportGlyphs(per_component=True, free="·", separator=" · ")
    return ReportGlyphs(per_component=False, free="░", separator=" · ")


def _render_phase(breakdown: PhaseBreakdown, glyphs: ReportGlyphs) -> Panel:
    """One phase: a stacked bar scaled to usable capacity, totals, and a row per component.

    A run over capacity is truncated at the bar's end and flagged with ``>>``.
    """
    usable = breakdown.device_usable_bytes
    parts = [(c.label, c.n_bytes) for c in breakdown.components if c.n_bytes > 0]
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
    bar.append(glyphs.free * (BAR_WIDTH - drawn), style="grey30")
    if not breakdown.fits:
        bar.append(" >>", style="bold red")

    headroom = breakdown.headroom_bytes
    summary = Text.assemble(
        (f"{breakdown.total_bytes / GiB:.2f}", "bold"),
        f" / {usable / GiB:.2f} GiB usable{glyphs.separator}"
        f"{breakdown.device_total_bytes / GiB:.0f} GiB card{glyphs.separator}",
        (
            (f"{headroom / GiB:.2f} GiB headroom", "green")
            if headroom >= 0
            else (f"SHORTFALL {-headroom / GiB:.2f} GiB", "bold red")
        ),
    )
    warnings = [Text(f"! {w}", style="yellow") for w in breakdown.warnings]
    return Panel(
        Group(bar, summary, Text(), rows, *warnings),
        title=Text(breakdown.phase.capitalize(), style="bold"),
        title_align="left",
        subtitle=(
            Text("FITS", style="bold green")
            if breakdown.fits
            else Text("OVER BUDGET", style="bold red")
        ),
        subtitle_align="right",
        border_style="green" if breakdown.fits else "red",
        width=PANEL_WIDTH,
    )


def _render_fixes(config: RunConfig) -> Panel:
    """The ranked setting changes for an over-budget run."""
    fixes = Table.grid(padding=(0, 1))
    for rank, suggestion in enumerate(advise(config), start=1):
        fixes.add_row(
            Text(f"{rank}.", style="dim"),
            Text(suggestion.phase, style="cyan"),
            str(suggestion),
        )
    return Panel(
        fixes,
        title=Text("Fixes by memory saved", style="bold"),
        title_align="left",
        border_style="cyan",
        width=PANEL_WIDTH,
    )


def _resolve_device(
    gpu: str | None, device_gb: float | None, *, flag: str, capacity_flag: str
) -> DeviceSpec | None:
    """A device from ``--gpu`` (catalogue) and/or a raw capacity in GiB."""
    if device_gb is not None and not (device_gb > 0 and math.isfinite(device_gb)):
        msg = f"{capacity_flag}: capacity must be a positive number of GiB, got {device_gb}."
        raise ValueError(msg)
    if gpu is not None:
        info = lookup_gpu(gpu)
        if info is None:
            msg = (
                f"{flag}: unknown GPU {gpu!r}; pass the capacity via "
                f"{capacity_flag} instead."
            )
            raise ValueError(msg)
        spec = info.device_spec()
        if device_gb is not None:
            spec = spec.model_copy(update={"total_bytes": int(device_gb * GiB)})
        return spec
    if device_gb is not None:
        return DeviceSpec(total_bytes=int(device_gb * GiB))
    return None


@click.group("memory")
def memory_group() -> None:
    """Estimate GPU memory for a training manifest."""


@memory_group.command("estimate")
@click.argument("manifest_path", metavar="MANIFEST")
@click.option(
    "--gpu",
    default=None,
    help="Training GPU the resource class provides, e.g. 'NVIDIA L4' or 'A100-80GB'.",
)
@click.option(
    "--device-gb",
    type=float,
    default=None,
    help="Training GPU memory in GiB, for a GPU the catalogue does not know.",
)
@click.option(
    "--gen-gpu",
    default=None,
    help="Rollout-engine GPU for async manifests; defaults to the training GPU.",
)
@click.option(
    "--gen-device-gb",
    type=float,
    default=None,
    help="Rollout-engine GPU memory in GiB for async manifests.",
)
@click.option(
    "--config",
    "config_path",
    default=None,
    help="Local config.json for the manifest's model (offline mode).",
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
def memory_estimate(
    manifest_path: str,
    gpu: str | None,
    device_gb: float | None,
    gen_gpu: str | None,
    gen_device_gb: float | None,
    config_path: str | None,
    as_json: bool,
    no_color: bool,
) -> None:
    """Size MANIFEST (a training manifest path) against a GPU.

    Run settings come from the manifest; name the device with --gpu or --device-gb.
    """
    sys.exit(
        main(
            manifest_path,
            gpu=gpu,
            device_gb=device_gb,
            gen_gpu=gen_gpu,
            gen_device_gb=gen_device_gb,
            config_path=config_path,
            as_json=as_json,
            no_color=no_color,
        )
    )


def main(
    manifest_path: str,
    *,
    gpu: str | None = None,
    device_gb: float | None = None,
    gen_gpu: str | None = None,
    gen_device_gb: float | None = None,
    config_path: str | None = None,
    as_json: bool = False,
    no_color: bool = False,
) -> int:
    """Run the estimate and return the exit code."""
    try:
        device = _resolve_device(
            gpu, device_gb, flag="--gpu", capacity_flag="--device-gb"
        )
        gen_device = _resolve_device(
            gen_gpu, gen_device_gb, flag="--gen-gpu", capacity_flag="--gen-device-gb"
        )
    except ValueError as err:
        print(err, file=sys.stderr)
        return EXIT_USAGE
    if device is None:
        print("Pass --gpu or --device-gb (see --help).", file=sys.stderr)
        return EXIT_USAGE

    try:
        manifest = TrainingManifest.get_validated(manifest_path, mode="python")
    except (OSError, ValueError, ValidationError, yaml.YAMLError) as err:
        print(f"could not validate {manifest_path}: {err}", file=sys.stderr)
        return EXIT_USAGE

    try:
        spec = llm_spec(manifest)
    except ValueError as err:
        print(err, file=sys.stderr)
        return EXIT_USAGE
    model_id = spec.pretrained_model_name_or_path
    if model_id is None:
        print("manifest names no pretrained model.", file=sys.stderr)
        return EXIT_USAGE
    try:
        model_config = _load_model_config(model_id, config_path)
    except (OSError, ValueError) as err:
        print(f"could not load the model config: {err}", file=sys.stderr)
        return EXIT_USAGE
    if config_path is not None:
        print(
            "note: --config skips the Hub parameter count; check the file "
            "matches the manifest's model.",
            file=sys.stderr,
        )

    try:
        config = run_config_from_manifest(
            manifest,
            device,
            model_config,
            n_params=(
                None
                if config_path is not None
                else checkpoint_param_count(
                    model_id, ModelArch.from_hf_config(model_config)
                )
            ),
            gen_device=gen_device,
        )
    except KeyError as err:
        print(
            f"model config is missing {err}; pass the checkpoint's config.json "
            "via --config.",
            file=sys.stderr,
        )
        return EXIT_USAGE
    except ValueError as err:
        print(str(err), file=sys.stderr)
        return EXIT_USAGE

    estimate = estimate_run(config)
    if as_json:
        payload = estimate.model_dump(by_alias=True)
        payload["fits"] = estimate.fits
        payload["advice"] = [a.model_dump() for a in advise(config)]
        print(json.dumps(payload, indent=2))
        return EXIT_OK if estimate.fits else EXIT_OVER_BUDGET

    # Rich drops colour off a terminal and honours NO_COLOR.
    console = Console(highlight=False, color_system=None if no_color else "auto")
    glyphs = _report_glyphs(console)
    console.print(
        Text.assemble(
            ("Memory estimate", "bold"), f"{glyphs.separator}{config.model.model_id}"
        )
    )
    console.print(
        Text("Training and generation are separate peaks, one bar each.", style="dim")
    )
    console.print(_render_phase(estimate.training, glyphs))
    console.print(_render_phase(estimate.generation, glyphs))
    if not estimate.fits:
        console.print(_render_fixes(config))
        logger.info("Blocked: apply a fix above or use a larger GPU.")
    return EXIT_OK if estimate.fits else EXIT_OVER_BUDGET

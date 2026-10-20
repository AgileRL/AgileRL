# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
r"""Size a training manifest against a GPU, or invert one field.

Settings come from the manifest; the device from ``--gpu`` / ``--device-gb``.
Without either, ``estimate`` picks the cheapest Arena resource tier that fits.
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

from agilerl.arena.config import CommandConfig, arena_client
from agilerl.arena.memory.advice import advise
from agilerl.arena.memory.estimator import PhaseBreakdown, estimate_run
from agilerl.arena.memory.formulas import MAX_UNDERPREDICTION
from agilerl.arena.memory.manifest import (
    llm_spec,
    lookup_gpu,
    run_config_from_manifest,
)
from agilerl.arena.memory.resources import TierCheck, check_tiers
from agilerl.arena.memory.solver import (
    INFERENCE_GPU_MEMORY_UTILIZATION,
    INFERENCE_MAX_NUM_SEQS,
    SOLVABLE_FIELDS,
    CannotSolve,
    architectural_context_limit,
    inference_run_config,
    solve,
    solve_inference,
)
from agilerl.arena.memory.specs import (
    DeviceSpec,
    GenerationSettings,
    GiB,
    ModelArch,
    ModelSpec,
    RunConfig,
    TrainingSettings,
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
    fits = breakdown.fits_with_buffer
    bar.append(glyphs.free * (BAR_WIDTH - drawn), style="grey30")
    if not fits:
        bar.append(" >>", style="bold red")

    # Training keeps the measured underprediction free. Generation is the
    # engine's own budget and stays on the point estimate.
    if breakdown.phase == "training":
        limit = int(usable * (1 - MAX_UNDERPREDICTION))
        headroom = limit - breakdown.total_bytes
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
        headroom = breakdown.headroom_bytes
        room = (
            (f"{headroom / GiB:.2f} GiB headroom", "green")
            if headroom >= 0
            else (f"SHORTFALL {-headroom / GiB:.2f} GiB", "bold red")
        )
    summary = Text.assemble(
        (f"{breakdown.total_bytes / GiB:.2f}", "bold"),
        f" / {usable / GiB:.2f} GiB usable{glyphs.separator}"
        f"{breakdown.device_total_bytes / GiB:.0f} GiB card{glyphs.separator}",
        room,
    )
    warnings = [Text(f"! {w}", style="yellow") for w in breakdown.warnings]
    return Panel(
        Group(bar, summary, Text(), rows, *warnings),
        title=Text(breakdown.phase.capitalize(), style="bold"),
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


def _model_spec_from_checkpoint(
    model_id: str, config_path: str | None
) -> tuple[dict[str, Any], ModelSpec]:
    """Geometry (and Hub parameter count) for a model id or a local config."""
    model_config = _load_model_config(model_id, config_path)
    arch = ModelArch.from_hf_config(model_config)
    n_params = (
        None if config_path is not None else checkpoint_param_count(model_id, arch)
    )
    return model_config, ModelSpec(model_id=model_id, arch=arch, n_params=n_params)


def _report_missing_key(err: KeyError) -> int:
    """Print a missing model-config key as a usage error."""
    print(
        f"model config is missing {err}; pass the checkpoint's config.json "
        "via --config.",
        file=sys.stderr,
    )
    return EXIT_USAGE


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
    """Estimate GPU memory, or solve one setting against a card."""


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
@click.pass_context
def memory_estimate(
    ctx: click.Context,
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

    Run settings come from the manifest; name the device with --gpu or
    --device-gb. Without either, the cheapest Arena resource tier that fits
    is picked (needs arena login).
    """
    tiers = None
    command_config = ctx.find_object(CommandConfig)
    if gpu is None and device_gb is None and command_config is not None:
        with arena_client(command_config) as client:
            tiers = list(client.list_resources()["tiers"].values())
    sys.exit(
        main(
            manifest_path,
            gpu=gpu,
            device_gb=device_gb,
            tiers=tiers,
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
    tiers: list[dict[str, Any]] | None = None,
    gen_gpu: str | None = None,
    gen_device_gb: float | None = None,
    config_path: str | None = None,
    as_json: bool = False,
    no_color: bool = False,
) -> int:
    """Run the estimate and return the exit code.

    :param tiers: Resource tiers to pick the cheapest fitting one from when
        neither ``gpu`` nor ``device_gb`` is given.
    """
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
    if device is None and tiers is None:
        print(
            "Pass --gpu or --device-gb, or run `arena memory estimate` logged in "
            "to pick a resource tier.",
            file=sys.stderr,
        )
        return EXIT_USAGE
    if device is None and gen_device is not None:
        print("--gen-gpu/--gen-device-gb needs --gpu or --device-gb.", file=sys.stderr)
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
        model_config, model_spec = _model_spec_from_checkpoint(model_id, config_path)
    except KeyError as err:
        return _report_missing_key(err)
    except (OSError, ValueError) as err:
        print(f"could not load the model config: {err}", file=sys.stderr)
        return EXIT_USAGE
    if config_path is not None:
        print(
            "note: --config skips the Hub parameter count; check the file "
            "matches the manifest's model.",
            file=sys.stderr,
        )

    def build_config(train_device: DeviceSpec) -> RunConfig:
        return run_config_from_manifest(
            manifest,
            train_device,
            model_config,
            n_params=model_spec.n_params,
            gen_device=gen_device,
        )

    chosen: TierCheck | None = None
    checks: list[TierCheck] = []
    try:
        if device is None:
            checks = check_tiers(tiers or [], manifest.training, build_config)
            chosen = next((check for check in checks if check.fits), None)
            if chosen is None or chosen.config is None:
                return _report_no_tier(checks, as_json)
            config = chosen.config
        else:
            config = build_config(device)
    except KeyError as err:
        return _report_missing_key(err)
    except ValueError as err:
        print(str(err), file=sys.stderr)
        return EXIT_USAGE

    estimate = estimate_run(config)
    if as_json:
        payload = estimate.model_dump(by_alias=True)
        payload["fits"] = estimate.fits_with_buffer
        payload["advice"] = [a.model_dump() for a in advise(config)]
        if chosen is not None:
            payload["resource"] = chosen.tier
            payload["tiers"] = _tier_rows(checks)
        print(json.dumps(payload, indent=2))
        return EXIT_OK if estimate.fits_with_buffer else EXIT_OVER_BUDGET

    # Rich drops colour off a terminal and honours NO_COLOR.
    console = Console(highlight=False, color_system=None if no_color else "auto")
    glyphs = _report_glyphs(console)
    if chosen is not None:
        console.print(
            Text.assemble(
                ("Cheapest resource tier that fits: ", "bold"),
                (_describe_tier(chosen.tier), "green"),
            )
        )
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
    if not estimate.fits_with_buffer:
        console.print(_render_fixes(config))
        logger.info("Blocked: apply a fix above or use a larger GPU.")
    return EXIT_OK if estimate.fits_with_buffer else EXIT_OVER_BUDGET


def _describe_tier(tier: dict[str, Any]) -> str:
    return (
        f"{tier['name']} ({tier['num_gpus']}x {tier['gpu_type']}, "
        f"{tier['price_per_node_hour']:.2f} credits/node-hour)"
    )


def _tier_rows(checks: list[TierCheck]) -> list[dict[str, Any]]:
    return [
        {
            "name": check.tier["name"],
            "price_per_node_hour": check.tier["price_per_node_hour"],
            "fits": check.fits,
            "reason": check.reason,
        }
        for check in checks
    ]


def _report_no_tier(checks: list[TierCheck], as_json: bool) -> int:
    """Why each tier was ruled out, when none fits."""
    if as_json:
        print(
            json.dumps(
                {"fits": False, "resource": None, "tiers": _tier_rows(checks)}, indent=2
            )
        )
        return EXIT_OVER_BUDGET
    print("No resource tier fits this manifest:")
    for check in checks:
        print(f"  - {check.tier['name']}: {check.reason}")
    return EXIT_OVER_BUDGET


@dataclass(frozen=True)
class SolveRequest:
    """One ``arena memory solve`` invocation."""

    field: str
    manifest_path: str | None = None
    inference: bool = False
    gpu: str | None = None
    device_gb: float | None = None
    gen_gpu: str | None = None
    gen_device_gb: float | None = None
    model_id: str | None = None
    config_path: str | None = None
    max_num_seqs: int | None = None
    gpu_memory_utilization: float | None = None
    max_model_len: int | None = None
    enforce_eager: bool | None = None
    max_loras: int | None = None
    hi: int | None = None
    as_json: bool = False
    no_color: bool = False


@memory_group.command("solve")
@click.argument(
    "field", type=click.Choice(sorted(SOLVABLE_FIELDS), case_sensitive=True)
)
@click.argument("manifest_path", required=False, metavar="[MANIFEST]")
@click.option(
    "--inference",
    is_flag=True,
    help=(
        "Dedicated serving GPU: no trainer residual. Requires --model or "
        "--config; do not pass a training manifest."
    ),
)
@click.option(
    "--gpu",
    default=None,
    help="GPU the resource class provides, e.g. 'NVIDIA L4' or 'A100-80GB'.",
)
@click.option(
    "--device-gb",
    type=float,
    default=None,
    help="GPU memory in GiB, for a GPU the catalogue does not know.",
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
    "--model",
    "model_id",
    default=None,
    help="Hub id; defaults to the --config file's directory name.",
)
@click.option(
    "--config",
    "config_path",
    default=None,
    help="Local config.json (offline mode).",
)
@click.option(
    "--max-num-seqs",
    type=int,
    default=None,
    help="Concurrent sequences. Default 8 in --inference; from the manifest otherwise.",
)
@click.option(
    "--gpu-memory-utilization",
    type=float,
    default=None,
    help="vLLM gpu_memory_utilization. Default 0.9 in --inference.",
)
@click.option(
    "--max-model-len",
    type=int,
    default=None,
    help="Context cap when solving a different setting.",
)
@click.option(
    "--enforce-eager",
    is_flag=True,
    default=None,
    help="Skip CUDA-graph capture, at some decode-throughput cost.",
)
@click.option(
    "--max-loras",
    type=int,
    default=None,
    help="LoRA adapter slots on the engine.",
)
@click.option(
    "--hi",
    type=int,
    default=None,
    help="Search ceiling. max_model_len also caps at the checkpoint's RoPE limit.",
)
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
@click.pass_context
def memory_solve(ctx: click.Context, **params: Any) -> None:
    """Solve FIELD: the largest value that still fits, given everything else.

    Invertible settings: max_model_len, max_num_seqs.
    --inference sizes a dedicated serving GPU; without it, MANIFEST is required and the
    solve is that training run.
    """
    sys.exit(solve_main(SolveRequest(**params)))


def solve_main(request: SolveRequest) -> int:
    """Invert one setting. Separated from Click so callers get a plain int."""
    field = request.field
    manifest_path = request.manifest_path
    inference = request.inference
    gpu = request.gpu
    device_gb = request.device_gb
    gen_gpu = request.gen_gpu
    gen_device_gb = request.gen_device_gb
    model_id = request.model_id
    config_path = request.config_path
    max_num_seqs = request.max_num_seqs
    gpu_memory_utilization = request.gpu_memory_utilization
    max_model_len = request.max_model_len
    enforce_eager = request.enforce_eager
    max_loras = request.max_loras
    hi = request.hi
    as_json = request.as_json
    no_color = request.no_color
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

    if inference and manifest_path is not None:
        print("--inference does not take a training manifest.", file=sys.stderr)
        return EXIT_USAGE
    if not inference and manifest_path is None:
        print("Pass a MANIFEST, or --inference with --model.", file=sys.stderr)
        return EXIT_USAGE
    if inference and gen_device is not None:
        print(
            "--inference sizes one GPU; drop --gen-gpu/--gen-device-gb.",
            file=sys.stderr,
        )
        return EXIT_USAGE

    overrides = {
        key: value
        for key, value in {
            "max_num_seqs": max_num_seqs,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len,
            "enforce_eager": enforce_eager,
            "max_loras": max_loras,
        }.items()
        if value is not None
    }
    if field in overrides:
        flag = field.replace("_", "-")
        print(f"--{flag} is the setting being solved; drop it.", file=sys.stderr)
        return EXIT_USAGE

    try:
        # Past the checks above, no manifest means --inference.
        if manifest_path is None:
            config, arch_limit = _inference_config(
                device, model_id=model_id, config_path=config_path, overrides=overrides
            )
        else:
            config, arch_limit = _manifest_config(
                manifest_path,
                device,
                config_path=config_path,
                gen_device=gen_device,
                overrides=overrides,
            )
    except KeyError as err:
        return _report_missing_key(err)
    except (OSError, ValueError, ValidationError, yaml.YAMLError) as err:
        print(str(err), file=sys.stderr)
        return EXIT_USAGE

    bound = hi
    if field == "max_model_len":
        bound = arch_limit if bound is None else min(bound, arch_limit)
    solver = solve_inference if inference else solve
    try:
        result = solver(config, field, hi=bound)
    except CannotSolve as err:
        print(str(err), file=sys.stderr)
        return EXIT_OVER_BUDGET
    except ValueError as err:
        print(str(err), file=sys.stderr)
        return EXIT_USAGE

    if as_json:
        payload = {
            "field": result.field,
            "value": result.value,
            "limited_by": result.limited_by,
            "bound": result.bound,
            "mode": "inference" if inference else "training",
            "model": result.config.model.model_id,
            "unchecked_over_budget": list(result.unchecked_over_budget),
            "generation": result.estimate.generation.model_dump(by_alias=True),
        }
        if not inference:
            payload["training"] = result.estimate.training.model_dump(by_alias=True)
        print(json.dumps(payload, indent=2))
        return EXIT_OK

    reason = "checkpoint / --hi cap" if result.limited_by == "bound" else "GPU memory"
    mode = "inference" if inference else "training"
    if result.unchecked_over_budget:
        unchecked = ", ".join(result.unchecked_over_budget)
        print(
            f"warning: {unchecked} is over budget, but this solve did not check it.",
            file=sys.stderr,
        )
    console = Console(highlight=False, color_system=None if no_color else "auto")
    glyphs = _report_glyphs(console)
    console.print(
        Text.assemble(
            ("Solved ", "bold"),
            (f"{result.field} = {result.value}", "bold cyan"),
            f"  ({reason})",
        )
    )
    console.print(
        Text(
            f"{result.config.model.model_id}{glyphs.separator}{mode}{glyphs.separator}"
            f"{result.config.generation.max_num_seqs} seqs{glyphs.separator}"
            f"gmu={result.config.generation.gpu_memory_utilization:g}",
            style="dim",
        )
    )
    if not inference:
        console.print(_render_phase(result.estimate.training, glyphs))
    console.print(_render_phase(result.estimate.generation, glyphs))
    return EXIT_OK


def _inference_config(
    device: DeviceSpec,
    *,
    model_id: str | None,
    config_path: str | None,
    overrides: dict[str, object],
) -> tuple[RunConfig, int]:
    resolved_id = model_id or (
        Path(config_path).parent.name if config_path is not None else None
    )
    if resolved_id is None:
        msg = "--inference needs --model or --config."
        raise ValueError(msg)
    model_config, model = _model_spec_from_checkpoint(resolved_id, config_path)
    settings = GenerationSettings.model_validate(
        {
            "gpu_memory_utilization": INFERENCE_GPU_MEMORY_UTILIZATION,
            "max_num_seqs": INFERENCE_MAX_NUM_SEQS,
            **overrides,
        }
    )
    return inference_run_config(model, device, settings), architectural_context_limit(
        model_config
    )


def _manifest_config(
    manifest_path: str,
    device: DeviceSpec,
    *,
    config_path: str | None,
    gen_device: DeviceSpec | None,
    overrides: dict[str, object],
) -> tuple[RunConfig, int]:
    manifest = TrainingManifest.get_validated(manifest_path, mode="python")
    resolved_id = llm_spec(manifest).pretrained_model_name_or_path
    if resolved_id is None:
        msg = "manifest names no pretrained model."
        raise ValueError(msg)
    model_config, model_spec = _model_spec_from_checkpoint(resolved_id, config_path)
    config = run_config_from_manifest(
        manifest,
        device,
        model_config,
        n_params=model_spec.n_params,
        gen_device=gen_device,
    )
    config_updates: dict[str, object] = {}
    if overrides:
        # model_copy skips validation; re-validate so a bad flag is a usage
        # error, not a downstream crash.
        config_updates["generation"] = GenerationSettings.model_validate(
            {**config.generation.model_dump(), **overrides}
        )
    if "max_model_len" in overrides:
        # The manifest holds one context length; a generation-only override
        # would desync the training side from the submitted run.
        config_updates["training"] = TrainingSettings.model_validate(
            {
                **config.training.model_dump(),
                "max_model_len": overrides["max_model_len"],
            }
        )
    if config_updates:
        config = config.model_copy(update=config_updates)
    return config, architectural_context_limit(model_config)

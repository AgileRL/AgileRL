# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Size a training manifest against a GPU.

No GPU, no profiling, no weight download. Settings come from the manifest;
the device from ``--gpu`` / ``--device-gb``. Pass ``--config`` to stay
offline.

Exit 0 if both phases fit, 3 if either is over budget, 2 on a usage error.
``--allow-oversize`` reports the shortfall but exits 0.

    arena memory estimate manifest.yaml --gpu "NVIDIA L4"
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import click
import yaml
from pydantic import ValidationError

from agilerl.arena.memory.advice import advise
from agilerl.arena.memory.estimator import PhaseBreakdown, estimate_run
from agilerl.arena.memory.manifest import (
    lookup_gpu,
    run_config_from_manifest,
)
from agilerl.arena.memory.specs import (
    DeviceSpec,
    GiB,
    ModelArch,
)
from agilerl.arena.models.algorithms.base import LLMAlgorithmSpec
from agilerl.arena.models.manifest import TrainingManifest

# Exit codes: fit, usage error, over budget.
EXIT_OK = 0
EXIT_OVER_BUDGET = 3
EXIT_USAGE = 2


def checkpoint_param_count(model_id: str, arch: ModelArch) -> int | None:
    """Resident parameters from the Hub's safetensors index (metadata only).

    A tied-embedding checkpoint may still ship ``lm_head.weight`` as a second
    copy of the table; ``from_pretrained`` re-ties them, so only one copy is
    resident. ``None`` when the repo publishes no index.
    """
    try:
        from huggingface_hub import get_safetensors_metadata

        metadata = get_safetensors_metadata(model_id)
    except (ImportError, OSError, ValueError, KeyError):
        return None
    total = sum(metadata.parameter_count.values())
    if not total:
        return None
    if arch.tied_embeddings and "lm_head.weight" in metadata.weight_map:
        total -= arch.vocab_size * arch.hidden_size
    return int(total)


def _load_model_config(model_id: str, config_path: str | None) -> dict[str, Any]:
    """The checkpoint's ``config.json``: from disk, or fetched by model id."""
    if config_path is not None:
        return json.loads(Path(config_path).read_text())
    try:
        from transformers import AutoConfig
    except ImportError as err:
        msg = (
            "transformers is not installed, so the config for "
            f"{model_id!r} cannot be fetched. Pass "
            "--config path/to/config.json instead."
        )
        raise ValueError(msg) from err
    try:
        return AutoConfig.from_pretrained(model_id).to_dict()
    except (OSError, ValueError) as err:
        msg = (
            f"could not fetch the config for {model_id!r} ({err}); pass "
            "--config path/to/config.json instead."
        )
        raise ValueError(msg) from err


# Bar width in characters. 48 keeps a phase line inside 80 columns once the
# label and the GiB figure are allowed for.
BAR_WIDTH = 48
# One colour per component, cycled. 256-colour codes so the bar reads on both
# light and dark terminals; ``BAR_GLYPHS`` is used when colour is unavailable,
# because a stacked bar is meaningless if every segment looks the same.
BAR_COLOURS = (33, 208, 71, 170, 214, 45, 203, 100)
BAR_GLYPHS = ("#", "=", "+", "*", "o", ":", "~", ".")


def _use_colour(stream: object) -> bool:
    """Colour only for an interactive terminal that has not opted out."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return bool(getattr(stream, "isatty", lambda: False)())


def _segments(breakdown: PhaseBreakdown) -> list[tuple[str, int]]:
    return [(c.label, c.n_bytes) for c in breakdown.components if c.n_bytes > 0]


def _stacked_bar(breakdown: PhaseBreakdown, colour: bool) -> tuple[str, list[str]]:
    """One stacked bar scaled to device capacity, plus its legend rows.

    Scaled to capacity rather than to the total, so the bar answers "how full
    is the card" instead of "what is the mix". A run that does not fit is
    truncated at the capacity marker and flagged with ``>>``, rather than
    renormalised -- renormalising would make a 3x overshoot look identical to
    a perfect fit.
    """
    usable = breakdown.device_usable_bytes or 1
    segments = _segments(breakdown)
    bar = ""
    drawn = 0
    legend: list[str] = []
    for index, (label, size) in enumerate(segments):
        want = max(1, round(BAR_WIDTH * size / usable))
        cells = max(0, min(want, BAR_WIDTH - drawn))
        drawn += cells
        glyph = "█" if colour else BAR_GLYPHS[index % len(BAR_GLYPHS)]
        chunk = glyph * cells
        if colour and cells:
            chunk = f"\033[38;5;{BAR_COLOURS[index % len(BAR_COLOURS)]}m{chunk}\033[0m"
        bar += chunk
        key = (
            f"\033[38;5;{BAR_COLOURS[index % len(BAR_COLOURS)]}m█\033[0m"
            if colour
            else BAR_GLYPHS[index % len(BAR_GLYPHS)]
        )
        legend.append(
            f"    {key} {label:<30.30} {size / GiB:>7.2f} GiB  {size / usable:>6.1%}"
        )

    if drawn < BAR_WIDTH:
        bar += "·" * (BAR_WIDTH - drawn)
    return bar, legend


def _render_phase(breakdown: PhaseBreakdown, colour: bool = False) -> str:
    usable = breakdown.device_usable_bytes
    total = breakdown.total_bytes
    status = "FITS" if breakdown.fits else "OVER BUDGET"
    if colour:
        status = (
            f"\033[32m{status}\033[0m"
            if breakdown.fits
            else f"\033[31m\033[1m{status}\033[0m"
        )

    bar, legend = _stacked_bar(breakdown, colour)
    capacity = (
        f"{'':<11}{total / GiB:.2f} of {usable / GiB:.2f} GiB usable "
        f"({breakdown.device_total_bytes / GiB:.0f} GiB card)"
    )
    overflow = " >>" if not breakdown.fits else ""
    lines = [
        f"{breakdown.phase.capitalize():<11}{bar}|{overflow} {status}",
        capacity,
        "",
        *legend,
    ]
    headroom = breakdown.headroom_bytes
    label = "Headroom" if headroom >= 0 else "SHORTFALL"
    lines.append(f"    {' '} {label:<30.30} {abs(headroom) / GiB:>7.2f} GiB")
    lines.extend(f"    ! {warning}" for warning in breakdown.warnings)
    return "\n".join(lines)


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
    "--allow-oversize",
    is_flag=True,
    help=(
        "Exit 0 even when a phase is over budget. The shortfall is still "
        "reported; this only changes the exit code."
    ),
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
    allow_oversize: bool,
    no_color: bool,
) -> None:
    """Size MANIFEST (a training manifest path) against a GPU.

    All run settings come from the manifest itself; only the device has to be
    named, because the manifest does not know what it will be scheduled on.
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
            allow_oversize=allow_oversize,
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
    allow_oversize: bool = False,
    no_color: bool = False,
) -> int:
    """Run the estimate and return an exit code. Separated from Click for tests."""
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

    algo = manifest.algorithm
    if not isinstance(algo, LLMAlgorithmSpec):
        print(
            f"Memory estimation covers the LLM fine-tuning algorithms; "
            f"{algo.name} builds its own (small) networks and is not sized.",
            file=sys.stderr,
        )
        return EXIT_USAGE
    model_id = algo.pretrained_model_name_or_path
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
            "note: --config geometry suppresses Hub n_params reconciliation; "
            "make sure the file matches the manifest's model.",
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
    blocked = not estimate.fits and not allow_oversize
    if as_json:
        payload = estimate.model_dump(by_alias=True)
        payload["fits"] = estimate.fits
        payload["blocked"] = blocked
        payload["advice"] = [a.model_dump() for a in advise(config)]
        print(json.dumps(payload, indent=2))
        return EXIT_OVER_BUDGET if blocked else EXIT_OK

    colour = not no_color and _use_colour(sys.stdout)
    print(f"Memory estimate — {config.model.model_id}")
    print("Training and generation are separate peaks, one bar each.\n")
    print(_render_phase(estimate.training, colour))
    print()
    print(_render_phase(estimate.generation, colour))
    if not estimate.fits:
        print("\nCheapest fixes:")
        for suggestion in advise(config):
            print(f"  - [{suggestion.phase}] {suggestion}")
        if allow_oversize:
            print(
                "\n--allow-oversize: over budget, submitting anyway. "
                "Expect an OOM unless the estimate is wrong in your favour."
            )
        else:
            print("\nBlocked. Re-run with --allow-oversize to submit regardless.")
    return EXIT_OVER_BUDGET if blocked else EXIT_OK


if __name__ == "__main__":
    memory_group()

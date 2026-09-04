# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
r"""Size a training manifest's GPU memory before it is submitted, and gate on
the answer.

Runs entirely on the closed-form core: no GPU, no profiling, no model
download. Every setting comes from the manifest — the same document the
submission carries — and the device from the resource class the run would be
scheduled on. The only thing fetched is the named checkpoint's own
``config.json`` (pass ``--config`` to read it from disk instead and stay
offline).

Prints the two independent phase bars (training and generation) as stacked
components against device capacity, and when a phase does not fit, ranks the
cheapest manifest changes that would make it.

Exit codes are the contract for callers that gate submissions:

===== ==========================================================
0      fits — safe to submit
3      over budget on at least one phase — submission blocked
2      usage error (bad arguments, unreadable manifest)
1      never returned deliberately: an exit of 1 is an unhandled
       crash (Python's default), so a caller can always tell "the
       gate ran and blocked you" from "the gate itself fell over"
===== ==========================================================

2 matches the usage-error convention Click and argparse already exit
with, and 3 is the first code that neither the runtime nor the argument
parser will ever produce on its own — which is what makes the verdict
codes unambiguous.

``--allow-oversize`` downgrades 3 to 0 for a caller that wants to submit
anyway; the report still shows the shortfall, so the override is recorded
rather than hidden. Estimates carry real error (a few percent, worse on the
corners), so this is a guard rail, not a proof.

Usage::

    arena memory estimate manifest.yaml --gpu "NVIDIA L4"
    arena memory solve max_model_len --inference --gpu "NVIDIA L4" \\
        --model Qwen/Qwen2.5-7B-Instruct

``python -m agilerl.arena.memory`` is equivalent, for a checkout without the
console script installed.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import click
from pydantic import ValidationError

from agilerl.arena.memory.advice import advise
from agilerl.arena.memory.estimator import PhaseBreakdown, estimate_run
from agilerl.arena.memory.manifest import (
    lookup_gpu,
    run_config_from_manifest,
)
from agilerl.arena.memory.solver import (
    INFERENCE_GPU_MEMORY_UTILIZATION,
    INFERENCE_MAX_NUM_SEQS,
    SOLVABLE_KNOBS,
    CannotSolve,
    architectural_context_limit,
    inference_run_config,
    solve,
    solve_inference,
)
from agilerl.arena.memory.specs import (
    DeviceSpec,
    GenerationKnobs,
    GiB,
    ModelArch,
    ModelSpec,
    RunConfig,
)
from agilerl.arena.models.algorithms.base import LLMAlgorithmSpec
from agilerl.arena.models.manifest import TrainingManifest

#: Exit codes -- the contract with the submission gate.
EXIT_OK = 0
EXIT_OVER_BUDGET = 3
EXIT_USAGE = 2


def checkpoint_param_count(model_id: str, arch: ModelArch) -> int | None:
    """Parameters the checkpoint puts on the device, from the Hub's safetensors
    index. Metadata only — no weights are fetched.

    Not simply the stored total: a tied-embedding checkpoint may still ship
    ``lm_head.weight`` as a second copy of the embedding table (Qwen3-0.6B and
    Qwen3-1.7B do; Qwen2.5 and Gemma 4 do not). ``from_pretrained`` re-ties
    them, so only one copy is ever resident, and counting the stored pair would
    over-state a 0.6B model by 311 MiB — which blocks runs that fit.

    ``None`` whenever the repo publishes no index, leaving the analytic count
    in charge.
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
    return AutoConfig.from_pretrained(model_id).to_dict()


#: Bar width in characters. 48 keeps a phase line inside 80 columns once the
#: label and the GiB figure are allowed for.
BAR_WIDTH = 48
#: One colour per component, cycled. 256-colour codes so the bar reads on both
#: light and dark terminals; ``BAR_GLYPHS`` is used when colour is unavailable,
#: because a stacked bar is meaningless if every segment looks the same.
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
    return [(c.label, c.bytes_) for c in breakdown.components if c.bytes_ > 0]


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
        want = max(1, round(BAR_WIDTH * size / usable)) if size else 0
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


def _resolve_device(
    gpu: str | None, device_gb: float | None, *, flag: str
) -> DeviceSpec | None:
    """A device from ``--gpu`` (catalogue) and/or a raw capacity in GiB."""
    if gpu is not None:
        info = lookup_gpu(gpu)
        if info is None:
            print(
                f"{flag}: unknown GPU {gpu!r}; pass the capacity via "
                f"{flag.replace('--gpu', '--device-gb')} instead.",
                file=sys.stderr,
            )
            raise SystemExit(EXIT_USAGE)
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
    "--allow-oversize",
    is_flag=True,
    help=(
        "Exit 0 even when a phase is over budget. The shortfall is still "
        "reported; this only downgrades the gate."
    ),
)
@click.option(
    "--contracted-moe",
    is_flag=True,
    help=(
        "Estimate as if the packed-expert adapter path computed its low-rank "
        "delta by contraction (the planned fix) instead of materializing "
        "effective weights. The default models the code as it runs today."
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
    contracted_moe: bool,
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
            contracted_moe=contracted_moe,
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
    contracted_moe: bool = False,
    no_color: bool = False,
) -> int:
    """The gate itself, separated from Click so callers get a plain int."""
    device = _resolve_device(gpu, device_gb, flag="--gpu")
    if device is None:
        print("Pass --gpu or --device-gb (see --help).", file=sys.stderr)
        return EXIT_USAGE
    gen_device = _resolve_device(gen_gpu, gen_device_gb, flag="--gen-gpu")

    try:
        manifest = TrainingManifest.get_validated(manifest_path, mode="python")
    except (OSError, ValueError, ValidationError) as err:
        print(f"could not validate {manifest_path}: {err}", file=sys.stderr)
        return EXIT_USAGE

    algo = manifest.algorithm
    if not isinstance(algo, LLMAlgorithmSpec):
        print(
            f"Memory estimation covers the LLM fine-tuning algorithms; "
            f"{algo.name} builds its own (small) networks and does not need a gate.",
            file=sys.stderr,
        )
        return EXIT_USAGE
    model_id = algo.pretrained_model_name_or_path
    if model_id is None:
        print(f"{manifest_path} names no pretrained model to size.", file=sys.stderr)
        return EXIT_USAGE
    try:
        model_config = _load_model_config(model_id, config_path)
    except (OSError, ValueError) as err:
        print(f"could not load the model config: {err}", file=sys.stderr)
        return EXIT_USAGE

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
    except ValueError as err:
        print(str(err), file=sys.stderr)
        return EXIT_USAGE

    if contracted_moe:
        config = config.model_copy(
            update={
                "training": config.training.model_copy(
                    update={"packed_moe_dispatch": "contracted"}
                )
            }
        )
    estimate = estimate_run(config)
    blocked = not estimate.fits and not allow_oversize
    if as_json:
        payload = estimate.model_dump(by_alias=True)
        payload["fits"] = estimate.fits
        payload["blocked"] = blocked
        print(json.dumps(payload, indent=2))
        return EXIT_OVER_BUDGET if blocked else EXIT_OK

    colour = not no_color and _use_colour(sys.stdout)
    placement = "colocated" if config.colocated else "disaggregated"
    print(f"Memory estimate — {config.model.model_id} ({placement})")
    print("Training and generation are separate peaks, never summed.\n")
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


@memory_group.command("solve")
@click.argument("knob", type=click.Choice(sorted(SOLVABLE_KNOBS), case_sensitive=True))
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
    "--model",
    "model_id",
    default=None,
    help="HuggingFace model id (required with --inference unless --config names one).",
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
    help="Context cap when solving a different setting. Ignored for max_model_len.",
)
@click.option(
    "--kv-cache-dtype",
    type=click.Choice(["auto", "fp8", "int8"], case_sensitive=True),
    default=None,
    help="KV cache dtype constraint.",
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
def memory_solve(
    knob: str,
    manifest_path: str | None,
    inference: bool,
    gpu: str | None,
    device_gb: float | None,
    model_id: str | None,
    config_path: str | None,
    max_num_seqs: int | None,
    gpu_memory_utilization: float | None,
    max_model_len: int | None,
    kv_cache_dtype: str | None,
    hi: int | None,
    as_json: bool,
    no_color: bool,
) -> None:
    """Solve SETTING: the largest value that still fits, given everything else.

    Invertible settings: max_model_len, max_num_seqs, micro_batch_size_per_gpu.
    --inference sizes a serving pod (the L4 context-window question);
    without it, MANIFEST is required and the solve is that training run.
    """
    sys.exit(
        solve_main(
            knob,
            manifest_path,
            inference=inference,
            gpu=gpu,
            device_gb=device_gb,
            model_id=model_id,
            config_path=config_path,
            max_num_seqs=max_num_seqs,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            kv_cache_dtype=kv_cache_dtype,
            hi=hi,
            as_json=as_json,
            no_color=no_color,
        )
    )


def solve_main(
    knob: str,
    manifest_path: str | None = None,
    *,
    inference: bool = False,
    gpu: str | None = None,
    device_gb: float | None = None,
    model_id: str | None = None,
    config_path: str | None = None,
    max_num_seqs: int | None = None,
    gpu_memory_utilization: float | None = None,
    max_model_len: int | None = None,
    kv_cache_dtype: str | None = None,
    hi: int | None = None,
    as_json: bool = False,
    no_color: bool = False,
) -> int:
    """Invert one setting. Separated from Click so callers get a plain int."""
    device = _resolve_device(gpu, device_gb, flag="--gpu")
    if device is None:
        print("Pass --gpu or --device-gb (see --help).", file=sys.stderr)
        return EXIT_USAGE

    if inference and manifest_path is not None:
        print("--inference does not take a training manifest.", file=sys.stderr)
        return EXIT_USAGE
    if not inference and manifest_path is None:
        print("Pass a MANIFEST, or --inference with --model.", file=sys.stderr)
        return EXIT_USAGE

    arch_limit: int | None = None
    try:
        if inference:
            config, arch_limit = _inference_config(
                device,
                model_id=model_id,
                config_path=config_path,
                max_num_seqs=max_num_seqs,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                kv_cache_dtype=kv_cache_dtype,
            )
        else:
            assert manifest_path is not None
            config, arch_limit = _manifest_config(
                manifest_path,
                device,
                config_path=config_path,
                max_num_seqs=max_num_seqs,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                kv_cache_dtype=kv_cache_dtype,
            )
    except (OSError, ValueError, ValidationError) as err:
        print(str(err), file=sys.stderr)
        return EXIT_USAGE

    bound = hi
    if knob == "max_model_len" and arch_limit is not None:
        bound = arch_limit if bound is None else min(bound, arch_limit)
    solver = solve_inference if inference else solve
    try:
        result = solver(config, knob, hi=bound)
    except CannotSolve as err:
        print(str(err), file=sys.stderr)
        return EXIT_OVER_BUDGET
    except ValueError as err:
        print(str(err), file=sys.stderr)
        return EXIT_USAGE

    if as_json:
        payload = {
            "knob": result.knob,
            "value": result.value,
            "limited_by": result.limited_by,
            "bound": result.bound,
            "mode": "inference" if inference else "training",
            "model": result.config.model.model_id,
            "generation": result.estimate.generation.model_dump(by_alias=True),
        }
        if not inference:
            payload["training"] = result.estimate.training.model_dump(by_alias=True)
        print(json.dumps(payload, indent=2))
        return EXIT_OK

    colour = not no_color and _use_colour(sys.stdout)
    reason = "checkpoint / --hi cap" if result.limited_by == "bound" else "GPU memory"
    mode = "inference" if inference else "training"
    print(f"Solved {result.knob} = {result.value}  ({reason})")
    print(
        f"  {result.config.model.model_id} · {mode} · "
        f"{result.config.generation.max_num_seqs} seqs · "
        f"gmu={result.config.generation.gpu_memory_utilization:g}"
    )
    print()
    if inference:
        print(_render_phase(result.estimate.generation, colour))
    else:
        print(_render_phase(result.estimate.training, colour))
        print()
        print(_render_phase(result.estimate.generation, colour))
    return EXIT_OK


def _generation_overrides(
    *,
    max_num_seqs: int | None,
    gpu_memory_utilization: float | None,
    max_model_len: int | None,
    kv_cache_dtype: str | None,
) -> dict[str, object]:
    updates: dict[str, object] = {}
    if max_num_seqs is not None:
        updates["max_num_seqs"] = max_num_seqs
    if gpu_memory_utilization is not None:
        updates["gpu_memory_utilization"] = gpu_memory_utilization
    if max_model_len is not None:
        updates["max_model_len"] = max_model_len
    if kv_cache_dtype is not None:
        updates["kv_cache_dtype"] = kv_cache_dtype
    return updates


def _inference_config(
    device: DeviceSpec,
    *,
    model_id: str | None,
    config_path: str | None,
    max_num_seqs: int | None,
    gpu_memory_utilization: float | None,
    max_model_len: int | None,
    kv_cache_dtype: str | None,
) -> tuple[RunConfig, int]:
    resolved_id = model_id or (
        Path(config_path).parent.name if config_path is not None else None
    )
    if resolved_id is None:
        msg = "--inference needs --model or --config."
        raise ValueError(msg)
    model_config, model = _model_spec_from_checkpoint(resolved_id, config_path)
    knobs = GenerationKnobs(
        gpu_memory_utilization=(
            INFERENCE_GPU_MEMORY_UTILIZATION
            if gpu_memory_utilization is None
            else gpu_memory_utilization
        ),
        max_num_seqs=(INFERENCE_MAX_NUM_SEQS if max_num_seqs is None else max_num_seqs),
        **({"max_model_len": max_model_len} if max_model_len is not None else {}),
        **({"kv_cache_dtype": kv_cache_dtype} if kv_cache_dtype is not None else {}),
    )
    return inference_run_config(model, device, knobs), architectural_context_limit(
        model_config
    )


def _manifest_config(
    manifest_path: str,
    device: DeviceSpec,
    *,
    config_path: str | None,
    max_num_seqs: int | None,
    gpu_memory_utilization: float | None,
    max_model_len: int | None,
    kv_cache_dtype: str | None,
) -> tuple[RunConfig, int]:
    manifest = TrainingManifest.get_validated(manifest_path, mode="python")
    algo = manifest.algorithm
    if not isinstance(algo, LLMAlgorithmSpec):
        msg = (
            f"Memory estimation covers the LLM fine-tuning algorithms; "
            f"{algo.name} builds its own (small) networks and does not need a gate."
        )
        raise ValueError(msg)
    resolved_id = algo.pretrained_model_name_or_path
    if resolved_id is None:
        msg = f"{manifest_path} names no pretrained model to size."
        raise ValueError(msg)
    model_config, model_spec = _model_spec_from_checkpoint(resolved_id, config_path)
    config = run_config_from_manifest(
        manifest,
        device,
        model_config,
        n_params=model_spec.n_params,
    )
    updates = _generation_overrides(
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        kv_cache_dtype=kv_cache_dtype,
    )
    if updates:
        config = config.model_copy(
            update={"generation": config.generation.model_copy(update=updates)}
        )
    return config, architectural_context_limit(model_config)


if __name__ == "__main__":
    memory_group()

# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
r"""Size an LLM RL run before it is submitted, and gate on the answer.

Runs entirely on the closed-form core: no GPU, no profiling, no model
download. The geometry comes from the checkpoint's own ``config.json``,
fetched by id or read from disk, so this is fast enough to sit in front of
every submission.

Prints the two independent phase bars (training and generation) as stacked
components against device capacity, and when a phase does not fit, ranks the
cheapest knob changes that would make it.

Exit codes are the contract for callers such as agilerl-arena:

===== ==========================================================
0      fits — safe to submit
3      over budget on at least one phase — submission blocked
2      usage error (bad or missing arguments)
===== ==========================================================

``--allow-oversize`` downgrades 3 to 0 for a caller that wants to submit
anyway; the report still shows the shortfall, so the override is recorded
rather than hidden. Estimates carry real error (a few percent, worse on the
corners), so this is a guard rail, not a proof.

Usage::

    agilerl-memory --model Qwen/Qwen2.5-3B-Instruct \\
        --device-gb 24 --max-model-len 4096 --group-size 8

``python -m agilerl.memory`` is equivalent, for a checkout without the
console script installed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from agilerl.memory.advice import advise
from agilerl.memory.estimator import PhaseBreakdown, estimate_run
from agilerl.memory.specs import (
    DeviceSpec,
    GenerationKnobs,
    GiB,
    ModelArch,
    ModelSpec,
    RunConfig,
    TrainingKnobs,
)

#: Exit codes -- the contract with agilerl-arena.
EXIT_OK = 0
EXIT_OVER_BUDGET = 3


def _load_model(
    args: argparse.Namespace,
) -> ModelSpec:
    """Resolve the model spec from a local config.json or a HF model id."""
    if args.config is not None:
        config = json.loads(Path(args.config).read_text())
        arch = ModelArch.from_hf_config(config)
        model_id = args.model or Path(args.config).parent.name or "local-model"
        return ModelSpec(model_id=model_id, arch=arch)

    try:
        from transformers import AutoConfig
    except ImportError:
        msg = (
            "transformers is not installed, so the config for "
            f"{args.model!r} cannot be fetched. Pass "
            "--config path/to/config.json instead."
        )
        raise SystemExit(msg) from None
    hf_config = AutoConfig.from_pretrained(args.model)
    arch = ModelArch.from_hf_config(hf_config.to_dict())
    return ModelSpec(model_id=args.model, arch=arch)


#: Bar width in characters. 48 keeps a phase line inside 80 columns once the
#: label and the GiB figure are allowed for.
BAR_WIDTH = 48
#: One colour per component, cycled. 256-colour codes so the bar reads on both
#: light and dark terminals; ``_STYLES`` falls back to distinct glyphs when
#: colour is unavailable, because a stacked bar is meaningless if every
#: segment looks the same.
_COLOURS = (33, 208, 71, 170, 214, 45, 203, 100)
_GLYPHS = ("#", "=", "+", "*", "o", ":", "~", ".")


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
        glyph = "█" if colour else _GLYPHS[index % len(_GLYPHS)]
        chunk = glyph * cells
        if colour and cells:
            chunk = f"\033[38;5;{_COLOURS[index % len(_COLOURS)]}m{chunk}\033[0m"
        bar += chunk
        key = (
            f"\033[38;5;{_COLOURS[index % len(_COLOURS)]}m█\033[0m"
            if colour
            else _GLYPHS[index % len(_GLYPHS)]
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agilerl-memory",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", help="HF model id (its config.json is fetched)")
    parser.add_argument(
        "--config", help="Path to a local HF config.json (offline mode)"
    )
    parser.add_argument(
        "--device-gb",
        type=float,
        default=24.0,
        help="Training device memory in GiB (default 24)",
    )
    parser.add_argument(
        "--gen-device-gb",
        type=float,
        default=None,
        help="Generation device memory in GiB; omit for colocated",
    )
    parser.add_argument("--algorithm", default="grpo")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--micro-batch", type=int, default=None)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument(
        "--quantization", choices=["none", "nf4", "int8"], default="none"
    )
    parser.add_argument("--weight-dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.3)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument(
        "--kv-cache-dtype", choices=["auto", "fp8", "int8"], default="auto"
    )
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the full estimate as JSON instead of the report",
    )
    parser.add_argument(
        "--allow-oversize",
        action="store_true",
        help=(
            "Exit 0 even when a phase is over budget. The shortfall is still "
            "reported; this only downgrades the gate."
        ),
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Plain glyphs instead of colour (also honours NO_COLOR)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.model and not args.config:
        print("Pass --model or --config (see --help).", file=sys.stderr)
        return 2

    model = _load_model(args)
    train_device = DeviceSpec(total_bytes=int(args.device_gb * GiB))
    gen_device = (
        DeviceSpec(total_bytes=int(args.gen_device_gb * GiB))
        if args.gen_device_gb is not None
        else None
    )
    config = RunConfig(
        model=model,
        train_device=train_device,
        gen_device=gen_device,
        training=TrainingKnobs(
            algorithm=args.algorithm,
            batch_size=args.batch_size,
            micro_batch_size_per_gpu=args.micro_batch,
            group_size=args.group_size,
            max_model_len=args.max_model_len,
            lora_rank=args.lora_rank,
            quantization=args.quantization,
            weight_dtype=args.weight_dtype,
        ),
        generation=GenerationKnobs(
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_num_seqs=args.max_num_seqs,
            max_model_len=args.max_model_len,
            kv_cache_dtype=args.kv_cache_dtype,
            enforce_eager=args.enforce_eager or None,
            max_lora_rank=args.lora_rank,
            weight_dtype=args.weight_dtype,
            concurrent_requests=args.batch_size * args.group_size,
        ),
        trainer_weight_variant="base",
    )

    estimate = estimate_run(config)
    blocked = not estimate.fits and not args.allow_oversize
    if args.json:
        payload = estimate.model_dump(by_alias=True)
        payload["fits"] = estimate.fits
        payload["blocked"] = blocked
        print(json.dumps(payload, indent=2))
        return EXIT_OVER_BUDGET if blocked else EXIT_OK

    colour = not args.no_color and _use_colour(sys.stdout)
    placement = "colocated" if config.colocated else "disaggregated"
    print(f"Memory estimate — {model.model_id} ({placement})")
    print("Training and generation are separate peaks, never summed.\n")
    print(_render_phase(estimate.training, colour))
    print()
    print(_render_phase(estimate.generation, colour))
    if not estimate.fits:
        print("\nCheapest fixes:")
        for suggestion in advise(config):
            print(f"  - [{suggestion.phase}] {suggestion}")
        if args.allow_oversize:
            print(
                "\n--allow-oversize: over budget, submitting anyway. "
                "Expect an OOM unless the estimate is wrong in your favour."
            )
        else:
            print("\nBlocked. Re-run with --allow-oversize to submit regardless.")
    return EXIT_OVER_BUDGET if blocked else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())

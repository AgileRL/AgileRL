# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
r"""CLI preflight: size an LLM RL run before launching it.

Prints the two independent phase bars (training and generation), flags
anything over budget, and ranks the cheapest fixes. Runs entirely on the
closed-form core: no GPU, no profiling, no model download. The geometry
comes from the checkpoint's own ``config.json``, fetched by id or read from
disk.

Usage::

    python -m agilerl.memory.preflight --model Qwen/Qwen2.5-3B-Instruct \\
        --device-gb 24 --max-model-len 4096 --group-size 8
"""

from __future__ import annotations

import argparse
import json
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


def _render_phase(breakdown: PhaseBreakdown) -> str:
    lines: list[str] = []
    usable = breakdown.device_usable_bytes
    total = breakdown.total_bytes
    status = "OK" if breakdown.fits else "OVER BUDGET"
    lines.append(
        f"{breakdown.phase.capitalize()} device: "
        f"{total / GiB:.1f} / {usable / GiB:.1f} GiB usable "
        f"({breakdown.device_total_bytes / GiB:.1f} GiB total) — {status}"
    )
    scale = max(total, usable)
    for component in breakdown.components:
        if component.bytes_ == 0:
            continue
        width = round(40 * component.bytes_ / scale) if scale else 0
        bar = "#" * max(width, 1)
        lines.append(
            f"  {component.label:<34} {component.bytes_ / GiB:>7.2f} GiB  {bar}"
        )
    headroom = breakdown.headroom_bytes
    if headroom >= 0:
        lines.append(f"  {'Headroom':<34} {headroom / GiB:>7.2f} GiB")
    else:
        lines.append(f"  {'Shortfall':<34} {-headroom / GiB:>7.2f} GiB")
    lines.extend(f"  ! {warning}" for warning in breakdown.warnings)
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agilerl.memory.preflight", description=__doc__
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
    if args.json:
        print(estimate.model_dump_json(by_alias=True, indent=2))
        return 0 if estimate.fits else 1

    placement = "colocated" if config.colocated else "disaggregated"
    print(f"Memory preflight — {model.model_id} ({placement})\n")
    print(_render_phase(estimate.training))
    print()
    print(_render_phase(estimate.generation))
    if not estimate.fits:
        print("\nCheapest fixes:")
        for suggestion in advise(config):
            print(f"  - [{suggestion.phase}] {suggestion}")
    return 0 if estimate.fits else 1


if __name__ == "__main__":
    sys.exit(main())

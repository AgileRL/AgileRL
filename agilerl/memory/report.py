# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Per-component memory breakdown for every calibrated model.

The sweep summary reports one error figure per phase, which says whether a
profile is trustworthy but not what it is made of. This prints the component
split behind those numbers — the same segments the Arena widget renders —
next to the measured peak for the same configuration, so predicted and
observed can be compared model by model.

Usage::

    python -m agilerl.memory.report
    python -m agilerl.memory.report --seq-len 4096 --micro-batch 8
"""

from __future__ import annotations

import argparse
import sys
from typing import cast

from agilerl.memory import formulas
from agilerl.memory.calibration import (
    ModelProfile,
    curated_profiles,
    load_profile,
)
from agilerl.memory.estimator import (
    estimate_generation,
    estimate_training,
    recommend_engine_budget,
)
from agilerl.memory.profiling.harness import SweepPoint
from agilerl.memory.specs import (
    Algorithm,
    DeviceSpec,
    GenerationKnobs,
    GiB,
    LoraTargetScope,
    TrainingKnobs,
)

TRAINING_KEYS = (
    "base_weights",
    "adapters",
    "grads_optimizer",
    "activations",
    "logits_workspace",
    "vllm_residual",
    "overhead",
)
GENERATION_KEYS = (
    "weights",
    "kv_cache",
    "activation_peak",
    "cuda_graphs",
    "lora_slots",
    "trainer_residual",
    "overhead",
)


def _device_of(profile: ModelProfile) -> DeviceSpec:
    assert profile.device is not None
    major, _, minor = (profile.device.compute_capability or "8.0").partition(".")
    return DeviceSpec.from_compute_capability(
        profile.device.total_bytes,
        int(major),
        int(minor or 0),
        name=profile.device.name,
    )


def _measured_total(
    profile: ModelProfile, phase: str, knobs: dict[str, int]
) -> float | None:
    """Measured peak for exactly this configuration, when the sweep hit it."""
    for point in profile.measured:
        if point.phase != phase:
            continue
        if all(int(point.knobs.get(k, -1)) == v for k, v in knobs.items()):
            return point.device_peak_bytes / GiB
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agilerl.memory.report", description=__doc__)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    args = parser.parse_args(argv)

    pairs = curated_profiles()
    if not pairs:
        print("No calibration fixtures checked in.", file=sys.stderr)
        return 1

    knob_match = {
        "seq_len": args.seq_len,
        "micro_batch": args.micro_batch,
        "group_size": args.group_size,
        "lora_rank": args.lora_rank,
    }
    print(
        f"Configuration: seq_len={args.seq_len} micro_batch={args.micro_batch} "
        f"group_size={args.group_size} lora_rank={args.lora_rank} "
        f"util={args.gpu_memory_utilization}\n"
    )

    for model_id, device_name in pairs:
        profile = load_profile(model_id, device_name=device_name)
        if profile is None or profile.model_spec is None or profile.device is None:
            continue
        model = profile.apply_realised_weights(profile.model_spec)
        device = _device_of(profile)
        # Inherit how the sweep actually configured LoRA. Defaulting to
        # all-linear would size adapters across every expert of an MoE
        # profiled with attention-only targets, over-predicting by GiBs.
        measured_point = SweepPoint.from_dict(profile.measured[0].knobs)
        train_knobs = TrainingKnobs(
            micro_batch_size_per_gpu=args.micro_batch,
            batch_size=args.micro_batch,
            group_size=args.group_size,
            trajectories_per_update=args.group_size,
            max_model_len=args.seq_len,
            lora_rank=args.lora_rank,
            lora_target_scope=cast("LoraTargetScope", measured_point.scope),
            algorithm=cast("Algorithm", measured_point.algorithm),
        )
        gen_knobs = GenerationKnobs(
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_num_seqs=args.group_size,
            max_model_len=args.seq_len,
            max_prompt_len=max(args.seq_len // 4, 8),
            max_lora_rank=args.lora_rank,
            concurrent_requests=args.group_size,
        )
        training = estimate_training(
            model,
            device,
            train_knobs,
            colocated=True,
            profile=profile,
            colocated_engine_reservation_bytes=(
                profile.sleeping_engine_residual_bytes
                if profile.sleeping_engine_residual_bytes is not None
                else formulas.SLEEPING_ENGINE_RESIDUAL_BYTES
            ),
        )
        generation = estimate_generation(
            model, device, gen_knobs, colocated=True, profile=profile
        )
        util, terms = recommend_engine_budget(model, device, gen_knobs)

        params = formulas.param_counts(model.arch).total
        arch = model.arch
        kind = (
            f"MoE {arch.n_experts}x{arch.n_experts_per_tok}active"
            if arch.is_moe
            else "dense"
        )
        print(f"=== {model_id}  @ {device_name} ===")
        print(
            f"    {params / 1e9:.1f}B params, {kind}, vocab {arch.vocab_size}, "
            f"{arch.n_layers} layers, {arch.n_kv_heads} kv-heads"
            + (f", window {arch.sliding_window}" if arch.sliding_window else "")
        )
        for phase, breakdown, keys in (
            ("TRAIN", training, TRAINING_KEYS),
            ("INFER", generation, GENERATION_KEYS),
        ):
            components = {c.key: c.bytes_ / GiB for c in breakdown.components}
            line = "  ".join(
                f"{k.split('_')[0]}={components.get(k, 0):.2f}"
                for k in keys
                if components.get(k, 0) > 0.005
            )
            measured = _measured_total(
                profile, "training" if phase == "TRAIN" else "generation", knob_match
            )
            suffix = (
                f"  [measured {measured:.2f}]"
                if measured is not None
                else "  [not measured at this config]"
            )
            print(f"  {phase} total {breakdown.total_bytes / GiB:6.2f} GiB{suffix}")
            print(f"        {line}")
        print(
            f"  ENGINE BUDGET to serve {gen_knobs.concurrency} seqs at "
            f"{args.seq_len} tokens: util >= {util:.2f} "
            f"(KV demand {terms['kv_demand'] / GiB:.2f} GiB)"
        )
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

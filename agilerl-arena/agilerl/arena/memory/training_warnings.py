# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""User-facing warnings for the training-phase estimate."""

from __future__ import annotations

from agilerl.arena.memory.formulas import ParamCounts, geometry_gap_warning
from agilerl.arena.memory.specs import TrainingSettings


def training_warnings(
    counts: ParamCounts,
    settings: TrainingSettings,
) -> list[str]:
    """User-facing training-phase warnings."""
    warnings: list[str] = []
    if not settings.uses_reference and settings.algorithm in ("ppo", "reinforce"):
        warnings.append(
            "At beta=0 the estimate skips the reference forward, but the fused "
            "no-grad pass still builds that row, so the run pays for one extra "
            "row of activations."
        )
    if settings.n_training_gpus > 1:
        n_gpus = settings.n_training_gpus
        if settings.fsdp is None:
            warnings.append(
                f"DDP over {n_gpus} GPUs. Each GPU holds a full copy of the "
                "weights, gradients, and Adam state."
            )
        elif settings.fsdp.reshard_after_forward:
            warnings.append(
                f"FSDP2 over {n_gpus} GPUs. Parameters are resharded after "
                "each unit's forward."
            )
        else:
            warnings.append(
                f"FSDP2 over {n_gpus} GPUs. Parameters stay gathered from "
                "forward through backward."
            )
    if (
        settings.lora_packed_target_matrices
        and settings.packed_moe_dispatch == "contracted"
    ):
        warnings.append(
            "packed_moe_dispatch='contracted' charges the split expert-LoRA "
            "residual (top-k gather, per-expert cat, fp32 GEMMs) on the "
            "checkpointed backward."
        )
    gap = geometry_gap_warning(counts)
    if gap:
        warnings.append(gap)
    if not settings.gradient_checkpointing:
        warnings.append(
            "Activation checkpointing is off: every block keeps its forward "
            "activations until its backward."
        )
    if settings.micro_batch_tokens is not None:
        warnings.append(
            "The gradient forward packs each micro-batch into one row of at "
            f"most {settings.micro_batch_tokens} tokens; the estimate prices "
            "that full row."
        )
    elif settings.sequence_packing:
        warnings.append(
            "The gradient forward packs its sequences into one row. "
            "This estimate sizes every row padded to max_model_len."
        )
    return warnings

# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Whether the trainer checkpoints activations."""

from __future__ import annotations

from collections.abc import Callable

from agilerl.arena.memory.breakdown import PhaseBreakdown
from agilerl.arena.memory.specs import DeviceSpec, ModelSpec, TrainingSettings


def resolve_gradient_checkpointing(
    model: ModelSpec,
    device: DeviceSpec,
    settings: TrainingSettings,
    training_breakdown: Callable[
        [ModelSpec, DeviceSpec, TrainingSettings, bool],
        PhaseBreakdown,
    ],
    orchestrated: bool = False,
) -> bool:
    """Whether to checkpoint activations at the settings' micro-batch and FSDP fields.

    Off skips the backward recompute of every block. It is taken when the
    step fits without checkpointing, with the margin the FSDP fields keep.
    Activation offload keeps checkpointing on: offloading every block's
    tensors costs more host traffic than the recompute saves.

    :param model: Training-side model spec.
    :param device: Training GPU.
    :param settings: Training settings, micro-batch and FSDP fields resolved.
    :param training_breakdown: Training-phase peak for one model, device, and
        settings.
    :param orchestrated: Charge the per-process job overhead.
    :return: Whether the trainer checkpoints activations.
    """
    if settings.activation_offload:
        return True
    uncheckpointed = settings.model_copy(update={"gradient_checkpointing": False})
    return not training_breakdown(
        model, device, uncheckpointed, orchestrated
    ).fits_with_buffer

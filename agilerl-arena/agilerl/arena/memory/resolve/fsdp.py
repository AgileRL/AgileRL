# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Unset FSDP chunk size and optimizer placement."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from agilerl.arena.memory.breakdown import PhaseBreakdown
from agilerl.arena.memory.formulas import ROUTED_CHUNK_MIB_CHOICES
from agilerl.arena.memory.specs import DeviceSpec, ModelSpec, TrainingSettings
from agilerl.arena.models.fsdp import FSDPConfig


def resolve_fsdp_memory(
    model: ModelSpec | None,
    device: DeviceSpec | None,
    settings: TrainingSettings,
    training_breakdown: Callable[
        [ModelSpec, DeviceSpec, TrainingSettings, bool],
        PhaseBreakdown,
    ],
    orchestrated: bool = False,
) -> FSDPConfig:
    """FSDP config with unset chunk size and optimizer placement filled in.

    :param model: Training-side model spec. Required when ``device`` is set.
    :param device: Training GPU. ``None`` skips the estimate: an unset chunk is
        the smallest choice, and an unset optimizer sits on the CPU unless
        ``cpu_offload`` is set. On a device, the chunk is the largest whose
        peak ``.fits``, and the optimizer stays on the GPU when that peak
        ``.fits_with_buffer``.
    :param settings: Training settings; ``fsdp`` must be set.
    :param training_breakdown: Training-phase peak for one model, device, and
        settings. Called when ``device`` is set.
    :param orchestrated: Charge the per-process job overhead.
    :return: Config with every memory field set. Set values are kept.
    """
    fsdp = settings.fsdp
    if fsdp is None:
        msg = "resolve_fsdp_memory needs settings.fsdp"
        raise ValueError(msg)

    def with_choice(chunk: int, offload: bool) -> FSDPConfig:
        return replace(fsdp, routed_expert_chunk_mib=chunk, optim_cpu_offload=offload)

    if device is None:
        chunk = fsdp.routed_expert_chunk_mib
        if chunk is None:
            chunk = ROUTED_CHUNK_MIB_CHOICES[-1]
        offload = fsdp.optim_cpu_offload
        if offload is None:
            offload = not fsdp.cpu_offload
        return with_choice(chunk, offload)

    if model is None:
        msg = "resolve_fsdp_memory needs a model when device is set"
        raise ValueError(msg)

    def estimate(chunk: int, offload: bool) -> PhaseBreakdown:
        trial = settings.model_copy(update={"fsdp": with_choice(chunk, offload)})
        return training_breakdown(model, device, trial, orchestrated)

    if fsdp.cpu_offload:
        lightest = False
    elif fsdp.optim_cpu_offload is None:
        lightest = True
    else:
        lightest = fsdp.optim_cpu_offload
    chunk = fsdp.routed_expert_chunk_mib
    if chunk is None:
        chunk = next(
            (c for c in ROUTED_CHUNK_MIB_CHOICES if estimate(c, lightest).fits),
            ROUTED_CHUNK_MIB_CHOICES[-1],
        )
    offload = fsdp.optim_cpu_offload
    if offload is None:
        offload = not fsdp.cpu_offload and not estimate(chunk, False).fits_with_buffer
    return with_choice(chunk, offload)

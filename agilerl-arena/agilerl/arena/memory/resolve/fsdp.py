# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Unset FSDP memory fields, and where full CPU offload runs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import TypeVar

from agilerl.arena.memory.breakdown import PhaseBreakdown
from agilerl.arena.memory.formulas import ROUTED_CHUNK_MIB_CHOICES
from agilerl.arena.memory.specs import DeviceSpec, ModelSpec, TrainingSettings
from agilerl.arena.models.fsdp import FSDPConfig

Choice = TypeVar("Choice", bool, int)


def cpu_offload_blocker(fsdp: FSDPConfig | None, async_rollout: bool) -> str | None:
    """Why FSDP ``cpu_offload`` cannot run, or ``None`` when it can.

    :param fsdp: The run's FSDP config, or ``None``.
    :param async_rollout: vLLM generates on rollout engines outside the trainer.
    :return: A short reason, or ``None`` when full CPU offload is off or served.
    """
    if fsdp is None or not fsdp.cpu_offload or not async_rollout:
        return None
    return (
        "fsdp.cpu_offload needs colocated vLLM generation: FSDP2's "
        "CPUOffloadPolicy keeps parameters on CPU. Use fsdp.optim_cpu_offload "
        "instead, which keeps parameters on the GPU."
    )


def _pick(value: Choice | None, default: Choice) -> Choice:
    """Return ``value``, or ``default`` when it is unset.

    :param value: Requested value, or ``None``.
    :param default: Value used when ``value`` is ``None``.
    :return: The chosen value.
    """
    return default if value is None else value


def _lightest(fsdp: FSDPConfig) -> FSDPConfig:
    """``fsdp`` with unset fields at their lightest, activation offload untouched.

    :param fsdp: The run's FSDP config.
    :return: Lightest config that keeps every set value.
    """
    return replace(
        fsdp,
        reshard_after_forward=_pick(fsdp.reshard_after_forward, True),
        routed_expert_chunk_mib=_pick(
            fsdp.routed_expert_chunk_mib, ROUTED_CHUNK_MIB_CHOICES[-1]
        ),
        optim_cpu_offload=(
            False if fsdp.cpu_offload else _pick(fsdp.optim_cpu_offload, True)
        ),
        cpu_offload=_pick(fsdp.cpu_offload, False),
    )


def _may_pick_cpu_offload(fsdp: FSDPConfig, settings: TrainingSettings) -> bool:
    """Whether the search may turn on full CPU offload for this run.

    CPUOffloadPolicy keeps parameters on CPU, which only colocated vLLM serves.

    :param fsdp: The run's FSDP config.
    :param settings: Training settings.
    :return: ``True`` when both offload fields are unset or off and vLLM is colocated.
    """
    return (
        fsdp.cpu_offload is None
        and not fsdp.optim_cpu_offload
        and settings.uses_generation_engine
        and not settings.async_rollout
    )


def resolve_fsdp_memory(
    model: ModelSpec | None,
    device: DeviceSpec | None,
    settings: TrainingSettings,
    training_breakdown: Callable[
        [ModelSpec, DeviceSpec, TrainingSettings, bool],
        PhaseBreakdown,
    ],
    orchestrated: bool = False,
    micro_batch_floor: int = 1,
) -> FSDPConfig:
    """FSDP config with unset memory fields set to the fastest choice that fits.

    The fastest run keeps parameters gathered after forward, activations on
    the GPU, the optimizer on the GPU and the largest routed-expert chunk.
    Speed is given up in this order: resharding after forward, a smaller
    chunk, a smaller micro-batch (searched by the caller afterwards), the
    optimizer to CPU, activation offload, then full CPU offload of parameters
    and gradients. Activation offload keeps every other field at its
    lightest. Full CPU offload keeps activation offload, turns optimizer
    offload off (the two are exclusive), and needs colocated vLLM. The
    optimizer, activation and full CPU offload choices are priced at
    ``micro_batch_floor`` with every other field at its lightest; resharding
    and the chunk at ``settings.micro_batch_size``.

    :param model: Training-side model spec. Required when ``device`` is set.
    :param device: Training GPU. ``None`` skips the estimate: unset fields take
        the lightest choice except activation offload and full CPU offload,
        which stay off.
    :param settings: Training settings; ``fsdp`` must be set.
    :param training_breakdown: Training-phase peak for one model, device, and
        settings. Called when ``device`` is set.
    :param orchestrated: Charge the per-process job overhead.
    :param micro_batch_floor: Smallest micro-batch the run can drop to.
        Activation offload turns on only when the lightest other choices are
        over the GPU at this micro-batch; full CPU offload only when they are
        still over it with activation offload on.
    :return: Config with every memory field set. Set values are kept.
    """
    fsdp = settings.fsdp
    if fsdp is None:
        msg = "resolve_fsdp_memory needs settings.fsdp"
        raise ValueError(msg)

    lightest = _lightest(fsdp)
    cpu_offload = lightest.cpu_offload

    if device is None:
        return replace(
            lightest, activation_offload=_pick(fsdp.activation_offload, False)
        )

    if model is None:
        msg = "resolve_fsdp_memory needs a model when device is set"
        raise ValueError(msg)

    def estimate(trial: FSDPConfig, micro: int | None = None) -> PhaseBreakdown:
        update: dict[str, object] = {
            "fsdp": trial,
            "activation_offload": bool(trial.activation_offload),
        }
        if micro is not None:
            update["micro_batch_size"] = micro
        return training_breakdown(
            model, device, settings.model_copy(update=update), orchestrated
        )

    activation_offload = fsdp.activation_offload
    if activation_offload is None:
        activation_offload = not estimate(
            replace(lightest, activation_offload=False), micro_batch_floor
        ).fits
    if activation_offload:
        offloaded = replace(lightest, activation_offload=True)
        if (
            _may_pick_cpu_offload(fsdp, settings)
            and not estimate(offloaded, micro_batch_floor).fits
        ):
            return replace(offloaded, cpu_offload=True, optim_cpu_offload=False)
        return offloaded
    offload = fsdp.optim_cpu_offload
    if offload is None:
        offload = (
            not fsdp.cpu_offload
            and not estimate(
                replace(
                    lightest,
                    optim_cpu_offload=False,
                    activation_offload=activation_offload,
                ),
                micro_batch_floor,
            ).fits_with_buffer
        )
    reshard = fsdp.reshard_after_forward
    if reshard is None:
        fastest = replace(
            fsdp,
            reshard_after_forward=False,
            routed_expert_chunk_mib=_pick(
                fsdp.routed_expert_chunk_mib, ROUTED_CHUNK_MIB_CHOICES[0]
            ),
            optim_cpu_offload=offload,
            activation_offload=activation_offload,
            cpu_offload=cpu_offload,
        )
        reshard = not estimate(fastest).fits_with_buffer

    def with_chunk(chunk: int) -> FSDPConfig:
        return replace(
            fsdp,
            reshard_after_forward=reshard,
            activation_offload=activation_offload,
            routed_expert_chunk_mib=chunk,
            optim_cpu_offload=offload,
            cpu_offload=cpu_offload,
        )

    chunk = fsdp.routed_expert_chunk_mib
    if chunk is None:
        chunk = next(
            (c for c in ROUTED_CHUNK_MIB_CHOICES if estimate(with_chunk(c)).fits),
            ROUTED_CHUNK_MIB_CHOICES[-1],
        )
    return with_chunk(chunk)

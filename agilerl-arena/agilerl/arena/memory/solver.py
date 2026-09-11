# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Invert one memory setting: the best value that still fits, given the rest.

The estimator answers "does this config fit?". This module answers the
inverse: hold every other input fixed and search one numeric field. That is
how a serving stack picks ``max_model_len`` for an L4.

Search, not a closed form. ``max_model_len`` feeds KV demand, the
scheduler token cap, the start-up profiling peak, and (on a hybrid in
``all`` mode) the Mamba cache — inverting any one term would miss the
others. Binary search over the estimator is exact against the same model
the gate uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from agilerl.arena.memory.estimator import (
    RunEstimate,
    estimate_run,
    generation_can_serve,
)
from agilerl.arena.memory.formulas import KV_BLOCK_SIZE_DEFAULT
from agilerl.arena.memory.specs import (
    DeviceSpec,
    GenerationSettings,
    ModelSpec,
    RunConfig,
)

PhaseName = Literal["training", "generation"]
LimitReason = Literal["memory", "bound"]

#: vLLM's own default. Dedicated inference has the card to itself, so the
#: engine may take nearly all of it.
INFERENCE_GPU_MEMORY_UTILIZATION = 0.9
#: Concurrent sequences a serving pod is sized for when the caller does not
#: say. ``1`` gives the longest single-request context; raise this to trade
#: context for throughput.
INFERENCE_MAX_NUM_SEQS = 8
#: Fallback architectural cap when ``config.json`` has no
#: ``max_position_embeddings``.
DEFAULT_CONTEXT_LIMIT = 131_072


@dataclass(frozen=True)
class FieldSpec:
    """One field the solver knows how to invert."""

    name: str
    group: Literal["training", "generation"]
    field: str
    lo: int
    default_hi: int
    #: Round the search down to this multiple. KV pages are 16 tokens.
    align: int = 1
    #: Generation only: the KV pool must cover worst-case demand, not just
    #: leave the resident peak under the card.
    require_kv_headroom: bool = False
    #: Other ``(group, field)`` pairs kept in lockstep. Training and
    #: generation ``max_model_len`` are the same manifest value.
    sync: tuple[tuple[str, str], ...] = ()


SOLVABLE_FIELDS: dict[str, FieldSpec] = {
    "max_model_len": FieldSpec(
        name="max_model_len",
        group="generation",
        field="max_model_len",
        lo=KV_BLOCK_SIZE_DEFAULT,
        default_hi=DEFAULT_CONTEXT_LIMIT,
        align=KV_BLOCK_SIZE_DEFAULT,
        require_kv_headroom=True,
        sync=(("training", "max_model_len"), ("generation", "max_model_len")),
    ),
    "max_num_seqs": FieldSpec(
        name="max_num_seqs",
        group="generation",
        field="max_num_seqs",
        lo=1,
        default_hi=256,
        require_kv_headroom=True,
    ),
    "micro_batch_size_per_gpu": FieldSpec(
        name="micro_batch_size_per_gpu",
        group="training",
        field="micro_batch_size_per_gpu",
        lo=1,
        default_hi=4096,
    ),
}


class CannotSolve(ValueError):
    """Even the lowest legal value of this setting does not fit."""


class SolveResult(BaseModel):
    """The best value of one setting, and the config that realises it."""

    model_config = ConfigDict(frozen=True)

    field: str
    value: int
    limited_by: LimitReason
    bound: int
    config: RunConfig
    estimate: RunEstimate


def architectural_context_limit(model_config: dict[str, Any]) -> int:
    """Hard cap from the checkpoint: RoPE / ``max_position_embeddings``."""
    text = model_config.get("text_config", model_config)
    raw = text.get("max_position_embeddings")
    if raw is None:
        return DEFAULT_CONTEXT_LIMIT
    return max(int(raw), 1)


def inference_run_config(
    model: ModelSpec,
    device: DeviceSpec,
    settings: GenerationSettings | None = None,
) -> RunConfig:
    """A dedicated serving GPU: no trainer residual, no job overhead.

    ``gen_device`` is set so the run is not colocated. Training settings stay
    at defaults and are ignored — inference has no trainer on the card.
    """
    generation = settings or GenerationSettings(
        gpu_memory_utilization=INFERENCE_GPU_MEMORY_UTILIZATION,
        max_num_seqs=INFERENCE_MAX_NUM_SEQS,
    )
    return RunConfig(
        model=model,
        train_device=device,
        gen_device=device,
        generation=generation,
        orchestrated=False,
    )


def solve_inference(
    config: RunConfig,
    field: str,
    *,
    hi: int | None = None,
) -> SolveResult:
    """``solve`` for a dedicated serving GPU: only the generation bar counts."""
    return solve(config, field, hi=hi, phases=("generation",))


def solve(
    config: RunConfig,
    field: str,
    *,
    hi: int | None = None,
    phases: tuple[PhaseName, ...] | None = None,
) -> SolveResult:
    """Largest value of ``setting`` at which ``config`` still fits.

    ``phases`` defaults to the phases that setting actually moves. Dedicated
    inference must pass ``("generation",)`` (or call :func:`solve_inference`)
    so a trainer bar that is not on the card cannot cap the search.
    """
    if field not in SOLVABLE_FIELDS:
        known = ", ".join(sorted(SOLVABLE_FIELDS))
        msg = f"Unknown field {field!r}; solvable: {known}."
        raise ValueError(msg)
    spec = SOLVABLE_FIELDS[field]
    bound = spec.default_hi if hi is None else hi
    if bound < spec.lo:
        msg = f"{field} upper bound {bound} is below the minimum {spec.lo}."
        raise ValueError(msg)
    checked = phases if phases is not None else _default_phases(config, spec)

    lo = spec.lo
    if not _fits(_apply(config, spec, lo), spec, checked):
        msg = (
            f"{field}={lo} already does not fit on this device; "
            "the model is larger than the card at the other settings given."
        )
        raise CannotSolve(msg)

    aligned_hi = _align_down(bound, spec.align)
    if aligned_hi < spec.lo:
        aligned_hi = spec.lo
    if _fits(_apply(config, spec, aligned_hi), spec, checked):
        solved = _apply(config, spec, aligned_hi)
        return SolveResult(
            field=field,
            value=aligned_hi,
            limited_by="bound",
            bound=bound,
            config=solved,
            estimate=estimate_run(solved),
        )

    best = lo
    low, high = lo, aligned_hi
    while low <= high:
        mid = _align_down((low + high) // 2, spec.align)
        if mid < spec.lo:
            mid = spec.lo
        if _fits(_apply(config, spec, mid), spec, checked):
            best = mid
            low = mid + spec.align
        else:
            high = mid - spec.align

    solved = _apply(config, spec, best)
    return SolveResult(
        field=field,
        value=best,
        limited_by="memory",
        bound=bound,
        config=solved,
        estimate=estimate_run(solved),
    )


def _default_phases(config: RunConfig, spec: FieldSpec) -> tuple[PhaseName, ...]:
    if spec.group == "training":
        return ("training",)
    if spec.name == "max_model_len" and config.training.uses_generation_engine:
        return ("training", "generation")
    if spec.name == "max_model_len":
        return ("training",)
    return ("generation",)


def _apply(config: RunConfig, spec: FieldSpec, value: int) -> RunConfig:
    updates: dict[str, object] = {}
    targets = spec.sync or ((spec.group, spec.field),)
    for group_name, field in targets:
        current = updates.get(group_name)
        if current is not None:
            group = current
        elif group_name == "training":
            group = config.training
        else:
            group = config.generation
        updates[group_name] = group.model_copy(update={field: value})
    return config.model_copy(update=updates)


def _fits(config: RunConfig, spec: FieldSpec, phases: tuple[PhaseName, ...]) -> bool:
    estimate = estimate_run(config)
    for name in phases:
        breakdown = estimate.generation if name == "generation" else estimate.training
        if spec.require_kv_headroom and name == "generation":
            if not generation_can_serve(breakdown):
                return False
        elif not breakdown.fits:
            return False
    return True


def _align_down(value: int, align: int) -> int:
    if align <= 1:
        return value
    return value - (value % align)

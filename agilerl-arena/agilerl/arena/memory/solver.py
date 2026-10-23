# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Invert one memory setting: the largest value that still fits, given the rest.

Scan up from the minimum to the first fit, then bisect for the top of that
run. Generation context is not monotonic (the resident bar dips while the
engine is underfilled), so plain bisection can miss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from agilerl.arena.memory.estimator import (
    PhaseName,
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

LimitReason = Literal["memory", "bound"]
SyncTarget = tuple[PhaseName, str]

# vLLM's own default. Dedicated inference has the card to itself, so the
# engine may take nearly all of it.
INFERENCE_GPU_MEMORY_UTILIZATION = 0.9
# Concurrent sequences a dedicated serving GPU is sized for when the caller does not
# say. ``1`` gives the longest single-request context; raise this to trade
# context for throughput.
INFERENCE_MAX_NUM_SEQS = 8
# Fallback architectural cap when ``config.json`` has no
# ``max_position_embeddings``.
DEFAULT_CONTEXT_LIMIT = 131_072


@dataclass(frozen=True)
class FieldSpec:
    """One field the solver knows how to invert."""

    name: str
    group: PhaseName
    field: str
    lo: int
    default_hi: int
    # Round the search down to this multiple. KV pages are 16 tokens.
    align: int = 1
    # Generation only: the KV pool must cover worst-case demand, not just
    # leave the resident peak under the card.
    require_kv_headroom: bool = False
    # Other ``(group, field)`` pairs kept in lockstep. Training and
    # generation ``max_model_len`` are the same manifest value.
    sync: tuple[SyncTarget, ...] = ()


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
}


class CannotSolve(ValueError):
    """No value of this setting fits on this device."""


class SolveResult(BaseModel):
    """The best value of one setting, and the config that realises it."""

    model_config = ConfigDict(frozen=True)

    field: str
    value: int
    limited_by: LimitReason
    bound: int
    config: RunConfig
    estimate: RunEstimate
    # Non-checked phases that are over budget at the solved value.
    unchecked_over_budget: tuple[str, ...] = ()


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
    """A dedicated serving GPU: generation only, no job overhead.

    Training settings stay at defaults and are ignored — inference has no
    trainer on the card.
    """
    generation = (
        settings
        if settings is not None
        else GenerationSettings(
            gpu_memory_utilization=INFERENCE_GPU_MEMORY_UTILIZATION,
            max_num_seqs=INFERENCE_MAX_NUM_SEQS,
        )
    )
    return RunConfig(
        model=model,
        train_device=device,
        gen_device=device,
        generation=generation,
    )


def solve_inference(
    config: RunConfig,
    field: str,
    *,
    hi: int | None = None,
) -> SolveResult:
    """``solve`` for a dedicated serving GPU: only the generation bar counts."""
    result = solve(config, field, hi=hi, phases=("generation",))
    # No trainer on a serving GPU; the dummy training bar is not a signal.
    return result.model_copy(update={"unchecked_over_budget": ()})


def solve(
    config: RunConfig,
    field: str,
    *,
    hi: int | None = None,
    phases: tuple[PhaseName, ...] | None = None,
) -> SolveResult:
    """Largest value of ``field`` at which ``config`` still fits.

    Training uses the same underprediction buffer as ``arena memory estimate``.
    Generation stays on the point estimate.

    :param phases: Phases to check; defaults to the phases ``field`` moves.
        Dedicated inference passes ``("generation",)``.
    :raises CannotSolve: No value fits; a ``ValueError`` subclass.
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

    aligned_hi = _align_down(bound, spec.align)
    # Generation fits() has one valley, so scan to the first fit, then bisect.
    found = _first_fit(config, spec, checked, aligned_hi)
    if found is None:
        msg = (
            f"no {field} value up to {bound} fits on this device; "
            "the model is larger than the card at the other settings given."
        )
        raise CannotSolve(msg)
    first, first_estimate = found

    solved = _apply(config, spec, aligned_hi)
    ok, estimate = _fits(solved, spec, checked)
    if ok:
        return SolveResult(
            field=field,
            value=aligned_hi,
            limited_by="bound",
            bound=bound,
            config=solved,
            estimate=estimate,
            unchecked_over_budget=_unchecked(estimate, checked),
        )

    best, best_estimate = first, first_estimate
    low, high = first, aligned_hi
    while low <= high:
        mid = _align_down((low + high) // 2, spec.align)
        candidate = _apply(config, spec, mid)
        ok, estimate = _fits(candidate, spec, checked)
        if ok:
            best, best_estimate = mid, estimate
            low = mid + spec.align
        else:
            high = mid - spec.align

    return SolveResult(
        field=field,
        value=best,
        limited_by="memory",
        bound=bound,
        config=_apply(config, spec, best),
        estimate=best_estimate,
        unchecked_over_budget=_unchecked(best_estimate, checked),
    )


def _default_phases(config: RunConfig, spec: FieldSpec) -> tuple[PhaseName, ...]:
    engine = config.training.uses_generation_engine
    if spec.name == "max_model_len":
        return ("training", "generation") if engine else ("training",)
    if not engine:
        msg = (
            f"{spec.name} sizes the generation engine, but "
            f"{config.training.algorithm} starts no engine."
        )
        raise ValueError(msg)
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


def _first_fit(
    config: RunConfig,
    spec: FieldSpec,
    phases: tuple[PhaseName, ...],
    aligned_hi: int,
) -> tuple[int, RunEstimate] | None:
    """Lowest fitting value, scanning up from the minimum in align steps."""
    value = spec.lo
    while value <= aligned_hi:
        candidate = _apply(config, spec, value)
        ok, estimate = _fits(candidate, spec, phases)
        if ok:
            return value, estimate
        value += spec.align
    return None


def _unchecked(
    estimate: RunEstimate, checked: tuple[PhaseName, ...]
) -> tuple[str, ...]:
    """Non-checked phases that are over budget at the solved candidate."""
    bars = (("training", estimate.training), ("generation", estimate.generation))
    return tuple(
        name for name, bar in bars if name not in checked and not bar.fits_with_buffer
    )


def _fits(
    config: RunConfig, spec: FieldSpec, phases: tuple[PhaseName, ...]
) -> tuple[bool, RunEstimate]:
    estimate = estimate_run(config)
    for name in phases:
        breakdown = estimate.generation if name == "generation" else estimate.training
        if spec.require_kv_headroom and name == "generation":
            if not generation_can_serve(breakdown):
                return False, estimate
        elif not breakdown.fits_with_buffer:
            return False, estimate
    return True, estimate


def _align_down(value: int, align: int) -> int:
    return value - (value % align)

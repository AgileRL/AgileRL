# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Prescriptive advice: which setting to change when a phase is over budget.

Each candidate is a setting change re-run through the estimator; the reported
saving is the exact delta the model predicts.
"""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel, ConfigDict

from agilerl.arena.memory.estimator import PhaseName, estimate_run
from agilerl.arena.memory.specs import GiB, RunConfig


class Advice(BaseModel):
    """One ranked suggestion for a phase."""

    model_config = ConfigDict(frozen=True)

    phase: PhaseName
    action: str
    saves_bytes: int

    def __str__(self) -> str:
        return f"{self.action} (saves ~{self.saves_bytes / GiB:.1f} GiB)"


def _update(group: str, **updates: object) -> Callable[[RunConfig], RunConfig]:
    def apply(c: RunConfig) -> RunConfig:
        current = c.training if group == "training" else c.generation
        return c.model_copy(update={group: current.model_copy(update=updates)})

    return apply


def _shorter_context_candidate(
    current: int,
) -> tuple[str, Callable[[RunConfig], RunConfig]]:
    """Shorten max_model_len on training and generation together."""
    shorter = int(current * 0.75)

    def apply(c: RunConfig) -> RunConfig:
        training = c.training.model_copy(
            update={"max_model_len": int(c.training.max_model_len * 0.75)}
        )
        generation = c.generation.model_copy(
            update={"max_model_len": int(c.generation.max_model_len * 0.75)}
        )
        return c.model_copy(update={"training": training, "generation": generation})

    return (
        f"Reduce max_model_len ({current} -> {shorter}; training and engine together)",
        apply,
    )


def _training_candidates(
    config: RunConfig,
) -> list[tuple[str, Callable[[RunConfig], RunConfig]]]:
    t = config.training

    def with_training(**updates: object) -> Callable[[RunConfig], RunConfig]:
        return _update("training", **updates)

    candidates: list[tuple[str, Callable[[RunConfig], RunConfig]]] = []
    update_rows = t.trajectories_per_update or t.group_size
    if update_rows > 1:
        halved = max(1, update_rows // 2)
        candidates.append(
            (
                (
                    f"Halve the update rows ({update_rows} -> {halved}); shrink "
                    "batch_size or group_size"
                ),
                with_training(trajectories_per_update=halved),
            )
        )
    if t.max_model_len > 512:
        candidates.append(_shorter_context_candidate(t.max_model_len))
    if not t.activation_offload:
        candidates.append(
            (
                (
                    "Offload backward-saved activations to host RAM "
                    "(activation_offload=True)"
                ),
                with_training(activation_offload=True),
            )
        )
    if t.lora_rank > 8:
        candidates.append(
            (
                f"Halve the LoRA rank ({t.lora_rank} -> {t.lora_rank // 2})",
                with_training(lora_rank=t.lora_rank // 2),
            )
        )
    if t.use_separate_reference_adapter:
        candidates.append(
            (
                (
                    "Drop the separate reference adapter (pins the reference to "
                    "the initial policy)"
                ),
                with_training(use_separate_reference_adapter=False),
            )
        )
    return candidates


def _generation_candidates(
    config: RunConfig,
) -> list[tuple[str, Callable[[RunConfig], RunConfig]]]:
    g = config.generation

    def with_generation(**updates: object) -> Callable[[RunConfig], RunConfig]:
        return _update("generation", **updates)

    candidates: list[tuple[str, Callable[[RunConfig], RunConfig]]] = []
    if not g.enforce_eager:
        candidates.append(
            (
                "Skip CUDA-graph capture (enforce_eager=True; costs decode throughput)",
                with_generation(enforce_eager=True),
            )
        )
    if g.max_num_seqs > 1:
        candidates.append(
            (
                (
                    f"Halve max_num_seqs ({g.max_num_seqs} -> "
                    f"{max(1, g.max_num_seqs // 2)})"
                ),
                with_generation(max_num_seqs=max(1, g.max_num_seqs // 2)),
            )
        )
    if g.max_model_len > 512:
        candidates.append(_shorter_context_candidate(g.max_model_len))
    if g.gpu_memory_utilization > 0.15:
        lower = round(g.gpu_memory_utilization - 0.1, 2)
        candidates.append(
            (
                (
                    f"Lower gpu_memory_utilization "
                    f"({g.gpu_memory_utilization} -> {lower}; shrinks the KV pool)"
                ),
                with_generation(gpu_memory_utilization=lower),
            )
        )
    return candidates


def advise(
    config: RunConfig,
    phase: PhaseName | None = None,
    top_n: int | None = 5,
) -> tuple[Advice, ...]:
    """Rank setting changes by the memory they save on the given phase (or on
    whichever phases are over budget when ``phase`` is None; falls back to
    both phases when everything fits).
    """
    baseline = estimate_run(config)
    phases: list[PhaseName]
    if phase is not None:
        phases = [phase]
    else:
        phases = []
        if not baseline.training.fits:
            phases.append("training")
        if not baseline.generation.fits:
            phases.append("generation")
        if not phases:
            phases = ["training", "generation"]

    results: list[Advice] = []
    for target in phases:
        candidates = (
            _training_candidates(config)
            if target == "training"
            else _generation_candidates(config)
        )
        before = (
            baseline.training.total_bytes
            if target == "training"
            else baseline.generation.total_bytes
        )
        for action, apply in candidates:
            after_estimate = estimate_run(apply(config))
            after = (
                after_estimate.training.total_bytes
                if target == "training"
                else after_estimate.generation.total_bytes
            )
            saved = before - after
            if saved > 0:
                results.append(Advice(phase=target, action=action, saves_bytes=saved))
    results.sort(key=lambda a: a.saves_bytes, reverse=True)
    return tuple(results[:top_n] if top_n is not None else results)

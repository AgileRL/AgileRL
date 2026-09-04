# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Prescriptive advice: which setting to change when a phase is over budget.

Rather than hand-maintaining savings formulas per lever, each candidate is a
setting change that is re-run through the estimator; the reported saving is
the exact delta the model predicts. This keeps advice automatically
consistent with the calculation core and restricted to settings that actually
exist in the framework.
"""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel, ConfigDict

from agilerl.arena.memory import formulas
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


def _update(field: str, **updates: object) -> Callable[[RunConfig], RunConfig]:
    def apply(c: RunConfig) -> RunConfig:
        return c.model_copy(
            update={field: getattr(c, field).model_copy(update=updates)}
        )

    return apply


def _training_candidates(
    config: RunConfig,
) -> list[tuple[str, Callable[[RunConfig], RunConfig]]]:
    t = config.training

    def with_training(**updates: object) -> Callable[[RunConfig], RunConfig]:
        return _update("training", **updates)

    candidates: list[tuple[str, Callable[[RunConfig], RunConfig]]] = []
    # Offered first because it is the only fix here that changes nothing about
    # the run: same batch, same context, same optimisation. eager builds an
    # S x S score matrix that is quadratic in context; SDPA and flex_attention
    # do not.
    resolved = formulas.resolve_attn_implementation(
        t.attn_implementation, config.train_device.flash_attn_installed
    )
    if formulas.materializes_attention_scores(resolved):
        candidates.append(
            (
                (
                    f"Switch attn_implementation ({resolved} -> "
                    "flex_attention); eager builds the score matrix, which "
                    "flex_attention tiles instead"
                ),
                with_training(attn_implementation="flex_attention"),
            )
        )
    if t.grad_rows > 1:
        candidates.append(
            (
                (
                    f"Halve the training micro-batch "
                    f"({t.grad_rows} -> {max(1, t.grad_rows // 2)}); keep the "
                    "effective batch via gradient accumulation"
                ),
                with_training(micro_batch_size_per_gpu=max(1, t.grad_rows // 2)),
            )
        )
    if t.batch_size > 1:
        candidates.append(
            (
                (
                    f"Halve the no-grad logprob batch "
                    f"({t.batch_size} -> {max(1, t.batch_size // 2)})"
                ),
                with_training(batch_size=max(1, t.batch_size // 2)),
            )
        )
    if t.max_model_len > 512:
        shorter = int(t.max_model_len * 0.75)
        candidates.append(
            (
                f"Reduce max_model_len ({t.max_model_len} -> {shorter})",
                with_training(max_model_len=shorter),
            )
        )
    if t.quantization == "none":
        candidates.append(
            (
                "Quantize the trainer base to NF4 (QLoRA)",
                with_training(quantization="nf4"),
            )
        )
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
    device = config.generation_device

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
    if g.kv_cache_dtype == "auto" and device.supports_fp8:
        candidates.append(
            (
                "Use fp8 KV cache (kv_cache_dtype='fp8')",
                with_generation(kv_cache_dtype="fp8"),
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
        shorter = int(g.max_model_len * 0.75)
        candidates.append(
            (
                f"Reduce engine max_model_len ({g.max_model_len} -> {shorter})",
                with_generation(max_model_len=shorter),
            )
        )
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
    quantized = [
        v.name
        for v in config.model.variants
        if v.quantization != "none" and v.name != g.weight_variant
    ]
    if quantized:
        candidates.append(
            (
                f"Load the quantized engine variant {quantized[0]!r}",
                with_generation(weight_variant=quantized[0]),
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
        before = getattr(baseline, target).total_bytes
        for action, apply in candidates:
            after_estimate = estimate_run(apply(config))
            saved = before - getattr(after_estimate, target).total_bytes
            if saved > 0:
                results.append(Advice(phase=target, action=action, saves_bytes=saved))
    results.sort(key=lambda a: a.saves_bytes, reverse=True)
    return tuple(results[:top_n] if top_n is not None else results)

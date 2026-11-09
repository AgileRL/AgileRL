# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Choose omitted vLLM engine width for rollout engines."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from agilerl.arena.memory.breakdown import PhaseBreakdown
from agilerl.arena.memory.estimator import estimate_generation
from agilerl.arena.memory.generation import KV_MARGIN_BYTES, drop_engine_fast_paths
from agilerl.arena.memory.kv_cache import VLLM_ENGINE_MAX_NUM_BATCHED_TOKENS
from agilerl.arena.memory.manifest import (
    MANIFEST_WEIGHT_VARIANTS,
    configured_rollout_engines,
    generation_settings_from_manifest,
)
from agilerl.arena.memory.resolve.batch import catalog_model_spec
from agilerl.arena.memory.solver import solve
from agilerl.arena.memory.specs import DeviceSpec, ModelSpec, RunConfig
from agilerl.arena.models.algorithms.rollout_llm import RolloutLLMSpec
from agilerl.arena.models.manifest import TrainingManifest
from agilerl.arena.models.model_info import SUPPORTED_MODEL_INFO


@lru_cache(maxsize=len(SUPPORTED_MODEL_INFO))
def _engine_model(hub_id: str) -> ModelSpec:
    """Catalog model the rollout engine serves, with the manifest's weight variants."""
    return catalog_model_spec(hub_id).model_copy(
        update={"variants": MANIFEST_WEIGHT_VARIANTS}
    )


def resolve_rollout_engines_per_agent(
    configured: int | Literal["auto"],
    total_gpus: int,
    pop_size: int,
    training_gpus_per_agent: int,
    gpus_per_engine: int,
) -> int:
    """Rollout engines each population member starts.

    :param configured: Manifest ``rollout_engines_per_agent``. A count is kept.
    :param total_gpus: GPUs in the allocation.
    :param pop_size: Population members.
    :param training_gpus_per_agent: Trainer GPUs per member.
    :param gpus_per_engine: GPUs per engine, its tensor-parallel width.
    :return: The set count, else the GPUs left after the trainers split
        evenly across members, rounded down.
    """
    if configured != "auto":
        return configured
    surplus = total_gpus - pop_size * training_gpus_per_agent
    return surplus // (pop_size * gpus_per_engine)


def _width_candidates(
    leftover: int,
    rollout_engines_per_agent: int | Literal["auto"],
    gpus_per_node: int,
) -> list[tuple[int, int]]:
    """Tensor-parallel widths that fit the leftover GPUs, with their engine counts.

    :param leftover: GPUs left after the trainers are placed.
    :param rollout_engines_per_agent: Engines per member, or ``auto``.
    :param gpus_per_node: GPUs on one node. One engine stays on one node.
    :return: ``(tensor_parallel_size, engines)`` pairs.
    """
    if rollout_engines_per_agent == "auto":
        return [
            (width, leftover // width)
            for width in range(1, leftover + 1)
            if leftover % width == 0 and width <= gpus_per_node
        ]
    engines = int(rollout_engines_per_agent)
    return [
        (width, engines)
        for width in range(1, leftover + 1)
        if engines * width <= leftover and width <= gpus_per_node
    ]


def _resident_sequences(
    manifest: TrainingManifest,
    model: ModelSpec,
    device: DeviceSpec,
    width: int,
    engines: int,
    adapter_sync: bool = False,
) -> int:
    """Sequences ``engines`` engines of ``width`` GPUs keep resident; 0 when none fits.

    A pinned ``max_num_seqs`` counts the share of it the KV pool holds; the
    rest queue (:func:`queue_warning`).
    """
    vllm = manifest.algorithm.vllm_config
    try:
        trial = generation_settings_from_manifest(
            manifest,
            tensor_parallel_size=width,
            rollout_engines=engines,
            adapter_sync=adapter_sync,
        )
        if "max_num_seqs" in vllm.model_fields_set:
            breakdown = estimate_generation(model, device, trial)
            if not breakdown.fits:
                return 0
            return engines * sequences_held(breakdown, trial.max_num_seqs)
        run = RunConfig(
            model=model,
            train_device=device,
            gen_device=device,
            generation=trial,
        )
        # KV beyond an engine's episodes stays idle; ties favour more engines.
        solved = solve(
            run,
            "max_num_seqs",
            hi=trial.max_num_seqs,
            phases=("generation",),
            kv_margin_bytes=KV_MARGIN_BYTES,
        )
    except ValueError:
        # Illegal Mamba TP widths raise; a width that cannot be solved scores 0.
        return 0
    return engines * solved.value


def sequences_held(breakdown: PhaseBreakdown, requested: int) -> int:
    """How many of ``requested`` sequences this generation bar can keep resident.

    :param breakdown: Generation-phase breakdown.
    :param requested: Sequences the engine was asked to hold.
    :return: ``requested`` when the KV pool covers them, otherwise the share
        of them the pool can hold.
    """
    kv = next((c for c in breakdown.components if c.key == "kv_cache"), None)
    if kv is None:
        return requested
    demand = kv.detail["worst_case_demand"]
    usable = max(kv.detail["pool"] - KV_MARGIN_BYTES, 0)
    if demand <= usable:
        return requested
    return int(requested * usable / demand)


def queue_warning(breakdown: PhaseBreakdown, requested: int, width: int) -> str | None:
    """Warn when an engine of ``width`` GPUs holds fewer than the requested sequences.

    :param breakdown: Generation-phase breakdown at ``width``.
    :param requested: Sequences the engine was asked to hold.
    :param width: Engine tensor-parallel width.
    :return: The warning, or ``None`` when every sequence fits.
    """
    held = sequences_held(breakdown, requested)
    if held >= requested:
        return None
    return (
        f"tensor_parallel_size={width} holds {held} of {requested} sequences; "
        "generation may queue"
    )


def servable_engine_widths(
    manifest: TrainingManifest,
    device: DeviceSpec,
    max_width: int,
    cache: dict[tuple, bool] | None = None,
) -> list[int]:
    """Engine widths up to ``max_width`` that vLLM can shard and one engine keeps a sequence resident at.

    :param manifest: Validated rollout manifest of a catalogued model.
    :param device: The rollout engine's GPU.
    :param max_width: Widest engine to try.
    :param cache: Widths already checked for this manifest's engine.
    :return: Those widths, ascending.
    """
    hub_id = manifest.algorithm.pretrained_model_name_or_path
    model = _engine_model(hub_id)
    arch = model.arch
    checked = {} if cache is None else cache

    def serves(width: int) -> bool:
        key = (device.name, device.total_bytes, width)
        if key not in checked:
            checked[key] = _resident_sequences(manifest, model, device, width, 1) > 0
        return checked[key]

    # vLLM splits attention heads evenly; KV heads split or replicate evenly.
    return [
        width
        for width in range(1, max_width + 1)
        if arch.n_heads % width == 0
        and (arch.n_kv_heads % width == 0 or width % arch.n_kv_heads == 0)
        and serves(width)
    ]


def resolve_engine_tensor_parallel(
    manifest: TrainingManifest,
    total_gpus: int,
    device: DeviceSpec,
    gpus_per_node: int | None = None,
    adapter_sync: bool = False,
) -> int | None:
    """Assign the omitted engine width that keeps the most sequences resident.

    A width that holds every requested sequence beats one that queues.
    Sized at the manifest's trainer count; a searched GPU split picks its
    width in :func:`~agilerl.arena.memory.resolve.run.resolve_run_memory`.

    :param manifest: Validated training manifest.
    :param total_gpus: GPUs in the allocation.
    :param device: One GPU of that allocation.
    :param gpus_per_node: GPUs on one node. Defaults to ``total_gpus``.
    :param adapter_sync: Weight sync pushes the adapter even where the engines
        could serve merged weights, so each engine holds a LoRA slot.
    :return: The assigned width, or None when the manifest is left unchanged.
    """
    spec = manifest.algorithm
    if not isinstance(spec, RolloutLLMSpec) or spec.vllm_config is None:
        return None
    vllm = spec.vllm_config
    if vllm.tensor_parallel_size is not None:
        return None
    hub_id = spec.pretrained_model_name_or_path
    info = SUPPORTED_MODEL_INFO.get(hub_id) if hub_id is not None else None
    if info is None:
        return None
    per_node = total_gpus if gpus_per_node is None else gpus_per_node
    leftover = (
        total_gpus
        - manifest.training.pop_size * manifest.training.training_gpus_per_agent
    )
    if leftover < 1:
        return None
    candidates = _width_candidates(
        leftover, configured_rollout_engines(manifest), per_node
    )
    if not candidates:
        return None

    model = _engine_model(hub_id)
    # A pinned sequence count that fits beats a higher count that queues.
    requested = vllm.max_num_seqs if "max_num_seqs" in vllm.model_fields_set else None
    best: tuple[bool, int, int, int] | None = None
    for width, engines in candidates:
        score = _resident_sequences(
            manifest, model, device, width, engines, adapter_sync
        )
        serves = requested is not None and score == engines * requested
        rank = (serves, score, engines, width)
        if score > 0 and (best is None or rank > best):
            best = rank
    if best is None:
        return None
    chosen = best[3]
    vllm.tensor_parallel_size = chosen
    return chosen


def resolve_engine_fast_path(
    manifest: TrainingManifest, device: DeviceSpec, adapter_sync: bool = False
) -> RolloutLLMSpec | None:
    """Pin off the engine defaults :func:`drop_engine_fast_paths` turns off on ``device``.

    Writes ``vllm_engine_args.speculative_config: null``,
    ``vllm_engine_args.lora_dual_stream: false`` and
    ``vllm_config.max_num_batched_tokens`` for the defaults it drops.

    :param manifest: Validated async rollout manifest.
    :param device: The rollout engine's GPU.
    :param adapter_sync: Weight sync pushes the adapter even where the engines
        could serve merged weights.
    :return: The rewritten algorithm spec, or None when the manifest is left unchanged.
    """
    spec = manifest.algorithm
    if not isinstance(spec, RolloutLLMSpec) or spec.vllm_config is None:
        return None
    vllm = spec.vllm_config
    hub_id = spec.pretrained_model_name_or_path
    info = SUPPORTED_MODEL_INFO.get(hub_id) if hub_id is not None else None
    if info is None or vllm.tensor_parallel_size is None:
        return None
    settings = generation_settings_from_manifest(manifest, adapter_sync=adapter_sync)
    lean = drop_engine_fast_paths(_engine_model(hub_id), device, settings)
    if lean == settings:
        return None
    if settings.speculative_drafting is None:
        spec.vllm_engine_args["speculative_config"] = None
    if settings.lora_dual_stream:
        spec.vllm_engine_args["lora_dual_stream"] = False
    if settings.max_num_batched_tokens is None:
        batched = VLLM_ENGINE_MAX_NUM_BATCHED_TOKENS
        if spec.vllm_engine_args.get("enable_chunked_prefill") is False:
            # Without chunked prefill vLLM rejects a step smaller than one full prompt.
            batched = max(batched, spec.max_model_len)
        vllm.max_num_batched_tokens = batched
    return spec

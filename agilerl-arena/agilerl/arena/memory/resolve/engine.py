# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Choose omitted vLLM engine width for rollout engines."""

from __future__ import annotations

from typing import Literal

from agilerl.arena.memory.estimator import estimate_generation, generation_can_serve
from agilerl.arena.memory.generation import KV_MARGIN_BYTES, drop_engine_fast_paths
from agilerl.arena.memory.kv_cache import VLLM_ENGINE_MAX_NUM_BATCHED_TOKENS
from agilerl.arena.memory.manifest import (
    MANIFEST_WEIGHT_VARIANTS,
    configured_rollout_engines,
    generation_settings_from_manifest,
)
from agilerl.arena.memory.solver import solve
from agilerl.arena.memory.specs import DeviceSpec, ModelArch, ModelSpec, RunConfig
from agilerl.arena.models.algorithms.rollout_llm import RolloutLLMSpec
from agilerl.arena.models.manifest import TrainingManifest
from agilerl.arena.models.model_info import SUPPORTED_MODEL_INFO, ModelInfo


def _engine_model(hub_id: str, info: ModelInfo) -> ModelSpec:
    """Catalog model the rollout engine serves, with the manifest's weight variants."""
    counted = info.inspected["num_params"]
    n_params = counted if isinstance(counted, int) else None
    return ModelSpec(
        model_id=hub_id,
        arch=ModelArch.from_hf_config(info.config),
        n_params=n_params,
        variants=MANIFEST_WEIGHT_VARIANTS,
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
) -> int:
    """Sequences ``engines`` engines of ``width`` GPUs keep resident; 0 when none fits."""
    vllm = manifest.algorithm.vllm_config
    try:
        trial = generation_settings_from_manifest(
            manifest, tensor_parallel_size=width, rollout_engines=engines
        )
        if "max_num_seqs" in vllm.model_fields_set:
            breakdown = estimate_generation(model, device, trial)
            if not generation_can_serve(breakdown, KV_MARGIN_BYTES):
                return 0
            return engines * trial.max_num_seqs
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


def servable_engine_widths(
    manifest: TrainingManifest, device: DeviceSpec, max_width: int
) -> list[int]:
    """Engine widths up to ``max_width`` that vLLM can shard and one engine keeps a sequence resident at.

    :param manifest: Validated rollout manifest of a catalogued model.
    :param device: The rollout engine's GPU.
    :param max_width: Widest engine to try.
    :return: Those widths, ascending.
    """
    hub_id = manifest.algorithm.pretrained_model_name_or_path
    model = _engine_model(hub_id, SUPPORTED_MODEL_INFO[hub_id])
    arch = model.arch
    # vLLM splits attention heads evenly; KV heads split or replicate evenly.
    return [
        width
        for width in range(1, max_width + 1)
        if arch.n_heads % width == 0
        and (arch.n_kv_heads % width == 0 or width % arch.n_kv_heads == 0)
        and _resident_sequences(manifest, model, device, width, 1) > 0
    ]


def resolve_engine_tensor_parallel(
    manifest: TrainingManifest,
    total_gpus: int,
    device: DeviceSpec,
    gpus_per_node: int | None = None,
) -> int | None:
    """Assign the omitted engine width that keeps the most sequences resident.

    Sized at the manifest's trainer count; a searched GPU split picks its
    width in :func:`~agilerl.arena.memory.resolve.split.resolve_run_memory`.

    :param manifest: Validated training manifest.
    :param total_gpus: GPUs in the allocation.
    :param device: One GPU of that allocation.
    :param gpus_per_node: GPUs on one node. Defaults to ``total_gpus``.
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

    model = _engine_model(hub_id, info)
    best: tuple[int, int, int] | None = None
    for width, engines in candidates:
        score = _resident_sequences(manifest, model, device, width, engines)
        rank = (score, engines, width)
        if score > 0 and (best is None or rank > best):
            best = rank
    if best is None:
        return None
    chosen = best[2]
    vllm.tensor_parallel_size = chosen
    return chosen


def resolve_engine_fast_path(
    manifest: TrainingManifest, device: DeviceSpec
) -> RolloutLLMSpec | None:
    """Pin off the engine defaults :func:`drop_engine_fast_paths` turns off on ``device``.

    Writes ``vllm_engine_args.speculative_config: null``,
    ``vllm_engine_args.lora_dual_stream: false`` and
    ``vllm_config.max_num_batched_tokens`` for the defaults it drops.

    :param manifest: Validated async rollout manifest.
    :param device: The rollout engine's GPU.
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
    settings = generation_settings_from_manifest(manifest)
    lean = drop_engine_fast_paths(_engine_model(hub_id, info), device, settings)
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

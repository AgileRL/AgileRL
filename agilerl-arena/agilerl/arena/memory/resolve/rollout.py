# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Whether a Ray run generates on async rollout engines, how many it starts, its cycle size, whether it drafts, how it syncs LoRA and its LoRA streams."""

from __future__ import annotations

import math
from typing import Any, Literal

from agilerl.arena.models.algorithms.base import AlgorithmSpec
from agilerl.arena.models.algorithms.rollout_llm import (
    RolloutLLMSpec,
    drafting_rejected_sampling,
)


def async_rollout_algorithm(algorithm: type[AlgorithmSpec]) -> bool:
    """Whether Ray runs this algorithm on async rollout engines.

    Ray runs every algorithm that generates its own rollouts async and ignores
    ``training.rollout_mode``. The other env type fails in Ray either way.

    :param algorithm: The manifest's algorithm spec class.
    :return: ``True`` for the rollout LLM algorithms.
    """
    return issubclass(algorithm, RolloutLLMSpec)


def run_rollout_engines(
    async_rollout: bool, configured: int | Literal["auto"]
) -> int | Literal["auto"]:
    """Rollout engines per member a Ray run starts: an async ``0`` is ``auto``.

    :param async_rollout: :func:`async_rollout_algorithm` of the run.
    :param configured: ``training.rollout_engines_per_agent`` as written.
    :return: ``configured``, or ``auto`` for an async run that leaves it 0.
    """
    if async_rollout and not configured:
        return "auto"
    return configured


def rollout_engines_blocker(
    async_rollout: bool, configured: int | Literal["auto"]
) -> str | None:
    """Why a run cannot start the configured rollout engines, or ``None``.

    :param async_rollout: :func:`async_rollout_algorithm` of the run.
    :param configured: ``training.rollout_engines_per_agent`` as written.
    :return: The error for engines on a run that does not generate, or ``None``.
    """
    if async_rollout or configured == 0:
        return None
    return f"rollout_engines_per_agent={configured} requires async rollout."


def default_rollout_batch_size(batch_size: int, trainers: int, engines: int) -> int:
    """Groups per rollout cycle for an unset ``training.rollout_batch_size``.

    One learn step of groups, rounded up so every trainer shard and every
    engine gets an equal share.

    :param batch_size: The algorithm's ``batch_size``.
    :param trainers: Trainer GPUs per member.
    :param engines: Rollout engines per member.
    :return: ``batch_size`` rounded up to a multiple of ``lcm(trainers, engines)``.
    """
    multiple = math.lcm(trainers, engines)
    return math.ceil(batch_size / multiple) * multiple


def apply_run_rollout_mode(training: dict[str, Any], async_rollout: bool) -> None:
    """Set a raw training section's rollout mode and engines as Ray runs them.

    :param training: The manifest's ``training`` section, edited in place.
    :param async_rollout: :func:`async_rollout_algorithm` of the run.
    """
    if not async_rollout:
        training.pop("rollout_mode", None)
        return
    training["rollout_mode"] = "async"
    training["rollout_engines_per_agent"] = run_rollout_engines(
        True, training.get("rollout_engines_per_agent", 0)
    )


MULTI_TURN_SPECULATIVE_CONFIG: dict[str, Any] = {
    "method": "ngram",
    "num_speculative_tokens": 4,
    "prompt_lookup_max": 4,
    "prompt_lookup_min": 2,
}
"""vLLM ``speculative_config`` a drafting rollout engine starts with by default."""


def drafts_by_default(
    multi_turn: bool, hybrid_ssm: bool, min_p: float, logit_bias: bool
) -> bool:
    """Whether a rollout engine drafts with n-gram lookup when ``speculative_config`` is omitted.

    Later turns repeat env text from earlier turns, so n-gram drafts are
    accepted. A single-turn completion shares little with its prompt and
    decodes slower with drafting, since vLLM turns off async scheduling for
    n-gram drafts. Hybrid recurrent-state models decode without drafting,
    since vLLM must roll their state back past rejected drafts. vLLM rejects
    every request that sets :func:`drafting_rejected_sampling` on a drafting
    engine.

    :param multi_turn: An episode can run more than one turn.
    :param hybrid_ssm: The model has Mamba or linear-attention layers.
    :param min_p: The algorithm's ``min_p``.
    :param logit_bias: Rollout requests carry a ``logit_bias``.
    :return: ``True`` for a multi-turn engine on an attention-only model
        whose requests set neither ``min_p`` nor ``logit_bias``.
    """
    return (
        multi_turn
        and not hybrid_ssm
        and not drafting_rejected_sampling(min_p, logit_bias)
    )


MERGED_SYNC_BUCKET_BYTES = 256 << 20
"""Bytes of merged weights one merged-sync transfer round moves, unless one tensor is larger."""


def drafts_with_mtp(speculative_config: dict[str, Any] | None) -> bool:
    """Whether a vLLM ``speculative_config`` can draft with MTP layers.

    vLLM 0.32 detects the method from a draft ``model``'s config when
    ``method`` is omitted or ``draft_model``, and picks ``mtp`` for an MTP
    draft config, so such a config counts as MTP.

    :param speculative_config: ``vllm_engine_args.speculative_config``.
    :return: ``True`` for method ``mtp``, a model-specific ``*_mtp`` method,
        or a draft ``model`` other than n-gram with the method left to vLLM.
    """
    if speculative_config is None:
        return False
    method = speculative_config.get("method") or ""
    if method in ("", "draft_model"):
        return speculative_config.get("model") not in (None, "ngram", "[ngram]")
    return method == "mtp" or method.endswith("_mtp")


def checkpoint_quantized(config: dict[str, Any]) -> bool:
    """Whether vLLM loads a checkpoint with this ``config.json`` as quantized weights.

    vLLM 0.32 reads ``quantization_config``, or compressed-tensors'
    ``compression_config``, from the config or its ``text_config``.

    :param config: The checkpoint's raw ``config.json``.
    :return: ``True`` when either section carries one of those keys.
    """
    sections = (config, config.get("text_config") or {})
    return any(
        section.get(key) is not None
        for section in sections
        for key in ("quantization_config", "compression_config")
    )


def merged_lora_sync_eligible(
    quantization: str | None,
    quantized_checkpoint: bool,
    tower_lora: bool,
    mtp_drafting: bool,
) -> bool:
    """Whether a rollout engine can serve merged LoRA weights with LoRA off.

    Every rollout engine serves one policy, so it can hold that policy's
    merged weights. vLLM quantizes weights at load, both for the
    ``quantization`` engine argument and for a checkpoint that ships a
    quantization config, so such an engine cannot take merged bf16 tensors
    and keeps serving the adapter. vLLM keys cached vision-encoder outputs by
    LoRA name only, so an engine with tower LoRA keeps serving the adapter
    too. Merged sync does not update MTP draft weights, so an engine that
    drafts with MTP keeps serving the adapter.

    :param quantization: ``vllm_config.quantization``.
    :param quantized_checkpoint: :func:`checkpoint_quantized` of the engine's checkpoint.
    :param tower_lora: Whether the adapter targets vision-tower modules.
    :param mtp_drafting: :func:`drafts_with_mtp` of the engine's ``speculative_config``.
    :return: ``True`` for an unquantized engine on an unquantized checkpoint,
        without tower LoRA or MTP drafting.
    """
    unquantized = quantization is None or quantization.lower() == "none"
    return (
        unquantized and not quantized_checkpoint and not tower_lora and not mtp_drafting
    )


def merged_sync_bucket_bytes(total_bytes: int, largest_tensor_bytes: int) -> int:
    """Bytes of the NIXL buffer that streams merged weights, one round at a time.

    :param total_bytes: Merged bytes of every tensor the adapter touches.
    :param largest_tensor_bytes: Merged bytes of the largest of them.
    :return: :data:`MERGED_SYNC_BUCKET_BYTES`, or the largest tensor when it
        is larger, and never more than ``total_bytes``.
    """
    return min(total_bytes, max(MERGED_SYNC_BUCKET_BYTES, largest_tensor_bytes))


def runs_dual_stream_lora(moe: bool) -> bool:
    """Whether a rollout engine runs LoRA layers on two CUDA streams (``VLLM_LORA_ENABLE_DUAL_STREAM``).

    On H100, vLLM 0.32 dual-stream LoRA hits an illegal memory access in the
    startup profile run of Nemotron-3.5 Lightning and gives a NaN loss at the
    first learn on Qwen3.6-35B-A3B, so MoE engines run LoRA on one stream.

    :param moe: The model has routed experts.
    :return: ``True`` for a dense model.
    """
    return not moe

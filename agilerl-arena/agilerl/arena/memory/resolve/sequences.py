# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Omitted vLLM ``max_num_seqs`` and ``gpu_memory_utilization``."""

from __future__ import annotations

import math

# Sequences a colocated engine decodes when ``vllm_config`` sets none.
COLOCATED_MAX_NUM_SEQS = 16
# Share of the card an omitted ``gpu_memory_utilization`` reserves when
# generation shares the trainer GPU.
OMITTED_GPU_MEMORY_UTILIZATION = 0.3
# vLLM's own default, for an engine with its GPUs to itself: a rollout engine
# or dedicated inference.
ROLLOUT_ENGINE_GPU_MEMORY_UTILIZATION = 0.9


def prompts_per_engine(rollout_batch_size: int, n_engines: int) -> int:
    """Prompts one engine receives from a job batch.

    :param rollout_batch_size: Prompts in the job.
    :param n_engines: Engines sharing that batch.
    :return: ``ceil(rollout_batch_size / n_engines)``.
    """
    return math.ceil(rollout_batch_size / n_engines)


def resolve_gpu_memory_utilization(configured: float | None, colocated: bool) -> float:
    """Share of the card one vLLM engine reserves.

    :param configured: Manifest ``gpu_memory_utilization``. A set value is kept.
    :param colocated: Generation shares the trainer GPU.
    :return: The set value, else 0.3 colocated and 0.9 on a rollout engine.
    """
    if configured is not None:
        return configured
    if colocated:
        return OMITTED_GPU_MEMORY_UTILIZATION
    return ROLLOUT_ENGINE_GPU_MEMORY_UTILIZATION


def resolve_max_num_seqs(
    configured: int | None,
    prompts: int,
    group_size: int,
    colocated: bool,
) -> int:
    """Sequences one vLLM engine decodes at once.

    :param configured: Manifest ``max_num_seqs``. A set value is kept.
    :param prompts: Prompts this engine receives (:func:`prompts_per_engine`).
    :param group_size: Completions per prompt, at the HPO ceiling when
        ``group_size`` mutates.
    :param colocated: Generation shares the trainer GPU. An omitted field is
        :data:`COLOCATED_MAX_NUM_SEQS`.
    :return: The set value, 16 when colocated, otherwise ``prompts * group_size``.
    """
    if configured is not None:
        return configured
    if colocated:
        return COLOCATED_MAX_NUM_SEQS
    return prompts * group_size

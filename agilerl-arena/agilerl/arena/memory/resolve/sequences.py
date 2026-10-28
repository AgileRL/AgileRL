# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Omitted vLLM ``max_num_seqs``."""

from __future__ import annotations

import math

# Sequences a colocated engine decodes when ``vllm_config`` sets none.
COLOCATED_MAX_NUM_SEQS = 8
# Share of the card an omitted ``gpu_memory_utilization`` reserves.
OMITTED_GPU_MEMORY_UTILIZATION = 0.3


def prompts_per_engine(rollout_batch_size: int, n_engines: int) -> int:
    """Prompts one engine receives from a job batch.

    :param rollout_batch_size: Prompts in the job.
    :param n_engines: Engines sharing that batch. ``auto`` is already 1 here.
    :return: ``ceil(rollout_batch_size / n_engines)``.
    """
    return math.ceil(rollout_batch_size / n_engines)


def resolve_max_num_seqs(
    configured: int | None,
    prompts: int,
    group_size: int,
    colocated: bool,
) -> int:
    """Sequences one vLLM engine decodes at once.

    :param configured: Manifest ``max_num_seqs``. A set value is kept.
    :param prompts: Prompts this engine receives.
    :param group_size: Completions per prompt.
    :param colocated: Generation shares the trainer GPU. An omitted field is
        :data:`COLOCATED_MAX_NUM_SEQS`.
    :return: The set value, 8 when colocated, otherwise ``prompts * group_size``.
    """
    if configured is not None:
        return configured
    if colocated:
        return COLOCATED_MAX_NUM_SEQS
    return prompts * group_size

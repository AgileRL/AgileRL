# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Generation settings an OPD teacher changes on the rollout engine."""

from __future__ import annotations

from typing import Any

from agilerl.arena.models.distillation import DistillationTeacherConfig
from agilerl.arena.models.networks import VLLMConfig


def rollout_engine_teacher_settings(
    teacher: DistillationTeacherConfig | None,
    max_model_len: int,
    vllm: VLLMConfig,
) -> dict[str, Any]:
    """Settings an OPD teacher changes on the rollout engine.

    A self-distillation or adapter teacher scores in the rollout engine, ahead
    of a full episode. A teacher adapter takes a second LoRA slot.

    :param teacher: The OPD teacher, or ``None``.
    :param max_model_len: The policy's context length.
    :param vllm: The rollout engine's vLLM config.
    :return: ``max_model_len``, prompt-logprob flags, and ``served_loras``.
    """
    scores = teacher is not None and teacher.model is None
    context_tokens = 0 if teacher is None else teacher.rollout_engine_context_tokens
    return {
        "max_model_len": max_model_len + context_tokens,
        "scores_prompt_logprobs": scores,
        "reserves_prompt_logprobs": scores and vllm.gpu_memory_utilization is None,
        "served_loras": 2 if teacher is not None and teacher.adapter else 1,
    }

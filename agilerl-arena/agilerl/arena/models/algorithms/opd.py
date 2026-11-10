# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""On-policy distillation (OPD) algorithm specification."""

from __future__ import annotations

from pydantic import Field, model_validator
from typing_extensions import Self

from agilerl.arena.models.algorithms.base import LLMAlgorithmSpec
from agilerl.arena.models.algorithms.grpo import GRPOSpec
from agilerl.arena.models.descriptions import BETA, GROUP_SIZE, TEMPERATURE, TOP_K
from agilerl.arena.models.distillation import (
    CheckpointPoolConfig,
    DistillationTeacherConfig,
)
from agilerl.arena.models.registry import register


@register()
class OPDSpec(GRPOSpec):
    """GRPO whose per-token advantage adds the teacher's log-prob gap to the policy's."""

    checkpoint_pool: CheckpointPoolConfig | None = Field(
        default=None,
        description=(
            "Opt-in stage that scores one finished run's checkpoints and "
            "distils the one covering the most tasks the student misses."
        ),
    )
    teacher: DistillationTeacherConfig = Field(
        default_factory=DistillationTeacherConfig,
        description=(
            "The model that scores every rollout: by default the live policy, "
            "given each task's reference trace; else teacher.adapter or "
            "teacher.model. The per-token gap is added to the task advantage."
        ),
    )
    group_size: int = Field(
        default=4,
        ge=1,
        description=GROUP_SIZE
        + " With no task reward the teacher term is the whole update, and a "
        "group of 1 spends the same samples on more prompts.",
    )
    temperature: float = Field(default=1.0, description=TEMPERATURE)
    top_k: int = Field(default=0, ge=0, description=TOP_K)
    beta: float = Field(default=0.0, ge=0.0, le=1.0, description=BETA)

    @model_validator(mode="after")
    def _teacher_scores_the_unmodified_policy(self) -> Self:
        """The teacher compares log-probs of the distribution rollout sampled from."""
        problems = [
            f"{name}={value}"
            for name, value, plain in (
                ("temperature", self.temperature, 1.0),
                ("top_k", self.top_k, 0),
                ("top_p", self.top_p, 1.0),
                ("min_p", self.min_p, 0.0),
                ("repetition_penalty", self.repetition_penalty, 1.0),
                ("whiten_advantages", self.whiten_advantages, False),
                ("filter_zero_adv", self.filter_zero_adv, False),
                ("answer_continuation", self.answer_continuation, False),
            )
            if value != plain
        ]
        if self.constrain_answer_pattern is not None:
            problems.append("constrain_answer_pattern is set")
        if problems:
            msg = (
                "OPD's teacher scores the policy's unmodified distribution, so "
                "sampling must use temperature 1, top_k 0, top_p 1, min_p 0 and "
                "repetition_penalty 1, with no answer grammar, and the group "
                "advantages must stay unwhitened and unfiltered. Got: "
                + ", ".join(problems)
            )
            raise ValueError(msg)
        return self


def distillation_teacher(spec: LLMAlgorithmSpec) -> DistillationTeacherConfig | None:
    """The teacher of an OPD spec, or ``None`` for every other algorithm.

    :param spec: Algorithm spec.
    :type spec: LLMAlgorithmSpec
    :return: The teacher settings, or ``None`` when the spec trains without one.
    :rtype: DistillationTeacherConfig | None
    """
    return spec.teacher if isinstance(spec, OPDSpec) else None


def teacher_engine_gpus(spec: LLMAlgorithmSpec) -> int:
    """GPUs a separate teacher model adds to each rollout engine.

    :param spec: Algorithm spec.
    :type spec: LLMAlgorithmSpec
    :return: The teacher engine's GPUs, or 0 when no separate model scores.
    :rtype: int
    """
    teacher = distillation_teacher(spec)
    return 0 if teacher is None or teacher.model is None else teacher.model.gpus

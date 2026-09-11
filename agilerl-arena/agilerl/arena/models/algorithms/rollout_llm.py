# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Shared rollout-LLM algorithm specification."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field, model_validator
from typing_extensions import Self

from agilerl.arena.models.algorithms.base import LLMAlgorithmSpec
from agilerl.arena.models.descriptions import (
    BETA,
    COSINE_LR,
    IS_CAP,
    IS_CORRECTION,
    MAX_OUTPUT_TOKENS,
    MICRO_BATCH,
    MIN_OUTPUT_TOKENS,
    MIN_P,
    MINI_BATCH,
    REPETITION_PENALTY,
    TEMPERATURE,
    TOP_K,
    TOP_P,
    USE_VLLM,
    VLLM_CONFIG,
)
from agilerl.arena.models.env import LLMEnvType
from agilerl.arena.models.networks import CosineLRScheduleConfig, VLLMConfig


class RolloutLLMSpec(LLMAlgorithmSpec):
    """Shared surface for the LLM algorithms that generate their own rollouts."""

    temperature: float = Field(default=0.9, description=TEMPERATURE)
    repetition_penalty: float = Field(
        default=1.0, ge=0.0, description=REPETITION_PENALTY
    )
    top_p: float = Field(default=0.95, ge=0.0, le=1.0, description=TOP_P)
    top_k: int = Field(default=50, ge=0, description=TOP_K)
    min_p: float = Field(default=0.0, ge=0.0, le=1.0, description=MIN_P)
    max_output_tokens: int | None = Field(default=None, description=MAX_OUTPUT_TOKENS)
    beta: float = Field(default=0.001, ge=0.0, le=1.0, description=BETA)
    mini_batch_size: int | None = Field(default=None, ge=1, description=MINI_BATCH)
    micro_batch_size_per_gpu: int | None = Field(
        default=None, ge=1, description=MICRO_BATCH
    )
    min_output_tokens: int | None = Field(
        default=None, ge=0, description=MIN_OUTPUT_TOKENS
    )
    cosine_lr_schedule_config: CosineLRScheduleConfig | None = Field(
        default=None, description=COSINE_LR
    )
    vllm_config: VLLMConfig | None = Field(default=None, description=VLLM_CONFIG)
    use_vllm: bool = Field(default=False, description=USE_VLLM)
    vllm_importance_sampling_correction: bool = Field(
        default=True, description=IS_CORRECTION
    )
    vllm_importance_sampling_cap: float = Field(default=2.0, ge=0.0, description=IS_CAP)

    env_type: ClassVar[LLMEnvType] = LLMEnvType.ROLLOUT

    @model_validator(mode="after")
    def _validate_vllm_config(self) -> Self:
        if self.use_vllm and self.vllm_config is None:
            msg = (
                "use_vllm is set but no vllm_config was provided in the algorithm "
                "section of the manifest."
            )
            raise ValueError(msg)
        return self

    def _max_output_tokens(self) -> int | None:
        return self.max_output_tokens

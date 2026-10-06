# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""GRPO algorithm specification."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from agilerl.arena.models.algorithms.rollout_llm import RolloutLLMSpec
from agilerl.arena.models.descriptions import (
    ADVANTAGE_GRANULARITY,
    CLIP_COEF,
    GROUP_SIZE,
    IS_LEVEL,
    LR,
    WHITEN_ADVANTAGES,
)
from agilerl.arena.models.profiling import ProfilingConfig
from agilerl.arena.models.registry import register


@register()
class GRPOSpec(RolloutLLMSpec):
    """Group Relative Policy Optimization."""

    group_size: int = Field(default=8, ge=1, description=GROUP_SIZE)
    lr: float = Field(default=5e-7, ge=0.0, description=LR)
    clip_coef: float = Field(default=0.2, ge=0.0, le=1.0, description=CLIP_COEF)

    adv_norm: Literal["mean_only", "mean_std"] = Field(
        default="mean_std",
        description=(
            "How group advantages are normalized. 'mean_only' subtracts the "
            "group mean; 'mean_std' also divides by its standard deviation."
        ),
    )
    use_kl_advantage_shaping: bool = Field(
        default=False,
        description=(
            "Fold the KL penalty into the advantage instead of adding it as a "
            "separate loss term."
        ),
    )
    importance_sampling_level: Literal["token", "turn", "trajectory"] | None = Field(
        default=None, description=IS_LEVEL
    )
    advantage_granularity: Literal["auto", "trajectory", "turn"] = Field(
        default="auto", description=ADVANTAGE_GRANULARITY
    )
    whiten_advantages: bool = Field(default=False, description=WHITEN_ADVANTAGES)
    adv_clip_range: float | None = Field(
        default=None,
        description="Clip advantages to +/- this value. Unset leaves them unclipped.",
    )
    filter_zero_adv: bool = Field(
        default=False,
        description=(
            "Drop groups whose completions all scored the same. They contribute "
            "no gradient, so training on them only costs compute."
        ),
    )
    adv_filter_eps: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "With filter_zero_adv, also drop groups whose largest absolute "
            "advantage is below this threshold."
        ),
    )
    offload_trainer_during_rollout: bool = Field(
        default=True,
        description=(
            "For colocated vLLM, offload the trainer's parameters while the "
            "engine generates, trading a copy for headroom."
        ),
    )
    turn_advantage_trajectory_fallback: bool = Field(
        default=True,
        description=(
            "Fall back to a trajectory-level advantage when per-turn credit "
            "cannot be assigned."
        ),
    )
    loss_norm: Literal["micro_batch", "accumulation_window", "episode"] = Field(
        default="micro_batch",
        description=(
            "Token count the loss is averaged over: each micro-batch on its "
            "own, or the whole accumulation window. The window is the unbiased "
            "choice when micro-batches have uneven lengths. 'episode' gives "
            "every episode of the window the same total policy weight, "
            "averaging tokens within an episode, so long episodes do not "
            "outweigh short ones."
        ),
    )
    old_logprobs_source: Literal["trainer", "rollout"] = Field(
        default="trainer",
        description=(
            "Old policy for the clipped ratio. 'trainer' scores the learn-start "
            "policy, skipping the extra forward for micro-batches before the "
            "first optimizer step. 'rollout' uses the inference engine's "
            "sampling log-probs and runs the extra forward only for completions "
            "missing some of them, e.g. after env truncation."
        ),
    )
    loss_type: Literal["grpo", "gspo", "cispo"] = Field(
        default="grpo",
        description=(
            "Policy-gradient loss variant. GSPO and CISPO pin this on their own specs."
        ),
    )
    profiling_config: ProfilingConfig | None = Field(
        default=None,
        description=(
            "Opt-in learn profiling: CUDA memory snapshot on OOM and a "
            "torch.profiler trace of one micro-batch. Unset profiles nothing."
        ),
    )

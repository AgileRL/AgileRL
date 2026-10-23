# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""LatentPPO algorithm specification."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field, model_validator
from typing_extensions import Self

from agilerl.arena.models.algorithms.base import AlgorithmSpec
from agilerl.arena.models.descriptions import LR
from agilerl.arena.models.hpo import RLHyperparameter
from agilerl.arena.models.registry import AgentType, register

LATENT_PPO_HPO_RANGES: dict[str, RLHyperparameter] = {
    "lr": RLHyperparameter(min=0.00001, max=0.01, grow_factor=1.25, shrink_factor=0.75),
    "batch_size": RLHyperparameter(min=2, max=512, grow_factor=1.5, shrink_factor=0.75),
    "clip_coef": RLHyperparameter(min=0.1, max=0.4, grow_factor=1.1, shrink_factor=0.9),
    "lambda_seg": RLHyperparameter(
        min=0.1, max=5.0, grow_factor=1.2, shrink_factor=0.8
    ),
    "lambda_policy": RLHyperparameter(
        min=0.1, max=5.0, grow_factor=1.2, shrink_factor=0.8
    ),
    "beta_kl": RLHyperparameter(min=0.001, max=0.1, grow_factor=1.1, shrink_factor=0.9),
}


@register()
class LatentPPOSpec(AlgorithmSpec):
    """Latent-space PPO for semantic segmentation."""

    batch_size: int = Field(
        default=32,
        ge=1,
        description="Images per learn step.",
    )
    lr: float = Field(default=0.0001, ge=0.0, description=LR)
    entrypoint: str | None = Field(
        default=None, description="Module path of the model to build."
    )
    clip_coef: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="PPO surrogate clipping range.",
    )
    lambda_seg: float = Field(
        default=1.0, ge=0.0, description="Weight on the segmentation loss."
    )
    lambda_policy: float = Field(
        default=1.0, ge=0.0, description="Weight on the policy loss."
    )
    lambda_kl: float = Field(default=1.0, ge=0.0, description="Weight on the KL loss.")
    beta_kl: float = Field(default=0.01, ge=0.0, description="KL penalty coefficient.")
    advantage_min: float = Field(
        default=-1.0, description="Floor advantages are clamped to."
    )
    advantage_max: float = Field(
        default=1.0, description="Ceiling advantages are clamped to."
    )
    max_grad_norm: float = Field(
        default=0.5,
        ge=0.0,
        description="Gradients are clipped to this global norm before each step.",
    )
    ema_decay: float = Field(
        default=0.995,
        ge=0.0,
        le=1.0,
        description="Decay on the baseline's exponential moving average.",
    )
    baseline_momentum: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Momentum on the baseline update.",
    )
    reward_components: list[str] = Field(
        default_factory=lambda: ["mean_iou", "dice", "ce", "boundary"],
        description="Reward terms combined into the scalar reward.",
    )
    reward_weights: list[float] = Field(
        default_factory=lambda: [1.0, 1.0, 1.0, 1.0],
        description="Weight per reward component, in order.",
    )

    agent_type: ClassVar[AgentType] = AgentType.LatentPPOAgent
    env_type: ClassVar[str] = "non_tabular"
    default_evo_steps: ClassVar[int] = 8
    hpo_ranges: ClassVar[dict[str, RLHyperparameter]] = LATENT_PPO_HPO_RANGES

    @model_validator(mode="after")
    def _check_reward_weights(self) -> Self:
        if len(self.reward_weights) != len(self.reward_components):
            msg = (
                f"reward_weights ({len(self.reward_weights)}) must match "
                f"reward_components ({len(self.reward_components)})."
            )
            raise ValueError(msg)
        return self

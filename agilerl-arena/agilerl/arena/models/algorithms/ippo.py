# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""IPPO algorithm specification."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from agilerl.arena.models.algorithms.base import (
    ON_POLICY_HPO_RANGES,
    MultiAgentAlgorithmSpec,
)
from agilerl.arena.models.descriptions import (
    ACTION_STD_INIT,
    CLIP_COEF,
    ENT_COEF,
    GAE_LAMBDA,
    LR,
    MAX_GRAD_NORM,
    NET_CONFIG_PER_AGENT,
    TARGET_KL,
    UPDATE_EPOCHS,
    VF_COEF,
)
from agilerl.arena.models.networks import StochasticActorSpec
from agilerl.arena.models.registry import register


@register()
class IPPOSpec(MultiAgentAlgorithmSpec):
    """Independent PPO."""

    learn_step: int = Field(
        default=2048,
        ge=1,
        description="Environment steps collected per rollout, before each update.",
    )
    gae_lambda: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description=GAE_LAMBDA,
    )
    action_std_init: float = Field(default=0.0, description=ACTION_STD_INIT)
    clip_coef: float = Field(default=0.2, ge=0.0, le=1.0, description=CLIP_COEF)
    ent_coef: float = Field(default=0.01, ge=0.0, le=1.0, description=ENT_COEF)
    vf_coef: float = Field(default=0.5, ge=0.0, le=1.0, description=VF_COEF)
    max_grad_norm: float = Field(default=0.5, description=MAX_GRAD_NORM)
    target_kl: float | None = Field(default=None, description=TARGET_KL)
    update_epochs: int = Field(default=4, ge=1, description=UPDATE_EPOCHS)
    action_batch_size: int | None = Field(
        default=None,
        description=(
            "Agents whose actions are computed in one batched forward pass. "
            "Unset does them all at once."
        ),
    )
    lr: float = Field(default=0.0001, ge=0.0, description=LR)
    net_config: StochasticActorSpec | dict[str, StochasticActorSpec] | None = Field(
        default=None, description=NET_CONFIG_PER_AGENT
    )

    hpo_ranges: ClassVar = ON_POLICY_HPO_RANGES

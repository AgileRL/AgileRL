# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""PPO algorithm specification."""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import Field

from agilerl.arena.models.algorithms.base import (
    ON_POLICY_HPO_RANGES,
    SingleAgentAlgorithmSpec,
)
from agilerl.arena.models.descriptions import (
    ACTION_STD_INIT,
    CLIP_COEF,
    ENT_COEF,
    GAE_LAMBDA,
    LR,
    MAX_GRAD_NORM,
    NET_CONFIG,
    SHARE_ENCODERS,
    TARGET_KL,
    UPDATE_EPOCHS,
    VF_COEF,
)
from agilerl.arena.models.hpo import RLHyperparameter
from agilerl.arena.models.networks import StochasticActorSpec
from agilerl.arena.models.registry import register


@register("Recurrent PPO")
@register("RecurrentPPO")
@register()
class PPOSpec(SingleAgentAlgorithmSpec):
    """Proximal Policy Optimization."""

    num_envs: int = Field(
        default=1,
        ge=1,
        description=(
            "Environments stepped in parallel. Resolved from environment.num_envs."
        ),
    )
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
    action_std_init: float = Field(default=0.0, ge=0.0, description=ACTION_STD_INIT)
    clip_coef: float = Field(default=0.2, ge=0.0, le=1.0, description=CLIP_COEF)
    ent_coef: float = Field(default=0.01, ge=0.0, le=1.0, description=ENT_COEF)
    vf_coef: float = Field(default=0.5, ge=0.0, le=1.0, description=VF_COEF)
    max_grad_norm: float = Field(default=0.5, ge=0.0, description=MAX_GRAD_NORM)
    target_kl: float | None = Field(default=None, ge=0.0, description=TARGET_KL)
    update_epochs: int = Field(default=4, ge=1, description=UPDATE_EPOCHS)
    rollout_buffer_config: dict[str, Any] | None = Field(
        default=None,
        description="Overrides for the on-policy rollout buffer.",
    )
    recurrent: bool = Field(
        default=False,
        description=(
            "Use an LSTM encoder and train through time. Cannot be combined "
            "with a SimBa encoder."
        ),
    )
    max_seq_len: int | None = Field(
        default=None,
        ge=1,
        description="Sequence length backpropagated through when recurrent.",
    )
    share_encoders: bool = Field(default=True, description=SHARE_ENCODERS)
    bptt_sequence_type: Literal["chunked", "maximum", "fifty_percent_overlap"] = Field(
        default="chunked",
        description=(
            "How rollouts are cut into training sequences for "
            "backpropagation through time."
        ),
    )
    lr: float = Field(default=0.0001, ge=0.0, description=LR)
    net_config: StochasticActorSpec | None = Field(default=None, description=NET_CONFIG)

    alias_implies: ClassVar[dict[str, dict[str, Any]]] = {
        "Recurrent PPO": {"recurrent": True},
        "RecurrentPPO": {"recurrent": True},
    }
    hpo_ranges: ClassVar[dict[str, RLHyperparameter]] = ON_POLICY_HPO_RANGES

    @property
    def name(self) -> str:
        """``Recurrent PPO`` when a recurrent encoder is requested, else ``PPO``."""
        prefix = "Recurrent " if self.recurrent else ""
        return f"{prefix}{self.__class__.__name__.removesuffix('Spec')}"

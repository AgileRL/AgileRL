"""DDPG algorithm specification."""

from __future__ import annotations

from agilerl.arena.models.algo import RLAlgorithmSpec, register
from agilerl.arena.models.networks import DeterministicActorSpec
from pydantic import Field


@register()
class DDPGSpec(RLAlgorithmSpec):
    """Specification for DDPG algorithm."""

    vect_noise_dim: int = Field(default=1, ge=1)
    lr_actor: float = Field(default=0.0001, ge=0.0)
    lr_critic: float = Field(default=0.001, ge=0.0)
    tau: float = Field(default=0.001, ge=0.0, le=1.0)
    policy_freq: int = Field(default=2, ge=1)
    O_U_noise: bool = Field(default=True)
    expl_noise: float = Field(default=0.1)
    mean_noise: float = Field(default=0.0)
    theta: float = Field(default=0.15)
    dt: float = Field(default=0.01)
    share_encoders: bool = Field(default=False)
    net_config: DeterministicActorSpec | None = Field(default=None)

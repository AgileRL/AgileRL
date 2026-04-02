"""IPPO algorithm specification."""

from collections.abc import Callable
from typing import Any, ClassVar

from pydantic import Field

from agilerl.algorithms import IPPO
from agilerl.models.algo import MultiAgentRLAlgorithmSpec, register
from agilerl.models.networks import StochasticActorSpec
from agilerl.training.train_multi_agent_on_policy import train_multi_agent_on_policy


@register(arena=True)
class IPPOSpec(MultiAgentRLAlgorithmSpec):
    """Specification for IPPO algorithm."""

    learn_step: int = Field(default=2048, ge=1)
    gae_lambda: float = Field(default=0.95, ge=0.0, le=1.0)
    action_std_init: float = Field(default=0.0)
    clip_coef: float = Field(default=0.2, ge=0.0, le=1.0)
    ent_coef: float = Field(default=0.01, ge=0.0, le=1.0)
    vf_coef: float = Field(default=0.5, ge=0.0, le=1.0)
    max_grad_norm: float = Field(default=0.5)
    target_kl: float | None = Field(default=None)
    update_epochs: int = Field(default=4, ge=1)
    action_batch_size: int | None = Field(default=None)
    lr: float = Field(default=0.0001, ge=0.0)
    torch_compiler: str | None = Field(default=None)
    net_config: StochasticActorSpec | None = Field(default=None)

    algo_class: ClassVar[type[IPPO]] = IPPO

    @staticmethod
    def get_training_fn() -> Callable[..., Any]:
        """Get the training function for IPPO.

        :return: Training function
        :rtype: Callable[..., Any]
        """
        return train_multi_agent_on_policy

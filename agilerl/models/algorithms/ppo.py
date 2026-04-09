"""PPO algorithm specification."""

from collections.abc import Callable
from typing import Annotated, Any, ClassVar

from pydantic import BeforeValidator, Field

from agilerl.algorithms import PPO
from agilerl.models.algo import RLAlgorithmSpec, register
from agilerl.models.networks import StochasticActorSpec
from agilerl.modules import EvolvableModule
from agilerl.training.train_on_policy import train_on_policy
from agilerl.typing import BPTTSequenceType


def _coerce_bptt_sequence_type(v: Any) -> BPTTSequenceType:
    if isinstance(v, BPTTSequenceType):
        return v
    if isinstance(v, str):
        return BPTTSequenceType(v)
    return v


BPTTSequenceTypeFromManifest = Annotated[
    BPTTSequenceType, BeforeValidator(_coerce_bptt_sequence_type)
]


@register(arena=True)
class PPOSpec(RLAlgorithmSpec):
    """Specification for PPO algorithm."""

    num_envs: int = Field(default=1, ge=1)
    learn_step: int = Field(default=2048, ge=1)
    gae_lambda: float = Field(default=0.95, ge=0.0, le=1.0)
    action_std_init: float = Field(default=0.6, ge=0.0)
    clip_coef: float = Field(default=0.2, ge=0.0, le=1.0)
    ent_coef: float = Field(default=0.01, ge=0.0, le=1.0)
    vf_coef: float = Field(default=0.5, ge=0.0, le=1.0)
    max_grad_norm: float = Field(default=0.5, ge=0.0)
    target_kl: float | None = Field(default=None, ge=0.0)
    update_epochs: int = Field(default=4, ge=1)
    rollout_buffer_config: dict[str, Any] | None = Field(default_factory=dict)
    recurrent: bool = Field(default=False)
    max_seq_len: int | None = Field(default=32, ge=1)
    share_encoders: bool = Field(default=True)
    bptt_sequence_type: BPTTSequenceTypeFromManifest = Field(
        default=BPTTSequenceType.CHUNKED
    )
    lr: float = Field(default=0.0001, ge=0.0)
    net_config: StochasticActorSpec | None = Field(default=None)
    actor_network: EvolvableModule | None = Field(default=None)
    critic_network: EvolvableModule | None = Field(default=None)

    algo_class: ClassVar[type[PPO]] = PPO

    @property
    def name(self) -> str:
        """Return the name of the algorithm."""
        algo_name = self.algo_class.__name__
        prefix = "Recurrent " if self.recurrent else ""
        return f"{prefix}{algo_name}"

    @staticmethod
    def get_training_fn() -> Callable[..., Any]:
        """Get the training function for PPO.

        :return: Training function
        :rtype: Callable[..., Any]
        """
        return train_on_policy

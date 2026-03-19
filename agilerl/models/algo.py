from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypedDict

from pydantic import BaseModel, Field

from agilerl import HAS_LLM_DEPENDENCIES
from agilerl.algorithms.core.registry import HyperparameterConfig
from agilerl.typing import BPTTSequenceType

from .networks import MlpSpec, NetworkSpec

if TYPE_CHECKING or HAS_LLM_DEPENDENCIES:
    from peft import LoraConfig  # noqa: TC002


class AlgorithmSpec(BaseModel):
    """Pydantic model for `EvolvableAlgorithm` objects."""

    model_config = {"arbitrary_types_allowed": True}

    batch_size: int = Field(default=64, ge=1)
    hp_config: HyperparameterConfig | None = None


def _default_net_config() -> NetworkSpec:
    """Sensible MLP [64, 64] default for zero-config usage."""
    return NetworkSpec(
        encoder_config=MlpSpec(hidden_size=[64, 64]),
        head_config=MlpSpec(hidden_size=[64, 64]),
    )


class RLAlgorithmSpec(AlgorithmSpec):
    """Pydantic model for `RLAlgorithm` and `MultiAgentRLAlgorithm` objects."""

    net_config: NetworkSpec = Field(default_factory=_default_net_config)
    learn_step: int = Field(default=5, ge=1)
    gamma: float = Field(default=0.99, ge=0.0, le=1.0)


class LoraConfigDict(TypedDict):
    """Configuration for LoRA (Low-Rank Adaptation) fine-tuning.

    This dictionary defines the parameters for LoRA fine-tuning, which is a technique
    for adapting pre-trained language models to specific tasks by adding trainable
    low-rank matrices to the model.
    """

    lora_r: int
    lora_alpha: int
    lora_dropout: float
    task_type: str


class LLMAlgorithmSpec(AlgorithmSpec):
    """Pydantic model for `LLMAlgorithm` objects.

    Extends AlgorithmSpec with LLM-specific fields including LoRA configuration,
    model parameters, and training hyperparameters.
    """

    beta: float = Field(default=0.001, ge=0.0, le=1.0)
    max_grad_norm: float = Field(default=0.1, ge=0.0)
    update_epochs: int = Field(..., ge=1)
    reduce_memory_peak: bool = Field(default=False)
    lora_config: LoraConfig
    max_model_len: int
    use_separate_reference_adapter: bool
    pretrained_model_name_or_path: str
    calc_position_embeddings: bool


class DDPGSpec(RLAlgorithmSpec):
    """Pydantic model for DDPG algorithm specification."""

    lr_actor: float = Field(default=0.0001)
    lr_critic: float = Field(default=0.001)
    tau: float = Field(default=0.001)
    policy_freq: int = Field(default=2)
    O_U_noise: bool = Field(default=True)
    expl_noise: float = Field(default=0.1)
    mean_noise: float = Field(default=0.0)
    theta: float = Field(default=0.15)
    dt: float = Field(default=0.01)
    share_encoders: bool = Field(default=False)


class TD3Spec(RLAlgorithmSpec):
    """Pydantic model for TD3 algorithm specification."""

    lr_actor: float = Field(default=0.0001)
    lr_critic: float = Field(default=0.001)
    tau: float = Field(default=0.005)
    policy_freq: int = Field(default=2)
    O_U_noise: bool = Field(default=True)
    expl_noise: float = Field(default=0.1)
    mean_noise: float = Field(default=0.0)
    theta: float = Field(default=0.15)
    dt: float = Field(default=0.01)
    share_encoders: bool = Field(default=False)


class DQNSpec(RLAlgorithmSpec):
    """Pydantic model for DQN algorithm specification."""

    tau: float = Field(default=0.001)
    double: bool = Field(default=False)
    lr: float = Field(default=0.0001, ge=0.0)


class RainbowDQNSpec(RLAlgorithmSpec):
    """Pydantic model for Rainbow DQN algorithm specification."""

    tau: float = Field(default=0.001)
    beta: float = Field(default=0.4)
    prior_eps: float = Field(default=1e-6)
    num_atoms: int = Field(default=51, ge=1)
    v_min: float = Field(default=-200)
    v_max: float = Field(default=200)
    noise_std: float = Field(default=0.5)
    n_step: int = Field(default=3, ge=1)
    combined_reward: bool = Field(default=False)
    lr: float = Field(default=0.0001, ge=0.0)


class PPOSpec(RLAlgorithmSpec):
    """Pydantic model for PPO algorithm specification."""

    learn_step: int = Field(default=2048, ge=1)
    num_envs: int = Field(default=1, ge=1)
    gae_lambda: float = Field(default=0.95, ge=0.0, le=1.0)
    action_std_init: float = Field(default=0.6, ge=0.0)
    clip_coef: float = Field(default=0.2, ge=0.0, le=1.0)
    ent_coef: float = Field(default=0.01, ge=0.0, le=1.0)
    vf_coef: float = Field(default=0.5, ge=0.0, le=1.0)
    max_grad_norm: float = Field(default=0.5, ge=0.0)
    target_kl: float | None = Field(default=None, ge=0.0)
    update_epochs: int = Field(default=4, ge=1)
    use_rollout_buffer: bool = Field(default=True)
    rollout_buffer_config: dict[str, Any] | None = Field(default_factory=dict)
    recurrent: bool = Field(default=False)
    max_seq_len: int | None = Field(default=32, ge=1)
    share_encoders: bool = Field(default=True)
    bptt_sequence_type: BPTTSequenceType = Field(default=BPTTSequenceType.CHUNKED)
    lr: float = Field(default=0.0001, ge=0.0)


class MADDPGSpec(RLAlgorithmSpec):
    """Pydantic model for MADDPG algorithm specification."""

    vect_noise_dim: int
    lr_actor: float = Field(default=0.001, ge=0.0)
    lr_critic: float = Field(default=0.01, ge=0.0)
    tau: float = Field(default=0.01)
    O_U_noise: bool = Field(default=True)
    expl_noise: float = Field(default=0.1)
    mean_noise: float = Field(default=0.0)
    theta: float = Field(default=0.15)
    dt: float = Field(default=0.01)
    torch_compiler: str | None = Field(default=None)


class MATD3Spec(RLAlgorithmSpec):
    """Pydantic model for MATD3 algorithm specification."""

    vect_noise_dim: int
    lr_actor: float = Field(default=0.001, ge=0.0)
    lr_critic: float = Field(default=0.01, ge=0.0)
    tau: float = Field(default=0.015, ge=0.0, le=1.0)
    O_U_noise: bool = Field(default=True)
    expl_noise: float = Field(default=0.1)
    policy_freq: int = Field(default=2, ge=1)
    mean_noise: float = Field(default=0.0)
    theta: float = Field(default=0.15)
    dt: float = Field(default=0.01)
    torch_compiler: str | None = Field(default=None)


class IPPOSpec(RLAlgorithmSpec):
    """Pydantic model for IPPO algorithm specification."""

    learn_step: int = Field(default=2048, ge=1)
    gae_lambda: float = Field(default=0.95, ge=0.0, le=1.0)
    action_std_init: float = Field(default=0.0)
    clip_coef: float = Field(default=0.2, ge=0.0, le=1.0)
    ent_coef: float = Field(default=0.01, ge=0.0, le=1.0)
    vf_coef: float = Field(default=0.5, ge=0.0, le=1.0)
    max_grad_norm: float = Field(default=0.5)
    target_kl: float = Field(default=None)
    update_epochs: int = Field(default=4, ge=1)
    action_batch_size: int = Field(default=None)
    lr: float = Field(default=0.0001, ge=0.0)
    torch_compiler: str | None = Field(default=None)


@dataclass(frozen=True, slots=True)
class AlgorithmMeta:
    """Metadata for a registered algorithm.

    Used by the Trainer to look up the correct spec class, training
    function, and whether a replay buffer is required.
    """

    name: str
    spec_cls: type[RLAlgorithmSpec]
    algo_path: str
    train_fn_name: str
    requires_buffer: bool


ALGO_REGISTRY: dict[str, AlgorithmMeta] = {
    "PPO": AlgorithmMeta(
        "PPO", PPOSpec, "agilerl.algorithms.PPO", "train_on_policy", False
    ),
    "DQN": AlgorithmMeta(
        "DQN", DQNSpec, "agilerl.algorithms.DQN", "train_off_policy", True
    ),
    "DDPG": AlgorithmMeta(
        "DDPG", DDPGSpec, "agilerl.algorithms.DDPG", "train_off_policy", True
    ),
    "TD3": AlgorithmMeta(
        "TD3", TD3Spec, "agilerl.algorithms.TD3", "train_off_policy", True
    ),
    "RainbowDQN": AlgorithmMeta(
        "RainbowDQN",
        RainbowDQNSpec,
        "agilerl.algorithms.RainbowDQN",
        "train_off_policy",
        True,
    ),
    "CQN": AlgorithmMeta(
        "CQN", RLAlgorithmSpec, "agilerl.algorithms.CQN", "train_offline", True
    ),
    "NeuralUCB": AlgorithmMeta(
        "NeuralUCB",
        RLAlgorithmSpec,
        "agilerl.algorithms.NeuralUCB",
        "train_bandits",
        True,
    ),
    "NeuralTS": AlgorithmMeta(
        "NeuralTS",
        RLAlgorithmSpec,
        "agilerl.algorithms.NeuralTS",
        "train_bandits",
        True,
    ),
    "IPPO": AlgorithmMeta(
        "IPPO",
        IPPOSpec,
        "agilerl.algorithms.IPPO",
        "train_multi_agent_on_policy",
        False,
    ),
    "MADDPG": AlgorithmMeta(
        "MADDPG",
        MADDPGSpec,
        "agilerl.algorithms.MADDPG",
        "train_multi_agent_off_policy",
        True,
    ),
    "MATD3": AlgorithmMeta(
        "MATD3",
        MATD3Spec,
        "agilerl.algorithms.MATD3",
        "train_multi_agent_off_policy",
        True,
    ),
}


class GRPOSpec(LLMAlgorithmSpec):
    """Pydantic model for GRPO algorithm specification."""

    group_size: int = Field(..., ge=1)
    lr: float = Field(default=0.0001, ge=0.0)
    clip_coef: float = Field(default=0.2, ge=0.0, le=1.0)
    temperature: float
    vllm_config: dict[str, Any] = Field(
        default_factory=lambda: {
            "max_num_seqs": 16,
            "gpu_memory_utilization": 0.4,
            "tensor_parallel_size": 1,
        },
    )
    use_vllm: bool = Field(default=True)


class DPOSpec(LLMAlgorithmSpec):
    """Pydantic model for DPO algorithm specification."""

    lr: float = Field(default=0.000005)

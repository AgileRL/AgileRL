from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

from pydantic import BaseModel, Field

from agilerl import HAS_LLM_DEPENDENCIES
from agilerl.algorithms.core.registry import HyperparameterConfig
from agilerl.arena.models.networks import NetworkSpec
from agilerl.typing import BPTTSequenceType, ConfigType

if TYPE_CHECKING or HAS_LLM_DEPENDENCIES:
    from peft import LoraConfig  # noqa: TC002


class AlgorithmSpec(BaseModel):
    """Pydantic model for `EvolvableAlgorithm` objects."""

    batch_size: int = Field(default=64, ge=1)
    hp_config: HyperparameterConfig | None = None


class RLAlgorithmSpec(AlgorithmSpec):
    """Pydantic model for `RLAlgorithm` and `MultiAgentRLAlgorithm` objects."""

    net_config: NetworkSpec
    learn_step: int = Field(..., ge=1)
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


# ---------------------------------------- #
# --------- Algorithms in Arena ---------- #
# ---------------------------------------- #


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

    num_envs: int = Field(..., ge=1)
    gae_lambda: float = Field(default=0.95, ge=0.0, le=1.0)
    action_std_init: float = Field(default=0.6, ge=0.0)
    clip_coef: float = Field(default=0.2, ge=0.0, le=1.0)
    ent_coef: float = Field(default=0.01, ge=0.0, le=1.0)
    vf_coef: float = Field(default=0.5, ge=0.0, le=1.0)
    max_grad_norm: float = Field(default=0.5, ge=0.0)
    target_kl: float | None = Field(default=None, ge=0.0)
    update_epochs: int = Field(default=4, ge=1)
    use_rollout_buffer: bool = Field(default=True)
    rollout_buffer_config: ConfigType | None = Field(default_factory=dict)
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


class GRPOSpec(LLMAlgorithmSpec):
    """Pydantic model for GRPO algorithm specification."""

    group_size: int = Field(..., ge=1)
    lr: float = Field(default=0.0001, ge=0.0)
    clip_coef: float = Field(default=0.2, ge=0.0, le=1.0)
    temperature: float
    vllm_config: ConfigType = Field(
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

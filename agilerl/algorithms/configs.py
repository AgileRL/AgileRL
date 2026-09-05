# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Thematic constructor configs for AgileRL algorithms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy.typing as npt

from agilerl.typing import BPTTSequenceType, DeviceType
from agilerl.utils.algo_utils import CosineLRScheduleConfig, VLLMConfig
from agilerl.utils.constructor_kwargs import from_hparams as construct_from_hparams

if TYPE_CHECKING:
    from accelerate import Accelerator
    from peft import LoraConfig
    from transformers import BitsAndBytesConfig

    from agilerl.algorithms.core.registry import HyperparameterConfig
    from agilerl.modules.base import EvolvableModule, ModuleDict
    from agilerl.protocols import PreTrainedModelProtocol


@dataclass
class PopulationIndex:
    """Tournament index and mutation bookkeeping for one population member."""

    index: int = 0
    hp_config: HyperparameterConfig | None = None
    mut: str | None = None


@dataclass
class AlgorithmRuntime:
    """Device placement, wrapping, and compiler mode."""

    device: DeviceType = "cpu"
    accelerator: Accelerator | None = None
    wrap: bool = True
    torch_compiler: str | None = None
    name: str | None = None


@dataclass
class LLMRuntime(AlgorithmRuntime):
    """LLM constructors leave device unset so it can be auto-detected."""

    device: DeviceType | None = None


@dataclass
class OffPolicyLearnConfig:
    """Replay-batch learning hyperparameters shared by DQN-style agents."""

    lr: float = 1e-4
    batch_size: int = 64
    learn_step: int = 5
    gamma: float = 0.99
    tau: float = 1e-3


@dataclass
class QNetworkSetup:
    """Q-network construction for DQN and CQN."""

    net_config: dict[str, Any] | None = None
    actor_network: EvolvableModule | None = None
    double: bool = False
    normalize_images: bool = True
    cudagraphs: bool = False


@dataclass
class RainbowLearnConfig:
    """Distributional / n-step / PER learning for Rainbow DQN."""

    lr: float = 1e-4
    batch_size: int = 64
    learn_step: int = 5
    gamma: float = 0.99
    tau: float = 1e-3
    beta: float = 0.4
    prior_eps: float = 1e-6
    num_atoms: int = 51
    v_min: float = 0.0
    v_max: float = 200.0
    n_step: int = 3
    noise_std: float = 0.5
    combined_reward: bool = False


@dataclass
class ActorCriticLearnConfig:
    """Twin learning rates and delayed policy updates for DDPG."""

    lr_actor: float = 1e-4
    lr_critic: float = 1e-3
    batch_size: int = 64
    learn_step: int = 5
    gamma: float = 0.99
    tau: float = 1e-3
    policy_freq: int = 2


@dataclass
class TD3LearnConfig(ActorCriticLearnConfig):
    """DDPG-style learning with TD3's target-update rate."""

    tau: float = 0.005


@dataclass
class MADDPGLearnConfig:
    """Shared-critic off-policy learning for MADDPG / MATD3."""

    lr_actor: float = 0.001
    lr_critic: float = 0.01
    batch_size: int = 64
    learn_step: int = 5
    gamma: float = 0.95
    tau: float = 0.01
    policy_freq: int = 2


@dataclass
class ActorCriticNetworkSetup:
    """Actor and critic construction for continuous-control agents."""

    net_config: dict[str, Any] | None = None
    actor_network: EvolvableModule | None = None
    critic_network: EvolvableModule | None = None
    critic_networks: list[EvolvableModule] | None = None
    share_encoders: bool = False
    normalize_images: bool = True


@dataclass
class ExplorationNoise:
    """Ornstein-Uhlenbeck / Gaussian exploration for DDPG / TD3 / MADDPG."""

    O_U_noise: bool = True
    expl_noise: float | npt.NDArray = 0.1
    vect_noise_dim: int = 1
    mean_noise: float | npt.NDArray = 0.0
    theta: float = 0.15
    dt: float = 0.01


@dataclass
class PPOLearnConfig:
    """PPO clip, value, and entropy objective."""

    lr: float = 1e-4
    batch_size: int = 64
    learn_step: int = 2048
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None
    update_epochs: int = 4


@dataclass
class PPONetworkSetup:
    """Stochastic actor and value network construction for PPO."""

    net_config: dict[str, Any] | None = None
    actor_network: EvolvableModule | None = None
    critic_network: EvolvableModule | None = None
    share_encoders: bool = True
    action_std_init: float = 0.0
    normalize_images: bool = True


@dataclass
class IPPONetworkSetup:
    """Per-agent stochastic actor and value networks for IPPO."""

    net_config: dict[str, Any] | None = None
    actor_networks: list[EvolvableModule] | ModuleDict | None = None
    critic_networks: list[EvolvableModule] | ModuleDict | None = None
    action_std_init: float = 0.0
    action_batch_size: int | None = None


@dataclass
class MultiAgentActorCriticSetup:
    """Per-agent actor and centralized critic construction for MADDPG / MATD3."""

    net_config: dict[str, Any] | None = None
    actor_networks: list[EvolvableModule] | ModuleDict | None = None
    critic_networks: (
        list[EvolvableModule]
        | ModuleDict
        | list[list[EvolvableModule] | ModuleDict]
        | None
    ) = None


@dataclass
class PPORolloutConfig:
    """Vectorized rollout buffer and recurrent BPTT settings."""

    num_envs: int = 1
    rollout_buffer_config: dict[str, Any] | None = None
    recurrent: bool = False
    bptt_sequence_type: str | BPTTSequenceType = BPTTSequenceType.CHUNKED
    max_seq_len: int | None = None


@dataclass
class MultiAgentEnvConfig:
    """PettingZoo agent IDs and missing-observation placeholder."""

    agent_ids: list[str] | None = None
    placeholder_value: float | None = -1
    normalize_images: bool = True


@dataclass
class IPPOAgentSetup(MultiAgentEnvConfig):
    """IPPO leaves missing observations unset rather than filling a placeholder."""

    placeholder_value: float | None = None


@dataclass
class BanditLearnConfig:
    """Neural UCB learning hyperparameters."""

    lr: float = 1e-3
    batch_size: int = 64
    learn_step: int = 2
    gamma: float = 1.0
    lamb: float = 1.0
    reg: float = 0.000625


@dataclass
class NeuralTSLearnConfig(BanditLearnConfig):
    """Thompson sampling uses a higher default learning rate than UCB."""

    lr: float = 3e-3


@dataclass
class BanditNetworkSetup:
    """Context encoder for neural bandits."""

    net_config: dict[str, Any] | None = None
    actor_network: EvolvableModule | None = None
    normalize_images: bool = True


@dataclass
class LLMModelSetup:
    """Base model, tokenizer pad tokens, and LoRA attachment."""

    pad_token_id: int
    pad_token: str
    model_name: str | None = None
    actor_network: PreTrainedModelProtocol | None = None
    model_config: dict[str, Any] | None = None
    lora_config: LoraConfig | None = None
    lora_target_scope: str | None = None
    quantization_config: BitsAndBytesConfig | None = None
    calc_position_embeddings: bool = True
    seed: int = 42
    use_separate_reference_adapter: bool = False


@dataclass
class LLMTrainSetup:
    """Optimizer, batching, and memory settings for LLM finetuning."""

    batch_size: int = 16
    lr: float = 5e-7
    lr_critic: float | None = None
    max_grad_norm: float = 0.1
    micro_batch_size_per_gpu: int | None = None
    mini_batch_size: int | None = None
    cosine_lr_schedule_config: CosineLRScheduleConfig | None = None
    use_liger_loss: bool = False
    use_value_head: bool = False
    gradient_checkpointing: bool = True
    activation_offload: bool = False
    use_sequence_packing: bool = False
    chunk_rows: int | None = None
    cast_logprobs_to_fp32: bool = True
    clone: bool = False


@dataclass
class LLMVLLMSetup:
    """Colocated vLLM rollout and importance-sampling correction."""

    use_vllm: bool = False
    vllm_config: VLLMConfig | None = None
    use_memory_efficient_params: bool = True
    vllm_importance_sampling_correction: bool = True
    vllm_importance_sampling_cap: float = 2.0


@dataclass
class LLMGenerationSetup:
    """Sampling limits used by HuggingFace generate and vLLM."""

    temperature: float = 0.9
    repetition_penalty: float = 1.0
    top_p: float = 0.95
    top_k: int = 50
    min_p: float = 0.0
    max_output_tokens: int | None = None
    min_output_tokens: int | None = None
    max_model_len: int | None = 1024
    hf_generate_chunk_size: int | None = None


@dataclass
class LLMSetup:
    """Grouped LLM model, train, serving, generation, and device settings."""

    model: LLMModelSetup
    train: LLMTrainSetup = field(default_factory=LLMTrainSetup)
    vllm: LLMVLLMSetup = field(default_factory=LLMVLLMSetup)
    generation: LLMGenerationSetup = field(default_factory=LLMGenerationSetup)
    runtime: LLMRuntime = field(default_factory=LLMRuntime)


@dataclass
class GRPOModelSetup(LLMModelSetup):
    """GRPO attaches a separate reference adapter by default."""

    use_separate_reference_adapter: bool = True


@dataclass
class GRPOSetup(LLMSetup):
    """LLM setup for GRPO / GSPO / CISPO, with GRPO adapter defaults."""

    model: GRPOModelSetup


@dataclass
class GRPOObjective:
    """GRPO / GSPO / CISPO group-relative objective."""

    beta: float = 0.001
    clip_coef: float | tuple[float, float] = 0.2
    update_epochs: int = 1
    group_size: int = 8
    loss_type: Literal["grpo", "gspo", "cispo"] = "grpo"
    importance_sampling_level: Literal["token", "turn", "trajectory"] | None = None
    advantage_granularity: Literal["auto", "trajectory", "turn"] = "auto"
    action_granularity: Literal["auto", "trajectory", "turn"] | None = None
    use_kl_advantage_shaping: bool = False
    adv_norm: str = "mean_std"
    whiten_advantages: bool = False
    adv_clip_range: float | None = None
    filter_zero_adv: bool = False
    adv_filter_eps: float = 0.0
    turn_advantage_trajectory_fallback: bool = True
    loss_norm: Literal["micro_batch", "accumulation_window"] = "micro_batch"


@dataclass
class SeparateReferenceModelSetup(LLMModelSetup):
    """Keep a dedicated reference LoRA adapter (PPO_LLM / REINFORCE / DPO)."""

    use_separate_reference_adapter: bool = True


@dataclass
class PPOLLMTrainSetup(LLMTrainSetup):
    """PPO_LLM uses a larger grad clip, a critic LR, and a value head."""

    max_grad_norm: float = 1.0
    lr_critic: float | None = 5e-5
    use_value_head: bool = True


@dataclass
class PPOLLMGenerationSetup(LLMGenerationSetup):
    """PPO_LLM samples with temperature 1 and no nucleus cutoff."""

    temperature: float = 1.0
    top_p: float = 1.0


@dataclass
class PPOLLMSetup(LLMSetup):
    """LLM setup for PPO_LLM."""

    model: SeparateReferenceModelSetup
    train: PPOLLMTrainSetup = field(default_factory=PPOLLMTrainSetup)
    generation: PPOLLMGenerationSetup = field(default_factory=PPOLLMGenerationSetup)


@dataclass
class PPOLLMObjective:
    """PPO clip, value, and turn-level advantage settings for LLM PPO."""

    beta: float = 0.01
    vf_coef: float = 0.5
    clip_coef: float = 0.2
    gamma: float = 1.0
    gae_lambda: float = 1.0
    update_epochs: int = 1
    turn_level_clip: bool = True
    importance_sampling_level: Literal["auto", "token", "turn", "trajectory"] = "auto"
    advantage_granularity: Literal["turn", "token", "auto"] = "auto"
    turn_ratio_pooling: Literal["sum", "mean"] = "sum"
    action_granularity: Literal["turn", "token", "auto"] | None = None
    turn_value_reduction: Literal["mean", "final_value"] = "final_value"
    whiten_advantages: bool = True


@dataclass
class REINFORCELLMTrainSetup(LLMTrainSetup):
    """REINFORCE_LLM uses a larger grad clip than GRPO."""

    max_grad_norm: float = 1.0


@dataclass
class REINFORCELLMSetup(LLMSetup):
    """LLM setup for REINFORCE_LLM."""

    model: SeparateReferenceModelSetup
    train: REINFORCELLMTrainSetup = field(default_factory=REINFORCELLMTrainSetup)
    generation: PPOLLMGenerationSetup = field(default_factory=PPOLLMGenerationSetup)


@dataclass
class REINFORCELLMObjective:
    """REINFORCE baseline and importance-sampling settings."""

    beta: float = 0.01
    clip_coef: float = 0.2
    gamma: float = 1.0
    update_epochs: int = 1
    advantage_granularity: Literal["turn", "token", "auto"] = "auto"
    action_granularity: Literal["turn", "token", "auto"] | None = None
    importance_sampling_level: Literal["token", "turn", "trajectory"] = "token"
    turn_ratio_pooling: Literal["sum", "mean"] = "sum"


@dataclass
class DPOTrainSetup(LLMTrainSetup):
    """DPO's default learning rate is 5e-6."""

    lr: float = 5e-6


@dataclass
class DPOSetup(LLMSetup):
    """LLM setup for DPO."""

    model: SeparateReferenceModelSetup
    train: DPOTrainSetup = field(default_factory=DPOTrainSetup)


@dataclass
class DPOObjective:
    """DPO preference objective."""

    beta: float = 0.1
    nll_alpha: float = 1.0
    update_epochs: int = 1


@dataclass
class SFTTrainSetup(LLMTrainSetup):
    """SFT's default learning rate is 5e-5."""

    lr: float = 5e-5


@dataclass
class SFTSetup(LLMSetup):
    """LLM setup for SFT."""

    train: SFTTrainSetup = field(default_factory=SFTTrainSetup)


@dataclass
class SFTObjective:
    """SFT epoch count."""

    update_epochs: int = 1


__all__ = [
    "ActorCriticLearnConfig",
    "ActorCriticNetworkSetup",
    "AlgorithmRuntime",
    "BanditLearnConfig",
    "BanditNetworkSetup",
    "DPOObjective",
    "DPOSetup",
    "DPOTrainSetup",
    "ExplorationNoise",
    "GRPOModelSetup",
    "GRPOObjective",
    "GRPOSetup",
    "IPPOAgentSetup",
    "IPPONetworkSetup",
    "LLMGenerationSetup",
    "LLMModelSetup",
    "LLMRuntime",
    "LLMSetup",
    "LLMTrainSetup",
    "LLMVLLMSetup",
    "MADDPGLearnConfig",
    "MultiAgentActorCriticSetup",
    "MultiAgentEnvConfig",
    "NeuralTSLearnConfig",
    "OffPolicyLearnConfig",
    "PPOLLMGenerationSetup",
    "PPOLLMObjective",
    "PPOLLMSetup",
    "PPOLLMTrainSetup",
    "PPOLearnConfig",
    "PPONetworkSetup",
    "PPORolloutConfig",
    "PopulationIndex",
    "QNetworkSetup",
    "REINFORCELLMObjective",
    "REINFORCELLMSetup",
    "REINFORCELLMTrainSetup",
    "RainbowLearnConfig",
    "SFTObjective",
    "SFTSetup",
    "SFTTrainSetup",
    "SeparateReferenceModelSetup",
    "TD3LearnConfig",
    "construct_from_hparams",
]

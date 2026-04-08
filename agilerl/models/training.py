from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import AliasChoices, BaseModel, Field

from agilerl.components.replay_buffer import (
    MultiAgentReplayBuffer,
    MultiStepReplayBuffer,
    PrioritizedReplayBuffer,
    ReplayBuffer,
)
from agilerl.models.algo import (
    LLMAlgorithmSpec,
    MultiAgentRLAlgorithmSpec,
    RLAlgorithmSpec,
)
from agilerl.models.algorithms import RainbowDQNSpec
from agilerl.protocols import AgentType

BufferT = (
    ReplayBuffer
    | MultiStepReplayBuffer
    | PrioritizedReplayBuffer
    | MultiAgentReplayBuffer
)
AlgoSpecT = RLAlgorithmSpec | MultiAgentRLAlgorithmSpec | LLMAlgorithmSpec

if TYPE_CHECKING:
    import torch


class NStepBufferArgs(BaseModel):
    """Arguments for the n-step replay buffer."""

    n_step: int = Field(default=3, ge=1)


class PerBufferArgs(BaseModel):
    """Arguments for the prioritized experience replay buffer."""

    alpha: float = Field(default=0.5, ge=0.0, le=1.0)


class ReplayBufferSpec(BaseModel):
    """Pydantic model for AgileRL replay buffers.

    :param memory_size: The memory size of the replay buffer. Defaults to 100,000.
    :type memory_size: int
    :param standard_buffer: Whether to use the standard replay buffer. Defaults to True.
    :type standard_buffer: bool
    :param n_step_buffer: Whether to use the n-step replay buffer. Defaults to False.
    :type n_step_buffer: bool
    :param n_step_buffer_args: The arguments for the n-step replay buffer. Defaults to NStepBufferArgs.
    :type n_step_buffer_args: NStepBufferArgs
    :param combined_buffers: Whether to use combined buffers. Defaults to False.
    :type combined_buffers: bool
    :param per_buffer: Whether to use the prioritized experience replay buffer. Defaults to False.
    :type per_buffer: bool
    :param per_buffer_args: The arguments for the prioritized experience replay buffer. Defaults to PerBufferArgs.
    :type per_buffer_args: PerBufferArgs
    :param n_step: The number of steps to use for the n-step replay buffer. Defaults to None.
    :type n_step: int | None
    """

    max_size: int = Field(
        default=100_000, ge=1, validation_alias=AliasChoices("max_size", "memory_size")
    )
    standard_buffer: bool = Field(default=True)
    combined_buffers: bool = Field(default=False)
    n_step_buffer: bool = Field(default=False)
    n_step_buffer_args: NStepBufferArgs = Field(default_factory=NStepBufferArgs)
    per_buffer: bool = Field(default=False)
    per_buffer_args: PerBufferArgs = Field(default_factory=PerBufferArgs)

    def init_buffer(
        self, algo_spec: AlgoSpecT, device: str | torch.device = "cpu"
    ) -> BufferT:
        """Initialize the replay buffer.

        :param algo_spec: Algorithm specification
        :type algo_spec: AlgoSpecT
        :param device: Device
        :type device: str | torch.device
        :return: Replay buffer
        :rtype: BufferT
        """
        buffer_args = {}
        is_multi_agent = algo_spec.agent_type == AgentType.MultiAgent
        if not is_multi_agent:
            if self.n_step_buffer:
                if not hasattr(algo_spec, "gamma"):
                    msg = "Gamma must be specified for N-step buffer"
                    raise ValueError(msg)

                n_step = self.n_step_buffer_args.n_step
                n_step_args = {"n_step": n_step, "gamma": algo_spec.gamma}
                buffer_args |= n_step_args
                buffer_class = MultiStepReplayBuffer

            elif self.per_buffer:
                if not isinstance(algo_spec, RainbowDQNSpec):
                    msg = "PER buffer is only supported for Rainbow DQN"
                    raise ValueError(msg)

                alpha = self.per_buffer_args.alpha
                per_args = {"alpha": alpha}
                buffer_args |= per_args
                buffer_class = PrioritizedReplayBuffer
            else:
                buffer_class = ReplayBuffer
        else:
            buffer_class = MultiAgentReplayBuffer

        return buffer_class(
            max_size=self.max_size,
            device=device,
            **buffer_args,
        )


class TrainingSpec(BaseModel):
    """Pydantic model for AgileRL training.

    :param max_steps: Maximum number of steps to train for
    :type max_steps: int
    :param evo_steps: Number of steps to train between evolutions.
    :type evo_steps: int
    :param eval_loop: Number of evaluation episodes
    :type eval_loop: int
    :param eval_steps: Number of steps to train for evaluation
    :type eval_steps: int | None
    :param reporting_interval: Number of steps between metrics reporting. This is only applicable to
        training on Arena. When training locally, we report metrics every `evo_steps` steps. Defaults to 1024.
    :type reporting_interval: int
    :param replay_buffer: Replay buffer configuration.
    :type replay_buffer: ReplayBufferSpec | None
    :param population_size: Population size
    :type population_size: int
    :param hpo: Whether to use evolutionary hyperparameter optimisation during training.
    :type hpo: bool
    :param experience_sharing: Whether to share experiences between individuals in a population.
    :type experience_sharing: bool
    :param learning_delay: Number of steps before starting learning.
    :type learning_delay: int
    :param eps_start: Probability of taking a random action at the start of training.
    :type eps_start: float | None
    :param eps_end: Probability of taking a random action at the end of training.
    :type eps_end: float | None
    :param eps_decay: Rate of decay of the exploration probability.
    :type eps_decay: float | None
    :param target_score: Target score for early stopping.
    :type target_score: float | None
    :param checkpoint_steps: The number of steps between checkpoints.
    :type checkpoint_steps: int | None
    :param checkpoint_path: The path to save the checkpoints.
    :type checkpoint_path: str | None
    :param overwrite_checkpoints: If ``True``, overwrite the checkpoints in the checkpoint directory.
    :type overwrite_checkpoints: bool
    """

    max_steps: int = Field(..., ge=1)
    evo_steps: int = Field(..., ge=1)
    population_size: int = Field(
        ..., ge=1, validation_alias=AliasChoices("population_size", "pop_size")
    )
    eval_steps: int | None = Field(default=None)
    eval_loop: int = Field(default=1, ge=1)
    replay_buffer: ReplayBufferSpec | None = Field(default=None)
    hpo: bool = Field(default=True)
    target_score: float | None = Field(default=None)

    # Experience sharing / learning delay only applicable for off policy algorithms
    experience_sharing: bool = Field(default=False)
    learning_delay: int = Field(default=0)

    # Off-policy exploration parameters
    eps_start: float | None = Field(default=None)
    eps_end: float | None = Field(default=None)
    eps_decay: float | None = Field(default=None)

    # Model checkpoints (only relevant for local training)
    checkpoint_steps: int | None = Field(default=None)
    checkpoint_path: str | None = Field(default=None)
    overwrite_checkpoints: bool = Field(default=False)

    # NOTE: The following are only applicable to Arena training
    reporting_interval: int = Field(default=1024, ge=1)

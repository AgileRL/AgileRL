# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import AliasChoices, BaseModel, Field, model_validator
from typing_extensions import Self

if TYPE_CHECKING:
    import torch

    from agilerl.components.replay_buffer import BufferType
    from agilerl.models.algo import AlgoSpec


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
    :param per_buffer: Whether to use the prioritized experience replay buffer. Defaults to False.
    :type per_buffer: bool
    :param per_buffer_args: The arguments for the prioritized experience replay buffer. Defaults to PerBufferArgs.
    :type per_buffer_args: PerBufferArgs
    :param n_step: The number of steps to use for the n-step replay buffer. Defaults to None.
    :type n_step: int | None
    :param name: Buffer name forwarded to the Arena runtime.
    :type name: str | None
    :param kind: Arena runtime buffer kind — ``"classic"`` (default) or ``"llm"``.
    :type kind: Literal["classic", "llm"] | None
    """

    name: str | None = Field(default=None)
    kind: Literal["classic", "llm"] | None = Field(default=None)
    max_size: int = Field(
        default=100_000, ge=1, validation_alias=AliasChoices("max_size", "memory_size")
    )
    standard_buffer: bool = Field(default=True)
    n_step_buffer: bool = Field(default=False)
    n_step_buffer_args: NStepBufferArgs = Field(default_factory=NStepBufferArgs)
    per_buffer: bool = Field(default=False)
    per_buffer_args: PerBufferArgs = Field(default_factory=PerBufferArgs)

    def init_buffer(
        self, algo_spec: AlgoSpec, device: str | torch.device = "cpu"
    ) -> BufferType:
        """Initialize the replay buffer.

        :param algo_spec: Algorithm specification
        :type algo_spec: AlgoSpec
        :param device: Device
        :type device: str | torch.device
        :return: Replay buffer
        :rtype: BufferType
        """
        # Import lazily to avoid heavy dependencies for Arena manifest validation
        from agilerl import AgentType
        from agilerl.components.replay_buffer import (
            MultiStepReplayBuffer,
            PrioritizedReplayBuffer,
            ReplayBuffer,
        )
        from agilerl.models.algorithms import RainbowDQNSpec

        buffer_args: dict[str, Any] = {}
        is_multi_agent = algo_spec.agent_type == AgentType.MultiAgent
        if not is_multi_agent:
            # PER takes precedence for the main memory: in combined
            # PER + n-step setups the n-step buffer is a secondary buffer
            # built by ``init_n_step_buffer``.
            if self.per_buffer:
                if not isinstance(algo_spec, RainbowDQNSpec):
                    msg = "PER buffer is only supported for Rainbow DQN"
                    raise ValueError(msg)

                alpha = self.per_buffer_args.alpha
                per_args = {"alpha": alpha}
                buffer_args |= per_args
                buffer_class = PrioritizedReplayBuffer

            elif self.n_step_buffer:
                if not hasattr(algo_spec, "gamma"):
                    msg = "Gamma must be specified for N-step buffer"
                    raise ValueError(msg)

                n_step = self.n_step_buffer_args.n_step
                n_step_args = {"n_step": n_step, "gamma": algo_spec.gamma}
                buffer_args |= n_step_args
                buffer_class = MultiStepReplayBuffer
            else:
                buffer_class = ReplayBuffer
        else:
            buffer_class = ReplayBuffer

        return buffer_class(
            max_size=self.max_size,
            device=device,
            **buffer_args,
        )

    def init_n_step_buffer(
        self, algo_spec: AlgoSpec, device: str | torch.device = "cpu"
    ) -> BufferType | None:
        """Initialize the n-step replay buffer for combined PER + n-step setups.

        Returns ``None`` unless both ``per_buffer`` and ``n_step_buffer`` are
        ``True``.

        :param algo_spec: Algorithm specification.
        :type algo_spec: AlgoSpec
        :param device: Device.
        :type device: str | torch.device
        :returns: A :class:`MultiStepReplayBuffer` or ``None``.
        :rtype: BufferType | None
        """
        if not (self.per_buffer and self.n_step_buffer):
            return None

        from agilerl.components.replay_buffer import MultiStepReplayBuffer

        gamma = getattr(algo_spec, "gamma", None)
        if not isinstance(gamma, (int, float)):
            msg = "Gamma must be specified for N-step buffer"
            raise ValueError(msg)

        return MultiStepReplayBuffer(
            max_size=self.max_size,
            device=device,
            n_step=self.n_step_buffer_args.n_step,
            gamma=gamma,
        )


class TrainingSpec(BaseModel):
    """Pydantic model for AgileRL training arguments.

    :param max_steps: Maximum number of steps to train for. Defaults to 1,000,000.
    :type max_steps: int
    :param evo_steps: Number of steps to train between evolutions.
    :type evo_steps: int | None
    :param pop_size: Number of agents in the population. Defaults to 1.
    :type pop_size: int
    :param eval_steps: Number of steps to train for evaluation. Defaults to None.
    :type eval_steps: int | None
    :param eval_loop: Number of evaluation episodes. Defaults to 1.
    :type eval_loop: int
    :param replay_buffer: Replay buffer specification.
    :type replay_buffer: ReplayBufferSpec | None
    :param hpo: Whether to use hyperparameter optimization.
    :type hpo: bool
    :param target_score: Target score for early stopping.
    :type target_score: float | None
    :param learning_delay: Number of steps before starting learning.
    :type learning_delay: int
    :param eps_start: Initial exploration probability.
    :type eps_start: float | None
    :param eps_end: Final exploration probability.
    :type eps_end: float | None
    :param eps_decay: Rate of decay of the exploration probability.
    :type eps_decay: float | None
    :param checkpoint_steps: The number of steps between checkpoints.
    :type checkpoint_steps: int | None
    :param checkpoint_path: The path to save the checkpoints.
    :type checkpoint_path: str | None
    :param overwrite_checkpoints: If ``True``, overwrite the checkpoints in the checkpoint directory.
    :type overwrite_checkpoints: bool
    :param evaluation_interval: Number of steps between evaluations.
    :type evaluation_interval: int
    :param num_epochs: Number of epochs to train for.
    :type num_epochs: int | None
    :param evo_epochs: Number of epochs between evolutions (Arena converts to
        ``evo_steps`` server-side alongside ``num_epochs``).
    :type evo_epochs: int | None
    :param max_wall_seconds: Wall-clock limit for multi-turn LLM fine-tuning runs.
    :type max_wall_seconds: float | None
    :param episode_steps: Number of steps to train for each episode (only applicable for bandits).
    :type episode_steps: int
    :param sum_scores: Whether to sum sub-agent scores (only applicable for multi-agent).
        Typically ``True`` for cooperative environments. Defaults to ``True``.
    :type sum_scores: bool
    :param reporting_interval: Number of steps between reporting.
    :type reporting_interval: int
    :param experience_sharing: Whether to share experiences between agents.
    :type experience_sharing: bool
    """

    max_steps: int = Field(default=1_000_000, ge=1)
    evo_steps: int | None = Field(
        default=None,
        ge=1,
        validation_alias=AliasChoices("metrics_interval", "evo_steps"),
    )
    pop_size: int = Field(
        default=1, ge=1, validation_alias=AliasChoices("population_size", "pop_size")
    )
    eval_steps: int | None = Field(default=None)
    eval_loop: int = Field(default=1, ge=1)
    replay_buffer: ReplayBufferSpec | None = Field(default=None)
    hpo: bool = Field(default=True)
    target_score: float | None = Field(default=None)

    # Learning delay / exploration parameters only applicable for off policy algorithms
    learning_delay: int = Field(default=0)
    eps_start: float | None = Field(default=None)
    eps_end: float | None = Field(default=None)
    eps_decay: float | None = Field(default=None)

    # Model checkpoints (only relevant for local training)
    checkpoint_steps: int | None = Field(default=None)
    checkpoint_path: str | None = Field(default=None)
    overwrite_checkpoints: bool = Field(default=False)

    # LLM-specific training parameters
    evaluation_interval: int = Field(default=10, ge=1)
    num_epochs: int | None = Field(default=None, ge=1)
    evo_epochs: int | None = Field(default=None, ge=0)
    max_wall_seconds: float | None = Field(default=None, gt=0)

    # Bandit-specific training parameters
    episode_steps: int = Field(default=500, ge=1)

    # Multi-agent specific training parameters
    sum_scores: bool = Field(default=True)

    # NOTE: The following are only applicable to training on Arena
    reporting_interval: int = Field(default=1024, ge=1)
    experience_sharing: bool = Field(
        default=False
    )  # when training locally, we always share experiences

    @model_validator(mode="after")
    def _validate_training_parameters(self) -> Self:
        if self.eval_steps is not None and self.eval_steps <= self.evaluation_interval:
            msg = "eval_steps must be greater than evaluation_interval"
            raise ValueError(msg)
        if self.evo_steps is not None and self.evo_steps > self.max_steps:
            msg = f"evo_steps ({self.evo_steps}) must be less than or equal to max_steps ({self.max_steps})."
            raise ValueError(msg)
        if (
            self.eps_start is not None
            and self.eps_end is not None
            and self.eps_start < self.eps_end
        ):
            msg = f"eps_start ({self.eps_start}) must be greater than or equal to eps_end ({self.eps_end})."
            raise ValueError(msg)
        return self

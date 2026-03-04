from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Status of a job."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    STOPPED = "STOPPED"


class NStepBufferArgs(BaseModel):
    """Arguments for the n-step replay buffer."""

    n_step: int = 3


class ReplayBufferSpec(BaseModel):
    """Pydantic model for replay buffer specification.

    :param name: The name of the replay buffer. Defaults to "ReplayBuffer".
    :type name: str
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
    """

    name: str = "ReplayBuffer"
    memory_size: int = Field(default=100_000, ge=1)
    standard_buffer: bool = True
    n_step_buffer: bool = False
    n_step_buffer_args: NStepBufferArgs = Field(default_factory=NStepBufferArgs)
    combined_buffers: bool = False


class TrainingSpec(BaseModel):
    """Training loop parameters section of an Arena training manifest.

    :param max_steps: The maximum number of steps to train for.
        Defaults to 1,000,000.
    :type max_steps: int
    :param pop_size: The population size. Defaults to 4.
    :type pop_size: int
    :param evo_steps: The number of evolution steps. Defaults to 10,000.
    :type evo_steps: int
    :param eval_loop: The number of evaluation loops. Defaults to 1.
    :type eval_loop: int
    :param learning_delay: The number of steps to wait before learning. Defaults to 0.
    :type learning_delay: int
    :param reporting_interval: The number of steps to report the training progress. Defaults to 4096.
    :type reporting_interval: int
    :param experience_sharing: Whether to share experience between agents. Defaults to True.
    :type experience_sharing: bool
    :param target_score: The target score to reach. Defaults to None.
    :type target_score: float | None
    """

    max_steps: int = Field(default=1_000_000, ge=1)
    pop_size: int = Field(default=4, ge=1)
    evo_steps: int = Field(default=10_000, ge=1)
    eval_loop: int = Field(default=1, ge=1)
    learning_delay: int = Field(default=0, ge=0)
    reporting_interval: int = Field(default=4096, ge=1)
    experience_sharing: bool = True
    target_score: float | None = None

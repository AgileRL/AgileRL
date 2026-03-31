from __future__ import annotations

from enum import Enum
from typing import Any

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

    n_step: int = Field(default=3, ge=1)


class ReplayBufferSpec(BaseModel):
    """Pydantic model for replay buffer specification.

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

    memory_size: int = Field(default=100_000, ge=1)
    standard_buffer: bool = True
    n_step_buffer: bool = False
    n_step_buffer_args: NStepBufferArgs = Field(default_factory=NStepBufferArgs)
    combined_buffers: bool = False

    def to_manifest(self) -> dict[str, Any]:
        """Serialize this replay buffer spec for Arena manifest payloads."""
        return {
            "name": "ReplayBuffer",
            **self.model_dump(mode="json", exclude_none=True),
        }


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
    :param target_score: The target score to reach. Defaults to None.
    :type target_score: float | None
    """

    max_steps: int = Field(default=1_000_000, ge=1)
    pop_size: int = Field(default=4, ge=1)
    evo_steps: int = Field(default=10_000, ge=1)
    eval_loop: int = Field(default=1, ge=1)
    learning_delay: int = Field(default=0, ge=0)

    target_score: float | None = None

    def to_manifest(self, *, name: str | None = None) -> dict[str, Any]:
        """Serialize this training spec for Arena manifest payloads."""
        payload: dict[str, Any] = self.model_dump(mode="json", exclude_none=True)
        if name is not None:
            payload["name"] = name
        return payload

# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Training specs (the arena contract) and the functions that build buffers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agilerl.arena.models.algorithms import (
    AlgoSpec,
    RainbowDQNSpec,
    SingleAgentAlgorithmSpec,
)
from agilerl.arena.models.registry import AgentType
from agilerl.arena.models.training import (
    BufferSpec,
    LLMRolloutBufferSpec,
    NStepBufferArgs,
    PerBufferArgs,
    ReplayBufferSpec,
    TrainingSpec,
)
from agilerl.components.replay_buffer import (
    MultiStepReplayBuffer,
    PrioritizedReplayBuffer,
    ReplayBuffer,
)

if TYPE_CHECKING:
    import torch

    from agilerl.components.replay_buffer import BufferType

__all__ = [
    "BufferSpec",
    "LLMRolloutBufferSpec",
    "NStepBufferArgs",
    "PerBufferArgs",
    "ReplayBufferSpec",
    "TrainingSpec",
    "init_buffer",
    "init_n_step_buffer",
]


def init_buffer(
    spec: ReplayBufferSpec,
    algo_spec: AlgoSpec,
    device: str | torch.device = "cpu",
) -> BufferType:
    """Initialize the replay buffer described by *spec*."""
    buffer_args: dict[str, Any] = {}
    is_multi_agent = algo_spec.agent_type == AgentType.MultiAgent
    if not is_multi_agent:
        # PER takes precedence for the main memory: in combined
        # PER + n-step setups the n-step buffer is a secondary buffer
        # built by ``init_n_step_buffer``.
        if spec.per_buffer:
            if not isinstance(algo_spec, RainbowDQNSpec):
                msg = "PER buffer is only supported for Rainbow DQN"
                raise ValueError(msg)

            buffer_args |= {"alpha": spec.per_buffer_args.alpha}
            buffer_class = PrioritizedReplayBuffer

        elif spec.n_step_buffer:
            if not isinstance(algo_spec, SingleAgentAlgorithmSpec):
                msg = "Gamma must be specified for N-step buffer"
                raise ValueError(msg)

            buffer_args |= {
                "n_step": spec.n_step_buffer_args.n_step,
                "gamma": algo_spec.gamma,
            }
            buffer_class = MultiStepReplayBuffer
        else:
            buffer_class = ReplayBuffer
    else:
        buffer_class = ReplayBuffer

    return buffer_class(
        max_size=spec.max_size,
        device=device,
        **buffer_args,
    )


def init_n_step_buffer(
    spec: ReplayBufferSpec,
    algo_spec: AlgoSpec,
    device: str | torch.device = "cpu",
) -> BufferType | None:
    """Initialize the n-step replay buffer for combined PER + n-step setups.

    Returns ``None`` unless both ``per_buffer`` and ``n_step_buffer`` are
    ``True``.
    """
    if not (spec.per_buffer and spec.n_step_buffer):
        return None

    if not isinstance(algo_spec, SingleAgentAlgorithmSpec):
        msg = "Gamma must be specified for N-step buffer"
        raise ValueError(msg)

    return MultiStepReplayBuffer(
        max_size=spec.max_size,
        device=device,
        n_step=spec.n_step_buffer_args.n_step,
        gamma=algo_spec.gamma,
    )

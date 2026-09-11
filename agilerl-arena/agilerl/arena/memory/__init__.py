# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""GPU memory estimation for LLM RL training and generation.

A first-principles model of peak GPU memory occupancy for the framework's LLM
RL stack. Everything is derived closed-form from the model geometry, the
device, and the training and generation settings. Pure python — pydantic and
nothing else — so the same calculation runs without torch or the training stack.
"""

from agilerl.arena.memory.estimator import (
    MemoryComponent,
    PhaseBreakdown,
    RunEstimate,
    estimate_generation,
    estimate_run,
    estimate_training,
    generation_can_serve,
)
from agilerl.arena.memory.specs import (
    DeviceSpec,
    GenerationSettings,
    ModelArch,
    ModelSpec,
    RunConfig,
    TrainingSettings,
    WeightVariant,
)

__all__ = [
    "DeviceSpec",
    "GenerationSettings",
    "MemoryComponent",
    "ModelArch",
    "ModelSpec",
    "PhaseBreakdown",
    "RunConfig",
    "RunEstimate",
    "TrainingSettings",
    "WeightVariant",
    "estimate_generation",
    "estimate_run",
    "estimate_training",
    "generation_can_serve",
]

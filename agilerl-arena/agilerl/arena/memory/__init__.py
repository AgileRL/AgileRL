# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""GPU memory estimation for LLM RL training and generation.

Peak occupancy from model geometry, device, and training and generation
settings. Manifest parsing lives in ``agilerl.arena.memory.manifest``.
"""

from agilerl.arena.memory.advice import Advice, advise
from agilerl.arena.memory.estimator import (
    MemoryComponent,
    PhaseBreakdown,
    RunEstimate,
    estimate_generation,
    estimate_run,
    estimate_training,
    generation_can_serve,
)
from agilerl.arena.memory.resolve.fsdp import resolve_fsdp_memory
from agilerl.arena.memory.solver import (
    SOLVABLE_FIELDS,
    CannotSolve,
    SolveResult,
    inference_run_config,
    solve,
    solve_inference,
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
    "SOLVABLE_FIELDS",
    "Advice",
    "CannotSolve",
    "DeviceSpec",
    "GenerationSettings",
    "MemoryComponent",
    "ModelArch",
    "ModelSpec",
    "PhaseBreakdown",
    "RunConfig",
    "RunEstimate",
    "SolveResult",
    "TrainingSettings",
    "WeightVariant",
    "advise",
    "estimate_generation",
    "estimate_run",
    "estimate_training",
    "generation_can_serve",
    "inference_run_config",
    "resolve_fsdp_memory",
    "solve",
    "solve_inference",
]

# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""GPU memory estimation for LLM RL training and generation.

Peak occupancy from model geometry, device, and training and generation
settings.
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
from agilerl.arena.memory.manifest import (
    GPU_CATALOGUE,
    GpuInfo,
    device_spec_from_resource_class,
    estimate_manifest,
    generation_settings_from_manifest,
    lookup_gpu,
    run_config_from_manifest,
    training_settings_from_manifest,
)
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
    "GPU_CATALOGUE",
    "SOLVABLE_FIELDS",
    "Advice",
    "CannotSolve",
    "DeviceSpec",
    "GenerationSettings",
    "GpuInfo",
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
    "device_spec_from_resource_class",
    "estimate_generation",
    "estimate_manifest",
    "estimate_run",
    "estimate_training",
    "generation_can_serve",
    "generation_settings_from_manifest",
    "inference_run_config",
    "lookup_gpu",
    "run_config_from_manifest",
    "solve",
    "solve_inference",
    "training_settings_from_manifest",
]

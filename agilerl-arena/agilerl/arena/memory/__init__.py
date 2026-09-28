# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""GPU memory estimation for LLM RL training and generation.

Peak occupancy from model geometry, device, and training and generation
settings. Pydantic only — no torch, no training stack.
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
    "Advice",
    "DeviceSpec",
    "GenerationSettings",
    "GpuInfo",
    "MemoryComponent",
    "ModelArch",
    "ModelSpec",
    "PhaseBreakdown",
    "RunConfig",
    "RunEstimate",
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
    "lookup_gpu",
    "run_config_from_manifest",
    "training_settings_from_manifest",
]

# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""GPU memory estimation for LLM RL training and generation.

A first-principles model of peak GPU memory occupancy for the framework's LLM
RL stack. Everything is derived closed-form from three inputs: the training
manifest (the same document a submission carries), the resource class the run
would be scheduled on, and the named checkpoint's own ``config.json``. Pure
python — pydantic and nothing else — so the same calculation runs in the CLI,
in a backend service, and client-side in the Arena widget, without dragging
in torch or any of the training stack.

:mod:`agilerl.arena.memory.manifest` is the front door: it turns the three
inputs into the estimator's working :class:`~agilerl.arena.memory.specs.RunConfig`,
so no caller assembles the run settings by hand.
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

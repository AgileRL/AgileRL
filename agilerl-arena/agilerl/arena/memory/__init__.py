# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""GPU memory estimation for LLM RL training and generation.

A first-principles model of peak GPU memory occupancy for the framework's LLM
RL stack. Everything is derived closed-form from three inputs: the training
manifest (the same document a submission carries), the resource class the run
would be scheduled on, and the named checkpoint's own ``config.json`` — there
is no per-model profiling step and no fitted correction. Pure python —
pydantic and nothing else — so the same calculation runs in the CLI, in a
backend service, and client-side in the Arena widget, without dragging in
torch or any of the training stack.

:mod:`agilerl.arena.memory.manifest` is the front door: it turns the three
inputs into the estimator's working :class:`~agilerl.arena.memory.specs.RunConfig`,
so no caller assembles the run settings by hand.

The measurement rig that built this model lives in the hub's
``scripts/memory_profiling`` alongside the measurements it collected; it
drives vLLM and NVML on a GPU box and is how the terms were found — not
something a caller needs.

See this package's ``README.md`` for the component model and its grounding
in the framework's actual code paths.
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
    generation_knobs_from_manifest,
    lookup_gpu,
    run_config_from_manifest,
    training_knobs_from_manifest,
)
from agilerl.arena.memory.solver import (
    SOLVABLE_KNOBS,
    CannotSolve,
    SolveResult,
    inference_run_config,
    solve,
    solve_inference,
)
from agilerl.arena.memory.specs import (
    DeviceSpec,
    GenerationKnobs,
    ModelArch,
    ModelSpec,
    RunConfig,
    TrainingKnobs,
    WeightVariant,
)

__all__ = [
    "GPU_CATALOGUE",
    "SOLVABLE_KNOBS",
    "Advice",
    "CannotSolve",
    "DeviceSpec",
    "GenerationKnobs",
    "GpuInfo",
    "MemoryComponent",
    "ModelArch",
    "ModelSpec",
    "PhaseBreakdown",
    "RunConfig",
    "RunEstimate",
    "SolveResult",
    "TrainingKnobs",
    "WeightVariant",
    "advise",
    "device_spec_from_resource_class",
    "estimate_generation",
    "estimate_manifest",
    "estimate_run",
    "estimate_training",
    "generation_can_serve",
    "generation_knobs_from_manifest",
    "inference_run_config",
    "lookup_gpu",
    "run_config_from_manifest",
    "solve",
    "solve_inference",
    "training_knobs_from_manifest",
]

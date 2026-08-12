# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""GPU memory estimation for LLM RL training and generation.

A first-principles model of peak GPU memory occupancy for the framework's LLM
RL stack. Everything is derived closed-form from the checkpoint's own
``config.json`` geometry, the run's knobs, and a small table of measured
per-device constants — there is no per-model profiling step and no fitted
correction. Pure python with no torch dependency, so the same calculation runs
in the CLI, in a backend service, and client-side in the Arena widget.

The measurement rig that built and validates this model lives outside the
package in ``tools/memory_profiling`` — it is how the terms were found, not
something a caller needs.

See ``agilerl/memory/README.md`` for the component model and its grounding in
the framework's actual code paths.
"""

from agilerl.memory.advice import Advice, advise
from agilerl.memory.estimator import (
    MemoryComponent,
    PhaseBreakdown,
    RunEstimate,
    estimate_generation,
    estimate_run,
    estimate_training,
)
from agilerl.memory.specs import (
    DeviceSpec,
    GenerationKnobs,
    ModelArch,
    ModelSpec,
    RunConfig,
    TrainingKnobs,
    WeightVariant,
)

__all__ = [
    "Advice",
    "DeviceSpec",
    "GenerationKnobs",
    "MemoryComponent",
    "ModelArch",
    "ModelSpec",
    "PhaseBreakdown",
    "RunConfig",
    "RunEstimate",
    "TrainingKnobs",
    "WeightVariant",
    "advise",
    "estimate_generation",
    "estimate_run",
    "estimate_training",
]

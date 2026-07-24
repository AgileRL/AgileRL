"""GPU memory estimation for LLM RL training and generation.

A first-principles model of peak GPU memory occupancy for the framework's
LLM RL stack: a closed-form calculation core (:mod:`~agilerl.memory.formulas`,
:mod:`~agilerl.memory.estimator`) calibrated per curated model by a small
profiled constant bundle (:mod:`~agilerl.memory.calibration`,
:mod:`~agilerl.memory.profiling`). The core is pure python with no torch
dependency so the same calculation can run client-side in the Arena widget.

See ``agilerl/memory/README.md`` for the component model and its grounding in
the framework's actual code paths.
"""

from agilerl.memory.advice import Advice, advise
from agilerl.memory.calibration import (
    ModelProfile,
    curated_models,
    load_profile,
    save_profile,
)
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
    "ModelProfile",
    "ModelSpec",
    "PhaseBreakdown",
    "RunConfig",
    "RunEstimate",
    "TrainingKnobs",
    "WeightVariant",
    "advise",
    "curated_models",
    "estimate_generation",
    "estimate_run",
    "estimate_training",
    "load_profile",
    "save_profile",
]

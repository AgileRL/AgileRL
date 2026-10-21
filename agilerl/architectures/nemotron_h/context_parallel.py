# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Context-parallel Nemotron-H Mamba2 mixers."""

from __future__ import annotations

from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHMamba2Mixer

from agilerl.architectures.runtime import ContextParallelPlan
from agilerl.distributed.context_parallel.mixer import install_mamba_context_parallel


def wrap_nemotron_h_context_parallel(model: nn.Module, cp_mesh: DeviceMesh) -> int:
    """Wrap each Nemotron-H Mamba2 mixer for context-parallel all-to-all."""
    count = 0
    for module in model.modules():
        if isinstance(module, NemotronHMamba2Mixer):
            count += int(install_mamba_context_parallel(module, cp_mesh))
    return count


NEMOTRON_H_CONTEXT_PARALLEL_PLAN = ContextParallelPlan(
    wrap=wrap_nemotron_h_context_parallel,
)

# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Context-parallel Granite 4 hybrid Mamba2 layers."""

from __future__ import annotations

from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from transformers.models.granitemoehybrid.modeling_granitemoehybrid import (
    GraniteMoeHybridMambaLayer,
)

from agilerl.architectures.runtime import ContextParallelPlan
from agilerl.distributed.context_parallel.mamba import install_mamba_context_parallel


def wrap_granitemoehybrid_context_parallel(
    model: nn.Module, cp_mesh: DeviceMesh
) -> int:
    """Wrap each Granite 4 Mamba2 layer for context-parallel all-to-all."""
    count = 0
    for module in model.modules():
        if isinstance(module, GraniteMoeHybridMambaLayer):
            # The gated RMSNorm spans every head, whatever ``n_groups`` is.
            count += int(
                install_mamba_context_parallel(
                    module, cp_mesh, int(module.intermediate_size)
                )
            )
    return count


CONTEXT_PARALLEL_PLAN = ContextParallelPlan(
    wrap=wrap_granitemoehybrid_context_parallel,
)

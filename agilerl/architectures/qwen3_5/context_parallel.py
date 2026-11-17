# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Context-parallel Qwen3.5 GatedDeltaNet mixers."""

from __future__ import annotations

from typing import cast

from torch import nn
from torch.distributed.device_mesh import DeviceMesh

from agilerl.architectures.catalog import QWEN3_5_RUNTIME_CONFIG
from agilerl.architectures.qwen3_5.gated_delta_net import mixer_classes
from agilerl.architectures.runtime import ContextParallelPlan, GdnPatchConfig
from agilerl.distributed.context_parallel.gdn import install_gdn_context_parallel


def wrap_qwen3_5_context_parallel(model: nn.Module, cp_mesh: DeviceMesh) -> int:
    """Wrap each Qwen3.5 GatedDeltaNet for context-parallel state passing."""
    gdn = cast("GdnPatchConfig", QWEN3_5_RUNTIME_CONFIG.patch.gdn)
    classes = mixer_classes(tuple(entry.mixer for entry in gdn.mixers))
    count = 0
    for module in model.modules():
        if isinstance(module, classes):
            count += int(install_gdn_context_parallel(module, cp_mesh))
    return count


QWEN3_5_CONTEXT_PARALLEL_PLAN = ContextParallelPlan(
    wrap=wrap_qwen3_5_context_parallel,
)

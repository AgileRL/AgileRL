# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Context-parallel Qwen3.5 GatedDeltaNet mixers."""

from __future__ import annotations

from torch import nn
from torch.distributed.device_mesh import DeviceMesh

from agilerl.architectures.qwen3_5.packed import PACKED_TARGETS
from agilerl.architectures.runtime import ContextParallelPlan
from agilerl.distributed.context_parallel.mixer import install_gdn_context_parallel
from agilerl.utils.patching import try_import


def _gdn_classes() -> tuple[type, ...]:
    classes: list[type] = []
    for mixer_path, _block in PACKED_TARGETS:
        module_path, _, class_name = mixer_path.rpartition(".")
        module = try_import(module_path)
        if module is None:
            continue
        mixer_cls = getattr(module, class_name, None)
        if mixer_cls is not None:
            classes.append(mixer_cls)
    return tuple(classes)


def wrap_qwen3_5_context_parallel(model: nn.Module, cp_mesh: DeviceMesh) -> int:
    """Wrap each Qwen3.5 GatedDeltaNet for context-parallel gather/shard."""
    classes = _gdn_classes()
    if not classes:
        return 0
    count = 0
    for module in model.modules():
        if isinstance(module, classes):
            count += int(install_gdn_context_parallel(module, cp_mesh))
    return count


QWEN3_5_CONTEXT_PARALLEL_PLAN = ContextParallelPlan(
    wrap=wrap_qwen3_5_context_parallel,
)

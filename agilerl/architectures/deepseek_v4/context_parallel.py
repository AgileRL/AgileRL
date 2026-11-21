# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Context-parallel DeepSeek-V4 attention.

While a sequence shard trains, each attention layer receives the
context-parallel group and builds its keys over the whole sequence; see
:mod:`agilerl.architectures.deepseek_v4`. Rollout and generation run the
unsharded forward.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from torch import nn
from torch.distributed.device_mesh import DeviceMesh

from agilerl.architectures.deepseek_v4 import CONTEXT_PARALLEL_GROUP
from agilerl.architectures.runtime import ContextParallelPlan
from agilerl.distributed.context_parallel.ulysses import ULYSSES_PARAMS


def install_deepseek_v4_context_parallel(
    attention: nn.Module, cp_mesh: DeviceMesh
) -> bool:
    """Pass the context-parallel group to ``attention`` while a sequence shard trains.

    :param attention: ``DeepseekV4Attention`` instance.
    :param cp_mesh: Context-parallel mesh.
    :return: Whether the forward was wrapped; False when already wrapped.
    """
    if getattr(attention, "_agilerl_deepseek_v4_cp", False):
        return False
    group = cp_mesh.get_group()
    forward = cast("Callable[..., object]", attention.forward)

    def cp_forward(*args: object, **kwargs: object) -> object:
        # Published by ContextParallelSequenceForward and its checkpoint recompute.
        if "cu_seqlens" in ULYSSES_PARAMS:
            kwargs[CONTEXT_PARALLEL_GROUP] = group
        return forward(*args, **kwargs)

    object.__setattr__(attention, "forward", cp_forward)
    object.__setattr__(attention, "_agilerl_deepseek_v4_cp", True)
    return True


def wrap_deepseek_v4_context_parallel(model: nn.Module, cp_mesh: DeviceMesh) -> int:
    """Wrap each DeepSeek-V4 attention layer for context-parallel keys."""
    count = 0
    for module in model.modules():
        # Match by name: Liger can reassign modeling classes after instances exist.
        if type(module).__name__ == "DeepseekV4Attention":
            count += int(install_deepseek_v4_context_parallel(module, cp_mesh))
    return count


DEEPSEEK_V4_CONTEXT_PARALLEL_PLAN = ContextParallelPlan(
    wrap=wrap_deepseek_v4_context_parallel,
)

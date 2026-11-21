# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Context-parallel GLM-5.3: Kimi Delta Attention and DSA attention on a sequence shard.

KDA conv and scan take a flash-linear-attention context, which carries their
state across shards. DSA attention layers receive one shared sequence shard
while a sequence shard trains; see :mod:`agilerl.architectures.glm5_next.patches`.
Rollout and generation run the unsharded forward.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh

from agilerl.architectures.glm5_next.patches import (
    CAUSAL_SDPA_WHEN_TOPK_COVERS,
    SEQUENCE_SHARD,
    SequenceShard,
)
from agilerl.architectures.runtime import ContextParallelPlan
from agilerl.distributed.context_parallel.gdn import install_gdn_context_parallel
from agilerl.distributed.context_parallel.ulysses import ULYSSES_PARAMS
from agilerl.utils.patching import class_is_patched

LINEAR_ATTENTION_CLASS = "Glm5NextTextLinearAttention"
ATTENTION_CLASS = "Glm5NextTextAttention"


def install_glm5_next_attention_context_parallel(
    attention: nn.Module, shard: SequenceShard
) -> bool:
    """Pass ``shard`` to a DSA ``attention`` while a sequence shard trains.

    :param attention: ``Glm5NextTextAttention`` instance.
    :param shard: Sequence shard shared by the model's DSA layers.
    :return: Whether the forward was wrapped; False when already wrapped.
    """
    if getattr(attention, "_agilerl_glm5_next_cp", False):
        return False
    # The stock forward would hand SEQUENCE_SHARD to SDPA and attend to the shard only.
    if not class_is_patched(type(attention), CAUSAL_SDPA_WHEN_TOPK_COVERS):
        msg = "GLM-5.3 context parallel needs install_glm5_next_patches"
        raise RuntimeError(msg)
    forward = cast("Callable[..., object]", attention.forward)

    def cp_forward(*args: object, **kwargs: object) -> object:
        # Published by ContextParallelSequenceForward and its checkpoint recompute.
        if "cu_seqlens" in ULYSSES_PARAMS:
            shard.begin_forward(cast("torch.Tensor", ULYSSES_PARAMS["cu_seqlens"]))
            kwargs[SEQUENCE_SHARD] = shard
        return forward(*args, **kwargs)

    object.__setattr__(attention, "forward", cp_forward)
    object.__setattr__(attention, "_agilerl_glm5_next_cp", True)
    return True


def wrap_glm5_next_context_parallel(model: nn.Module, cp_mesh: DeviceMesh) -> int:
    """Wrap each GLM-5.3 KDA and DSA attention layer for a sequence shard."""
    shard = SequenceShard(cp_mesh.get_group())
    count = 0
    for module in model.modules():
        # Match by name: Liger can reassign modeling classes after instances exist.
        name = type(module).__name__
        if name == LINEAR_ATTENTION_CLASS:
            count += int(install_gdn_context_parallel(module, cp_mesh))
        elif name == ATTENTION_CLASS:
            count += int(install_glm5_next_attention_context_parallel(module, shard))
    return count


GLM5_NEXT_CONTEXT_PARALLEL_PLAN = ContextParallelPlan(
    wrap=wrap_glm5_next_context_parallel,
)

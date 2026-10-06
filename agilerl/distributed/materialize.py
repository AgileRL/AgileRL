# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Build FSDP2 shards from a CPU snapshot or a meta safetensors load."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FSDPModule, share_comm_ctx
from torch.distributed.tensor import DTensor

from agilerl.arena.models.fsdp import FSDPConfig
from agilerl.distributed.checkpoint import _load_sharded_weights_from_safetensors
from agilerl.distributed.meta import init_rope_buffers, restore_after_to_empty
from agilerl.distributed.state import (
    canonical_fsdp_param_fqn,
    set_full_model_state_dict,
)
from agilerl.distributed.wrap import apply_fsdp2


def _restore_nonpersistent_buffers(
    model: nn.Module, buffers: dict[str, torch.Tensor]
) -> int:
    """Copy buffers that ``state_dict()`` omitted (``persistent=False``)."""
    named = dict(model.named_buffers())
    restored = 0
    with torch.no_grad():
        for key, dest in named.items():
            value = buffers.get(key)
            if value is None:
                value = buffers.get(canonical_fsdp_param_fqn(key))
            if value is None:
                continue
            dest.copy_(value.to(device=dest.device, dtype=dest.dtype))
            restored += 1
    return restored


def materialize_fsdp2_from_cpu_state(
    model: nn.Module,
    device: str | torch.device,
    config: FSDPConfig | None = None,
    mesh: DeviceMesh | None = None,
    gradient_checkpointing: bool = False,
) -> nn.Module:
    """Shard a model and fill each rank's FSDP parameters.

    A meta module loads its checkpoint from safetensors: config
    ``_name_or_path`` or ``name_or_path`` is a local directory or a Hugging
    Face Hub id, and each rank copies only its shard (the full tensor when
    world size is one). A dense module is snapshotted on CPU and those
    tensors are scattered into the shards.

    :param model: Meta module with a checkpoint path, or a dense module.
    :type model: nn.Module
    :param device: Compute device for sharded parameter storage.
    :type device: str | torch.device
    :param config: FSDP2 settings.
    :type config: FSDPConfig | None
    :param mesh: Optional FSDP device mesh.
    :type mesh: DeviceMesh | None
    :param gradient_checkpointing: Wrap each transformer block with
        non-reentrant activation checkpointing before sharding.
    :type gradient_checkpointing: bool
    :return: The sharded model (same object).
    :rtype: nn.Module
    """
    config = config or FSDPConfig()
    has_meta = any(param.is_meta for param in model.parameters())
    cpu_state: dict[str, torch.Tensor] | None = None
    cpu_buffers: dict[str, torch.Tensor] | None = None
    if not has_meta:
        cpu_state = {
            key: value.detach().to("cpu").contiguous()
            for key, value in model.state_dict().items()
        }
        cpu_buffers = {
            key: value.detach().to("cpu").contiguous()
            for key, value in model.named_buffers()
            if key not in cpu_state
        }
    model.to_empty(device="meta")
    apply_fsdp2(
        model,
        config,
        mesh=mesh,
        gradient_checkpointing=gradient_checkpointing,
    )
    target = torch.device("cpu") if config.cpu_offload else torch.device(device)
    model.to_empty(device=target)
    restore_after_to_empty(model)
    if cpu_state is None:
        _load_sharded_weights_from_safetensors(model)
        init_rope_buffers(model)
    else:
        set_full_model_state_dict(model, cpu_state, strict=True)
        _restore_nonpersistent_buffers(model, cpu_buffers or {})
    _share_fsdp_comm_streams(model)
    return model


def _share_fsdp_comm_streams(model: nn.Module) -> None:
    """Create and share all-gather CUDA streams across FSDP units.

    Prefetch of a sibling needs those streams on the target's comm context.
    PEFT calls ``CausalLM.forward`` directly, so the first hook can be embed
    (a false root). Sharing streams here lets that prefetch succeed without
    running root ``_lazy_init`` before the first real forward.
    """
    units = cast(
        "list[FSDPModule]",
        [module for module in model.modules() if isinstance(module, FSDPModule)],
    )
    if not units:
        return
    if len(units) > 1:
        share_comm_ctx(units)
    device = next(model.parameters()).device
    if device.type == "meta":
        return
    comm = units[0]._get_fsdp_state()._comm_ctx
    comm.lazy_init(device)
    for unit in units:
        param_group = unit._get_fsdp_state()._fsdp_param_group
        if param_group is None:
            continue
        if not all(
            isinstance(fsdp_param.sharded_param, DTensor)
            for fsdp_param in param_group.fsdp_params
        ):
            continue
        param_group.lazy_init()

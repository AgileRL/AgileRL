# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Context-parallel mixer entry point and helpers shared by the mixer wraps.

Family catalog plans call
:func:`~agilerl.distributed.context_parallel.mamba.install_mamba_context_parallel`
or :func:`~agilerl.distributed.context_parallel.gdn.install_gdn_context_parallel`
on their own mixer classes. :func:`apply_mixer_context_parallel` runs every
catalog plan.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.func import functional_call

from agilerl.architectures import family_context_parallel_plans


def local_tensor(value: torch.Tensor) -> torch.Tensor:
    """This rank's shard of a ``DTensor``; any other tensor as is."""
    if isinstance(value, DTensor):
        return value.to_local()
    return value


def drop_sequence_shard_mask(
    args: tuple[object, ...], kwargs: dict[str, object]
) -> tuple[tuple[object, ...], dict[str, object]]:
    """Inner scan is the full sequence; shard mask / seq_idx cannot apply."""
    kwargs = {
        key: value
        for key, value in kwargs.items()
        if key
        not in {
            "attention_mask",
            "seq_idx",
            "cu_seq_lens_q",
            "cu_seq_lens_k",
            "cu_seqlens",
        }
    }
    if len(args) >= 2:
        args = (args[0], None, *args[2:])
    return args, kwargs


@dataclass(frozen=True)
class LocalParamSlot:
    """Where one parameter of a :func:`call_local` module lives."""

    path: str
    owner: nn.Module
    attr: str


def local_param_slots(module: nn.Module) -> tuple[LocalParamSlot, ...]:
    """Owner and attribute of every parameter of ``module``.

    :param module: Module :func:`call_local` will run.
    :return: One slot per parameter, in ``named_parameters`` order.
    """
    return tuple(
        LocalParamSlot(f"{owner_path}.{attr}" if owner_path else attr, owner, attr)
        for owner_path, owner in module.named_modules()
        for attr, _param in owner.named_parameters(recurse=False)
    )


def call_local(
    module: nn.Module,
    slots: tuple[LocalParamSlot, ...],
    *args: object,
    **kwargs: object,
) -> torch.Tensor:
    """Run ``module`` on this rank's local shards of its parameters.

    :param module: Module to run.
    :param slots: :func:`local_param_slots` of ``module``. Parameters are read
        from their owners on each call, since FSDP2 swaps them every forward.
    :return: ``module`` output.
    """
    params = [getattr(slot.owner, slot.attr) for slot in slots]
    if not any(isinstance(param, DTensor) for param in params):
        return module(*args, **kwargs)
    local_params = {
        slot.path: local_tensor(param)
        for slot, param in zip(slots, params, strict=True)
    }
    return functional_call(module, local_params, args, kwargs)


def assign_view(module: nn.Module, name: str, value: torch.Tensor | None) -> None:
    """Put a view in a Parameter slot without wrapping it (keeps autograd)."""
    module._parameters[name] = cast("nn.Parameter | None", value)


def apply_mixer_context_parallel(model: nn.Module, cp_mesh: DeviceMesh | None) -> int:
    """Run each family's context-parallel mixer wrap on ``model``.

    :param model: Module tree that may contain mixers.
    :param cp_mesh: Context-parallel mesh. ``None`` or size 1 wraps nothing.
    :return: How many mixers were wrapped.
    """
    if cp_mesh is None or int(cp_mesh.size()) <= 1:
        return 0
    return sum(plan.wrap(model, cp_mesh) for plan in family_context_parallel_plans())

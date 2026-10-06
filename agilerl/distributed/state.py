# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Full state-dict scatter, gather, and reshard for FSDP2."""

from __future__ import annotations

from collections.abc import Generator, Iterable, Sequence
from contextlib import _GeneratorContextManager, contextmanager
from typing import Any, overload

import torch
from torch import nn
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor, distribute_tensor

CHECKPOINT_FQN_PART = "_checkpoint_wrapped_module"


def canonical_fsdp_param_fqn(name: str) -> str:
    """Map a live checkpoint-wrapped FQN to the pre-wrap state-dict key.

    :param name: Live parameter FQN.
    :type name: str
    :return: FQN with checkpoint-wrapper segments removed.
    :rtype: str
    """
    return ".".join(part for part in name.split(".") if part != CHECKPOINT_FQN_PART)


def _state_dict_value(state_dict: dict[str, Any], name: str) -> torch.Tensor | None:
    if name in state_dict:
        value = state_dict[name]
    else:
        value = state_dict.get(canonical_fsdp_param_fqn(name))
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        msg = f"state_dict[{name!r}] must be a Tensor, got {type(value).__name__}"
        raise TypeError(msg)
    return value


def reshard_fsdp_modules(model: nn.Module) -> None:
    """Return FSDP2 units to sharded state after HF ``generate``.

    ``reshard_after_forward=False`` leaves all-gathered params live. The fused
    lm_head gemm in ``learn`` then ``full_tensor()``s a stale buffer.

    :param model: Model whose FSDP2 units to reshard.
    :type model: nn.Module
    """
    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.reshard()


def set_full_model_state_dict(
    model: nn.Module,
    state_dict: dict[str, Any],
    strict: bool = False,
) -> None:
    """Scatter a full (unsharded) state dict onto an FSDP2-sharded model.

    Replicated parameters (FSDP ``ignored_params``, e.g. token embeddings)
    are plain ``Parameter``s. DCP's full-state loader walks every parameter
    and assumes DTensors, so mixed models cannot use it. DTensor shards are
    scattered with ``distribute_tensor``; replicated params are copied.

    :param model: FSDP2-sharded model to write into.
    :type model: nn.Module
    :param state_dict: Full tensors keyed by pre-wrap FQN.
    :type state_dict: dict[str, Any]
    :param strict: Raise on missing or unexpected keys.
    :type strict: bool
    """
    values = {
        key: _state_dict_value(state_dict, key) for key, _ in model.named_parameters()
    }
    missing = [key for key, value in values.items() if value is None]
    if strict and missing:
        preview = ", ".join(missing[:8])
        suffix = "…" if len(missing) > 8 else ""
        msg = f"Missing keys in state_dict ({len(missing)}): {preview}{suffix}"
        raise RuntimeError(msg)
    with torch.no_grad():
        for key, dest in model.named_parameters():
            value = values[key]
            if value is None:
                continue
            _write_full_tensor(dest, value)
        for key, buf in model.named_buffers():
            value = _state_dict_value(state_dict, key)
            if value is None:
                continue
            buf.copy_(value.to(device=buf.device, dtype=buf.dtype))


def _write_full_tensor(dest: nn.Parameter, value: torch.Tensor) -> None:
    """Copy a full tensor into a plain parameter or an FSDP2 DTensor shard."""
    value = value.to(device=dest.device, dtype=dest.dtype)
    if isinstance(dest, DTensor):
        sharded = distribute_tensor(value, dest.device_mesh, dest.placements)
        dest.to_local().copy_(sharded.to_local())
    else:
        dest.data.copy_(value)


@overload
def materialize_dtensors(
    t0: torch.Tensor, t1: torch.Tensor | None, /
) -> _GeneratorContextManager[tuple[torch.Tensor, torch.Tensor | None], None, None]: ...


@overload
def materialize_dtensors(
    *tensors: torch.Tensor | None,
) -> _GeneratorContextManager[list[torch.Tensor | None], None, None]: ...


@contextmanager
def materialize_dtensors(
    *tensors: torch.Tensor | None,
) -> Generator[Sequence[torch.Tensor | None], None, None]:
    """All-gather ``DTensor`` shards to dense locals without swapping module params.

    Prefer this for ephemeral matmuls (fused lm_head logprobs, Liger). Use
    :func:`gather_params` when in-module reads must
    see dense weights (``state_dict`` / PEFT ``save_pretrained``). All ranks
    must enter and exit together. Yields a list parallel to ``tensors``.

    :param tensors: Tensors to densify; ``None`` passes through.
    :type tensors: torch.Tensor | None
    """
    gathered_tensors: list[torch.Tensor | None] = []
    for tensor in tensors:
        if tensor is None or not isinstance(tensor, DTensor):
            gathered_tensors.append(tensor)
        else:
            gathered_tensors.append(tensor.full_tensor())
    yield gathered_tensors


def parameter_owners(
    root: nn.Module,
    params: Iterable[torch.Tensor],
) -> dict[int, tuple[nn.Module, str]]:
    """Map ``id(param)`` to ``(module, attr_name)`` for params registered under ``root``.

    Params not registered on ``root`` or its submodules are absent.

    :param root: Module whose submodules own the parameters.
    :type root: nn.Module
    :param params: Parameters to look up.
    :type params: Iterable[torch.Tensor]
    :return: ``id(param)`` to ``(module, attr_name)``.
    :rtype: dict[int, tuple[nn.Module, str]]
    """
    wanted = {id(param) for param in params}
    owners: dict[int, tuple[nn.Module, str]] = {}
    if not wanted:
        return owners
    for module in root.modules():
        for name, value in module._parameters.items():
            if value is not None and id(value) in wanted:
                owners.setdefault(id(value), (module, name))
    return owners


@contextmanager
def full_shape_views(
    root: nn.Module,
    params: Sequence[torch.Tensor | None],
) -> Generator[None, None, None]:
    """Expose global-shape views on FSDP2 ``DTensor`` params for shape-only reads.

    Each sharded param is temporarily replaced on its owning module with a
    zero-storage view (a scalar expanded to the DTensor global shape) so
    shape, dtype and device reads see the full tensor without
    :meth:`~torch.distributed.tensor.DTensor.full_tensor`. Values must not
    be read inside the block; the original ``DTensor`` is restored on exit
    when the installed view is still the live parameter. Plain tensors pass
    through untouched. Duplicate references are processed once (by identity).

    :param root: Module whose submodules own ``params``.
    :type root: nn.Module
    :param params: Parameters to expose; ``None`` entries are skipped.
    :type params: Sequence[torch.Tensor | None]
    """
    restores: list[tuple[nn.Module, str, DTensor, nn.Parameter]] = []
    sharded = {id(param): param for param in params if isinstance(param, DTensor)}
    owners = parameter_owners(root, sharded.values())
    try:
        for param_id, param in sharded.items():
            owner = owners.get(param_id)
            if owner is None:
                continue
            module, name = owner
            view = torch.empty((), dtype=param.dtype, device=param.device).expand(
                tuple(param.shape)
            )
            view_param = nn.Parameter(view, requires_grad=bool(param.requires_grad))
            owned: dict[str, Any] = module._parameters
            owned[name] = view_param
            restores.append((module, name, param, view_param))
        yield
    finally:
        for module, name, original, view_param in reversed(restores):
            owned: dict[str, Any] = module._parameters
            if owned.get(name) is view_param:
                owned[name] = original


@contextmanager
def gather_params(
    root: nn.Module,
    params: Sequence[torch.Tensor | None],
) -> Generator[list[torch.Tensor | None], None, None]:
    """Materialize full (unsharded) views of ``params`` for the duration of the context.

    Plain tensors are left unchanged. For FSDP2 ``DTensor`` parameters, each
    tensor is all-gathered with :meth:`~torch.distributed.tensor.DTensor.full_tensor`
    and temporarily installed on its owning module so in-module reads (e.g.
    ``state_dict`` / PEFT ``save_pretrained``) see dense weights. Original
    shards are restored on exit.

    Yields a list parallel to ``params``: dense locals for any gathered
    ``DTensor``, and the original handles otherwise. Callers that hold
    pre-gather tensor references must use the yielded list for math — those
    references still point at the shard. For matmul-only gathers prefer
    :func:`materialize_dtensors` (no module Parameter install).

    Gathered parameters are read-only: writes are discarded when the sharded
    ``DTensor`` is restored. Write into a sharded model with
    :func:`set_full_model_state_dict`. All ranks must enter and exit together.

    :param root: Module whose submodules own ``params``.
    :type root: nn.Module
    :param params: Parameters to gather; ``None`` entries pass through.
    :type params: Sequence[torch.Tensor | None]
    """
    restores: list[tuple[nn.Module, str, torch.Tensor]] = []
    gathered_tensors: list[torch.Tensor | None] = []
    owners = parameter_owners(
        root, (param for param in params if isinstance(param, DTensor))
    )
    try:
        for param in params:
            if not isinstance(param, DTensor):
                gathered_tensors.append(param)
                continue
            full = param.full_tensor()
            gathered_tensors.append(full)
            owner = owners.get(id(param))
            if owner is None:
                continue
            module, name = owner
            restores.append((module, name, param))
            owned: dict[str, Any] = module._parameters
            owned[name] = nn.Parameter(full, requires_grad=bool(param.requires_grad))
        yield gathered_tensors
    finally:
        for module, name, original in reversed(restores):
            owned: dict[str, Any] = module._parameters
            owned[name] = original

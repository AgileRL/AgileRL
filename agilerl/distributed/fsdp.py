# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""FSDP2 state loading, gather, and CPU-offload optimizer helpers."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import re
from collections.abc import Callable, Generator, Iterable, Sequence
from contextlib import _GeneratorContextManager, contextmanager
from pathlib import Path
from typing import Any, Protocol, cast, overload

import torch
import torch.nn.init as init
from safetensors import safe_open
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FSDPModule, share_comm_ctx
from torch.distributed.tensor import DTensor, distribute_tensor
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset
from torch.distributed.tensor.placement_types import Shard
from transformers import PretrainedConfig, PreTrainedModel
from transformers.conversion_mapping import get_checkpoint_conversion_mapping
from transformers.core_model_loading import WeightTransform
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, cached_file

from agilerl.arena.models.fsdp import FSDPConfig
from agilerl.distributed.expert_parallel import (
    ParallelMesh,
    apply_expert_parallel,
    assert_packed_experts_ep_sharded,
    iter_packed_expert_modules,
    shard_experts_on_ep,
)
from agilerl.distributed.fsdp_blocks import (
    apply_fsdp2,
    compile_dense_block_modules,
    resolve_causal_lm,
)
from agilerl.distributed.fsdp_meta import (
    ModelWithTiedWeightKeys,
    init_rope_buffers,
    match_checkpoint_key,
    renamed_checkpoint_keys,
    restore_after_to_empty,
)
from agilerl.distributed.process import get_world_size
from agilerl.distributed.tensor_parallel import (
    apply_tensor_parallel,
    restore_tensor_parallel,
)

WRAPPER_FQN_PARTS = frozenset({"_checkpoint_wrapped_module", "_fsdp_wrapped_module"})


def _module_pretrained_config(module: nn.Module) -> PretrainedConfig | None:
    """Return a Hugging Face config carried by *module*, when present."""
    if isinstance(module, PreTrainedModel):
        return module.config
    module_config = vars(module).get("config")
    if isinstance(module_config, PretrainedConfig):
        return module_config
    return None


def canonical_fsdp_param_fqn(name: str) -> str:
    """Map a live wrapped FQN to the pre-wrap state-dict key.

    Activation-checkpoint and FSDP1 wrappers insert a module segment into
    ``named_modules`` / ``named_parameters`` names that ``state_dict`` drops.

    :param name: Live parameter or module FQN.
    :type name: str
    :return: FQN with wrapper segments removed.
    :rtype: str
    """
    return ".".join(part for part in name.split(".") if part not in WRAPPER_FQN_PARTS)


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
    skip: frozenset[str] = frozenset(),
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
    :param skip: Live parameter FQNs written by another path (e.g. sliced
        expert shards); excluded from the missing-key check.
    :type skip: frozenset[str]
    """
    values = {
        key: _state_dict_value(state_dict, key)
        for key, _ in model.named_parameters()
        if key not in skip
    }
    missing = [key for key, value in values.items() if value is None]
    if strict and missing:
        preview = ", ".join(missing[:8])
        suffix = "…" if len(missing) > 8 else ""
        msg = f"Missing keys in state_dict ({len(missing)}): {preview}{suffix}"
        raise RuntimeError(msg)
    with torch.no_grad():
        for key, dest in model.named_parameters():
            if key in skip:
                continue
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
        # EP reshapes some params after the CPU snapshot (stacked LoRA B
        # goes 2D [out, E*r] to 3D [out, r, E]); restore the global shape
        # before sharding. Same element count, so no values move.
        if value.dim() != dest.dim():
            value = value.reshape(dest.shape)
        sharded = distribute_tensor(value, dest.device_mesh, dest.placements)
        with torch.no_grad():
            dest.to_local().copy_(sharded.to_local())
    else:
        dest.data.copy_(value)


def _write_ep_expert_slice(dest: DTensor, value: torch.Tensor) -> None:
    """Copy this rank's shard of a full expert tensor into local storage.

    Only the local block crosses to the destination device, so no full
    expert tensor ever materializes there. DTensor computes the local block,
    covering 1D EP meshes and FSDP-over-EP ``(_StridedShard, Shard)``
    layouts alike. Snapshot layouts that predate a sharding-time reshape
    (stacked LoRA B as 2D ``[out, E*r]``) are restored to the global
    sharded shape first; the element count is unchanged.
    """
    full = value.detach().to("cpu")
    if full.dim() != dest.dim():
        full = full.reshape(dest.shape)
    # Checkpoint import passes external tensors; copy_ would broadcast a size-1 dim.
    if full.shape != dest.shape:
        msg = (
            f"EP expert value shape {tuple(full.shape)} does not match "
            f"destination shape {tuple(dest.shape)}"
        )
        raise ValueError(msg)
    local_shape, offset = compute_local_shape_and_global_offset(
        full.shape, dest.device_mesh, dest.placements
    )
    piece = full[
        tuple(
            slice(start, start + size)
            for start, size in zip(offset, local_shape, strict=True)
        )
    ]
    local = dest.to_local()
    with torch.no_grad():
        local.copy_(piece.to(device=local.device, dtype=dest.dtype))


def _ep_expert_live_keys(
    model: nn.Module, packed_ep_modules: list[tuple[str, nn.Module]]
) -> frozenset[str]:
    """Live FQNs of packed-expert parameters, wrapper segments included.

    Matches on canonical module prefixes so checkpoint-wrapped live names
    still resolve to their expert modules.
    """
    prefixes = [canonical_fsdp_param_fqn(prefix) for prefix, _ in packed_ep_modules]

    def _is_expert(canonical: str) -> bool:
        return any(
            not prefix or canonical == prefix or canonical.startswith(prefix + ".")
            for prefix in prefixes
        )

    return frozenset(
        live
        for live, _ in model.named_parameters()
        if _is_expert(canonical_fsdp_param_fqn(live))
    )


def _scatter_ep_expert_slices(
    model: nn.Module,
    state_dict: dict[str, Any],
    expert_keys: frozenset[str],
) -> None:
    """Write full expert tensors into per-rank EP shards, chunk by chunk."""
    params = dict(model.named_parameters())
    values = {key: _state_dict_value(state_dict, key) for key in sorted(expert_keys)}
    full = {key: value for key, value in values.items() if value is not None}
    missing = [key for key in values if key not in full]
    if missing:
        preview = ", ".join(sorted(missing)[:8])
        suffix = "…" if len(missing) > 8 else ""
        msg = f"Missing expert keys in state_dict ({len(missing)}): {preview}{suffix}"
        raise RuntimeError(msg)
    with torch.no_grad():
        for key, value in full.items():
            _write_ep_expert_slice(cast("DTensor", params[key]), value)


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


PRETRAINED_MODEL_PREFIX = "pretrained_model."
PEFT_BASE_PREFIX = "base_model.model."
LORA_MARKERS = ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B")


class SafetensorsSliceView(Protocol):
    def __getitem__(self, key: slice | tuple[slice, ...]) -> torch.Tensor: ...


class SafetensorsFileHandle(Protocol):
    def get_slice(self, key: str) -> SafetensorsSliceView: ...
    def get_tensor(self, key: str) -> torch.Tensor: ...


def checkpoint_key_for_parameter(live_name: str) -> str:
    """Map a live FSDP parameter FQN to a HuggingFace safetensors key."""
    name = canonical_fsdp_param_fqn(live_name)
    name = name.removeprefix(PRETRAINED_MODEL_PREFIX)
    if name == "base_model.model":
        return ""
    name = name.removeprefix(PEFT_BASE_PREFIX)
    return _without_base_layer_modules(name)


def checkpoint_key_candidates(live_name: str) -> tuple[str, ...]:
    """Safetensors keys that may store this parameter.

    A checkpoint may name the language body ``backbone`` while the module
    built from config names that body ``model``.
    """
    key = checkpoint_key_for_parameter(live_name)
    alias = key.replace("language_model.model.", "language_model.backbone.", 1)
    if alias == key:
        return (key,)
    return (key, alias)


def _checkpoint_transform_groups(
    model: nn.Module,
) -> list[tuple[str, list[WeightTransform]]]:
    """Conversions registered for a module class or model type, under that module path.

    Longer paths come first so a nested module's mapping wins over its parent.
    """
    groups: list[tuple[str, list[WeightTransform]]] = []
    for name, module in model.named_modules():
        config = _module_pretrained_config(module)
        if config is None:
            continue
        model_type = config.model_type
        conversions = get_checkpoint_conversion_mapping(type(module).__name__)
        if not conversions and model_type:
            conversions = get_checkpoint_conversion_mapping(model_type)
        if not conversions:
            continue
        prefix = checkpoint_key_for_parameter(name) if name else ""
        groups.append((prefix, list(conversions)))
    groups.sort(key=lambda item: len(item[0]), reverse=True)
    return groups


def _shift_slices_for_split(
    index_slices: tuple[slice, ...],
    *,
    split_index: int,
    rows: int,
) -> tuple[slice, ...]:
    """Move a shard slice onto one piece of a stacked source tensor."""
    row = index_slices[0]
    offset = split_index * rows
    return (slice(offset + row.start, offset + row.stop), *index_slices[1:])


def _without_base_layer_modules(key: str) -> str:
    """Drop PEFT ``base_layer`` modules from a parameter path."""
    stripped = key
    while ".base_layer." in stripped:
        stripped = stripped.replace(".base_layer.", ".", 1)
    return stripped


def _contiguous_indexed_weight_keys(
    key: str, key_files: dict[str, str]
) -> list[str] | None:
    """Keys ``parent.{i}.leaf.weight`` that stack into ``parent.leaf``."""
    parent, dot, leaf = key.rpartition(".")
    if not dot:
        return None
    prefix = f"{parent}."
    suffix = f".{leaf}.weight"
    found: dict[int, str] = {}
    for candidate in key_files:
        if not (candidate.startswith(prefix) and candidate.endswith(suffix)):
            continue
        index = candidate[len(prefix) : -len(suffix)]
        if not index.isdigit():
            continue
        found[int(index)] = candidate
    if not found:
        return None
    indexes = sorted(found)
    if indexes != list(range(indexes[-1] + 1)):
        msg = f"Checkpoint indexes for {key!r} are not contiguous: {indexes}"
        raise RuntimeError(msg)
    return [found[index] for index in indexes]


def _packed_checkpoint_keys(
    candidates: tuple[str, ...], key_files: dict[str, str]
) -> list[str] | None:
    """Indexed weights that stack into this packed parameter."""
    for key in candidates:
        packed = _contiguous_indexed_weight_keys(
            _without_base_layer_modules(key), key_files
        )
        if packed is not None:
            return packed
    return None


def _take_local_expert_rows(
    source: torch.Tensor,
    dest: torch.Tensor,
    global_dim0: int,
) -> torch.Tensor:
    """Keep leftover-dp rows when ``source`` is still this rank's EP block."""
    if tuple(source.shape) == tuple(dest.shape):
        return source
    source_rows = int(source.shape[0])
    dest_rows = int(dest.shape[0])
    even_dim0_split = (
        source.ndim == dest.ndim
        and tuple(source.shape[1:]) == tuple(dest.shape[1:])
        and source_rows > dest_rows > 0
        and source_rows % dest_rows == 0
        and global_dim0 % source_rows == 0
    )
    if not even_dim0_split:
        return source
    ep = global_dim0 // source_rows
    leftover_dp = source_rows // dest_rows
    # rank = (replica * leftover_dp + dp_index) * ep + ep_index
    dp_index = (torch.distributed.get_rank() // ep) % leftover_dp
    return source.narrow(0, dp_index * dest_rows, dest_rows)


def _copy_indexed_weights(
    key_files: dict[str, str],
    keys: list[str],
    global_shape: tuple[int, ...],
    index_slices: tuple[slice, ...] | None,
    dest: torch.Tensor,
) -> None:
    """Stack ``keys`` on dim 0 and copy this rank's slice into ``dest``."""
    by_path: dict[str, list[str]] = {}
    for key in keys:
        by_path.setdefault(key_files[key], []).append(key)
    order = {key: index for index, key in enumerate(keys)}
    loaded: list[tuple[int, torch.Tensor]] = []
    for path, path_keys in by_path.items():
        with safe_open(path, framework="pt", device="cpu") as handle:
            loaded.extend((order[key], handle.get_tensor(key)) for key in path_keys)
    loaded.sort()
    stacked = torch.stack([tensor for _, tensor in loaded], dim=0)
    if tuple(stacked.shape) != global_shape:
        msg = (
            f"Stacked checkpoint shape {tuple(stacked.shape)} does not match "
            f"parameter shape {global_shape}"
        )
        raise RuntimeError(msg)
    source = stacked if index_slices is None else stacked[index_slices]
    source = _take_local_expert_rows(source, dest, global_shape[0])
    if tuple(source.shape) != tuple(dest.shape):
        msg = (
            f"Indexed weight slice {tuple(source.shape)} does not match "
            f"destination shape {tuple(dest.shape)}"
        )
        raise RuntimeError(msg)
    dest.copy_(source.to(device=dest.device, dtype=dest.dtype))


def global_shard_slices(
    global_shape: Sequence[int],
    placements: tuple[Any, ...],
    device_mesh: DeviceMesh,
) -> tuple[slice, ...]:
    """Per-dimension slices for this rank's shard of a logical tensor."""
    slices: list[slice] = [slice(0, int(size)) for size in global_shape]
    coordinate = device_mesh.get_coordinate()
    for mesh_dim, placement in enumerate(placements):
        if not isinstance(placement, Shard):
            continue
        shard_dim = placement.dim
        num_chunks = device_mesh.size(mesh_dim=mesh_dim)
        if coordinate is None:
            rank_at_mesh = device_mesh.get_local_rank(mesh_dim=mesh_dim)
        else:
            rank_at_mesh = int(coordinate[mesh_dim])
        dim_size = int(global_shape[shard_dim])
        local_size, offset = Shard.local_shard_size_and_offset(
            dim_size, num_chunks, rank_at_mesh
        )
        start = int(offset)
        end = start + int(local_size)
        slices[shard_dim] = slice(start, end)
    return tuple(slices)


def _stable_generator_seed(name: str) -> int:
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**63)


def _is_lora_parameter_name(name: str) -> bool:
    return any(marker in name for marker in LORA_MARKERS)


def _lora_is_b_matrix(name: str) -> bool:
    return "lora_B" in name or "lora_embedding_B" in name


def _is_value_head_parameter(candidates: tuple[str, ...]) -> bool:
    return any(key == "v_head" or key.startswith("v_head.") for key in candidates)


def _init_value_head_parameter(
    param: nn.Parameter,
    canonical_name: str,
    global_shape: Sequence[int],
    placements: tuple[Any, ...],
    device_mesh: DeviceMesh | None,
) -> None:
    """Fill a value-head shard the pretrained checkpoint does not contain."""
    if device_mesh is None or not isinstance(param, DTensor):
        slices = tuple(slice(0, int(size)) for size in global_shape)
    else:
        slices = global_shard_slices(global_shape, placements, device_mesh)
    with torch.no_grad():
        full = torch.empty(tuple(int(size) for size in global_shape), device="cpu")
        if canonical_name.endswith(".bias"):
            full.zero_()
        else:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(_stable_generator_seed(canonical_name))
            init.kaiming_uniform_(full, a=5**0.5, generator=generator)
        local = _parameter_dest_local(param)
        local.copy_(full[slices].to(device=local.device, dtype=local.dtype))


def _parameter_dest_local(param: torch.Tensor) -> torch.Tensor:
    if isinstance(param, DTensor):
        return param.to_local()
    return param


def _init_lora_parameter(
    param: nn.Parameter,
    canonical_name: str,
    global_shape: Sequence[int],
    placements: tuple[Any, ...],
    device_mesh: DeviceMesh | None,
) -> None:
    """Fill one LoRA parameter shard; identical on every rank."""
    if device_mesh is None or not isinstance(param, DTensor):
        slices = tuple(slice(0, int(size)) for size in global_shape)
    else:
        slices = global_shard_slices(global_shape, placements, device_mesh)
    with torch.no_grad():
        full = torch.empty(tuple(int(s) for s in global_shape), device="cpu")
        if _lora_is_b_matrix(canonical_name):
            full.zero_()
        else:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(_stable_generator_seed(canonical_name))
            init.kaiming_uniform_(full, a=5**0.5, generator=generator)
        local = _parameter_dest_local(param)
        local.copy_(full[slices].to(device=local.device, dtype=local.dtype))


def _copy_safetensors_slice(
    handle: SafetensorsFileHandle,
    checkpoint_key: str,
    index_slices: tuple[slice, ...],
    dest: torch.Tensor,
    global_dim0: int | None = None,
) -> None:
    source = handle.get_slice(checkpoint_key)[index_slices]
    if global_dim0 is not None:
        source = _take_local_expert_rows(source, dest, global_dim0)
    dest.copy_(source.to(device=dest.device, dtype=dest.dtype))


def _checkpoint_source_from_config(config: PretrainedConfig) -> str | None:
    raw_path = config.name_or_path
    if not raw_path:
        return None
    source = str(raw_path)
    index_file = cached_file(
        source,
        SAFE_WEIGHTS_INDEX_NAME,
        _raise_exceptions_for_missing_entries=False,
    )
    if index_file is not None:
        return source
    weights_file = cached_file(
        source,
        SAFE_WEIGHTS_NAME,
        _raise_exceptions_for_missing_entries=False,
    )
    if weights_file is not None:
        return source
    return None


def _next_unwrap_module(module: nn.Module) -> nn.Module | None:
    pretrained = getattr(module, "pretrained_model", None)
    if isinstance(pretrained, nn.Module):
        return pretrained
    get_base = getattr(module, "get_base_model", None)
    if callable(get_base):
        base = get_base()
        if isinstance(base, nn.Module):
            return base
    wrapped = getattr(module, "base_model", None)
    if isinstance(wrapped, nn.Module):
        return wrapped
    inner = getattr(module, "model", None)
    if isinstance(inner, nn.Module) and inner is not module:
        return inner
    return None


def _resolve_checkpoint_source(model: nn.Module) -> str:
    current: nn.Module | None = model
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        config = _module_pretrained_config(current)
        if config is not None:
            source = _checkpoint_source_from_config(config)
            if source is not None:
                return source
        current = _next_unwrap_module(current)
    msg = (
        "FSDP shard load requires a module config whose name_or_path resolves to "
        f"{SAFE_WEIGHTS_NAME} or "
        f"{SAFE_WEIGHTS_INDEX_NAME} (local directory or Hugging Face Hub id)"
    )
    raise RuntimeError(msg)


def _tied_weight_checkpoint_targets(model: nn.Module) -> set[str]:
    causal = resolve_causal_lm(model)
    if not isinstance(causal, ModelWithTiedWeightKeys):
        return set()
    tied = causal.all_tied_weights_keys
    if not isinstance(tied, dict):
        return set()
    return {key for key in tied if isinstance(key, str)}


def _candidate_is_tied_head(
    candidates: tuple[str, ...],
    model: nn.Module,
    tied_targets: set[str],
) -> bool:
    """Whether this checkpoint key is a tied output embedding, not its own tensor."""
    if any(key in tied_targets for key in candidates):
        return True
    causal = resolve_causal_lm(model)
    config = _module_pretrained_config(causal)
    if config is None or not config.tie_word_embeddings:
        return False
    return any(
        key in {"lm_head.weight", "embed_out.weight"}
        or key.endswith((".lm_head.weight", ".embed_out.weight"))
        for key in candidates
    )


def _build_safetensors_key_files(model_path: str) -> dict[str, str]:
    index_file = cached_file(
        model_path,
        SAFE_WEIGHTS_INDEX_NAME,
        _raise_exceptions_for_missing_entries=False,
    )
    if index_file is not None:
        weight_map = json.loads(Path(index_file).read_text(encoding="utf-8"))[
            "weight_map"
        ]
        key_files: dict[str, str] = {}
        shard_paths: dict[str, str] = {}
        for key, filename in weight_map.items():
            if filename not in shard_paths:
                shard = cached_file(
                    model_path,
                    filename,
                    _raise_exceptions_for_missing_entries=False,
                )
                if shard is None:
                    msg = (
                        f"FSDP shard load missing shard {filename!r} "
                        f"for checkpoint {model_path!r}"
                    )
                    raise RuntimeError(msg)
                shard_paths[filename] = shard
            key_files[key] = shard_paths[filename]
        return key_files
    weights_file = cached_file(
        model_path,
        SAFE_WEIGHTS_NAME,
        _raise_exceptions_for_missing_entries=False,
    )
    if weights_file is None:
        msg = (
            f"FSDP shard load requires {SAFE_WEIGHTS_NAME} or "
            f"{SAFE_WEIGHTS_INDEX_NAME} at checkpoint {model_path!r}"
        )
        raise RuntimeError(msg)
    with safe_open(weights_file, framework="pt") as handle:
        return dict.fromkeys(handle.keys(), weights_file)


def _initialize_ignored_missing_parameter(model: nn.Module, live_name: str) -> bool:
    """Fill a checkpoint-omitted parameter from the module that marks it ignorable.

    :param model: Module being loaded.
    :type model: nn.Module
    :param live_name: Parameter name from ``named_parameters``.
    :type live_name: str
    :return: True when an ancestor's ``_init_weights`` filled the parameter.
    :rtype: bool
    """
    module_path = live_name.rpartition(".")[0]
    owner = model.get_submodule(module_path) if module_path else model
    nodes = [owner]
    current_path = module_path
    while current_path:
        current_path = current_path.rpartition(".")[0]
        nodes.append(model.get_submodule(current_path) if current_path else model)
    key = checkpoint_key_for_parameter(live_name)
    for node in nodes:
        patterns = getattr(node, "_keys_to_ignore_on_load_missing", None)
        init_fn = getattr(node, "_init_weights", None)
        if not patterns or not callable(init_fn):
            continue
        matched = False
        for pattern in patterns:
            matched = (
                re.search(pattern, key) is not None
                or re.search(pattern, live_name) is not None
            )
            if matched:
                break
        if not matched:
            continue
        init_fn(owner)
        return True
    return False


def _load_sharded_weights_from_safetensors(model: nn.Module) -> None:
    """Copy each rank's parameter shards from on-disk safetensors."""
    model_path = _resolve_checkpoint_source(model)
    key_files = _build_safetensors_key_files(model_path)
    tied_targets = _tied_weight_checkpoint_targets(model)
    transform_groups = _checkpoint_transform_groups(model)
    renamed_keys = renamed_checkpoint_keys(key_files, transform_groups)
    vision_parameters_copied = 0
    unmatched_outside_language: list[str] = []
    # Parameters outside language_model stay empty when no checkpoint key matches.
    has_language_tower = any(
        "language_model." in name for name, _ in model.named_parameters()
    )
    copies_by_path: dict[
        str, list[tuple[str, tuple[slice, ...] | None, torch.Tensor, int]]
    ] = {}
    indexed: list[
        tuple[list[str], tuple[int, ...], tuple[slice, ...] | None, torch.Tensor]
    ] = []
    with torch.no_grad():
        for live_name, param in model.named_parameters():
            canonical = canonical_fsdp_param_fqn(live_name)
            candidates = checkpoint_key_candidates(live_name)
            checkpoint_key, split_index = match_checkpoint_key(
                candidates, key_files, renamed_keys, transform_groups
            )
            if checkpoint_key is not None:
                path = key_files[checkpoint_key]
                if isinstance(param, DTensor):
                    index_slices: tuple[slice, ...] | None = global_shard_slices(
                        param.shape,
                        param.placements,
                        param.device_mesh,
                    )
                else:
                    index_slices = tuple(slice(0, int(size)) for size in param.shape)
                if split_index is not None:
                    index_slices = _shift_slices_for_split(
                        index_slices,
                        split_index=split_index,
                        rows=int(param.shape[0]),
                    )
                copies_by_path.setdefault(path, []).append(
                    (
                        checkpoint_key,
                        index_slices,
                        _parameter_dest_local(param),
                        int(param.shape[0]),
                    )
                )
                if "vision_model." in canonical:
                    vision_parameters_copied += 1
                continue
            if _is_lora_parameter_name(live_name):
                mesh = param.device_mesh if isinstance(param, DTensor) else None
                placements = param.placements if isinstance(param, DTensor) else ()
                _init_lora_parameter(
                    param,
                    canonical,
                    param.shape,
                    placements,
                    mesh,
                )
                continue
            if _candidate_is_tied_head(candidates, model, tied_targets):
                continue
            if _is_value_head_parameter(candidates):
                mesh = param.device_mesh if isinstance(param, DTensor) else None
                placements = param.placements if isinstance(param, DTensor) else ()
                _init_value_head_parameter(
                    param,
                    canonical,
                    param.shape,
                    placements,
                    mesh,
                )
                continue
            if has_language_tower and "language_model." not in candidates[0]:
                if _initialize_ignored_missing_parameter(model, live_name):
                    continue
                unmatched_outside_language.append(canonical)
                continue
            packed = _packed_checkpoint_keys(candidates, key_files)
            if packed is None:
                renamed_packed = _packed_checkpoint_keys(candidates, renamed_keys)
                if renamed_packed is not None:
                    packed = [renamed_keys[key] for key in renamed_packed]
            if packed is not None:
                global_shape = tuple(int(size) for size in param.shape)
                slices = (
                    global_shard_slices(
                        global_shape, param.placements, param.device_mesh
                    )
                    if isinstance(param, DTensor)
                    else None
                )
                indexed.append(
                    (packed, global_shape, slices, _parameter_dest_local(param))
                )
                continue
            msg = (
                f"Missing checkpoint weight for parameter {live_name!r} "
                f"(mapped key {candidates[0]!r})"
            )
            raise RuntimeError(msg)
        for live_name, buf in model.named_buffers():
            checkpoint_key, _ = match_checkpoint_key(
                checkpoint_key_candidates(live_name),
                key_files,
                renamed_keys,
                transform_groups,
            )
            if checkpoint_key is None:
                continue
            path = key_files[checkpoint_key]
            copies_by_path.setdefault(path, []).append(
                (checkpoint_key, None, buf, int(buf.shape[0]) if buf.ndim else 0),
            )
        for path in sorted(copies_by_path):
            with safe_open(path, framework="pt", device="cpu") as handle:
                for checkpoint_key, index_slices, dest, global_dim0 in copies_by_path[
                    path
                ]:
                    if index_slices is None:
                        full = handle.get_tensor(checkpoint_key)
                        dest.copy_(full.to(device=dest.device, dtype=dest.dtype))
                    else:
                        _copy_safetensors_slice(
                            handle,
                            checkpoint_key,
                            index_slices,
                            dest,
                            global_dim0=global_dim0,
                        )
        for keys, global_shape, slices, dest in indexed:
            _copy_indexed_weights(key_files, keys, global_shape, slices, dest)
    first_unmatched = (
        unmatched_outside_language[0] if unmatched_outside_language else "none"
    )
    logging.getLogger(__name__).info(
        "FSDP safetensors load copied %d vision_model parameters; "
        "%d parameters outside language_model had no checkpoint key (first %s)",
        vision_parameters_copied,
        len(unmatched_outside_language),
        first_unmatched,
    )


def materialize_fsdp2_from_cpu_state(
    model: nn.Module,
    device: str | torch.device,
    config: FSDPConfig | None = None,
    parallel_mesh: ParallelMesh | None = None,
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
    :param parallel_mesh: HSDP / EP / TP mesh views; required when
        ``config.ep`` or ``config.tp`` is set, or ``config.shard_group_size``
        is smaller than the world.
        ``None`` shards over the default process group.
    :type parallel_mesh: ParallelMesh | None
    :param gradient_checkpointing: Wrap each transformer block with
        non-reentrant activation checkpointing before sharding.
    :type gradient_checkpointing: bool
    :return: The sharded model (same object).
    :rtype: nn.Module
    """
    config = config or FSDPConfig()
    hsdp = config.shard_group_size not in (None, get_world_size())
    if parallel_mesh is None and (config.ep > 1 or config.tp > 1 or hsdp):
        msg = (
            f"FSDPConfig(ep={config.ep}, tp={config.tp}, "
            f"shard_group_size={config.shard_group_size}) needs a ParallelMesh"
        )
        raise ValueError(msg)
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
    mesh = None if parallel_mesh is None else parallel_mesh.hsdp
    ep_mesh = None if parallel_mesh is None else parallel_mesh.ep
    tp_mesh = None if parallel_mesh is None or config.tp <= 1 else parallel_mesh.tp
    expert_mesh = None
    experts_skip_fsdp = False
    packed_ep_modules: list[tuple[str, nn.Module]] = []
    if config.ep > 1:
        assert parallel_mesh is not None
        apply_expert_parallel(
            model,
            ep_mesh,
            tp_mesh=tp_mesh,
            token_blocks=config.ep_token_blocks,
        )
        packed_ep_modules = list(iter_packed_expert_modules(model))
        if not packed_ep_modules:
            msg = (
                f"ep={config.ep} but apply_expert_parallel found no packed "
                "expert modules."
            )
            raise RuntimeError(msg)
        experts_skip_fsdp = parallel_mesh.leftover_dp == 1
        if not experts_skip_fsdp:
            expert_mesh = parallel_mesh.dp_mod_ep
    if config.tp > 1:
        apply_tensor_parallel(model, tp_mesh)
    # Before apply_fsdp2: fully_shard renames block classes, hiding them from
    # the _no_split_modules lookup.
    if config.compile_blocks:
        compile_dense_block_modules(model, config.compile_backend)
    apply_fsdp2(
        model,
        config,
        mesh=mesh,
        expert_mesh=expert_mesh,
        gradient_checkpointing=gradient_checkpointing,
    )
    target = torch.device("cpu") if config.cpu_offload else torch.device(device)
    model.to_empty(device=target)
    restore_after_to_empty(model)
    if experts_skip_fsdp:
        # Experts skip FSDP here, so their parameters are dense after
        # empty-materialization. Re-shard the empties so each load path
        # writes straight into per-rank shards.
        for _name, module in packed_ep_modules:
            shard_experts_on_ep(module, ep_mesh)
    if cpu_state is None:
        _load_sharded_weights_from_safetensors(model)
        init_rope_buffers(model)
    else:
        expert_keys = frozenset()
        if packed_ep_modules:
            expert_keys = _ep_expert_live_keys(model, packed_ep_modules)
            _scatter_ep_expert_slices(model, cpu_state, expert_keys)
        set_full_model_state_dict(model, cpu_state, strict=True, skip=expert_keys)
        _restore_nonpersistent_buffers(model, cpu_buffers or {})
    if experts_skip_fsdp:
        assert_packed_experts_ep_sharded(
            model, config.ep, modules=[mod for _name, mod in packed_ep_modules]
        )
    if tp_mesh is not None:
        restore_tensor_parallel(model, tp_mesh)
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


class CPUOffloadOptimizer:
    """Wrap an optimizer so optimizer states (AdamW m+v) stay on CPU.

    States are moved to GPU only for ``step()``, then back to CPU.  Parameters
    and gradients remain on GPU throughout, so compute is unaffected.  With
    activation checkpointing, activations and optimizer states are never on
    GPU at the same time: peak memory becomes ``max(activations, opt_states)``
    instead of ``sum``.

    Handles FSDP2 ``DTensor`` optimizer states by swapping ``_local_tensor``
    between CPU and GPU while preserving the DTensor wrapper.
    """

    def __init__(
        self, optimizer: torch.optim.Optimizer, pin_memory: bool = True
    ) -> None:
        self.optimizer = optimizer
        self.pin_memory = pin_memory
        self._initialized = False

    def _to_device(self, tensor: torch.Tensor, device: str) -> torch.Tensor:
        if device == "cpu":
            moved = tensor.to("cpu")
            if self.pin_memory and not moved.is_pinned():
                return moved.pin_memory()
            return moved
        return tensor.to(device, non_blocking=True)

    def _move_states(self, device: str) -> None:
        # Non-fused, non-capturable Adam reads ``step`` on CPU; a device copy
        # forces a host sync per parameter in ``step()``.
        step_on_device = bool(
            self.optimizer.defaults.get("fused")
            or self.optimizer.defaults.get("capturable")
        )
        for p in self.optimizer.state:
            state = self.optimizer.state[p]
            for k, v in state.items():
                if k == "step" and not step_on_device:
                    continue
                if isinstance(v, DTensor):
                    new_dt = copy.copy(v)
                    new_dt._local_tensor = self._to_device(v._local_tensor, device)
                    state[k] = new_dt
                elif isinstance(v, torch.Tensor):
                    state[k] = self._to_device(v, device)

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        if not self._initialized:
            result = self.optimizer.step(closure)
            self._move_states("cpu")
            self._initialized = True
            return result
        self._move_states("cuda")
        result = self.optimizer.step(closure)
        self._move_states("cpu")
        return result

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:
        if self._initialized:
            self._move_states("cuda")
        sd = self.optimizer.state_dict()
        if self._initialized:
            self._move_states("cpu")
        return sd

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.optimizer.load_state_dict(state_dict)
        self.offload_states()

    def offload_states(self) -> None:
        """Move populated optimizer states to CPU; later steps round-trip them."""
        self._move_states("cpu")
        self._initialized = True

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return self.optimizer.param_groups

    @param_groups.setter
    def param_groups(self, value: list[dict[str, Any]]) -> None:
        self.optimizer.param_groups = value

    @property
    def state(self) -> dict[torch.Tensor, Any]:
        return self.optimizer.state

    @property
    def base_optimizer(self) -> torch.optim.Optimizer:
        return self.optimizer

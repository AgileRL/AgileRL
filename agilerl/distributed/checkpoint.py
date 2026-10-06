# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Safetensors key mapping and sharded FSDP2 load."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import torch
import torch.nn.init as init
from safetensors import safe_open
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Shard
from transformers import PretrainedConfig, PreTrainedModel
from transformers.conversion_mapping import get_checkpoint_conversion_mapping
from transformers.core_model_loading import WeightTransform
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, cached_file

from agilerl.distributed.meta import (
    ModelWithTiedWeightKeys,
    match_checkpoint_key,
    renamed_checkpoint_keys,
)
from agilerl.distributed.state import canonical_fsdp_param_fqn
from agilerl.distributed.wrap import _resolve_causal_lm


def _module_pretrained_config(module: nn.Module) -> PretrainedConfig | None:
    """Return a Hugging Face config carried by *module*, when present."""
    if isinstance(module, PreTrainedModel):
        return module.config
    module_config = vars(module).get("config")
    if isinstance(module_config, PretrainedConfig):
        return module_config
    return None


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
) -> None:
    source = handle.get_slice(checkpoint_key)[index_slices]
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
    causal = _resolve_causal_lm(model)
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
    causal = _resolve_causal_lm(model)
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
        str, list[tuple[str, tuple[slice, ...] | None, torch.Tensor]]
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
                (checkpoint_key, None, buf),
            )
        for path in sorted(copies_by_path):
            with safe_open(path, framework="pt", device="cpu") as handle:
                for checkpoint_key, index_slices, dest in copies_by_path[path]:
                    if index_slices is None:
                        full = handle.get_tensor(checkpoint_key)
                        dest.copy_(full.to(device=dest.device, dtype=dest.dtype))
                    else:
                        _copy_safetensors_slice(
                            handle,
                            checkpoint_key,
                            index_slices,
                            dest,
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

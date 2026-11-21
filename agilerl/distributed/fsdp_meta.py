# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Load FSDP models built on the meta device.

Find the checkpoint's safetensors files, map live parameter names onto
their keys, then re-tie embeddings and fill RoPE buffers after
``to_empty``. FP8 and MXFP4
checkpoints dequantize to the load dtype on both this path and
``from_pretrained``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

import torch
from safetensors import safe_open
from torch import nn

from agilerl import HAS_LLM_DEPENDENCIES

if TYPE_CHECKING or HAS_LLM_DEPENDENCIES:
    from transformers import PretrainedConfig, PreTrainedModel
    from transformers.core_model_loading import (
        Chunk,
        WeightConverter,
        WeightRenaming,
        WeightTransform,
        dot_natural_key,
        rename_source_key,
    )
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    from transformers.quantizers import AutoHfQuantizer, AutoQuantizationConfig
    from transformers.utils import (
        SAFE_WEIGHTS_INDEX_NAME,
        SAFE_WEIGHTS_NAME,
        cached_file,
    )
    from transformers.utils.quantization_config import (
        FineGrainedFP8Config,
        Mxfp4Config,
    )


@runtime_checkable
class ModelWithTiedWeightKeys(Protocol):
    all_tied_weights_keys: Mapping[str, str]


class PerLayerTypeConfig(Protocol):
    per_layer_config: Mapping[str, object]


def restore_after_to_empty(model: nn.Module) -> None:
    """Re-tie input/output embeddings, which ``to_empty`` unties.

    HuggingFace ``tie_weights`` on the head's owner also ties its submodules.

    :raises RuntimeError: If a tied parameter is still separate from its
        source, which the safetensors load would leave empty.
    """
    owner = None
    for module in nn.Module.modules(model):
        # PEFT __getattr__ forwards lm_head; ownership is the registered child.
        children = dict(module.named_children())
        if isinstance(children.get("lm_head"), nn.Module) or isinstance(
            children.get("embed_out"), nn.Module
        ):
            owner = module
            break
    if owner is None:
        return
    tie = getattr(owner, "tie_weights", None)
    if callable(tie):
        tie()
    if not isinstance(owner, ModelWithTiedWeightKeys):
        return
    for target, source in owner.all_tied_weights_keys.items():
        if owner.get_parameter(target) is not owner.get_parameter(source):
            msg = (
                f"{type(owner).__name__}.{target} is not tied to {source} "
                "after tie_weights()"
            )
            raise RuntimeError(msg)


def init_rope_buffers(model: nn.Module) -> None:
    """Fill RoPE frequency buffers after a meta-device safetensors load.

    ``inv_freq`` / ``original_inv_freq``, and Gemma 4's
    ``{layer_type}_inv_freq`` pairs, are non-persistent, so they are not in
    the checkpoint. ``to_empty`` allocates them uninitialised. Needed when
    the model was built on meta and loaded from safetensors (FSDP ranks
    sharing one checkpoint). Same computation as HuggingFace ``_init_weights``.
    """
    with torch.no_grad():
        for module in nn.Module.modules(model):
            if "RotaryEmbedding" not in type(module).__name__:
                continue
            layer_fns = getattr(module, "rope_init_fns", None)
            if isinstance(layer_fns, dict) and layer_fns:
                _init_per_layer_type_rope(module, layer_fns)
                continue
            rope_types = getattr(module, "rope_type", None)
            if isinstance(rope_types, dict):
                _init_per_rope_type_rope(module, rope_types)
                continue
            if hasattr(module, "original_inv_freq"):
                _init_single_rope(module)


def _init_single_rope(module: nn.Module) -> None:
    rope_type = cast("str", module.rope_type)
    # "default" and vision "axial" live on the module, as in HF ``_init_weights``.
    rope_fn = cast(
        "Callable[..., tuple[torch.Tensor, object]]",
        getattr(module, f"compute_{rope_type}_rope_parameters", None)
        or ROPE_INIT_FUNCTIONS[rope_type],
    )
    inv_freq_buf = cast("torch.Tensor", module.inv_freq)
    original_buf = cast("torch.Tensor", module.original_inv_freq)
    # Keyword device: the default fn is static and takes ``(config, **kwargs)``.
    inv_freq, _ = rope_fn(module.config, device=inv_freq_buf.device)
    inv_freq_buf.copy_(inv_freq)
    original_buf.copy_(inv_freq)


def _init_per_layer_type_rope(
    module: nn.Module,
    layer_fns: Mapping[str, Callable[..., tuple[torch.Tensor, object]]],
) -> None:
    # Head dims differ by layer type; each type's config carries its own.
    layer_configs = cast("PerLayerTypeConfig", module.config).per_layer_config
    for layer_type, rope_fn in layer_fns.items():
        inv_freq_buf = cast("torch.Tensor", getattr(module, f"{layer_type}_inv_freq"))
        original_buf = cast(
            "torch.Tensor", getattr(module, f"{layer_type}_original_inv_freq")
        )
        inv_freq, _ = rope_fn(layer_configs[layer_type], layer_type=layer_type)
        inv_freq = inv_freq.to(device=inv_freq_buf.device, dtype=inv_freq_buf.dtype)
        inv_freq_buf.copy_(inv_freq)
        original_buf.copy_(inv_freq)


def _init_per_rope_type_rope(module: nn.Module, rope_types: Mapping[str, str]) -> None:
    # DeepSeek-V4 keys buffers by rope type (``main`` / ``compress``), not layer type.
    for rope_name, rope_type in rope_types.items():
        rope_fn = cast(
            "Callable[..., tuple[torch.Tensor, object]]",
            module.compute_default_rope_parameters
            if rope_type == "default"
            else ROPE_INIT_FUNCTIONS[rope_type],
        )
        inv_freq_buf = cast("torch.Tensor", getattr(module, f"{rope_name}_inv_freq"))
        original_buf = cast(
            "torch.Tensor", getattr(module, f"{rope_name}_original_inv_freq")
        )
        inv_freq, _ = rope_fn(
            module.config, device=inv_freq_buf.device, layer_type=rope_name
        )
        inv_freq_buf.copy_(inv_freq)
        original_buf.copy_(inv_freq)


def init_embed_scale_buffers(model: nn.Module) -> None:
    """Fill non-persistent ``embed_scale`` from ``scalar_embed_scale``.

    Gemma-style scaled word embeddings keep the multiplier as a buffer the
    checkpoint omits. ``to_empty`` leaves it uninitialised.
    """
    with torch.no_grad():
        for module in nn.Module.modules(model):
            scale = getattr(module, "scalar_embed_scale", None)
            buf = getattr(module, "embed_scale", None)
            if not isinstance(scale, (int, float)) or not isinstance(buf, torch.Tensor):
                continue
            if buf.is_meta:
                continue
            buf.fill_(float(scale))


def converted_checkpoint_key(
    key: str,
    transforms: Sequence[WeightTransform],
) -> tuple[str, int | None]:
    """Map a live parameter name back to the checkpoint key it was saved under.

    A transform that splits one source tensor records which piece this name is.

    :param key: Live parameter name.
    :type key: str
    :param transforms: Conversions registered for one module.
    :type transforms: Sequence[WeightTransform]
    :return: Checkpoint key, and the split index when one source feeds several
        parameters.
    :rtype: tuple[str, int | None]
    """
    current = key
    split_index: int | None = None
    for transform in reversed(transforms):
        renamed, pattern = transform.reverse_transform().rename_source_key(current)
        if pattern is None:
            continue
        if isinstance(transform, WeightConverter):
            operations = transform.operations
            if any(isinstance(operation, Chunk) for operation in operations):
                split_index = list(transform.target_patterns).index(pattern)
        current = renamed
    return current, split_index


def iter_converted_checkpoint_keys(
    key: str,
    groups: Sequence[tuple[str, Sequence[WeightTransform]]],
) -> Iterable[tuple[str, int | None]]:
    """Yield checkpoint keys produced by applying each group under its module path."""
    for prefix, transforms in groups:
        if prefix:
            head = f"{prefix}."
            if not key.startswith(head):
                continue
            relative = key[len(head) :]
        else:
            relative = key
        converted, split_index = converted_checkpoint_key(relative, transforms)
        if converted == relative:
            continue
        full = f"{prefix}.{converted}" if prefix else converted
        yield full, split_index


def _one_to_one_live_name(
    checkpoint_key: str,
    prefix: str,
    transforms: Sequence[WeightTransform],
) -> str | None:
    """Return the live name after one-to-one renames, or None if unchanged."""
    head = f"{prefix}." if prefix else ""
    if not checkpoint_key.startswith(head):
        return None
    current = checkpoint_key[len(head) :]
    for transform in transforms:
        if not isinstance(transform, WeightConverter):
            current, _ = transform.rename_source_key(current)
    live = f"{head}{current}"
    if live == checkpoint_key:
        return None
    return live


def renamed_checkpoint_keys(
    checkpoint_keys: Iterable[str],
    groups: Sequence[tuple[str, Sequence[WeightTransform]]],
) -> dict[str, str]:
    """Map live names to stored keys through each one-to-one rename that matches.

    Packing converters are resolved by :func:`iter_converted_checkpoint_keys`.

    :param checkpoint_keys: Keys stored in the safetensors files.
    :type checkpoint_keys: Iterable[str]
    :param groups: Conversions per module path.
    :type groups: Sequence[tuple[str, Sequence[WeightTransform]]]
    :return: Live parameter name to checkpoint key.
    :rtype: dict[str, str]
    """
    renamed: dict[str, str] = {}
    for checkpoint_key in checkpoint_keys:
        for prefix, transforms in groups:
            live = _one_to_one_live_name(checkpoint_key, prefix, transforms)
            if live is None:
                continue
            existing = renamed.setdefault(live, checkpoint_key)
            if existing != checkpoint_key:
                msg = (
                    f"Checkpoint keys {existing!r} and {checkpoint_key!r} "
                    f"both rename to {live!r}"
                )
                raise ValueError(msg)
    return renamed


@dataclass
class ConverterRoute:
    """One converter instance: the live key it is named by and the stored keys it reads."""

    converter: WeightConverter
    target: str
    sources: list[tuple[str, str]] = field(default_factory=list)


def converter_checkpoint_routes(
    checkpoint_keys: Iterable[str],
    groups: Sequence[tuple[str, Sequence[WeightTransform]]],
) -> dict[str, ConverterRoute]:
    """Group stored keys by the converter instance that loads them, as ``from_pretrained`` does.

    :param checkpoint_keys: Keys stored in the safetensors files.
    :type checkpoint_keys: Iterable[str]
    :param groups: Conversions per module path, longest path first.
    :type groups: Sequence[tuple[str, Sequence[WeightTransform]]]
    :return: Live parameter name to its route, sources in the loader's natural
        key order; every target of a one-to-many converter shares one route.
    :rtype: dict[str, ConverterRoute]
    """
    routes: dict[str, ConverterRoute] = {}
    for checkpoint_key in sorted(checkpoint_keys, key=dot_natural_key):
        for prefix, transforms in groups:
            head = f"{prefix}." if prefix else ""
            if not checkpoint_key.startswith(head):
                continue
            converters = [t for t in transforms if isinstance(t, WeightConverter)]
            renamed, pattern = rename_source_key(
                checkpoint_key[len(head) :],
                [t for t in transforms if isinstance(t, WeightRenaming)],
                converters,
            )
            if pattern is None:
                continue
            target = f"{head}{renamed}"
            route = routes.get(target)
            if route is None:
                converter = next(c for c in converters if pattern in c.source_patterns)
                route = ConverterRoute(converter, target)
                first_target = converter.target_patterns[0]
                for target_pattern in converter.target_patterns:
                    routes[target.replace(first_target, target_pattern)] = route
            route.sources.append((pattern, checkpoint_key))
            break
    return routes


def match_checkpoint_key(
    candidates: Sequence[str],
    key_files: dict[str, str],
    renamed_keys: dict[str, str],
    transform_groups: Sequence[tuple[str, Sequence[WeightTransform]]],
) -> tuple[str | None, int | None]:
    """Resolve candidates to a stored checkpoint key and optional split index.

    :param candidates: Live-name keys to try, from :func:`checkpoint_key_candidates`.
    :type candidates: Sequence[str]
    :param key_files: Stored checkpoint key to safetensors path.
    :type key_files: dict[str, str]
    :param renamed_keys: Live name to stored key from :func:`renamed_checkpoint_keys`.
    :type renamed_keys: dict[str, str]
    :param transform_groups: Conversions per module path.
    :type transform_groups: Sequence[tuple[str, Sequence[WeightTransform]]]
    :return: Stored key and split index, or ``(None, None)``.
    :rtype: tuple[str | None, int | None]
    """
    direct = next((key for key in candidates if key in key_files), None)
    if direct is not None:
        return direct, None
    renamed = next(
        (renamed_keys[key] for key in candidates if key in renamed_keys),
        None,
    )
    if renamed is not None:
        return renamed, None
    for candidate in candidates:
        for converted, split_index in iter_converted_checkpoint_keys(
            candidate, transform_groups
        ):
            if converted in key_files and converted != candidate:
                return converted, split_index
    return None, None


def checkpoint_quantization_config(
    config: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    """Return the quantization a checkpoint was saved with, where ``from_pretrained`` reads it.

    :param config: Checkpoint ``config.json`` contents.
    :type config: Mapping[str, Any]
    :return: ``quantization_config`` from the top level or the text config.
    :rtype: Mapping[str, Any] | None
    """
    text_config = config.get("text_config") or {}
    return config.get("quantization_config") or text_config.get("quantization_config")


def dequantizing_quantization_config(
    quantization_config: Mapping[str, Any],
) -> FineGrainedFP8Config | Mxfp4Config:
    """Build the checkpoint's quantization config with ``dequantize`` set.

    :param quantization_config: The checkpoint's ``quantization_config``.
    :type quantization_config: Mapping[str, Any]
    :return: Config that loads the weights in the model dtype on any GPU.
    :rtype: FineGrainedFP8Config | Mxfp4Config
    :raises ValueError: If the quantization does not dequantize on load.
    """
    config = AutoQuantizationConfig.from_dict(dict(quantization_config))
    if not isinstance(config, (FineGrainedFP8Config, Mxfp4Config)):
        msg = (
            f"Cannot train on {type(config).__name__} checkpoints: only FP8 "
            "and MXFP4 checkpoints dequantize on load"
        )
        raise ValueError(msg)
    config.dequantize = True
    return config


def dequantizing_transform_groups(
    groups: Sequence[tuple[str, Sequence[WeightTransform]]],
    quantization_config: Mapping[str, Any],
) -> list[tuple[str, list[WeightTransform]]]:
    """Add the quantizer's dequantize conversions to each group, as ``from_pretrained`` does.

    :param groups: Conversions per module path, longest path first.
    :type groups: Sequence[tuple[str, Sequence[WeightTransform]]]
    :param quantization_config: The checkpoint's ``quantization_config``.
    :type quantization_config: Mapping[str, Any]
    :return: Groups whose converters fold block scales into the weights.
    :rtype: list[tuple[str, list[WeightTransform]]]
    """
    quantizer = AutoHfQuantizer.from_config(
        dequantizing_quantization_config(quantization_config), pre_quantized=True
    )
    catch_all = len(quantizer.get_weight_conversions())
    if all(prefix for prefix, _ in groups):
        groups = [*groups, ("", [])]
    dequantizing: list[tuple[str, list[WeightTransform]]] = []
    for prefix, transforms in groups:
        # The quantizer appends its catch-all converters last. Only the root
        # keeps them, so a key no nested mapping converts falls through to it.
        updated = quantizer.update_weight_conversions(list(transforms))
        if prefix:
            updated = updated[: len(updated) - catch_all]
        dequantizing.append((prefix, updated))
    return dequantizing


def module_pretrained_config(module: nn.Module) -> PretrainedConfig | None:
    """Return a Hugging Face config carried by *module*, when present."""
    if isinstance(module, PreTrainedModel):
        return module.config
    module_config = vars(module).get("config")
    if isinstance(module_config, PretrainedConfig):
        return module_config
    return None


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


def resolve_checkpoint_source(model: nn.Module) -> str:
    current: nn.Module | None = model
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        config = module_pretrained_config(current)
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


def build_safetensors_key_files(model_path: str) -> dict[str, str]:
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

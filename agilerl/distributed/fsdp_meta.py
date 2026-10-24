# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Load FSDP models built on the meta device.

Map live parameter names onto Hugging Face safetensors keys, then re-tie
embeddings and fill RoPE buffers after ``to_empty``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Protocol, cast, runtime_checkable

import torch
from torch import nn
from transformers.core_model_loading import Chunk, WeightConverter, WeightTransform
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS


@runtime_checkable
class ModelWithTiedWeightKeys(Protocol):
    all_tied_weights_keys: Mapping[str, str]


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
    rope_types = getattr(module, "rope_type", {})
    for layer_type, rope_fn in layer_fns.items():
        kwargs: dict[str, object] = {"layer_type": layer_type}
        if (
            layer_type == "full_attention"
            and isinstance(rope_types, dict)
            and rope_types.get(layer_type) == "proportional"
        ):
            kwargs["head_dim_key"] = "global_head_dim"
        inv_freq_buf = cast("torch.Tensor", getattr(module, f"{layer_type}_inv_freq"))
        original_buf = cast(
            "torch.Tensor", getattr(module, f"{layer_type}_original_inv_freq")
        )
        inv_freq, _ = rope_fn(module.config, **kwargs)
        inv_freq = inv_freq.to(device=inv_freq_buf.device, dtype=inv_freq_buf.dtype)
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

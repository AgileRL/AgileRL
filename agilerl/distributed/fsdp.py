# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""FSDP2 wrap, gather, and CPU-offload optimizer helpers."""

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
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    checkpoint_wrapper,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    FSDPModule,
    MixedPrecisionPolicy,
    fully_shard,
    register_fsdp_forward_method,
    share_comm_ctx,
)
from torch.distributed.tensor import DTensor, distribute_tensor
from torch.distributed.tensor.placement_types import Shard
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, cached_file

from agilerl.arena.models.fsdp import FSDPConfig
from agilerl.distributed.process import is_distributed
from agilerl.utils.patching import class_is_patched

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


def _transformer_blocks(model: nn.Module) -> list[nn.Module]:
    """Outermost HuggingFace ``_no_split_modules`` hits.

    Nested no_split modules (packed experts inside a decoder layer) stay
    leaves of the parent FSDP unit. Empty when the model has no no-split
    names (root-only sharding).
    """
    no_split: set[str] = set()
    for module in model.modules():
        names = getattr(module, "_no_split_modules", None)
        if names:
            no_split.update(names)
    if not no_split:
        return []
    hits = [module for module in model.modules() if type(module).__name__ in no_split]
    nested: set[int] = set()
    for hit in hits:
        for child in hit.modules():
            if child is hit or type(child).__name__ not in no_split:
                continue
            nested.add(id(child))
    return [module for module in hits if id(module) not in nested]


def _unwrap_checkpoint(module: nn.Module) -> nn.Module:
    """Inner module when ``module`` is an activation-checkpoint wrapper."""
    inner = getattr(module, "_checkpoint_wrapped_module", None)
    return inner if isinstance(inner, nn.Module) else module


class FSDPBlockGroup(nn.Module):
    """Consecutive transformer blocks treated as one FSDP2 unit."""

    def __init__(self, blocks: Sequence[nn.Module]) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(list(blocks))
        first = _unwrap_checkpoint(self.blocks[0])
        # HF indexes ``layer.block_type`` / ``layer_idx`` on ModuleList entries.
        self.block_type = getattr(first, "block_type", None)
        self.layer_idx = getattr(first, "layer_idx", None)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        mask = kwargs.get("attention_mask")
        result: torch.Tensor | tuple[torch.Tensor, ...] = hidden_states
        for block in self.blocks:
            hidden = result[0] if isinstance(result, tuple) else result
            if isinstance(mask, dict):
                raw = _unwrap_checkpoint(block)
                block_kwargs = dict(kwargs)
                block_kwargs["attention_mask"] = mask.get(
                    getattr(raw, "block_type", None)
                )
                result = block(hidden, *args, **block_kwargs)
            else:
                result = block(hidden, *args, **kwargs)
        return result


def _replace_consecutive_blocks(
    root: nn.Module, blocks: Sequence[nn.Module]
) -> FSDPBlockGroup:
    """Lift a consecutive ``ModuleList`` span into an :class:`FSDPBlockGroup`."""
    target = list(blocks)
    if not target:
        msg = "blocks must be non-empty"
        raise ValueError(msg)
    for parent in root.modules():
        if not isinstance(parent, nn.ModuleList):
            continue
        children = list(parent)
        span = len(target)
        for start in range(len(children) - span + 1):
            if children[start : start + span] != target:
                continue
            for offset in range(span - 1, -1, -1):
                del parent[start + offset]
            group = FSDPBlockGroup(target)
            parent.insert(start, group)
            return group
    msg = "Could not find consecutive ModuleList span for FSDP block group"
    raise RuntimeError(msg)


def _group_transformer_units(
    model: nn.Module, units: Sequence[nn.Module], every_n: int
) -> list[nn.Module]:
    """Bundle consecutive wrap targets into groups of ``every_n``."""
    unit_list = list(units)
    if every_n == 1 or len(unit_list) <= 1:
        return unit_list
    grouped: list[nn.Module] = []
    for start in range(0, len(unit_list), every_n):
        chunk = unit_list[start : start + every_n]
        if len(chunk) == 1:
            grouped.append(chunk[0])
            continue
        grouped.append(_replace_consecutive_blocks(model, chunk))
    return grouped


def _replace_child(root: nn.Module, old: nn.Module, new: nn.Module) -> None:
    """Swap ``old`` for ``new`` on its parent under ``root``."""
    for parent in root.modules():
        for name, child in parent.named_children():
            if child is old:
                setattr(parent, name, new)
                return
    msg = f"Could not find parent module for {type(old).__name__}"
    raise RuntimeError(msg)


def _resolve_causal_lm(model: nn.Module) -> nn.Module:
    """Unwrap value-head / PEFT shells to the HuggingFace causal LM."""
    causal = model
    pretrained = getattr(causal, "pretrained_model", None)
    if isinstance(pretrained, nn.Module):
        causal = pretrained
    get_base = getattr(causal, "get_base_model", None)
    if callable(get_base):
        base = get_base()
        if isinstance(base, nn.Module):
            causal = base
    wrapped = getattr(causal, "base_model", None)
    if isinstance(wrapped, nn.Module):
        causal = wrapped
    inner = getattr(causal, "model", None)
    if isinstance(inner, nn.Module) and (
        hasattr(inner, "lm_head")
        or hasattr(inner, "embed_out")
        or hasattr(inner, "model")
    ):
        causal = inner
    children = dict(causal.named_children())
    language_tower = children.get("language_model")
    if isinstance(language_tower, nn.Module) and (
        hasattr(language_tower, "lm_head")
        or hasattr(language_tower, "embed_out")
        or hasattr(language_tower, "backbone")
        or hasattr(language_tower, "model")
    ):
        return language_tower
    return causal


def _language_model(causal: nn.Module) -> nn.Module | None:
    """Return the transformer body that owns ``embed_tokens`` / ``layers``."""
    children = dict(causal.named_children())
    language_tower = children.get("language_model")
    if isinstance(language_tower, nn.Module):
        causal = language_tower
    for attr in ("model", "backbone"):
        inner = getattr(causal, attr, None)
        if isinstance(inner, nn.Module) and (
            hasattr(inner, "embed_tokens")
            or hasattr(inner, "embeddings")
            or hasattr(inner, "layers")
        ):
            return inner
    if hasattr(causal, "embed_tokens") or hasattr(causal, "layers"):
        return causal
    return None


def _owned_parameters(module: nn.Module) -> set[nn.Parameter]:
    """Parameters of ``module`` not already owned by a child FSDP unit."""
    owned = set(module.parameters())
    for child in module.modules():
        if child is module or not isinstance(child, FSDPModule):
            continue
        owned.difference_update(child.parameters())
    return owned


def _persistent_params(module: nn.Module, threshold: int) -> set[nn.Parameter]:
    """Unsharded parameters: ``numel`` below ``threshold``."""
    if threshold <= 0:
        return set()
    return {param for param in _owned_parameters(module) if param.numel() < threshold}


def _cast_params(params: Iterable[nn.Parameter], dtype: torch.dtype) -> None:
    """Put unsharded parameters in the mixed-precision compute dtype."""
    for param in params:
        if param.dtype != dtype:
            param.data = param.data.to(dtype=dtype)


def _embed_params(model: nn.Module) -> set[nn.Parameter]:
    """Token-embedding parameters to leave replicated on the root unit.

    ``to_empty`` splits tied ``lm_head.weight`` from ``embed_tokens``. Both
    must be ignored or FSDP shards the head, then ``tie_weights()`` replaces
    that DTensor with the dense embed Parameter.
    """
    causal = _resolve_causal_lm(model)
    language = _language_model(causal)
    if language is None:
        return set()
    embed = getattr(language, "embed_tokens", None) or getattr(
        language, "embeddings", None
    )
    if embed is None:
        return set()
    ignored = set(embed.parameters())
    config = getattr(causal, "config", None)
    if config is None or not bool(getattr(config, "tie_word_embeddings", False)):
        return ignored
    lm_head = getattr(causal, "lm_head", None) or getattr(causal, "embed_out", None)
    if lm_head is not None:
        ignored.update(lm_head.parameters())
    return ignored


def _shard_unit(
    module: nn.Module,
    shard_kwargs: dict,
    persistence_threshold: int,
    extra_ignored: set[nn.Parameter] | None = None,
) -> None:
    """``fully_shard`` ``module``, leaving tiny owned parameters replicated."""
    ignored = _persistent_params(module, persistence_threshold)
    if extra_ignored:
        ignored = ignored | extra_ignored
    if ignored:
        mp_policy = shard_kwargs.get("mp_policy")
        param_dtype = getattr(mp_policy, "param_dtype", None)
        if param_dtype is not None:
            _cast_params(ignored, param_dtype)
        shard_kwargs = dict(shard_kwargs)
        shard_kwargs["ignored_params"] = ignored
    fully_shard(module, **shard_kwargs)


def _shard_embed_and_lm_head(
    model: nn.Module, shard_kwargs: dict, persistence_threshold: int
) -> None:
    """Replicate token embeddings; shard an untied ``lm_head`` as its own unit.

    Embeddings stay dense on every rank. An untied head is sharded alone
    with ``reshard_after_forward=False`` so the last all-gather stays live
    into the unembedding. Tied embeddings skip the head unit so tying
    stays intact.
    """
    causal = _resolve_causal_lm(model)
    config = getattr(causal, "config", None)
    if config is not None and bool(getattr(config, "tie_word_embeddings", False)):
        return

    lm_head = getattr(causal, "lm_head", None) or getattr(causal, "embed_out", None)
    if lm_head is not None:
        head_kwargs = dict(shard_kwargs)
        head_kwargs["reshard_after_forward"] = False
        _shard_unit(lm_head, head_kwargs, persistence_threshold)


def _set_prefetch(
    model: nn.Module,
    block_units: Sequence[nn.Module],
    prefetch_units: int = 1,
) -> None:
    """Overlap neighbour FSDP all-gathers with the current unit's compute.

    Dense path only: embed → first block, consecutive transformer blocks,
    last block → ``lm_head``. Forward prefetches the next ``prefetch_units``
    modules. Backward prefetches the previous ``prefetch_units`` so layer
    i-1 gathers while layer i runs backward. ``block_units`` are the
    modules just ``fully_shard``ed (the checkpoint wrappers when
    activation checkpointing is on). Walking ``_transformer_blocks`` after
    wrap would also see     inner decoder layers.
    """
    units: list[Any] = []
    causal = _resolve_causal_lm(model)
    language = _language_model(causal)
    if language is not None:
        embed = getattr(language, "embed_tokens", None) or getattr(
            language, "embeddings", None
        )
        if isinstance(embed, FSDPModule):
            units.append(embed)
    units.extend(unit for unit in block_units if isinstance(unit, FSDPModule))
    lm_head = getattr(causal, "lm_head", None) or getattr(causal, "embed_out", None)
    if isinstance(lm_head, FSDPModule):
        units.append(lm_head)

    for index, current in enumerate(units):
        nxt = units[index + 1 : index + 1 + prefetch_units]
        if nxt:
            current.set_modules_to_forward_prefetch(list(nxt))
        prev = units[max(0, index - prefetch_units) : index]
        if prev:
            current.set_modules_to_backward_prefetch(list(reversed(prev)))


def _inner_block_types(group: FSDPBlockGroup) -> set[object]:
    """``block_type`` of each inner decoder block in ``group``."""
    types: set[object] = set()
    for block in group.blocks:
        raw = _unwrap_checkpoint(block)
        types.add(getattr(raw, "block_type", None))
    return types


def _mixed_fsdp_groups(module: nn.Module) -> list[FSDPBlockGroup]:
    """Grouped wrap units whose inner blocks do not share one ``block_type``."""
    return [
        child
        for child in module.modules()
        if isinstance(child, FSDPBlockGroup) and len(_inner_block_types(child)) > 1
    ]


def _mixed_group_attention_mask(
    language: nn.Module,
    input_ids: torch.Tensor | None,
    inputs_embeds: torch.Tensor | None,
    position_ids: torch.Tensor | None,
    past_key_values: object | None,
    attention_mask: torch.Tensor | dict[str, object] | None,
) -> tuple[
    torch.Tensor | dict[str, object] | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    list[tuple[FSDPBlockGroup, object]],
]:
    """Rewrite mixed-type group ``block_type`` so HF forwards the full mask dict."""
    mixed = _mixed_fsdp_groups(language)
    mapping = attention_mask if isinstance(attention_mask, dict) else None
    restored: list[tuple[FSDPBlockGroup, object]] = []
    if mapping is None and mixed:
        needs_linear = any(
            "linear_attention" in _inner_block_types(group) for group in mixed
        )
        embeds = inputs_embeds
        if embeds is None and input_ids is not None:
            embed_fn = getattr(language, "embeddings", None) or getattr(
                language, "embed_tokens", None
            )
            if callable(embed_fn):
                embeds = embed_fn(input_ids)
        if needs_linear and embeds is not None:
            # Nemotron hybrid mask mapping; this architecture is not
            # loaded for the rest of this module.
            from agilerl.architectures.nemotron_h.mamba import (  # lazy import of optional architecture extra
                block_type_mask_mapping,
            )

            mapping, position_ids = block_type_mask_mapping(
                language,
                embeds=embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
            if inputs_embeds is None:
                inputs_embeds = embeds
                input_ids = None
    if mapping and mixed:
        mapping = dict(mapping)
        payload = dict(mapping)
        mapping["_agilerl_fsdp_group"] = payload
        attention_mask = mapping
        for group in mixed:
            restored.append((group, group.block_type))
            group.block_type = "_agilerl_fsdp_group"
    return attention_mask, input_ids, inputs_embeds, position_ids, restored


def _run_grouped_language_forward(
    language: nn.Module,
    original: Callable[..., object],
    input_ids: torch.Tensor | None,
    inputs_embeds: torch.Tensor | None,
    position_ids: torch.Tensor | None,
    past_key_values: object | None,
    use_cache: bool | None,
    attention_mask: torch.Tensor | dict[str, object] | None,
    kwargs: dict[str, object],
) -> object:
    """HF language forward with mixed-type FSDP group masks restored after."""
    (
        attention_mask,
        input_ids,
        inputs_embeds,
        position_ids,
        restored,
    ) = _mixed_group_attention_mask(
        language,
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        position_ids=position_ids,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
    )
    try:
        return original(
            language,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            attention_mask=attention_mask,
            **kwargs,
        )
    finally:
        for group, previous in restored:
            group.block_type = previous


def _install_grouped_mask_forward(language: nn.Module) -> None:
    """Pass the full mask dict into mixed-type ``FSDPBlockGroup`` units.

    HuggingFace does ``mapping.get(layer.block_type)`` before calling the layer.
    """
    cls = type(language)
    if class_is_patched(cls, "_agilerl_grouped_fsdp_forward_patched"):
        return
    original = cls.forward

    def wrapped(
        self: nn.Module,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: object | None = None,
        use_cache: bool | None = None,
        attention_mask: torch.Tensor | dict[str, object] | None = None,
        **kwargs: object,
    ) -> object:
        return _run_grouped_language_forward(
            self,
            original,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            attention_mask=attention_mask,
            kwargs=kwargs,
        )

    cls.forward = wrapped
    type.__setattr__(cls, "_agilerl_grouped_fsdp_forward_patched", True)


def apply_fsdp2(
    model: nn.Module,
    config: FSDPConfig | None = None,
    mesh: DeviceMesh | None = None,
    gradient_checkpointing: bool = False,
) -> nn.Module:
    """Shard ``model`` with FSDP2: blocks, embed/(untied) lm_head, root, prefetch.

    Parameters become DTensors in place, so any optimizer must be (re)built
    after this call. Callers must not pass a dense full-model replica on
    CUDA — use :func:`materialize_fsdp2_from_cpu_state` so weights stay on
    CPU/meta until only local shards are allocated on the compute device.

    When ``mesh`` is provided, ``fully_shard`` shards over that device mesh;
    when ``None``, FSDP uses the default process group (flat path).

    When ``gradient_checkpointing`` is on, each transformer block is wrapped
    with ``checkpoint_wrapper`` before ``fully_shard`` so the checkpoint
    boundary sits inside the FSDP unit.

    :param model: Model to shard (CPU or meta parameters).
    :type model: nn.Module
    :param config: Sharding settings; defaults to :class:`FSDPConfig`'s
        defaults.
    :type config: FSDPConfig | None
    :param mesh: Optional FSDP device mesh. Ignored when ``None``.
    :type mesh: DeviceMesh | None
    :param gradient_checkpointing: Wrap each transformer block with
        non-reentrant activation checkpointing before sharding.
    :type gradient_checkpointing: bool
    :return: The sharded model (same object).
    :rtype: nn.Module
    """
    if not is_distributed():
        msg = (
            "FSDP2 sharding requires an initialised process group. Launch "
            "with torchrun (or set RANK, WORLD_SIZE, MASTER_ADDR, and "
            "MASTER_PORT) so init_distributed() succeeds."
        )
        raise RuntimeError(msg)
    config = config or FSDPConfig()
    kwargs: dict = {
        "reshard_after_forward": config.reshard_after_forward,
        "mp_policy": MixedPrecisionPolicy(
            param_dtype=getattr(torch, config.param_dtype),
            reduce_dtype=getattr(torch, config.reduce_dtype),
        ),
    }
    if config.cpu_offload:
        kwargs["offload_policy"] = CPUOffloadPolicy()
    if mesh is not None:
        kwargs["mesh"] = mesh

    # Packed-expert skip lives in moe_lora; algorithms.core imports this
    # module via base, so the import stays in this function.
    from agilerl.algorithms.core.llm_ops.moe_lora import (  # cycle: algorithms.core imports this module via base
        _is_packed_experts_module,
    )

    wrap_units: list[nn.Module] = []
    for block in _transformer_blocks(model):
        if _is_packed_experts_module(block):
            continue
        unit = block
        if gradient_checkpointing:
            unit = checkpoint_wrapper(
                block,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                preserve_rng_state=False,
            )
            _replace_child(model, block, unit)
        wrap_units.append(unit)
    sharded_blocks = _group_transformer_units(
        model, wrap_units, config.wrap_every_n_blocks
    )
    if _mixed_fsdp_groups(model):
        language = _language_model(_resolve_causal_lm(model))
        if language is not None:
            _install_grouped_mask_forward(language)
    threshold = config.param_persistence_threshold
    for unit in sharded_blocks:
        _shard_unit(unit, kwargs, threshold)
    _shard_embed_and_lm_head(model, kwargs, threshold)
    _shard_unit(model, kwargs, threshold, extra_ignored=_embed_params(model))
    _set_prefetch(model, sharded_blocks, config.prefetch_units)
    # PEFT ``generate`` delegates to ``base_model.generate`` and never enters
    # the FSDP-rooted ``forward``, so remaining root shards stay DTensors
    # against plain ``input_ids``. Register ``generate`` so FSDP2 all-gathers
    # the same way as ``forward``. Register ``forward`` so a direct
    # ``module.forward(...)`` still runs root ``_pre_forward``.
    if hasattr(model, "generate"):
        register_fsdp_forward_method(model, "generate")
    register_fsdp_forward_method(model, "forward")
    return model


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


def _restore_after_to_empty(model: nn.Module) -> None:
    """Re-tie input/output embeddings, which ``to_empty`` unties.

    HuggingFace ``tie_weights`` on the causal LM also ties its submodules.
    """
    tie = getattr(_resolve_causal_lm(model), "tie_weights", None)
    if callable(tie):
        tie()


PEFT_BASE_PREFIX = "base_model.model."
LORA_MARKERS = ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B")


class SafetensorsSliceView(Protocol):
    def __getitem__(self, key: slice | tuple[slice, ...]) -> torch.Tensor: ...


class SafetensorsFileHandle(Protocol):
    def get_slice(self, key: str) -> SafetensorsSliceView: ...


def checkpoint_key_for_parameter(live_name: str) -> str:
    """Map a live FSDP parameter FQN to a HuggingFace safetensors key."""
    name = canonical_fsdp_param_fqn(live_name).removeprefix(PEFT_BASE_PREFIX)
    if name.endswith(".base_layer.weight"):
        return name[: -len(".base_layer.weight")] + ".weight"
    if name.endswith(".base_layer.bias"):
        return name[: -len(".base_layer.bias")] + ".bias"
    return name


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
) -> list[tuple[str, list[Any]]]:
    """Conversions registered for a module class or model type, under that module path.

    Longer paths come first so a nested module's mapping wins over its parent.
    """
    from transformers.conversion_mapping import get_checkpoint_conversion_mapping

    groups: list[tuple[str, list[Any]]] = []
    for name, module in model.named_modules():
        config = getattr(module, "config", None)
        model_type = getattr(config, "model_type", None)
        conversions = get_checkpoint_conversion_mapping(type(module).__name__)
        if not conversions and model_type:
            conversions = get_checkpoint_conversion_mapping(model_type)
        if not conversions:
            continue
        prefix = checkpoint_key_for_parameter(name) if name else ""
        groups.append((prefix, list(conversions)))
    groups.sort(key=lambda item: len(item[0]), reverse=True)
    return groups


def converted_checkpoint_key(
    key: str,
    transforms: Sequence[Any],
) -> tuple[str, int | None]:
    """Map a live parameter name back to the checkpoint key it was saved under.

    A transform that splits one source tensor records which piece this name is.

    :param key: Live parameter name.
    :type key: str
    :param transforms: Conversions registered for one module.
    :type transforms: Sequence[Any]
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
        operations = getattr(transform, "operations", ())
        if any(type(operation).__name__ == "Chunk" for operation in operations):
            split_index = list(transform.target_patterns).index(pattern)
        current = renamed
    return current, split_index


def _iter_converted_checkpoint_keys(
    key: str,
    groups: Sequence[tuple[str, Sequence[Any]]],
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
            # get_slice is a view of the open file.
            loaded.extend(
                (order[key], handle.get_slice(key)[:].clone()) for key in path_keys
            )
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


def _checkpoint_source_from_config(config: object) -> str | None:
    raw_path = getattr(config, "_name_or_path", None) or getattr(
        config, "name_or_path", None
    )
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
        config = getattr(current, "config", None)
        if config is not None:
            source = _checkpoint_source_from_config(config)
            if source is not None:
                return source
        current = _next_unwrap_module(current)
    msg = (
        "FSDP shard load requires a module config whose _name_or_path or "
        f"name_or_path resolves to {SAFE_WEIGHTS_NAME} or "
        f"{SAFE_WEIGHTS_INDEX_NAME} (local directory or Hugging Face Hub id)"
    )
    raise RuntimeError(msg)


def _tied_weight_checkpoint_targets(model: nn.Module) -> set[str]:
    causal = _resolve_causal_lm(model)
    tied = getattr(causal, "all_tied_weights_keys", None)
    if not isinstance(tied, dict):
        return set()
    return {key for key in tied if isinstance(key, str)}


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
            checkpoint_key = next((key for key in candidates if key in key_files), None)
            split_index: int | None = None
            if checkpoint_key is None:
                for candidate in candidates:
                    for converted, split in _iter_converted_checkpoint_keys(
                        candidate, transform_groups
                    ):
                        if converted in key_files and converted != candidate:
                            checkpoint_key = converted
                            split_index = split
                            break
                    if checkpoint_key is not None:
                        break
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
            if any(key in tied_targets for key in candidates):
                continue
            if has_language_tower and "language_model." not in candidates[0]:
                if _initialize_ignored_missing_parameter(model, live_name):
                    continue
                unmatched_outside_language.append(candidates[0])
                continue
            packed = _packed_checkpoint_keys(candidates, key_files)
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
            buffer_candidates = checkpoint_key_candidates(live_name)
            checkpoint_key = next(
                (key for key in buffer_candidates if key in key_files),
                None,
            )
            if checkpoint_key is None:
                for candidate in buffer_candidates:
                    for converted, _split in _iter_converted_checkpoint_keys(
                        candidate, transform_groups
                    ):
                        if converted in key_files and converted != candidate:
                            checkpoint_key = converted
                            break
                    if checkpoint_key is not None:
                        break
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
                        full = handle.get_slice(checkpoint_key)[:]
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
    _restore_after_to_empty(model)
    if cpu_state is None:
        _load_sharded_weights_from_safetensors(model)
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

# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""FSDP2 wrap, gather, and CPU-offload optimizer helpers."""

from __future__ import annotations

import copy
from collections.abc import Callable, Generator, Iterable, Sequence
from contextlib import _GeneratorContextManager, contextmanager
from typing import Any, cast, overload

import torch
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
            value = value.to(device=dest.device, dtype=dest.dtype)
            if isinstance(dest, DTensor):
                sharded = distribute_tensor(value, dest.device_mesh, dest.placements)
                dest.to_local().copy_(sharded.to_local())
            else:
                dest.data.copy_(value)
        for key, buf in model.named_buffers():
            value = _state_dict_value(state_dict, key)
            if value is None:
                continue
            buf.copy_(value.to(device=buf.device, dtype=buf.dtype))


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
            return base
    wrapped = getattr(causal, "base_model", None)
    if isinstance(wrapped, nn.Module):
        causal = wrapped
    inner = getattr(causal, "model", None)
    if isinstance(inner, nn.Module) and (
        hasattr(inner, "lm_head")
        or hasattr(inner, "embed_out")
        or hasattr(inner, "model")
    ):
        return inner
    return causal


def _language_model(causal: nn.Module) -> nn.Module | None:
    """Return the transformer body that owns ``embed_tokens`` / ``layers``."""
    inner = getattr(causal, "model", None)
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


def materialize_fsdp2_from_cpu_state(
    model: nn.Module,
    device: str | torch.device,
    config: FSDPConfig | None = None,
    mesh: DeviceMesh | None = None,
    gradient_checkpointing: bool = False,
) -> nn.Module:
    """Shard a CPU-resident model without placing a dense full replica on GPU.

    Captures a CPU state dict, moves parameters to meta, applies FSDP2,
    allocates empty sharded storages on ``device`` (or CPU when
    ``config.cpu_offload``), then scatters the CPU state into DTensor shards.

    :param model: Dense actor on CPU (base + LoRA already attached).
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
    set_full_model_state_dict(model, cpu_state, strict=True)
    _restore_nonpersistent_buffers(model, cpu_buffers)
    del cpu_state, cpu_buffers
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

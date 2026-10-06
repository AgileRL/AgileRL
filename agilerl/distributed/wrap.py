# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""FSDP2 wrap units, prefetch, mixed-mask forward, and block grouping."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

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
)

from agilerl.arena.models.fsdp import FSDPConfig
from agilerl.distributed.process import is_distributed
from agilerl.utils.patching import class_is_patched


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
    children = dict(causal.named_children())
    for attr in ("model", "backbone"):
        inner = children.get(attr)
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
    for module in nn.Module.modules(model):
        # PEFT __getattr__ forwards lm_head; ownership is the registered child.
        children = dict(module.named_children())
        head = children.get("lm_head")
        if not isinstance(head, nn.Module):
            head = children.get("embed_out")
        if isinstance(head, nn.Module):
            ignored.update(head.parameters())
            break
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

# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""FSDP2 sharding of transformer blocks, embeddings, and root; per-block compile."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from typing import Any, cast

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
from torch.distributed.tensor import DTensor

from agilerl.architectures.nemotron_h.mamba import block_type_mask_mapping
from agilerl.arena.models.fsdp import FSDPConfig
from agilerl.distributed.expert_parallel import (
    iter_packed_expert_modules,
    routed_counts_contexts,
)
from agilerl.distributed.process import is_distributed
from agilerl.distributed.tensor_parallel import set_tp_compute_dtype
from agilerl.lora.moe.layouts import is_packed_experts_module
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


def resolve_causal_lm(model: nn.Module) -> nn.Module:
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
    causal = resolve_causal_lm(model)
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
    keep_dtype: set[nn.Parameter] | None = None,
) -> None:
    """``fully_shard`` ``module``, leaving tiny owned parameters replicated.

    :param keep_dtype: Ignored parameters whose stored dtype is kept.
    """
    ignored = _persistent_params(module, persistence_threshold)
    if extra_ignored:
        ignored = ignored | (extra_ignored & _owned_parameters(module))
    if ignored:
        mp_policy = shard_kwargs.get("mp_policy")
        param_dtype = getattr(mp_policy, "param_dtype", None)
        if param_dtype is not None:
            _cast_params(ignored - (keep_dtype or set()), param_dtype)
        shard_kwargs = dict(shard_kwargs)
        shard_kwargs["ignored_params"] = ignored
    fully_shard(module, **shard_kwargs)


def _shard_embed_and_lm_head(
    model: nn.Module,
    shard_kwargs: dict,
    persistence_threshold: int,
    extra_ignored: set[nn.Parameter] | None = None,
    keep_dtype: set[nn.Parameter] | None = None,
) -> None:
    """Replicate token embeddings; shard an untied ``lm_head`` as its own unit.

    Embeddings stay dense on every rank. An untied head is sharded alone
    with ``reshard_after_forward=False`` so the last all-gather stays live
    into the unembedding. Tied embeddings skip the head unit so tying
    stays intact. A head whose parameters are already on the expert-parallel
    mesh stays there.
    """
    causal = resolve_causal_lm(model)
    config = getattr(causal, "config", None)
    if config is not None and bool(getattr(config, "tie_word_embeddings", False)):
        return

    lm_head = getattr(causal, "lm_head", None) or getattr(causal, "embed_out", None)
    if lm_head is None:
        return
    ignored = extra_ignored or set()
    owned = _owned_parameters(lm_head)
    if owned and owned <= ignored:
        return
    head_kwargs = dict(shard_kwargs)
    head_kwargs["reshard_after_forward"] = False
    _shard_unit(
        lm_head,
        head_kwargs,
        persistence_threshold,
        extra_ignored=ignored,
        keep_dtype=keep_dtype,
    )


def _prefetch_chains(
    model: nn.Module, block_units: Sequence[nn.Module]
) -> list[list[Any]]:
    """FSDP block units in forward order, one chain per parent module.

    Blocks of one parent (a decoder ``layers`` list, a vision ``blocks``
    stack) run back to back, so only siblings prefetch each other. A unit
    with no explicit backward prefetch falls back to FSDP's recorded
    post-forward order, which covers the jump between towers. Embed and
    ``lm_head`` join the language model's chain, or the last chain when no
    language model is found.
    """
    unit_ids = {id(unit) for unit in block_units if isinstance(unit, FSDPModule)}
    causal = resolve_causal_lm(model)
    language = _language_model(causal)
    language_ids = (
        {id(module) for module in language.modules()} if language is not None else set()
    )
    chains: list[list[Any]] = []
    language_chain: list[Any] | None = None
    for parent in model.modules():
        chain = [child for child in parent.children() if id(child) in unit_ids]
        if not chain:
            continue
        chains.append(chain)
        if language_chain is None and id(parent) in language_ids:
            language_chain = chain
    if language_chain is None:
        if not chains:
            chains.append([])
        language_chain = chains[-1]

    if language is not None:
        embed = getattr(language, "embed_tokens", None) or getattr(
            language, "embeddings", None
        )
        if isinstance(embed, FSDPModule):
            language_chain.insert(0, embed)
    lm_head = getattr(causal, "lm_head", None) or getattr(causal, "embed_out", None)
    if isinstance(lm_head, FSDPModule):
        language_chain.append(lm_head)
    return chains


def _set_prefetch(
    model: nn.Module,
    block_units: Sequence[nn.Module],
    forward_units: int = 1,
    backward_units: int = 1,
) -> None:
    """Overlap neighbour FSDP all-gathers with the current unit's compute.

    Forward prefetches the next ``forward_units`` units of the same chain
    (see :func:`_prefetch_chains`). Backward prefetches the previous
    ``backward_units`` so unit i-1 gathers while unit i runs backward.
    ``block_units`` are the modules just ``fully_shard``ed with the
    non-expert mesh (checkpoint wrappers when checkpointing is on), so
    every prefetch stays on that mesh: under HSDP each all-gather runs
    inside one shard group. Nested packed-expert units are never listed.
    """
    for units in _prefetch_chains(model, block_units):
        for index, current in enumerate(units):
            nxt = units[index + 1 : index + 1 + forward_units]
            if nxt:
                current.set_modules_to_forward_prefetch(list(nxt))
            prev = units[max(0, index - backward_units) : index]
            if prev:
                current.set_modules_to_backward_prefetch(list(reversed(prev)))


def _block_kind(block: nn.Module) -> str:
    """Kind of a transformer block for activation-checkpoint selection.

    :param block: Unwrapped transformer block.
    :type block: nn.Module
    :return: The block's ``block_type`` string (hybrid models name the
        mixer, e.g. ``linear_attention`` / ``full_attention`` / ``moe``), else
        its class name.
    :rtype: str
    """
    kind = getattr(block, "block_type", None)
    return kind if isinstance(kind, str) else type(block).__name__


def _blocks_to_checkpoint(blocks: Sequence[nn.Module], config: FSDPConfig) -> set[int]:
    """``id`` of each block that ``config``'s checkpoint policy wraps."""
    kinds = [_block_kind(block) for block in blocks]
    skipped = set(config.checkpoint_skip_layer_types)
    unknown = skipped - set(kinds)
    if unknown:
        msg = (
            f"FSDPConfig.checkpoint_skip_layer_types {sorted(unknown)} match no "
            f"transformer block; block kinds are {sorted(set(kinds))}"
        )
        raise ValueError(msg)
    eligible = [
        (block, kind)
        for block, kind in zip(blocks, kinds, strict=True)
        if kind not in skipped
    ]
    chosen = eligible[:: config.checkpoint_every_n_blocks]
    totals = Counter(kinds)
    wrapped = Counter(kind for _block, kind in chosen)
    logging.getLogger(__name__).info(
        "Activation checkpointing wraps %s",
        ", ".join(f"{kind} {wrapped[kind]}/{totals[kind]}" for kind in sorted(totals)),
    )
    return {id(block) for block, _kind in chosen}


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
    expert_mesh: DeviceMesh | None = None,
    gradient_checkpointing: bool = False,
) -> nn.Module:
    """Shard ``model`` with FSDP2: blocks, embed/(untied) lm_head, root, prefetch.

    Parameters become DTensors in place, so any optimizer must be (re)built
    after this call. Callers must not pass a dense full-model replica on
    CUDA — use :func:`materialize_fsdp2_from_cpu_state` so weights stay on
    CPU/meta until only local shards are allocated on the compute device.

    When ``mesh`` is provided, non-expert ``fully_shard`` uses that mesh;
    when ``None``, FSDP uses the default process group (flat path). Packed
    expert modules are sharded on ``expert_mesh`` when that is provided.

    When ``gradient_checkpointing`` is on, the transformer blocks chosen by
    ``config.checkpoint_skip_layer_types`` and
    ``config.checkpoint_every_n_blocks`` are wrapped with
    ``checkpoint_wrapper`` before ``fully_shard`` so the checkpoint boundary
    sits inside the FSDP unit.

    :param model: Model to shard (CPU or meta parameters).
    :type model: nn.Module
    :param config: Sharding settings; defaults to :class:`FSDPConfig`'s
        defaults.
    :type config: FSDPConfig | None
    :param mesh: Optional FSDP device mesh for non-expert units. Ignored
        when ``None``.
    :type mesh: DeviceMesh | None
    :param expert_mesh: Optional FSDP mesh for packed-expert modules
        (the leftover data-parallel axis). Ignored when ``None``.
    :type expert_mesh: DeviceMesh | None
    :param gradient_checkpointing: Wrap the transformer blocks selected by
        ``config`` with non-reentrant activation checkpointing before sharding.
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

    packed_experts = [expert for _name, expert in iter_packed_expert_modules(model)]
    # fully_shard on the EP group densifies packed experts to full-E.
    skip_expert_fsdp = False
    if config.ep > 1 and expert_mesh is None:
        skip_expert_fsdp = True
    elif expert_mesh is not None:
        size_fn = getattr(expert_mesh, "size", None)
        skip_expert_fsdp = callable(size_fn) and int(expert_mesh.size()) <= 1
        if not skip_expert_fsdp:
            expert_kwargs = dict(kwargs)
            expert_kwargs["mesh"] = expert_mesh
            # Packed experts take leftover-dp FSDP. LoRA parameters stay EP-only.
            # An EP owner's grad already sums its EP group, so the FSDP
            # reduce divides by the full world. PreMulSum is NCCL-only;
            # force SUM so the custom factor also works on gloo.
            for expert in packed_experts:
                _shard_unit(expert, expert_kwargs, config.param_persistence_threshold)
                unit = cast("FSDPModule", expert)
                unit.set_gradient_divide_factor(expert_mesh.size() * config.ep)
                unit.set_force_sum_reduction_for_comms(True)
    expert_params: set[nn.Parameter] = set()
    if skip_expert_fsdp:
        for expert in packed_experts:
            expert_params.update(expert.parameters())
    # EP / TP shards (including LoRA on PEFT wrapper links outside the packed
    # modules) are DTensors on their own mesh. No non-expert unit may manage
    # them: fully_shard would treat that mesh as TP and fail concatenating it
    # with its own. TP-replicated copies hold rows each TP rank reads in full.
    expert_params.update(
        param for param in model.parameters() if isinstance(param, DTensor)
    )
    expert_params.update(
        module.get_parameter(name)
        for module in model.modules()
        for name in getattr(module, "tp_replicated_params", ())
    )
    # TP params skip FSDP, so TP regions apply the mixed-precision policy.
    tp_trainable: set[nn.Parameter] = set()
    if config.tp > 1:
        tp_trainable = set_tp_compute_dtype(model, getattr(torch, config.param_dtype))

    blocks = [
        block
        for block in _transformer_blocks(model)
        if not is_packed_experts_module(block)
    ]
    checkpointed = (
        _blocks_to_checkpoint(blocks, config) if gradient_checkpointing else set()
    )
    wrap_units: list[nn.Module] = []
    for block in blocks:
        unit = block
        if id(block) in checkpointed:
            unit = checkpoint_wrapper(
                block,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                preserve_rng_state=False,
                context_fn=routed_counts_contexts,
            )
            _replace_child(model, block, unit)
        wrap_units.append(unit)
    sharded_blocks = _group_transformer_units(
        model, wrap_units, config.wrap_every_n_blocks
    )
    if _mixed_fsdp_groups(model):
        language = _language_model(resolve_causal_lm(model))
        if language is not None:
            _install_grouped_mask_forward(language)
    threshold = config.param_persistence_threshold
    for unit in sharded_blocks:
        _shard_unit(
            unit,
            kwargs,
            threshold,
            extra_ignored=expert_params,
            keep_dtype=tp_trainable,
        )
    _shard_embed_and_lm_head(
        model,
        kwargs,
        threshold,
        extra_ignored=expert_params,
        keep_dtype=tp_trainable,
    )
    _shard_unit(
        model,
        kwargs,
        threshold,
        extra_ignored=_embed_params(model) | expert_params,
        keep_dtype=tp_trainable,
    )
    _set_prefetch(
        model, sharded_blocks, config.prefetch_units, config.backward_prefetch_units
    )
    # PEFT ``generate`` delegates to ``base_model.generate`` and never enters
    # the FSDP-rooted ``forward``, so remaining root shards stay DTensors
    # against plain ``input_ids``. Register ``generate`` so FSDP2 all-gathers
    # the same way as ``forward``. Register ``forward`` so a direct
    # ``module.forward(...)`` still runs root ``_pre_forward``.
    if hasattr(model, "generate"):
        register_fsdp_forward_method(model, "generate")
    register_fsdp_forward_method(model, "forward")
    return model

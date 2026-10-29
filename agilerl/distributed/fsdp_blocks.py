# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""FSDP2 sharding of transformer blocks, embeddings, and root; per-block compile."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import ExitStack, contextmanager
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
from torch.utils.checkpoint import (
    CheckpointPolicy,
    SelectiveCheckpointContext,
    create_selective_checkpoint_contexts,
)
from torch.utils.hooks import RemovableHandle

from agilerl.architectures.moe_router import patch_moe_routers
from agilerl.arena.models.fsdp import (
    PARAM_PERSISTENCE_THRESHOLD,
    PREFETCH_UNITS,
    FSDPConfig,
)
from agilerl.distributed.expert_parallel import (
    iter_packed_expert_modules,
    routed_counts_contexts,
)
from agilerl.distributed.process import is_distributed
from agilerl.distributed.tensor_parallel import set_tp_compute_dtype
from agilerl.lora.moe.layouts import is_packed_experts_module
from agilerl.utils.patching import library_shim


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


class UnforwardedFSDPUnits:
    """FSDP2 units that have not run a forward since this tracker was built."""

    def __init__(self, modules: Iterable[nn.Module]) -> None:
        """Track the FSDP2 units in ``modules`` until each one's first forward.

        :param modules: Modules to scan; those that are not FSDP2 units are skipped.
        :type modules: Iterable[nn.Module]
        """
        self.units: dict[int, FSDPModule] = {}
        self.handles: dict[int, RemovableHandle] = {}
        for module in modules:
            if not isinstance(module, FSDPModule):
                continue
            self.units[id(module)] = module
            self.handles[id(module)] = module.register_forward_pre_hook(
                self._mark_forwarded
            )

    def _mark_forwarded(self, module: nn.Module, _args: tuple[Any, ...]) -> None:
        """Stop tracking ``module`` once it runs a forward."""
        self.units.pop(id(module))
        self.handles.pop(id(module)).remove()

    def __iter__(self) -> Iterator[FSDPModule]:
        return iter(self.units.values())


def _persistent_params(module: nn.Module) -> set[nn.Parameter]:
    """Unsharded parameters: ``numel`` below ``PARAM_PERSISTENCE_THRESHOLD``."""
    return {
        param
        for param in _owned_parameters(module)
        if param.numel() < PARAM_PERSISTENCE_THRESHOLD
    }


def _cast_params(params: Iterable[nn.Parameter], dtype: torch.dtype) -> None:
    """Put unsharded parameters in the mixed-precision compute dtype."""
    for param in params:
        if param.dtype != dtype:
            param.data = param.data.to(dtype=dtype)


def _ple_embed(language: nn.Module | None) -> nn.Module | None:
    """Packed per-layer embedding table on the language body, if present."""
    if language is None:
        return None
    module = getattr(language, "embed_tokens_per_layer", None)
    return module if isinstance(module, nn.Module) else None


def _embed_params(model: nn.Module) -> set[nn.Parameter]:
    """Token-embedding parameters to leave replicated on the root unit.

    ``to_empty`` splits tied ``lm_head.weight`` from ``embed_tokens``. Both
    must be ignored or FSDP shards the head, then ``tie_weights()`` replaces
    that DTensor with the dense embed Parameter. Per-layer embeddings are a
    separate FSDP unit, not ignored here.
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
    extra_ignored: set[nn.Parameter] | None = None,
    keep_dtype: set[nn.Parameter] | None = None,
) -> None:
    """``fully_shard`` ``module``, leaving tiny owned parameters replicated.

    :param keep_dtype: Ignored parameters whose stored dtype is kept.
    """
    ignored = _persistent_params(module)
    owned = _owned_parameters(module)
    if extra_ignored:
        ignored = ignored | (extra_ignored & owned)
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
        extra_ignored=ignored,
        keep_dtype=keep_dtype,
    )


def _shard_ple_embeddings(
    model: nn.Module,
    shard_kwargs: dict,
    extra_ignored: set[nn.Parameter] | None = None,
    keep_dtype: set[nn.Parameter] | None = None,
) -> None:
    """Shard packed per-layer embeddings as their own FSDP unit.

    Gemma 4 PLE is vocab * layers * ple_dim (billions of elements on E4B).
    A dedicated unit all-gathers that table without mixing it into the root
    leftover gather.
    """
    ple = _ple_embed(_language_model(resolve_causal_lm(model)))
    if ple is None:
        return
    _shard_unit(
        ple,
        shard_kwargs,
        extra_ignored=extra_ignored,
        keep_dtype=keep_dtype,
    )


def _prefetch_chains(
    model: nn.Module, block_units: Sequence[nn.Module]
) -> list[list[Any]]:
    """FSDP block units in forward order, one chain per parent module.

    Blocks of one parent (a decoder ``layers`` list, a vision ``blocks``
    stack) run back to back, so only siblings prefetch each other. A unit
    with no explicit backward prefetch falls back to FSDP's recorded
    post-forward order, which covers the jump between towers. Embed,
    per-layer embeddings, and ``lm_head`` join the language model's chain,
    or the last chain when no language model is found.
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
        insert_at = 0
        if isinstance(embed, FSDPModule):
            language_chain.insert(0, embed)
            insert_at = 1
        ple = _ple_embed(language)
        if isinstance(ple, FSDPModule):
            language_chain.insert(insert_at, ple)
    lm_head = getattr(causal, "lm_head", None) or getattr(causal, "embed_out", None)
    if isinstance(lm_head, FSDPModule):
        language_chain.append(lm_head)
    return chains


def _set_prefetch(model: nn.Module, block_units: Sequence[nn.Module]) -> None:
    """Overlap neighbour FSDP all-gathers with the current unit's compute.

    Forward prefetches the next ``PREFETCH_UNITS`` units of the same chain
    (see :func:`_prefetch_chains`). Backward prefetches the previous
    ``PREFETCH_UNITS`` so unit i-1 gathers while unit i runs backward.
    ``block_units`` are the modules just ``fully_shard``ed with the
    non-expert mesh (checkpoint wrappers when checkpointing is on), so
    every prefetch stays on that mesh: under HSDP each all-gather runs
    inside one shard group. Nested packed-expert units are never listed.
    """
    for units in _prefetch_chains(model, block_units):
        for index, current in enumerate(units):
            nxt = units[index + 1 : index + 1 + PREFETCH_UNITS]
            if nxt:
                current.set_modules_to_forward_prefetch(list(nxt))
            prev = units[max(0, index - PREFETCH_UNITS) : index]
            if prev:
                current.set_modules_to_backward_prefetch(list(reversed(prev)))


# Resolved once: the routing policy runs on every op of a routed block.
TOPK_OP = torch.ops.aten.topk
TO_COPY_OP = torch.ops.aten._to_copy


def _save_routing_policy(
    _context: SelectiveCheckpointContext,
    op: torch._ops.OpOverload,
    *_args: Any,
    **kwargs: Any,
) -> CheckpointPolicy:
    """Keep router ``topk`` and device-to-host copies from forward; recompute the rest.

    A recomputed router can pick other experts, which changes dispatch shapes
    and the all-to-all splits every expert-parallel peer expects. Saved host
    copies spare the recompute a host sync each.

    Saved outputs replay in forward order per op, so a saved op must run the
    same number of times in the recompute.
    """
    packet = op.overloadpacket
    if packet is TOPK_OP:
        return CheckpointPolicy.MUST_SAVE
    if packet is not TO_COPY_OP:
        return CheckpointPolicy.PREFER_RECOMPUTE
    device = kwargs.get("device")
    if isinstance(device, torch.device) and device.type == "cpu":
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE


@contextmanager
def _together(*cms: Any) -> Iterator[None]:
    with ExitStack() as stack:
        for cm in cms:
            stack.enter_context(cm)
        yield


@library_shim("torch", "==2.13.*")
def _block_checkpoint_contexts(save_router: bool) -> Callable[[], tuple[Any, Any]]:
    """EP count replay, plus router topk save on packed-expert blocks."""

    def context_fn() -> tuple[Any, Any]:
        routing_fwd, routing_recompute = routed_counts_contexts()
        if not save_router:
            return routing_fwd, routing_recompute
        save_fwd, save_recompute = create_selective_checkpoint_contexts(
            _save_routing_policy
        )
        return (
            _together(save_fwd, routing_fwd),
            _together(save_recompute, routing_recompute),
        )

    return context_fn


def apply_fsdp2(
    model: nn.Module,
    config: FSDPConfig | None = None,
    mesh: DeviceMesh | None = None,
    expert_mesh: DeviceMesh | None = None,
    gradient_checkpointing: bool = False,
) -> nn.Module:
    """Shard ``model`` with FSDP2: blocks, PLE, embed/(untied) lm_head, root, prefetch.

    Parameters become DTensors in place, so any optimizer must be (re)built
    after this call. Callers must not pass a dense full-model replica on
    CUDA — use :func:`materialize_fsdp2_from_cpu_state` so weights stay on
    CPU/meta until only local shards are allocated on the compute device.

    When ``mesh`` is provided, non-expert ``fully_shard`` uses that mesh;
    when ``None``, FSDP uses the default process group (flat path). Packed
    expert modules are sharded on ``expert_mesh`` when that is provided.

    Each transformer block is one FSDP unit. When ``gradient_checkpointing``
    is on, every block is wrapped with ``checkpoint_wrapper`` before
    ``fully_shard`` so the checkpoint boundary sits inside the FSDP unit. Blocks holding packed experts reuse their
    forward routing in the recompute; their listed routers
    (:data:`~agilerl.architectures.moe_router.ROUTER_FORWARDS`) get
    out-of-place top-k renormalization.

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
    :param gradient_checkpointing: Wrap every transformer block with
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
                _shard_unit(expert, expert_kwargs)
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
    wrap_units: list[nn.Module] = []
    for block in blocks:
        unit = block
        if gradient_checkpointing:
            save_router = any(
                is_packed_experts_module(module) for module in block.modules()
            )
            if save_router:
                patch_moe_routers(block)
            unit = checkpoint_wrapper(
                block,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                preserve_rng_state=False,
                context_fn=_block_checkpoint_contexts(save_router),
            )
            _replace_child(model, block, unit)
        wrap_units.append(unit)
    for unit in wrap_units:
        _shard_unit(
            unit,
            kwargs,
            extra_ignored=expert_params,
            keep_dtype=tp_trainable,
        )
    _shard_embed_and_lm_head(
        model,
        kwargs,
        extra_ignored=expert_params,
        keep_dtype=tp_trainable,
    )
    _shard_ple_embeddings(
        model,
        kwargs,
        extra_ignored=expert_params,
        keep_dtype=tp_trainable,
    )
    _shard_unit(
        model,
        kwargs,
        extra_ignored=_embed_params(model) | expert_params,
        keep_dtype=tp_trainable,
    )
    _set_prefetch(model, wrap_units)
    # PEFT ``generate`` delegates to ``base_model.generate`` and never enters
    # the FSDP-rooted ``forward``, so remaining root shards stay DTensors
    # against plain ``input_ids``. Register ``generate`` so FSDP2 all-gathers
    # the same way as ``forward``. Register ``forward`` so a direct
    # ``module.forward(...)`` still runs root ``_pre_forward``.
    if hasattr(model, "generate"):
        register_fsdp_forward_method(model, "generate")
    register_fsdp_forward_method(model, "forward")
    return model

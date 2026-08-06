# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Fused multi-adapter LoRA forward pass.

Runs several LoRA adapters (e.g. actor + critic) in one forward pass by
routing each batch row to an adapter: the frozen base layer runs once over
the full batch and each adapter's low-rank delta is added to its rows only.
This removes adapter switching from ``learn()``, which would otherwise
desynchronise gradient-checkpoint recomputation.

Rows routed to the same adapter form contiguous runs, so each adapter's
rows are sliced with ``narrow`` and its delta added out of place. Adding out
of place (rather than accumulating into the base output) is what lets this
work unchanged on a quantized (bitsandbytes) base, whose forward returns a
view of a custom autograd Function that autograd forbids editing in place.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any
from weakref import WeakKeyDictionary

import torch
import torch.nn as nn
from peft.tuners.lora.layer import LoraLayer, ParamWrapper

# Remembers each model's LoRA layers so we don't rescan on every call.
# Weak keys: caching a model here never stops it being garbage-collected.
_LORA_LAYER_CACHE: WeakKeyDictionary[nn.Module, list[LoraLayer]] = WeakKeyDictionary()

# Which adapter each layer currently routes to (kept off the layer so it stays typed).
# Weak keys: tracking a layer here never stops it being garbage-collected.
_ROUTING_STATE: WeakKeyDictionary[LoraLayer, list[str] | None] = WeakKeyDictionary()


def uniform_routed_adapter(layer: LoraLayer) -> str | None:
    """The single adapter the active routing assigns *layer*, or ``None`` when routing is inactive.

    Parameter-level LoRA applies its delta to whole parameters (and packed
    experts see rows grouped by expert, not by sample), so only uniform
    routings are computable there; mixed routings raise.
    """
    routing = _ROUTING_STATE.get(layer)
    if routing is None:
        return None
    names = set(routing)
    if len(names) > 1:
        msg = (
            "Fused multi-adapter routing is not supported on parameter-level "
            "LoRA layers (LoraConfig.target_parameters). Use a single-adapter "
            "configuration (e.g. use_separate_reference_adapter=False, no "
            "value head)."
        )
        raise RuntimeError(msg)
    return next(iter(names))


def _handles_own_routing(module: LoraLayer) -> bool:
    """Whether *module*'s class forward consults the routing state itself."""
    return getattr(module, "_self_routed_lora", False)


def _is_routing_managed(module: LoraLayer) -> bool:
    """Whether *module* will honor fused routing (patched or self-routing)."""
    return _is_routed_layer(module) or _handles_own_routing(module)


def _lora_delta(layer: LoraLayer, adapter: str, rows: torch.Tensor) -> torch.Tensor:
    """One adapter's low-rank delta for a slice of input rows."""
    lora_a = layer.lora_A[adapter]
    # lora_A is an nn.ModuleDict, so its members' `.weight` resolves through
    # nn.Module.__getattr__ as the loose Tensor | Module; narrow it to Tensor.
    lora_a_weight = lora_a.weight
    assert isinstance(lora_a_weight, torch.Tensor)
    rows = layer.lora_dropout[adapter](
        layer._cast_input_dtype(rows, lora_a_weight.dtype)
    )
    return layer.lora_B[adapter](lora_a(rows)) * layer.scaling[adapter]


def _needs_peft_mixed_forward(layer: LoraLayer, routing: Sequence[str]) -> bool:
    """Whether any routed adapter needs PEFT's own mixed-batch forward.

    The sliced path computes the standard linear delta; embedding adapters
    and LoRA variants (e.g. aLoRA) use different maths, so those layers
    delegate to ``LoraLayer._mixed_batch_forward``.
    """
    embedding = getattr(layer, "lora_embedding_A", None) or {}
    variants = getattr(layer, "lora_variant", None) or {}
    return any(name in embedding or name in variants for name in set(routing))


def _routed_forward(
    layer: LoraLayer,
    original_forward: Callable[..., torch.Tensor],
    x: torch.Tensor,
    *forward_args: Any,
    **forward_kwargs: Any,
) -> torch.Tensor:
    """Replacement ``forward`` installed on patched LoRA layers.

    ``routing`` names the adapter for each batch row (e.g. ``["actor"] * B +
    ["critic"] * B``). Same-adapter rows are contiguous, so we run the base
    layer once and add each adapter's delta to its slice — walking the
    contiguous runs with :func:`itertools.groupby`. ``"__base__"`` and
    adapters that don't wrap this layer leave their rows on the base output,
    matching PEFT's mixed-batch behaviour. Falls back to the layer's ordinary
    forward while routing is unset.
    """
    routing = _ROUTING_STATE.get(layer)
    if routing is None:
        return original_forward(layer, x, *forward_args, **forward_kwargs)

    # Layers that flatten (batch, seq, hidden) -> (batch * seq, hidden) before
    # their linears (OPT's MLP, MoE experts) show seq rows per routed sample.
    # The flatten is row-major, so each run just covers ``factor`` times as
    # many contiguous rows.
    factor, remainder = divmod(x.shape[0], len(routing))
    if remainder:
        msg = (
            f"Fused adapter routing covers {len(routing)} rows but the layer "
            f"input's leading dimension is {x.shape[0]}."
        )
        raise ValueError(msg)

    if _needs_peft_mixed_forward(layer, routing):
        names = [name for name in routing for _ in range(factor)]
        return layer._mixed_batch_forward(
            x, *forward_args, adapter_names=names, **forward_kwargs
        )

    base_out = layer.base_layer(x, *forward_args, **forward_kwargs)

    pieces = []
    start = 0
    for name, run in itertools.groupby(routing):
        n = sum(1 for _ in run) * factor
        rows = base_out.narrow(0, start, n)
        if name != "__base__" and name in layer.lora_A:
            rows = rows + _lora_delta(layer, name, x.narrow(0, start, n)).to(rows.dtype)
        pieces.append(rows)
        start += n
    return pieces[0] if len(pieces) == 1 else torch.cat(pieces)


def _param_wrapper_routed_forward(
    layer: LoraLayer,
    original_forward: Callable[..., torch.Tensor],
    x: torch.Tensor,
    *forward_args: Any,
    **forward_kwargs: Any,
) -> torch.Tensor:
    """Replacement ``forward`` for parameter-level LoRA wrappers under fused routing.

    The routed adapter runs the ordinary forward; ``"__base__"`` (or a
    foreign adapter) runs with adapters disabled.
    """
    name = uniform_routed_adapter(layer)
    if name is None:
        return original_forward(layer, x, *forward_args, **forward_kwargs)
    if name in layer.lora_A:
        if list(layer.active_adapters) != [name]:
            msg = (
                f"Fused routing requested adapter {name!r} on a "
                "parameter-level LoRA wrapper whose active adapters are "
                f"{list(layer.active_adapters)}; set the adapter before "
                "routing."
            )
            raise RuntimeError(msg)
        return original_forward(layer, x, *forward_args, **forward_kwargs)
    previously_disabled = layer.disable_adapters
    layer.enable_adapters(False)
    try:
        return original_forward(layer, x, *forward_args, **forward_kwargs)
    finally:
        if not previously_disabled:
            layer.enable_adapters(True)


def _is_routed_layer(module: LoraLayer) -> bool:
    """Whether *module* has the fused-routing ``forward`` installed."""
    fwd = module.__dict__.get("forward")
    return isinstance(fwd, partial) and fwd.func in (
        _routed_forward,
        _param_wrapper_routed_forward,
    )


def adapter_aligned_chunks(
    routing: Sequence[str], batch_size: int
) -> list[tuple[int, int]]:
    """Micro-batch ``(start, end)`` spans of at most *batch_size* rows that never straddle an adapter run."""
    chunks: list[tuple[int, int]] = []
    run_start = 0
    for _, run in itertools.groupby(routing):
        run_len = sum(1 for _ in run)
        chunks.extend(
            (start, min(start + batch_size, run_start + run_len))
            for start in range(run_start, run_start + run_len, batch_size)
        )
        run_start += run_len
    return chunks


def _store_layer_cache(model: nn.Module, layers: list[LoraLayer]) -> None:
    _LORA_LAYER_CACHE[model] = layers


def get_cached_lora_layers(model: nn.Module) -> list[LoraLayer]:
    """All ``LoraLayer`` modules under *model*, cached after the first traversal.

    :param model: A ``PeftModel`` or any module containing ``LoraLayer`` s.
    :return: The LoRA layers under *model*.
    """
    cached = _LORA_LAYER_CACHE.get(model)
    if cached is not None:
        return cached
    # nn.Module.modules rather than model.modules: EvolvableModule overrides
    # modules() to return only evolvable children, which excludes LoRA layers.
    layers: list[LoraLayer] = [
        m for m in nn.Module.modules(model) if isinstance(m, LoraLayer)
    ]
    _store_layer_cache(model, layers)
    return layers


def _validate_routing(layers: list[LoraLayer], routing: Sequence[str]) -> None:
    """Reject routings PEFT would compute silently wrongly (unknown names)
    or that the fused forward cannot compute (merged weights, DoRA).

    :param layers: All LoRA layers under the model.
    :param routing: Adapter name per batch row.
    :raises RuntimeError: If any layer has merged adapters.
    :raises ValueError: If any adapter name is unknown or a DoRA adapter.
    """
    requested = set(routing) - {"__base__"}
    available: set[str] = set()
    for module in layers:
        if getattr(module, "merged", False):
            msg = (
                "Cannot use fused adapter routing while adapters are merged "
                "into the base weights; unmerge them first."
            )
            raise RuntimeError(msg)
        use_dora = getattr(module, "use_dora", None) or {}
        dora = sorted(name for name in requested if use_dora.get(name))
        if dora:
            msg = f"Fused adapter routing does not support DoRA adapters: {dora}."
            raise ValueError(msg)
        for attr in getattr(module, "adapter_layer_names", ()):
            container = getattr(module, attr, None)
            if container is not None:
                available.update(container.keys())
    # Best-effort: when no adapter containers are discoverable (e.g. test
    # doubles), routing is applied without name validation.
    if available and not requested <= available:
        unknown = sorted(requested - available)
        msg = (
            f"Unknown adapter name(s) in fused routing: {unknown}. Known "
            f"adapters: {sorted(available)} (plus '__base__' for "
            "base-only rows)."
        )
        raise ValueError(msg)


def patch_lora_for_fused_forward(model: nn.Module) -> None:
    """Swap every LoRA layer's ``forward`` for the fused-routing forward.

    Routing starts inactive, so the patched layers behave exactly as before
    until ``set_fused_adapter_routing`` is called. Idempotent — call again
    after adding adapters that wrap new modules; only the new layers are
    patched and the cached layer list is refreshed either way.

    :param model: A ``PeftModel`` or any module containing ``LoraLayer`` s.
    """
    layers: list[LoraLayer] = []
    for module in nn.Module.modules(model):
        if not isinstance(module, LoraLayer):
            continue
        layers.append(module)
        if not _is_routing_managed(module):
            routed = (
                _param_wrapper_routed_forward
                if isinstance(module, ParamWrapper)
                else _routed_forward
            )
            module.forward = partial(routed, module, type(module).forward)
    _store_layer_cache(model, layers)


def unpatch_lora_for_fused_forward(model: nn.Module) -> None:
    """Restore every LoRA layer's original ``forward`` and drop routing state.

    After this, ``set_fused_adapter_routing`` raises until the model is
    patched again.

    :param model: The model to release from fused-routing control.
    """
    for module in get_cached_lora_layers(model):
        if _is_routed_layer(module):
            # Drop the monkeypatched instance forward, restoring the class one.
            module.__dict__.pop("forward", None)
        _ROUTING_STATE.pop(module, None)
    _LORA_LAYER_CACHE.pop(model, None)


def set_fused_adapter_routing(model: nn.Module, routing: Sequence[str]) -> None:
    """Route each row of the next forward's batch through its own adapter.

    Routing stays active — including for gradient-checkpoint recomputation
    during ``backward()`` — until ``unset_fused_adapter_routing`` is called.

    :param model: The patched model whose LoRA layers should route rows.
    :param routing: Adapter name per batch row, e.g. ``["actor"] * B +
        ["critic"] * B``. ``"__base__"`` runs a row through the frozen base
        weights with no delta.
    :raises RuntimeError: If LoRA layers are unpatched (the routing would be
        silently ignored) or have adapters merged into the base weights.
    :raises ValueError: If *routing* is empty, names an unknown adapter, or
        names a DoRA adapter (unsupported, as in PEFT's mixed-batch forward).
    """
    layers = get_cached_lora_layers(model)
    if not layers:
        # Plain base model (no adapters) or PEFT not installed: nothing to
        # route, and the unfused forward is already correct.
        return
    if any(not _is_routing_managed(m) for m in layers):
        msg = (
            "set_fused_adapter_routing called on a model with LoRA layers that "
            "have no fused-routing forward; the routing would be silently "
            "ignored. Call patch_lora_for_fused_forward(model) first (again, "
            "if adapters were added after the last call)."
        )
        raise RuntimeError(msg)

    routing = list(routing)
    if not routing:
        msg = "Fused adapter routing must name at least one row."
        raise ValueError(msg)
    _validate_routing(layers, routing)
    for module in layers:
        _ROUTING_STATE[module] = routing


def unset_fused_adapter_routing(model: nn.Module) -> None:
    """Deactivate fused routing, restoring standard single-adapter forward.

    :param model: The model whose LoRA layers should clear fused routing.
    """
    for module in get_cached_lora_layers(model):
        if _is_routing_managed(module):
            _ROUTING_STATE[module] = None

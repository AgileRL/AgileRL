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
import torch.nn.functional as F
from peft.tuners.lora.layer import LoraLayer, ParamWrapper

from agilerl.utils.patching import library_shim

# Remembers each model's LoRA layers so we don't rescan on every call.
# Weak keys: caching a model here never stops it being garbage-collected.
LORA_LAYER_CACHE: WeakKeyDictionary[nn.Module, list[LoraLayer]] = WeakKeyDictionary()

# Which adapter each layer currently routes to (kept off the layer so it stays typed).
# Weak keys: tracking a layer here never stops it being garbage-collected.
ROUTING_STATE: WeakKeyDictionary[nn.Module | LoraLayer, list[str] | None] = (
    WeakKeyDictionary()
)


def uniform_routed_adapter(layer: LoraLayer) -> str | None:
    """The single adapter the active routing assigns *layer*, or ``None`` when routing is inactive.

    Default ``ParamWrapper`` applies a delta to a whole parameter, so mixed
    routings raise. Expert wrappers that set ``_self_routed_lora`` apply
    per-token masks in their own forward.

    :param layer: LoRA layer to check.
    :type layer: LoraLayer
    :return: The routed adapter name, or ``None``.
    :rtype: str | None
    """
    routing = ROUTING_STATE.get(layer)
    if routing is None:
        return None
    names = set(routing)
    if len(names) > 1:
        msg = (
            "Fused multi-adapter routing is not supported on parameter-level "
            "LoRA layers (LoraConfig.target_parameters) that do not implement "
            "per-token adapter masks."
        )
        raise RuntimeError(msg)
    return next(iter(names))


def _handles_own_routing(module: LoraLayer) -> bool:
    """Whether *module*'s class forward consults the routing state itself."""
    return getattr(module, "_self_routed_lora", False)


def _is_routing_managed(module: LoraLayer) -> bool:
    """Whether *module* will honor fused routing (patched or self-routing)."""
    return _is_routed_layer(module) or _handles_own_routing(module)


def _has_stock_linear_base(layer: LoraLayer) -> bool:
    """Whether *layer* wraps a base whose output the LoRA delta may overwrite.

    On 2-D input a stock ``nn.Linear`` forward returns a fresh tensor whose
    backward saves only the input and weight. Quantized (bitsandbytes) and
    other custom base layers may return views autograd forbids editing.
    """
    return type(layer.base_layer).forward is nn.Linear.forward


def _apply_lora_factor(
    layer: LoraLayer, factor: nn.Module, x: torch.Tensor
) -> torch.Tensor:
    """One LoRA factor (``lora_A`` or ``lora_B``) applied to *x* in *x*'s dtype.

    A linear factor casts its weight to *x*'s dtype in the forward, so fp32
    adapters on a bf16 base multiply in bf16 while autograd still accumulates
    fp32 weight gradients. FSDP's mixed-precision policy gives the same
    compute dtype by casting the parameters. Other factors take PEFT's cast
    of *x* to the weight dtype.

    :param layer: LoRA layer holding the factor.
    :param factor: The factor module.
    :param x: Factor input.
    :return: The factor output.
    """
    weight = factor.weight
    assert isinstance(weight, torch.Tensor)
    if not isinstance(factor, nn.Linear):
        return factor(layer._cast_input_dtype(x, weight.dtype))
    bias = None if factor.bias is None else factor.bias.to(x.dtype)
    return F.linear(x, weight.to(x.dtype), bias)


def _add_lora_delta(
    layer: LoraLayer,
    adapter: str,
    base_rows: torch.Tensor,
    rows: torch.Tensor,
    in_place: bool,
) -> torch.Tensor:
    """``base_rows`` plus one adapter's low-rank delta for the matching input rows.

    The delta runs in the dtype of the layer input (:func:`_apply_lora_factor`).
    A bias-free linear ``lora_B`` whose output dtype matches ``base_rows`` runs
    its matmul, ``scaling`` and the residual add as one ``addmm``.

    :param layer: LoRA layer holding the adapter.
    :param adapter: Adapter whose delta is added.
    :param base_rows: Base-layer output for the rows.
    :param rows: Layer input for the same rows.
    :param in_place: Accumulate into the 2-D ``base_rows`` and return it.
    :return: ``base_rows`` with the adapter delta added.
    """
    lora_b = layer.lora_B[adapter]
    # lora_B is an nn.ModuleDict, so its members' `.weight` resolves through
    # nn.Module.__getattr__ as the loose Tensor | Module; narrow it to Tensor.
    lora_b_weight = lora_b.weight
    assert isinstance(lora_b_weight, torch.Tensor)
    hidden = _apply_lora_factor(
        layer, layer.lora_A[adapter], layer.lora_dropout[adapter](rows)
    )
    scaling = layer.scaling[adapter]
    fusable = (
        isinstance(lora_b, nn.Linear)
        and lora_b.bias is None
        and hidden.dtype == base_rows.dtype
        and hidden.shape[:-1] == base_rows.shape[:-1]
        and base_rows.shape[-1] == lora_b_weight.shape[0]
    )
    if not fusable:
        delta = (_apply_lora_factor(layer, lora_b, hidden) * scaling).to(
            base_rows.dtype
        )
        return base_rows.add_(delta) if in_place else base_rows + delta
    weight = lora_b_weight.t().to(hidden.dtype)
    if in_place:
        return base_rows.addmm_(hidden, weight, alpha=scaling)
    out = torch.addmm(
        base_rows.reshape(-1, base_rows.shape[-1]),
        hidden.reshape(-1, hidden.shape[-1]),
        weight,
        alpha=scaling,
    )
    return out.view(base_rows.shape)


def _needs_peft_mixed_forward(layer: LoraLayer, routing: Sequence[str]) -> bool:
    """Whether any routed adapter needs PEFT's own mixed-batch forward.

    The sliced path computes the standard linear delta; embedding adapters
    and LoRA variants (e.g. aLoRA) use different maths, so those layers
    delegate to ``LoraLayer._mixed_batch_forward``.
    """
    embedding = getattr(layer, "lora_embedding_A", None) or {}
    variants = getattr(layer, "lora_variant", None) or {}
    return any(name in embedding or name in variants for name in set(routing))


def _unrouted_fused_adapter(
    layer: LoraLayer,
    forward_args: tuple[Any, ...],
    forward_kwargs: dict[str, Any],
) -> str | None:
    """The single active adapter an unrouted forward can run fused, or ``None`` to defer to PEFT.

    The fused run reproduces PEFT's vanilla LoRA delta only on a stock linear
    base without training-mode dropout, LoRA variants (DoRA, aLoRA), merged
    weights or extra forward arguments.

    :param layer: LoRA layer about to run.
    :param forward_args: Extra positional arguments to the layer forward.
    :param forward_kwargs: Keyword arguments to the layer forward.
    :return: The adapter name, or ``None``.
    """
    if forward_args or forward_kwargs or layer.disable_adapters or layer.merged:
        return None
    if len(layer.active_adapters) != 1:
        return None
    name = layer.active_adapters[0]
    if name not in layer.lora_A or name in layer.lora_variant:
        return None
    dropout = layer.lora_dropout[name]
    if not isinstance(dropout, nn.Identity) and dropout.training:
        return None
    return name if _has_stock_linear_base(layer) else None


def _single_run_forward(
    layer: LoraLayer,
    name: str,
    x: torch.Tensor,
    *forward_args: Any,
    **forward_kwargs: Any,
) -> torch.Tensor:
    """Stock linear base over every row of *x* with adapter *name*'s delta added in place.

    :param layer: LoRA layer with a stock ``nn.Linear`` base.
    :param name: Adapter for every row, or ``"__base__"`` for none.
    :param x: Layer input.
    :param forward_args: Extra positional arguments to the base layer.
    :param forward_kwargs: Keyword arguments to the base layer.
    :return: The layer output.
    """
    # An in-place delta on a reshaped view makes autograd clone the gradient.
    flat = x.reshape(-1, x.shape[-1])
    out = layer.base_layer(flat, *forward_args, **forward_kwargs)
    if name != "__base__" and name in layer.lora_A:
        out = _add_lora_delta(layer, name, out, flat, True)
    return out.view(*x.shape[:-1], out.shape[-1])


def _routing_spans(routing: Sequence[str], n_rows: int) -> list[tuple[str, int]]:
    """``(adapter, rows)`` for each contiguous run of *routing* over a layer input with *n_rows* leading rows.

    :raises ValueError: If the runs cannot cover *n_rows*.
    """
    runs = [(name, sum(1 for _ in run)) for name, run in itertools.groupby(routing)]
    # Layers that flatten (batch, seq, hidden) -> (batch * seq, hidden) before
    # their linears (OPT's MLP, MoE experts) show seq rows per routed sample.
    # The flatten is row-major, so each run just covers ``factor`` times as
    # many contiguous rows. divmod graph-breaks torch.compile on a symbolic
    # batch size.
    factor = n_rows // len(routing)
    remainder = n_rows % len(routing)
    if not remainder:
        return [(name, length * factor) for name, length in runs]
    if len({length for _, length in runs}) == 1 and n_rows % len(runs) == 0:
        # Vision-tower layers see image rows, not sample rows. Equal-length
        # runs repeat the same samples, so each run owns an equal image share.
        share = n_rows // len(runs)
        return [(name, share) for name, _ in runs]
    msg = (
        f"Fused adapter routing covers {len(routing)} rows but the layer "
        f"input's leading dimension is {n_rows}."
    )
    raise ValueError(msg)


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
    matching PEFT's mixed-batch behaviour. While routing is unset, a single
    eligible active adapter runs fused and anything else runs the layer's
    ordinary forward.
    """
    routing = ROUTING_STATE.get(layer)
    if routing is None:
        name = _unrouted_fused_adapter(layer, forward_args, forward_kwargs)
        if name is None:
            return original_forward(layer, x, *forward_args, **forward_kwargs)
        return _single_run_forward(layer, name, x)

    spans = _routing_spans(routing, x.shape[0])
    if _needs_peft_mixed_forward(layer, routing):
        names = [name for name, n in spans for _ in range(n)]
        return layer._mixed_batch_forward(
            x, *forward_args, adapter_names=names, **forward_kwargs
        )

    # Slicing a single run would add a zero-fill and copy to backward.
    whole = len(spans) == 1
    if whole and _has_stock_linear_base(layer):
        return _single_run_forward(
            layer, spans[0][0], x, *forward_args, **forward_kwargs
        )

    base_out = layer.base_layer(x, *forward_args, **forward_kwargs)
    pieces = []
    start = 0
    for name, n in spans:
        rows = base_out if whole else base_out.narrow(0, start, n)
        if name != "__base__" and name in layer.lora_A:
            lora_rows = x if whole else x.narrow(0, start, n)
            rows = _add_lora_delta(layer, name, rows, lora_rows, False)
        pieces.append(rows)
        start += n
    return pieces[0] if whole else torch.cat(pieces)


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
        previous = list(layer.active_adapters)
        if previous != [name]:
            layer.set_adapter(name)
        try:
            return original_forward(layer, x, *forward_args, **forward_kwargs)
        finally:
            if previous != [name] and previous:
                layer.set_adapter(previous if len(previous) > 1 else previous[0])
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


def _store_layer_cache(model: nn.Module, layers: list[LoraLayer]) -> None:
    LORA_LAYER_CACHE[model] = layers


def get_cached_lora_layers(model: nn.Module) -> list[LoraLayer]:
    """All ``LoraLayer`` modules under *model*, cached after the first traversal.

    :param model: A ``PeftModel`` or any module containing ``LoraLayer`` s.
    :return: The LoRA layers under *model*.
    """
    cached = LORA_LAYER_CACHE.get(model)
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


@library_shim("peft", "==0.21.*")
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
        ROUTING_STATE.pop(module, None)
    LORA_LAYER_CACHE.pop(model, None)


def set_fused_adapter_routing(model: nn.Module, routing: Sequence[str]) -> None:
    """Route each row of the next forward's batch through its own adapter.

    Routing stays active — including for gradient-checkpoint recomputation
    during ``backward()`` — until ``unset_fused_adapter_routing`` is called.

    :param model: The patched model whose LoRA layers should route rows.
    :param routing: Adapter name per batch row, e.g. ``["actor"] * B +
        ["critic"] * B``. ``"__base__"`` runs a row through the frozen base
        weights with no delta. A mixed routing over image inputs must repeat
        the same samples in equal-length runs, so each run's images are an
        equal share of the vision batch.
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
        ROUTING_STATE[module] = routing


def unset_fused_adapter_routing(model: nn.Module) -> None:
    """Deactivate fused routing, restoring standard single-adapter forward.

    :param model: The model whose LoRA layers should clear fused routing.
    """
    for module in get_cached_lora_layers(model):
        if _is_routing_managed(module):
            ROUTING_STATE[module] = None

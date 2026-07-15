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
from collections.abc import Sequence
from functools import partial

import torch
import torch.nn as nn

try:
    from peft.tuners.lora.layer import LoraLayer
except ImportError:  # pragma: no cover
    LoraLayer = None  # type: ignore[assignment, misc]

# The batch is a stack of rows, and ``routing`` labels each row with an adapter
# name. ``_contiguous_groups`` collapses consecutive same-name rows into one
# ``(name, start, stop)`` run, where ``[start, stop)`` are the row indices that
# adapter owns. Worked example — routing for a
# 4-row batch::
#
#     routing   = ["actor", "actor", "critic", "critic"]
#     row index =    0         1         2          3
#
# becomes two groups::
#
#     ("actor",  0, 2)   ->  rows 0, 1  (start=0, stop=2)
#     ("critic", 2, 4)   ->  rows 2, 3  (start=2, stop=4)
AdapterGroups = list[tuple[str, int, int]]


def _contiguous_groups(routing: Sequence[str]) -> AdapterGroups:
    """Compress a per-row adapter list into ``(name, start, stop)`` runs.

    :param routing: Adapter name for each batch row.
    :raises ValueError: If *routing* is empty.
    """
    groups: AdapterGroups = []
    start = 0
    for name, run in itertools.groupby(routing):
        stop = start + sum(1 for _ in run)
        groups.append((name, start, stop))
        start = stop
    if not groups:
        msg = "Fused adapter routing must name at least one row."
        raise ValueError(msg)
    return groups


def _lora_delta(layer: nn.Module, adapter: str, rows: torch.Tensor) -> torch.Tensor:
    """One adapter's low-rank delta for a slice of input rows."""
    lora_a = layer.lora_A[adapter]
    rows = layer.lora_dropout[adapter](rows.to(lora_a.weight.dtype))
    return layer.lora_B[adapter](lora_a(rows)) * layer.scaling[adapter]


def _needs_peft_mixed_forward(layer: nn.Module, groups: AdapterGroups) -> bool:
    """Whether any routed adapter needs PEFT's own mixed-batch forward.

    The sliced path computes the standard linear delta; embedding adapters
    and LoRA variants (e.g. aLoRA) use different maths, so those layers
    delegate to ``LoraLayer._mixed_batch_forward``.
    """
    embedding = getattr(layer, "lora_embedding_A", None) or {}
    variants = getattr(layer, "lora_variant", None) or {}
    return any(name in embedding or name in variants for name, _, _ in groups)


def _groups_for_leading_dim(groups: AdapterGroups, n_rows: int) -> AdapterGroups:
    """Rescale routing groups to a layer input with *n_rows* leading rows.

    Some layers flatten ``(batch, seq, hidden)`` to ``(batch * seq, hidden)``
    before their linears (OPT's MLP, MoE experts). The flatten is row-major,
    so sample ``i`` becomes the contiguous run ``[i * seq, (i + 1) * seq)``
    and the groups scale by ``seq``. No-op when the leading dim already
    matches the routed batch.

    :raises ValueError: If *n_rows* is not a whole multiple of the routed
        batch — the routing cannot be lined up with this input.
    """
    n_samples = groups[-1][2]
    if n_rows == n_samples:
        return groups
    if n_rows % n_samples != 0:
        msg = (
            f"Fused adapter routing covers {n_samples} rows but the input's "
            f"leading dimension is {n_rows}."
        )
        raise ValueError(msg)
    factor = n_rows // n_samples
    return [(name, start * factor, stop * factor) for name, start, stop in groups]


def _routed_forward(
    layer: nn.Module,
    x: torch.Tensor,
    *forward_args,
    **forward_kwargs,
) -> torch.Tensor:
    """Replacement ``forward`` installed on patched LoRA layers.

    Runs the layer's ordinary single-adapter forward until routing is set.
    Adapters that do not wrap this layer leave their rows on the base output,
    matching PEFT's mixed-batch behaviour.
    """
    groups = layer._fused_adapter_groups
    if groups is None:
        return type(layer).forward(layer, x, *forward_args, **forward_kwargs)

    groups = _groups_for_leading_dim(groups, x.shape[0])

    if _needs_peft_mixed_forward(layer, groups):
        adapter_names = [
            name for name, start, stop in groups for _ in range(stop - start)
        ]
        return layer._mixed_batch_forward(
            x, *forward_args, adapter_names=adapter_names, **forward_kwargs
        )

    base_out = layer.base_layer(x, *forward_args, **forward_kwargs)

    if len(groups) == 1:
        name = groups[0][0]
        if name == "__base__" or name not in layer.lora_A:
            return base_out
        return base_out + _lora_delta(layer, name, x).to(base_out.dtype)

    pieces = []
    for name, start, stop in groups:
        rows = base_out.narrow(0, start, stop - start)
        if name != "__base__" and name in layer.lora_A:
            delta = _lora_delta(layer, name, x.narrow(0, start, stop - start))
            rows = rows + delta.to(rows.dtype)
        pieces.append(rows)
    return torch.cat(pieces)


def _store_layer_cache(model: nn.Module, layers: list[nn.Module]) -> None:
    try:
        model._fused_lora_layers = layers  # type: ignore[attr-defined]
    except (AttributeError, TypeError):
        pass  # Best-effort cache; routing falls back to a module traversal.


def _get_cached_lora_layers(model: nn.Module) -> list[nn.Module]:
    """All ``LoraLayer`` modules under *model*, cached after the first traversal."""
    cached = getattr(model, "_fused_lora_layers", None)
    if cached is not None:
        return cached
    if LoraLayer is None:
        return []
    # nn.Module.modules rather than model.modules: EvolvableModule overrides
    # modules() to return only evolvable children, which excludes LoRA layers.
    layers = [m for m in nn.Module.modules(model) if isinstance(m, LoraLayer)]
    _store_layer_cache(model, layers)
    return layers


def patch_lora_for_fused_forward(model: nn.Module) -> None:
    """Swap every LoRA layer's ``forward`` for the fused-routing forward.

    Routing starts inactive, so the patched layers behave exactly as before
    until ``set_fused_adapter_routing`` is called. Idempotent — call again
    after adding adapters that wrap new modules; only the new layers are
    patched and the cached layer list is refreshed either way.

    :param model: A ``PeftModel`` or any module containing ``LoraLayer`` s.
    """
    if LoraLayer is None:
        return
    layers: list[nn.Module] = []
    for module in nn.Module.modules(model):
        if not isinstance(module, LoraLayer):
            continue
        layers.append(module)
        if not hasattr(module, "_fused_adapter_groups"):
            module._fused_adapter_groups = None  # type: ignore[attr-defined]
            module.forward = partial(_routed_forward, module)  # type: ignore[method-assign]
    _store_layer_cache(model, layers)


def unpatch_lora_for_fused_forward(model: nn.Module) -> None:
    """Restore every LoRA layer's original ``forward`` and drop routing state.

    After this, ``set_fused_adapter_routing`` raises until the model is
    patched again.

    :param model: The model to release from fused-routing control.
    """
    for module in _get_cached_lora_layers(model):
        if hasattr(module, "_fused_adapter_groups"):
            del module._fused_adapter_groups
            del module.forward
    if hasattr(model, "_fused_lora_layers"):
        del model._fused_lora_layers


def set_fused_adapter_routing(model: nn.Module, routing: Sequence[str]) -> None:
    """Route each row of the next forward's batch through its own adapter.

    Routing stays active — including for gradient-checkpoint recomputation
    during ``backward()`` — until ``clear_fused_adapter_routing`` is called.

    :param model: The patched model whose LoRA layers should route rows.
    :param routing: Adapter name per batch row, e.g. ``["actor"] * B +
        ["critic"] * B``. ``"__base__"`` runs a row through the frozen base
        weights with no delta.
    :raises RuntimeError: If LoRA layers are unpatched (the routing would be
        silently ignored) or have adapters merged into the base weights.
    :raises ValueError: If *routing* is empty, names an unknown adapter, or
        names a DoRA adapter (unsupported, as in PEFT's mixed-batch forward).
    """
    layers = _get_cached_lora_layers(model)
    if not layers:
        # Plain base model (no adapters) or PEFT not installed: nothing to
        # route, and the unfused forward is already correct.
        return
    if any(not hasattr(m, "_fused_adapter_groups") for m in layers):
        msg = (
            "set_fused_adapter_routing called on a model with LoRA layers that "
            "have no fused-routing forward; the routing would be silently "
            "ignored. Call patch_lora_for_fused_forward(model) first (again, "
            "if adapters were added after the last call)."
        )
        raise RuntimeError(msg)

    groups = _contiguous_groups(routing)
    _validate_routing(layers, groups)
    for module in layers:
        module._fused_adapter_groups = groups  # type: ignore[attr-defined]


def _validate_routing(layers: list[nn.Module], groups: AdapterGroups) -> None:
    """Reject routings PEFT would compute silently wrongly (unknown names)
    or that the fused forward cannot compute (merged weights, DoRA).

    :param layers: All LoRA layers under the model.
    :param groups: Contiguous runs of adapter names in the routing.
    :raises RuntimeError: If any layer has merged adapters.
    :raises ValueError: If any adapter name is unknown or a DoRA adapter.
    """
    requested = {name for name, _, _ in groups if name != "__base__"}
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


def clear_fused_adapter_routing(model: nn.Module) -> None:
    """Deactivate fused routing, restoring standard single-adapter forward.

    :param model: The model whose LoRA layers should clear fused routing.
    """
    for module in _get_cached_lora_layers(model):
        if hasattr(module, "_fused_adapter_groups"):
            module._fused_adapter_groups = None  # type: ignore[attr-defined]

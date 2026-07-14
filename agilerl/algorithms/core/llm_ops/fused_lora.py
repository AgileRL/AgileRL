"""Fused multi-adapter LoRA forward pass.

Enables running multiple LoRA adapters (e.g. actor + critic) in a single
forward pass by partitioning the batch dimension.  Each adapter's LoRA
weights are applied only to its assigned batch rows, while the frozen base
weights are computed once for the entire batch.

This eliminates adapter switching during training and resolves the
gradient-checkpointing / DeepSpeed incompatibility that arises when
different adapters must be active for different parts of the computation
graph.

The mechanism piggy-backs on PEFT's existing ``_mixed_batch_forward``
(which handles the per-row LoRA routing) but bypasses the inference-only
gate and uses persistent per-layer attributes instead of ephemeral hooks
so that routing survives gradient-checkpoint recomputation.

.. note::
    LoRA layer references are cached on the model during
    ``patch_lora_for_fused_forward`` and reused by ``set_`` / ``clear_``
    to avoid repeated ``nn.Module.modules()`` traversals (which are
    expensive when called dozens of times per ``learn()``).
    We use ``nn.Module.modules()`` rather than ``model.modules()``
    because ``EvolvableModule`` overrides ``modules()`` to return only
    evolvable children, which excludes the LoRA layers.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

try:
    from peft.tuners.lora.layer import LoraLayer
except ImportError:  # pragma: no cover
    LoraLayer = None  # type: ignore[assignment, misc]


def _align_routing_to_leading_dim(routing: Sequence[str], args: tuple) -> Sequence[str]:
    """Repeat each per-row adapter name to match the input's leading dim.

    Models that flatten ``(batch, seq, hidden)`` to ``(batch * seq, hidden)``
    before a LoRA linear (OPT, Switch/NLLB-MoE) need one name per flattened row;
    a no-op when the leading dim already equals the routing length.

    :param routing: Adapter names, one per logical batch row.
    :type routing: Sequence[str]
    :param args: Positional forward args; ``args[0]`` is the input tensor.
    :type args: tuple
    :return: Routing expanded to the input's leading dim, or unchanged.
    :rtype: Sequence[str]
    """
    if not args or not isinstance(args[0], torch.Tensor):
        return routing
    leading = args[0].shape[0]
    n = len(routing)
    if n == 0 or leading == n or leading % n != 0:
        return routing
    factor = leading // n
    return [name for name in routing for _ in range(factor)]


def _fused_routing_pre_hook(
    module: nn.Module,
    args: tuple,
    kwargs: dict,
) -> tuple[tuple, dict]:
    """Inject leading-dim-aligned ``adapter_names`` when fused routing is active.

    :param module: The LoRA layer about to run ``forward``.
    :type module: nn.Module
    :param args: Positional forward args; ``args[0]`` is the input tensor.
    :type args: tuple
    :param kwargs: Keyword args passed to ``forward``.
    :type kwargs: dict
    :return: ``args`` unchanged and ``kwargs`` with ``adapter_names`` set when routing is active.
    :rtype: tuple[tuple, dict]
    """
    routing = getattr(module, "_fused_adapter_routing", None)
    if routing is not None:
        kwargs["adapter_names"] = _align_routing_to_leading_dim(routing, args)
    return args, kwargs


def _get_cached_lora_layers(model: nn.Module) -> list[nn.Module]:
    """Return the cached list of LoRA layers, falling back to a full traversal.

    :param model: Root module that may store ``_fused_lora_layers`` after
        ``patch_lora_for_fused_forward`` runs.
    :type model: nn.Module
    :return: All ``LoraLayer`` instances under ``model``, or an empty list if PEFT
        is not installed.
    :rtype: list[nn.Module]
    """
    cached = getattr(model, "_fused_lora_layers", None)
    if cached is not None:
        return cached
    if LoraLayer is None:
        return []
    layers = [m for m in nn.Module.modules(model) if isinstance(m, LoraLayer)]
    try:
        model._fused_lora_layers = layers  # type: ignore[attr-defined]
    except (AttributeError, TypeError):
        # Best-effort cache: some wrapped/slotted modules may reject dynamic attrs.
        # This only impacts caching/performance, not correctness.
        pass
    return layers


def _make_base_output_clone_hook(lora_layer: nn.Module):
    """Clone the base-layer output while fused routing is active.

    Fused adapter routing otherwise breaks on a quantized base: PEFT
    accumulates the LoRA delta in place into the base output, which is a view
    of bitsandbytes' quantized matmul output, and autograd forbids that
    in-place edit. Cloning the output fixes it; no-op when routing is inactive.
    """

    def _hook(module: nn.Module, args: tuple, output):
        if getattr(lora_layer, "_fused_adapter_routing", None) is None:
            return None
        if isinstance(output, torch.Tensor):
            return output.clone()
        return None

    return _hook


def patch_lora_for_fused_forward(model: nn.Module) -> None:
    """Register forward pre-hooks on all LoRA layers.

    The hooks inject per-sample adapter routing into each layer's forward
    call when ``_fused_adapter_routing`` is set, triggering PEFT's
    ``_mixed_batch_forward`` code path.  When the attribute is ``None``
    (the default), the hooks are no-ops and standard single-adapter
    forward runs unchanged.

    Also caches the list of LoRA layers on the model for fast access
    by ``set_fused_adapter_routing`` / ``clear_fused_adapter_routing``.

    Idempotent: layers that already carry a fused-routing hook are skipped,
    so it is safe (and required) to call again whenever adapters wrapping
    **new** modules are added after the first call — only the new layers are
    hooked, and the cached layer list is refreshed either way.

    :param model: A ``PeftModel`` (or any ``nn.Module`` containing
        ``LoraLayer`` sub-modules).
    :type model: nn.Module
    :return: ``None``
    :rtype: None
    """
    if LoraLayer is None:
        return
    layers: list[nn.Module] = []
    for module in nn.Module.modules(model):
        if not isinstance(module, LoraLayer):
            continue
        layers.append(module)
        if hasattr(module, "_fused_adapter_routing"):
            # Already patched (re-patch after adding adapters): keep the
            # existing hook; double-registering would run it twice per forward.
            continue
        module._fused_adapter_routing = None  # type: ignore[attr-defined]
        module._fused_routing_hook_handle = (  # type: ignore[attr-defined]
            module.register_forward_pre_hook(
                _fused_routing_pre_hook,
                with_kwargs=True,
            )
        )
        # Clone the frozen base output during fused routing so PEFT's in-place
        # LoRA accumulation (``result[idx] += ...``) never mutates a bnb 4-bit/
        # 8-bit custom-Function output view (forbidden by autograd; crashes the
        # multi-adapter PPO gradient forward). No-op when routing is inactive.
        base_layer = getattr(module, "base_layer", None)
        if base_layer is not None:
            module._fused_base_clone_handle = (  # type: ignore[attr-defined]
                base_layer.register_forward_hook(_make_base_output_clone_hook(module))
            )
    try:
        model._fused_lora_layers = layers  # type: ignore[attr-defined]
    except (AttributeError, TypeError):
        # Best-effort cache only: some wrapped/frozen modules may reject
        # dynamic attribute assignment. Fused routing still works by
        # discovering LoRA layers when needed.
        pass


def unpatch_lora_for_fused_forward(model: nn.Module) -> None:
    """Remove the hooks and per-layer state installed by ``patch_lora_for_fused_forward``.

    Restores every LoRA layer to its pre-patch state (no routing attribute,
    no pre-hook) and drops the cached layer list. After this,
    ``set_fused_adapter_routing`` raises until the model is patched again.

    :param model: The model to release from fused-routing control.
    :type model: nn.Module
    :return: ``None``
    :rtype: None
    """
    for module in _get_cached_lora_layers(model):
        handle = getattr(module, "_fused_routing_hook_handle", None)
        if handle is not None:
            handle.remove()
            del module._fused_routing_hook_handle
        clone_handle = getattr(module, "_fused_base_clone_handle", None)
        if clone_handle is not None:
            clone_handle.remove()
            del module._fused_base_clone_handle
        if hasattr(module, "_fused_adapter_routing"):
            del module._fused_adapter_routing
    try:
        del model._fused_lora_layers
    except (AttributeError, TypeError):
        pass


def set_fused_adapter_routing(model: nn.Module, routing: Sequence[str]) -> None:
    """Activate fused adapter routing on all LoRA layers.

    :param model: The model whose LoRA layers should use fused routing.
    :type model: nn.Module
    :param routing: Adapter names, one per row of the fused batch (e.g.
        ``["actor"] * B + ["critic"] * B`` when the batch concatenates actor and
        critic inputs).  ``"__base__"`` routes a row through the frozen base
        weights with no LoRA delta.
    :type routing: Sequence[str]
    :raises RuntimeError: If LoRA layers exist that
        ``patch_lora_for_fused_forward`` has not hooked — without the hooks
        the routing would be silently ignored and every row would run under
        the currently active adapter.
    :raises ValueError: If *routing* names an adapter that no LoRA layer
        registers — PEFT's ``_mixed_batch_forward`` silently treats unknown
        names as base-only rows, so a typo would corrupt training without
        an error.
    :return: ``None``
    :rtype: None
    """
    layers = _get_cached_lora_layers(model)
    if not layers:
        # Plain base model (no adapters) or PEFT not installed: nothing to
        # route, and the unfused forward is already correct.
        return

    if any(not hasattr(m, "_fused_adapter_routing") for m in layers):
        msg = (
            "set_fused_adapter_routing called on a model with LoRA layers that "
            "have no fused-routing hook; the routing would be silently ignored. "
            "Call patch_lora_for_fused_forward(model) first (again, if adapters "
            "were added after the last call)."
        )
        raise RuntimeError(msg)

    routing = list(routing)
    requested = set(routing) - {"__base__"}
    available: set[str] = set()
    for module in layers:
        for attr in getattr(module, "adapter_layer_names", ()):
            container = getattr(module, attr, None)
            if container is not None:
                available.update(container.keys())
        if requested <= available:
            break
    # Validation is best-effort: real PEFT layers always expose their adapter
    # containers via ``adapter_layer_names``; when none are discoverable (e.g.
    # test doubles), routing is applied without validation.
    if available and not requested <= available:
        unknown = sorted(requested - available)
        msg = (
            f"Unknown adapter name(s) in fused routing: {unknown}. Known "
            f"adapters: {sorted(available)} (plus '__base__' for "
            "base-only rows)."
        )
        raise ValueError(msg)

    for module in layers:
        module._fused_adapter_routing = routing  # type: ignore[attr-defined]


def clear_fused_adapter_routing(model: nn.Module) -> None:
    """Deactivate fused routing, restoring standard single-adapter forward.

    Lenient by design so it is safe to call from error-cleanup paths: layers
    never patched are left untouched (rather than gaining a routing attribute
    that would defeat ``set_fused_adapter_routing``'s patched-layer check).

    :param model: The model whose LoRA layers should clear fused routing.
    :type model: nn.Module
    :return: ``None``
    :rtype: None
    """
    for module in _get_cached_lora_layers(model):
        if hasattr(module, "_fused_adapter_routing"):
            module._fused_adapter_routing = None  # type: ignore[attr-defined]

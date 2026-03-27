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

import torch.nn as nn

try:
    from peft.tuners.lora.layer import LoraLayer
except ImportError:
    LoraLayer = None  # type: ignore[assignment, misc]

_CACHE_ATTR = "_fused_lora_layers"


def _fused_routing_pre_hook(
    module: nn.Module,
    args: tuple,
    kwargs: dict,
) -> tuple[tuple, dict]:
    """Forward pre-hook that injects ``adapter_names`` when fused routing is active."""
    routing = getattr(module, "_fused_adapter_routing", None)
    if routing is not None:
        kwargs["adapter_names"] = routing
    return args, kwargs


def _get_cached_lora_layers(model: nn.Module) -> list[nn.Module]:
    """Return the cached list of LoRA layers, falling back to a full traversal."""
    cached = getattr(model, _CACHE_ATTR, None)
    if cached is not None:
        return cached
    if LoraLayer is None:
        return []
    layers = [m for m in nn.Module.modules(model) if isinstance(m, LoraLayer)]
    try:
        model._fused_lora_layers = layers  # type: ignore[attr-defined]
    except (AttributeError, TypeError):
        pass
    return layers


def patch_lora_for_fused_forward(model: nn.Module) -> None:
    """Register permanent forward pre-hooks on all LoRA layers.

    The hooks inject per-sample adapter routing into each layer's forward
    call when ``_fused_adapter_routing`` is set, triggering PEFT's
    ``_mixed_batch_forward`` code path.  When the attribute is ``None``
    (the default), the hooks are no-ops and standard single-adapter
    forward runs unchanged.

    Also caches the list of LoRA layers on the model for fast access
    by ``set_fused_adapter_routing`` / ``clear_fused_adapter_routing``.

    Must be called **once** after the PEFT model is fully constructed
    (all adapters added).

    :param model: A ``PeftModel`` (or any ``nn.Module`` containing
        ``LoraLayer`` sub-modules).
    """
    if LoraLayer is None:
        return
    layers: list[nn.Module] = []
    for module in nn.Module.modules(model):
        if isinstance(module, LoraLayer):
            module._fused_adapter_routing = None  # type: ignore[attr-defined]
            module.register_forward_pre_hook(
                _fused_routing_pre_hook,
                with_kwargs=True,
            )
            layers.append(module)
    try:
        model._fused_lora_layers = layers  # type: ignore[attr-defined]
    except (AttributeError, TypeError):
        pass


def set_fused_adapter_routing(model: nn.Module, routing: list[str]) -> None:
    """Activate fused adapter routing on all LoRA layers.

    :param model: The model whose LoRA layers should use fused routing.
    :param routing: A list of adapter names, one per sample in the batch.
        For example ``["actor"] * B + ["critic"] * B`` when the batch is
        the concatenation of actor and critic inputs.
    """
    for module in _get_cached_lora_layers(model):
        module._fused_adapter_routing = routing  # type: ignore[attr-defined]


def clear_fused_adapter_routing(model: nn.Module) -> None:
    """Deactivate fused routing, restoring standard single-adapter forward."""
    for module in _get_cached_lora_layers(model):
        module._fused_adapter_routing = None  # type: ignore[attr-defined]

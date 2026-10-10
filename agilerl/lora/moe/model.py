# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Whole-model passes: LoRA target discovery, wrapper upgrade, recompute policy, grouped GEMM."""

from __future__ import annotations

import warnings
from types import MethodType
from typing import Any

import torch
import torch.nn as nn
from peft.tuners.lora.layer import ParamWrapper
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointWrapper,
)
from transformers.modeling_layers import GradientCheckpointingLayer

from agilerl.lora.fused import patch_lora_for_fused_forward
from agilerl.lora.moe.adapters import wrapper_chain
from agilerl.lora.moe.grouped_gemm import grouped_linear
from agilerl.lora.moe.layouts import (
    is_routed_experts_module,
    is_sorted_experts_module,
    is_transposed_experts_module,
    routed_projection_names,
)
from agilerl.lora.moe.routed import routed_experts_local_forward
from agilerl.lora.moe.wrappers import (
    RoutedExpertsLoraWrapper,
    SortedExpertsLoraWrapper,
    TransposedExpertsLoraWrapper,
)


def moe_expert_target_parameters(model: nn.Module) -> list[str]:
    """Parameter-path suffixes of packed expert weights for ``LoraConfig.target_parameters``.

    :param model: Model to scan for packed expert modules.
    :type model: nn.Module
    :return: Sorted parameter-path suffixes.
    :rtype: list[str]
    """
    suffixes: set[str] = set()
    for name, module in model.named_modules():
        prefix = ".".join(name.split(".")[-2:])
        if is_sorted_experts_module(module):
            suffixes.add(f"{prefix}.weight")
        elif (projections := routed_projection_names(module)) is not None:
            suffixes.add(f"{prefix}.{projections[0]}")
            suffixes.add(f"{prefix}.down_proj")
        elif is_transposed_experts_module(module):
            suffixes.add(f"{prefix}.gate_up_proj")
            suffixes.add(f"{prefix}.down_proj")
    return sorted(suffixes)


def _bind_gate_token_index(model: nn.Module) -> None:
    """Copy each sorted-MoE gate's ``batch_index`` onto sibling expert wrappers.

    The sibling ``router`` returns ``(index_sorted_experts, batch_index, ...)``.
    Fused routing is in token order; the wrappers permute adapter ids with
    that index (see ``token_adapter_ids``).
    """
    for parent in model.modules():
        router = getattr(parent, "router", None)
        if router is None:
            continue
        experts = [
            child
            for child in parent.children()
            if type(child) is SortedExpertsLoraWrapper
        ]
        if not experts or getattr(router, "agilerl_token_index_hook", False):
            continue

        def hook(
            _module: nn.Module,
            args: tuple[Any, ...],
            output: object,
            _experts: list[SortedExpertsLoraWrapper] = experts,
        ) -> None:
            if not (isinstance(output, tuple) and len(output) >= 2):
                return
            token_index = output[1]
            hidden = args[0] if args else None
            if not isinstance(token_index, torch.Tensor) or not isinstance(
                hidden, torch.Tensor
            ):
                return
            n_tokens = int(hidden.shape[0])
            for expert in _experts:
                expert.token_index = token_index
                expert.n_tokens = n_tokens

        router.register_forward_hook(hook)
        router.agilerl_token_index_hook = True


def upgrade_moe_param_wrappers(model: nn.Module) -> int:
    """Swap eligible ``ParamWrapper`` instances to split-LoRA execution, returning how many.

    :param model: PEFT model holding expert ``ParamWrapper`` layers.
    :type model: nn.Module
    :return: Number of wrappers upgraded.
    :rtype: int
    """
    wrapped_bases = {
        id(module.base_layer)
        for module in model.modules()
        if isinstance(module, ParamWrapper)
    }
    upgraded = 0
    fallbacks: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, ParamWrapper) or id(module) in wrapped_bases:
            continue
        if type(module) is not ParamWrapper:
            continue
        chain = wrapper_chain(module)
        base = module.get_base_layer()
        projections = routed_projection_names(base)
        if (
            len(chain) == 1
            and module.parameter_name == "weight"
            and is_sorted_experts_module(base)
        ):
            module.__class__ = SortedExpertsLoraWrapper
            upgraded += 1
        elif projections is not None and set(chain) <= {projections[0], "down_proj"}:
            module.__class__ = RoutedExpertsLoraWrapper
            upgraded += 1
        elif is_transposed_experts_module(base) and set(chain) <= {
            "gate_up_proj",
            "down_proj",
        }:
            module.__class__ = TransposedExpertsLoraWrapper
            upgraded += 1
        elif module.get_param().ndim == 3:
            fallbacks.append(name)
    _bind_gate_token_index(model)
    # Class swaps drop the instance fused-routing forward; re-attach it.
    if upgraded:
        patch_lora_for_fused_forward(model)
    if fallbacks:
        warnings.warn(
            "Packed-experts LoRA wrappers on unrecognized module conventions "
            "stay on PEFT's delta-materializing forward (memory-hungry): "
            f"{fallbacks}.",
            stacklevel=2,
        )
    return upgraded


def materializes_expert_lora(model: nn.Module) -> bool:
    """Whether any packed-experts LoRA stays on PEFT's delta-materializing forward.

    :param model: PEFT model after :func:`upgrade_moe_param_wrappers`.
    :type model: nn.Module
    :return: ``True`` when an outermost ``ParamWrapper`` was not upgraded.
    :rtype: bool
    """
    wrapped_bases = {
        id(module.base_layer)
        for module in model.modules()
        if isinstance(module, ParamWrapper)
    }
    return any(
        type(module) is ParamWrapper and id(module) not in wrapped_bases
        for module in model.modules()
    )


def _checkpoints_activations(module: nn.Module) -> bool:
    """Whether ``module`` reruns its forward during backward."""
    return isinstance(module, CheckpointWrapper) or (
        isinstance(module, GradientCheckpointingLayer) and module.gradient_checkpointing
    )


def set_routed_experts_recompute(model: nn.Module, enabled: bool | None) -> None:
    """Choose recompute-in-backward or the full autograd graph for every routed-experts LoRA wrapper.

    :param model: Model holding ``RoutedExpertsLoraWrapper`` layers.
    :type model: nn.Module
    :param enabled: ``True`` runs :class:`~agilerl.lora.moe.recompute.LoraExpertsFunction`
        on frozen base weights and ``False`` keeps the full graph. ``None``
        keeps the full graph inside activation-checkpointed blocks, which
        already rerun the expert forward in backward, and recomputes
        everywhere else.
    :type enabled: bool | None
    """
    checkpointed: set[int] = set()
    if enabled is None:
        for module in model.modules():
            if _checkpoints_activations(module):
                checkpointed.update(id(inner) for inner in module.modules())
    for module in model.modules():
        if isinstance(module, RoutedExpertsLoraWrapper):
            module.recompute = (
                id(module) not in checkpointed if enabled is None else enabled
            )


def set_routed_experts_chunk_bytes(model: nn.Module, chunk_bytes: int) -> None:
    """Set the row-chunk budget of every routed and sorted experts LoRA wrapper.

    :param model: Model holding ``RoutedExpertsLoraWrapper`` or
        ``SortedExpertsLoraWrapper`` layers.
    :type model: nn.Module
    :param chunk_bytes: Widest ``[rows, features]`` activation of one row
        chunk, one fp32 chunk of the expert-parallel combine, and one grouped
        LoRA GEMM output. Bigger chunks launch fewer kernels per MoE layer.
    :type chunk_bytes: int
    """
    for module in model.modules():
        if isinstance(module, (RoutedExpertsLoraWrapper, SortedExpertsLoraWrapper)):
            module.chunk_bytes = chunk_bytes


def bind_routed_experts_config(model: nn.Module) -> None:
    """Point packed experts that have no ``act_fn`` at the model config.

    Liger-patched packed experts drop ``act_fn``; the fused path then reads
    ``config.hidden_act`` off the experts module.

    :param model: PEFT model with a ``config`` carrying ``hidden_act``.
    :type model: nn.Module
    """
    for module in model.modules():
        if is_routed_experts_module(module) and not callable(
            getattr(module, "act_fn", None)
        ):
            module.config = model.config


def install_packed_expert_grouped_gemm(model: nn.Module) -> int:
    """Replace packed-expert Python loops with grouped GEMM.

    Walks routed (Nemotron-H / Qwen3-MoE) and sorted (Granite) expert modules.
    PEFT ``ParamWrapper`` shells are unwrapped via ``get_base_layer`` so expert
    LoRA still runs on ``RoutedExpertsLoraWrapper`` / ``SortedExpertsLoraWrapper``
    on top of this kernel. Idempotent.

    :param model: Model (or PEFT wrapper) to patch.
    :type model: nn.Module
    :return: Number of expert modules whose ``forward`` was replaced.
    :rtype: int
    """
    patched = 0
    for module in model.modules():
        target = module.get_base_layer() if isinstance(module, ParamWrapper) else module
        if is_routed_experts_module(target):
            fn = routed_experts_local_forward
        elif is_sorted_experts_module(target):
            fn = grouped_linear
        else:
            continue
        if getattr(target.forward, "__func__", None) is fn:
            continue
        target.forward = MethodType(fn, target)
        patched += 1
    return patched

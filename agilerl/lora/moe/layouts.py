# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Packed mixture-of-experts layouts: detection and expert activations."""

from __future__ import annotations

import inspect
from collections.abc import Callable

import torch
import torch.nn as nn
from transformers.activations import get_activation

from agilerl.architectures.gptoss import experts as gptoss_experts

is_transposed_experts_module = gptoss_experts.is_gpt_oss_experts_module


def _forward_param_names(module: nn.Module) -> list[str]:
    """Positional parameter names of a module's ``forward``, excluding ``self``."""
    try:
        signature = inspect.signature(type(module).forward)
    except (TypeError, ValueError):
        return []
    return [name for name in signature.parameters if name != "self"]


def is_sorted_experts_module(module: nn.Module) -> bool:
    """Whether *module* is a grouped linear over expert-sorted rows with a stacked 3D ``weight``."""
    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 3:
        return False
    return _forward_param_names(module)[:2] == ["inputs", "expert_size"]


def routed_projection_names(module: nn.Module) -> tuple[str, bool] | None:
    """The up-projection parameter name and gatedness of a self-routing packed-experts block.

    Gated (``gate_up_proj``, Qwen3-MoE/granite: ``act(gate) * up``) and
    ungated (``up_proj``, NemotronH: ``act(up)``) variants are supported;
    per-expert biases or other layouts are off-convention and return ``None``.
    """
    down = getattr(module, "down_proj", None)
    if not isinstance(down, torch.Tensor) or down.ndim != 3:
        return None
    if getattr(module, "down_proj_bias", None) is not None:
        return None
    if _forward_param_names(module)[:3] != [
        "hidden_states",
        "top_k_index",
        "top_k_weights",
    ]:
        return None
    for up_name, gated in (("gate_up_proj", True), ("up_proj", False)):
        up = getattr(module, up_name, None)
        if not isinstance(up, torch.Tensor) or up.ndim != 3:
            continue
        if getattr(module, f"{up_name}_bias", None) is not None:
            continue
        num_experts, up_out, in_dim = up.shape
        if gated and up_out % 2:
            continue
        intermediate = up_out // 2 if gated else up_out
        if down.shape == (num_experts, in_dim, intermediate):
            return up_name, gated
    return None


def is_routed_experts_module(module: nn.Module) -> bool:
    """Whether *module* is a self-routing packed-experts block."""
    return routed_projection_names(module) is not None


def is_packed_experts_module(module: nn.Module) -> bool:
    """Whether *module* is a packed expert stack (routed, sorted, or transposed)."""
    return (
        is_routed_experts_module(module)
        or is_sorted_experts_module(module)
        or is_transposed_experts_module(module)
    )


def routed_experts_act_fn(
    experts: nn.Module,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Activation for the packed grouped-GEMM path; never guessed.

    Stock HF packed experts expose ``act_fn``. Fused kernels (LigerExperts)
    drop it but keep the config that chose the activation. If neither exists,
    raise instead of assuming a default.
    """
    act_fn = getattr(experts, "act_fn", None)
    if callable(act_fn):
        return act_fn
    config = getattr(experts, "config", None)
    hidden_act = getattr(config, "hidden_act", None)
    resolved = get_activation(hidden_act) if isinstance(hidden_act, str) else None
    if resolved is None:
        msg = (
            "Packed-experts module does not expose an activation function. "
            "Set ``act_fn`` on the module or provide ``config.hidden_act``."
        )
        raise RuntimeError(msg)
    return resolved


def expert_activation(
    projected: torch.Tensor,
    act_fn: Callable[[torch.Tensor], torch.Tensor],
    gated: bool,
) -> torch.Tensor:
    """``act(gate) * up`` for a gated up-projection, else ``act(up)``."""
    if gated:
        gate, up = projected.chunk(2, dim=-1)
        return act_fn(gate) * up
    return act_fn(projected)

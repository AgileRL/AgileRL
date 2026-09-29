# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""GptOssExperts layout: ``x @ W + b`` and interleaved even/odd SwiGLU."""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor


@dataclass(frozen=True)
class GptOssExpertParams:
    """Tensor parameters of a module that matches the GptOssExperts layout."""

    gate_up_proj: torch.Tensor
    down_proj: torch.Tensor
    gate_up_proj_bias: torch.Tensor
    down_proj_bias: torch.Tensor
    alpha: float
    limit: float


def _forward_param_names(module: nn.Module) -> list[str]:
    """Positional parameter names of a module's ``forward``, excluding ``self``."""
    try:
        signature = inspect.signature(type(module).forward)
    except (TypeError, ValueError):
        return []
    return [name for name in signature.parameters if name != "self"]


def _counts_list(counts: Sequence[int] | torch.Tensor) -> list[int]:
    """Per-expert row counts as a plain list."""
    if isinstance(counts, torch.Tensor):
        return [int(count) for count in counts.tolist()]
    return [int(count) for count in counts]


def gpt_oss_expert_params(module: nn.Module) -> GptOssExpertParams | None:
    """Return expert tensors when *module* matches the GptOssExperts layout."""
    if _forward_param_names(module)[:3] != [
        "hidden_states",
        "router_indices",
        "routing_weights",
    ]:
        return None
    up = getattr(module, "gate_up_proj", None)
    down = getattr(module, "down_proj", None)
    up_bias = getattr(module, "gate_up_proj_bias", None)
    down_bias = getattr(module, "down_proj_bias", None)
    alpha = getattr(module, "alpha", None)
    limit = getattr(module, "limit", None)
    if not isinstance(up, torch.Tensor) or up.ndim != 3:
        return None
    if not isinstance(down, torch.Tensor) or down.ndim != 3:
        return None
    if not isinstance(up_bias, torch.Tensor) or not isinstance(down_bias, torch.Tensor):
        return None
    if not isinstance(alpha, (int, float)) or not isinstance(limit, (int, float)):
        return None
    num_experts, hidden, two_intermediate = up.shape
    if two_intermediate % 2:
        return None
    intermediate = two_intermediate // 2
    if (
        down.shape != (num_experts, intermediate, hidden)
        or up_bias.shape != (num_experts, two_intermediate)
        or down_bias.shape != (num_experts, hidden)
    ):
        return None
    return GptOssExpertParams(
        gate_up_proj=up,
        down_proj=down,
        gate_up_proj_bias=up_bias,
        down_proj_bias=down_bias,
        alpha=float(alpha),
        limit=float(limit),
    )


def is_gpt_oss_experts_module(module: nn.Module) -> bool:
    """Whether *module* is a packed experts block in the GptOssExperts convention."""
    return gpt_oss_expert_params(module) is not None


def _local_weight(weight: torch.Tensor) -> torch.Tensor:
    """Dense view of an expert weight, including an FSDP DTensor shard."""
    if isinstance(weight, DTensor):
        return weight.to_local()
    return weight


def expert_matmul_loop(
    x: torch.Tensor,
    weight: torch.Tensor,
    counts: Sequence[int] | torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-expert ``x @ W[+b]`` over expert-sorted rows with stacked ``[experts, in, out]`` weight."""
    weight = _local_weight(weight)
    if bias is not None:
        bias = _local_weight(bias)
    outputs = []
    for expert, rows in enumerate(x.split(_counts_list(counts))):
        projected = rows @ weight[expert]
        if bias is not None:
            projected = projected + bias[expert]
        outputs.append(projected)
    return torch.cat(outputs)


def apply_gpt_oss_gate(
    gate_up: torch.Tensor, alpha: float, limit: float
) -> torch.Tensor:
    """Interleaved SwiGLU used by GptOssExperts (even gate, odd up)."""
    gate, up = gate_up[..., ::2], gate_up[..., 1::2]
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    glu = gate * torch.sigmoid(gate * alpha)
    return (up + 1) * glu

# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""GptOssExperts layout: ``x @ W + b`` and interleaved even/odd SwiGLU."""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from peft.tuners.lora.layer import ParamWrapper
from torch.distributed.tensor import DTensor

from agilerl.algorithms.core.llm_ops.fused_lora import ROUTING_STATE


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


def _gpt_oss_expert_params(module: nn.Module) -> GptOssExpertParams | None:
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
    return _gpt_oss_expert_params(module) is not None


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


def _apply_gpt_oss_gate(
    gate_up: torch.Tensor, alpha: float, limit: float
) -> torch.Tensor:
    """Interleaved SwiGLU used by GptOssExperts (even gate, odd up)."""
    gate, up = gate_up[..., ::2], gate_up[..., 1::2]
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    glu = gate * torch.sigmoid(gate * alpha)
    return (up + 1) * glu


def gpt_oss_experts_local_forward(
    experts: nn.Module,
    hidden_states: torch.Tensor,
    router_indices: torch.Tensor,
    routing_weights: torch.Tensor,
    chain: dict[str, ParamWrapper] | None = None,
    adapters: dict[str, list[str]] | None = None,
    routing: Sequence[str] | None = None,
) -> torch.Tensor:
    """GptOssExperts forward with split-LoRA deltas on the matmul layout."""
    params = _gpt_oss_expert_params(experts)
    if params is None:
        msg = "Gpt-oss expert LoRA requires a GptOssExperts layout."
        raise RuntimeError(msg)
    # moe_lora imports this module to install the wrapper.
    from agilerl.algorithms.core.llm_ops import moe_lora as moe_lora_mod

    if chain is not None and adapters is None:
        adapters = {
            name: moe_lora_mod.resolve_adapters(wrapper)
            for name, wrapper in chain.items()
        }
    adapters = adapters or {}
    chain = chain or {}

    local_e = params.gate_up_proj.shape[0]
    top_k = router_indices.shape[-1]
    flat_experts = router_indices.reshape(-1)
    order = torch.argsort(flat_experts, stable=True)
    counts = torch.bincount(flat_experts, minlength=local_e)
    token_idx = torch.div(order, top_k, rounding_mode="floor")
    x = hidden_states[token_idx]
    routed_weights = routing_weights.reshape(-1)[order].unsqueeze(-1)
    row_ids: torch.Tensor | None = None
    id_map: dict[str, int] | None = None
    if routing is not None and len(set(routing)) > 1:
        row_ids, id_map = moe_lora_mod.token_adapter_ids(
            routing, hidden_states.shape[0], token_idx
        )

    offs = torch.cumsum(counts, dim=0).to(torch.int32) if x.is_cuda else None
    projected = expert_matmul_loop(
        x, params.gate_up_proj, counts, params.gate_up_proj_bias
    )
    for name in adapters.get("gate_up_proj", []):
        delta = moe_lora_mod.split_lora_delta(
            chain["gate_up_proj"], x, counts, name, offs, num_experts=local_e
        )
        if row_ids is not None and id_map is not None:
            mask = (row_ids == id_map[name]).to(delta.dtype).unsqueeze(-1)
            delta = delta * mask
        projected = projected + delta.to(projected.dtype)
    intermediate = _apply_gpt_oss_gate(projected, params.alpha, params.limit)
    down = expert_matmul_loop(
        intermediate, params.down_proj, counts, params.down_proj_bias
    )
    for name in adapters.get("down_proj", []):
        delta = moe_lora_mod.split_lora_delta(
            chain["down_proj"], intermediate, counts, name, offs, num_experts=local_e
        )
        if row_ids is not None and id_map is not None:
            mask = (row_ids == id_map[name]).to(delta.dtype).unsqueeze(-1)
            delta = delta * mask
        down = down + delta.to(down.dtype)

    result = torch.zeros_like(hidden_states)
    result.index_add_(0, token_idx, (down * routed_weights).to(result.dtype))
    return result


class GptOssExpertsLoraWrapper(ParamWrapper):
    """Split-LoRA ``ParamWrapper`` for GptOssExperts packed blocks."""

    _self_routed_lora = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_indices: torch.Tensor | None = None,
        routing_weights: torch.Tensor | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        if (
            args
            or kwargs
            or router_indices is None
            or routing_weights is None
            or hidden_states.dim() != 2
        ):
            return ParamWrapper.forward(
                self, hidden_states, router_indices, routing_weights, *args, **kwargs
            )
        # moe_lora imports this module to install the wrapper.
        from agilerl.algorithms.core.llm_ops import moe_lora as moe_lora_mod

        chain = moe_lora_mod.wrapper_chain(self)
        experts = self.get_base_layer()
        routing = ROUTING_STATE.get(self)
        if routing is not None and len(set(routing)) > 1:
            adapters = {
                name: moe_lora_mod.adapters_in_routing(wrapper, routing)
                for name, wrapper in chain.items()
            }
            mixed_routing: Sequence[str] | None = routing
        else:
            adapters = {
                name: moe_lora_mod.resolve_adapters(wrapper)
                for name, wrapper in chain.items()
            }
            mixed_routing = None
        if not any(adapters.values()):
            return experts(hidden_states, router_indices, routing_weights)
        if not is_gpt_oss_experts_module(experts):
            return ParamWrapper.forward(
                self, hidden_states, router_indices, routing_weights
            )
        return gpt_oss_experts_local_forward(
            experts,
            hidden_states,
            router_indices,
            routing_weights,
            chain=chain,
            adapters=adapters,
            routing=mixed_routing,
        )

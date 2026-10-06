# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Transposed packed-experts forward (``x @ W + b``) with split-LoRA deltas."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
from peft.tuners.lora.layer import ParamWrapper

from agilerl.architectures.gptoss import experts as gptoss_experts
from agilerl.lora.moe.adapters import (
    resolve_adapters,
    split_lora_delta,
    token_adapter_ids,
)

transposed_expert_params = gptoss_experts.gpt_oss_expert_params
apply_transposed_experts_gate = gptoss_experts.apply_gpt_oss_gate
expert_matmul_loop = gptoss_experts.expert_matmul_loop


def transposed_experts_local_forward(
    experts: nn.Module,
    hidden_states: torch.Tensor,
    router_indices: torch.Tensor,
    routing_weights: torch.Tensor,
    chain: dict[str, ParamWrapper] | None = None,
    adapters: dict[str, list[str]] | None = None,
    routing: Sequence[str] | None = None,
) -> torch.Tensor:
    """Transposed packed-experts forward with split-LoRA deltas on the matmul layout."""
    params = transposed_expert_params(experts)
    if params is None:
        msg = "Transposed experts LoRA requires a transposed packed-experts layout."
        raise RuntimeError(msg)
    if chain is not None and adapters is None:
        adapters = {name: resolve_adapters(wrapper) for name, wrapper in chain.items()}
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
        row_ids, id_map = token_adapter_ids(routing, hidden_states.shape[0], token_idx)

    offs = torch.cumsum(counts, dim=0).to(torch.int32) if x.is_cuda else None
    projected = expert_matmul_loop(
        x, params.gate_up_proj, counts, params.gate_up_proj_bias
    )
    for name in adapters.get("gate_up_proj", []):
        delta = split_lora_delta(
            chain["gate_up_proj"], x, counts, name, offs, num_experts=local_e
        )
        if row_ids is not None and id_map is not None:
            mask = (row_ids == id_map[name]).to(delta.dtype).unsqueeze(-1)
            delta = delta * mask
        projected = projected + delta.to(projected.dtype)
    intermediate = apply_transposed_experts_gate(projected, params.alpha, params.limit)
    down = expert_matmul_loop(
        intermediate, params.down_proj, counts, params.down_proj_bias
    )
    for name in adapters.get("down_proj", []):
        delta = split_lora_delta(
            chain["down_proj"], intermediate, counts, name, offs, num_experts=local_e
        )
        if row_ids is not None and id_map is not None:
            mask = (row_ids == id_map[name]).to(delta.dtype).unsqueeze(-1)
            delta = delta * mask
        down = down + delta.to(down.dtype)

    result = torch.zeros_like(hidden_states)
    result.index_add_(0, token_idx, (down * routed_weights).to(result.dtype))
    return result

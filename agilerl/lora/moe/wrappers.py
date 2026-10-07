# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Split-LoRA ``ParamWrapper`` subclasses for each packed-experts calling convention."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from peft.tuners.lora.layer import ParamWrapper

from agilerl.lora.fused import ROUTING_STATE
from agilerl.lora.moe.adapters import (
    adapters_in_routing,
    expert_counts,
    mixed_routing,
    resolve_adapters,
    split_lora_delta,
    token_adapter_ids,
    wrapper_chain,
)
from agilerl.lora.moe.grouped_gemm import group_offsets
from agilerl.lora.moe.layouts import (
    is_transposed_experts_module,
    routed_projection_names,
)
from agilerl.lora.moe.routed import routed_experts_local_forward
from agilerl.lora.moe.transposed import transposed_experts_local_forward


def _chain_adapters(
    outer: ParamWrapper, chain: dict[str, ParamWrapper]
) -> dict[str, list[str]]:
    """Adapters each chain link applies, routed by the outer wrapper when it has routing."""
    # Under EP the outer wrapper's routing covers this rank's rows only; inner
    # links still hold the whole batch's routing.
    routing = ROUTING_STATE.get(outer)
    if routing:
        return {
            name: adapters_in_routing(link, routing) for name, link in chain.items()
        }
    return {name: resolve_adapters(link) for name, link in chain.items()}


class SortedExpertsLoraWrapper(ParamWrapper):
    """Split-LoRA ``ParamWrapper`` for grouped linears taking expert-sorted rows."""

    _self_routed_lora = True
    token_index: torch.Tensor | None = None
    n_tokens: int | None = None

    def forward(
        self,
        x: torch.Tensor,
        expert_size: Sequence[int] | torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        routing = mixed_routing(self)
        row_ids: torch.Tensor | None = None
        id_map: dict[str, int] | None = None
        if routing is not None:
            adapters = adapters_in_routing(self, routing)
            token_idx = self.token_index
            n_tokens = self.n_tokens
            if token_idx is None or n_tokens is None:
                msg = (
                    "Mixed fused routing on sorted-experts LoRA needs "
                    "token_index from the gate."
                )
                raise RuntimeError(msg)
            if token_idx.shape[0] != x.shape[0]:
                msg = (
                    "Gate batch_index length "
                    f"{token_idx.shape[0]} does not match expert-sorted rows "
                    f"{x.shape[0]}."
                )
                raise ValueError(msg)
            row_ids, id_map = token_adapter_ids(routing, n_tokens, token_idx)
        else:
            adapters = resolve_adapters(self)
        base = self.base_layer
        result = base(x, expert_size, *args, **kwargs)
        if not adapters:
            return result
        counts = expert_counts(expert_size, self.num_experts)
        offs = group_offsets(counts, x.device) if x.is_cuda else None
        for name in adapters:
            if row_ids is None or id_map is None:
                split_lora_delta(self, x, counts, name, offs, destination=result)
                continue
            delta = split_lora_delta(self, x, counts, name, offs)
            mask = (row_ids == id_map[name]).to(delta.dtype).unsqueeze(-1)
            delta.mul_(mask)
            if delta.dtype != result.dtype:
                delta = delta.to(result.dtype)
            result.add_(delta)
        return result


class RoutedExpertsLoraWrapper(ParamWrapper):
    """Split-LoRA ``ParamWrapper`` for self-routing packed-experts modules."""

    _self_routed_lora = True
    # See ``recompute`` on :func:`routed_experts_local_forward`.
    recompute: bool = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
        *args: Any,
        already_grouped: bool | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if args or kwargs or hidden_states.dim() != 2:
            return ParamWrapper.forward(
                self, hidden_states, top_k_index, top_k_weights, *args, **kwargs
            )
        chain = wrapper_chain(self)
        experts = self.get_base_layer()
        routing = mixed_routing(self)
        adapters = _chain_adapters(self, chain)

        if not any(adapters.values()):
            return experts(hidden_states, top_k_index, top_k_weights)

        if routed_projection_names(experts) is None:
            return ParamWrapper.forward(self, hidden_states, top_k_index, top_k_weights)

        return routed_experts_local_forward(
            experts,
            hidden_states,
            top_k_index,
            top_k_weights,
            chain=chain,
            adapters=adapters,
            routing=routing,
            already_grouped=already_grouped,
            recompute=self.recompute,
        )


class TransposedExpertsLoraWrapper(ParamWrapper):
    """Split-LoRA ``ParamWrapper`` for transposed packed-experts blocks."""

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
        chain = wrapper_chain(self)
        experts = self.get_base_layer()
        routing = mixed_routing(self)
        adapters = _chain_adapters(self, chain)
        if not any(adapters.values()):
            return experts(hidden_states, router_indices, routing_weights)
        if not is_transposed_experts_module(experts):
            return ParamWrapper.forward(
                self, hidden_states, router_indices, routing_weights
            )
        return transposed_experts_local_forward(
            experts,
            hidden_states,
            router_indices,
            routing_weights,
            chain=chain,
            adapters=adapters,
            routing=routing,
        )

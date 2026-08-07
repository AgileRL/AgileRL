# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Split low-rank adapter execution for packed mixture-of-experts weights.

PEFT's ``ParamWrapper`` (``LoraConfig.target_parameters``) supports stacked 3D
expert weights but applies adapters by materializing the full-rank delta
``B @ A`` for every expert on every forward — an allocation the size of the
expert weights themselves, per wrapped parameter, per layer. The wrappers here
keep the low-rank factorization split instead: tokens are grouped per expert
and pushed through that expert's rank-``r`` slice of ``lora_A``/``lora_B``, so
the largest adapter intermediate is ``[tokens, r]``. Wrappers on modules
matching neither supported calling convention stay on PEFT's default path.
"""

from __future__ import annotations

import inspect
import logging
import warnings
from collections.abc import Sequence
from functools import cache
from typing import Any

import torch
import torch.nn as nn
from peft.tuners.lora.layer import ParamWrapper

from agilerl.algorithms.core.llm_ops.fused_lora import uniform_routed_adapter

logger = logging.getLogger(__name__)


@cache
def _grouped_mm_supported(device_index: int, dtype: torch.dtype) -> bool:
    """Whether ``torch._grouped_mm`` computes correct results (fwd and bwd, transposed views) here."""
    if not hasattr(torch, "_grouped_mm"):
        return False
    try:
        device = torch.device("cuda", device_index)
        generator = torch.Generator(device=device).manual_seed(0)
        x = torch.randn(8, 16, device=device, dtype=dtype, generator=generator)
        w = torch.randn(2, 4, 16, device=device, dtype=dtype, generator=generator)
        x = x.requires_grad_(True)
        w = w.requires_grad_(True)
        offs = torch.tensor([5, 8], device=device, dtype=torch.int32)
        out = torch._grouped_mm(x, w.transpose(-2, -1), offs=offs)
        reference = torch.cat([x[:5] @ w[0].mT, x[5:] @ w[1].mT])
        if not torch.allclose(out.float(), reference.float(), atol=1e-2):
            return False
        # square() materializes the incoming gradient; the op's backward
        # rejects the zero-stride expanded grad a bare sum() would feed it.
        out.square().sum().backward()
    except Exception:
        return False
    return x.grad is not None and w.grad is not None


def _use_grouped_mm(x: torch.Tensor) -> bool:
    """Whether the grouped-GEMM fast path applies to *x*'s device and dtype."""
    if not x.is_cuda:
        return False
    index = x.device.index
    if index is None:
        index = torch.cuda.current_device()
    return _grouped_mm_supported(index, x.dtype)


def _counts_list(counts: Sequence[int] | torch.Tensor) -> list[int]:
    """Per-expert row counts as a plain list (host sync only when needed)."""
    if isinstance(counts, torch.Tensor):
        return [int(count) for count in counts.tolist()]
    return [int(count) for count in counts]


def _counts_tensor(
    counts: Sequence[int] | torch.Tensor, device: torch.device
) -> torch.Tensor:
    """Per-expert row counts as a device tensor."""
    if isinstance(counts, torch.Tensor):
        return counts
    return torch.as_tensor(counts, device=device)


def _group_offsets(
    counts: Sequence[int] | torch.Tensor, device: torch.device
) -> torch.Tensor:
    """Cumulative per-expert row offsets in the layout ``torch._grouped_mm`` takes."""
    return torch.cumsum(_counts_tensor(counts, device), dim=0).to(torch.int32)


def _dims_aligned(itemsize: int, *dims: int) -> bool:
    """Whether row strides over these inner dims meet the op's 16-byte alignment."""
    return all(dim * itemsize % 16 == 0 for dim in dims)


def _is_partitioned(*tensors: torch.Tensor) -> bool:
    """Whether any tensor is an FSDP2 ``DTensor`` shard (not a dense local view)."""
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        return False
    return any(isinstance(tensor, DTensor) for tensor in tensors)


def _as_local_expert_weight(weight: torch.Tensor) -> torch.Tensor:
    """``to_local()`` for EP/FSDP ``DTensor`` expert weights; identity otherwise."""
    from agilerl.utils.expert_parallel import expert_local_tensor

    return expert_local_tensor(weight)


def _module_ep_degree(module: nn.Module) -> int:
    from agilerl.utils.expert_parallel import module_ep_degree

    return module_ep_degree(module)


def _grouped_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    counts: Sequence[int] | torch.Tensor,
    offs: torch.Tensor | None = None,
    *,
    copy_for_gmm: bool = False,
) -> torch.Tensor:
    """Per-expert linear over expert-sorted rows with a stacked ``[experts, out, in]`` weight."""
    if (
        x.dtype == weight.dtype
        and _dims_aligned(x.element_size(), weight.shape[1], weight.shape[2])
        and _use_grouped_mm(x)
    ):
        operand = weight.transpose(-2, -1)
        if copy_for_gmm:
            # grouped_mm is only validated for a plain last-two-dims transpose;
            # weights on other stride patterns are copied (rank-sized here).
            operand = operand.contiguous()
        if offs is None:
            offs = _group_offsets(counts, x.device)
        return torch._grouped_mm(x, operand, offs=offs)
    outputs = [
        nn.functional.linear(rows, weight[expert])
        for expert, rows in enumerate(x.split(_counts_list(counts)))
    ]
    return torch.cat(outputs)


def _forward_param_names(module: nn.Module) -> list[str]:
    """Positional parameter names of a module's ``forward``, excluding ``self``."""
    try:
        signature = inspect.signature(type(module).forward)
    except (TypeError, ValueError):
        return []
    return [name for name in signature.parameters if name != "self"]


def _is_sorted_experts_module(module: nn.Module) -> bool:
    """Whether *module* is a grouped linear over expert-sorted rows with a stacked 3D ``weight``."""
    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 3:
        return False
    return _forward_param_names(module)[:2] == ["inputs", "expert_size"]


def _routed_projection_names(module: nn.Module) -> tuple[str, bool] | None:
    """The up-projection parameter name and gatedness of a self-routing packed-experts block.

    Gated (``gate_up_proj``, Qwen3-MoE/granite: ``act(gate) * up``) and
    ungated (``up_proj``, NemotronH: ``act(up)``) variants are supported;
    per-expert biases or other layouts are off-convention and return ``None``.
    """
    down = getattr(module, "down_proj", None)
    if not isinstance(down, torch.Tensor) or down.ndim != 3:
        return None
    if not callable(getattr(module, "act_fn", None)):
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


def _is_routed_experts_module(module: nn.Module) -> bool:
    """Whether *module* is a self-routing packed-experts block in the transformers-5 convention."""
    return _routed_projection_names(module) is not None


def _expert_counts(
    expert_size: Sequence[int] | torch.Tensor, num_experts: int
) -> list[int]:
    """Normalize a per-expert row-count spec, validating the calling convention."""
    counts = _counts_list(expert_size)
    if len(counts) != num_experts:
        msg = (
            f"Expected {num_experts} per-expert counts, got {len(counts)}; "
            "the wrapped experts module does not follow the sorted-rows "
            "calling convention."
        )
        raise ValueError(msg)
    return counts


def _resolve_adapters(wrapper: ParamWrapper) -> list[str]:
    """Adapter names to apply on this forward, honoring fused routing and adapter state."""
    routed = uniform_routed_adapter(wrapper)
    if routed is not None:
        return [routed] if routed in wrapper.lora_A else []
    if wrapper.disable_adapters:
        if wrapper.merged:
            wrapper.unmerge()
        return []
    return [
        name
        for name in wrapper.active_adapters
        if name in wrapper.lora_A and name not in wrapper.merged_adapters
    ]


def _split_lora_delta(
    wrapper: ParamWrapper,
    x: torch.Tensor,
    counts: Sequence[int] | torch.Tensor,
    adapter: str,
    offs: torch.Tensor | None = None,
    *,
    num_experts: int | None = None,
) -> torch.Tensor:
    """Low-rank delta for expert-sorted rows without materializing per-expert full-rank weights.

    Under Expert Parallel, LoRA ``A`` is ``Shard(0)`` on expert-major ``[E*r, in]``
    and ``B`` is ``Shard(2)`` on ``[out, r, E]``; this path uses ``to_local()``
    slices for the local expert block and never gathers the full expert axis.
    """
    lora_a = wrapper.lora_A[adapter]
    lora_b = wrapper.lora_B[adapter]
    weight_a = lora_a.weight
    weight_b = lora_b.weight
    assert isinstance(weight_a, torch.Tensor)
    assert isinstance(weight_b, torch.Tensor)
    scaling = wrapper.scaling[adapter]
    rank = wrapper.r[adapter]
    local_a = _as_local_expert_weight(weight_a)
    local_b = _as_local_expert_weight(weight_b)
    x = x.to(local_a.dtype)

    # Prefer local grouped GEMM whenever we have dense local shards (EP or
    # post-gather). The Linear fallback is only for still-partitioned DTensors
    # that were not EP-sharded on dim 0 (non-EP FSDP leftover).
    if _is_partitioned(local_a, local_b):
        experts = num_experts if num_experts is not None else wrapper.num_experts
        total = x.shape[0]
        expert_ids = torch.repeat_interleave(
            torch.arange(experts, device=x.device),
            _counts_tensor(counts, x.device),
        )
        rows = torch.arange(total, device=x.device)
        a_full = lora_a(x).view(total, experts, rank)
        gated = torch.zeros(
            total, rank, experts, dtype=a_full.dtype, device=x.device
        )
        gated[rows, :, expert_ids] = a_full[rows, expert_ids]
        return lora_b(gated.reshape(total, rank * experts)) * scaling

    # Stacked PEFT layouts: A is ``[E*r, in]`` or already ``[E, r, in]`` local;
    # B is ``[out, E*r]`` / ``[out, r, E]``. Infer E from the local shard.
    if local_a.ndim == 3:
        a3 = local_a
        experts = a3.shape[0]
    else:
        experts = num_experts if num_experts is not None else wrapper.num_experts
        a3 = local_a.view(experts, rank, local_a.shape[1])
        experts = a3.shape[0]
    if local_b.ndim == 3:
        b3 = local_b
        if b3.shape[0] == experts:
            b_grouped = b3
        else:
            # ``[out, r, E]`` local on last dim
            b_grouped = b3.permute(2, 0, 1)
    else:
        b3 = local_b.view(local_b.shape[0], rank, experts)
        b_grouped = b3.permute(2, 0, 1)
    down = _grouped_linear(x, a3, counts, offs)
    up = _grouped_linear(down, b_grouped, counts, offs, copy_for_gmm=True)
    return up * scaling


def _routed_experts_local_forward(
    experts: nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    *,
    chain: dict[str, ParamWrapper] | None = None,
    adapters: dict[str, list[str]] | None = None,
) -> torch.Tensor:
    """Packed routed-experts forward on ``to_local()`` weights (EP-safe).

    When ``module_ep_degree(experts) > 1``, tokens are all-to-all dispatched to
    the ranks that own each expert, local GEMM (+ optional split LoRA) runs on
    ``E/ep`` experts, then outputs are combined. Never gathers full-E weights.
    """
    from agilerl.utils.expert_parallel import (
        module_ep_degree,
        module_ep_group,
        token_combine,
        token_dispatch,
    )

    projections = _routed_projection_names(experts)
    if projections is None:
        msg = "Routed experts module does not match a supported packed layout."
        raise RuntimeError(msg)
    up_name, gated = projections
    up_weight = _as_local_expert_weight(getattr(experts, up_name))
    down_weight = _as_local_expert_weight(experts.down_proj)
    act_fn = experts.act_fn
    assert isinstance(up_weight, torch.Tensor)
    assert isinstance(down_weight, torch.Tensor)

    ep_degree = module_ep_degree(experts)
    # Full expert count is ep * local_E when sharded; PEFT wrappers still know E.
    local_e = up_weight.shape[0]
    num_experts = local_e * ep_degree if ep_degree > 1 else local_e
    if chain is not None and adapters is None:
        adapters = {name: _resolve_adapters(w) for name, w in chain.items()}
    adapters = adapters or {}
    chain = chain or {}

    top_k = top_k_index.shape[-1]
    flat_experts = top_k_index.reshape(-1)
    order = torch.argsort(flat_experts, stable=True)
    counts = torch.bincount(flat_experts, minlength=num_experts)
    token_idx = torch.div(order, top_k, rounding_mode="floor")
    x = hidden_states[token_idx]
    routed_weights = top_k_weights.reshape(-1)[order].unsqueeze(-1)

    if ep_degree > 1:
        ep_group = module_ep_group(experts)
        if ep_group is None:
            msg = "Expert Parallel module is missing _ep_group; call shard_experts_on_ep."
            raise RuntimeError(msg)
        x, local_counts, state = token_dispatch(
            x,
            counts.to(torch.long),
            ep_group=ep_group,
            ep_degree=ep_degree,
            num_local_experts=local_e,
        )
        # Combine needs the pre-dispatch routed weights / token_idx; keep them.
        counts_for_gemm = local_counts
    else:
        if _is_partitioned(getattr(experts, up_name), experts.down_proj):
            msg = (
                "Split expert LoRA found FSDP2-sharded expert weights that are "
                "not gathered. Call :func:`~agilerl.utils.llm_utils.gather_params` "
                "on packed expert parameters before MoE wrapper upgrade so this "
                "forward can read dense packed weights, or enable Expert Parallel "
                "(ep > 1) so forwards use to_local() shards."
            )
            raise RuntimeError(msg)
        counts_for_gemm = counts
        state = None

    offs = (
        torch.cumsum(_counts_tensor(counts_for_gemm, x.device), dim=0).to(torch.int32)
        if x.is_cuda
        else None
    )
    projected = _grouped_linear(x, up_weight, counts_for_gemm, offs)
    for name in adapters.get(up_name, []):
        delta = _split_lora_delta(
            chain[up_name],
            x,
            counts_for_gemm,
            name,
            offs,
            num_experts=local_e,
        )
        projected = projected + delta.to(projected.dtype)
    if gated:
        gate, up = projected.chunk(2, dim=-1)
        intermediate = act_fn(gate) * up
    else:
        intermediate = act_fn(projected)
    down = _grouped_linear(intermediate, down_weight, counts_for_gemm, offs)
    for name in adapters.get("down_proj", []):
        delta = _split_lora_delta(
            chain["down_proj"],
            intermediate,
            counts_for_gemm,
            name,
            offs,
            num_experts=local_e,
        )
        down = down + delta.to(down.dtype)

    if state is not None:
        down = token_combine(down, state)

    result = torch.zeros_like(hidden_states)
    result.index_add_(0, token_idx, (down * routed_weights).to(result.dtype))
    return result


def _wrapper_chain(wrapper: ParamWrapper) -> dict[str, ParamWrapper]:
    """Map targeted parameter name to wrapper for a (possibly nested) wrapper chain."""
    chain: dict[str, ParamWrapper] = {}
    module: nn.Module = wrapper
    while isinstance(module, ParamWrapper):
        chain[module.parameter_name] = module
        module = module.base_layer
    return chain


class SortedExpertsLoraWrapper(ParamWrapper):
    """Split-LoRA ``ParamWrapper`` for grouped linears taking expert-sorted rows.

    Under Expert Parallel the caller dispatches tokens first so ``expert_size``
    / ``x`` cover only the local expert shard; base and LoRA weights are read
    via ``to_local()``.
    """

    _self_routed_lora = True

    def forward(
        self,
        x: torch.Tensor,
        expert_size: Sequence[int] | torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        adapters = _resolve_adapters(self)
        base = self.base_layer
        ep_degree = _module_ep_degree(base)
        if ep_degree > 1:
            # Local shard forward: grouped linear on to_local weights.
            weight = _as_local_expert_weight(base.weight)
            local_e = weight.shape[0]
            counts = _expert_counts(expert_size, local_e)
            offs = _group_offsets(counts, x.device) if x.is_cuda else None
            result = _grouped_linear(x, weight, counts, offs)
            for name in adapters:
                delta = _split_lora_delta(
                    self, x, counts, name, offs, num_experts=local_e
                )
                result = result + delta.to(result.dtype)
            return result

        result = base(x, expert_size, *args, **kwargs)
        if not adapters:
            return result
        counts = _expert_counts(expert_size, self.num_experts)
        offs = _group_offsets(counts, x.device) if x.is_cuda else None
        for name in adapters:
            delta = _split_lora_delta(self, x, counts, name, offs)
            result = result + delta.to(result.dtype)
        return result


class RoutedExpertsLoraWrapper(ParamWrapper):
    """Split-LoRA ``ParamWrapper`` for self-routing packed-experts modules.

    Under Expert Parallel (``ep > 1``) uses ``to_local()`` expert / LoRA shards
    and torch all-to-all dispatch/combine — never gathers full-E weights on CUDA.
    """

    _self_routed_lora = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        if args or kwargs or hidden_states.dim() != 2:
            return ParamWrapper.forward(
                self, hidden_states, top_k_index, top_k_weights, *args, **kwargs
            )
        chain = _wrapper_chain(self)
        experts = self.get_base_layer()
        adapters = {name: _resolve_adapters(w) for name, w in chain.items()}
        ep_degree = _module_ep_degree(experts)

        if not any(adapters.values()) and ep_degree <= 1:
            return experts(hidden_states, top_k_index, top_k_weights)

        projections = _routed_projection_names(experts)
        if projections is None:
            return ParamWrapper.forward(self, hidden_states, top_k_index, top_k_weights)

        return _routed_experts_local_forward(
            experts,
            hidden_states,
            top_k_index,
            top_k_weights,
            chain=chain,
            adapters=adapters,
        )


def upgrade_moe_param_wrappers(model: nn.Module) -> int:
    """Swap eligible ``ParamWrapper`` instances to split-LoRA execution, returning how many."""
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
        chain = _wrapper_chain(module)
        base = module.get_base_layer()
        projections = _routed_projection_names(base)
        if (
            len(chain) == 1
            and module.parameter_name == "weight"
            and _is_sorted_experts_module(base)
        ):
            module.__class__ = SortedExpertsLoraWrapper
            upgraded += 1
        elif projections is not None and set(chain) <= {projections[0], "down_proj"}:
            module.__class__ = RoutedExpertsLoraWrapper
            upgraded += 1
        elif module.get_param().ndim == 3:
            fallbacks.append(name)
    if fallbacks:
        warnings.warn(
            "Packed-experts LoRA wrappers on unrecognized module conventions "
            "stay on PEFT's delta-materializing forward (memory-hungry): "
            f"{fallbacks}.",
            stacklevel=2,
        )
    return upgraded


def moe_expert_target_parameters(model: nn.Module) -> list[str]:
    """Parameter-path suffixes of packed expert weights for ``LoraConfig.target_parameters``."""
    suffixes: set[str] = set()
    for name, module in model.named_modules():
        prefix = ".".join(name.split(".")[-2:])
        if _is_sorted_experts_module(module):
            suffixes.add(f"{prefix}.weight")
        elif (projections := _routed_projection_names(module)) is not None:
            suffixes.add(f"{prefix}.{projections[0]}")
            suffixes.add(f"{prefix}.down_proj")
    return sorted(suffixes)

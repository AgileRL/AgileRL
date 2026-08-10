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
from typing import Any

import torch
import torch.nn as nn
from peft.tuners.lora.layer import ParamWrapper

from agilerl.algorithms.core.llm_ops.fused_lora import uniform_routed_adapter

logger = logging.getLogger(__name__)


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


def _is_partitioned(*tensors: torch.Tensor) -> bool:
    """Whether any tensor is a ZeRO-3 shard that is not currently gathered.

    Inside a ZeRO-3 leaf module's forward (see
    :func:`mark_expert_wrappers_as_zero3_leaves`) partitioned parameters are
    gathered and report ``ds_status`` AVAILABLE, so raw reads see full data.
    """
    for tensor in tensors:
        if not hasattr(tensor, "ds_id"):
            continue
        status = getattr(tensor, "ds_status", None)
        if getattr(status, "name", None) != "AVAILABLE":
            return True
    return False


def _expert_linear_loop(
    x: torch.Tensor,
    weight: torch.Tensor,
    counts: Sequence[int] | torch.Tensor,
) -> torch.Tensor:
    """Per-expert ``F.linear`` over expert-sorted rows with a stacked ``[experts, out, in]`` weight."""
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


def _lora_compute_dtype(dtype: torch.dtype) -> torch.dtype:
    """fp32 for low-rank GEMMs when activations are fp16/bf16."""
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def _split_lora_delta(
    wrapper: ParamWrapper,
    x: torch.Tensor,
    counts: Sequence[int] | torch.Tensor,
    adapter: str,
) -> torch.Tensor:
    """Low-rank delta for expert-sorted rows without materializing per-expert full-rank weights."""
    lora_a = wrapper.lora_A[adapter]
    lora_b = wrapper.lora_B[adapter]
    weight_a = lora_a.weight
    weight_b = lora_b.weight
    assert isinstance(weight_a, torch.Tensor)
    assert isinstance(weight_b, torch.Tensor)
    scaling = wrapper.scaling[adapter]
    num_experts = wrapper.num_experts
    rank = wrapper.r[adapter]
    out_dtype = x.dtype
    compute_dtype = _lora_compute_dtype(out_dtype)
    x = x.to(compute_dtype)

    if _is_partitioned(weight_a, weight_b):
        # ZeRO-3 shards are only gathered by module pre-forward hooks, so the
        # adapter Linears run as modules and each token keeps only its own
        # expert's block: lora_A rows are expert-major, lora_B columns rank-major.
        total = x.shape[0]
        expert_ids = torch.repeat_interleave(
            torch.arange(num_experts, device=x.device),
            _counts_tensor(counts, x.device),
        )
        rows = torch.arange(total, device=x.device)
        a_full = lora_a(x.to(weight_a.dtype)).view(total, num_experts, rank)
        gated = torch.zeros(
            total, rank, num_experts, dtype=a_full.dtype, device=x.device
        )
        gated[rows, :, expert_ids] = a_full[rows, expert_ids]
        delta = lora_b(gated.reshape(total, rank * num_experts)) * scaling
        return delta.to(out_dtype)

    a3 = weight_a.view(num_experts, rank, weight_a.shape[1]).to(compute_dtype)
    b3 = weight_b.view(weight_b.shape[0], rank, num_experts).to(compute_dtype)
    down = _expert_linear_loop(x, a3, counts)
    up = _expert_linear_loop(down, b3.permute(2, 0, 1), counts)
    return (up * scaling).to(out_dtype)


def _routed_lora_residual(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    chain: dict[str, ParamWrapper],
    experts: nn.Module,
    adapters: dict[str, list[str]],
) -> torch.Tensor:
    """LoRA-only residual for a routed experts block (``full_lora - native_base``)."""
    projections = _routed_projection_names(experts)
    if projections is None:
        msg = "Routed expert LoRA residual requires a recognized projection layout."
        raise RuntimeError(msg)
    up_name, gated = projections
    up_weight = getattr(experts, up_name)
    down_weight = experts.down_proj
    act_fn = experts.act_fn
    assert isinstance(up_weight, torch.Tensor)
    assert isinstance(down_weight, torch.Tensor)
    assert not isinstance(act_fn, torch.Tensor)
    if _is_partitioned(up_weight, down_weight):
        msg = (
            "Split expert LoRA found ZeRO-3 partitioned expert weights "
            "that are not gathered. The expert wrappers must be ZeRO-3 "
            "leaf modules (mark_expert_wrappers_as_zero3_leaves; applied "
            "automatically at algorithm init) so the subtree gathers "
            "before this forward reads the packed weights."
        )
        raise RuntimeError(msg)

    num_experts = up_weight.shape[0]
    top_k = top_k_index.shape[-1]
    flat_experts = top_k_index.reshape(-1)
    order = torch.argsort(flat_experts, stable=True)
    counts = torch.bincount(flat_experts, minlength=num_experts)
    token_idx = torch.div(order, top_k, rounding_mode="floor")
    x = hidden_states[token_idx]

    projected_base = _expert_linear_loop(x, up_weight, counts)
    projected = projected_base.to(_lora_compute_dtype(projected_base.dtype))
    for name in adapters.get(up_name, []):
        delta = _split_lora_delta(chain[up_name], x, counts, name)
        projected = projected + delta.to(projected.dtype)
    projected = projected.to(projected_base.dtype)

    if gated:
        gate_b, up_b = projected_base.chunk(2, dim=-1)
        intermediate_base = act_fn(gate_b) * up_b
        gate, up = projected.chunk(2, dim=-1)
        intermediate = act_fn(gate) * up
    else:
        intermediate_base = act_fn(projected_base)
        intermediate = act_fn(projected)

    residual = _expert_linear_loop(
        intermediate - intermediate_base, down_weight, counts
    )
    residual = residual.to(_lora_compute_dtype(residual.dtype))
    for name in adapters.get("down_proj", []):
        delta = _split_lora_delta(chain["down_proj"], intermediate, counts, name)
        residual = residual + delta.to(residual.dtype)

    routed_weights = top_k_weights.reshape(-1)[order].unsqueeze(-1)
    result = torch.zeros_like(hidden_states)
    result.index_add_(
        0,
        token_idx,
        (residual.to(hidden_states.dtype) * routed_weights).to(result.dtype),
    )
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
    """Split-LoRA ``ParamWrapper`` for grouped linears taking expert-sorted rows."""

    _self_routed_lora = True

    def forward(
        self,
        x: torch.Tensor,
        expert_size: Sequence[int] | torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        adapters = _resolve_adapters(self)
        result = self.base_layer(x, expert_size, *args, **kwargs)
        if not adapters:
            return result
        counts = _expert_counts(expert_size, self.num_experts)
        for name in adapters:
            delta = _split_lora_delta(self, x, counts, name)
            result = result + delta.to(result.dtype)
        return result


class RoutedExpertsLoraWrapper(ParamWrapper):
    """Split-LoRA ``ParamWrapper`` for self-routing packed-experts modules."""

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
        base = experts(hidden_states, top_k_index, top_k_weights)
        if not any(adapters.values()):
            return base
        if _routed_projection_names(experts) is None:
            return ParamWrapper.forward(self, hidden_states, top_k_index, top_k_weights)
        delta = _routed_lora_residual(
            hidden_states, top_k_index, top_k_weights, chain, experts, adapters
        )
        return base + delta.to(base.dtype)


def mark_expert_wrappers_as_zero3_leaves(model: nn.Module) -> int:
    """Mark upgraded expert wrappers as DeepSpeed ZeRO-3 leaf modules, returning how many.

    A leaf module's whole parameter subtree — adapter Linears and the packed
    expert weights beneath the wrapper — is gathered at its pre-forward, so
    the split forward's raw weight reads see full tensors (and MoE avoids
    per-parameter gather storms). Call before ``deepspeed.initialize``.
    """
    wrapper_classes = (SortedExpertsLoraWrapper, RoutedExpertsLoraWrapper)
    count = sum(isinstance(m, wrapper_classes) for m in model.modules())
    if not count:
        return 0
    from deepspeed.utils import set_z3_leaf_modules

    set_z3_leaf_modules(model, list(wrapper_classes))
    return count


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

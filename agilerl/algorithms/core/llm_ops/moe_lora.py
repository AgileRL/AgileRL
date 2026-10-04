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
matching none of the supported calling conventions stay on PEFT's default path.
"""

from __future__ import annotations

import inspect
import logging
import warnings
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from functools import cache
from types import MethodType
from typing import Any

import torch
import torch.nn as nn
from peft.tuners.lora.layer import ParamWrapper
from torch.autograd.function import once_differentiable
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointWrapper,
)
from torch.distributed.tensor import DTensor
from transformers.activations import get_activation
from transformers.modeling_layers import GradientCheckpointingLayer

from agilerl.algorithms.core.llm_ops.fused_lora import (
    ROUTING_STATE,
    patch_lora_for_fused_forward,
    uniform_routed_adapter,
)
from agilerl.architectures.gptoss import experts as gptoss_experts

is_transposed_experts_module = gptoss_experts.is_gpt_oss_experts_module
transposed_expert_params = gptoss_experts.gpt_oss_expert_params
apply_transposed_experts_gate = gptoss_experts.apply_gpt_oss_gate
expert_matmul_loop = gptoss_experts.expert_matmul_loop

logger = logging.getLogger(__name__)


@cache
def grouped_mm_supported(device_index: int, dtype: torch.dtype) -> bool:
    """Whether ``torch._grouped_mm`` computes correct results (fwd and bwd, transposed views) here."""
    if not hasattr(torch, "_grouped_mm"):
        return False
    # The first call can run inside a checkpointed or no_grad forward. The
    # probe needs its own backward: grad on, and identity hooks keep its saved
    # tensors out of the checkpoint so recompute matches.
    with (
        torch.enable_grad(),
        torch.autograd.graph.saved_tensors_hooks(lambda t: t, lambda t: t),
    ):
        try:
            device = torch.device("cuda", device_index)
            generator = torch.Generator(device=device).manual_seed(0)
            # Both inner dims stay 16-byte aligned in bf16, including after transpose.
            x = torch.randn(8, 16, device=device, dtype=dtype, generator=generator)
            w = torch.randn(2, 16, 16, device=device, dtype=dtype, generator=generator)
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
    return grouped_mm_supported(index, x.dtype)


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


# A full LoRA up-projection sits beside the expert activation and does not fit.
GROUPED_LINEAR_CHUNK_BYTES = 16 * 1024 * 1024

# Widest [rows, features] activation of one routed-expert row chunk. A chunk's
# backward holds about a dozen buffers that size (peak near 0.8 GiB). Each
# chunk adds a fixed set of small kernel launches that bound the step on the host.
ROUTED_EXPERT_CHUNK_BYTES = 64 * 1024 * 1024


def _group_offsets(
    counts: Sequence[int] | torch.Tensor, device: torch.device
) -> torch.Tensor:
    """Cumulative per-expert row offsets in the layout ``torch._grouped_mm`` takes."""
    return torch.cumsum(_counts_tensor(counts, device), dim=0).to(torch.int32)


def _dims_aligned(itemsize: int, *dims: int) -> bool:
    """Whether row strides over these inner dims meet the op's 16-byte alignment."""
    return all(dim * itemsize % 16 == 0 for dim in dims)


def _grouped_mm_operand_ready(mat: torch.Tensor) -> bool:
    """Whether ``mat`` has a last-two-dims layout ``_grouped_mm`` accepts."""
    if mat.dim() != 3:
        return False
    alignment = 16 // mat.element_size()
    if mat.is_cuda and (mat.data_ptr() % 16 != 0 or mat.stride(0) % alignment != 0):
        return False
    stride_row, stride_col = mat.stride(-2), mat.stride(-1)
    size_row, size_col = mat.shape[-2], mat.shape[-1]
    column_major = (
        stride_row == 1
        and stride_col >= max(1, size_row)
        and stride_col % alignment == 0
    )
    row_major = (
        stride_col == 1
        and stride_row >= max(1, size_col)
        and stride_row % alignment == 0
    )
    return column_major or row_major


def _routed_experts_act_fn(
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


def _grouped_linear(
    x: torch.Tensor | nn.Module,
    weight: torch.Tensor,
    counts: Sequence[int] | torch.Tensor,
    offs: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-expert linear over expert-sorted rows with a stacked ``[experts, out, in]`` weight.

    Bound as sorted-experts ``forward(inputs, expert_size)``: ``x`` is the module
    and ``weight`` is ``x.weight``.
    """
    if isinstance(x, nn.Module):
        module_weight = x.weight
        if not isinstance(module_weight, torch.Tensor):
            msg = "expert module weight must be a Tensor"
            raise TypeError(msg)
        x = weight
        weight = module_weight
    if not isinstance(x, torch.Tensor):
        msg = "grouped linear input must be a Tensor"
        raise TypeError(msg)
    if isinstance(weight, DTensor):
        weight = weight.to_local()
    if (
        weight.dim() == 3
        and x.dtype == weight.dtype
        and _dims_aligned(x.element_size(), weight.shape[1], weight.shape[2])
        and _use_grouped_mm(x)
    ):
        operand = weight.transpose(-2, -1)
        # A plain transpose is a layout the op accepts. Other strides are copied.
        if not _grouped_mm_operand_ready(operand):
            operand = operand.contiguous()
        if offs is None:
            offs = _group_offsets(counts, x.device)
        return torch._grouped_mm(x, operand, offs=offs)
    # FSDP's gathered expert block is a narrow view. cuBLAS rejects that
    # stride, and a zero-row expert is an empty gemm it also rejects.
    weight = weight.contiguous()
    pieces = []
    for expert, rows in enumerate(x.split(_counts_list(counts))):
        if rows.shape[0] == 0:
            continue
        pieces.append(nn.functional.linear(rows, weight[expert]))
    if not pieces:
        return x.new_empty((0, weight.shape[1]))
    return torch.cat(pieces)


def _grouped_matmul(
    x: torch.Tensor,
    weight: torch.Tensor,
    counts: Sequence[int],
    offs: torch.Tensor,
) -> torch.Tensor:
    """Per-expert ``rows @ weight[e]`` over expert-sorted rows with a stacked ``[experts, in, out]`` weight."""
    if (
        x.dtype == weight.dtype
        and _dims_aligned(x.element_size(), weight.shape[1], weight.shape[2])
        and _use_grouped_mm(x)
        and _grouped_mm_operand_ready(weight)
    ):
        return torch._grouped_mm(x, weight, offs=offs)
    pieces = [
        rows @ weight[expert]
        for expert, rows in enumerate(x.split(counts))
        if rows.shape[0] > 0
    ]
    if not pieces:
        return x.new_empty((0, weight.shape[2]))
    return torch.cat(pieces)


def _iter_expert_row_chunks(
    counts: Sequence[int] | torch.Tensor,
    max_rows: int,
) -> Iterator[tuple[int, int, list[int], int, int]]:
    """Yield ``(start_row, taken, local_counts, start_expert, end_expert)``."""
    remaining = _counts_list(counts)
    total = sum(remaining)
    row = 0
    expert = 0
    n_experts = len(remaining)
    while row < total:
        start_row = row
        start_expert = expert
        taken = 0
        local_counts: list[int] = []
        end_expert = start_expert
        while expert < n_experts and taken < max_rows:
            left = remaining[expert]
            if left == 0:
                local_counts.append(0)
                expert += 1
                end_expert = expert
                continue
            use = min(left, max_rows - taken)
            local_counts.append(use)
            remaining[expert] = left - use
            taken += use
            if remaining[expert] == 0:
                expert += 1
                end_expert = expert
            else:
                end_expert = expert + 1
                break
        if taken == 0:
            break
        yield start_row, taken, local_counts, start_expert, end_expert
        row += taken


def _add_grouped_linear(
    destination: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    counts: Sequence[int] | torch.Tensor,
    scaling: float | torch.Tensor,
) -> None:
    """Add a grouped GEMM into ``destination`` in row chunks.

    Each chunk's output stays within ``GROUPED_LINEAR_CHUNK_BYTES``.
    """
    if x.shape[0] == 0:
        return
    row_bytes = weight.shape[1] * x.element_size()
    max_rows = max(1, GROUPED_LINEAR_CHUNK_BYTES // max(row_bytes, 1))
    for (
        start_row,
        taken,
        local_counts,
        start_expert,
        end_expert,
    ) in _iter_expert_row_chunks(counts, max_rows):
        chunk = _grouped_linear(
            x[start_row : start_row + taken],
            weight[start_expert:end_expert],
            local_counts,
        )
        if chunk.dtype != destination.dtype:
            chunk = chunk.to(dtype=destination.dtype)
        chunk.mul_(scaling)
        destination[start_row : start_row + taken].add_(chunk)


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
    """Whether *module* is a self-routing packed-experts block."""
    return _routed_projection_names(module) is not None


def _is_packed_experts_module(module: nn.Module) -> bool:
    """Whether *module* is a packed expert stack (routed, sorted, or transposed)."""
    return (
        _is_routed_experts_module(module)
        or _is_sorted_experts_module(module)
        or is_transposed_experts_module(module)
    )


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


def adapters_in_routing(wrapper: ParamWrapper, routing: Sequence[str]) -> list[str]:
    """Adapter names from *routing* that this wrapper actually hosts."""
    return [name for name in dict.fromkeys(routing) if name in wrapper.lora_A]


def token_adapter_ids(
    routing: Sequence[str], n_rows: int, token_idx: torch.Tensor
) -> tuple[torch.Tensor, dict[str, int]]:
    """Expand per-sample fused routing to tokens, then permute into expert-sorted order.

    ``token_idx`` is the gate's ``batch_index``: ``token_idx[i]`` is the original
    token for expert-sorted row ``i``.
    """
    factor, remainder = divmod(n_rows, len(routing))
    if remainder:
        msg = (
            f"Fused adapter routing covers {len(routing)} rows but the "
            f"experts input's leading dimension is {n_rows}."
        )
        raise ValueError(msg)
    expanded = [name for name in routing for _ in range(factor)]
    name_to_id = {name: index for index, name in enumerate(dict.fromkeys(expanded))}
    table = torch.tensor(
        [name_to_id[name] for name in expanded],
        device=token_idx.device,
        dtype=torch.int64,
    )
    return table[token_idx], name_to_id


def resolve_adapters(wrapper: ParamWrapper) -> list[str]:
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


def _stacked_lora_weights(
    wrapper: ParamWrapper,
    adapter: str,
    dtype: torch.dtype,
    num_experts: int | None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """LoRA A as ``[E, r, in]`` and contiguous B as ``[E, out, r]`` in ``dtype``, or ``None`` for DTensors."""
    weight_a = wrapper.lora_A[adapter].weight
    weight_b = wrapper.lora_B[adapter].weight
    assert isinstance(weight_a, torch.Tensor)
    assert isinstance(weight_b, torch.Tensor)
    if isinstance(weight_a, DTensor) or isinstance(weight_b, DTensor):
        return None
    rank = wrapper.r[adapter]
    # Stacked PEFT layouts: A is ``[E*r, in]`` or ``[E, r, in]``;
    # B is ``[out, E*r]`` or ``[out, r, E]``.
    if weight_a.ndim == 3:
        a3 = weight_a
    else:
        experts = num_experts if num_experts is not None else wrapper.num_experts
        a3 = weight_a.view(experts, rank, weight_a.shape[1])
    experts = a3.shape[0]
    if weight_b.ndim == 3:
        if weight_b.shape[0] == experts:
            b_grouped = weight_b
        else:
            # ``[out, r, E]`` with experts on the last dim
            b_grouped = weight_b.permute(2, 0, 1)
    else:
        b_grouped = weight_b.view(weight_b.shape[0], rank, experts).permute(2, 0, 1)
    # The multiply stays in the activation dtype. A wider LoRA weight is cast.
    # The permuted B has strides no GEMM accepts; one copy serves every chunk.
    return (
        a3.to(dtype=dtype),
        b_grouped.to(dtype=dtype, memory_format=torch.contiguous_format),
    )


def _partitioned_lora_delta(
    wrapper: ParamWrapper,
    x: torch.Tensor,
    adapter: str,
    expert_ids: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """LoRA delta through the adapter ``Linear`` modules, for still-partitioned DTensor weights."""
    rank = wrapper.r[adapter]
    total = x.shape[0]
    rows = torch.arange(total, device=x.device)
    a_full = wrapper.lora_A[adapter](x).view(total, num_experts, rank)
    gated = torch.zeros(total, rank, num_experts, dtype=a_full.dtype, device=x.device)
    gated[rows, :, expert_ids] = a_full[rows, expert_ids]
    delta = wrapper.lora_B[adapter](gated.reshape(total, rank * num_experts))
    return delta * wrapper.scaling[adapter]


def split_lora_delta(
    wrapper: ParamWrapper,
    x: torch.Tensor,
    counts: Sequence[int] | torch.Tensor,
    adapter: str,
    offs: torch.Tensor | None = None,
    num_experts: int | None = None,
    destination: torch.Tensor | None = None,
) -> torch.Tensor:
    """Low-rank delta for expert-sorted rows without materializing per-expert full-rank weights."""
    stacked = _stacked_lora_weights(wrapper, adapter, x.dtype, num_experts)
    # Prefer the grouped GEMM on dense weights. The Linear fallback is for
    # still-partitioned DTensors (FSDP leftover outside a rooted forward).
    if stacked is None:
        experts = num_experts if num_experts is not None else wrapper.num_experts
        expert_ids = torch.repeat_interleave(
            torch.arange(experts, device=x.device),
            _counts_tensor(counts, x.device),
        )
        delta = _partitioned_lora_delta(wrapper, x, adapter, expert_ids, experts)
        if destination is None:
            return delta
        if delta.dtype != destination.dtype:
            delta = delta.to(dtype=destination.dtype)
        destination.add_(delta)
        return destination

    a3, b3 = stacked
    scaling = wrapper.scaling[adapter]
    down = _grouped_linear(x, a3, counts, offs)
    if destination is None:
        up = _grouped_linear(down, b3, counts, offs)
        # ``up`` is a fresh GEMM output. Scaling it in place skips a second full copy.
        return up.mul_(scaling)
    _add_grouped_linear(destination, down, b3, counts, scaling)
    return destination


@dataclass(frozen=True)
class ExpertLora:
    """One adapter on a routed-experts projection, with weights prepared once per forward."""

    wrapper: ParamWrapper
    adapter: str
    # ``None`` while the LoRA weights are partitioned DTensors.
    stacked: tuple[torch.Tensor, torch.Tensor] | None
    # Adapter id in the per-row ids under mixed routing, else ``None``.
    row_id: int | None


def _low_rank_delta(
    rows: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    counts: list[int],
    offs: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    """Scaled ``rows @ A[e]^T @ B[e]^T`` for expert-sorted rows with stacked ``[E, r, in]`` / ``[E, out, r]`` factors."""
    down = _grouped_linear(rows, lora_a, counts, offs)
    return _grouped_linear(down, lora_b, counts, offs).mul_(scaling)


def _chunk_offsets(
    group_ends: torch.Tensor, experts: slice, start_row: int, stop: int
) -> torch.Tensor:
    """Grouped-GEMM offsets of one row chunk, kept on device so no host sync runs."""
    return (group_ends[experts].clamp(max=stop) - start_row).to(torch.int32)


def _expert_activation(
    projected: torch.Tensor,
    act_fn: Callable[[torch.Tensor], torch.Tensor],
    gated: bool,
) -> torch.Tensor:
    """``act(gate) * up`` for a gated up-projection, else ``act(up)``."""
    if gated:
        gate, up = projected.chunk(2, dim=-1)
        return act_fn(gate) * up
    return act_fn(projected)


def _add_expert_loras(
    destination: torch.Tensor,
    rows: torch.Tensor,
    loras: Sequence[ExpertLora],
    counts: list[int],
    offs: torch.Tensor,
    experts: slice,
    num_experts: int,
    row_ids: torch.Tensor | None,
) -> None:
    """Add each adapter's low-rank delta for one chunk of expert-sorted rows."""
    for lora in loras:
        if lora.stacked is None:
            expert_ids = torch.repeat_interleave(
                torch.arange(experts.start, experts.stop), torch.tensor(counts)
            ).to(rows.device)
            delta = _partitioned_lora_delta(
                lora.wrapper, rows, lora.adapter, expert_ids, num_experts
            )
        else:
            a3, b3 = lora.stacked
            delta = _low_rank_delta(
                rows,
                a3[experts],
                b3[experts],
                counts,
                offs,
                lora.wrapper.scaling[lora.adapter],
            )
        if lora.row_id is not None:
            assert row_ids is not None
            delta.mul_((row_ids == lora.row_id).to(delta.dtype).unsqueeze(-1))
        if delta.dtype != destination.dtype:
            delta = delta.to(dtype=destination.dtype)
        destination.add_(delta)


class ScatterRows(torch.autograd.Function):
    """Add source rows into a buffer."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        base: torch.Tensor,
        index: torch.Tensor,
        source: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(index)
        ctx.mark_dirty(base)
        base.index_add_(0, index, source)
        return base

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_out: torch.Tensor,
    ) -> tuple[torch.Tensor, None, torch.Tensor]:
        (index,) = ctx.saved_tensors
        return grad_out, None, grad_out.index_select(0, index)


def _scatter_routed_expert_chunks(
    hidden_states: torch.Tensor,
    x: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    act_fn: Callable[[torch.Tensor], torch.Tensor],
    gated: bool,
    counts: torch.Tensor,
    token_idx: torch.Tensor,
    routed_weights: torch.Tensor,
    up_loras: Sequence[ExpertLora],
    down_loras: Sequence[ExpertLora],
    row_ids: torch.Tensor | None,
) -> torch.Tensor:
    """Run the routed expert forward in row chunks and scatter into the layer output.

    Expanded expert rows are ``top_k`` times the token count, so the full up
    and down activations do not fit beside the gathered hidden states.
    """
    result = torch.zeros_like(hidden_states)
    num_experts = up_weight.shape[0]
    row_bytes = max(up_weight.shape[1], down_weight.shape[1]) * x.element_size()
    max_rows = max(1, ROUTED_EXPERT_CHUNK_BYTES // max(row_bytes, 1))
    plan = list(_iter_expert_row_chunks(counts, max_rows))
    sizes = [taken for _, taken, _, _, _ in plan]
    # Chunk offsets stay on device so the loop issues no host sync.
    group_ends = torch.cumsum(counts, dim=0)
    # Split once so backward concatenates the chunk grads into one buffer.
    chunks = zip(plan, x.split(sizes), routed_weights.split(sizes), strict=True)
    for (
        start_row,
        taken,
        local_counts,
        start_expert,
        end_expert,
    ), rows, weights in chunks:
        stop = start_row + taken
        experts = slice(start_expert, end_expert)
        offs = _chunk_offsets(group_ends, experts, start_row, stop)
        chunk_ids = None if row_ids is None else row_ids[start_row:stop]
        projected = _grouped_linear(rows, up_weight[experts], local_counts, offs)
        _add_expert_loras(
            projected,
            rows,
            up_loras,
            local_counts,
            offs,
            experts,
            num_experts,
            chunk_ids,
        )
        intermediate = _expert_activation(projected, act_fn, gated)
        del projected
        down = _grouped_linear(intermediate, down_weight[experts], local_counts, offs)
        _add_expert_loras(
            down,
            intermediate,
            down_loras,
            local_counts,
            offs,
            experts,
            num_experts,
            chunk_ids,
        )
        del intermediate
        down.mul_(weights)
        if down.dtype != result.dtype:
            down = down.to(dtype=result.dtype)
        result = ScatterRows.apply(result, token_idx[start_row:stop], down)
    return result


@dataclass(frozen=True)
class AdapterSlot:
    """Scaling and mixed-routing row id of one adapter inside :class:`LoraExpertsFunction`."""

    scaling: float
    # Adapter id in the per-row ids under mixed routing, else ``None``.
    row_id: int | None


def _slot_factors(
    factors: Sequence[torch.Tensor],
    slots: Sequence[AdapterSlot],
    experts: slice,
    row_ids: torch.Tensor | None,
    dtype: torch.dtype,
) -> list[tuple[torch.Tensor, torch.Tensor, float, torch.Tensor | None]]:
    """Per-adapter ``(A, B, scaling, row_mask)`` for one chunk; ``factors`` alternate A and B."""
    adapters = []
    for slot, lora_a, lora_b in zip(slots, factors[::2], factors[1::2], strict=True):
        mask = None
        if slot.row_id is not None:
            assert row_ids is not None
            mask = (row_ids == slot.row_id).to(dtype).unsqueeze(-1)
        adapters.append((lora_a[experts], lora_b[experts], slot.scaling, mask))
    return adapters


def _adapter_delta(
    rows: torch.Tensor,
    adapters: Sequence[tuple[torch.Tensor, torch.Tensor, float, torch.Tensor | None]],
    counts: list[int],
    offs: torch.Tensor,
) -> torch.Tensor | None:
    """Summed low-rank delta of every adapter for one chunk, or ``None`` without adapters."""
    total: torch.Tensor | None = None
    for lora_a, lora_b, scaling, mask in adapters:
        delta = _low_rank_delta(rows, lora_a, lora_b, counts, offs, scaling)
        if mask is not None:
            delta.mul_(mask)
        total = delta if total is None else total.add_(delta)
    return total


class LoraExpertsFunction(torch.autograd.Function):
    """Routed experts on frozen packed weights with split LoRA, recomputing the up-projection in backward.

    Saves the token rows, routing, and LoRA factors only. Rows are gathered and
    outputs scattered one expert chunk at a time, so neither the expert-sorted
    copy nor any ``[rows, intermediate]`` activation outlives its chunk.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        hidden_states: torch.Tensor,
        token_idx: torch.Tensor,
        routed_weights: torch.Tensor,
        group_ends: torch.Tensor,
        row_ids: torch.Tensor | None,
        up_weight: torch.Tensor,
        down_weight: torch.Tensor,
        act_fn: Callable[[torch.Tensor], torch.Tensor],
        gated: bool,
        plan: list[tuple[int, int, list[int], int, int]],
        up_slots: tuple[AdapterSlot, ...],
        down_slots: tuple[AdapterSlot, ...],
        *factors: torch.Tensor,
    ) -> torch.Tensor:
        up_factors = factors[: 2 * len(up_slots)]
        down_factors = factors[2 * len(up_slots) :]
        result = torch.zeros_like(hidden_states, dtype=torch.float32)
        for start_row, taken, counts, start_expert, end_expert in plan:
            stop = start_row + taken
            experts = slice(start_expert, end_expert)
            offs = _chunk_offsets(group_ends, experts, start_row, stop)
            chunk_ids = None if row_ids is None else row_ids[start_row:stop]
            index = token_idx[start_row:stop]
            rows = hidden_states.index_select(0, index)
            projected = _grouped_linear(rows, up_weight[experts], counts, offs)
            up_delta = _adapter_delta(
                rows,
                _slot_factors(up_factors, up_slots, experts, chunk_ids, rows.dtype),
                counts,
                offs,
            )
            if up_delta is not None:
                projected.add_(up_delta)
            intermediate = _expert_activation(projected, act_fn, gated)
            out = _grouped_linear(intermediate, down_weight[experts], counts, offs)
            down_delta = _adapter_delta(
                intermediate,
                _slot_factors(down_factors, down_slots, experts, chunk_ids, rows.dtype),
                counts,
                offs,
            )
            if down_delta is not None:
                out.add_(down_delta)
            out.mul_(routed_weights[start_row:stop])
            result.index_add_(0, index, out.to(torch.float32))
        ctx.save_for_backward(
            hidden_states,
            token_idx,
            routed_weights,
            group_ends,
            row_ids,
            up_weight,
            down_weight,
            *factors,
        )
        ctx.act_fn = act_fn
        ctx.gated = gated
        ctx.plan = plan
        ctx.up_slots = up_slots
        ctx.down_slots = down_slots
        return result.to(hidden_states.dtype)

    @staticmethod
    @once_differentiable
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor | None, ...]:
        (
            hidden_states,
            token_idx,
            routed_weights,
            group_ends,
            row_ids,
            up_weight,
            down_weight,
            *factors,
        ) = ctx.saved_tensors
        # Forward inputs ahead of ``*factors``.
        n_fixed = 12
        needs_rows = ctx.needs_input_grad[0]
        needs_weights = ctx.needs_input_grad[2]
        needs_factors = ctx.needs_input_grad[n_fixed:]
        n_up = 2 * len(ctx.up_slots)
        grad_hidden = (
            torch.zeros_like(hidden_states, dtype=torch.float32) if needs_rows else None
        )
        grad_weights = torch.zeros_like(routed_weights) if needs_weights else None
        grad_factors = [
            torch.zeros_like(factor, dtype=torch.float32) if needs else None
            for factor, needs in zip(factors, needs_factors, strict=True)
        ]
        for start_row, taken, counts, start_expert, end_expert in ctx.plan:
            stop = start_row + taken
            experts = slice(start_expert, end_expert)
            offs = _chunk_offsets(group_ends, experts, start_row, stop)
            chunk_ids = None if row_ids is None else row_ids[start_row:stop]
            index = token_idx[start_row:stop]
            weights = routed_weights[start_row:stop]
            rows = hidden_states.index_select(0, index)
            grad_rows = grad_output.index_select(0, index).to(rows.dtype)
            # Leaves hold only this chunk's experts, so the local graph and its
            # factor grads stay chunk-sized.
            leaves = [
                factor[experts].detach().requires_grad_(needs)
                for factor, needs in zip(factors, needs_factors, strict=True)
            ]
            whole = slice(None)
            with torch.enable_grad():
                rows.requires_grad_(needs_rows)
                projected = _grouped_linear(rows, up_weight[experts], counts, offs)
                up_delta = _adapter_delta(
                    rows,
                    _slot_factors(
                        leaves[:n_up], ctx.up_slots, whole, chunk_ids, rows.dtype
                    ),
                    counts,
                    offs,
                )
                if up_delta is not None:
                    projected = projected + up_delta
                intermediate = _expert_activation(projected, ctx.act_fn, ctx.gated)
                down_delta = _adapter_delta(
                    intermediate,
                    _slot_factors(
                        leaves[n_up:], ctx.down_slots, whole, chunk_ids, rows.dtype
                    ),
                    counts,
                    offs,
                )
            # The output is linear in ``intermediate`` through the frozen base,
            # so its transpose gives both the intermediate grad and the
            # router-weight grad without recomputing the down projection.
            base_grad = _grouped_matmul(grad_rows, down_weight[experts], counts, offs)
            if grad_weights is not None:
                score = (intermediate.detach().float() * base_grad.float()).sum(
                    -1, keepdim=True
                )
                if down_delta is not None:
                    score += (down_delta.detach().float() * grad_rows.float()).sum(
                        -1, keepdim=True
                    )
                grad_weights[start_row:stop] = score.to(grad_weights.dtype)
            scale = weights.to(grad_rows.dtype)
            outputs: list[torch.Tensor] = []
            output_grads: list[torch.Tensor] = []
            if intermediate.requires_grad:
                outputs.append(intermediate)
                output_grads.append(base_grad.mul_(scale))
            if down_delta is not None and down_delta.requires_grad:
                outputs.append(down_delta)
                output_grads.append(grad_rows * scale)
            targets = [
                (leaf, position)
                for position, leaf in enumerate([rows, *leaves])
                if leaf.requires_grad
            ]
            if not outputs or not targets:
                continue
            grads = torch.autograd.grad(
                outputs,
                [leaf for leaf, _ in targets],
                output_grads,
                allow_unused=True,
            )
            for (_, position), grad in zip(targets, grads, strict=True):
                if grad is None:
                    continue
                if position == 0:
                    assert grad_hidden is not None
                    grad_hidden.index_add_(0, index, grad.float())
                    continue
                target = grad_factors[position - 1]
                assert target is not None
                target[experts] += grad.float()
        return (
            None if grad_hidden is None else grad_hidden.to(hidden_states.dtype),
            None,
            grad_weights,
            *([None] * (n_fixed - 3)),
            *(
                None if grad is None else grad.to(factor.dtype)
                for grad, factor in zip(grad_factors, factors, strict=True)
            ),
        )


def _recompute_routed_experts(
    hidden_states: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    act_fn: Callable[[torch.Tensor], torch.Tensor],
    gated: bool,
    counts: torch.Tensor,
    token_idx: torch.Tensor,
    routed_weights: torch.Tensor,
    up_loras: Sequence[ExpertLora],
    down_loras: Sequence[ExpertLora],
    row_ids: torch.Tensor | None,
) -> torch.Tensor:
    """Run :class:`LoraExpertsFunction` over the routed rows with stacked LoRA factors."""
    if isinstance(up_weight, DTensor):
        up_weight = up_weight.to_local()
    if isinstance(down_weight, DTensor):
        down_weight = down_weight.to_local()
    row_bytes = (
        max(up_weight.shape[1], down_weight.shape[1]) * hidden_states.element_size()
    )
    max_rows = max(1, ROUTED_EXPERT_CHUNK_BYTES // row_bytes)
    plan = list(_iter_expert_row_chunks(counts, max_rows))
    factors = []
    for lora in (*up_loras, *down_loras):
        assert lora.stacked is not None
        factors.extend(lora.stacked)
    return LoraExpertsFunction.apply(
        hidden_states,
        token_idx,
        routed_weights,
        torch.cumsum(counts, dim=0),
        row_ids,
        up_weight,
        down_weight,
        act_fn,
        gated,
        plan,
        tuple(
            AdapterSlot(lora.wrapper.scaling[lora.adapter], lora.row_id)
            for lora in up_loras
        ),
        tuple(
            AdapterSlot(lora.wrapper.scaling[lora.adapter], lora.row_id)
            for lora in down_loras
        ),
        *factors,
    )


def _routed_experts_local_forward(
    experts: nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    chain: dict[str, ParamWrapper] | None = None,
    adapters: dict[str, list[str]] | None = None,
    routing: Sequence[str] | None = None,
    *,
    already_grouped: bool | None = None,
    recompute: bool = False,
) -> torch.Tensor:
    """Packed routed-experts forward with optional split-LoRA deltas.

    :param already_grouped: ``True`` when each row has one expert and rows are
        in expert order, which skips the sort and gather; ``None`` detects it.
    :param recompute: Run :class:`LoraExpertsFunction`, which saves only the
        token rows, routing and LoRA factors and recomputes the up-projection in
        backward. Applies when both base weights are frozen and the LoRA factors
        are not partitioned DTensors.
    """
    projections = _routed_projection_names(experts)
    if projections is None:
        msg = "Routed experts module does not match a supported packed layout."
        raise RuntimeError(msg)
    up_name, gated = projections
    up_weight = getattr(experts, up_name)
    down_weight = experts.down_proj
    act_fn = _routed_experts_act_fn(experts)
    assert isinstance(up_weight, torch.Tensor)
    assert isinstance(down_weight, torch.Tensor)

    num_experts = up_weight.shape[0]
    if chain is not None and adapters is None:
        adapters = {name: resolve_adapters(w) for name, w in chain.items()}
    adapters = adapters or {}
    chain = chain or {}

    top_k = top_k_index.shape[-1]
    if already_grouped and top_k != 1:
        msg = f"already_grouped needs one expert per row, got top_k={top_k}."
        raise ValueError(msg)
    flat_experts = top_k_index.reshape(-1)
    order = (
        torch.arange(flat_experts.shape[0], device=flat_experts.device)
        if already_grouped
        else torch.argsort(flat_experts, stable=True)
    )
    if already_grouped is None:
        # Rows already sit in expert order. Indexing them would clone the activation.
        already_grouped = top_k == 1 and torch.equal(
            order, torch.arange(order.shape[0], device=order.device)
        )
    counts = torch.bincount(flat_experts, minlength=num_experts)
    token_idx = torch.div(order, top_k, rounding_mode="floor")
    routed_weights = top_k_weights.reshape(-1)[order].unsqueeze(-1)
    row_ids: torch.Tensor | None = None
    id_map: dict[str, int] | None = None
    if routing is not None and len(set(routing)) > 1:
        row_ids, id_map = token_adapter_ids(routing, hidden_states.shape[0], token_idx)

    def expert_loras(param_name: str) -> list[ExpertLora]:
        return [
            ExpertLora(
                chain[param_name],
                name,
                _stacked_lora_weights(
                    chain[param_name], name, hidden_states.dtype, num_experts
                ),
                None if id_map is None else id_map[name],
            )
            for name in adapters.get(param_name, [])
        ]

    up_loras = expert_loras(up_name)
    down_loras = expert_loras("down_proj")
    if (
        recompute
        and not up_weight.requires_grad
        and not down_weight.requires_grad
        and all(lora.stacked is not None for lora in (*up_loras, *down_loras))
    ):
        return _recompute_routed_experts(
            hidden_states,
            up_weight,
            down_weight,
            act_fn,
            gated,
            counts,
            token_idx,
            routed_weights,
            up_loras,
            down_loras,
            row_ids,
        )
    x = hidden_states if already_grouped else hidden_states[token_idx]
    return _scatter_routed_expert_chunks(
        hidden_states,
        x,
        up_weight,
        down_weight,
        act_fn,
        gated,
        counts,
        token_idx,
        routed_weights,
        up_loras,
        down_loras,
        row_ids,
    )


def wrapper_chain(wrapper: ParamWrapper) -> dict[str, ParamWrapper]:
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
    token_index: torch.Tensor | None = None
    n_tokens: int | None = None

    def forward(
        self,
        x: torch.Tensor,
        expert_size: Sequence[int] | torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        routing = ROUTING_STATE.get(self)
        mixed = routing is not None and len(set(routing)) > 1
        row_ids: torch.Tensor | None = None
        id_map: dict[str, int] | None = None
        if mixed:
            assert routing is not None
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
        counts = _expert_counts(expert_size, self.num_experts)
        offs = _group_offsets(counts, x.device) if x.is_cuda else None
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
    # See ``recompute`` on :func:`_routed_experts_local_forward`.
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
        routing = ROUTING_STATE.get(self)
        mixed = routing is not None and len(set(routing)) > 1
        if mixed:
            assert routing is not None
            adapters = {
                name: adapters_in_routing(wrapper, routing)
                for name, wrapper in chain.items()
            }
        else:
            adapters = {name: resolve_adapters(w) for name, w in chain.items()}

        if not any(adapters.values()):
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
            routing=routing if mixed else None,
            already_grouped=already_grouped,
            recompute=self.recompute,
        )


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
        routing = ROUTING_STATE.get(self)
        if routing is not None and len(set(routing)) > 1:
            adapters = {
                name: adapters_in_routing(wrapper, routing)
                for name, wrapper in chain.items()
            }
            mixed_routing: Sequence[str] | None = routing
        else:
            adapters = {
                name: resolve_adapters(wrapper) for name, wrapper in chain.items()
            }
            mixed_routing = None
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
            routing=mixed_routing,
        )


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


def _checkpoints_activations(module: nn.Module) -> bool:
    """Whether ``module`` reruns its forward during backward."""
    return isinstance(module, CheckpointWrapper) or (
        isinstance(module, GradientCheckpointingLayer) and module.gradient_checkpointing
    )


def set_routed_experts_recompute(model: nn.Module, enabled: bool | None) -> None:
    """Choose recompute-in-backward or the full autograd graph for every routed-experts LoRA wrapper.

    :param model: Model holding ``RoutedExpertsLoraWrapper`` layers.
    :type model: nn.Module
    :param enabled: ``True`` runs :class:`LoraExpertsFunction` on frozen base
        weights and ``False`` keeps the full graph. ``None`` keeps the full
        graph inside activation-checkpointed blocks, which already rerun the
        expert forward in backward, and recomputes everywhere else.
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


def bind_routed_experts_config(model: nn.Module) -> None:
    """Point packed experts that have no ``act_fn`` at the model config.

    Liger-patched packed experts drop ``act_fn``; the fused path then reads
    ``config.hidden_act`` off the experts module.

    :param model: PEFT model with a ``config`` carrying ``hidden_act``.
    :type model: nn.Module
    """
    for module in model.modules():
        if _is_routed_experts_module(module) and not callable(
            getattr(module, "act_fn", None)
        ):
            module.config = model.config


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
        if _is_sorted_experts_module(module):
            suffixes.add(f"{prefix}.weight")
        elif (projections := _routed_projection_names(module)) is not None:
            suffixes.add(f"{prefix}.{projections[0]}")
            suffixes.add(f"{prefix}.down_proj")
        elif is_transposed_experts_module(module):
            suffixes.add(f"{prefix}.gate_up_proj")
            suffixes.add(f"{prefix}.down_proj")
    return sorted(suffixes)


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
        if _is_routed_experts_module(target):
            fn = _routed_experts_local_forward
        elif _is_sorted_experts_module(target):
            fn = _grouped_linear
        else:
            continue
        if getattr(target.forward, "__func__", None) is fn:
            continue
        target.forward = MethodType(fn, target)
        patched += 1
    return patched

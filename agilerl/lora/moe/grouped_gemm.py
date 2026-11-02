# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Per-expert GEMMs over expert-sorted rows, and their row chunking."""

from __future__ import annotations

import importlib.util
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from functools import cache
from typing import Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import once_differentiable
from torch.distributed.tensor import DTensor
from torch.utils._python_dispatch import _disable_current_modes

from agilerl.arena.memory.resolve.moe import (
    ExpertGemmBackend,
    expert_gemm_backend,
    padded_expert_lora_rank,
)

HAS_TRITON = importlib.util.find_spec("triton") is not None

if HAS_TRITON:
    from agilerl.lora.moe import triton_kernels


@cache
def device_expert_backend(device_index: int, dtype: torch.dtype) -> ExpertGemmBackend:
    """:func:`expert_gemm_backend` of a CUDA device; ``grouped_mm`` must pass :func:`probe_grouped_mm`."""
    backend = expert_gemm_backend(
        torch.cuda.get_device_capability(device_index),
        str(dtype).removeprefix("torch."),
        HAS_TRITON,
    )
    if backend == "grouped_mm":
        probe_grouped_mm(device_index, dtype)
    return backend


def probe_grouped_mm(device_index: int, dtype: torch.dtype) -> None:
    """Raise unless ``F.grouped_mm`` computes correct results here (fwd and bwd, transposed views).

    :param device_index: CUDA device index.
    :param dtype: Activation dtype.
    :raises RuntimeError: If ``F.grouped_mm`` fails or differs from per-expert matmuls.
    """
    major, minor = torch.cuda.get_device_capability(device_index)
    failure = f"torch.nn.functional.grouped_mm failed on SM {major}.{minor}"
    # The first call can run inside a checkpointed or no_grad forward. The
    # probe needs its own backward: grad on, and identity hooks keep its saved
    # tensors out of the checkpoint. Its ops run once, so no dispatch mode may
    # record them: selective checkpointing indexes every op call on recompute.
    with (
        torch.enable_grad(),
        torch.autograd.graph.saved_tensors_hooks(lambda t: t, lambda t: t),
        _disable_current_modes(),
    ):
        device = torch.device("cuda", device_index)
        generator = torch.Generator(device=device).manual_seed(0)
        # Both inner dims stay 16-byte aligned in bf16, including after transpose.
        x = torch.randn(8, 16, device=device, dtype=dtype, generator=generator)
        w = torch.randn(2, 16, 16, device=device, dtype=dtype, generator=generator)
        x = x.requires_grad_(True)
        w = w.requires_grad_(True)
        offs = torch.tensor([5, 8], device=device, dtype=torch.int32)
        reference = torch.cat([x[:5] @ w[0].mT, x[5:] @ w[1].mT])
        try:
            out = F.grouped_mm(x, w.transpose(-2, -1), offs=offs)
        except Exception as error:
            msg = f"{failure}: {error}"
            raise RuntimeError(msg) from error
        if not torch.allclose(out.float(), reference.float(), atol=1e-2):
            msg = f"{failure}: output differs from per-expert matmuls"
            raise RuntimeError(msg)
        try:
            # square() materializes the incoming gradient; the op's backward
            # rejects the zero-stride expanded grad a bare sum() would feed it.
            out.square().sum().backward()
        except Exception as error:
            msg = f"{failure} in backward: {error}"
            raise RuntimeError(msg) from error


def expert_backend(x: torch.Tensor) -> ExpertGemmBackend:
    """Kernel the per-expert GEMMs over *x* run on; ``grouped_mm`` off CUDA."""
    if not x.is_cuda:
        return "grouped_mm"
    index = x.device.index
    if index is None:
        index = torch.cuda.current_device()
    return device_expert_backend(index, x.dtype)


def _use_grouped_mm(x: torch.Tensor) -> bool:
    """Whether the grouped-GEMM fast path applies to *x*'s device and dtype."""
    return expert_backend(x) == "grouped_mm"


@dataclass(frozen=True)
class ExpertKernels:
    """Kernels over expert-sorted rows that read the group offsets on device."""

    # ``(x [rows, in], weight [experts, in, out], offs) -> [rows, out]``
    matmul: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
    # ``(x [rows, in], grad [rows, out], offs) -> [experts, in, out]``
    weight_grad: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
    # ``(out fp32 [tokens, h], index, rows [rows, h], weights [rows, 1] | None)``
    # adds each scaled row into ``out[index]`` in place.
    scatter_add: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None], None
    ]


TRITON_EXPERT_KERNELS = (
    ExpertKernels(
        triton_kernels.grouped_matmul,
        triton_kernels.grouped_weight_grad,
        triton_kernels.scatter_add_rows,
    )
    if HAS_TRITON
    else None
)


def expert_kernels(x: torch.Tensor) -> ExpertKernels | None:
    """Triton expert kernels when *x* runs on the ``triton`` backend, else ``None`` to stay on torch ops."""
    if expert_backend(x) != "triton":
        return None
    return TRITON_EXPERT_KERNELS


def unaligned_gemm_kernels(x: torch.Tensor) -> ExpertKernels:
    """Triton kernels for the expert GEMMs over *x* that ``grouped_mm`` cannot run.

    :raises RuntimeError: Off CUDA or without Triton, where nothing else runs them.
    """
    if x.is_cuda and TRITON_EXPERT_KERNELS is not None:
        return TRITON_EXPERT_KERNELS
    msg = (
        f"MoE expert GEMMs in {str(x.dtype).removeprefix('torch.')} on "
        f"{x.device.type} have no kernel: torch.nn.functional.grouped_mm needs "
        "inner dims on a 16-byte boundary, and the Triton kernels need CUDA "
        "with Triton installed"
    )
    raise RuntimeError(msg)


class GroupedMatmulCtx(Protocol):
    saved_tensors: tuple[torch.Tensor | None, ...]
    needs_input_grad: tuple[bool, ...]
    kernels: ExpertKernels

    def save_for_backward(self, *tensors: torch.Tensor | None) -> None: ...


class GroupedMatmul(torch.autograd.Function):
    """Per-expert ``rows @ weight[e]`` over expert-sorted rows on :class:`ExpertKernels`."""

    @staticmethod
    def forward(
        ctx: GroupedMatmulCtx,
        x: torch.Tensor,
        weight: torch.Tensor,
        offs: torch.Tensor,
        kernels: ExpertKernels,
    ) -> torch.Tensor:
        needs_x, needs_weight = ctx.needs_input_grad[:2]
        ctx.save_for_backward(
            x if needs_weight else None, weight if needs_x else None, offs
        )
        ctx.kernels = kernels
        return kernels.matmul(x, weight, offs)

    @staticmethod
    @once_differentiable
    def backward(
        ctx: GroupedMatmulCtx, grad: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, None, None]:
        x, weight, offs = ctx.saved_tensors
        assert offs is not None
        grad_x = (
            None
            if weight is None
            else ctx.kernels.matmul(grad, weight.transpose(-2, -1), offs)
        )
        grad_weight = None if x is None else ctx.kernels.weight_grad(x, grad, offs)
        return grad_x, grad_weight, None, None


def counts_list(counts: Sequence[int] | torch.Tensor) -> list[int]:
    """Per-expert row counts as a plain list (host sync only when needed)."""
    if isinstance(counts, torch.Tensor):
        return [int(count) for count in counts.tolist()]
    return [int(count) for count in counts]


def counts_tensor(
    counts: Sequence[int] | torch.Tensor, device: torch.device
) -> torch.Tensor:
    """Per-expert row counts as a device tensor."""
    if isinstance(counts, torch.Tensor):
        return counts
    return torch.as_tensor(counts, device=device)


def group_offsets(
    counts: Sequence[int] | torch.Tensor, device: torch.device
) -> torch.Tensor:
    """Cumulative per-expert row offsets in the layout ``torch.nn.functional.grouped_mm`` takes."""
    return torch.cumsum(counts_tensor(counts, device), dim=0).to(torch.int32)


def expert_row_counts(expert_ids: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Rows per expert; ``bincount`` on CUDA reads the max id back to the host, this does not."""
    counts = torch.zeros(num_experts, dtype=torch.long, device=expert_ids.device)
    ones = torch.ones_like(expert_ids, dtype=torch.long)
    return counts.index_add_(0, expert_ids, ones)


def row_chunk_offsets(
    counts: torch.Tensor, total_rows: int, max_rows: int
) -> tuple[list[int], torch.Tensor]:
    """Fixed row chunks of expert-sorted rows and every chunk's grouped-GEMM offsets.

    Chunks may cut through an expert. Row ``i`` of the offsets covers every
    expert (empty groups included) for chunk ``i``. Only ``total_rows`` comes
    from the host, so planning issues no host sync.

    :param counts: Rows per expert, on device.
    :param total_rows: Sum of ``counts``.
    :param max_rows: Rows per chunk; the last chunk takes the remainder.
    :return: Chunk sizes and ``[chunks, experts]`` int32 offsets.
    """
    sizes = [
        min(max_rows, total_rows - start) for start in range(0, total_rows, max_rows)
    ]
    starts = torch.arange(0, total_rows, max_rows, device=counts.device)
    ends = (torch.cumsum(counts, dim=0) - starts.unsqueeze(1)).clamp_(min=0)
    taken = (total_rows - starts).clamp_(max=max_rows).unsqueeze(1)
    return sizes, torch.minimum(ends, taken).to(torch.int32)


def _dims_aligned(itemsize: int, *dims: int) -> bool:
    """Whether row strides over these inner dims meet the op's 16-byte alignment."""
    return all(dim * itemsize % 16 == 0 for dim in dims)


def _grouped_mm_operand_ready(mat: torch.Tensor) -> bool:
    """Whether ``mat`` has a last-two-dims layout ``grouped_mm`` accepts."""
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


def grouped_linear(
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
    if offs is None:
        offs = group_offsets(counts, x.device)
    return grouped_matmul(x, weight.transpose(-2, -1), offs)


def grouped_operand(weight: torch.Tensor) -> torch.Tensor:
    """``[experts, in, out]`` operand of a stacked ``[experts, out, in]`` weight, in a layout :func:`grouped_matmul` runs on the grouped GEMM."""
    if isinstance(weight, DTensor):
        weight = weight.to_local()
    operand = weight.transpose(-2, -1)
    return operand if _grouped_mm_operand_ready(operand) else operand.contiguous()


def pad_lora_rank(
    lora_a: torch.Tensor, lora_b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero-pad stacked ``[E, r, in]`` / ``[E, out, r]`` LoRA factors to a rank the grouped GEMM accepts.

    The added rows of ``A`` and columns of ``B`` are zero, so every product is
    exact, and gradients reach the unpadded factors through the pad.

    :param lora_a: Stacked ``A`` factors.
    :param lora_b: Stacked ``B`` factors.
    :return: Padded factors when the grouped GEMM serves ``lora_a``, else the inputs.
    """
    rank = lora_a.shape[1]
    padded = padded_expert_lora_rank(rank, lora_a.element_size())
    if padded == rank or not _use_grouped_mm(lora_a):
        return lora_a, lora_b
    return (
        nn.functional.pad(lora_a, (0, 0, 0, padded - rank)),
        nn.functional.pad(lora_b, (0, padded - rank)),
    )


def grouped_matmul(
    x: torch.Tensor,
    weight: torch.Tensor,
    offs: torch.Tensor,
) -> torch.Tensor:
    """Per-expert ``rows @ weight[e]`` over expert-sorted rows with a stacked ``[experts, in, out]`` weight.

    :param offs: Cumulative row offsets, one per expert; the last is ``x``'s row count.
    :raises RuntimeError: When neither ``grouped_mm`` nor Triton runs the GEMM.
    """
    if x.shape[0] == 0:
        return x.new_empty((0, weight.shape[2]))
    # The grouped kernels take one dtype; weights follow the activations.
    weight = weight.to(x.dtype)
    if _use_grouped_mm(x) and _dims_aligned(
        x.element_size(), weight.shape[1], weight.shape[2]
    ):
        # A plain transpose is a layout the op accepts. Other strides are copied.
        if not _grouped_mm_operand_ready(weight):
            weight = weight.contiguous()
        return F.grouped_mm(x, weight, offs=offs)
    return GroupedMatmul.apply(x, weight, offs, unaligned_gemm_kernels(x))


def iter_expert_row_chunks(
    counts: Sequence[int] | torch.Tensor,
    max_rows: int,
) -> Iterator[tuple[int, int, list[int], int, int]]:
    """Yield ``(start_row, taken, local_counts, start_expert, end_expert)``."""
    remaining = counts_list(counts)
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


def add_grouped_linear(
    destination: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    counts: Sequence[int] | torch.Tensor,
    scaling: float | torch.Tensor,
    chunk_bytes: int,
) -> None:
    """Add a grouped GEMM into ``destination`` in row chunks.

    :param chunk_bytes: Largest output of one chunk.
    """
    if x.shape[0] == 0:
        return
    row_bytes = weight.shape[1] * x.element_size()
    max_rows = max(1, chunk_bytes // max(row_bytes, 1))
    for (
        start_row,
        taken,
        local_counts,
        start_expert,
        end_expert,
    ) in iter_expert_row_chunks(counts, max_rows):
        chunk = grouped_linear(
            x[start_row : start_row + taken],
            weight[start_expert:end_expert],
            local_counts,
        )
        if chunk.dtype != destination.dtype:
            chunk = chunk.to(dtype=destination.dtype)
        chunk.mul_(scaling)
        destination[start_row : start_row + taken].add_(chunk)

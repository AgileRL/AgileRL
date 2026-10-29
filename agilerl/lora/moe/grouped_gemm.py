# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Per-expert GEMMs over expert-sorted rows, and their row chunking."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from functools import cache

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor

# No public API pops the active torch dispatch modes.
from torch.utils._python_dispatch import _disable_current_modes

from agilerl.arena.memory.resolve.moe import (
    grouped_gemm_supported,
    padded_expert_lora_rank,
)


@cache
def grouped_mm_supported(device_index: int, dtype: torch.dtype) -> bool:
    """Whether expert GEMMs on this device and dtype run on ``torch._grouped_mm``.

    Where :func:`grouped_gemm_supported` requires the grouped GEMM, the op must
    compute correct results (fwd and bwd, transposed views) or this raises.

    :param device_index: CUDA device index.
    :param dtype: Activation dtype.
    :return: ``True`` where the grouped GEMM is required, ``False`` for the per-expert loop.
    :raises RuntimeError: If the grouped GEMM is required and ``torch._grouped_mm`` fails.
    """
    major, minor = torch.cuda.get_device_capability(device_index)
    if not grouped_gemm_supported((major, minor), str(dtype).removeprefix("torch.")):
        return False
    failure = f"torch._grouped_mm failed on SM {major}.{minor}"
    if not hasattr(torch, "_grouped_mm"):
        msg = f"{failure}: torch {torch.__version__} has no torch._grouped_mm"
        raise RuntimeError(msg)
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
            out = torch._grouped_mm(x, w.transpose(-2, -1), offs=offs)
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
    return True


def _use_grouped_mm(x: torch.Tensor) -> bool:
    """Whether the grouped-GEMM fast path applies to *x*'s device and dtype."""
    if not x.is_cuda:
        return False
    index = x.device.index
    if index is None:
        index = torch.cuda.current_device()
    return grouped_mm_supported(index, x.dtype)


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
    """Cumulative per-expert row offsets in the layout ``torch._grouped_mm`` takes."""
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


def offset_counts(offs: torch.Tensor) -> list[int]:
    """Per-group row counts of grouped-GEMM offsets (host sync)."""
    return torch.diff(offs, prepend=offs.new_zeros(1)).tolist()


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
            offs = group_offsets(counts, x.device)
        return torch._grouped_mm(x, operand, offs=offs)
    # FSDP's gathered expert block is a narrow view. cuBLAS rejects that
    # stride, and a zero-row expert is an empty gemm it also rejects.
    weight = weight.contiguous()
    pieces = []
    for expert, rows in enumerate(x.split(counts_list(counts))):
        if rows.shape[0] == 0:
            continue
        pieces.append(nn.functional.linear(rows, weight[expert]))
    if not pieces:
        return x.new_empty((0, weight.shape[1]))
    return torch.cat(pieces)


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
    """
    if (
        x.dtype == weight.dtype
        and _dims_aligned(x.element_size(), weight.shape[1], weight.shape[2])
        and _use_grouped_mm(x)
        and _grouped_mm_operand_ready(weight)
    ):
        return torch._grouped_mm(x, weight, offs=offs)
    pieces = [
        rows @ weight[expert]
        for expert, rows in enumerate(x.split(offset_counts(offs)))
        if rows.shape[0] > 0
    ]
    if not pieces:
        return x.new_empty((0, weight.shape[2]))
    return torch.cat(pieces)


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

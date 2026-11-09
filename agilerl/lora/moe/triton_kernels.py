# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Triton grouped GEMMs and weighted row scatter over expert-sorted rows.

Group offsets stay on device: each program finds its expert from the
offsets, so a call is one launch with no host sync and no per-expert loop.
"""

from collections.abc import Callable

import torch
import triton
import triton.language as tl

# Configs a shape may use, best first. Each autotune key (weight shape and
# dtypes, never the row count) keeps at most AUTOTUNE_CANDIDATES of them.
MATMUL_CONFIGS = [
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=3
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=4
    ),
    # Rank-deep LoRA ``B``.
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=4
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=4
    ),
    # Rank-wide LoRA ``A``.
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 128}, num_warps=4, num_stages=3
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 64}, num_warps=4, num_stages=4
    ),
]

WEIGHT_GRAD_CONFIGS = [
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3
    ),
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=3
    ),
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 128}, num_warps=4, num_stages=3
    ),
    # Rank-deep LoRA ``B`` gradient.
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=4
    ),
    # Rank-wide LoRA ``A`` gradient.
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 64}, num_warps=4, num_stages=4
    ),
]

AUTOTUNE_CANDIDATES = 2
SCATTER_BLOCK = 512


def fitting_configs(
    configs: list[triton.Config],
    named_args: dict[str, int | torch.Tensor],
    **_kwargs: object,
) -> list[triton.Config]:
    """The first :data:`AUTOTUNE_CANDIDATES` configs whose N and K tiles do not overshoot the shape."""
    n_tile = max(32, triton.next_power_of_2(int(named_args["N"])))
    k_tile = max(32, triton.next_power_of_2(int(named_args["K"])))
    fitting = [
        config
        for config in configs
        if config.kwargs["BLOCK_N"] <= n_tile and config.kwargs["BLOCK_K"] <= k_tile
    ]
    return (fitting or configs)[:AUTOTUNE_CANDIDATES]


def quick_bench(
    kernel_call: Callable[[], object], quantiles: tuple[float, ...]
) -> list[float]:
    """Autotune timing over ~25 ms per config."""
    return triton.testing.do_bench(kernel_call, warmup=5, rep=20, quantiles=quantiles)


@triton.autotune(
    configs=MATMUL_CONFIGS,
    key=["N", "K"],
    prune_configs_by={"early_config_prune": fitting_configs},
    do_bench=quick_bench,
)
@triton.jit
def grouped_matmul_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    offs_ptr,
    N,
    K,
    E,
    stride_xm,
    stride_we,
    stride_wk,
    stride_wn,
    E_POW2: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """``y[rows_e] = x[rows_e] @ w[e]``; program ``(m, n)`` owns one row tile of one expert.

    ``x`` has unit column stride and ``y`` is contiguous ``[rows, N]``.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    experts = tl.arange(0, E_POW2)
    ends = tl.load(offs_ptr + experts, mask=experts < E, other=0)
    starts = tl.load(
        offs_ptr + experts - 1, mask=(experts > 0) & (experts < E), other=0
    )
    tiles = (ends - starts + BLOCK_M - 1) // BLOCK_M
    tile_ends = tl.cumsum(tiles, axis=0)
    # Experts whose tiles all lie before this program; the next one owns it.
    expert = tl.sum((tile_ends <= pid_m).to(tl.int32), axis=0)
    # The grid holds one spare tile per expert for partial last tiles.
    if expert >= E:
        return
    owner = experts == expert
    tile = pid_m - tl.sum(tl.where(owner, tile_ends - tiles, 0), axis=0)
    row_start = tl.sum(tl.where(owner, starts, 0), axis=0) + tile * BLOCK_M
    row_end = tl.sum(tl.where(owner, ends, 0), axis=0)

    rows = row_start + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    inner = tl.arange(0, BLOCK_K)
    row_mask = rows < row_end
    col_mask = cols < N
    x_ptrs = x_ptr + rows[:, None].to(tl.int64) * stride_xm + inner[None, :]
    w_ptrs = (
        w_ptr
        + expert.to(tl.int64) * stride_we
        + inner[:, None] * stride_wk
        + cols[None, :] * stride_wn
    )
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(tl.cdiv(K, BLOCK_K)):
        k_mask = inner < K - k * BLOCK_K
        a = tl.load(x_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0)
        b = tl.load(w_ptrs, mask=k_mask[:, None] & col_mask[None, :], other=0.0)
        acc += tl.dot(a, b)
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K * stride_wk
    y_ptrs = y_ptr + rows[:, None].to(tl.int64) * N + cols[None, :]
    tl.store(
        y_ptrs,
        acc.to(y_ptr.dtype.element_ty),
        mask=row_mask[:, None] & col_mask[None, :],
    )


@triton.autotune(
    configs=WEIGHT_GRAD_CONFIGS,
    key=["N", "K"],
    prune_configs_by={"early_config_prune": fitting_configs},
    do_bench=quick_bench,
)
@triton.jit
def grouped_weight_grad_kernel(
    x_ptr,
    g_ptr,
    dw_ptr,
    offs_ptr,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_gm,
    stride_gn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """``dw[e] = x[rows_e]^T @ g[rows_e]``; program ``(e, tile)`` owns one ``[K, N]`` tile of one expert.

    ``dw`` is contiguous ``[experts, K, N]``.
    """
    expert = tl.program_id(0)
    pid = tl.program_id(1)
    tiles_n = tl.cdiv(N, BLOCK_N)
    pid_k = pid // tiles_n
    pid_n = pid % tiles_n
    row_end = tl.load(offs_ptr + expert)
    row_start = tl.where(expert > 0, tl.load(offs_ptr + tl.maximum(expert - 1, 0)), 0)

    ks = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    ns = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_mask = ks < K
    n_mask = ns < N
    acc = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)
    for start in range(row_start, row_end, BLOCK_M):
        rows = start + tl.arange(0, BLOCK_M)
        row_mask = rows < row_end
        x = tl.load(
            x_ptr + rows[:, None].to(tl.int64) * stride_xm + ks[None, :] * stride_xk,
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        g = tl.load(
            g_ptr + rows[:, None].to(tl.int64) * stride_gm + ns[None, :] * stride_gn,
            mask=row_mask[:, None] & n_mask[None, :],
            other=0.0,
        )
        acc += tl.dot(tl.trans(x), g)
    dw_ptrs = dw_ptr + expert.to(tl.int64) * K * N + ks[:, None] * N + ns[None, :]
    # Experts without rows store zeros.
    tl.store(
        dw_ptrs,
        acc.to(dw_ptr.dtype.element_ty),
        mask=k_mask[:, None] & n_mask[None, :],
    )


@triton.heuristics(
    {
        "HAS_WEIGHTS": lambda args: args["weights_ptr"] is not None,
        "BLOCK_H": lambda _args: SCATTER_BLOCK,
    }
)
@triton.jit
def scatter_add_rows_kernel(
    out_ptr,
    index_ptr,
    rows_ptr,
    weights_ptr,
    H,
    stride_om,
    stride_rm,
    stride_rh,
    stride_w,
    HAS_WEIGHTS: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """``out[index[r]] += weights[r] * rows[r]`` in fp32 atomics."""
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = cols < H
    token = tl.load(index_ptr + row).to(tl.int64)
    values = tl.load(
        rows_ptr + row.to(tl.int64) * stride_rm + cols * stride_rh,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    if HAS_WEIGHTS:
        values *= tl.load(weights_ptr + row.to(tl.int64) * stride_w).to(tl.float32)
    tl.atomic_add(out_ptr + token * stride_om + cols, values, mask=mask, sem="relaxed")


def grouped_matmul(
    x: torch.Tensor, weight: torch.Tensor, offs: torch.Tensor
) -> torch.Tensor:
    """Per-expert ``rows @ weight[e]`` over expert-sorted rows with a stacked ``[experts, in, out]`` weight of any strides.

    :param offs: Cumulative int32 row offsets, one per expert, on device.
    """
    x = x.contiguous()
    rows, inner = x.shape
    experts, _, cols = weight.shape
    out = x.new_empty((rows, cols))

    def grid(meta: dict[str, int]) -> tuple[int, int]:
        return (
            triton.cdiv(rows, meta["BLOCK_M"]) + experts,
            triton.cdiv(cols, meta["BLOCK_N"]),
        )

    grouped_matmul_kernel[grid](
        x,
        weight,
        out,
        offs,
        cols,
        inner,
        experts,
        x.stride(0),
        *weight.stride(),
        E_POW2=triton.next_power_of_2(experts),
    )
    return out


def grouped_weight_grad(
    x: torch.Tensor, grad: torch.Tensor, offs: torch.Tensor
) -> torch.Tensor:
    """``[experts, in, out]`` gradient of :func:`grouped_matmul`'s weight, in ``x``'s dtype."""
    experts = offs.shape[0]
    inner = x.shape[1]
    cols = grad.shape[1]
    out = x.new_empty((experts, inner, cols))

    def grid(meta: dict[str, int]) -> tuple[int, int]:
        return (
            experts,
            triton.cdiv(inner, meta["BLOCK_K"]) * triton.cdiv(cols, meta["BLOCK_N"]),
        )

    grouped_weight_grad_kernel[grid](
        x, grad, out, offs, cols, inner, *x.stride(), *grad.stride()
    )
    return out


def scatter_add_rows(
    out: torch.Tensor,
    index: torch.Tensor,
    rows: torch.Tensor,
    weights: torch.Tensor | None,
) -> None:
    """Add ``weights[r] * rows[r]`` into row ``index[r]`` of the contiguous fp32 ``out``.

    :param weights: ``[rows, 1]`` per-row scale, or ``None``.
    """
    n_rows, width = rows.shape
    if n_rows == 0:
        return
    scatter_add_rows_kernel[(n_rows, triton.cdiv(width, SCATTER_BLOCK))](
        out,
        index,
        rows,
        weights,
        width,
        out.stride(0),
        rows.stride(0),
        rows.stride(1),
        0 if weights is None else weights.stride(0),
    )

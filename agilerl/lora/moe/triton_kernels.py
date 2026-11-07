# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Triton grouped GEMMs, weighted row scatter and gather, and SwiGLU over expert-sorted rows.

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
# Elements of one SwiGLU program's ``[rows, inter]`` tile.
SWIGLU_TILE = 2048
# ``tl.dot`` needs an inner dim of at least 16.
MIN_DOT_K = 16


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


@triton.jit
def expert_row_tile(offs_ptr, E, E_POW2: tl.constexpr, BLOCK_M: tl.constexpr):
    """``(expert, row_start, row_end)`` of this program's row tile; ``expert >= E`` on a spare tile."""
    pid_m = tl.program_id(0)
    experts = tl.arange(0, E_POW2)
    ends = tl.load(offs_ptr + experts, mask=experts < E, other=0)
    starts = tl.load(
        offs_ptr + experts - 1, mask=(experts > 0) & (experts < E), other=0
    )
    tiles = (ends - starts + BLOCK_M - 1) // BLOCK_M
    tile_ends = tl.cumsum(tiles, axis=0)
    # Experts whose tiles all lie before this program; the next one owns it.
    expert = tl.sum((tile_ends <= pid_m).to(tl.int32), axis=0)
    owner = experts == expert
    tile = pid_m - tl.sum(tl.where(owner, tile_ends - tiles, 0), axis=0)
    row_start = tl.sum(tl.where(owner, starts, 0), axis=0) + tile * BLOCK_M
    row_end = tl.sum(tl.where(owner, ends, 0), axis=0)
    return expert, row_start, row_end


@triton.jit
def lora_tile_dot(lora, expert, rows, row_mask, cols, col_mask, BLOCK_R: tl.constexpr):
    """fp32 ``xa[rows] @ b[expert][:, cols]`` of ``lora = (xa, b, R, *b.stride())`` with contiguous ``xa``."""
    xa_ptr, b_ptr, R, stride_be, stride_bk, stride_bn = lora
    ranks = tl.arange(0, BLOCK_R)
    rank_mask = ranks < R
    xa = tl.load(
        xa_ptr + rows[:, None].to(tl.int64) * R + ranks[None, :],
        mask=row_mask[:, None] & rank_mask[None, :],
        other=0.0,
    )
    lora_b = tl.load(
        b_ptr
        + expert.to(tl.int64) * stride_be
        + ranks[:, None] * stride_bk
        + cols[None, :] * stride_bn,
        mask=rank_mask[:, None] & col_mask[None, :],
        other=0.0,
    )
    return tl.dot(xa, lora_b)


@triton.autotune(
    configs=MATMUL_CONFIGS,
    key=["N", "K", "HAS_LORA"],
    prune_configs_by={"early_config_prune": fitting_configs},
    do_bench=quick_bench,
)
@triton.jit
def grouped_matmul_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    offs_ptr,
    lora,
    N,
    K,
    E,
    w_strides,
    E_POW2: tl.constexpr,
    HAS_LORA: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """``y[rows_e] = x[rows_e] @ w[e]`` plus the :func:`lora_tile_dot` of ``lora`` with ``HAS_LORA``.

    Program ``(m, n)`` owns one row tile of one expert. ``x`` is contiguous
    ``[rows, K]`` and ``y`` is contiguous ``[rows, N]``.
    """
    expert, row_start, row_end = expert_row_tile(offs_ptr, E, E_POW2, BLOCK_M)
    # The grid holds one spare tile per expert for partial last tiles.
    if expert >= E:
        return
    stride_we, stride_wk, stride_wn = w_strides
    rows = row_start + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    inner = tl.arange(0, BLOCK_K)
    row_mask = rows < row_end
    col_mask = cols < N
    x_ptrs = x_ptr + rows[:, None].to(tl.int64) * K + inner[None, :]
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
    if HAS_LORA:
        acc += lora_tile_dot(lora, expert, rows, row_mask, cols, col_mask, BLOCK_R)
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


@triton.heuristics({"BLOCK_H": lambda _args: SCATTER_BLOCK})
@triton.jit
def gather_scaled_rows_kernel(
    grad_ptr,
    index_ptr,
    rows_ptr,
    weights_ptr,
    grad_rows_ptr,
    grad_weights_ptr,
    H,
    stride_gm,
    stride_rm,
    stride_w,
    BLOCK_H: tl.constexpr,
):
    """``grad_rows[r] = weights[r] * grad[index[r]]`` and ``grad_weights[r] = grad[index[r]] . rows[r]`` in fp32.

    ``grad`` and ``rows`` have unit column stride; ``grad_rows`` is contiguous.
    """
    row = tl.program_id(0).to(tl.int64)
    token = tl.load(index_ptr + row).to(tl.int64)
    weight = tl.load(weights_ptr + row * stride_w).to(tl.float32)
    dot = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for start in range(0, H, BLOCK_H):
        cols = start + tl.arange(0, BLOCK_H)
        mask = cols < H
        grad = tl.load(grad_ptr + token * stride_gm + cols, mask=mask, other=0.0).to(
            tl.float32
        )
        values = tl.load(rows_ptr + row * stride_rm + cols, mask=mask, other=0.0).to(
            tl.float32
        )
        dot += grad * values
        tl.store(
            grad_rows_ptr + row * H + cols,
            (grad * weight).to(grad_rows_ptr.dtype.element_ty),
            mask=mask,
        )
    tl.store(grad_weights_ptr + row, tl.sum(dot, axis=0))


# Heuristics run in order, so BLOCK_M can read BLOCK_N.
SWIGLU_BLOCKS = {
    "BLOCK_N": lambda args: min(SWIGLU_TILE, triton.next_power_of_2(args["N"])),
    "BLOCK_M": lambda args: SWIGLU_TILE // args["BLOCK_N"],
}


@triton.heuristics(SWIGLU_BLOCKS)
@triton.jit
def swiglu_kernel(
    projected_ptr,
    out_ptr,
    M,
    N,
    limit,
    CLAMP: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """``out = silu(gate) * up`` in fp32 over contiguous ``projected = [gate | up]`` of ``[M, 2N]``.

    With ``CLAMP``, ``gate`` is first clamped to at most ``limit`` and ``up``
    to ``[-limit, limit]``.
    """
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rows < M)[:, None] & (cols < N)[None, :]
    gate_ptrs = projected_ptr + rows[:, None].to(tl.int64) * 2 * N + cols[None, :]
    gate = tl.load(gate_ptrs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(gate_ptrs + N, mask=mask, other=0.0).to(tl.float32)
    if CLAMP:
        gate = tl.minimum(gate, limit)
        up = tl.minimum(tl.maximum(up, -limit), limit)
    out = gate * tl.sigmoid(gate) * up
    tl.store(
        out_ptr + rows[:, None].to(tl.int64) * N + cols[None, :],
        out.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.heuristics(SWIGLU_BLOCKS)
@triton.jit
def swiglu_backward_kernel(
    grad_ptr,
    projected_ptr,
    grad_projected_ptr,
    M,
    N,
    limit,
    CLAMP: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """``[grad * up * silu'(gate) | grad * silu(gate)]`` in fp32 into contiguous ``grad_projected``.

    With ``CLAMP``, the gradient of :func:`swiglu_kernel`'s clamps: it passes
    where ``gate <= limit`` and ``-limit <= up <= limit``, bounds included,
    as ``torch.clamp`` does, and is zero elsewhere.
    """
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rows < M)[:, None] & (cols < N)[None, :]
    packed = rows[:, None].to(tl.int64) * 2 * N + cols[None, :]
    grad = tl.load(
        grad_ptr + rows[:, None].to(tl.int64) * N + cols[None, :], mask=mask, other=0.0
    ).to(tl.float32)
    gate = tl.load(projected_ptr + packed, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(projected_ptr + packed + N, mask=mask, other=0.0).to(tl.float32)
    if CLAMP:
        gate_passes = gate <= limit
        up_passes = (up >= -limit) & (up <= limit)
        gate = tl.minimum(gate, limit)
        up = tl.minimum(tl.maximum(up, -limit), limit)
    sigmoid = tl.sigmoid(gate)
    grad_gate = grad * up * sigmoid * (1.0 + gate * (1.0 - sigmoid))
    grad_up = grad * gate * sigmoid
    if CLAMP:
        grad_gate = tl.where(gate_passes, grad_gate, 0.0)
        grad_up = tl.where(up_passes, grad_up, 0.0)
    dtype = grad_projected_ptr.dtype.element_ty
    tl.store(grad_projected_ptr + packed, grad_gate.to(dtype), mask=mask)
    tl.store(grad_projected_ptr + packed + N, grad_up.to(dtype), mask=mask)


def swiglu_grid(rows: int, inter: int) -> Callable[[dict[str, int]], tuple[int, int]]:
    """Launch grid of a SwiGLU kernel over ``[rows, inter]`` from its heuristic block sizes."""

    def grid(meta: dict[str, int]) -> tuple[int, int]:
        return triton.cdiv(rows, meta["BLOCK_M"]), triton.cdiv(inter, meta["BLOCK_N"])

    return grid


def swiglu(projected: torch.Tensor, limit: float | None) -> torch.Tensor:
    """``silu(gate) * up`` of a ``[rows, 2 * inter]`` projection ``[gate | up]``, in its dtype.

    :param projected: ``[rows, 2 * inter]`` packed ``[gate | up]``.
    :param limit: ``swiglu_limit``: clamp ``gate`` to at most it and ``up``
        to ``[-limit, limit]`` first, or ``None`` for no clamp.
    """
    projected = projected.contiguous()
    rows, inter = projected.shape[0], projected.shape[1] // 2
    out = projected.new_empty((rows, inter))
    if rows == 0:
        return out
    swiglu_kernel[swiglu_grid(rows, inter)](
        projected, out, rows, inter, 0.0 if limit is None else limit, limit is not None
    )
    return out


def swiglu_backward(
    grad: torch.Tensor, projected: torch.Tensor, limit: float | None
) -> torch.Tensor:
    """Gradient of :func:`swiglu` w.r.t. ``projected``, in ``projected``'s dtype.

    :param grad: ``[rows, inter]`` gradient of the SwiGLU output.
    :param projected: The ``[rows, 2 * inter]`` forward input.
    :param limit: The forward's ``swiglu_limit``, or ``None``.
    """
    grad = grad.contiguous()
    projected = projected.contiguous()
    rows, inter = grad.shape
    grad_projected = torch.empty_like(projected)
    if rows == 0:
        return grad_projected
    swiglu_backward_kernel[swiglu_grid(rows, inter)](
        grad,
        projected,
        grad_projected,
        rows,
        inter,
        0.0 if limit is None else limit,
        limit is not None,
    )
    return grad_projected


def grouped_matmul(
    x: torch.Tensor,
    weight: torch.Tensor,
    offs: torch.Tensor,
    lora: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Per-expert ``rows @ weight[e]`` over expert-sorted rows with a stacked ``[experts, in, out]`` weight of any strides.

    :param offs: Cumulative int32 row offsets, one per expert, on device.
    :param lora: ``(xa [rows, r], b [experts, r, out])`` whose per-expert
        ``xa @ b[e]`` adds into the output in the same fp32 accumulator.
    """
    x = x.contiguous()
    rows, inner = x.shape
    experts, _, cols = weight.shape
    out = x.new_empty((rows, cols))
    if lora is None:
        xa, lora_b = x, weight
        rank = 0
    else:
        xa, lora_b = lora
        xa = xa.contiguous()
        rank = xa.shape[1]

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
        (xa, lora_b, rank, *lora_b.stride()),
        cols,
        inner,
        experts,
        weight.stride(),
        E_POW2=triton.next_power_of_2(experts),
        HAS_LORA=lora is not None,
        BLOCK_R=max(MIN_DOT_K, triton.next_power_of_2(rank)),
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


def gather_scaled_rows(
    grad: torch.Tensor,
    index: torch.Tensor,
    rows: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward of :func:`scatter_add_rows` with weights: the grads of ``rows`` and ``weights``.

    :param grad: fp32 ``[tokens, h]`` grad of the scatter output.
    :param weights: ``[rows, 1]`` per-row scale.
    :return: ``weights * grad[index]`` in ``rows``' dtype and the fp32 ``[rows, 1]``
        row sums of ``grad[index] * rows``.
    """
    grad = grad.contiguous()
    rows = rows.contiguous()
    n_rows, width = rows.shape
    grad_rows = torch.empty_like(rows)
    grad_weights = torch.empty((n_rows, 1), device=rows.device, dtype=torch.float32)
    if n_rows == 0:
        return grad_rows, grad_weights
    gather_scaled_rows_kernel[(n_rows,)](
        grad,
        index,
        rows,
        weights,
        grad_rows,
        grad_weights,
        width,
        grad.stride(0),
        rows.stride(0),
        weights.stride(0),
    )
    return grad_rows, grad_weights

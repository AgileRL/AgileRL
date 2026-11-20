# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""SRAM-safe Triton tiles for transformers flex attention."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from agilerl.utils.patching import library_shim

if TYPE_CHECKING:
    from torch.nn.attention.flex_attention import BlockMask


def flex_decode_kernel_options(
    query: torch.Tensor, attention_mask: torch.Tensor | BlockMask | None = None
) -> dict[str, Any] | None:
    """Return a ``BLOCK_M`` that keeps short-query flex forwards compilable.

    Queries under 128 tokens hit inductor's flex-decoding kernel, whose default
    ``BLOCK_M`` (``next_power_of_2(seq_len_q * gqa_shared_heads)``) can exceed
    the mask's Q block size; every candidate config then fails a divisibility
    filter and compilation raises ``NoValidChoicesError``. Pinning ``BLOCK_M``
    to the mask's Q block size avoids that on every GPU generation.

    :param query: Query tensor, shaped ``[batch, heads, seq_len_q, head_dim]``.
    :type query: torch.Tensor
    :param attention_mask: Mask passed to flex attention; a ``BlockMask``
        carries the Q block size, anything else uses torch's 128 default.
    :type attention_mask: torch.Tensor | BlockMask | None
    :return: ``{"BLOCK_M": ...}`` for short queries, else ``None``.
    :rtype: dict[str, Any] | None
    """
    shape = getattr(query, "shape", None)
    if shape is None or len(shape) < 2:
        return None
    seq_len_q = shape[-2]
    if not isinstance(seq_len_q, int) or seq_len_q >= 128:
        return None

    q_block_size = 128
    block_size = getattr(attention_mask, "BLOCK_SIZE", None)
    if isinstance(block_size, tuple) and block_size:
        # Maskless calls get one giant block; cap at torch's 128 default.
        q_block_size = min(int(block_size[0]), 128)
    return {"BLOCK_M": q_block_size}


@library_shim("torch", "==2.13.*")
def large_head_dim_kernel_options(
    qk_head_dim: int, v_head_dim: int, element_size: int, shared_memory: int
) -> dict[str, int]:
    """Flex attention Triton tiles for head dims over 256 that fit one thread block.

    Torch's default tiles for these head dims overflow shared memory on A100
    and H100, and their fp32 accumulators exceed 255 registers per thread.

    :param qk_head_dim: Query and key head dim.
    :param v_head_dim: Value head dim.
    :param element_size: Bytes per query element.
    :param shared_memory: Opt-in shared memory per thread block, in bytes.
    :return: Forward and backward ``kernel_options``.
    """
    row_bytes = (qk_head_dim + v_head_dim) * element_size
    # num_stages pipelined tiles plus the tile held across the loop, with 1/8 spare for layout scratch.
    budget = shared_memory * 7 // 8
    # The query tile is live for the whole loop. Halve it until one 16-wide K/V stage fits.
    block_m = 64
    while (
        block_m > 16
        and block_m * qk_head_dim * element_size + 2 * 16 * row_bytes > budget
    ):
        block_m //= 2
    query_tile_bytes = block_m * qk_head_dim * element_size
    forward_block_n = 32 if 2 * 32 * row_bytes + query_tile_bytes <= budget else 16
    forward_stages = (
        2 if 2 * forward_block_n * row_bytes + query_tile_bytes <= budget else 1
    )
    backward_stages = 3 if (3 * 16 + 32) * row_bytes <= budget else 2
    block_m1, block_n1 = 16, 32
    block_m2, block_n2 = 32, 16
    if (backward_stages * block_m1 + block_n1) * row_bytes > budget:
        block_n1 = 16
    if (backward_stages * block_n2 + block_m2) * row_bytes > budget:
        block_m2 = 16
    if (backward_stages * block_m1 + block_n1) * row_bytes > budget:
        backward_stages = 1
    return {
        "BLOCK_M": block_m,
        "BLOCK_N": forward_block_n,
        "fwd_num_stages": forward_stages,
        "fwd_num_warps": 8,
        "BLOCK_M1": block_m1,
        "BLOCK_N1": block_n1,
        "BLOCK_M2": block_m2,
        "BLOCK_N2": block_n2,
        "bwd_num_stages": backward_stages,
        "bwd_num_warps": 8,
    }


def device_shared_memory(query: torch.Tensor) -> int | None:
    """Opt-in shared memory of ``query``'s CUDA device, in bytes.

    :param query: Tensor whose device is queried.
    :return: ``shared_memory_per_block_optin``, or None when ``query`` is not on CUDA.
    """
    if not isinstance(query, torch.Tensor) or query.device.type != "cuda":
        return None
    return int(
        torch.cuda.get_device_properties(query.device).shared_memory_per_block_optin
    )


def _packed_head(tensor: torch.Tensor) -> torch.Tensor:
    """Pack a head row so flex attention's vector load is 16-byte aligned.

    A transpose leaves each head at ``stride(seq) = heads * head_dim``. The
    Triton load addresses that row as a contiguous vector and faults with
    ``misaligned address`` when the base or the row stride is not 16-byte
    aligned.
    """
    if isinstance(tensor, torch.Tensor) and not tensor.is_contiguous():
        return tensor.contiguous()
    return tensor


def _flex_kernel_options(
    query: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | BlockMask | None,
    static: dict[str, Any] | None,
    choose_for_head: bool,
) -> dict[str, Any] | None:
    """Tiles for this query: wide heads from device SRAM, else ``static``."""
    shape = getattr(query, "shape", None)
    head_dim = shape[-1] if shape is not None and len(shape) >= 1 else None
    shared = (
        device_shared_memory(query)
        if choose_for_head and isinstance(head_dim, int) and head_dim > 256
        else None
    )
    if shared is not None and isinstance(head_dim, int):
        value_shape = getattr(value, "shape", None)
        value_dim = (
            value_shape[-1]
            if value_shape is not None and len(value_shape) >= 1
            else head_dim
        )
        if not isinstance(value_dim, int):
            value_dim = head_dim
        return large_head_dim_kernel_options(
            head_dim, value_dim, query.element_size(), shared
        )
    if static is not None:
        return static
    return flex_decode_kernel_options(query, attention_mask)


@library_shim("torch", "==2.13.*")
@library_shim("transformers", "==5.19.*")
def patch_flex_attention_kernel_options(options: dict[str, Any] | None = None) -> None:
    """Inject SRAM-safe Triton ``kernel_options`` into transformers' flex-attn path.

    FlexAttention's default Triton config for large head dims (e.g. Gemma's
    head_dim=256) needs more shared memory than an A100 has (~208 KB required vs
    ~163 KB available), so the autotuner finds "no valid triton configs"
    (OutOfMemoryError: out of resource: triton_tem_fused_flex_attention_0).
    The flex attention function accepts ``kernel_options`` to shrink the block
    sizes / pipeline stages; this registers a wrapper over the
    ``"flex_attention"`` entry that supplies safe defaults when the caller
    passes none. Idempotent; no-op if transformers/flex is unavailable.

    **Auto-detect**: when ``options`` is ``None``, Hopper (SM90+) fits the
    stock tiles for head dims up to 256, so only short-query forwards get a
    ``BLOCK_M`` there (see :func:`flex_decode_kernel_options`) and the
    autotuner keeps everything else. A100 (SM80) gets SRAM-safe small blocks;
    other pre-Hopper GPUs (L4/A10, ~99 KB shared memory) get smaller backward
    tiles. A head dim over 256 uses :func:`large_head_dim_kernel_options` sized
    to that device's shared memory. Query, key, and value are packed first:
    flex attention vector-loads each head as a contiguous row.

    :param options: Override the default kernel options (forward + backward
        block sizes, ``num_warps``, ``num_stages``). Installed unconditionally
        when given.
    :type options: dict[str, Any] | None
    """
    try:
        # transformers flex-attn is optional; no-op when missing.
        from transformers.integrations.flex_attention import flex_attention_forward
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    except Exception:
        return
    patched_flag = "_agilerl_kernel_opts_patched"
    # Wrappers registered over this patch (Ulysses CP) carry the flag too.
    if getattr(ALL_ATTENTION_FUNCTIONS["flex_attention"], patched_flag, False):
        return

    # Small blocks exist only to fit pre-SM90 SRAM; Hopper keeps the
    # autotuner's blocks and gets a decode-safe BLOCK_M per call instead.
    needs_sram_safe_blocks = True
    needs_tiny_blocks = False
    if options is None and torch.cuda.is_available():
        try:
            capability = torch.cuda.get_device_capability()
        except Exception:
            capability = (0, 0)
        needs_sram_safe_blocks = capability < (9, 0)
        # A100 has ~163 KB of shared memory; other pre-Hopper GPUs (L4/A10:
        # ~99 KB) need smaller backward tiles to stay under their limit.
        needs_tiny_blocks = needs_sram_safe_blocks and capability != (8, 0)

    # head_dim<=256 makes the Q/K/V tiles (BLOCK x head_dim) the dominant SRAM
    # cost, so use small 32-wide blocks to fit the A100's ~163 KB shared memory.
    opts = options
    choose_for_head = options is None
    if opts is None and needs_sram_safe_blocks:
        opts = {
            # Forward kernel blocks.
            "BLOCK_M": 32,
            "BLOCK_N": 32,
            # Backward kernel blocks (training).
            "BLOCK_M1": 16,
            "BLOCK_N1": 32,
            "BLOCK_M2": 32,
            "BLOCK_N2": 16,
            "num_warps": 4,
            "num_stages": 2,
        }
        if needs_tiny_blocks:
            opts["BLOCK_N1"] = 16
            opts["BLOCK_M2"] = 16

    def _flex_with_opts(
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | BlockMask,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        query = _packed_head(query)
        key = _packed_head(key)
        value = _packed_head(value)
        resolved = _flex_kernel_options(
            query, value, attention_mask, opts, choose_for_head
        )
        if resolved is not None:
            kwargs.setdefault("kernel_options", resolved)
        return flex_attention_forward(
            module, query, key, value, attention_mask, **kwargs
        )

    # Dynamic marker attribute on the wrapper function (checked via getattr
    # above); function attributes are not statically declarable.
    object.__setattr__(_flex_with_opts, patched_flag, True)
    try:
        ALL_ATTENTION_FUNCTIONS["flex_attention"] = _flex_with_opts
    except Exception:
        ALL_ATTENTION_FUNCTIONS.register("flex_attention", _flex_with_opts)

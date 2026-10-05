# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""SRAM-safe Triton tiles for transformers flex attention."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

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
    stock tiles, so only short-query forwards get a ``BLOCK_M`` there (see
    :func:`flex_decode_kernel_options`) and the autotuner keeps everything
    else. A100 (SM80) gets SRAM-safe small blocks; other pre-Hopper GPUs
    (L4/A10, ~99 KB shared memory) get smaller backward tiles.

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
    if getattr(flex_attention_forward, patched_flag, False):
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

    # head_dim=256 makes the Q/K/V tiles (BLOCK x head_dim) the dominant SRAM
    # cost, so use small 32-wide blocks to fit the A100's ~163 KB shared memory.
    opts = options
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
        resolved = (
            opts
            if opts is not None
            else flex_decode_kernel_options(query, attention_mask)
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

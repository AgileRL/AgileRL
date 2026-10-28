# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Torch-native varlen attention as a transformers attention implementation.

Registers ``torch_varlen`` in ``ALL_ATTENTION_FUNCTIONS``: packed rows run
:func:`torch.nn.attention.varlen.varlen_attn` (no flash-attn wheel),
unpacked rows fall back to SDPA. Opt-in per run with
``attn_implementation="torch_varlen"``.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.nn.attention.varlen import varlen_attn
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.modeling_flash_attention_utils import (
    _is_packed_sequence,
    prepare_fa_kwargs_from_position_ids,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

TORCH_VARLEN_IMPLEMENTATION = "torch_varlen"

VARLEN_KWARGS = frozenset(
    {"cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k"}
)


def _varlen_layout(
    query: torch.Tensor, causal: bool, kwargs: dict[str, Any]
) -> tuple[Any, Any, Any, Any]:
    """Varlen segment layout from kwargs, or derived from packed positions."""
    cu_q = kwargs.get("cu_seq_lens_q")
    cu_k = kwargs.get("cu_seq_lens_k")
    max_q = kwargs.get("max_length_q")
    max_k = kwargs.get("max_length_k")
    if (
        (cu_q is None or cu_k is None or max_q is None or max_k is None)
        and causal
        and _is_packed_sequence(kwargs.get("position_ids"), query.shape[0])
    ):
        (cu_q, cu_k), (max_q, max_k) = prepare_fa_kwargs_from_position_ids(
            kwargs["position_ids"]
        )
    return cu_q, cu_k, max_q, max_k


def _sdpa_fallback(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float,
    scaling: float | None,
    is_causal: bool | None,
    kwargs: dict[str, Any],
) -> tuple[torch.Tensor, None]:
    """SDPA for rows the varlen path cannot take, without the layout kwargs."""
    passthrough = {
        name: value for name, value in kwargs.items() if name not in VARLEN_KWARGS
    }
    return sdpa_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask,
        dropout=dropout,
        scaling=scaling,
        is_causal=is_causal,
        **passthrough,
    )


def torch_varlen_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    sliding_window: int | None = None,
    softcap: float | None = None,
    is_causal: bool | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Run packed rows on torch varlen, unpacked rows on SDPA.

    ``cu_seq_lens_q/k`` plus ``max_length_q/k`` select the varlen path
    directly; otherwise a single packed row derives them from
    ``position_ids`` like the flash path. Anything else (unpacked rows,
    paged cache, non-causal) runs SDPA, which also decides dropout there.

    :param module: Attention module (heads and ``is_causal``/``config``).
    :param query: ``(B, Hq, Lq, D)`` queries.
    :param key: ``(B, Hkv, Lk, D)`` keys.
    :param value: ``(B, Hkv, Lk, Dv)`` values.
    :param attention_mask: Mask for the SDPA fallback; ignored on the varlen
        path, where ``cu_seq_lens`` define the segments.
    :param dropout: Attention dropout; only the SDPA fallback supports it.
    :param scaling: Softmax scale; defaults to ``1/sqrt(D)``.
    :param sliding_window: Sliding-window size, mirrored from the flash
        mapping; ``None`` reads ``module.config.sliding_window``.
    :param softcap: Rejected, varlen has none.
    :param is_causal: Causality; defaults to ``module.is_causal``.
    :return: ``((B, Lq, Hq, D), None)``.
    """
    if softcap is not None:
        msg = (
            "torch_varlen has no softcap support; use flash_attention_2 or "
            "flex_attention."
        )
        raise ValueError(msg)
    causal = bool(is_causal if is_causal is not None else module.is_causal)
    cu_q, cu_k, max_q, max_k = _varlen_layout(query, causal, kwargs)
    if (
        cu_q is None
        or cu_k is None
        or max_q is None
        or max_k is None
        or not causal
        or kwargs.get("cache") is not None
    ):
        return _sdpa_fallback(
            module,
            query,
            key,
            value,
            attention_mask,
            dropout,
            scaling,
            is_causal,
            kwargs,
        )
    if dropout:
        msg = (
            "torch_varlen has no dropout support; use flash_attention_2 or "
            "flex_attention."
        )
        raise ValueError(msg)
    if kwargs.get("s_aux") is not None:
        msg = "torch_varlen has no attention-sink support; use flash_attention_2."
        raise ValueError(msg)
    window = sliding_window
    if window is None:
        window = getattr(module.config, "sliding_window", None)
    batches, heads, rows, head_dim = query.shape
    kv_heads = key.shape[1]
    query = query.transpose(1, 2).reshape(-1, heads, head_dim)
    key = key.transpose(1, 2).reshape(-1, kv_heads, head_dim)
    value = value.transpose(1, 2).reshape(-1, kv_heads, value.shape[3])
    out = varlen_attn(
        query,
        key,
        value,
        cu_q,
        cu_k,
        int(max_q),
        int(max_k),
        scale=scaling if scaling is not None else head_dim**-0.5,
        window_size=(-1, 0) if window is None else (window - 1, 0),
        enable_gqa=heads != kv_heads,
    )
    assert isinstance(out, torch.Tensor)
    return out.reshape(batches, rows, heads, head_dim), None


def register_torch_varlen_attention() -> None:
    """Register ``torch_varlen`` in transformers' attention registry."""
    if TORCH_VARLEN_IMPLEMENTATION in ALL_ATTENTION_FUNCTIONS.valid_keys():
        return
    ALL_ATTENTION_FUNCTIONS.register(
        TORCH_VARLEN_IMPLEMENTATION, torch_varlen_attention_forward
    )

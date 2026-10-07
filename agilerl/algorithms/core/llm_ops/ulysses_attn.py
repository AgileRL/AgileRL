# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Ulysses context parallelism for dense attention layers.

Sequence-sharded Q/K/V are redistributed to head-sharded full-sequence
tensors with two all-to-alls, local flash attention runs on the full
sequence with ``H/cp`` heads, then the inverse all-to-all restores the
sequence-shard layout. Leading batch dims are flattened through the exchange
and restored after, so packed ``(1, N)`` rows and row batches share one path.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from types import TracebackType
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
import transformers.modeling_flash_attention_utils
from torch import nn

if TYPE_CHECKING:
    from transformers import PretrainedConfig

ULYSSES_PARAMS: dict[str, torch.Tensor | int] = {}


def update_ulysses_params(cu_seqlens: torch.Tensor, max_seqlen: int) -> None:
    """Publish full-sequence varlen params for the next Ulysses forward."""
    ULYSSES_PARAMS["cu_seqlens"] = cu_seqlens
    ULYSSES_PARAMS["max_seqlen"] = int(max_seqlen)


def clear_ulysses_params() -> None:
    """Drop published varlen params (stock attention resumes)."""
    ULYSSES_PARAMS.clear()


class PublishedUlyssesParams(AbstractContextManager[None]):
    """Publish fixed params inside the block, then restore what was published.

    Re-enterable: checkpoint may run one block's recompute more than once.
    """

    def __init__(self, params: dict[str, torch.Tensor | int]) -> None:
        self.params = params
        self.previous: dict[str, torch.Tensor | int] = {}

    def __enter__(self) -> None:
        self.previous = dict(ULYSSES_PARAMS)
        ULYSSES_PARAMS.clear()
        ULYSSES_PARAMS.update(self.params)

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> bool | None:
        ULYSSES_PARAMS.clear()
        ULYSSES_PARAMS.update(self.previous)
        return None


def ulysses_checkpoint_context_fn() -> tuple[
    AbstractContextManager[None, bool | None],
    AbstractContextManager[None, bool | None],
]:
    """``context_fn`` for non-reentrant activation checkpointing.

    The recompute runs in backward, after the forward cleared
    :data:`ULYSSES_PARAMS`. Re-publish the params captured at forward time so
    the recomputed block uses the same attention path as its forward.
    """
    return nullcontext(), PublishedUlyssesParams(dict(ULYSSES_PARAMS))


def replicate_kv_heads(data: torch.Tensor, cp_size: int) -> torch.Tensor:
    """Replicate KV heads so GQA with ``H_kv < cp`` can be head-sharded.

    ``(..., S_local, H_kv, D) -> (..., S_local, cp_size, D)``. Each KV head is
    repeated ``cp_size / H_kv`` times; backward sums over the replicas.
    """
    heads = data.shape[-2]
    if cp_size % heads != 0:
        msg = (
            f"num_key_value_heads ({heads}) must divide cp_size ({cp_size}) "
            "for the Ulysses KV-replication path"
        )
        raise ValueError(msg)
    return data.repeat_interleave(cp_size // heads, dim=-2)


def all_to_all_seq_to_head(
    data: torch.Tensor, cp_size: int, cp_group: dist.ProcessGroup
) -> torch.Tensor:
    """Redistribute ``(..., S_local, H, D) -> (..., S_global, H_local, D)``."""
    *leading, seq_local, heads, depth = data.shape
    if heads % cp_size != 0:
        msg = (
            f"num_heads ({heads}) must be divisible by cp_size ({cp_size}); "
            "for GQA KV tensors with fewer heads, replicate first via "
            "replicate_kv_heads"
        )
        raise ValueError(msg)
    heads_local = heads // cp_size
    flat = data.reshape(-1, seq_local, heads, depth)
    batch = flat.shape[0]
    # all_to_all_single splits dimension 0.
    swapped = (
        flat.reshape(batch, seq_local, cp_size, heads_local, depth)
        .permute(2, 0, 1, 3, 4)
        .contiguous()
    )
    exchanged = dist_nn.all_to_all_single(
        torch.empty_like(swapped), swapped, group=cp_group
    )
    gathered = (
        exchanged.permute(1, 0, 2, 3, 4)
        .contiguous()
        .reshape(batch, cp_size * seq_local, heads_local, depth)
    )
    return gathered.reshape(*leading, cp_size * seq_local, heads_local, depth)


def all_to_all_head_to_seq(
    data: torch.Tensor, cp_size: int, cp_group: dist.ProcessGroup
) -> torch.Tensor:
    """Inverse of :func:`all_to_all_seq_to_head`."""
    *leading, seq_global, heads_local, depth = data.shape
    if seq_global % cp_size != 0:
        msg = (
            f"global sequence length ({seq_global}) must be divisible by "
            f"cp_size ({cp_size})"
        )
        raise ValueError(msg)
    seq_local = seq_global // cp_size
    flat = data.reshape(-1, seq_global, heads_local, depth)
    batch = flat.shape[0]
    swapped = (
        flat.reshape(batch, cp_size, seq_local, heads_local, depth)
        .permute(1, 0, 2, 3, 4)
        .contiguous()
    )
    exchanged = dist_nn.all_to_all_single(
        torch.empty_like(swapped), swapped, group=cp_group
    )
    restored = (
        exchanged.permute(1, 2, 0, 3, 4)
        .contiguous()
        .reshape(batch, seq_local, heads_local * cp_size, depth)
    )
    return restored.reshape(*leading, seq_local, heads_local * cp_size, depth)


def ulysses_flash_attn_varlen_func(
    flash_fn: Callable[..., torch.Tensor],
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: bool,
    cp_group: dist.ProcessGroup,
    cp_size: int,
    window_size: tuple[int, int] = (-1, -1),
    softmax_scale: float | None = None,
    dropout_p: float = 0.0,
    deterministic: bool | None = None,
) -> torch.Tensor:
    """Run varlen flash attention under Ulysses CP (full seq, ``H/cp`` heads)."""
    query = all_to_all_seq_to_head(query, cp_size, cp_group)
    if key.shape[-2] < cp_size:
        key = replicate_kv_heads(key, cp_size)
        value = replicate_kv_heads(value, cp_size)
    key = all_to_all_seq_to_head(key, cp_size, cp_group)
    value = all_to_all_seq_to_head(value, cp_size, cp_group)

    kwargs: dict[str, bool | float | tuple[int, int] | None] = {"causal": causal}
    if window_size != (-1, -1):
        kwargs["window_size"] = window_size
    if softmax_scale is not None:
        kwargs["softmax_scale"] = softmax_scale
    if dropout_p:
        kwargs["dropout_p"] = dropout_p
    if deterministic is not None:
        kwargs["deterministic"] = deterministic

    # varlen flash is (total_tokens, heads, dim). Packed (1, N) and row
    # batches keep a leading dim through the all-to-alls; flatten it here.
    q_shape = query.shape
    query = query.reshape(-1, query.shape[-2], query.shape[-1]).contiguous()
    key = key.reshape(-1, key.shape[-2], key.shape[-1]).contiguous()
    value = value.reshape(-1, value.shape[-2], value.shape[-1]).contiguous()
    out = flash_fn(
        query,
        key,
        value,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        **kwargs,
    )
    if isinstance(out, tuple):
        out = out[0]
    out = out.reshape(q_shape)
    return all_to_all_head_to_seq(out, cp_size, cp_group)


def model_config(model: nn.Module) -> PretrainedConfig | None:
    """Innermost HF config, unwrapping PEFT containers.

    Checks the model, its ``base_model``, and the base's ``model`` in order,
    covering plain causal LMs and PEFT wrappers.
    """
    candidates = [model]
    base = getattr(model, "base_model", None)
    if base is not None:
        candidates.append(base)
        inner = getattr(base, "model", None)
        if inner is not None:
            candidates.append(inner)
    for candidate in candidates:
        config = getattr(candidate, "config", None)
        if config is not None:
            return config
    return None


def attention_head_counts(model: nn.Module) -> tuple[int, int | None]:
    """Read ``(num_attention_heads, num_key_value_heads?)`` off a model.

    Works on plain and PEFT-wrapped causal LMs via their ``config``.
    """
    config = model_config(model)
    heads = getattr(config, "num_attention_heads", None) if config is not None else None
    if heads is None:
        msg = (
            "CP requires a model config with num_attention_heads; "
            f"got {type(model).__name__} without one."
        )
        raise ValueError(msg)
    kv_heads = getattr(config, "num_key_value_heads", None)
    return int(heads), None if kv_heads is None else int(kv_heads)


def model_attention_backend(model: nn.Module) -> str | None:
    """Resolved transformers attention backend, if the model records one."""
    config = model_config(model)
    return getattr(config, "_attn_implementation", None) if config is not None else None


def substitute_hf_ulysses_attn(
    process_group: dist.ProcessGroup,
    *,
    patch_all_attention_functions: bool = True,
) -> None:
    """Patch HF flash-attention-2 entrypoints for Ulysses all-to-all.

    Forwards without published :data:`ULYSSES_PARAMS` (rollout, generate, any
    full-sequence path) keep stock flash attention, so CP stays train-only.
    Safe to call once per process; repeat calls are no-ops.
    """
    if importlib.util.find_spec("flash_attn") is None:
        msg = (
            "Context parallel needs the flash_attn package for the Ulysses "
            "forward; install it to train with cp > 1."
        )
        raise ImportError(msg)
    from flash_attn import flash_attn_varlen_func

    if getattr(
        transformers.modeling_flash_attention_utils._flash_attention_forward,
        "_agilerl_ulysses",
        False,
    ):
        return
    cp_size = dist.get_world_size(group=process_group)
    _original_flash_attention_forward = (
        transformers.modeling_flash_attention_utils._flash_attention_forward
    )

    def _ulysses_flash_attention_forward(
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        query_length: int,
        is_causal: bool,
        dropout: float = 0.0,
        position_ids: torch.Tensor | None = None,
        softmax_scale: float | None = None,
        sliding_window: int | None = None,
        use_top_left_mask: bool = False,
        softcap: float | None = None,
        deterministic: bool | None = None,
        cu_seq_lens_q: torch.LongTensor | None = None,
        cu_seq_lens_k: torch.LongTensor | None = None,
        max_length_q: int | None = None,
        max_length_k: int | None = None,
        target_dtype: torch.dtype | None = None,
        attn_implementation: str | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if "cu_seqlens" not in ULYSSES_PARAMS:
            return _original_flash_attention_forward(
                query_states,
                key_states,
                value_states,
                attention_mask,
                query_length,
                is_causal,
                dropout=dropout,
                position_ids=position_ids,
                softmax_scale=softmax_scale,
                sliding_window=sliding_window,
                use_top_left_mask=use_top_left_mask,
                softcap=softcap,
                deterministic=deterministic,
                cu_seq_lens_q=cu_seq_lens_q,
                cu_seq_lens_k=cu_seq_lens_k,
                max_length_q=max_length_q,
                max_length_k=max_length_k,
                target_dtype=target_dtype,
                attn_implementation=attn_implementation,
                **kwargs,
            )
        if not is_causal:
            msg = "Ulysses CP only supports causal attention."
            raise AssertionError(msg)
        if softcap is not None:
            msg = "Ulysses CP path does not support attention softcap."
            raise AssertionError(msg)
        cu_seqlens = ULYSSES_PARAMS["cu_seqlens"]
        max_seqlen = ULYSSES_PARAMS["max_seqlen"]
        if not isinstance(cu_seqlens, torch.Tensor):
            msg = "Ulysses cu_seqlens is not published."
            raise RuntimeError(msg)
        if not isinstance(max_seqlen, int):
            msg = "Ulysses max_seqlen is not published."
            raise RuntimeError(msg)
        window_size = (-1, -1)
        if sliding_window is not None and key_states.shape[1] > sliding_window:
            window_size = (sliding_window, sliding_window)
        return ulysses_flash_attn_varlen_func(
            flash_attn_varlen_func,
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            causal=True,
            cp_group=process_group,
            cp_size=cp_size,
            window_size=window_size,
            softmax_scale=softmax_scale,
            dropout_p=dropout,
            deterministic=deterministic,
        )

    patched = cast("Any", _ulysses_flash_attention_forward)
    patched._agilerl_ulysses = True
    cast(
        "Any", transformers.modeling_flash_attention_utils
    )._flash_attention_forward = _ulysses_flash_attention_forward

    if not patch_all_attention_functions:
        return
    attention_functions = None
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        attention_functions = ALL_ATTENTION_FUNCTIONS
    except ImportError:
        attention_functions = None
    if attention_functions is None:
        return
    _original_all_attn = attention_functions.get("flash_attention_2")

    def _ulysses_flash_attention_forward_v2(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        dropout: float = 0.0,
        scaling: float | None = None,
        sliding_window: int | None = None,
        softcap: float | None = None,
        **kw: Any,
    ) -> tuple[torch.Tensor, None]:
        if "cu_seqlens" not in ULYSSES_PARAMS and _original_all_attn is not None:
            return _original_all_attn(
                module,
                query,
                key,
                value,
                attention_mask,
                dropout=dropout,
                scaling=scaling,
                sliding_window=sliding_window,
                softcap=softcap,
                **kw,
            )
        seq_len = query.shape[2]
        query_t = query.transpose(1, 2)
        key_t = key.transpose(1, 2)
        value_t = value.transpose(1, 2)
        kw.pop("is_causal", None)
        attn_out = _ulysses_flash_attention_forward(
            query_t,
            key_t,
            value_t,
            attention_mask,
            query_length=seq_len,
            is_causal=bool(module.is_causal),
            dropout=dropout,
            softmax_scale=scaling,
            sliding_window=sliding_window,
            softcap=softcap,
        )
        return attn_out, None

    attention_functions["flash_attention_2"] = _ulysses_flash_attention_forward_v2

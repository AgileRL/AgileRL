# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Ulysses context parallelism for dense attention layers.

Sequence-sharded Q/K/V are redistributed to head-sharded full-sequence
tensors with one all-to-all, local flash or flex attention runs on the full
sequence with ``H/cp`` heads, then the inverse all-to-all restores the
sequence-shard layout. Leading batch dims are flattened through the exchange
and restored after, so packed ``(1, N)`` rows and row batches share one path.
"""

from __future__ import annotations

import importlib.util
import inspect
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from functools import lru_cache, wraps
from itertools import pairwise
from types import TracebackType
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

from agilerl import HAS_LLM_DEPENDENCIES
from agilerl.distributed.context_parallel.sequence import (
    all_to_all_hidden_to_seq,
    all_to_all_seq_to_hidden,
)

if TYPE_CHECKING or HAS_LLM_DEPENDENCIES:
    import transformers.modeling_flash_attention_utils
    from peft import PeftModel
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

if TYPE_CHECKING:
    from transformers import PretrainedConfig

ULYSSES_PARAMS: dict[str, object] = {}


def update_ulysses_params(
    cu_seqlens: torch.Tensor, host_cu_seqlens: tuple[int, ...]
) -> None:
    """Publish full-sequence varlen offsets for the next Ulysses forward.

    Replaces every published param, so kernel contexts built from earlier
    offsets drop with them.

    :param cu_seqlens: Varlen offsets on the forward's device.
    :param host_cu_seqlens: The same offsets on the host.
    """
    ULYSSES_PARAMS.clear()
    ULYSSES_PARAMS["cu_seqlens"] = cu_seqlens
    ULYSSES_PARAMS["host_cu_seqlens"] = host_cu_seqlens
    ULYSSES_PARAMS["max_seqlen"] = max(
        (stop - start for start, stop in pairwise(host_cu_seqlens)), default=0
    )
    # Mixer kernel contexts of these offsets; recompute shares this dict.
    ULYSSES_PARAMS["contexts"] = {}


def clear_ulysses_params() -> None:
    """Drop published varlen params (stock attention resumes)."""
    ULYSSES_PARAMS.clear()


class PublishedUlyssesParams(AbstractContextManager[None]):
    """Publish fixed params inside the block, then restore what was published.

    Re-enterable: checkpoint may run one block's recompute more than once.
    """

    def __init__(self, params: dict[str, object]) -> None:
        self.params = params
        self.previous: dict[str, object] = {}

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
    # (forward, recompute): forward already has ULYSSES_PARAMS published.
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


def all_to_all_qkv_seq_to_head(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cp_size: int,
    cp_group: dist.ProcessGroup,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One all-to-all of q, k and v ``(..., S_local, H, D) -> (..., S_global, H/cp, D)``.

    :param query: Sequence-shard queries.
    :param key: Sequence-shard keys, already replicated to at least ``cp_size`` heads.
    :param value: Sequence-shard values, with as many heads as ``key``.
    :param cp_size: Context-parallel degree.
    :param cp_group: Context-parallel process group.
    :return: Head-shard full-sequence query, key and value.
    """
    tensors = (query, key, value)
    for tensor in tensors:
        if tensor.shape[-2] % cp_size != 0:
            msg = (
                f"num_heads ({tensor.shape[-2]}) must be divisible by cp_size "
                f"({cp_size}); for GQA KV tensors with fewer heads, replicate "
                "first via replicate_kv_heads"
            )
            raise ValueError(msg)
    *leading, seq_local, _heads, _depth = query.shape
    # Each rank's heads of q, k and v sit side by side in its 1/cp of the last dim.
    per_rank = [
        tensor.reshape(
            -1, seq_local, cp_size, tensor.shape[-2] // cp_size * tensor.shape[-1]
        )
        for tensor in tensors
    ]
    packed = torch.cat(per_rank, dim=-1)
    exchanged = all_to_all_seq_to_hidden(
        packed.reshape(packed.shape[0], seq_local, -1), cp_size, cp_group
    )
    split = exchanged.split([part.shape[-1] for part in per_rank], dim=-1)
    query_h, key_h, value_h = (
        part.reshape(
            *leading,
            cp_size * seq_local,
            tensor.shape[-2] // cp_size,
            tensor.shape[-1],
        )
        for part, tensor in zip(split, tensors, strict=True)
    )
    return query_h, key_h, value_h


def all_to_all_head_to_seq(
    data: torch.Tensor, cp_size: int, cp_group: dist.ProcessGroup
) -> torch.Tensor:
    """Redistribute ``(..., S_global, H/cp, D) -> (..., S_local, H, D)``."""
    *leading, seq_global, heads_local, depth = data.shape
    restored = all_to_all_hidden_to_seq(
        data.reshape(-1, seq_global, heads_local * depth), cp_size, cp_group
    )
    return restored.reshape(
        *leading, seq_global // cp_size, heads_local * cp_size, depth
    )


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
    if key.shape[-2] < cp_size:
        key = replicate_kv_heads(key, cp_size)
        value = replicate_kv_heads(value, cp_size)
    query, key, value = all_to_all_qkv_seq_to_head(query, key, value, cp_size, cp_group)

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
    out = out.reshape(q_shape)
    return all_to_all_head_to_seq(out, cp_size, cp_group)


def model_config(model: nn.Module) -> PretrainedConfig:
    """Language-model HF config of ``model`` or its PEFT base; a VL config yields its ``text_config``."""
    if isinstance(model, PeftModel):
        model = model.get_base_model()
    return cast("PretrainedConfig", model.config).get_text_config()


def attention_head_counts(model: nn.Module) -> tuple[int, int | None]:
    """Read ``(num_attention_heads, num_key_value_heads?)`` off a model.

    Works on plain and PEFT-wrapped causal LMs via their ``config``.
    """
    config = model_config(model)
    # Multi-head-attention configs omit num_key_value_heads.
    kv_heads = getattr(config, "num_key_value_heads", None)
    return int(config.num_attention_heads), None if kv_heads is None else int(kv_heads)


def model_attention_backend(model: nn.Module) -> str | None:
    """Resolved transformers attention backend, if the model records one."""
    return model_config(model)._attn_implementation


def flash_window_size(sliding_window: int | None, key_length: int) -> tuple[int, int]:
    """Inclusive flash window for the gathered sequence, or full causal.

    :param sliding_window: Tokens in the window, counting the query, or ``None``.
    :param key_length: Sequence length flash sees after the all-to-all.
    :return: ``(left, right)`` for flash, or ``(-1, -1)`` for unwindowed causal.
    """
    if sliding_window is None or key_length <= sliding_window:
        return (-1, -1)
    return (sliding_window - 1, sliding_window - 1)


@lru_cache(maxsize=8)
def full_sequence_block_mask(
    seq_len: int, sliding_window: int | None, device: torch.device
) -> BlockMask:
    """Causal flex mask over the gathered sequence, windowed when ``sliding_window`` is set.

    :param seq_len: Full sequence length after the sequence-to-head all-to-all.
    :param sliding_window: Keys per query, counting the query, or ``None`` for full causal.
    :param device: Device the mask lives on.
    :return: A ``BlockMask`` shared by every batch row and head.
    """

    def causal(
        _batch: torch.Tensor,
        _head: torch.Tensor,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
    ) -> torch.Tensor:
        return kv_idx <= q_idx

    def windowed(
        _batch: torch.Tensor,
        _head: torch.Tensor,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
    ) -> torch.Tensor:
        return (kv_idx <= q_idx) & (kv_idx > q_idx - cast("int", sliding_window))

    return create_block_mask(
        causal if sliding_window is None else windowed,
        B=None,
        H=None,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device=device,
        # Uncompiled masks materialise seq_len x seq_len on the device.
        _compile=device.type == "cuda",
    )


def cp_local_heads(per_head: torch.Tensor, cp_group: dist.ProcessGroup) -> torch.Tensor:
    """This rank's contiguous chunk of a per-head tensor, matching the all-to-all head split."""
    cp_size = dist.get_world_size(group=cp_group)
    return per_head.reshape(cp_size, -1)[dist.get_rank(group=cp_group)]


def _substitute_ulysses_flex_attn(process_group: dist.ProcessGroup) -> None:
    original = ALL_ATTENTION_FUNCTIONS["flex_attention"]
    if getattr(original, "_agilerl_ulysses", False):
        return
    cp_size = dist.get_world_size(group=process_group)

    @wraps(original)
    def _ulysses_flex_attention_forward(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | BlockMask | None,
        sliding_window: int | None = None,
        s_aux: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        if "cu_seqlens" not in ULYSSES_PARAMS:
            return original(
                module,
                query,
                key,
                value,
                attention_mask,
                sliding_window=sliding_window,
                s_aux=s_aux,
                **kwargs,
            )
        if not module.is_causal:
            msg = "Ulysses CP only supports causal attention."
            raise ValueError(msg)
        # (B, H, S_local, D) -> (B, S, H/cp, D) for the exchange.
        key_t = key.transpose(1, 2)
        value_t = value.transpose(1, 2)
        if key_t.shape[-2] < cp_size:
            key_t = replicate_kv_heads(key_t, cp_size)
            value_t = replicate_kv_heads(value_t, cp_size)
        query_t, key_t, value_t = all_to_all_qkv_seq_to_head(
            query.transpose(1, 2), key_t, value_t, cp_size, process_group
        )
        # The model built its mask for the shard; the gathered sequence needs its own.
        block_mask = full_sequence_block_mask(
            query_t.shape[1], sliding_window, query_t.device
        )
        local_sinks = None if s_aux is None else cp_local_heads(s_aux, process_group)
        out, _ = original(
            module,
            query_t.transpose(1, 2),
            key_t.transpose(1, 2),
            value_t.transpose(1, 2),
            block_mask,
            sliding_window=sliding_window,
            s_aux=local_sinks,
            **kwargs,
        )
        return all_to_all_head_to_seq(out, cp_size, process_group), None

    patched = cast("Any", _ulysses_flex_attention_forward)
    patched._agilerl_ulysses = True
    ALL_ATTENTION_FUNCTIONS["flex_attention"] = _ulysses_flex_attention_forward


def substitute_hf_ulysses_attn(process_group: dist.ProcessGroup, backend: str) -> None:
    """Patch HF flash-attention-2 or flex-attention entrypoints for Ulysses all-to-all.

    Forwards without published :data:`ULYSSES_PARAMS` (rollout, generate, any
    full-sequence path) keep stock attention, so CP stays train-only.
    Safe to call once per process; repeat calls are no-ops.

    :param process_group: Context-parallel process group.
    :param backend: The actor's attention backend, ``flash_attention_2`` or ``flex_attention``.
    """
    if backend == "flex_attention":
        _substitute_ulysses_flex_attn(process_group)
        return
    if importlib.util.find_spec("flash_attn") is None:
        msg = (
            "Context parallel needs the flash_attn package for the Ulysses "
            "forward; install it to train with cp > 1."
        )
        raise ImportError(msg)
    # flash_attn is an optional extra.
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

    original_signature = inspect.signature(_original_flash_attention_forward)

    def _ulysses_flash_attention_forward(*args: Any, **kwargs: Any) -> torch.Tensor:
        if "cu_seqlens" not in ULYSSES_PARAMS:
            return _original_flash_attention_forward(*args, **kwargs)
        bound = original_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        call = bound.arguments
        if not call["is_causal"]:
            msg = "Ulysses CP only supports causal attention."
            raise ValueError(msg)
        if call["softcap"] is not None:
            msg = "Ulysses CP path does not support attention softcap."
            raise ValueError(msg)
        cu_seqlens = cast("torch.Tensor", ULYSSES_PARAMS["cu_seqlens"])
        max_seqlen = cast("int", ULYSSES_PARAMS["max_seqlen"])
        window_size = flash_window_size(call["sliding_window"], max_seqlen)
        return ulysses_flash_attn_varlen_func(
            flash_attn_varlen_func,
            call["query_states"],
            call["key_states"],
            call["value_states"],
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            causal=True,
            cp_group=process_group,
            cp_size=cp_size,
            window_size=window_size,
            softmax_scale=call["softmax_scale"],
            dropout_p=call["dropout"],
            deterministic=call["deterministic"],
        )

    patched = cast("Any", _ulysses_flash_attention_forward)
    patched._agilerl_ulysses = True
    cast(
        "Any", transformers.modeling_flash_attention_utils
    )._flash_attention_forward = _ulysses_flash_attention_forward

    _original_all_attn = ALL_ATTENTION_FUNCTIONS["flash_attention_2"]

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
        if "cu_seqlens" not in ULYSSES_PARAMS:
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

    ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = _ulysses_flash_attention_forward_v2

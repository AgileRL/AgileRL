"""Ulysses-style context parallelism via all-to-all on head/seq dimensions.

Sequence-sharded Q/K/V are redistributed to head-sharded full-sequence tensors,
local FA2 runs on the full sequence with ``H/cp`` heads, then the inverse
all-to-all restores the sequence shard layout.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn

# Populated by ``update_ulysses_params`` before each forward. Mirrors
# ring_flash_attn's DATA_PARAMS pattern so patched attention can reach the
# full (un-sharded) cu_seqlens / max_seqlen at call time.
ULYSSES_PARAMS: dict = {}


def update_ulysses_params(cu_seqlens: torch.Tensor, max_seqlen: int) -> None:
    """Publish full-sequence FA2 varlen params for the next Ulysses forward."""
    ULYSSES_PARAMS["cu_seqlens"] = cu_seqlens
    ULYSSES_PARAMS["max_seqlen"] = int(max_seqlen)


def clear_ulysses_params() -> None:
    """Drop published FA2 varlen params (stock FA2 path resumes)."""
    ULYSSES_PARAMS.clear()


def _replicate_kv_heads(t: torch.Tensor, cp_size: int) -> torch.Tensor:
    """Replicate KV heads so GQA with ``H_kv < cp`` can be head-sharded.

    ``[S_local, H_kv, D] -> [S_local, cp_size, D]``. Each KV head is repeated
    ``cp_size / H_kv`` times. Backward sums gradients over replicas.
    """
    s_local, h, d = t.shape
    if cp_size % h != 0:
        raise ValueError(
            f"num_key_value_heads ({h}) must divide cp_size ({cp_size}) "
            "for the ulysses KV-replication path"
        )
    return t.repeat_interleave(cp_size // h, dim=1)


def _all_to_all_seq_to_head(
    t: torch.Tensor, cp_size: int, cp_group: dist.ProcessGroup
) -> torch.Tensor:
    """Redistribute ``[S_local, H, D] -> [S_global, H_local, D]``."""
    s_local, h, d = t.shape
    if h % cp_size != 0:
        raise ValueError(
            f"num_heads ({h}) must be divisible by cp_size ({cp_size}); "
            "for GQA KV tensors with fewer heads than cp_size, replicate first "
            "via _replicate_kv_heads"
        )
    h_local = h // cp_size
    t = t.reshape(s_local, cp_size, h_local, d).transpose(0, 1).contiguous()
    output = torch.empty_like(t)
    out = dist_nn.all_to_all_single(output, t, group=cp_group)
    return out.reshape(cp_size * s_local, h_local, d)


def _all_to_all_head_to_seq(
    t: torch.Tensor, cp_size: int, cp_group: dist.ProcessGroup
) -> torch.Tensor:
    """Inverse of ``_all_to_all_seq_to_head``: ``[S_global, H_local, D] -> [S_local, H, D]``."""
    s_global, h_local, d = t.shape
    if s_global % cp_size != 0:
        raise ValueError(
            f"global sequence length ({s_global}) must be divisible by cp_size ({cp_size})"
        )
    s_local = s_global // cp_size
    h = h_local * cp_size
    t = t.reshape(cp_size, s_local, h_local, d).contiguous()
    output = torch.empty_like(t)
    out = dist_nn.all_to_all_single(output, t, group=cp_group)
    return out.transpose(0, 1).contiguous().reshape(s_local, h, d)


def ulysses_flash_attn_varlen_func(
    flash_fn,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
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
    """Run varlen FA2 under Ulysses CP (full seq, ``H/cp`` heads locally)."""
    q = _all_to_all_seq_to_head(q, cp_size, cp_group)
    if k.shape[1] < cp_size:
        k = _replicate_kv_heads(k, cp_size)
        v = _replicate_kv_heads(v, cp_size)
    k = _all_to_all_seq_to_head(k, cp_size, cp_group)
    v = _all_to_all_seq_to_head(v, cp_size, cp_group)

    kwargs: dict = {"causal": causal}
    if window_size != (-1, -1):
        kwargs["window_size"] = window_size
    if softmax_scale is not None:
        kwargs["softmax_scale"] = softmax_scale
    if dropout_p:
        kwargs["dropout_p"] = dropout_p
    if deterministic is not None:
        kwargs["deterministic"] = deterministic

    out = flash_fn(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        **kwargs,
    )
    if isinstance(out, tuple):
        out = out[0]
    return _all_to_all_head_to_seq(out, cp_size, cp_group)


def substitute_hf_ulysses_attn(
    process_group: dist.ProcessGroup,
    *,
    patch_all_attention_functions: bool = True,
) -> None:
    """Patch HF FA2 entrypoints for Ulysses all-to-all + local FA2.

    Always patches ``modeling_flash_attention_utils._flash_attention_forward``.
    When ``patch_all_attention_functions`` is True (default), also replaces
    ``ALL_ATTENTION_FUNCTIONS["flash_attention_2"]`` — required on transformers
    builds that route attention through that registry.
    """
    import transformers.modeling_flash_attention_utils
    from flash_attn import flash_attn_varlen_func

    cp_size = dist.get_world_size(group=process_group)
    # Stock FA2 for rollouts / any forward that has not published CP params.
    # CP is train-only; HF generate keeps full sequences on every rank.
    _original_flash_attention_forward = (
        transformers.modeling_flash_attention_utils._flash_attention_forward
    )

    def _ulysses_flash_attention_forward(
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask,
        query_length: int,
        is_causal: bool,
        dropout: float = 0.0,
        position_ids=None,
        softmax_scale=None,
        sliding_window=None,
        use_top_left_mask: bool = False,
        softcap=None,
        deterministic=None,
        **kwargs,
    ):
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
                **kwargs,
            )

        if not is_causal:
            raise AssertionError("ulysses CP only supports causal attention")
        if softcap is not None:
            raise AssertionError("ulysses CP path does not support softcap")
        if query_states.size(0) != 1:
            raise AssertionError("varlen data should be processed with batch=1")

        cu_seqlens = ULYSSES_PARAMS["cu_seqlens"]
        max_seqlen = ULYSSES_PARAMS["max_seqlen"]

        window_size = (-1, -1)
        if sliding_window is not None and key_states.shape[1] > sliding_window:
            window_size = (sliding_window, sliding_window)

        out = ulysses_flash_attn_varlen_func(
            flash_attn_varlen_func,
            query_states.squeeze(0),
            key_states.squeeze(0),
            value_states.squeeze(0),
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
        return out.unsqueeze(0)

    transformers.modeling_flash_attention_utils._flash_attention_forward = (
        _ulysses_flash_attention_forward
    )

    if not patch_all_attention_functions:
        return

    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    except ImportError:  # pragma: no cover
        ALL_ATTENTION_FUNCTIONS = None

    if ALL_ATTENTION_FUNCTIONS is None:
        return

    _original_all_attn = ALL_ATTENTION_FUNCTIONS.get("flash_attention_2")

    def _ulysses_flash_attention_forward_v2(
        module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask=None,
        dropout: float = 0.0,
        scaling=None,
        sliding_window=None,
        softcap=None,
        **kw,
    ):
        # Rollouts / generate: keep the stock registry path (full sequence).
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
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        kw.pop("is_causal", None)
        attn_out = _ulysses_flash_attention_forward(
            query,
            key,
            value,
            attention_mask,
            query_length=seq_len,
            is_causal=module.is_causal,
            dropout=dropout,
            softmax_scale=scaling,
            sliding_window=sliding_window,
            softcap=softcap,
        )
        return attn_out, None

    ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = _ulysses_flash_attention_forward_v2

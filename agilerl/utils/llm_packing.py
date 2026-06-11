"""Padding-free sequence packing helpers for LLM RL training.

A right-padded ``(B, T)`` batch wastes the model forward on pad tokens — the
fraction wasted grows with the spread of completion lengths, which for RL is
large. Packing flattens the real tokens of every sequence into a single
``(1, N)`` row and hands the model the metadata it needs to keep sequences from
attending across one another (per-sequence ``position_ids`` that restart at 0,
plus ``cu_seqlens`` for a FlashAttention-varlen forward). The result is the same
loss with no pad-token compute.

This module is deliberately backend-agnostic and pure-tensor: it produces the
packed inputs and maps the packed result back onto the padded frame, so the
rest of the loss path is unchanged. Three unpackers cover the consumers:
:func:`unpack_logprobs` scatters packed per-token logprobs onto the
``(B, T-1)`` next-token frame (the standard fused-logprob path),
:func:`unpack_values` scatters packed per-token critic values onto that same
``(B, T-1)`` frame (the PPO value-head path), and
:func:`unpack_hidden_states` scatters the raw packed last-hidden-states onto
the ``(B, T, H)`` frame (so a fused loss kernel such as Liger consumes them
exactly as it would a padded forward's output).

Both unpackers are the same scatter: for each original sequence *b* of real
length ``L_b``, its real tokens occupy the contiguous flat span
``cu[b] .. cu[b] + L_b`` and land left-aligned at row *b* of the output,
leaving pad rows zero (the action mask discards them downstream). They differ
only in how many tokens per segment are scattered and the output row width:
logprobs map the ``L_b - 1`` within-sequence next-token predictions onto the
``(B, T-1)`` frame, dropping the cross-segment boundary prediction (segment
*b*'s last token "predicting" segment *b+1*'s first token) so no label leaks
between sequences; hidden states map all ``L_b`` real tokens onto the
``(B, T, H)`` frame (no boundary token to drop — the next-token shift is
applied by the kernel on the padded frame). Both are differentiable; gradients
flow back through ``index_select`` / ``index_put``.

Crucially, the packed forward passes the model only the per-sequence
``position_ids`` (with ``attention_mask=None``); it does **not** build an
attention mask itself. Transformers' own mask creation detects the packed
format from those ``position_ids``
(:func:`transformers.masking_utils.find_packed_sequence_indices`) and
AND-composes a block-diagonal ("same segment") constraint onto each layer's
*native* mask. So sliding-window layers stay windowed and full layers stay full
causal, each additionally confined to its own segment — packing is therefore
correct for sliding-window models (e.g. gemma), not only full-attention ones.

.. warning::

   Packing only avoids cross-sequence attention on a backend where that composed
   mask stays sparse: FlashAttention-2 varlen (``position_ids`` → ``cu_seqlens``
   + per-layer ``window_size``, no mask) or FlexAttention (a sparse
   block-diagonal ``BlockMask``, windowed on sliding layers). Dense backends
   (SDPA/eager) would materialize a dense ``O(N^2)`` mask — correct but defeating
   the memory win — so packing is not enabled there and falls back to padding.
   Callers must gate packing on a supported backend; see
   :meth:`LLMAlgorithm._packing_mode`.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F


class PackedBatch(NamedTuple):
    """Flattened batch plus the metadata needed to unpack it again.

    :param input_ids: ``(1, N)`` real tokens of every sequence, concatenated in
        row-major order (``N = sum(seq_lengths)``).
    :param position_ids: ``(1, N)`` per-sequence positions, each restarting at 0.
    :param cu_seqlens: ``(B + 1,)`` int32 cumulative sequence lengths
        (``[0, L_0, L_0 + L_1, ...]``) for FlashAttention-varlen.
    :param seq_lengths: ``(B,)`` real (non-pad) length of each original row.
    :param max_seqlen: Longest single sequence length (varlen kernel arg).
    :param batch_size: Number of original sequences ``B``.
    :param padded_seq_len: Original padded length ``T`` (so unpacking can
        rebuild the ``(B, T - 1)`` frame).
    """

    input_ids: torch.Tensor
    position_ids: torch.Tensor
    cu_seqlens: torch.Tensor
    seq_lengths: torch.Tensor
    max_seqlen: int
    batch_size: int
    padded_seq_len: int


def pack_padded_batch(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> PackedBatch:
    """Flatten a padded ``(B, T)`` batch into a single padding-free row.

    Real tokens are selected by ``attention_mask`` (robust to either padding
    side), so the i-th sequence contributes its ``attention_mask[i].sum()``
    unmasked tokens in their original order. Per-sequence ``position_ids``
    restart at 0, matching the ``cumsum(mask) - 1`` convention the dense path
    uses for real tokens.

    :param input_ids: ``(B, T)`` token ids.
    :type input_ids: torch.Tensor
    :param attention_mask: ``(B, T)`` mask; non-zero marks real tokens.
    :type attention_mask: torch.Tensor
    :return: A :class:`PackedBatch`.
    :rtype: PackedBatch
    """
    if input_ids.dim() != 2:
        msg = (
            f"pack_padded_batch expects (B, T) input_ids, got {tuple(input_ids.shape)}"
        )
        raise ValueError(msg)
    mask = attention_mask.bool()
    batch_size, padded_seq_len = input_ids.shape
    seq_lengths = mask.sum(dim=1).to(torch.long)  # (B,)

    flat_ids = input_ids[mask].unsqueeze(0)  # (1, N), row-major over real tokens

    cu = F.pad(seq_lengths.cumsum(0), (1, 0))  # (B+1,) int64 [0, L_0, L_0+L_1, ...]
    # position_ids: global index minus the start offset of each token's segment,
    # so each sequence's positions restart at 0 (vectorized arange-per-segment).
    n_tokens = int(cu[-1])
    starts = cu[:-1].repeat_interleave(seq_lengths)  # (N,) segment start per token
    position_ids = (torch.arange(n_tokens, device=input_ids.device) - starts).unsqueeze(
        0
    )  # (1, N)

    cu_seqlens = cu.to(torch.int32)  # (B+1,)
    max_seqlen = int(seq_lengths.max().item()) if seq_lengths.numel() else 0

    return PackedBatch(
        input_ids=flat_ids,
        position_ids=position_ids,
        cu_seqlens=cu_seqlens,
        seq_lengths=seq_lengths,
        max_seqlen=max_seqlen,
        batch_size=batch_size,
        padded_seq_len=padded_seq_len,
    )


def packed_segment_ids(
    packed: PackedBatch,
    device: torch.device | str,
) -> torch.Tensor:
    """``(N,)`` segment id per flat token (which original sequence it belongs to)."""
    return torch.repeat_interleave(
        torch.arange(packed.batch_size, device=device),
        packed.seq_lengths.to(device),
    )


def _scatter_packed(
    flat: torch.Tensor,
    packed: PackedBatch,
    tokens_per_seg: torch.Tensor,
    row_stride: int,
    trailing_dim: int | None,
) -> torch.Tensor:
    """Scatter the leading ``tokens_per_seg[b]`` rows of each packed segment.

    Shared core of both unpackers. For each original sequence *b*, the flat span
    starting at ``cu[b]`` and spanning ``tokens_per_seg[b]`` rows is copied to
    output rows ``b * row_stride + 0 .. tokens_per_seg[b] - 1`` (left-aligned),
    leaving all other rows zero. Segments with no contributing rows are skipped.
    Built with ``index_select`` / ``index_put`` so gradients flow back to ``flat``.

    :param flat: ``(N,)`` or ``(N, H)`` packed rows to scatter.
    :type flat: torch.Tensor
    :param packed: The :class:`PackedBatch` whose ``cu_seqlens`` give segment
        offsets.
    :type packed: PackedBatch
    :param tokens_per_seg: ``(B,)`` number of leading rows to take from each
        segment (e.g. ``L_b - 1`` for logprobs, ``L_b`` for hidden states).
    :type tokens_per_seg: torch.Tensor
    :param row_stride: Output row stride per segment (``T - 1`` or ``T``).
    :type row_stride: int
    :param trailing_dim: ``H`` for a ``(N, H)`` payload, else ``None`` for ``(N,)``.
    :type trailing_dim: int | None
    :return: Flat ``(B * row_stride,)`` (or ``(B * row_stride, H)``) output, to be
        viewed into its final padded shape by the caller.
    :rtype: torch.Tensor
    """
    device = flat.device
    cu = packed.cu_seqlens
    n_rows = packed.batch_size * row_stride
    out = (
        flat.new_zeros(n_rows)
        if trailing_dim is None
        else flat.new_zeros(n_rows, trailing_dim)
    )

    src_chunks: list[torch.Tensor] = []
    dst_chunks: list[torch.Tensor] = []
    for b in range(packed.batch_size):
        n = int(tokens_per_seg[b])
        if n <= 0:
            continue
        offset = int(cu[b])
        src_chunks.append(torch.arange(offset, offset + n, device=device))
        dst_chunks.append(b * row_stride + torch.arange(n, device=device))

    if src_chunks:
        src = torch.cat(src_chunks)
        dst = torch.cat(dst_chunks)
        out = out.index_put((dst,), flat.index_select(0, src))

    return out


def unpack_logprobs(
    packed_logprobs: torch.Tensor,
    packed: PackedBatch,
) -> torch.Tensor:
    """Scatter packed per-token logprobs back onto the padded ``(B, T-1)`` frame.

    ``packed_logprobs`` is the ``(N - 1,)`` fused next-token sequence over the
    flattened row. For each sequence *b* of real length ``L_b``, its ``L_b - 1``
    within-sequence predictions land at columns ``0 .. L_b - 2`` of row *b*; the
    cross-segment boundary prediction is dropped and pad columns stay zero. See
    the module docstring for the shared scatter semantics.

    :param packed_logprobs: ``(N - 1,)`` packed per-token logprobs.
    :type packed_logprobs: torch.Tensor
    :param packed: The :class:`PackedBatch` the logprobs were computed from.
    :type packed: PackedBatch
    :return: ``(B, T-1)`` per-token logprobs aligned to the dense frame.
    :rtype: torch.Tensor
    """
    flat = packed_logprobs.reshape(-1)
    tm1 = packed.padded_seq_len - 1
    out = _scatter_packed(
        flat,
        packed,
        tokens_per_seg=packed.seq_lengths - 1,  # L_b - 1 (skipped when <= 0)
        row_stride=tm1,
        trailing_dim=None,
    )
    return out.view(packed.batch_size, tm1)


def unpack_values(
    packed_values: torch.Tensor,
    packed: PackedBatch,
) -> torch.Tensor:
    """Scatter packed per-token critic values back onto the ``(B, T-1)`` frame.

    The value-head analogue of :func:`unpack_logprobs`. A critic value is a
    per-position scalar (not a next-token prediction), so unlike logprobs there
    is no cross-segment boundary to drop — every real token carries its own
    value. Each sequence *b* keeps its first ``min(L_b, T-1)`` values, landing
    left-aligned at columns ``0 .. min(L_b, T-1) - 1`` of row *b*; the clamp
    mirrors the padded path's ``value[:, :-1]`` (which never holds more than
    ``T - 1`` columns and drops the final position's value when a row is
    unpadded). Pad columns stay zero — the action mask discards them downstream,
    so the packed (zero) and padded (computed-then-masked) frames agree exactly
    on every action position. See the module docstring for the shared scatter
    semantics.

    :param packed_values: ``(N,)`` (or ``(1, N)`` / ``(N, 1)``) packed per-token
        critic values.
    :type packed_values: torch.Tensor
    :param packed: The :class:`PackedBatch` the values were computed from.
    :type packed: PackedBatch
    :return: ``(B, T-1)`` per-token values aligned to the dense frame.
    :rtype: torch.Tensor
    """
    flat = packed_values.reshape(-1)
    tm1 = packed.padded_seq_len - 1
    out = _scatter_packed(
        flat,
        packed,
        tokens_per_seg=packed.seq_lengths.clamp(max=tm1),  # L_b clamped to T-1
        row_stride=tm1,
        trailing_dim=None,
    )
    return out.view(packed.batch_size, tm1)


def unpack_hidden_states(
    packed_hidden: torch.Tensor,
    packed: PackedBatch,
) -> torch.Tensor:
    """Scatter packed ``(1, N, H)`` hidden states back onto the ``(B, T, H)`` frame.

    The hidden-state analogue of :func:`unpack_logprobs`: it maps all ``L_b``
    real last-hidden-states of each sequence onto the full padded frame so a
    fused loss kernel (e.g. Liger) consumes them exactly as it would a padded
    forward's output. Unlike logprob unpacking there is no boundary token to
    drop. See the module docstring for the shared scatter semantics.

    :param packed_hidden: ``(1, N, H)`` (or ``(N, H)``) packed last-hidden-states.
    :type packed_hidden: torch.Tensor
    :param packed: The :class:`PackedBatch` the hidden states were computed from.
    :type packed: PackedBatch
    :return: ``(B, T, H)`` hidden states aligned to the dense frame.
    :rtype: torch.Tensor
    """
    hidden_dim = packed_hidden.shape[-1]
    flat = packed_hidden.reshape(-1, hidden_dim)  # (N, H)
    t = packed.padded_seq_len
    out = _scatter_packed(
        flat,
        packed,
        tokens_per_seg=packed.seq_lengths,  # L_b (skipped when == 0)
        row_stride=t,
        trailing_dim=hidden_dim,
    )
    return out.view(packed.batch_size, t, hidden_dim)

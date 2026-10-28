# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Memory-bounded fused-linear log-prob computation.

Computes per-token target logprobs straight from last-hidden-states with the
``(B, T, V)`` logits slab never materialized: the lm_head matmul + log-softmax
reduction runs over bounded ``(chunk_rows, V)`` tiles in both the forward and
the (gradient-checkpointed) backward. Used by ``LLMAlgorithm`` for both the
no-grad old/reference passes and the gradient forward.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

import torch

from agilerl.distributed import materialize_dtensors


def fp32_lm_head_operands(
    head_w: torch.Tensor,
    head_b: torch.Tensor | None,
    cast_to_fp32: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Upcast the lm_head for one chunk loop, hoisted out of the loop body.

    A ``(V, H)`` fp32 copy is gigabytes at production vocab sizes, so a loop
    takes at most one, shared by every chunk. Call inside an FSDP/DTensor
    gather (or with already-dense weights), or the copy is taken from a shard.

    :param head_w: ``(V, H)``.
    :type head_w: torch.Tensor
    :param head_b: ``(V,)`` or ``None``.
    :type head_b: torch.Tensor | None
    :param cast_to_fp32: whether the loop runs its matmul in fp32.
    :type cast_to_fp32: bool
    :return: The head operands to pass to every chunk.
    :rtype: tuple[torch.Tensor, torch.Tensor | None]
    """
    if not cast_to_fp32 or head_w.dtype == torch.float32:
        return head_w, head_b
    bias = head_b.float() if head_b is not None else None
    return head_w.float(), bias


def support_scores(
    scores: torch.Tensor,
    support_ids: torch.Tensor,
    target_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Target score and log-sum-exp of ``scores`` over each row's sampling support.

    A support row lists the token ids the sampler could draw, padded with
    ``-1``; a row whose first id is ``-1`` has no recorded support. Its
    normalizer is the target's own score repeated, so it stays finite and its
    gradient is zero once the caller selects the full-vocab normalizer there.
    The target and the support share one gather, so the backward builds one
    ``(..., V)`` scatter buffer.

    :param scores: ``(..., V)`` logits or log-probs.
    :type scores: torch.Tensor
    :param support_ids: ``(..., K)`` support token ids, ``-1`` padded.
    :type support_ids: torch.Tensor
    :param target_ids: ``(...)`` sampled token ids.
    :type target_ids: torch.Tensor
    :return: ``(selected, normalizer, restricted)``, each ``(...)``: the
        target's score, the log-sum-exp over the support, and whether the row
        has a support.
    :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    """
    present = support_ids >= 0
    restricted = present[..., 0]
    targets = target_ids.unsqueeze(-1)
    ids = torch.where(present, support_ids.long(), targets)
    values = scores.gather(dim=-1, index=torch.cat([targets, ids], dim=-1))
    support = values[..., 1:].masked_fill(
        ~present & restricted.unsqueeze(-1), float("-inf")
    )
    return values[..., 0], torch.logsumexp(support, dim=-1), restricted


def _fused_logprob_chunk(
    h_chunk: torch.Tensor,
    head_w: torch.Tensor,
    head_b: torch.Tensor | None,
    target_chunk: torch.Tensor,
    temperature: float,
    cast_to_fp32: bool,
    softcap: float | None,
    logit_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-token logprobs for one flat ``(chunk_rows, H)`` slab of hidden states.

    Functional (no in-place ops) so it is safe under both ``torch.compile`` and
    the autograd recompute in :class:`FusedLinearLogProbsFunction`. The
    max-shift used elsewhere for stability is folded into ``logsumexp`` (which
    is already numerically stable), keeping the body fusion-friendly.

    :param h_chunk: ``(chunk_rows, H)`` hidden states.
    :type h_chunk: torch.Tensor
    :param head_w: ``(V, H)``.
    :type head_w: torch.Tensor
    :param head_b: ``(V,)`` or ``None``.
    :type head_b: torch.Tensor | None
    :param target_chunk: ``(chunk_rows,)`` target token ids.
    :type target_chunk: torch.Tensor
    :param temperature: logits divided by this before log_softmax (skip at 1.0).
    :type temperature: float
    :param cast_to_fp32: upcast ``h_chunk`` so the matmul and the reduction run
        in fp32; the head is expected to arrive fp32 already
        (:func:`fp32_lm_head_operands`).
    :type cast_to_fp32: bool
    :param softcap: squash logits to ``softcap * tanh(logits / softcap)`` before
        the temperature, as the model's own head does; ``None`` skips it.
    :type softcap: float | None
    :param logit_bias: ``(V,)`` fp32 bias added after the softcap and before the
        temperature, as vLLM applies its ``logit_bias``; ``None`` adds none.
    :type logit_bias: torch.Tensor | None
    :return: ``(chunk_rows,)`` per-token logprobs.
    :rtype: torch.Tensor
    """
    if cast_to_fp32:
        h_chunk = h_chunk.float()
    if h_chunk.dtype != head_w.dtype:
        # Chunk-tiling callers upcast the head once ahead of their loop, so this
        # only fires for a direct call or an fp16 checkpoint under bf16 autocast
        # (fp32 hidden, fp16 head). It copies the whole ``(V, H)`` head.
        compute_dtype = torch.promote_types(h_chunk.dtype, head_w.dtype)
        h_chunk = h_chunk.to(compute_dtype)
        head_w = head_w.to(compute_dtype)
    logits = h_chunk @ head_w.t()
    if head_b is not None:
        logits = logits + head_b
    if softcap is not None:
        logits = softcap * torch.tanh(logits / softcap)
    if logit_bias is not None:
        logits = logits + logit_bias.to(logits.dtype)
    if temperature != 1.0:
        logits = logits / temperature
    selected = logits.gather(dim=-1, index=target_chunk.unsqueeze(-1)).squeeze(-1)
    log_z = torch.logsumexp(logits, dim=-1)
    return selected - log_z


def support_logprob_chunk(
    h_chunk: torch.Tensor,
    head_w: torch.Tensor,
    head_b: torch.Tensor | None,
    target_chunk: torch.Tensor,
    temperature: float,
    cast_to_fp32: bool,
    softcap: float | None,
    support_chunk: torch.Tensor,
    logit_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-token logprobs over each row's sampling support, from the support's head rows alone.

    A logprob normalized over a support depends only on the support's logits,
    so the head is gathered to those ``K`` rows per token rather than
    multiplied in full. Every row must have a support; a target outside its
    support has probability zero.

    :param h_chunk: ``(n, H)`` hidden states.
    :type h_chunk: torch.Tensor
    :param head_w: ``(V, H)`` in the model's dtype; gathered rows are upcast
        when ``cast_to_fp32`` is set.
    :type head_w: torch.Tensor
    :param head_b: ``(V,)`` or ``None``.
    :type head_b: torch.Tensor | None
    :param target_chunk: ``(n,)`` target token ids.
    :type target_chunk: torch.Tensor
    :param temperature: logits divided by this before normalizing (skip at 1.0).
    :type temperature: float
    :param cast_to_fp32: run the dot products and the reduction in fp32.
    :type cast_to_fp32: bool
    :param softcap: final-logit softcap of the model's head, or ``None``.
    :type softcap: float | None
    :param support_chunk: ``(n, K)`` support token ids, ``-1`` padded, with a
        support on every row.
    :type support_chunk: torch.Tensor
    :param logit_bias: ``(V,)`` fp32 bias added after the softcap, or ``None``.
    :type logit_bias: torch.Tensor | None
    :return: ``(n,)`` per-token logprobs.
    :rtype: torch.Tensor
    """
    present = support_chunk >= 0
    ids = support_chunk.long().clamp(min=0)
    matches = present & (ids == target_chunk.unsqueeze(-1))
    rows = head_w[ids]
    if cast_to_fp32:
        h_chunk = h_chunk.float()
        rows = rows.float()
    compute_dtype = torch.promote_types(h_chunk.dtype, rows.dtype)
    logits = torch.bmm(
        rows.to(compute_dtype), h_chunk.to(compute_dtype).unsqueeze(-1)
    ).squeeze(-1)
    if head_b is not None:
        logits = logits + head_b[ids].to(logits.dtype)
    if softcap is not None:
        logits = softcap * torch.tanh(logits / softcap)
    if logit_bias is not None:
        logits = logits + logit_bias[ids].to(logits.dtype)
    if temperature != 1.0:
        logits = logits / temperature
    # The target's own support column, so a one-id support cancels exactly.
    selected = logits.gather(-1, matches.int().argmax(-1, keepdim=True)).squeeze(-1)
    log_z = torch.logsumexp(logits.masked_fill(~present, float("-inf")), dim=-1)
    return torch.where(
        matches.any(-1), selected - log_z, torch.full_like(log_z, float("-inf"))
    )


def support_chunk_rows(chunk_rows: int, head_w: torch.Tensor, width: int) -> int:
    """Rows per support chunk whose gathered head fits the full-vocab chunk's logits.

    :param chunk_rows: rows of a full-vocab ``(chunk_rows, V)`` chunk.
    :type chunk_rows: int
    :param head_w: ``(V, H)``.
    :type head_w: torch.Tensor
    :param width: support width ``K``.
    :type width: int
    :return: Rows whose ``(rows, K, H)`` gather has at most ``chunk_rows * V``
        elements, and at least ``chunk_rows``.
    :rtype: int
    """
    vocab, hidden = head_w.shape
    return max(chunk_rows, chunk_rows * vocab // (width * hidden))


def split_support_rows(
    flat_support: torch.Tensor | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Split flat rows into those scored over a support and those scored over the vocabulary.

    :param flat_support: ``(N, K)`` supports, or ``None``.
    :type flat_support: torch.Tensor | None
    :return: ``(support_rows, vocab_rows)`` long indices, or ``(None, None)``
        without supports, when every row is scored over the vocabulary.
    :rtype: tuple[torch.Tensor | None, torch.Tensor | None]
    """
    if flat_support is None:
        return None, None
    restricted = flat_support[:, 0] >= 0
    return restricted.nonzero().squeeze(-1), (~restricted).nonzero().squeeze(-1)


def row_chunks(
    rows: torch.Tensor | None, n_rows: int, step: int
) -> list[slice | torch.Tensor]:
    """Chunks of ``step`` flat rows: slices of every row, or slices of an index.

    :param rows: Long row indices, or ``None`` for all ``n_rows`` rows.
    :type rows: torch.Tensor | None
    :param n_rows: Total flat rows.
    :type n_rows: int
    :param step: Rows per chunk.
    :type step: int
    :return: Indexers for one chunk each.
    :rtype: list[slice | torch.Tensor]
    """
    if rows is None:
        return [slice(s, min(s + step, n_rows)) for s in range(0, n_rows, step)]
    return list(rows.split(step)) if rows.numel() else []


# torch.compile cache for ``_fused_logprob_chunk`` (CUDA-only; any failure
# latches ``disabled`` and the eager path is used for the rest of the process).
_FUSED_LOGPROB_COMPILE_STATE: dict[str, Any] = {"fn": None, "disabled": False}


def _fused_logprob_chunk_dispatch(
    device: torch.device,
    h_chunk: torch.Tensor,
    head_w: torch.Tensor,
    head_b: torch.Tensor | None,
    target_chunk: torch.Tensor,
    temperature: float,
    cast_to_fp32: bool,
    softcap: float | None,
    logit_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run :func:`_fused_logprob_chunk`, compiled on CUDA with eager fallback."""
    state = _FUSED_LOGPROB_COMPILE_STATE
    args = (
        h_chunk,
        head_w,
        head_b,
        target_chunk,
        temperature,
        cast_to_fp32,
        softcap,
        logit_bias,
    )
    if device.type != "cuda" or state["disabled"]:
        return _fused_logprob_chunk(*args)
    if state["fn"] is None:
        state["fn"] = torch.compile(_fused_logprob_chunk, dynamic=True)
    try:
        return state["fn"](*args)
    except Exception:
        # Triton/backend failure — drop to eager for the rest of the process.
        state["disabled"] = True
        return _fused_logprob_chunk(*args)


def scored_position_index(score_mask: torch.Tensor) -> torch.Tensor:
    """Column index of each row's scored positions, left-aligned in order.

    Every row is padded to the longest row's count (at least one column) with
    its unscored positions, so the index never repeats a column within a row.

    :param score_mask: ``(B, T)`` mask of the positions to score.
    :type score_mask: torch.Tensor
    :return: ``(B, L)`` long column indices into ``score_mask``.
    :rtype: torch.Tensor
    """
    keep = score_mask.to(torch.bool)
    width = max(int(keep.sum(dim=-1).max().item()), 1)
    order = torch.argsort((~keep).to(torch.int32), dim=-1, stable=True)
    return order[:, :width]


def fused_linear_logprobs_chunked(
    hidden: torch.Tensor,
    head_w: torch.Tensor,
    head_b: torch.Tensor | None,
    target_ids: torch.Tensor,
    temperature: float = 1.0,
    cast_to_fp32: bool = True,
    *,
    chunk_rows: int,
    softcap: float | None = None,
    support_ids: torch.Tensor | None = None,
    logit_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-token target logprobs from hidden states, vocab-chunked, no grad.

    Tiles flat over ``(B*T)`` so the transient logits workspace is bounded to
    ``(chunk_rows, V)`` per iteration; the full ``(B, T, V)`` slab is never
    built. Shared forward implementation backing both the no-grad static
    method :meth:`LLMAlgorithm._logprobs_from_hidden_fused` and the
    autograd-aware :class:`FusedLinearLogProbsFunction`. Must be called under
    ``no_grad``/``inference_mode`` (it writes results in place).

    :param hidden: ``(B, T, H)`` last-hidden-state.
    :type hidden: torch.Tensor
    :param head_w: ``(V, H)``.
    :type head_w: torch.Tensor
    :param head_b: ``(V,)`` or ``None``.
    :type head_b: torch.Tensor | None
    :param target_ids: ``(B, T)`` sampled token ids (caller does the shift).
    :type target_ids: torch.Tensor
    :param temperature: logits divided by this before log_softmax (skipped at 1.0).
    :type temperature: float, optional
    :param cast_to_fp32: run the per-chunk matmul and reduction in fp32, then
        cast back.
    :type cast_to_fp32: bool, optional
    :param chunk_rows: rows of the flattened ``(B*T)`` workspace per iteration.
    :type chunk_rows: int
    :param softcap: final-logit softcap of the model's head, or ``None``.
    :type softcap: float | None, optional
    :param support_ids: ``(B, T, K)`` sampling support of each position,
        ``-1`` padded, or ``None`` to normalize over the full vocabulary.
    :type support_ids: torch.Tensor | None
    :param logit_bias: ``(V,)`` fp32 bias added after the softcap and before
        the temperature, or ``None``.
    :type logit_bias: torch.Tensor | None
    :return: ``(B, T)`` per-token logprobs in ``hidden.dtype``.
    :rtype: torch.Tensor
    """
    orig_dtype = hidden.dtype
    B, T, H = hidden.shape
    flat_h = hidden.reshape(-1, H)
    flat_targets = target_ids.reshape(-1).to(torch.long)
    flat_support = flat_support_ids(support_ids, hidden.device)
    support_rows, vocab_rows = split_support_rows(flat_support)
    N = flat_h.shape[0]
    out = torch.empty(N, dtype=orig_dtype, device=hidden.device)

    with materialize_dtensors(head_w, head_b) as (full_weight, full_bias):
        if flat_support is not None and support_rows is not None:
            step = support_chunk_rows(chunk_rows, full_weight, flat_support.shape[1])
            for rows in row_chunks(support_rows, N, step):
                out[rows] = support_logprob_chunk(
                    flat_h[rows],
                    full_weight,
                    full_bias,
                    flat_targets[rows],
                    temperature,
                    cast_to_fp32,
                    softcap,
                    flat_support[rows],
                    logit_bias,
                ).to(orig_dtype)
        vocab_chunks = row_chunks(vocab_rows, N, chunk_rows)
        if vocab_chunks:
            compute_w, compute_b = fp32_lm_head_operands(
                full_weight, full_bias, cast_to_fp32
            )
            for rows in vocab_chunks:
                out[rows] = _fused_logprob_chunk_dispatch(
                    hidden.device,
                    flat_h[rows],
                    compute_w,
                    compute_b,
                    flat_targets[rows],
                    temperature,
                    cast_to_fp32,
                    softcap,
                    logit_bias,
                ).to(orig_dtype)

    return out.reshape(B, T)


def flat_support_ids(
    support_ids: torch.Tensor | None, device: torch.device
) -> torch.Tensor | None:
    """Flatten a ``(B, T, K)`` support frame to ``(B*T, K)`` on ``device``.

    :param support_ids: ``(B, T, K)`` sampling supports, or ``None``.
    :type support_ids: torch.Tensor | None
    :param device: Device the chunk loop runs on.
    :type device: torch.device
    :return: ``(B*T, K)`` supports, or ``None``.
    :rtype: torch.Tensor | None
    """
    if support_ids is None:
        return None
    return support_ids.reshape(-1, support_ids.shape[-1]).to(device)


def chunk_input_grads(
    logprob_fn: Callable[..., torch.Tensor],
    h_chunk: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    grad_chunk: torch.Tensor,
    needs: tuple[bool, bool, bool],
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Recompute one chunk's logprobs with grad and return the input gradients it needs.

    :param logprob_fn: Maps ``(hidden, weight, bias)`` to the chunk's logprobs.
    :type logprob_fn: Callable[..., torch.Tensor]
    :param h_chunk: ``(n, H)`` hidden states.
    :type h_chunk: torch.Tensor
    :param weight: Head weight the recompute reads.
    :type weight: torch.Tensor
    :param bias: Head bias, or ``None``.
    :type bias: torch.Tensor | None
    :param grad_chunk: ``(n,)`` upstream gradient.
    :type grad_chunk: torch.Tensor
    :param needs: Whether hidden, weight and bias need gradients.
    :type needs: tuple[bool, bool, bool]
    :return: Gradients of hidden, weight and bias, ``None`` where not needed.
    :rtype: tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]
    """
    leaves = (h_chunk, weight, bias)
    wanted = [
        need and leaf is not None for need, leaf in zip(needs, leaves, strict=True)
    ]
    if not any(wanted):
        return None, None, None
    inputs = [
        leaf.detach().requires_grad_(need) if leaf is not None else None
        for need, leaf in zip(wanted, leaves, strict=True)
    ]
    with torch.enable_grad():
        logps = logprob_fn(*inputs)
    grads = iter(
        torch.autograd.grad(
            logps,
            [
                leaf
                for leaf, need in zip(inputs, wanted, strict=True)
                if need and leaf is not None
            ],
            grad_outputs=grad_chunk.to(logps.dtype),
        )
    )
    hidden_grad, weight_grad, bias_grad = (
        next(grads) if need else None for need in wanted
    )
    return hidden_grad, weight_grad, bias_grad


class FusedLinearLogProbsFunction(torch.autograd.Function):
    """Gradient-checkpointed per-token logprobs over a chunked lm_head matmul.

    The forward computes per-token logprobs chunk-by-chunk under ``no_grad``
    (peak logits workspace bounded to ``(chunk_rows, V)``). The backward
    re-runs the same chunked matmul one chunk at a time with grad enabled,
    accumulates gradients into preallocated buffers, and frees each chunk's
    logits before the next. Peak logits memory is therefore ``O(chunk_rows *
    V)`` in *both* directions — the ``(B, T, V)`` slab is never materialized.

    Only ``(B, T, H)`` hidden states (saved for the recompute) are held across
    forward/backward, which the surrounding policy graph keeps alive anyway.

    Numerically the forward matches :func:`fused_linear_logprobs_chunked`; the
    backward yields the exact ``log_softmax`` gradient ``onehot(target) -
    softmax`` (the max-shift used for stability cancels in the derivative),
    with the softmax taken over the sampling support where one is given.
    """

    @staticmethod
    def forward(
        ctx: Any,  # noqa: ANN401 -- torch autograd context carries arbitrary attrs
        hidden: torch.Tensor,
        head_w: torch.Tensor,
        head_b: torch.Tensor | None,
        target_ids: torch.Tensor,
        temperature: float,
        cast_to_fp32: bool,
        chunk_rows: int,
        softcap: float | None,
        support_ids: torch.Tensor | None = None,
        logit_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            logps = fused_linear_logprobs_chunked(
                hidden,
                head_w,
                head_b,
                target_ids,
                temperature=temperature,
                cast_to_fp32=cast_to_fp32,
                chunk_rows=chunk_rows,
                softcap=softcap,
                support_ids=support_ids,
                logit_bias=logit_bias,
            )
        ctx.save_for_backward(
            hidden, head_w, head_b, target_ids, support_ids, logit_bias
        )
        ctx.temperature = temperature
        ctx.cast_to_fp32 = cast_to_fp32
        ctx.chunk_rows = chunk_rows
        ctx.softcap = softcap
        ctx.needs_hidden_grad = hidden.requires_grad
        ctx.needs_weight_grad = head_w.requires_grad
        ctx.needs_bias_grad = head_b is not None and head_b.requires_grad
        return logps

    @staticmethod
    def backward(
        ctx: Any,  # noqa: ANN401 -- torch autograd context carries arbitrary attrs
        *grad_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor | None, ...]:
        (grad_output,) = grad_outputs
        hidden, head_w, head_b, target_ids, support_ids, logit_bias = ctx.saved_tensors
        B, T, H = hidden.shape
        flat_h = hidden.reshape(-1, H)
        flat_targets = target_ids.reshape(-1).to(torch.long)
        flat_support = flat_support_ids(support_ids, hidden.device)
        support_rows, vocab_rows = split_support_rows(flat_support)
        flat_grad = grad_output.reshape(-1)
        N = flat_h.shape[0]
        options = {
            "temperature": ctx.temperature,
            "cast_to_fp32": ctx.cast_to_fp32,
            "softcap": ctx.softcap,
            "logit_bias": logit_bias,
        }
        needs = (ctx.needs_hidden_grad, ctx.needs_weight_grad, ctx.needs_bias_grad)

        with materialize_dtensors(head_w, head_b) as (
            full_weight,
            full_bias,
        ):
            grad_hidden = torch.zeros_like(flat_h) if ctx.needs_hidden_grad else None
            grad_weight = (
                torch.zeros_like(full_weight) if ctx.needs_weight_grad else None
            )
            grad_bias = (
                torch.zeros_like(full_bias)
                if ctx.needs_bias_grad and full_bias is not None
                else None
            )
            chunks: list[tuple[slice | torch.Tensor, Callable, torch.Tensor, Any]] = []
            if flat_support is not None and support_rows is not None:
                step = support_chunk_rows(
                    ctx.chunk_rows, full_weight, flat_support.shape[1]
                )
                chunks += [
                    (
                        rows,
                        partial(
                            support_logprob_chunk,
                            target_chunk=flat_targets[rows],
                            support_chunk=flat_support[rows],
                            **options,
                        ),
                        full_weight,
                        full_bias,
                    )
                    for rows in row_chunks(support_rows, N, step)
                ]
            vocab_chunks = row_chunks(vocab_rows, N, ctx.chunk_rows)
            if vocab_chunks:
                # Grad buffers above stay in the head's own dtype; the
                # full-vocab recompute runs against one hoisted fp32 copy of it.
                compute_w, compute_b = fp32_lm_head_operands(
                    full_weight, full_bias, ctx.cast_to_fp32
                )
                chunks += [
                    (
                        rows,
                        partial(
                            _fused_logprob_chunk_dispatch,
                            hidden.device,
                            target_chunk=flat_targets[rows],
                            **options,
                        ),
                        compute_w,
                        compute_b,
                    )
                    for rows in vocab_chunks
                ]

            for rows, logprob_fn, weight, bias in chunks:
                chunk_grad_hidden, chunk_grad_weight, chunk_grad_bias = (
                    chunk_input_grads(
                        logprob_fn, flat_h[rows], weight, bias, flat_grad[rows], needs
                    )
                )
                if grad_hidden is not None and chunk_grad_hidden is not None:
                    grad_hidden[rows] = chunk_grad_hidden.to(grad_hidden.dtype)
                if grad_weight is not None and chunk_grad_weight is not None:
                    grad_weight += chunk_grad_weight.to(grad_weight.dtype)
                if grad_bias is not None and chunk_grad_bias is not None:
                    grad_bias += chunk_grad_bias.to(grad_bias.dtype)

        grad_hidden_out = (
            grad_hidden.reshape(B, T, H) if grad_hidden is not None else None
        )
        return (
            grad_hidden_out,
            grad_weight,
            grad_bias,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

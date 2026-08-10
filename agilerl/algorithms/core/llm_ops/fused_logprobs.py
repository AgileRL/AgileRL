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

from typing import Any

import torch

from agilerl.utils.llm_utils import gather_if_ds_param as _gather_if_ds_param


def fp32_lm_head_operands(
    lm_head_weight: torch.Tensor,
    lm_head_bias: torch.Tensor | None,
    cast_to_fp32: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Upcast the lm_head for one chunk loop, hoisted out of the loop body.

    A ``(V, H)`` fp32 copy is gigabytes at production vocab sizes, so a loop
    takes at most one, shared by every chunk. Call inside the ZeRO-3 gather, or
    the copy is taken from an unmaterialized shard.

    :param lm_head_weight: ``(V, H)``.
    :type lm_head_weight: torch.Tensor
    :param lm_head_bias: ``(V,)`` or ``None``.
    :type lm_head_bias: torch.Tensor | None
    :param cast_to_fp32: whether the loop runs its matmul in fp32.
    :type cast_to_fp32: bool
    :return: The head operands to pass to every chunk.
    :rtype: tuple[torch.Tensor, torch.Tensor | None]
    """
    if not cast_to_fp32 or lm_head_weight.dtype == torch.float32:
        return lm_head_weight, lm_head_bias
    bias = lm_head_bias.float() if lm_head_bias is not None else None
    return lm_head_weight.float(), bias


def _fused_logprob_chunk(
    h_chunk: torch.Tensor,
    lm_head_weight: torch.Tensor,
    lm_head_bias: torch.Tensor | None,
    target_chunk: torch.Tensor,
    temperature: float,
    cast_to_fp32: bool,
) -> torch.Tensor:
    """Per-token logprobs for one flat ``(chunk_rows, H)`` slab of hidden states.

    Functional (no in-place ops) so it is safe under both ``torch.compile`` and
    the autograd recompute in :class:`FusedLinearLogProbsFunction`. The
    max-shift used elsewhere for stability is folded into ``logsumexp`` (which
    is already numerically stable), keeping the body fusion-friendly.

    :param h_chunk: ``(chunk_rows, H)`` hidden states.
    :type h_chunk: torch.Tensor
    :param lm_head_weight: ``(V, H)``.
    :type lm_head_weight: torch.Tensor
    :param lm_head_bias: ``(V,)`` or ``None``.
    :type lm_head_bias: torch.Tensor | None
    :param target_chunk: ``(chunk_rows,)`` target token ids.
    :type target_chunk: torch.Tensor
    :param temperature: logits divided by this before log_softmax (skip at 1.0).
    :type temperature: float
    :param cast_to_fp32: upcast ``h_chunk`` so the matmul and the reduction run
        in fp32; the head is expected to arrive fp32 already
        (:func:`fp32_lm_head_operands`).
    :type cast_to_fp32: bool
    :return: ``(chunk_rows,)`` per-token logprobs.
    :rtype: torch.Tensor
    """
    if cast_to_fp32:
        h_chunk = h_chunk.float()
    if h_chunk.dtype != lm_head_weight.dtype:
        # Chunk-tiling callers upcast the head once ahead of their loop, so this
        # only fires for a direct call or an fp16 checkpoint under bf16 autocast
        # (fp32 hidden, fp16 head). It copies the whole ``(V, H)`` head.
        compute_dtype = torch.promote_types(h_chunk.dtype, lm_head_weight.dtype)
        h_chunk = h_chunk.to(compute_dtype)
        lm_head_weight = lm_head_weight.to(compute_dtype)
    logits = h_chunk @ lm_head_weight.t()
    if lm_head_bias is not None:
        logits = logits + lm_head_bias
    if temperature != 1.0:
        logits = logits / temperature
    selected = logits.gather(dim=-1, index=target_chunk.unsqueeze(-1)).squeeze(-1)
    log_z = torch.logsumexp(logits, dim=-1)
    return selected - log_z


# torch.compile cache for ``_fused_logprob_chunk`` (CUDA-only; any failure
# latches ``disabled`` and the eager path is used for the rest of the process).
_FUSED_LOGPROB_COMPILE_STATE: dict[str, Any] = {"fn": None, "disabled": False}


def _fused_logprob_chunk_dispatch(
    device: torch.device,
    h_chunk: torch.Tensor,
    lm_head_weight: torch.Tensor,
    lm_head_bias: torch.Tensor | None,
    target_chunk: torch.Tensor,
    temperature: float,
    cast_to_fp32: bool,
) -> torch.Tensor:
    """Run :func:`_fused_logprob_chunk`, compiled on CUDA with eager fallback."""
    state = _FUSED_LOGPROB_COMPILE_STATE
    args = (
        h_chunk,
        lm_head_weight,
        lm_head_bias,
        target_chunk,
        temperature,
        cast_to_fp32,
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


def fused_linear_logprobs_chunked(
    hidden: torch.Tensor,
    lm_head_weight: torch.Tensor,
    lm_head_bias: torch.Tensor | None,
    target_ids: torch.Tensor,
    temperature: float = 1.0,
    cast_to_fp32: bool = True,
    *,
    chunk_rows: int,
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
    :param lm_head_weight: ``(V, H)``.
    :type lm_head_weight: torch.Tensor
    :param lm_head_bias: ``(V,)`` or ``None``.
    :type lm_head_bias: torch.Tensor | None
    :param target_ids: ``(B, T)`` sampled token ids (caller does the shift).
    :type target_ids: torch.Tensor
    :param temperature: logits divided by this before log_softmax (skipped at 1.0).
    :type temperature: float, optional
    :param cast_to_fp32: run the per-chunk matmul and reduction in fp32, then
        cast back.
    :type cast_to_fp32: bool, optional
    :param chunk_rows: rows of the flattened ``(B*T)`` workspace per iteration.
    :type chunk_rows: int
    :return: ``(B, T)`` per-token logprobs in ``hidden.dtype``.
    :rtype: torch.Tensor
    """
    orig_dtype = hidden.dtype
    B, T, H = hidden.shape
    flat_h = hidden.reshape(-1, H)
    flat_targets = target_ids.reshape(-1).to(torch.long)
    N = flat_h.shape[0]
    out = torch.empty(N, dtype=orig_dtype, device=hidden.device)

    with _gather_if_ds_param(lm_head_weight, lm_head_bias):
        head_weight, head_bias = fp32_lm_head_operands(
            lm_head_weight,
            lm_head_bias,
            cast_to_fp32,
        )
        for s in range(0, N, chunk_rows):
            e = min(s + chunk_rows, N)
            result = _fused_logprob_chunk_dispatch(
                hidden.device,
                flat_h[s:e],
                head_weight,
                head_bias,
                flat_targets[s:e],
                temperature,
                cast_to_fp32,
            )
            out[s:e].copy_(result.to(orig_dtype) if cast_to_fp32 else result)

    return out.reshape(B, T)


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
    softmax`` (the max-shift used for stability cancels in the derivative).
    """

    @staticmethod
    def forward(
        ctx: Any,  # noqa: ANN401 -- torch autograd context carries arbitrary attrs
        hidden: torch.Tensor,
        lm_head_weight: torch.Tensor,
        lm_head_bias: torch.Tensor | None,
        target_ids: torch.Tensor,
        temperature: float,
        cast_to_fp32: bool,
        chunk_rows: int,
    ) -> torch.Tensor:
        with torch.no_grad():
            logps = fused_linear_logprobs_chunked(
                hidden,
                lm_head_weight,
                lm_head_bias,
                target_ids,
                temperature=temperature,
                cast_to_fp32=cast_to_fp32,
                chunk_rows=chunk_rows,
            )
        ctx.save_for_backward(hidden, lm_head_weight, lm_head_bias, target_ids)
        ctx.temperature = temperature
        ctx.cast_to_fp32 = cast_to_fp32
        ctx.chunk_rows = chunk_rows
        ctx.needs_hidden_grad = hidden.requires_grad
        ctx.needs_weight_grad = lm_head_weight.requires_grad
        ctx.needs_bias_grad = lm_head_bias is not None and lm_head_bias.requires_grad
        return logps

    @staticmethod
    def backward(
        ctx: Any,  # noqa: ANN401 -- torch autograd context carries arbitrary attrs
        *grad_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor | None, ...]:
        (grad_output,) = grad_outputs
        hidden, lm_head_weight, lm_head_bias, target_ids = ctx.saved_tensors
        B, T, H = hidden.shape
        flat_h = hidden.reshape(-1, H)
        flat_targets = target_ids.reshape(-1).to(torch.long)
        flat_grad = grad_output.reshape(-1)
        N = flat_h.shape[0]
        chunk_rows = ctx.chunk_rows
        temperature = ctx.temperature
        cast_to_fp32 = ctx.cast_to_fp32

        # Re-gather for the recompute: the forward gather has already exited
        # and ZeRO-3 has re-partitioned ``lm_head_weight`` to a zero-sized
        # shard. The Parameter object is the same; only ``.data`` was swapped.
        if ctx.needs_weight_grad and hasattr(lm_head_weight, "ds_id"):
            msg = (
                "ZeRO-3 full-finetune of lm_head through the fused-linear "
                "logprob path is not supported; use LoRA (lm_head frozen) "
                "or disable ZeRO-3."
            )
            raise RuntimeError(msg)
        with _gather_if_ds_param(lm_head_weight, lm_head_bias):
            grad_hidden = torch.zeros_like(flat_h) if ctx.needs_hidden_grad else None
            grad_weight = (
                torch.zeros_like(lm_head_weight) if ctx.needs_weight_grad else None
            )
            grad_bias = torch.zeros_like(lm_head_bias) if ctx.needs_bias_grad else None
            # Grad buffers above stay in the head's own dtype; the recompute
            # below runs against one hoisted fp32 copy of it.
            head_weight, head_bias = fp32_lm_head_operands(
                lm_head_weight,
                lm_head_bias,
                cast_to_fp32,
            )

            for s in range(0, N, chunk_rows):
                e = min(s + chunk_rows, N)
                h_chunk = flat_h[s:e].detach()
                if ctx.needs_hidden_grad:
                    h_chunk.requires_grad_(True)
                weight = head_weight
                if ctx.needs_weight_grad:
                    weight = weight.detach().requires_grad_(True)
                bias = head_bias
                if ctx.needs_bias_grad and bias is not None:
                    bias = bias.detach().requires_grad_(True)

                with torch.enable_grad():
                    logps_chunk = _fused_logprob_chunk_dispatch(
                        hidden.device,
                        h_chunk,
                        weight,
                        bias,
                        flat_targets[s:e],
                        temperature,
                        cast_to_fp32,
                    )

                inputs: list[torch.Tensor] = []
                if ctx.needs_hidden_grad:
                    inputs.append(h_chunk)
                if ctx.needs_weight_grad:
                    inputs.append(weight)
                if ctx.needs_bias_grad and bias is not None:
                    inputs.append(bias)
                if not inputs:
                    continue

                grads = torch.autograd.grad(
                    logps_chunk,
                    inputs,
                    grad_outputs=flat_grad[s:e].to(logps_chunk.dtype),
                )
                idx = 0
                if ctx.needs_hidden_grad and grad_hidden is not None:
                    grad_hidden[s:e] = grads[idx].to(grad_hidden.dtype)
                    idx += 1
                if ctx.needs_weight_grad and grad_weight is not None:
                    grad_weight += grads[idx].to(grad_weight.dtype)
                    idx += 1
                if ctx.needs_bias_grad and grad_bias is not None:
                    grad_bias += grads[idx].to(grad_bias.dtype)
                    idx += 1

        grad_hidden_out = (
            grad_hidden.reshape(B, T, H) if grad_hidden is not None else None
        )
        return (grad_hidden_out, grad_weight, grad_bias, None, None, None, None)

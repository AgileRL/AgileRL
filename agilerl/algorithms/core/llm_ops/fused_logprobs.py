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

# Target byte budget for one transient fp32 ``(chunk_rows, V)`` logits slab.
# ``chunk_rows`` is derived so the slab stays near this size, trading
# kernel-launch count against peak memory.
_FUSED_LOGPROBS_WORKSPACE_BYTES = 256 * 1024 * 1024


def _resolve_fused_logprobs_chunk_rows(vocab_size: int) -> int:
    """Vocab-aware row count for the fused-linear-logprob workspace.

    Sizes ``chunk_rows`` so the transient fp32 ``(chunk_rows, vocab)`` logits
    slab stays near :data:`_FUSED_LOGPROBS_WORKSPACE_BYTES`: large-vocab models
    (e.g. gemma's 262k) get fewer rows per chunk, small-vocab models more. This
    is the default used when the ``fused_logprobs_chunk_rows`` constructor
    kwarg is left ``None``.

    :param vocab_size: lm_head vocabulary dimension ``V``.
    :type vocab_size: int
    :return: rows of the flattened ``(B*T)`` workspace per chunk (128..4096).
    :rtype: int
    """
    rows = _FUSED_LOGPROBS_WORKSPACE_BYTES // max(1, int(vocab_size) * 4)
    return int(min(max(rows, 128), 4096))


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
    the autograd recompute in :class:`_FusedLinearLogProbsFunction`. The
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
    :param cast_to_fp32: run the reduction in fp32.
    :type cast_to_fp32: bool
    :return: ``(chunk_rows,)`` per-token logprobs.
    :rtype: torch.Tensor
    """
    logits = h_chunk @ lm_head_weight.t()
    if lm_head_bias is not None:
        logits = logits + lm_head_bias
    if temperature != 1.0:
        logits = logits / temperature
    if cast_to_fp32:
        logits = logits.float()
    selected = logits.gather(dim=-1, index=target_chunk.unsqueeze(-1)).squeeze(-1)
    log_z = torch.logsumexp(logits, dim=-1)
    return selected - log_z


# Lazily-compiled variant of :func:`_fused_logprob_chunk`. ``torch.compile``
# fuses the matmul + log-softmax reduction into Triton kernels (the same chunked
# GRPO log-softmax idea used by other fine-tuning stacks), which is both faster and lower-peak
# than eager. Compilation is attempted on first CUDA use; any failure (no
# triton, unsupported backend, CPU/MPS) falls back to eager permanently.
# Call :func:`disable_fused_logprob_compile` to force the eager path.
# State held in a dict so the dispatch can mutate it without ``global``:
# ``fn`` caches the compiled callable, ``disabled`` latches eager fallback.
_FUSED_LOGPROB_COMPILE_STATE: dict[str, Any] = {"fn": None, "disabled": False}


def disable_fused_logprob_compile() -> None:
    """Force the eager fused-logprob path for the rest of the process.

    Compilation (and this latch) is process-global — the compiled kernel is
    shared by every agent in the process — so the escape hatch is a module
    function rather than a per-algorithm argument. Use it if the compiled
    kernel misbehaves on a particular torch/triton combination; compilation
    failures themselves already fall back to eager automatically.
    """
    _FUSED_LOGPROB_COMPILE_STATE["disabled"] = True


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


def _fused_linear_logprobs_chunked(
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
    autograd-aware :class:`_FusedLinearLogProbsFunction`. Must be called under
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
    :param cast_to_fp32: run the per-chunk reduction in fp32 then cast back.
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

    for s in range(0, N, chunk_rows):
        e = min(s + chunk_rows, N)
        result = _fused_logprob_chunk_dispatch(
            hidden.device,
            flat_h[s:e],
            lm_head_weight,
            lm_head_bias,
            flat_targets[s:e],
            temperature,
            cast_to_fp32,
        )
        out[s:e].copy_(result.to(orig_dtype) if cast_to_fp32 else result)

    return out.reshape(B, T)


class _FusedLinearLogProbsFunction(torch.autograd.Function):
    """Gradient-checkpointed per-token logprobs over a chunked lm_head matmul.

    The forward computes per-token logprobs chunk-by-chunk under ``no_grad``
    (peak logits workspace bounded to ``(chunk_rows, V)``). The backward
    re-runs the same chunked matmul one chunk at a time with grad enabled,
    accumulates gradients into preallocated buffers, and frees each chunk's
    logits before the next. Peak logits memory is therefore ``O(chunk_rows *
    V)`` in *both* directions — the ``(B, T, V)`` slab is never materialized.

    Only ``(B, T, H)`` hidden states (saved for the recompute) are held across
    forward/backward, which the surrounding policy graph keeps alive anyway.

    Numerically the forward matches :func:`_fused_linear_logprobs_chunked`; the
    backward yields the exact ``log_softmax`` gradient ``onehot(target) -
    softmax`` (the max-shift used for stability cancels in the derivative).
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        hidden: torch.Tensor,
        lm_head_weight: torch.Tensor,
        lm_head_bias: torch.Tensor | None,
        target_ids: torch.Tensor,
        temperature: float,
        cast_to_fp32: bool,
        chunk_rows: int,
    ) -> torch.Tensor:
        with torch.no_grad():
            logps = _fused_linear_logprobs_chunked(
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
    def backward(  # type: ignore[override]
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor | None, ...]:
        hidden, lm_head_weight, lm_head_bias, target_ids = ctx.saved_tensors
        B, T, H = hidden.shape
        flat_h = hidden.reshape(-1, H)
        flat_targets = target_ids.reshape(-1).to(torch.long)
        flat_grad = grad_output.reshape(-1)
        N = flat_h.shape[0]
        chunk_rows = ctx.chunk_rows
        temperature = ctx.temperature
        cast_to_fp32 = ctx.cast_to_fp32

        grad_hidden = torch.zeros_like(flat_h) if ctx.needs_hidden_grad else None
        grad_weight = (
            torch.zeros_like(lm_head_weight) if ctx.needs_weight_grad else None
        )
        grad_bias = torch.zeros_like(lm_head_bias) if ctx.needs_bias_grad else None

        for s in range(0, N, chunk_rows):
            e = min(s + chunk_rows, N)
            h_chunk = flat_h[s:e].detach()
            if ctx.needs_hidden_grad:
                h_chunk.requires_grad_(True)
            weight = lm_head_weight
            if ctx.needs_weight_grad:
                weight = weight.detach().requires_grad_(True)
            bias = lm_head_bias
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

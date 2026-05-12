"""Fused linear PPO-style policy gradient loss with per-token / per-turn advantages.

Liger's ``LigerFusedLinearGRPOFunction`` only handles GRPO's
per-trajectory scalar advantage. LLMPPO carries ``(B, T)`` per-token
advantages (GAE returns); turn-PPO with ``turn_level_clip=True`` carries
``(B, max_turns)`` per-turn advantages. This module provides a sibling
autograd Function that handles both per-token and per-turn
advantages via a ``turn_ids`` switch.

Same chunked forward+backward idea as Liger's base: each chunk computes
the loss + accumulates ``grad_input``/``grad_weight`` and is then freed,
so the gradient-time ``(B, T, V)`` logits tensor is never materialized.
PPO's value-head loss runs outside this fusion (the value tensor is small).

The per-chunk math lives in :func:`agilerl.algorithms.core.llm_policy_loss.llm_policy_loss_fn`
(Liger-free, unit-testable). This module imports it and wraps it in an
autograd Function inheriting ``chunk_forward`` and ``backward`` from
Liger's base, overriding ``forward`` minimally to slice ``turn_ids``
alongside the other chunked inputs.

Importing this module requires ``liger-kernel`` to be installed; callers
should gate the import behind ``HAS_LIGER_KERNEL``.
"""

from __future__ import annotations

import torch

from agilerl import HAS_LIGER_KERNEL
from agilerl.algorithms.core.llm_policy_loss import llm_policy_loss_fn

if not HAS_LIGER_KERNEL:
    msg = (
        "agilerl.algorithms.core.fused_llm_policy_loss requires `liger-kernel`. "
        "Gate the import behind `agilerl.HAS_LIGER_KERNEL`, or use the "
        "unfused loss in `agilerl.algorithms.core.llm_policy_loss`."
    )
    raise ImportError(msg)

from liger_kernel.chunked_loss.fused_linear_ppo import LigerFusedLinearPPOBase


class LigerFusedLinearPolicyLossFunction(LigerFusedLinearPPOBase):
    """Fused linear PPO-style policy loss with per-token or per-turn ratios.

    Inherits ``chunk_forward`` (matmul + log-softmax) and ``backward``
    (saved-grad plumbing) from
    :class:`liger_kernel.chunked_loss.fused_linear_ppo.LigerFusedLinearPPOBase`,
    but overrides ``forward`` with our own chunk loop so we can slice
    ``turn_ids`` along dim 0 alongside the other chunked inputs —
    the base class hardcodes its chunked-arg list and doesn't expose
    an injection point.
    """

    @classmethod
    def forward(
        cls,
        ctx,
        _input,
        weight,
        selected_token_ids,
        attention_mask,
        advantages,
        bias=None,
        ref_per_token_logps=None,
        old_per_token_logps=None,
        beta=0.0,
        epsilon_low=0.2,
        epsilon_high=0.2,
        temperature=1.0,
        compiled=False,
        chunk_size=1,
        turn_ids=None,
        full_turn_mask=None,
        max_turns=None,
    ):
        """Chunked forward + backward.

        Mirrors the structure of
        :meth:`LigerFusedLinearPPOBase.forward` but with one extra
        chunked tensor (``turn_ids``) and a leaner static-arg list
        (Liger's SAPO/CISPO/vllm-IS knobs aren't reachable from this
        wrapper). When ``turn_ids`` is ``None`` this reduces to the
        existing token-mode behavior.
        """
        loss_acc = torch.zeros((), device=_input.device, dtype=torch.float32)
        grad_weight = torch.zeros_like(weight)
        grad_inputs: list[torch.Tensor] = []
        grad_bias = torch.zeros_like(bias) if bias is not None else None
        aggregated_metrics: list[torch.Tensor] = []

        full_attention_mask = attention_mask

        def _compute_chunk_loss(
            input_chunk,
            weight_local,
            selected_token_ids_chunk,
            attention_mask_chunk,
            advantages_chunk,
            bias_local=None,
            ref_per_token_logps_chunk=None,
            old_per_token_logps_chunk=None,
            turn_ids_chunk=None,
        ):
            log_probs, _ = LigerFusedLinearPPOBase.chunk_forward(
                input_chunk,
                weight_local,
                bias=bias_local,
                temperature=temperature,
            )
            return llm_policy_loss_fn(
                log_probs=log_probs,
                selected_token_ids=selected_token_ids_chunk,
                attention_mask=attention_mask_chunk,
                advantages=advantages_chunk,
                full_attention_mask=full_attention_mask,
                ref_per_token_logps=(
                    ref_per_token_logps_chunk.float()
                    if ref_per_token_logps_chunk is not None
                    else None
                ),
                old_per_token_logps=(
                    old_per_token_logps_chunk.float()
                    if old_per_token_logps_chunk is not None
                    else None
                ),
                epsilon_low=epsilon_low,
                epsilon_high=epsilon_high,
                beta=beta,
                turn_ids=turn_ids_chunk,
                full_turn_mask=full_turn_mask,
                max_turns=max_turns,
            )

        def fused_fwd_bwd(
            input_chunk,
            selected_token_ids_chunk,
            attention_mask_chunk,
            advantages_chunk,
            ref_per_token_logps_chunk,
            old_per_token_logps_chunk,
            turn_ids_chunk,
        ):
            argnums = (0, 1, 5) if bias is not None else (0, 1)
            return torch.func.grad_and_value(
                _compute_chunk_loss, argnums=argnums, has_aux=True
            )(
                input_chunk,
                weight,
                selected_token_ids_chunk,
                attention_mask_chunk,
                advantages_chunk,
                bias,
                ref_per_token_logps_chunk=ref_per_token_logps_chunk,
                old_per_token_logps_chunk=old_per_token_logps_chunk,
                turn_ids_chunk=turn_ids_chunk,
            )

        if compiled:  # pragma: no cover -- requires torch.compile warmup
            fused_fwd_bwd = torch.compile(fused_fwd_bwd)

        def accumulate_chunk(
            input_chunk,
            selected_token_ids_chunk,
            attention_mask_chunk,
            advantages_chunk,
            ref_per_token_logps_chunk,
            old_per_token_logps_chunk,
            turn_ids_chunk,
        ):
            (
                (chunk_grad_input, chunk_grad_weight, *chunk_grad_bias),
                (
                    chunk_loss,
                    chunk_metrics,
                ),
            ) = fused_fwd_bwd(
                input_chunk,
                selected_token_ids_chunk,
                attention_mask_chunk,
                advantages_chunk,
                ref_per_token_logps_chunk,
                old_per_token_logps_chunk,
                turn_ids_chunk,
            )
            if grad_bias is not None:
                grad_bias.add_(chunk_grad_bias[0])
            grad_weight.add_(chunk_grad_weight)
            grad_inputs.append(chunk_grad_input)
            loss_acc.add_(chunk_loss)
            if not aggregated_metrics:
                for metric in chunk_metrics:
                    if metric.ndim == 0:
                        aggregated_metrics.append(torch.zeros((), device=metric.device))
                    else:  # pragma: no cover -- llm_policy_loss_fn only returns scalars
                        aggregated_metrics.append([])  # type: ignore[arg-type]
            for i, metric in enumerate(chunk_metrics):
                if metric.ndim == 0:
                    aggregated_metrics[i].add_(metric)
                else:  # pragma: no cover -- llm_policy_loss_fn only returns scalars
                    aggregated_metrics[i].append(metric)  # type: ignore[union-attr]

        chunks = max(1, _input.shape[0] // chunk_size)
        _input_chunks = torch.chunk(_input, chunks=chunks, dim=0)
        _ids_chunks = torch.chunk(selected_token_ids, chunks=chunks, dim=0)
        _mask_chunks = torch.chunk(attention_mask, chunks=chunks, dim=0)
        _adv_chunks = torch.chunk(advantages, chunks=chunks, dim=0)
        _ref_chunks = (
            torch.chunk(ref_per_token_logps, chunks=chunks, dim=0)
            if ref_per_token_logps is not None
            else [None] * chunks
        )
        _old_chunks = (
            torch.chunk(old_per_token_logps, chunks=chunks, dim=0)
            if old_per_token_logps is not None
            else [None] * chunks
        )
        _turn_chunks = (
            torch.chunk(turn_ids, chunks=chunks, dim=0)
            if turn_ids is not None
            else [None] * chunks
        )

        for ic, idc, mc, ac, rc, oc, tc in zip(
            _input_chunks,
            _ids_chunks,
            _mask_chunks,
            _adv_chunks,
            _ref_chunks,
            _old_chunks,
            _turn_chunks,
            strict=True,
        ):
            accumulate_chunk(ic, idc, mc, ac, rc, oc, tc)

        grad_input = torch.cat(grad_inputs, dim=0)
        ctx.save_for_backward(grad_input, grad_weight, grad_bias)

        final_metrics: list[torch.Tensor] = []
        for metric in aggregated_metrics:
            if isinstance(metric, list):  # pragma: no cover -- scalars only
                final_metrics.append(torch.cat(metric, dim=0))
            else:
                final_metrics.append(metric)
        return loss_acc, tuple(final_metrics)

    @staticmethod
    def backward(ctx, grad_output, *grad_metrics):
        grads = LigerFusedLinearPPOBase.backward(ctx, grad_output)
        # forward arity after ctx: 17 inputs (added turn_ids,
        # full_turn_mask, max_turns to the original 14).
        return (
            *grads[
                :6
            ],  # _input, weight, selected_token_ids, attention_mask, advantages, bias
            None,  # ref_per_token_logps
            None,  # old_per_token_logps
            None,  # beta
            None,  # epsilon_low
            None,  # epsilon_high
            None,  # temperature
            None,  # compiled
            None,  # chunk_size
            None,  # turn_ids
            None,  # full_turn_mask
            None,  # max_turns
        )

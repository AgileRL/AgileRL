"""Fused linear PPO/REINFORCE loss with per-token / per-turn advantages.

Liger's stock ``LigerFusedLinearGRPOFunction`` only handles GRPO's
per-trajectory scalar advantage. LLMPPO carries ``(B, T)`` per-token
advantages (GAE returns); turn-PPO with ``turn_level_clip=True`` carries
``(B, max_turns)`` per-turn advantages. This module provides a sibling
autograd Function that handles both via a ``turn_ids`` switch.

Same chunked forward+backward idea as Liger's base: each chunk computes
the loss + accumulates ``grad_input``/``grad_weight`` and is then freed,
so the gradient-time ``(B, T, V)`` logits tensor is never materialized.
PPO's value-head loss runs outside this fusion (the value tensor is small).

The math sits in :func:`llm_ppo_loss_fn` at module top level so it's
unit-testable without Liger installed. The autograd Function
:class:`LigerFusedLinearLLMPPOFunction` resolves only when
``liger-kernel`` is importable, inherits ``chunk_forward`` and
``backward`` from Liger's base, and overrides ``forward`` minimally to
slice ``turn_ids`` alongside the other chunked inputs.
"""

from __future__ import annotations

import torch

from agilerl import HAS_LIGER_KERNEL


def _k3_kl(log_p: torch.Tensor, log_q: torch.Tensor) -> torch.Tensor:
    """K3 estimator of ``KL[q || p]`` (Schulman 2020).

    Identical to the helper in Liger's ``grpo_loss.k3_loss_fn`` — duplicated
    here to keep this module self-contained.
    """
    return torch.exp(log_p - log_q) - (log_p - log_q) - 1.0


def llm_ppo_loss_fn(
    log_probs: torch.Tensor,
    selected_token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    advantages: torch.Tensor,
    full_attention_mask: torch.Tensor,
    ref_per_token_logps: torch.Tensor | None = None,
    old_per_token_logps: torch.Tensor | None = None,
    epsilon_low: float = 0.2,
    epsilon_high: float = 0.2,
    beta: float = 0.0,
    turn_ids: torch.Tensor | None = None,
    full_turn_mask: torch.Tensor | None = None,
    max_turns: int | None = None,
    **_unused: object,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Per-chunk policy + KL loss with per-token or per-turn ratio clipping.

    Importable without ``liger_kernel`` so the math can be unit-tested on
    boxes that don't have Liger; the full
    :class:`LigerFusedLinearLLMPPOFunction` autograd Function below wraps
    this fn behind a chunked forward+backward.

    Branches:

    * ``turn_ids is None`` — **token mode**: ratio, clip, max(-adv*r, ...)
      at the token level. ``advantages`` is ``(chunk_B, T)``.
    * ``turn_ids is not None`` — **turn mode**: token log-ratios are
      ``scatter_add_``'d into ``(chunk_B, max_turns)`` per-turn log
      ratios; clipping and the policy formula run on those.
      ``advantages`` is ``(chunk_B, max_turns)``. ``full_turn_mask`` is
      the global ``(B, max_turns)`` mask (for the cross-chunk reduction
      denominator). KL stays token-level (matches unfused turn-PPO).

    :param log_probs: ``(chunk_B, T, V)`` fp32 log-softmax (caller passes
        the output of Liger's ``chunk_forward``).
    :param selected_token_ids: ``(chunk_B, T)`` target token ids.
    :param attention_mask: ``(chunk_B, T)`` token-level action mask.
    :param advantages: token mode ``(chunk_B, T)``; turn mode
        ``(chunk_B, max_turns)``.
    :param full_attention_mask: ``(B, T)`` global mask used as the
        token-level reduction denominator (KL, and the policy-loss in
        token mode).
    :param ref_per_token_logps: ``(chunk_B, T)`` reference logprobs (for
        the KL metric and, when ``beta > 0``, the KL penalty).
    :param old_per_token_logps: ``(chunk_B, T)`` old-policy logprobs.
    :param epsilon_low: PPO clip lower bound.
    :param epsilon_high: PPO clip upper bound.
    :param beta: KL penalty weight (0 disables — REINFORCE folds KL into
        advantages upstream).
    :param turn_ids: ``(chunk_B, T)`` turn index per token, ``-1`` for
        non-action tokens. ``None`` selects token mode.
    :param full_turn_mask: ``(B, max_turns)`` global per-turn existence
        mask (used as the reduction denominator in turn mode).
    :param max_turns: ``int``, total turn buckets across the batch.
    :return: ``(chunk_loss, [kl, clipfrac, pg_loss, entropy])`` — first
        element backprops; metrics are detached scalars contributing to
        the global mean across chunks.
    """
    per_token_logps = log_probs.gather(
        dim=-1, index=selected_token_ids.unsqueeze(-1)
    ).squeeze(-1)

    if old_per_token_logps is None:
        # PPO/REINFORCE always pass old_log_probs; this guards against a
        # caller forgetting (ratio == 1, gradient still well-defined).
        old_per_token_logps = per_token_logps.detach()
    token_log_ratio = per_token_logps - old_per_token_logps

    # KL is logged regardless of whether it's added to the loss — REINFORCE
    # folds KL into advantages upstream and runs with beta=0 here, but
    # still wants the kl scalar for monitoring. KL stays token-level in
    # both branches (matches the unfused PPO/REINFORCE convention).
    kl_div: torch.Tensor | None = None
    if ref_per_token_logps is not None:
        kl_div = _k3_kl(ref_per_token_logps, per_token_logps)

    token_global_count = full_attention_mask.float().sum().clamp(min=1.0)

    if turn_ids is None:
        # Token mode: ratio + clip + max-formula at token level.
        ratio = torch.exp(token_log_ratio)
        clipped_ratio = torch.clamp(ratio, 1.0 - epsilon_low, 1.0 + epsilon_high)
        pg_unit_loss = torch.max(-advantages * ratio, -advantages * clipped_ratio)
        unit_mask = attention_mask
        unit_global_count = token_global_count
    else:
        # Turn mode: pool token log-ratios per turn, then clip + max at turn level.
        if max_turns is None or full_turn_mask is None:
            msg = (
                "turn-mode loss requires max_turns and full_turn_mask. "
                "Got turn_ids without one of them."
            )
            raise ValueError(msg)
        chunk_b = token_log_ratio.shape[0]
        # Mask non-action tokens out of the per-turn sum and clamp -1
        # turn_ids to bucket 0 (mask handles the exclusion).
        masked_token_log_ratio = token_log_ratio * attention_mask
        safe_turn_ids = turn_ids.clamp(min=0)
        # Sum-pool token log-ratios into per-turn log-ratios. scatter_add
        # is autograd-friendly along the value tensor, so gradients flow
        # back through token_log_ratio → per_token_logps → log_probs.
        turn_log_ratio = torch.zeros(
            chunk_b,
            max_turns,
            dtype=token_log_ratio.dtype,
            device=token_log_ratio.device,
        )
        turn_log_ratio = turn_log_ratio.scatter_add(
            1, safe_turn_ids, masked_token_log_ratio
        )
        # Per-chunk turn mask: a turn is active in this chunk iff at
        # least one of its tokens has mask=1.
        chunk_turn_active = torch.zeros_like(turn_log_ratio)
        chunk_turn_active = chunk_turn_active.scatter_add(
            1, safe_turn_ids, attention_mask.float()
        )
        chunk_turn_mask = (chunk_turn_active > 0).to(turn_log_ratio.dtype)

        ratio = torch.exp(turn_log_ratio)
        clipped_ratio = torch.clamp(ratio, 1.0 - epsilon_low, 1.0 + epsilon_high)
        pg_unit_loss = torch.max(-advantages * ratio, -advantages * clipped_ratio)
        unit_mask = chunk_turn_mask
        unit_global_count = full_turn_mask.float().sum().clamp(min=1.0)

    chunk_loss = (pg_unit_loss * unit_mask).sum() / unit_global_count
    if beta != 0.0 and kl_div is not None:
        # KL term added at the token level — unfused PPO does the same.
        chunk_loss = chunk_loss + beta * (
            (kl_div * attention_mask).sum() / token_global_count
        )

    # Metrics — detached scalars contributing to the global mean. The
    # subclass forward .add_()s them into running totals across chunks.
    with torch.no_grad():
        kl_metric = (
            (kl_div * attention_mask).sum() / token_global_count
            if kl_div is not None
            else torch.zeros((), device=log_probs.device, dtype=log_probs.dtype)
        )
        is_clipped = ratio != clipped_ratio
        clipfrac_metric = (
            is_clipped.to(unit_mask.dtype) * unit_mask
        ).sum() / unit_global_count
        pg_loss_metric = (pg_unit_loss * unit_mask).sum() / unit_global_count
        # Entropy proxy: -log p of chosen tokens, masked at the token
        # level regardless of mode.
        entropy_metric = (
            -per_token_logps.detach() * attention_mask
        ).sum() / token_global_count

    return chunk_loss, [kl_metric, clipfrac_metric, pg_loss_metric, entropy_metric]


if not HAS_LIGER_KERNEL:
    # No Liger installed — the autograd Function below isn't available.
    # ``llm_ppo_loss_fn`` is still importable for unit tests of the math.
    LigerFusedLinearLLMPPOFunction = None  # type: ignore[assignment]
else:
    from liger_kernel.chunked_loss.fused_linear_ppo import LigerFusedLinearPPOBase

    class LigerFusedLinearLLMPPOFunction(LigerFusedLinearPPOBase):  # type: ignore[no-redef]
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
            ppo_loss_fn = llm_ppo_loss_fn

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
                return ppo_loss_fn(
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

            if compiled:
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
                if bias is not None:
                    grad_bias.add_(chunk_grad_bias[0])
                grad_weight.add_(chunk_grad_weight)
                grad_inputs.append(chunk_grad_input)
                loss_acc.add_(chunk_loss)
                if not aggregated_metrics:
                    for metric in chunk_metrics:
                        if metric.ndim == 0:
                            aggregated_metrics.append(
                                torch.zeros((), device=metric.device)
                            )
                        else:
                            aggregated_metrics.append([])  # type: ignore[arg-type]
                for i, metric in enumerate(chunk_metrics):
                    if metric.ndim == 0:
                        aggregated_metrics[i].add_(metric)
                    else:
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
                if isinstance(metric, list):
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

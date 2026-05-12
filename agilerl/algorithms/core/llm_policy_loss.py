"""Pure-tensor PPO-style policy + KL loss with per-token / per-turn advantages.

The math here drives both the unfused PPO/REINFORCE path (called once on the
full ``(B, T)`` tensors) and the Liger-fused autograd Function in
:mod:`agilerl.algorithms.core.fused_llm_policy_loss` (called per chunk after
``LigerFusedLinearPPOBase.chunk_forward`` produces fp32 ``log_probs``).
"""

from __future__ import annotations

import torch


def _k3_kl(log_p: torch.Tensor, log_q: torch.Tensor) -> torch.Tensor:
    """K3 estimator of ``KL[q || p]`` (Schulman 2020).

    Identical to the helper in Liger's ``grpo_loss.k3_loss_fn`` — duplicated
    here to keep this module Liger-free.
    """
    return torch.exp(log_p - log_q) - (log_p - log_q) - 1.0


def llm_policy_loss_fn(
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

    :class:`agilerl.algorithms.core.fused_llm_policy_loss.LigerFusedLinearPolicyLossFunction`
    wraps this fn behind a chunked forward+backward.

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

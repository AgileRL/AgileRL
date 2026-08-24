# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

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

The per-chunk math (:func:`llm_policy_loss_fn`) lives in this module;
the K3 KL-divergence estimator it uses (:func:`calculate_k3_kl`) is canonically
defined in :mod:`agilerl.utils.llm_utils` and re-exported here for
backward-compatible imports. This module requires ``liger-kernel`` at
import time — gate on :data:`agilerl.HAS_LIGER_KERNEL` before importing.
"""

from __future__ import annotations

from typing import Protocol

import torch
from torch.autograd.function import FunctionCtx

from agilerl import HAS_LIGER_KERNEL

if not HAS_LIGER_KERNEL:
    msg = (
        "Liger fused loss functions are only available when liger-kernel "
        "is installed. Check ``HAS_LIGER_KERNEL`` before importing or "
        "using this module."
    )
    raise ImportError(msg)

from liger_kernel.chunked_loss.dpo_loss import LigerFusedLinearDPOFunction
from liger_kernel.chunked_loss.fused_linear_ppo import LigerFusedLinearPPOBase
from liger_kernel.chunked_loss.fused_linear_preference import (
    LigerFusedLinearPreferenceBase,
)

from agilerl.utils.llm_utils import calculate_k3_kl


class SaveForBackwardCtx(Protocol):
    """The one autograd-``ctx`` capability these forwards use."""

    def save_for_backward(self, *tensors: torch.Tensor | None) -> None: ...


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
    importance_sampling_level: str = "token",
    turn_log_ratio_reduction: str = "mean",
    vllm_is_ratio: torch.Tensor | None = None,
    # Liger's ``LigerFusedLinearPPOBase._compute_loss`` invokes the loss fn
    # with kwargs this fn does not consume (``ref_log_probs``, ``loss_type``,
    # ``max_completion_length``, ``sapo_temperature_pos``, ...), so a
    # catch-all is required to absorb them.
    **_unused: object,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Per-chunk policy + KL loss at token / turn / trajectory ratio granularity.

    :class:`LigerFusedLinearPolicyLossFunction` (below) wraps this fn
    behind a chunked forward+backward. ``importance_sampling_level`` selects
    the branch; ``turn_ids`` / ``full_turn_mask`` / ``max_turns`` are required
    by (and only used in) the turn branch:

    * ``"token"`` — ratio, clip, max(-adv*r, ...) at the token level.
      ``advantages`` is ``(chunk_B, T)``.
    * ``"turn"`` — token log-ratios are pooled into
      ``(chunk_B, max_turns)`` per-turn log ratios; clipping and the policy
      formula run on those. ``turn_log_ratio_reduction="mean"`` gives a
      length-normalized mean (geometric-mean ratio), ``"sum"`` gives a sum
      (product ratio). ``advantages`` is ``(chunk_B, max_turns)``.
      ``full_turn_mask`` is the global ``(B, max_turns)`` mask (for the
      cross-chunk reduction denominator). KL stays token-level (matches
      unfused turn-PPO).
    * ``"trajectory"`` — token log-ratios are mean-pooled over the whole
      completion (GSPO). ``advantages`` is ``(chunk_B, 1)``.

    :param log_probs: ``(chunk_B, T, V)`` fp32 log-softmax of the chunk's
        logits (the fused Function computes this inline per chunk).
    :type log_probs: torch.Tensor
    :param selected_token_ids: ``(chunk_B, T)`` target token ids.
    :type selected_token_ids: torch.Tensor
    :param attention_mask: ``(chunk_B, T)`` token-level action mask.
    :type attention_mask: torch.Tensor
    :param advantages: token mode ``(chunk_B, T)``; turn mode
        ``(chunk_B, max_turns)``; trajectory mode ``(chunk_B, 1)``.
    :type advantages: torch.Tensor
    :param full_attention_mask: ``(B, T)`` global mask used as the
        token-level reduction denominator (KL, and the policy-loss in
        token mode).
    :type full_attention_mask: torch.Tensor
    :param ref_per_token_logps: ``(chunk_B, T)`` reference logprobs (for
        the KL metric and, when ``beta > 0``, the KL penalty).
    :type ref_per_token_logps: torch.Tensor | None
    :param old_per_token_logps: ``(chunk_B, T)`` old-policy logprobs.
    :type old_per_token_logps: torch.Tensor | None
    :param epsilon_low: PPO clip lower bound.
    :type epsilon_low: float
    :param epsilon_high: PPO clip upper bound.
    :type epsilon_high: float
    :param beta: KL penalty weight (0 disables — REINFORCE folds KL into
        advantages upstream).
    :type beta: float
    :param turn_ids: ``(chunk_B, T)`` turn index per token, ``-1`` for
        non-action tokens. Required for the turn level.
    :type turn_ids: torch.Tensor | None
    :param full_turn_mask: ``(B, max_turns)`` global per-turn existence
        mask (used as the reduction denominator in turn mode).
    :type full_turn_mask: torch.Tensor | None
    :param max_turns: Total turn buckets across the batch.
    :type max_turns: int | None
    :param importance_sampling_level: Ratio-pooling granularity —
        ``"token"`` (default), ``"turn"`` or ``"trajectory"``.
    :type importance_sampling_level: str
    :param turn_log_ratio_reduction: Turn-level reduction for pooled log-ratios,
        one of ``"mean"`` or ``"sum"``.
    :type turn_log_ratio_reduction: str
    :param vllm_is_ratio: Optional detached, upper-clamped per-token vLLM
        sampling-mismatch ratio ``(chunk_B, T)`` (token mode only). When
        provided, the per-token policy loss is multiplied by it *before* the KL
        term, matching the standard PyTorch path. The per-token reweight cannot
        be pooled into the turn/trajectory ratio, so it is honoured for token
        mode only; ``None`` keeps the loss identical to the uncorrected path.
    :type vllm_is_ratio: torch.Tensor | None
    :return: ``(chunk_loss, [kl, clipfrac, pg_loss, entropy])`` — first
        element backprops; metrics are detached scalars contributing to
        the global mean across chunks.
    :rtype: tuple[torch.Tensor, list[torch.Tensor]]
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
        kl_div = calculate_k3_kl(ref_per_token_logps, per_token_logps)

    token_global_count = full_attention_mask.float().sum().clamp(min=1.0)

    if importance_sampling_level == "token":
        if turn_ids is not None:
            # The level is authoritative; fail loudly on a stray turn_ids.
            msg = (
                "turn_ids was provided but importance_sampling_level='token'; "
                "pass importance_sampling_level='turn' for turn-level pooling."
            )
            raise ValueError(msg)
        # Token mode: ratio + clip + max-formula at token level.
        ratio = torch.exp(token_log_ratio)
        clipped_ratio = torch.clamp(ratio, 1.0 - epsilon_low, 1.0 + epsilon_high)
        pg_unit_loss = torch.max(-advantages * ratio, -advantages * clipped_ratio)
        if vllm_is_ratio is not None:
            # Truncated importance sampling: reweight each token by the
            # detached, upper-clamped trainer/vLLM probability ratio, applied to
            # the policy term before the KL penalty (matches the standard path).
            pg_unit_loss = pg_unit_loss * vllm_is_ratio
        unit_mask = attention_mask
        unit_global_count = token_global_count
    elif importance_sampling_level == "turn":
        # Turn mode: pool token log-ratios per turn, then clip + max at turn
        # level. ``turn_log_ratio_reduction="mean"`` gives a geometric-mean
        # ratio; ``"sum"`` gives the product ratio. ``advantages`` is per-turn
        # ``(chunk_b, max_turns)``.
        if max_turns is None or full_turn_mask is None or turn_ids is None:
            msg = "turn-level loss requires turn_ids, max_turns and full_turn_mask."
            raise ValueError(msg)
        if turn_log_ratio_reduction not in {"mean", "sum"}:
            msg = (
                "turn_log_ratio_reduction must be one of ['mean', 'sum'], got "
                f"{turn_log_ratio_reduction!r}."
            )
            raise ValueError(msg)
        chunk_b = token_log_ratio.shape[0]
        # Mask non-action tokens out of the per-turn sum and clamp -1
        # turn_ids to bucket 0 (mask handles the exclusion).
        masked_token_log_ratio = token_log_ratio * attention_mask
        safe_turn_ids = turn_ids.clamp(min=0)
        # Pool token log-ratios into per-turn log-ratios. scatter_add is
        # autograd-friendly along the value tensor, so gradients flow back
        # through token_log_ratio -> per_token_logps -> log_probs.
        turn_log_ratio_sum = torch.zeros(
            chunk_b,
            max_turns,
            dtype=token_log_ratio.dtype,
            device=token_log_ratio.device,
        ).scatter_add(1, safe_turn_ids, masked_token_log_ratio)
        # Per-turn token count; also yields the per-chunk turn mask (a turn is
        # active iff it has >= 1 action token).
        chunk_turn_active = torch.zeros_like(turn_log_ratio_sum).scatter_add(
            1, safe_turn_ids, attention_mask.to(token_log_ratio.dtype)
        )
        chunk_turn_mask = (chunk_turn_active > 0).to(token_log_ratio.dtype)
        if turn_log_ratio_reduction == "mean":
            turn_log_ratio = turn_log_ratio_sum / chunk_turn_active.clamp(min=1.0)
        else:
            turn_log_ratio = turn_log_ratio_sum

        ratio = torch.exp(turn_log_ratio)
        clipped_ratio = torch.clamp(ratio, 1.0 - epsilon_low, 1.0 + epsilon_high)
        pg_unit_loss = torch.max(-advantages * ratio, -advantages * clipped_ratio)
        unit_mask = chunk_turn_mask
        unit_global_count = full_turn_mask.float().sum().clamp(min=1.0)
    elif importance_sampling_level == "trajectory":
        # Trajectory mode: length-normalized mean over all action tokens of the
        # completion (GSPO). ``advantages`` is per-trajectory ``(chunk_b, 1)``.
        mask_f = attention_mask.to(token_log_ratio.dtype)
        seq_count = mask_f.sum(dim=-1, keepdim=True)  # (chunk_b, 1)
        seq_log_ratio = (token_log_ratio * mask_f).sum(
            dim=-1, keepdim=True
        ) / seq_count.clamp(min=1.0)
        chunk_seq_mask = (seq_count > 0).to(token_log_ratio.dtype)  # (chunk_b, 1)
        ratio = torch.exp(seq_log_ratio)
        clipped_ratio = torch.clamp(ratio, 1.0 - epsilon_low, 1.0 + epsilon_high)
        pg_unit_loss = torch.max(-advantages * ratio, -advantages * clipped_ratio)
        unit_mask = chunk_seq_mask
        # Normalize by the number of active sequences in the full batch.
        unit_global_count = (
            (full_attention_mask.sum(dim=-1) > 0).to(token_log_ratio.dtype).sum()
        ).clamp(min=1.0)
    else:
        msg = (
            f"Unknown importance_sampling_level '{importance_sampling_level}'. "
            "Expected one of ['token', 'turn', 'trajectory']."
        )
        raise ValueError(msg)

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


class LigerFusedLinearPolicyLossFunction(LigerFusedLinearPPOBase):
    """Fused linear PPO-style policy loss with per-token or per-turn ratios.

    Inherits ``backward`` (saved-grad plumbing) from
    :class:`liger_kernel.chunked_loss.fused_linear_ppo.LigerFusedLinearPPOBase`,
    but overrides ``forward`` with our own chunk loop so we can slice
    ``turn_ids`` along dim 0 alongside the other chunked inputs —
    the base class hardcodes its chunked-arg list and doesn't expose
    an injection point. The per-chunk logits + log-softmax are computed
    inline: Liger >= 0.8.0's ``chunk_forward`` is a selective
    per-token-logp kernel, and this loss needs the full
    ``(chunk, T, V)`` log_probs.
    """

    @classmethod
    def forward(  # ty: ignore[invalid-method-override]  # intentionally diverges from Liger's base
        cls,
        ctx: SaveForBackwardCtx,
        _input: torch.Tensor,
        weight: torch.Tensor,
        selected_token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        advantages: torch.Tensor,
        bias: torch.Tensor | None = None,
        ref_per_token_logps: torch.Tensor | None = None,
        old_per_token_logps: torch.Tensor | None = None,
        beta: float = 0.0,
        epsilon_low: float = 0.2,
        epsilon_high: float = 0.2,
        temperature: float = 1.0,
        compiled: bool = False,
        chunk_size: int = 1,
        turn_ids: torch.Tensor | None = None,
        full_turn_mask: torch.Tensor | None = None,
        max_turns: int | None = None,
        importance_sampling_level: str = "token",
        turn_log_ratio_reduction: str = "mean",
        vllm_is_ratio: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Chunked forward + backward.

        Mirrors the structure of
        :meth:`LigerFusedLinearPPOBase.forward` but with two extra
        chunked tensors (``turn_ids`` and the optional ``vllm_is_ratio``)
        and a leaner static-arg list (Liger's SAPO/CISPO settings aren't
        reachable from this wrapper). When ``turn_ids`` is ``None`` this
        reduces to the existing token-mode behavior; ``vllm_is_ratio`` is the
        detached, upper-clamped per-token vLLM sampling-mismatch ratio applied
        to the per-token policy loss (token mode only), or ``None``.
        """
        loss_acc = torch.zeros((), device=_input.device, dtype=torch.float32)
        grad_weight = torch.zeros_like(weight)
        grad_inputs: list[torch.Tensor] = []
        grad_bias = torch.zeros_like(bias) if bias is not None else None
        # Scalar metrics accumulate into a zero tensor; non-scalar metrics
        # (not produced by llm_policy_loss_fn today) collect per-chunk values.
        aggregated_metrics: list[torch.Tensor | list[torch.Tensor]] = []

        full_attention_mask = attention_mask

        def _compute_chunk_loss(
            input_chunk: torch.Tensor,
            weight_local: torch.Tensor,
            selected_token_ids_chunk: torch.Tensor,
            attention_mask_chunk: torch.Tensor,
            advantages_chunk: torch.Tensor,
            bias_local: torch.Tensor | None = None,
            ref_per_token_logps_chunk: torch.Tensor | None = None,
            old_per_token_logps_chunk: torch.Tensor | None = None,
            turn_ids_chunk: torch.Tensor | None = None,
            vllm_is_ratio_chunk: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, list[torch.Tensor]]:
            # Liger 0.8.0 rewrote ``LigerFusedLinearPPOBase.chunk_forward`` into a
            # selective-logp kernel that doesn't compose with the grad_and_value
            # transform below, so inline the numerically identical 0.7.0 math.
            if input_chunk.dtype != weight_local.dtype:
                # fp16 checkpoint under bf16 autocast: fp32 hidden, fp16 head.
                compute_dtype = torch.promote_types(
                    input_chunk.dtype, weight_local.dtype
                )
                input_chunk = input_chunk.to(compute_dtype)
                weight_local = weight_local.to(compute_dtype)
            logits = torch.matmul(input_chunk, weight_local.t())
            if bias_local is not None:
                logits = logits + bias_local
            if temperature != 1.0:
                logits = logits / temperature
            log_probs = torch.log_softmax(logits.float(), dim=-1)
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
                importance_sampling_level=importance_sampling_level,
                turn_log_ratio_reduction=turn_log_ratio_reduction,
                vllm_is_ratio=vllm_is_ratio_chunk,
            )

        def fused_fwd_bwd(
            input_chunk: torch.Tensor,
            selected_token_ids_chunk: torch.Tensor,
            attention_mask_chunk: torch.Tensor,
            advantages_chunk: torch.Tensor,
            ref_per_token_logps_chunk: torch.Tensor | None,
            old_per_token_logps_chunk: torch.Tensor | None,
            turn_ids_chunk: torch.Tensor | None,
            vllm_is_ratio_chunk: torch.Tensor | None,
        ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, list[torch.Tensor]]]:
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
                vllm_is_ratio_chunk=vllm_is_ratio_chunk,
            )

        if compiled:  # pragma: no cover -- requires torch.compile warmup
            fwd_bwd_fn = torch.compile(fused_fwd_bwd)
        else:
            fwd_bwd_fn = fused_fwd_bwd

        def accumulate_chunk(
            input_chunk: torch.Tensor,
            selected_token_ids_chunk: torch.Tensor,
            attention_mask_chunk: torch.Tensor,
            advantages_chunk: torch.Tensor,
            ref_per_token_logps_chunk: torch.Tensor | None,
            old_per_token_logps_chunk: torch.Tensor | None,
            turn_ids_chunk: torch.Tensor | None,
            vllm_is_ratio_chunk: torch.Tensor | None,
        ) -> None:
            (
                (chunk_grad_input, chunk_grad_weight, *chunk_grad_bias),
                (
                    chunk_loss,
                    chunk_metrics,
                ),
            ) = fwd_bwd_fn(
                input_chunk,
                selected_token_ids_chunk,
                attention_mask_chunk,
                advantages_chunk,
                ref_per_token_logps_chunk,
                old_per_token_logps_chunk,
                turn_ids_chunk,
                vllm_is_ratio_chunk,
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
                        aggregated_metrics.append([])
            for metric, agg_metric in zip(
                chunk_metrics, aggregated_metrics, strict=True
            ):
                if isinstance(agg_metric, torch.Tensor):
                    agg_metric.add_(metric)
                else:  # pragma: no cover -- llm_policy_loss_fn only returns scalars
                    agg_metric.append(metric)

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
        _vllm_chunks = (
            torch.chunk(vllm_is_ratio, chunks=chunks, dim=0)
            if vllm_is_ratio is not None
            else [None] * chunks
        )

        for ic, idc, mc, ac, rc, oc, tc, vc in zip(
            _input_chunks,
            _ids_chunks,
            _mask_chunks,
            _adv_chunks,
            _ref_chunks,
            _old_chunks,
            _turn_chunks,
            _vllm_chunks,
            strict=True,
        ):
            accumulate_chunk(ic, idc, mc, ac, rc, oc, tc, vc)

        grad_input = torch.cat(grad_inputs, dim=0)
        # grad_bias is None when the layer has no bias; save_for_backward
        # accepts None at runtime, but torch types it as Tensor.
        ctx.save_for_backward(grad_input, grad_weight, grad_bias)

        final_metrics: list[torch.Tensor] = []
        for metric in aggregated_metrics:
            if isinstance(metric, torch.Tensor):
                final_metrics.append(metric)
            else:  # pragma: no cover -- scalars only
                final_metrics.append(torch.cat(metric, dim=0))
        return loss_acc, tuple(final_metrics)

    @staticmethod
    def backward(
        ctx: FunctionCtx,
        grad_output: torch.Tensor,
        *grad_metrics: torch.Tensor,
    ) -> tuple[torch.Tensor | None, ...]:
        grads = LigerFusedLinearPPOBase.backward(ctx, grad_output)
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
            None,  # importance_sampling_level
            None,  # turn_log_ratio_reduction
            None,  # vllm_is_ratio
        )


def flatten_tokens_for_fused_loss(
    policy_hidden: torch.Tensor,
    target_ids: torch.Tensor,
    mask: torch.Tensor,
    old_log_probs: torch.Tensor | None,
    reference_log_probs: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Flatten token-level inputs from ``(B, T, ...)`` to ``(B*T, 1, ...)``.

    For ``importance_sampling_level="token"`` the per-token objective is
    token-independent, so the ``(B, T_act, H)`` hidden states are flattened to
    ``(B*T_act, 1, H)`` and the fused kernel chunks over *tokens* instead of
    sequences. This performs that reshape for hidden states plus the matching
    ``(B, T_act) -> (B*T_act, 1)`` reshapes of the target ids, mask and (when
    present) the old / reference logprobs. The hidden tensor is made
    ``contiguous`` to mirror :func:`apply_fused_policy_loss`.

    Advantages are intentionally **not** handled here: their pre-flatten shape
    varies by caller (``apply_fused_policy_loss`` carries ``(B, T_act)``, while
    GRPO resolves several shapes to a flat ``(n_tokens,)`` vector), so advantage
    reshaping stays inline at each call site.

    :param policy_hidden: ``(B, T_act, H)`` hidden states sliced to action
        positions.
    :type policy_hidden: torch.Tensor
    :param target_ids: ``(B, T_act)`` next-token target ids.
    :type target_ids: torch.Tensor
    :param mask: ``(B, T_act)`` action-token mask.
    :type mask: torch.Tensor
    :param old_log_probs: ``(B, T_act)`` old-policy logprobs, or ``None``.
    :type old_log_probs: torch.Tensor | None
    :param reference_log_probs: ``(B, T_act)`` reference logprobs, or ``None``.
    :type reference_log_probs: torch.Tensor | None
    :return: ``(hidden, target_ids, mask, old_log_probs, reference_log_probs)``
        all flattened to the token-level layout; the logprob entries stay
        ``None`` when their input was ``None``.
    :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]
    """
    b, t_act, h = policy_hidden.shape
    n_tokens = b * t_act
    return (
        policy_hidden.reshape(n_tokens, 1, h).contiguous(),
        target_ids.reshape(n_tokens, 1),
        mask.reshape(n_tokens, 1),
        old_log_probs.reshape(n_tokens, 1) if old_log_probs is not None else None,
        reference_log_probs.reshape(n_tokens, 1)
        if reference_log_probs is not None
        else None,
    )


def apply_fused_policy_loss(
    policy_hidden: torch.Tensor,
    lm_head_weight: torch.Tensor,
    lm_head_bias: torch.Tensor | None,
    target_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    advantages: torch.Tensor,
    ref_per_token_logps: torch.Tensor | None,
    old_per_token_logps: torch.Tensor | None,
    beta: float,
    epsilon_low: float,
    epsilon_high: float,
    temperature: float,
    importance_sampling_level: str,
    turn_ids: torch.Tensor | None = None,
    full_turn_mask: torch.Tensor | None = None,
    max_turns: int | None = None,
    token_chunk_size: int = 2048,
    turn_log_ratio_reduction: str = "mean",
    vllm_is_ratio: torch.Tensor | None = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """Run :class:`LigerFusedLinearPolicyLossFunction`, bounded at token level.

    For ``importance_sampling_level="token"`` the per-token objective is
    token-independent, so the ``(B, T_act, H)`` hidden states are flattened to
    ``(B*T_act, 1, H)`` and the fused kernel chunks over *tokens* — each chunk
    materializes only ``(token_chunk, vocab)`` logits in both the forward and
    backward, the same memory-bounding trick GRPO/CISPO use. The global
    token-count denominator is unchanged by flattening, so the result is exact.
    ``token_chunk_size`` sets the tokens-per-chunk.

    Turn- and trajectory-level pooling couple a turn/trajectory's tokens, so they
    cannot be token-chunked: a chunk would only see part of the pooled unit.
    Those levels keep the batch path (one sequence per chunk), which holds a
    ``(seq_len, vocab)`` logits tensor per trajectory — not memory-bounded.
    Use the standard (fused-linear-logprob) path for bounded memory there.

    :param policy_hidden: ``(B, T_act, H)`` hidden states already sliced to the
        action positions (caller does the ``[:, :-1]`` shift).
    :type policy_hidden: torch.Tensor
    :param target_ids: ``(B, T_act)`` next-token target ids.
    :type target_ids: torch.Tensor
    :param attention_mask: ``(B, T_act)`` action-token mask.
    :type attention_mask: torch.Tensor
    :param advantages: token level ``(B, T_act)``; turn ``(B, max_turns)``;
        trajectory ``(B, 1)``.
    :type advantages: torch.Tensor
    :param vllm_is_ratio: Optional detached, upper-clamped per-token vLLM
        sampling-mismatch ratio ``(B, T_act)`` (token level only). Token-flattened
        and multiplied into the per-token policy loss before the KL term. Turn /
        trajectory pooling cannot express the per-token reweight, so it is ignored
        there (callers fall back to the standard path for those levels).
    :type vllm_is_ratio: torch.Tensor | None
    :param turn_log_ratio_reduction: Turn-level reduction for pooled log-ratios,
        one of ``"mean"`` or ``"sum"``.
    :type turn_log_ratio_reduction: str
    :return: ``(loss, aux)`` straight from the fused Function.
    :rtype: tuple[torch.Tensor, tuple[torch.Tensor, ...]]
    """
    if importance_sampling_level == "token":
        b, t_act, _ = policy_hidden.shape
        n_tokens = b * t_act
        (
            hidden_flat,
            target_ids_flat,
            mask_flat,
            old_log_probs_flat,
            ref_log_probs_flat,
        ) = flatten_tokens_for_fused_loss(
            policy_hidden,
            target_ids,
            attention_mask,
            old_per_token_logps,
            ref_per_token_logps,
        )
        return LigerFusedLinearPolicyLossFunction.apply(
            hidden_flat,
            lm_head_weight,
            target_ids_flat,
            mask_flat,
            advantages.reshape(n_tokens, 1),
            lm_head_bias,
            ref_log_probs_flat,
            old_log_probs_flat,
            beta,
            epsilon_low,
            epsilon_high,
            temperature,
            False,  # compiled
            token_chunk_size,  # tokens per chunk
            None,  # turn_ids
            None,  # full_turn_mask
            None,  # max_turns
            "token",
            "mean",  # turn_log_ratio_reduction (inert at token level)
            # vllm_is_ratio: token-flattened to match the loss layout above.
            vllm_is_ratio.reshape(n_tokens, 1).contiguous()
            if vllm_is_ratio is not None
            else None,
        )
    return LigerFusedLinearPolicyLossFunction.apply(
        policy_hidden.contiguous(),
        lm_head_weight,
        target_ids,
        attention_mask,
        advantages,
        lm_head_bias,
        ref_per_token_logps,
        old_per_token_logps,
        beta,
        epsilon_low,
        epsilon_high,
        temperature,
        False,  # compiled
        1,  # chunk_size — one sequence per chunk
        turn_ids,
        full_turn_mask,
        max_turns,
        importance_sampling_level,
        turn_log_ratio_reduction,
        None,  # vllm_is_ratio (token level only)
    )


class LigerDPOWithAlpha(LigerFusedLinearPreferenceBase):
    """Thin wrapper that exposes ``alpha`` for NLL scaling.

    ``LigerFusedLinearDPOFunction`` passes ``compute_nll_loss`` as a bool
    but never forwards ``alpha`` to the base class (which defaults to 1.0).
    This subclass reuses the DPO preference loss and adds ``alpha`` so the
    fused kernel correctly scales the NLL component.
    """

    preference_loss_fn = (
        staticmethod(LigerFusedLinearDPOFunction.preference_loss_fn)
        if HAS_LIGER_KERNEL
        else None
    )

    @classmethod
    def forward(  # ty: ignore[invalid-method-override]  # intentionally diverges from Liger's base
        cls,
        ctx: FunctionCtx,
        _input: torch.Tensor,
        weight: torch.Tensor,
        target: torch.Tensor,
        bias: torch.Tensor | None = None,
        ref_input: torch.Tensor | None = None,
        ref_weight: torch.Tensor | None = None,
        ref_bias: torch.Tensor | None = None,
        ignore_index: int = -100,
        beta: float = 0.1,
        alpha: float = 1.0,
        compute_nll_loss: bool = True,
        compiled: bool = True,
        use_ref_model: bool = True,
        average_log_prob: bool = False,
        chunk_size: int = 1,
        loss_type: str = "sigmoid",
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        return LigerFusedLinearPreferenceBase.forward(
            cls=cls,
            ctx=ctx,
            _input=_input,
            weight=weight,
            target=target,
            bias=bias,
            ignore_index=ignore_index,
            alpha=alpha,
            beta=beta,
            compute_nll_loss=compute_nll_loss,
            compiled=compiled,
            use_ref_model=use_ref_model,
            ref_input=ref_input,
            ref_weight=ref_weight,
            ref_bias=ref_bias,
            average_log_prob=average_log_prob,
            chunk_size=chunk_size,
            loss_type=loss_type,
        )

    @staticmethod
    def backward(
        ctx: FunctionCtx,
        *grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor | None, ...]:
        grads = LigerFusedLinearPreferenceBase.backward(ctx, grad_output)[:4]
        return (*grads, *(None,) * 12)

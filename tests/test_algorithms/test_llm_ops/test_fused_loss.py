"""Tests for ``agilerl.algorithms.core.llm_ops.fused_loss``.

``fused_loss`` requires ``liger-kernel`` at import time, so this whole
file is gated: on platforms without Liger, the module-level import-skip
below short-circuits collection and all tests are skipped. The
underlying math (``llm_policy_loss_fn``) is plain-tensor
PyTorch; only the autograd Functions need Liger to run.
"""

from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock, patch

import pytest
import torch

from agilerl import HAS_LIGER_KERNEL

if not HAS_LIGER_KERNEL:
    pytest.skip(
        "fused_loss tests require liger-kernel; skipping on this platform.",
        allow_module_level=True,
    )
from liger_kernel.chunked_loss.fused_linear_preference import (
    LigerFusedLinearPreferenceBase,
)
from agilerl.algorithms.core.llm_ops.fused_loss import (
    llm_policy_loss_fn,
    LigerFusedLinearPolicyLossFunction,
    _LigerDPOWithAlpha,
)


def test_no_liger_fused_module_raises_import_error(monkeypatch):
    """When ``HAS_LIGER_KERNEL=False``, importing ``fused_loss`` raises
    a clear ``ImportError`` rather than silently exposing broken
    autograd Functions. Exercise the guard by patching the flag and
    forcing a fresh import. A failed module body never enters
    ``sys.modules``, so the cached real module survives untouched.
    """
    import agilerl

    monkeypatch.setattr(agilerl, "HAS_LIGER_KERNEL", False)
    mod_name = "agilerl.algorithms.core.llm_ops.fused_loss"
    monkeypatch.delitem(sys.modules, mod_name, raising=False)

    with pytest.raises(ImportError, match="liger-kernel"):
        importlib.import_module(mod_name)


@pytest.mark.parametrize(
    "module_path,symbol",
    [
        ("agilerl.algorithms.grpo", "LigerFusedLinearGRPOFunction"),
        ("agilerl.algorithms.ppo_llm", "LigerFusedLinearPolicyLossFunction"),
        ("agilerl.algorithms.reinforce_llm", "LigerFusedLinearPolicyLossFunction"),
    ],
)
def test_no_liger_fallback_sets_symbol_to_none(module_path: str, symbol: str) -> None:
    """Each LLM-algo module has an ``if HAS_LIGER_KERNEL: from … import
    LigerFusedLinear*; else: LigerFusedLinear* = None`` guard at the top.
    The ``else`` branch is unreachable on Linux CI (Liger installed), so
    exercise it via ``importlib.reload`` with ``HAS_LIGER_KERNEL=False``.

    Using ``reload`` (not ``pop`` + ``import_module``) preserves module
    identity, so other test files' captured class references — e.g.
    ``test_ppo_llm.py``'s ``from agilerl.algorithms.ppo_llm import PPO``
    — continue to point at the same module after this test finishes.
    Restore is handled in the ``finally`` block, not by ``monkeypatch``,
    so the original module state is rebuilt with the original
    ``HAS_LIGER_KERNEL`` regardless of teardown ordering.
    """
    import agilerl

    mod = importlib.import_module(module_path)
    original_has_liger = agilerl.HAS_LIGER_KERNEL
    try:
        agilerl.HAS_LIGER_KERNEL = False
        importlib.reload(mod)
        assert getattr(mod, symbol) is None
    finally:
        agilerl.HAS_LIGER_KERNEL = original_has_liger
        importlib.reload(mod)


def test_llm_policy_loss_fn_with_old_per_token_logps_none_falls_back_to_detached():
    """``llm_policy_loss_fn`` allows ``old_per_token_logps=None`` for callers
    that forgot to pass it — fallback uses the current logprobs detached
    (ratio == 1, gradient still well-defined). Covers the
    ``if old_per_token_logps is None`` branch in the math fn."""
    torch.manual_seed(0)
    B, T, V = 2, 4, 16
    raw = torch.randn(B, T, V, requires_grad=True)
    log_probs = torch.log_softmax(raw, dim=-1)
    target_ids = torch.randint(0, V, (B, T))
    mask = torch.ones(B, T, dtype=torch.float32)
    adv = torch.randn(B, T) * 0.1

    loss_with_none, metrics_with_none = llm_policy_loss_fn(
        log_probs=log_probs,
        selected_token_ids=target_ids,
        attention_mask=mask,
        advantages=adv,
        full_attention_mask=mask,
        old_per_token_logps=None,
        beta=0.0,
    )

    # When old_per_token_logps falls back to per_token_logps.detach(),
    # ratio is exp(0) == 1 everywhere; clipped_ratio is also 1; both
    # max(-adv * ratio, -adv * clipped_ratio) terms equal -adv. Loss
    # reduces to -adv.mean() (all tokens unmasked).
    expected_pg = (-adv).mean()
    assert torch.allclose(loss_with_none, expected_pg, rtol=1e-5, atol=1e-5)
    # clipfrac is zero — ratio == clipped_ratio everywhere
    assert metrics_with_none[1].item() == 0.0
    # Gradient still flows back to the raw input (detach only on the
    # old-policy side; the current logprobs retain grad).
    loss_with_none.backward()
    assert raw.grad is not None


def _unfused_reference(
    log_probs: torch.Tensor,
    target_ids: torch.Tensor,
    mask: torch.Tensor,
    advantages: torch.Tensor,
    old_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor | None,
    epsilon: float,
    beta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """PyTorch reference matching what LLMPPO/LLMREINFORCE compute today."""
    per_token_logps = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(
        -1
    )
    log_ratio = per_token_logps - old_log_probs
    ratio = torch.exp(log_ratio)
    clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    pg_token = torch.max(-advantages * ratio, -advantages * clipped)

    per_token_loss = pg_token
    kl_div = None
    if ref_log_probs is not None:
        kl_div = (
            torch.exp(ref_log_probs - per_token_logps)
            - (ref_log_probs - per_token_logps)
            - 1.0
        )
        if beta != 0.0:
            per_token_loss = per_token_loss + beta * kl_div

    mask_f = mask.float()
    denom = mask_f.sum().clamp(min=1.0)
    loss = (per_token_loss * mask_f).sum() / denom

    metrics = {
        "pg_loss": float(((pg_token * mask_f).sum() / denom).item()),
        "kl": float(((kl_div * mask_f).sum() / denom).item())
        if kl_div is not None
        else 0.0,
        "clipfrac": float((((ratio != clipped).float() * mask_f).sum() / denom).item()),
        "entropy": float(((-per_token_logps.detach() * mask_f).sum() / denom).item()),
    }
    return loss, metrics


class TestLlmPpoLossFn:
    """Math agreement with the existing unfused PPO/REINFORCE token-level path."""

    def test_matches_unfused_with_kl(self) -> None:
        torch.manual_seed(0)
        B, T, V = 4, 8, 100
        log_probs = torch.log_softmax(torch.randn(B, T, V), dim=-1)
        target_ids = torch.randint(0, V, (B, T))
        mask = torch.ones(B, T, dtype=torch.float)
        mask[0, -3:] = 0.0
        adv = torch.randn(B, T) * 0.1
        old_log_probs = torch.randn(B, T) * 0.05
        ref_log_probs = torch.randn(B, T) * 0.05
        eps, beta = 0.2, 0.01

        ref_loss, ref_m = _unfused_reference(
            log_probs, target_ids, mask, adv, old_log_probs, ref_log_probs, eps, beta
        )
        fused_loss, metrics = llm_policy_loss_fn(
            log_probs=log_probs,
            selected_token_ids=target_ids,
            attention_mask=mask,
            advantages=adv,
            full_attention_mask=mask,
            ref_per_token_logps=ref_log_probs,
            old_per_token_logps=old_log_probs,
            epsilon_low=eps,
            epsilon_high=eps,
            beta=beta,
        )
        assert torch.allclose(fused_loss, ref_loss, rtol=1e-6, atol=1e-6)
        kl_m, clip_m, pg_m, ent_m = (m.item() for m in metrics)
        assert abs(kl_m - ref_m["kl"]) < 1e-6
        assert abs(clip_m - ref_m["clipfrac"]) < 1e-6
        assert abs(pg_m - ref_m["pg_loss"]) < 1e-6
        assert abs(ent_m - ref_m["entropy"]) < 1e-6

    def test_matches_unfused_beta_zero_reinforce_style(self) -> None:
        """REINFORCE folds KL into advantages upstream and runs with beta=0.
        KL must still be reported as a metric for monitoring."""
        torch.manual_seed(1)
        B, T, V = 3, 6, 64
        log_probs = torch.log_softmax(torch.randn(B, T, V), dim=-1)
        target_ids = torch.randint(0, V, (B, T))
        mask = torch.ones(B, T, dtype=torch.float)
        adv = torch.randn(B, T) * 0.1
        old_log_probs = torch.randn(B, T) * 0.05
        ref_log_probs = torch.randn(B, T) * 0.05

        ref_loss, ref_m = _unfused_reference(
            log_probs,
            target_ids,
            mask,
            adv,
            old_log_probs,
            ref_log_probs,
            epsilon=0.2,
            beta=0.0,
        )
        fused_loss, metrics = llm_policy_loss_fn(
            log_probs=log_probs,
            selected_token_ids=target_ids,
            attention_mask=mask,
            advantages=adv,
            full_attention_mask=mask,
            ref_per_token_logps=ref_log_probs,
            old_per_token_logps=old_log_probs,
            epsilon_low=0.2,
            epsilon_high=0.2,
            beta=0.0,
        )
        # Loss is pure clipped policy gradient — no KL term added.
        assert torch.allclose(fused_loss, ref_loss, rtol=1e-6, atol=1e-6)
        # KL metric still reported.
        assert metrics[0].item() == pytest.approx(ref_m["kl"], rel=1e-6)

    def test_chunk_accumulation_recovers_global_loss(self) -> None:
        """Splitting the batch and summing chunk losses must equal the
        single-shot loss — this is the invariant Liger's base class relies on
        when accumulating over chunks."""
        torch.manual_seed(2)
        B, T, V = 6, 5, 64
        log_probs = torch.log_softmax(torch.randn(B, T, V), dim=-1)
        target_ids = torch.randint(0, V, (B, T))
        mask = torch.ones(B, T, dtype=torch.float)
        mask[1, -2:] = 0.0  # uneven masking across samples
        adv = torch.randn(B, T) * 0.1
        old_log_probs = torch.randn(B, T) * 0.05
        ref_log_probs = torch.randn(B, T) * 0.05

        single, _ = llm_policy_loss_fn(
            log_probs=log_probs,
            selected_token_ids=target_ids,
            attention_mask=mask,
            advantages=adv,
            full_attention_mask=mask,
            ref_per_token_logps=ref_log_probs,
            old_per_token_logps=old_log_probs,
            beta=0.01,
        )

        # Split into 3 chunks of 2 samples each.
        chunk_sum = torch.zeros((), dtype=single.dtype)
        for s, e in [(0, 2), (2, 4), (4, 6)]:
            chunk_loss, _ = llm_policy_loss_fn(
                log_probs=log_probs[s:e],
                selected_token_ids=target_ids[s:e],
                attention_mask=mask[s:e],
                advantages=adv[s:e],
                full_attention_mask=mask,  # <- GLOBAL mask is the denominator
                ref_per_token_logps=ref_log_probs[s:e],
                old_per_token_logps=old_log_probs[s:e],
                beta=0.01,
            )
            chunk_sum = chunk_sum + chunk_loss

        assert torch.allclose(single, chunk_sum, rtol=1e-5, atol=1e-5)

    def test_no_ref_logprobs_zero_kl_metric(self) -> None:
        """If no reference is provided, KL metric is zero and loss is pure PG."""
        torch.manual_seed(3)
        B, T, V = 2, 4, 32
        log_probs = torch.log_softmax(torch.randn(B, T, V), dim=-1)
        target_ids = torch.randint(0, V, (B, T))
        mask = torch.ones(B, T, dtype=torch.float)
        adv = torch.randn(B, T) * 0.1
        old_log_probs = torch.randn(B, T) * 0.05

        fused_loss, metrics = llm_policy_loss_fn(
            log_probs=log_probs,
            selected_token_ids=target_ids,
            attention_mask=mask,
            advantages=adv,
            full_attention_mask=mask,
            ref_per_token_logps=None,
            old_per_token_logps=old_log_probs,
            beta=0.0,
        )
        assert metrics[0].item() == 0.0  # kl
        # Loss should equal pg_loss metric (since beta=0).
        assert torch.allclose(fused_loss, metrics[2], rtol=1e-6)

    def test_clip_engaged_reports_nonzero_clipfrac(self) -> None:
        """When ratios force clipping, clipfrac must be > 0 (sanity check on
        the metric path)."""
        torch.manual_seed(4)
        B, T, V = 2, 4, 16
        log_probs = torch.log_softmax(torch.randn(B, T, V), dim=-1)
        target_ids = torch.randint(0, V, (B, T))
        mask = torch.ones(B, T, dtype=torch.float)
        adv = torch.randn(B, T)
        # Force ratio far from 1 by making old_log_probs much smaller than
        # current → ratio = exp(big) gets clipped at 1 + eps.
        per_token_logps = log_probs.gather(
            dim=-1, index=target_ids.unsqueeze(-1)
        ).squeeze(-1)
        old_log_probs = per_token_logps.detach() - 1.0  # ratio ≈ e ≫ 1.2

        _, metrics = llm_policy_loss_fn(
            log_probs=log_probs,
            selected_token_ids=target_ids,
            attention_mask=mask,
            advantages=adv,
            full_attention_mask=mask,
            old_per_token_logps=old_log_probs,
            epsilon_low=0.2,
            epsilon_high=0.2,
            beta=0.0,
        )
        assert metrics[1].item() > 0.0  # clipfrac


def _unfused_turn_reference(
    log_probs: torch.Tensor,
    target_ids: torch.Tensor,
    token_mask: torch.Tensor,
    turn_ids: torch.Tensor,
    turn_advantages: torch.Tensor,
    turn_mask: torch.Tensor,
    old_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor | None,
    epsilon: float,
    beta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """PyTorch reference matching unfused turn-PPO with ``turn_level_clip=True``.

    Sums token log-ratios per turn, clips at the turn level, max-formula
    on per-turn quantities, mask-mean reduction over the turn mask. KL
    stays token-level (matches PPO's existing convention).
    """
    per_token_logps = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(
        -1
    )
    token_log_ratio = per_token_logps - old_log_probs
    # Sum-pool token log-ratios into per-turn log-ratios.
    masked_token_log_ratio = token_log_ratio * token_mask.float()
    safe_turn_ids = turn_ids.clamp(min=0)
    B, T = per_token_logps.shape
    max_turns = turn_mask.shape[1]
    turn_log_ratio = torch.zeros(B, max_turns, dtype=token_log_ratio.dtype)
    turn_log_ratio.scatter_add_(1, safe_turn_ids, masked_token_log_ratio)

    ratio = torch.exp(turn_log_ratio)
    clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    pg_turn = torch.max(-turn_advantages * ratio, -turn_advantages * clipped)
    turn_mask_f = turn_mask.float()
    turn_denom = turn_mask_f.sum().clamp(min=1.0)
    loss = (pg_turn * turn_mask_f).sum() / turn_denom

    if ref_log_probs is not None:
        kl_div = (
            torch.exp(ref_log_probs - per_token_logps)
            - (ref_log_probs - per_token_logps)
            - 1.0
        )
        token_denom = token_mask.float().sum().clamp(min=1.0)
        if beta != 0.0:
            loss = loss + beta * ((kl_div * token_mask.float()).sum() / token_denom)
        kl_metric = float(((kl_div * token_mask.float()).sum() / token_denom).item())
    else:
        kl_metric = 0.0

    return loss, {
        "kl": kl_metric,
        "clipfrac": float(
            (((ratio != clipped).float() * turn_mask_f).sum() / turn_denom).item()
        ),
        "pg_loss": float(((pg_turn * turn_mask_f).sum() / turn_denom).item()),
    }


class TestLlmPpoLossFnTurnMode:
    """Turn-mode loss must match the unfused turn-PPO path under
    ``turn_level_clip=True`` (sum-pool log-ratios, clip at turn level)
    and preserve the chunk-accumulation invariant.
    """

    @staticmethod
    def _build_turn_inputs(B: int, T: int, V: int, max_turns: int, seed: int):
        """Random batch where each sample has all max_turns active turns
        contiguously distributed over T tokens (deterministic, simple)."""
        torch.manual_seed(seed)
        log_probs = torch.log_softmax(torch.randn(B, T, V), dim=-1)
        target_ids = torch.randint(0, V, (B, T))
        # Assign turn ids: bucket the T tokens into max_turns roughly-equal slices.
        turn_ids = torch.zeros(B, T, dtype=torch.long)
        bucket_size = T // max_turns
        for t in range(max_turns):
            turn_ids[:, t * bucket_size : (t + 1) * bucket_size] = t
        # Token-level action mask — fully active in this fixture.
        token_mask = torch.ones(B, T, dtype=torch.float)
        # Per-turn advantages and turn mask (all turns present).
        turn_advantages = torch.randn(B, max_turns) * 0.1
        turn_mask = torch.ones(B, max_turns, dtype=torch.float)
        old_log_probs = torch.randn(B, T) * 0.05
        ref_log_probs = torch.randn(B, T) * 0.05
        return (
            log_probs,
            target_ids,
            token_mask,
            turn_ids,
            turn_advantages,
            turn_mask,
            old_log_probs,
            ref_log_probs,
        )

    def test_matches_unfused_turn_ppo_with_kl(self) -> None:
        B, T, V, max_turns = 4, 12, 64, 4
        (
            log_probs,
            target_ids,
            token_mask,
            turn_ids,
            turn_adv,
            turn_mask,
            old_lp,
            ref_lp,
        ) = self._build_turn_inputs(B, T, V, max_turns, seed=10)
        eps, beta = 0.2, 0.01

        ref_loss, ref_m = _unfused_turn_reference(
            log_probs,
            target_ids,
            token_mask,
            turn_ids,
            turn_adv,
            turn_mask,
            old_lp,
            ref_lp,
            eps,
            beta,
        )
        fused_loss, metrics = llm_policy_loss_fn(
            log_probs=log_probs,
            selected_token_ids=target_ids,
            attention_mask=token_mask,
            advantages=turn_adv,
            full_attention_mask=token_mask,
            ref_per_token_logps=ref_lp,
            old_per_token_logps=old_lp,
            epsilon_low=eps,
            epsilon_high=eps,
            beta=beta,
            turn_ids=turn_ids,
            full_turn_mask=turn_mask,
            max_turns=max_turns,
        )
        assert torch.allclose(fused_loss, ref_loss, rtol=1e-6, atol=1e-6)
        assert metrics[0].item() == pytest.approx(ref_m["kl"], rel=1e-6, abs=1e-6)
        assert metrics[1].item() == pytest.approx(ref_m["clipfrac"], abs=1e-6)
        assert metrics[2].item() == pytest.approx(ref_m["pg_loss"], rel=1e-6, abs=1e-6)

    def test_chunk_accumulation_recovers_global_loss_turn_mode(self) -> None:
        """Splitting along B and summing chunk losses must equal the
        single-shot turn-mode loss — Liger's chunk loop relies on this."""
        B, T, V, max_turns = 6, 12, 64, 4
        (
            log_probs,
            target_ids,
            token_mask,
            turn_ids,
            turn_adv,
            turn_mask,
            old_lp,
            ref_lp,
        ) = self._build_turn_inputs(B, T, V, max_turns, seed=11)

        single, _ = llm_policy_loss_fn(
            log_probs=log_probs,
            selected_token_ids=target_ids,
            attention_mask=token_mask,
            advantages=turn_adv,
            full_attention_mask=token_mask,
            ref_per_token_logps=ref_lp,
            old_per_token_logps=old_lp,
            beta=0.01,
            turn_ids=turn_ids,
            full_turn_mask=turn_mask,
            max_turns=max_turns,
        )
        chunked_total = torch.zeros((), dtype=single.dtype)
        for s, e in [(0, 2), (2, 4), (4, 6)]:
            chunk_loss, _ = llm_policy_loss_fn(
                log_probs=log_probs[s:e],
                selected_token_ids=target_ids[s:e],
                attention_mask=token_mask[s:e],
                advantages=turn_adv[s:e],
                full_attention_mask=token_mask,  # global denom
                ref_per_token_logps=ref_lp[s:e],
                old_per_token_logps=old_lp[s:e],
                beta=0.01,
                turn_ids=turn_ids[s:e],
                full_turn_mask=turn_mask,  # global turn denom
                max_turns=max_turns,
            )
            chunked_total = chunked_total + chunk_loss
        assert torch.allclose(single, chunked_total, rtol=1e-5, atol=1e-5)

    def test_uneven_turns_per_sample(self) -> None:
        """Samples with fewer than ``max_turns`` active turns must
        contribute only their active turns to the reduction (uses
        ``full_turn_mask`` to gate)."""
        torch.manual_seed(12)
        B, T, V, max_turns = 3, 9, 32, 3
        log_probs = torch.log_softmax(torch.randn(B, T, V), dim=-1)
        target_ids = torch.randint(0, V, (B, T))
        # Sample 0: 3 turns. Sample 1: 2 turns. Sample 2: 1 turn.
        turn_ids = torch.tensor(
            [
                [0, 0, 0, 1, 1, 1, 2, 2, 2],
                [0, 0, 0, 0, 1, 1, 1, 1, -1],
                [0, 0, 0, 0, 0, 0, 0, 0, -1],
            ]
        )
        token_mask = (turn_ids >= 0).float()
        turn_mask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        turn_adv = torch.randn(B, max_turns) * 0.1
        old_lp = torch.randn(B, T) * 0.05
        ref_lp = torch.randn(B, T) * 0.05

        ref_loss, _ = _unfused_turn_reference(
            log_probs,
            target_ids,
            token_mask,
            turn_ids,
            turn_adv,
            turn_mask,
            old_lp,
            ref_lp,
            epsilon=0.2,
            beta=0.01,
        )
        fused_loss, _ = llm_policy_loss_fn(
            log_probs=log_probs,
            selected_token_ids=target_ids,
            attention_mask=token_mask,
            advantages=turn_adv,
            full_attention_mask=token_mask,
            ref_per_token_logps=ref_lp,
            old_per_token_logps=old_lp,
            epsilon_low=0.2,
            epsilon_high=0.2,
            beta=0.01,
            turn_ids=turn_ids,
            full_turn_mask=turn_mask,
            max_turns=max_turns,
        )
        assert torch.allclose(fused_loss, ref_loss, rtol=1e-6, atol=1e-6)

    def test_raises_when_turn_args_incomplete(self) -> None:
        """Passing ``turn_ids`` without ``max_turns`` / ``full_turn_mask``
        is a programming error and should raise."""
        log_probs = torch.log_softmax(torch.randn(2, 4, 16), dim=-1)
        with pytest.raises(ValueError, match="turn-mode"):
            llm_policy_loss_fn(
                log_probs=log_probs,
                selected_token_ids=torch.randint(0, 16, (2, 4)),
                attention_mask=torch.ones(2, 4),
                advantages=torch.randn(2, 2) * 0.1,
                full_attention_mask=torch.ones(2, 4),
                old_per_token_logps=torch.randn(2, 4) * 0.05,
                turn_ids=torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]]),
                # full_turn_mask + max_turns missing
            )


class TestLigerFusedLinearPolicyLossFunction:
    """Drive the autograd Function on tiny shapes and assert it matches the
    unfused reference for both forward loss and backward gradient flow."""

    @staticmethod
    def _build_inputs(B, T, V, H, *, with_ref):
        torch.manual_seed(0)
        hidden = torch.randn(B, T, H, dtype=torch.float32, requires_grad=True)
        # Scale before setting requires_grad — otherwise the multiply
        # makes ``weight`` a non-leaf tensor and ``weight.grad`` won't
        # populate after backward().
        weight = (torch.randn(V, H, dtype=torch.float32) * 0.02).requires_grad_(True)
        target_ids = torch.randint(0, V, (B, T))
        mask = torch.ones(B, T, dtype=torch.float32)
        adv = torch.randn(B, T, dtype=torch.float32) * 0.1
        old_lp = torch.randn(B, T, dtype=torch.float32) * 0.05
        ref_lp = torch.randn(B, T, dtype=torch.float32) * 0.05 if with_ref else None
        return hidden, weight, target_ids, mask, adv, old_lp, ref_lp

    def test_forward_matches_unfused_loss_token_mode(self) -> None:
        B, T, V, H = 2, 5, 32, 8
        hidden, weight, ids, mask, adv, old_lp, ref_lp = self._build_inputs(
            B, T, V, H, with_ref=True
        )
        loss, _ = LigerFusedLinearPolicyLossFunction.apply(
            hidden,
            weight,
            ids,
            mask,
            adv,
            None,  # bias
            ref_lp,
            old_lp,
            0.01,  # beta
            0.2,
            0.2,  # epsilon_low, epsilon_high
            1.0,  # temperature
            False,  # compiled
            1,  # chunk_size
            None,
            None,
            None,  # turn_ids, full_turn_mask, max_turns
        )
        # Reference: compute loss the unfused way.
        logits = hidden @ weight.t()
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        ref_loss, _ = llm_policy_loss_fn(
            log_probs=log_probs,
            selected_token_ids=ids,
            attention_mask=mask,
            advantages=adv,
            full_attention_mask=mask,
            ref_per_token_logps=ref_lp,
            old_per_token_logps=old_lp,
            beta=0.01,
        )
        assert torch.allclose(loss, ref_loss, rtol=1e-4, atol=1e-4)

    def test_backward_gradients_flow_to_hidden_and_weight(self) -> None:
        B, T, V, H = 2, 4, 16, 8
        hidden, weight, ids, mask, adv, old_lp, _ = self._build_inputs(
            B, T, V, H, with_ref=False
        )
        loss, _ = LigerFusedLinearPolicyLossFunction.apply(
            hidden,
            weight,
            ids,
            mask,
            adv,
            None,
            None,
            old_lp,
            0.0,
            0.2,
            0.2,
            1.0,
            False,
            1,
            None,
            None,
            None,
        )
        loss.backward()
        assert hidden.grad is not None
        assert hidden.grad.shape == hidden.shape
        assert weight.grad is not None
        assert weight.grad.shape == weight.shape
        # Gradients should be non-zero (with the random advantages we set up).
        assert hidden.grad.abs().sum() > 0
        assert weight.grad.abs().sum() > 0

    def test_turn_mode_forward_runs(self) -> None:
        """Smoke test for the turn-mode branch in the autograd Function."""
        B, T, V, H = 2, 6, 16, 8
        hidden, weight, ids, mask, _, old_lp, ref_lp = self._build_inputs(
            B, T, V, H, with_ref=True
        )
        max_turns = 2
        turn_ids = torch.tensor([[0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 1, 1]])
        full_turn_mask = torch.ones(B, max_turns, dtype=torch.float32)
        adv_turn = torch.randn(B, max_turns, dtype=torch.float32) * 0.1
        loss, _ = LigerFusedLinearPolicyLossFunction.apply(
            hidden,
            weight,
            ids,
            mask,
            adv_turn,
            None,
            ref_lp,
            old_lp,
            0.0,
            0.2,
            0.2,
            1.0,
            False,
            1,
            turn_ids,
            full_turn_mask,
            max_turns,
        )
        loss.backward()
        assert hidden.grad is not None
        assert weight.grad is not None

    def test_backward_with_bias_accumulates_grad_bias(self) -> None:
        """When the lm_head has a bias, the chunked backward must
        accumulate ``grad_bias`` alongside ``grad_input`` and
        ``grad_weight``. Covers the ``if grad_bias is not None`` branch
        in the chunk-accumulator."""
        B, T, V, H = 2, 4, 16, 8
        hidden, weight, ids, mask, adv, old_lp, _ = self._build_inputs(
            B, T, V, H, with_ref=False
        )
        bias = (torch.randn(V, dtype=torch.float32) * 0.02).requires_grad_(True)
        loss, _ = LigerFusedLinearPolicyLossFunction.apply(
            hidden,
            weight,
            ids,
            mask,
            adv,
            bias,
            None,
            old_lp,
            0.0,
            0.2,
            0.2,
            1.0,
            False,
            1,
            None,
            None,
            None,
        )
        loss.backward()
        assert bias.grad is not None
        assert bias.grad.shape == bias.shape
        # With random advantages the bias gradient should be non-zero.
        assert bias.grad.abs().sum() > 0


class TestLigerDPOWithAlphaBackward:
    def test_liger_dpo_with_alpha_backward_returns_sixteen_outputs_with_trailing_nones(
        self,
    ) -> None:
        """``_LigerDPOWithAlpha.backward`` forwards to the base, keeps four grads, pads twelve ``None``."""

        def fake_parent_backward(ctx, grad_output):
            return tuple(range(16))

        with patch.object(
            LigerFusedLinearPreferenceBase,
            "backward",
            staticmethod(fake_parent_backward),
        ):
            out = _LigerDPOWithAlpha.backward(MagicMock(), torch.tensor(1.0))

        assert len(out) == 16
        assert out[:4] == (0, 1, 2, 3)
        assert out[4:] == (None,) * 12

# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for LLMPPO's per-token bound on the K3 KL penalty (``kl_clamp``).

Pure CPU on a tiny real model in fp32. The fused-path counterparts live in
``tests/test_algorithms/test_llm_ops/test_fused_loss.py``.
"""

from __future__ import annotations

import math

import pytest
import torch

pytest.importorskip("transformers", reason="LLM tests require transformers.")
pytest.importorskip("peft", reason="LLM tests require peft.")

from agilerl.algorithms.ppo_llm import PPO as LLMPPO
from agilerl.utils.llm_utils import calculate_k3_kl, masked_mean
from tests.test_algorithms.test_llms.test_ppo_llm_segments import _make_ppo

BETA = 0.05
# Policy log-prob this far below the reference: k3 = exp(11) - 12, about 6e4.
GAP = 11.0


def _inputs() -> dict[str, torch.Tensor]:
    """Two rows of four action tokens, every reference within 0.3 nats."""
    generator = torch.Generator().manual_seed(0)
    log_probs = -1.0 + 0.1 * torch.randn(2, 4, generator=generator)
    return {
        "log_probs": log_probs,
        "old": log_probs + 0.05 * torch.randn(2, 4, generator=generator),
        "reference": log_probs + 0.3 * torch.randn(2, 4, generator=generator),
        "advantages": 0.1 * torch.randn(2, 4, generator=generator),
        "mask": torch.ones(2, 4),
    }


def _with_gap(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """``inputs`` with token (0, 1) sitting ``GAP`` nats below its reference."""
    reference = inputs["reference"].clone()
    reference[0, 1] = inputs["log_probs"][0, 1] + GAP
    return {**inputs, "reference": reference}


def _loss_and_grad(
    agent: LLMPPO, inputs: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
    """Standard-path policy loss, metrics and gradient with respect to the log-probs."""
    log_probs = inputs["log_probs"].clone().requires_grad_(True)
    loss, metrics = agent._ppo_policy_loss(
        log_probs,
        inputs["mask"],
        inputs["old"],
        inputs["reference"],
        inputs["advantages"],
        torch.zeros(2, 4, dtype=torch.long),
        1,
        "token",
    )
    loss.backward()
    assert log_probs.grad is not None
    return loss.detach(), metrics, log_probs.grad


class TestLLMPPOKlClampConfig:
    def test_defaults_to_ten(self) -> None:
        agent = _make_ppo()

        assert agent.kl_clamp == 10.0

    @pytest.mark.parametrize("kl_clamp", [0.0, -1.0])
    def test_a_non_positive_bound_is_rejected(self, kl_clamp: float) -> None:
        with pytest.raises(ValueError, match="kl_clamp must be > 0 or None"):
            _make_ppo(kl_clamp=kl_clamp)


class TestLLMPPOPolicyLossKlClamp:
    def test_tokens_within_the_bound_are_unchanged(self) -> None:
        # Arrange
        clamped = _make_ppo(beta=BETA)
        unclamped = _make_ppo(beta=BETA, kl_clamp=None)
        inputs = _inputs()

        # Act
        clamped_loss, clamped_metrics, clamped_grad = _loss_and_grad(clamped, inputs)
        loss, metrics, grad = _loss_and_grad(unclamped, inputs)

        # Assert
        assert torch.equal(clamped_loss, loss)
        assert torch.equal(clamped_grad, grad)
        assert clamped_metrics["kl"] == metrics["kl"]
        assert clamped_metrics["kl_clamp_frac"] == 0.0

    def test_a_token_past_the_bound_gets_no_kl_gradient(self) -> None:
        # Arrange
        clamped = _make_ppo(beta=BETA)
        unclamped = _make_ppo(beta=BETA, kl_clamp=None)
        policy_only = _make_ppo(beta=0.0)
        inputs = _with_gap(_inputs())

        # Act
        _, _, clamped_grad = _loss_and_grad(clamped, inputs)
        _, _, unclamped_grad = _loss_and_grad(unclamped, inputs)
        _, _, policy_grad = _loss_and_grad(policy_only, inputs)

        # Assert
        assert clamped_grad[0, 1] == pytest.approx(float(policy_grad[0, 1]), rel=1e-6)
        assert abs(float(unclamped_grad[0, 1])) > 100.0
        others = torch.ones(2, 4, dtype=torch.bool)
        others[0, 1] = False
        assert torch.allclose(
            clamped_grad[others], unclamped_grad[others], rtol=1e-6, atol=0.0
        )

    def test_a_policy_far_above_the_reference_keeps_its_kl_gradient(self) -> None:
        # Arrange: token (0, 1) sits 14 nats above its reference, k3 about 13.
        clamped = _make_ppo(beta=BETA)
        unclamped = _make_ppo(beta=BETA, kl_clamp=None)
        policy_only = _make_ppo(beta=0.0)
        inputs = _inputs()
        reference = inputs["reference"].clone()
        reference[0, 1] = inputs["log_probs"][0, 1] - 14.0
        inputs = {**inputs, "reference": reference}

        # Act
        clamped_loss, clamped_metrics, clamped_grad = _loss_and_grad(clamped, inputs)
        loss, _, grad = _loss_and_grad(unclamped, inputs)
        _, _, policy_grad = _loss_and_grad(policy_only, inputs)

        # Assert
        assert torch.equal(clamped_loss, loss)
        assert torch.equal(clamped_grad, grad)
        assert not torch.isclose(clamped_grad[0, 1], policy_grad[0, 1])
        assert clamped_metrics["kl_clamp_frac"] == 0.0

    def test_grad_norm_stays_at_the_normal_scale(self) -> None:
        # Arrange
        clamped = _make_ppo(beta=BETA)
        unclamped = _make_ppo(beta=BETA, kl_clamp=None)
        inputs = _inputs()
        gapped = _with_gap(inputs)

        # Act
        normal_norm = _loss_and_grad(clamped, inputs)[2].norm()
        clamped_norm = _loss_and_grad(clamped, gapped)[2].norm()
        unclamped_norm = _loss_and_grad(unclamped, gapped)[2].norm()

        # Assert
        assert clamped_norm < 2 * normal_norm
        assert unclamped_norm > 1000 * normal_norm

    def test_metrics_report_the_unbounded_kl_and_the_clamped_share(self) -> None:
        # Arrange
        agent = _make_ppo(beta=BETA)
        inputs = _with_gap(_inputs())
        expected_kl = masked_mean(
            calculate_k3_kl(inputs["reference"], inputs["log_probs"]), inputs["mask"]
        )

        # Act
        _, metrics, _ = _loss_and_grad(agent, inputs)

        # Assert
        assert metrics["kl"] == pytest.approx(float(expected_kl), rel=1e-6)
        assert metrics["kl"] > (math.exp(GAP) - GAP - 1) / 8
        assert metrics["kl_clamp_frac"] == pytest.approx(1 / 8)

# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for GRPO's per-token bound on the K3 KL penalty (``kl_clamp``).

Pure CPU on a tiny real model in fp32. The parity class runs the real Liger
kernel, which is installed on Linux only.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch

pytest.importorskip("transformers", reason="LLM tests require transformers.")
pytest.importorskip("peft", reason="LLM tests require peft.")

from agilerl import HAS_LIGER_KERNEL
from agilerl.algorithms import grpo as grpo_module
from agilerl.algorithms.core.llm_ops.fused_logprobs import scored_position_index
from agilerl.utils.llm_utils import calculate_k3_kl, masked_mean
from tests.test_algorithms.test_llms.test_grpo_episode_loss_norm import KERNEL_NAME
from tests.test_algorithms.test_llms.test_grpo_off_policy_masks import (
    ICEPOP_BAND,
    RecordingFusedKernel,
)
from tests.test_algorithms.test_llms.test_grpo_old_logprobs import (
    NUM_ROWS,
    PROMPT_LEN,
    _batch,
    _make_grpo,
)

BETA = 0.05
# Policy log-prob this far below the reference: k3 = exp(11) - 12, about 6e4.
GAP = 11.0
REFERENCE_ARG = 6
FIRST_ACTION = PROMPT_LEN - 1


class RecordingKlKernel(RecordingFusedKernel):
    """Recording probe kernel with the ``[kl, clipfrac]`` aux of a KL coefficient."""

    @classmethod
    def apply(cls, *args):
        """Record the call, run the probe loss and prepend a zero KL."""
        loss, aux = super().apply(*args)
        return loss, (torch.zeros(()), *aux)


def _inputs() -> dict[str, torch.Tensor]:
    """Two rows of four action tokens, every reference within 0.3 nats."""
    generator = torch.Generator().manual_seed(0)
    log_probs = -1.0 + 0.1 * torch.randn(2, 4, generator=generator)
    return {
        "log_probs": log_probs,
        "old": log_probs + 0.05 * torch.randn(2, 4, generator=generator),
        "reference": log_probs + 0.3 * torch.randn(2, 4, generator=generator),
        "advantages": torch.tensor([[1.0], [-0.5]]),
        "mask": torch.ones(2, 4, dtype=torch.bool),
    }


def _loss_and_grad(
    agent: grpo_module.GRPO,
    inputs: dict[str, torch.Tensor],
    objective: str = "cispo",
    level: str = "token",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Standard-path loss, KL metric and gradient with respect to the log-probs."""
    log_probs = inputs["log_probs"].clone().requires_grad_(True)
    loss, kl, _clipfrac = agent._compute_policy_loss(
        inputs["mask"],
        log_probs,
        inputs["old"],
        inputs["reference"],
        inputs["advantages"],
        None,
        level,
        objective,
    )
    loss.backward()
    assert log_probs.grad is not None
    return loss.detach(), kl.detach(), log_probs.grad


def _with_gap(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """``inputs`` with token (0, 1) sitting ``GAP`` nats below its reference."""
    reference = inputs["reference"].clone()
    reference[0, 1] = inputs["log_probs"][0, 1] + GAP
    return {**inputs, "reference": reference}


class TestGRPOKlClampConfig:
    def test_defaults_to_ten(self) -> None:
        agent = _make_grpo()

        assert agent.kl_clamp == 10.0

    def test_none_disables_the_bound(self) -> None:
        agent = _make_grpo(kl_clamp=None)

        assert agent.kl_clamp is None

    @pytest.mark.parametrize("kl_clamp", [0.0, -1.0])
    def test_a_non_positive_bound_is_rejected(self, kl_clamp: float) -> None:
        with pytest.raises(ValueError, match="kl_clamp must be > 0 or None"):
            _make_grpo(kl_clamp=kl_clamp)


class TestGRPOComputePolicyLossKlClamp:
    @pytest.mark.parametrize(
        ("loss_type", "objective", "level"),
        [
            ("cispo", "cispo", "token"),
            ("grpo", "grpo", "token"),
            ("gspo", "grpo", "trajectory"),
        ],
    )
    def test_tokens_within_the_bound_are_unchanged(
        self, loss_type: str, objective: str, level: str
    ) -> None:
        # Arrange
        clamped = _make_grpo(loss_norm="micro_batch", loss_type=loss_type, beta=BETA)
        unclamped = _make_grpo(
            loss_norm="micro_batch", loss_type=loss_type, beta=BETA, kl_clamp=None
        )
        inputs = _inputs()

        # Act
        clamped_loss, clamped_kl, clamped_grad = _loss_and_grad(
            clamped, inputs, objective, level
        )
        loss, kl, grad = _loss_and_grad(unclamped, inputs, objective, level)

        # Assert
        assert torch.equal(clamped_loss, loss)
        assert torch.equal(clamped_kl, kl)
        assert torch.equal(clamped_grad, grad)

    @pytest.mark.parametrize("use_bias_correction_kl", [False, True])
    def test_a_token_past_the_bound_gets_no_kl_gradient(
        self, use_bias_correction_kl: bool
    ) -> None:
        # Arrange
        clamped = _make_grpo(
            loss_norm="micro_batch",
            loss_type="cispo",
            beta=BETA,
            use_bias_correction_kl=use_bias_correction_kl,
        )
        unclamped = _make_grpo(
            loss_norm="micro_batch",
            loss_type="cispo",
            beta=BETA,
            use_bias_correction_kl=use_bias_correction_kl,
            kl_clamp=None,
        )
        policy_only = _make_grpo(loss_norm="micro_batch", loss_type="cispo")
        inputs = _with_gap(_inputs())

        # Act
        _, _, clamped_grad = _loss_and_grad(clamped, inputs)
        _, _, unclamped_grad = _loss_and_grad(unclamped, inputs)
        _, _, policy_grad = _loss_and_grad(policy_only, inputs)

        # Assert: the gapped token keeps only its policy gradient; the others
        # keep their KL gradient.
        assert clamped_grad[0, 1] == pytest.approx(float(policy_grad[0, 1]), rel=1e-6)
        assert not torch.isclose(unclamped_grad[0, 1], clamped_grad[0, 1])
        others = torch.ones(2, 4, dtype=torch.bool)
        others[0, 1] = False
        assert torch.allclose(
            clamped_grad[others], unclamped_grad[others], rtol=1e-6, atol=0.0
        )

    def test_a_policy_far_above_the_reference_keeps_its_kl_gradient(self) -> None:
        # Arrange: token (0, 1) sits 14 nats above its reference, k3 about 13.
        clamped = _make_grpo(loss_norm="micro_batch", loss_type="cispo", beta=BETA)
        unclamped = _make_grpo(
            loss_norm="micro_batch", loss_type="cispo", beta=BETA, kl_clamp=None
        )
        policy_only = _make_grpo(loss_norm="micro_batch", loss_type="cispo")
        inputs = _inputs()
        reference = inputs["reference"].clone()
        reference[0, 1] = inputs["log_probs"][0, 1] - 14.0
        inputs = {**inputs, "reference": reference}

        # Act
        clamped_loss, _, clamped_grad = _loss_and_grad(clamped, inputs)
        loss, _, grad = _loss_and_grad(unclamped, inputs)
        _, _, policy_grad = _loss_and_grad(policy_only, inputs)

        # Assert
        assert torch.equal(clamped_loss, loss)
        assert torch.equal(clamped_grad, grad)
        assert not torch.isclose(clamped_grad[0, 1], policy_grad[0, 1])

    def test_grad_norm_stays_at_the_normal_scale(self) -> None:
        # Arrange: the plain K3 gradient grows as exp(GAP); the bias-corrected
        # one grows linearly in GAP.
        clamped = _make_grpo(
            loss_norm="micro_batch",
            loss_type="cispo",
            beta=BETA,
            use_bias_correction_kl=False,
        )
        unclamped = _make_grpo(
            loss_norm="micro_batch",
            loss_type="cispo",
            beta=BETA,
            use_bias_correction_kl=False,
            kl_clamp=None,
        )
        inputs = _inputs()
        gapped = _with_gap(inputs)

        # Act
        normal_norm = _loss_and_grad(clamped, inputs)[2].norm()
        clamped_norm = _loss_and_grad(clamped, gapped)[2].norm()
        unclamped_norm = _loss_and_grad(unclamped, gapped)[2].norm()

        # Assert
        assert clamped_norm < 2 * normal_norm
        assert unclamped_norm > 1000 * normal_norm

    def test_kl_metric_stays_unbounded(self) -> None:
        # Arrange
        agent = _make_grpo(
            loss_norm="micro_batch",
            loss_type="cispo",
            beta=BETA,
            use_bias_correction_kl=False,
        )
        inputs = _with_gap(_inputs())
        expected = masked_mean(
            calculate_k3_kl(inputs["reference"], inputs["log_probs"]), inputs["mask"]
        )

        # Act
        _, kl, _ = _loss_and_grad(agent, inputs)

        # Assert
        assert float(kl) == pytest.approx(float(expected), rel=1e-6)
        assert float(kl) > (math.exp(GAP) - GAP - 1) / 8

    def test_kl_advantage_shaping_uses_the_bound_and_skips_dropped_tokens(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange: token (0, 1) sits GAP nats below its reference and token
        # (1, 0) is dropped by the band (ratio exp(-1) < 0.5).
        agent = _make_grpo(
            loss_norm="micro_batch",
            loss_type="cispo",
            beta=BETA,
            use_kl_advantage_shaping=True,
            off_policy_token_mask_bounds=ICEPOP_BAND,
            off_policy_sequence_mask_threshold=None,
        )
        inputs = _with_gap(_inputs())
        old = inputs["old"].clone()
        old[1, 0] = inputs["log_probs"][1, 0] + 1.0
        log_probs = inputs["log_probs"].clone().requires_grad_(True)
        shaped: list[torch.Tensor] = []
        shape = agent._apply_kl_advantage_shaping

        def record_shaping(*args: torch.Tensor) -> torch.Tensor:
            shaped.append(shape(*args))
            return shaped[-1]

        monkeypatch.setattr(agent, "_apply_kl_advantage_shaping", record_shaping)
        k3 = calculate_k3_kl(inputs["reference"], inputs["log_probs"])
        row0_kl = k3[0].clone()
        row0_kl[1] = agent.kl_clamp
        row1_kl = k3[1, 1:]
        expected = torch.empty(2, 4)
        expected[0] = 1.0 + BETA * (row0_kl.mean() - row0_kl)
        expected[1, 0] = -0.5 + BETA * row1_kl.mean()
        expected[1, 1:] = -0.5 + BETA * (row1_kl.mean() - row1_kl)

        # Act
        agent._compute_policy_loss(
            inputs["mask"],
            log_probs,
            old,
            inputs["reference"],
            inputs["advantages"],
            None,
            "token",
            "cispo",
        )
        (advantages,) = shaped
        (grad,) = torch.autograd.grad(advantages.sum(), log_probs)

        # Assert
        assert torch.allclose(advantages.detach(), expected, rtol=1e-6, atol=1e-7)
        assert advantages[0, 1] < 1.0
        assert bool((advantages[0, [0, 2, 3]] > 1.0).all())
        assert grad[0, 1] == 0.0
        assert grad[1, 0] == 0.0


class TestGRPOSummarizeUpdateKlClampFrac:
    def test_reports_the_share_of_action_tokens_past_the_bound(self) -> None:
        # Arrange: one of seven action tokens sits past the bound.
        agent = _make_grpo(beta=BETA)
        inputs = _with_gap(_inputs())
        mask = inputs["mask"].clone()
        mask[1, 3] = False

        # Act
        stats = agent._summarize_update(
            inputs["log_probs"],
            inputs["old"],
            inputs["reference"],
            mask,
            inputs["advantages"],
            None,
        )

        # Assert
        assert stats["kl_clamp_frac"] == pytest.approx(1 / 7)

    def test_reports_zero_without_a_bound(self) -> None:
        agent = _make_grpo(beta=BETA, kl_clamp=None)
        inputs = _with_gap(_inputs())

        stats = agent._summarize_update(
            inputs["log_probs"],
            inputs["old"],
            inputs["reference"],
            inputs["mask"],
            inputs["advantages"],
            None,
        )

        assert stats["kl_clamp_frac"] == 0.0


class TestGRPOFusedKernelLossKlClamp:
    @pytest.fixture
    def recording_kernel(self, monkeypatch: pytest.MonkeyPatch) -> type:
        monkeypatch.setattr(grpo_module, "HAS_LIGER_KERNEL", True)
        monkeypatch.setattr(RecordingFusedKernel, "calls", [])
        monkeypatch.setattr(grpo_module, KERNEL_NAME, RecordingKlKernel, raising=False)
        return RecordingFusedKernel

    def test_clamped_and_dropped_tokens_reach_the_kernel_with_a_policy_reference(
        self, recording_kernel: type
    ) -> None:
        # Arrange: row 0's first action token sits GAP nats below the
        # reference; row 1's first action token is dropped by the band.
        agent = _make_grpo(
            loss_norm="micro_batch",
            loss_type="cispo",
            beta=BETA,
            use_liger_loss=True,
            off_policy_token_mask_bounds=ICEPOP_BAND,
        )
        ids, mask = _batch()
        _, policy, _ = agent._fused_forward_no_grad(
            ids, NUM_ROWS, include_reference=False
        )
        reference = policy + 0.3
        reference[0, FIRST_ACTION] = policy[0, FIRST_ACTION] + GAP
        old = policy.clone()
        old[1, FIRST_ACTION] += 1.0

        # Act
        _loss, kl, _clipfrac, kernel_policy = agent._fused_kernel_loss(
            ids, mask, torch.ones(NUM_ROWS, 1), old, reference
        )

        # Assert
        (args,) = recording_kernel.calls
        expected = reference.clone()
        expected[0, FIRST_ACTION] = kernel_policy[0, FIRST_ACTION]
        expected[1, FIRST_ACTION] = kernel_policy[1, FIRST_ACTION]
        scored = mask.gather(1, scored_position_index(mask))
        kernel_reference = args[REFERENCE_ARG].reshape(scored.shape)
        assert torch.allclose(kernel_reference[scored], expected[mask])
        unbounded = masked_mean(calculate_k3_kl(reference, kernel_policy), mask)
        assert float(kl) == pytest.approx(float(unbounded), rel=1e-5)


@pytest.mark.skipif(not HAS_LIGER_KERNEL, reason="liger-kernel is Linux-only.")
class TestGRPOKlClampLigerParity:
    @pytest.mark.parametrize("loss_type", ["grpo", "cispo"])
    def test_fused_and_standard_paths_match(self, loss_type: str) -> None:
        # Arrange: row 0's first action token sits GAP nats below the
        # reference; row 1's first action token is dropped by the band.
        ids, mask = _batch()
        advantages = torch.tensor([[1.0], [-1.0], [0.5], [-0.5]])
        results: list[dict[str, Any]] = []

        # Act
        for use_liger_loss in (False, True):
            agent = _make_grpo(
                loss_norm="micro_batch",
                loss_type=loss_type,
                beta=BETA,
                use_liger_loss=use_liger_loss,
                off_policy_token_mask_bounds=ICEPOP_BAND,
            )
            _, policy, _ = agent._fused_forward_no_grad(
                ids, NUM_ROWS, include_reference=False
            )
            reference = policy + 0.3
            reference[0, FIRST_ACTION] = policy[0, FIRST_ACTION] + GAP
            old = policy.clone()
            old[1, FIRST_ACTION] += 1.0
            loss, kl, _clipfrac, _ = agent._objective_loss(
                ids, mask, advantages, old, reference, None, None
            )
            loss.backward()
            results.append(
                {
                    "loss": loss.detach(),
                    "kl": kl.detach(),
                    "grads": {
                        name: param.grad.clone()
                        for name, param in agent.actor.named_parameters()
                        if param.grad is not None
                    },
                }
            )

        # Assert
        standard, fused = results
        assert torch.allclose(fused["loss"], standard["loss"], rtol=1e-5, atol=1e-6)
        assert torch.allclose(fused["kl"], standard["kl"], rtol=1e-5, atol=1e-6)
        assert fused["grads"].keys() == standard["grads"].keys()
        for name, grad in standard["grads"].items():
            assert torch.allclose(fused["grads"][name], grad, rtol=1e-4, atol=1e-6)

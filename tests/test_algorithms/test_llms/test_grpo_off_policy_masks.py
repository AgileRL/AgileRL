# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for GRPO off-policy token / sequence masking and the IS-corrected KL.

Pure CPU on a tiny real model in fp32.
"""

from __future__ import annotations

import math
from typing import Any, ClassVar

import pytest
import torch

pytest.importorskip("transformers", reason="LLM tests require transformers.")
pytest.importorskip("peft", reason="LLM tests require peft.")

from agilerl.algorithms import grpo as grpo_module
from tests.test_algorithms.test_llms.test_grpo_episode_loss_norm import (
    KERNEL_NAME,
    _ProbeFusedKernel,
)
from tests.test_algorithms.test_llms.test_grpo_old_logprobs import (
    NUM_ROWS,
    PROMPT_LEN,
    _batch,
    _experiences,
    _make_grpo,
    _record_step_gradients,
    _sampling_logps,
)

ICEPOP_BAND = (0.5, 5.0)
NO_OFF_POLICY_CORRECTION: dict[str, Any] = {
    "off_policy_token_mask_bounds": None,
    "off_policy_sequence_mask_threshold": None,
    "use_bias_correction_kl": False,
}
VLLM_IS_RATIO_ARG = 23
BIAS_CORRECTION_KL_ARG = 25


class RecordingFusedKernel(_ProbeFusedKernel):
    """Probe kernel that keeps the positional arguments of every call."""

    calls: ClassVar[list[tuple[Any, ...]]] = []

    @classmethod
    def apply(cls, *args):
        """Record the call, then run the probe loss."""
        cls.calls.append(args)
        return super().apply(*args)


def _policy_gradient(
    agent: grpo_module.GRPO,
    log_probs: torch.Tensor,
    behaviour_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Per-token gradient of the standard CISPO loss with respect to ``log_probs``."""
    log_probs = log_probs.clone().requires_grad_(True)
    loss, _kl, _clipfrac = agent._compute_policy_loss(
        mask,
        log_probs,
        behaviour_log_probs,
        torch.zeros_like(log_probs),
        advantages,
        None,
        "token",
        "cispo",
    )
    loss.backward()
    assert log_probs.grad is not None
    return log_probs.grad


class TestGRPOOffPolicyMaskConfig:
    def test_masks_and_bias_corrected_kl_are_on_by_default(self) -> None:
        agent = _make_grpo()

        assert agent.off_policy_token_mask_bounds == (0.5, 5.0)
        assert agent.off_policy_sequence_mask_threshold == 0.03
        assert agent.use_bias_correction_kl is True

    def test_none_and_false_turn_them_off(self) -> None:
        agent = _make_grpo(**NO_OFF_POLICY_CORRECTION)

        assert agent.off_policy_token_mask_bounds is None
        assert agent.off_policy_sequence_mask_threshold is None
        assert agent.use_bias_correction_kl is False
        assert agent._masks_off_policy_tokens is False

    @pytest.mark.parametrize("bounds", [(1.0, 5.0), (0.5, 1.0), (-0.1, 2.0)])
    def test_a_band_that_excludes_ratio_one_is_rejected(
        self, bounds: tuple[float, float]
    ) -> None:
        with pytest.raises(ValueError, match="0 <= low < 1 < high"):
            _make_grpo(off_policy_token_mask_bounds=bounds)

    def test_a_negative_sequence_threshold_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be >= 0"):
            _make_grpo(off_policy_sequence_mask_threshold=-0.1)


class TestGRPOOffPolicyDrops:
    def test_token_mask_drops_ratios_outside_the_band(self) -> None:
        # Arrange: ratios 1.0, 0.4, 6.0, 1.1 and one non-action token at 0.1.
        agent = _make_grpo(off_policy_token_mask_bounds=ICEPOP_BAND)
        behaviour = torch.zeros(1, 5)
        log_probs = torch.log(torch.tensor([[1.0, 0.4, 6.0, 1.1, 0.1]]))
        mask = torch.tensor([[True, True, True, True, False]])

        # Act
        token_drop, sequence_drop = agent._off_policy_drops(
            log_probs, behaviour, mask, torch.ones(1, 1)
        )

        # Assert
        assert token_drop.tolist() == [[False, True, True, False, False]]
        assert not sequence_drop.any()

    def test_sequence_mask_drops_drifted_negative_advantage_rows_only(self) -> None:
        # Arrange: rows 0 and 1 drift 0.3 below the sampler, row 2 drifts 0.05.
        agent = _make_grpo(off_policy_sequence_mask_threshold=0.1)
        behaviour = torch.zeros(3, 4)
        log_probs = torch.tensor([[-0.3] * 4, [-0.3] * 4, [-0.05] * 4])
        mask = torch.ones(3, 4, dtype=torch.bool)
        mask[0, 3] = False
        advantages = torch.tensor([[-1.0], [1.0], [-1.0]])

        # Act
        token_drop, sequence_drop = agent._off_policy_drops(
            log_probs, behaviour, mask, advantages
        )

        # Assert
        assert sequence_drop.tolist() == [
            [True, True, True, False],
            [False] * 4,
            [False] * 4,
        ]
        assert not token_drop.any()

    def test_sequence_mask_reads_per_token_advantage_signs(self) -> None:
        agent = _make_grpo(off_policy_sequence_mask_threshold=0.1)
        log_probs = torch.full((1, 4), -0.5)
        advantages = torch.tensor([[1.0, 1.0, -1.0, -1.0]])

        _token_drop, sequence_drop = agent._off_policy_drops(
            log_probs, torch.zeros(1, 4), torch.ones(1, 4), advantages
        )

        assert sequence_drop.tolist() == [[False, False, True, True]]

    def test_default_sequence_threshold_drops_negative_rows_past_it(self) -> None:
        # Arrange: row 0 drifts 0.05 past the sampler, row 1 drifts 0.02.
        agent = _make_grpo()
        log_probs = torch.tensor([[-0.05] * 3, [-0.02] * 3])

        # Act
        token_drop, sequence_drop = agent._off_policy_drops(
            log_probs,
            torch.zeros(2, 3),
            torch.ones(2, 3, dtype=torch.bool),
            -torch.ones(2, 1),
        )

        # Assert
        assert sequence_drop.tolist() == [[True] * 3, [False] * 3]
        assert not token_drop.any()

    def test_rows_without_sampling_log_probs_drop_nothing(self) -> None:
        # Arrange: both rows sit at ratio 0.4 and drift 0.9 with negative advantage.
        agent = _make_grpo(
            off_policy_token_mask_bounds=ICEPOP_BAND,
            off_policy_sequence_mask_threshold=0.1,
        )
        log_probs = torch.log(torch.full((2, 3), 0.4))

        # Act
        token_drop, sequence_drop = agent._off_policy_drops(
            log_probs,
            torch.zeros(2, 3),
            torch.ones(2, 3, dtype=torch.bool),
            -torch.ones(2, 1),
            torch.tensor([True, False]),
        )

        # Assert
        assert token_drop.tolist() == [[True] * 3, [False] * 3]
        assert sequence_drop.tolist() == [[True] * 3, [False] * 3]


class TestGRPOComputePolicyLossOffPolicyMasks:
    def test_dropped_tokens_get_no_policy_gradient(self) -> None:
        # Arrange: token 1 sits at ratio 0.4, outside the band.
        masked = _make_grpo(
            loss_norm="micro_batch",
            loss_type="cispo",
            off_policy_token_mask_bounds=ICEPOP_BAND,
        )
        unmasked = _make_grpo(
            loss_norm="micro_batch", loss_type="cispo", **NO_OFF_POLICY_CORRECTION
        )
        log_probs = torch.log(torch.tensor([[0.5, 0.2, 0.5, 0.5]]))
        behaviour = torch.log(torch.tensor([[0.5, 0.5, 0.5, 0.5]]))
        advantages = torch.ones(1, 1)
        mask = torch.ones(1, 4, dtype=torch.bool)

        # Act
        masked_grad = _policy_gradient(masked, log_probs, behaviour, advantages, mask)
        unmasked_grad = _policy_gradient(
            unmasked, log_probs, behaviour, advantages, mask
        )

        # Assert
        assert masked_grad[0, 1] == 0.0
        assert unmasked_grad[0, 1] != 0.0
        keep = [0, 2, 3]
        assert torch.allclose(masked_grad[0, keep], unmasked_grad[0, keep])

    def test_default_band_drops_a_token_outside_it(self) -> None:
        # Arrange: token 1 sits at ratio 0.4, below the default band's 0.5.
        agent = _make_grpo(loss_norm="micro_batch", loss_type="cispo")
        log_probs = torch.log(torch.tensor([[0.5, 0.2, 0.5, 0.5]]))
        behaviour = torch.log(torch.tensor([[0.5, 0.5, 0.5, 0.5]]))
        mask = torch.ones(1, 4, dtype=torch.bool)

        # Act
        grad = _policy_gradient(agent, log_probs, behaviour, torch.ones(1, 1), mask)

        # Assert
        assert grad[0, 1] == 0.0
        assert (grad[0, [0, 2, 3]] != 0.0).all()

    def test_sampling_log_probs_are_the_behaviour_policy(self) -> None:
        # Arrange: old log-probs match the policy; the sampler is 1 nat above.
        agent = _make_grpo(
            loss_norm="micro_batch",
            loss_type="cispo",
            off_policy_token_mask_bounds=ICEPOP_BAND,
        )
        log_probs = torch.full((1, 3), -1.0, requires_grad=True)
        sampling = torch.tensor([[-1.0, 0.0, -1.0]])
        mask = torch.ones(1, 3, dtype=torch.bool)

        # Act
        loss, _kl, _clipfrac = agent._compute_policy_loss(
            mask,
            log_probs,
            log_probs.detach(),
            torch.zeros(1, 3),
            torch.ones(1, 1),
            None,
            "token",
            "cispo",
            sampling_log_probs=sampling,
        )
        loss.backward()

        # Assert
        assert log_probs.grad is not None
        assert log_probs.grad[0, 1] == 0.0
        assert (log_probs.grad[0, [0, 2]] != 0.0).all()

    @pytest.mark.parametrize(
        ("mask_kwargs", "dropped"),
        [
            ({"off_policy_token_mask_bounds": ICEPOP_BAND}, [False, True]),
            ({"off_policy_sequence_mask_threshold": 0.1}, [True, True]),
        ],
    )
    def test_dropped_tokens_get_no_kl_gradient(
        self, mask_kwargs: dict[str, Any], dropped: list[bool]
    ) -> None:
        # Arrange: a near-zero negative advantage leaves the KL term. Token 1
        # sits 1 nat below its sampler, outside the band; the row drifts 0.5.
        agent = _make_grpo(
            loss_norm="micro_batch",
            loss_type="cispo",
            beta=0.1,
            **{**NO_OFF_POLICY_CORRECTION, **mask_kwargs},
        )
        log_probs = torch.full((1, 2), -1.0, requires_grad=True)
        behaviour = torch.tensor([[-1.0, 0.0]])
        reference = torch.full((1, 2), -0.5)
        mask = torch.ones(1, 2, dtype=torch.bool)

        # Act
        loss, kl, _clipfrac = agent._compute_policy_loss(
            mask,
            log_probs,
            behaviour,
            reference,
            -torch.full((1, 1), 1e-6),
            None,
            "token",
            "cispo",
        )
        loss.backward()

        # Assert: dropped tokens keep their KL in the metric only.
        assert log_probs.grad is not None
        assert (log_probs.grad[0] == 0.0).tolist() == dropped
        assert float(kl) == pytest.approx(math.exp(0.5) - 0.5 - 1, rel=1e-6)

    def test_trajectory_level_loss_drops_masked_tokens(self) -> None:
        # Arrange: token 2 sits at ratio exp(-2); the pooled ratio exp(-2/3)
        # is below the 0.8 clip, so every token's surrogate is -exp(-2/3).
        agent = _make_grpo(
            loss_norm="micro_batch",
            loss_type="gspo",
            off_policy_token_mask_bounds=ICEPOP_BAND,
        )
        mask = torch.ones(1, 3, dtype=torch.bool)

        # Act
        loss, _kl, _clipfrac = agent._compute_policy_loss(
            mask,
            torch.tensor([[-1.0, -1.0, -3.0]]),
            torch.full((1, 3), -1.0),
            torch.zeros(1, 3),
            torch.ones(1, 1),
            None,
            "trajectory",
            "grpo",
        )

        # Assert: the row mean keeps 2 of its 3 tokens.
        assert float(loss) == pytest.approx(-math.exp(-2 / 3) * 2 / 3, rel=1e-6)


class TestGRPOComputePolicyLossBiasCorrectionKL:
    def test_kl_term_is_weighted_by_the_importance_ratio(self) -> None:
        # Arrange: every token's ratio to the old policy is 2.
        corrected = _make_grpo(
            loss_norm="micro_batch", beta=0.1, use_bias_correction_kl=True
        )
        plain = _make_grpo(
            loss_norm="micro_batch", beta=0.1, use_bias_correction_kl=False
        )
        log_probs = torch.full((2, 3), -1.0)
        old = log_probs - math.log(2.0)
        reference = torch.full((2, 3), -0.7)
        mask = torch.ones(2, 3, dtype=torch.bool)
        args = (mask, log_probs, old, reference, torch.zeros(2, 1), None, "token")

        # Act
        corrected_loss, corrected_kl, _ = corrected._compute_policy_loss(*args, "cispo")
        plain_loss, plain_kl, _ = plain._compute_policy_loss(*args, "cispo")

        # Assert
        assert float(corrected_loss) == pytest.approx(2 * float(plain_loss), rel=1e-6)
        assert float(corrected_kl) == pytest.approx(2 * float(plain_kl), rel=1e-6)

    def test_kl_gradient_gains_the_k3_value_at_ratio_one(self) -> None:
        # Arrange
        corrected = _make_grpo(
            loss_norm="micro_batch", beta=0.1, use_bias_correction_kl=True
        )
        plain = _make_grpo(
            loss_norm="micro_batch", beta=0.1, use_bias_correction_kl=False
        )
        reference = torch.full((1, 3), -0.7)
        mask = torch.ones(1, 3, dtype=torch.bool)
        grads = []

        # Act
        for agent in (corrected, plain):
            log_probs = torch.full((1, 3), -1.0, requires_grad=True)
            loss, _kl, _ = agent._compute_policy_loss(
                mask,
                log_probs,
                log_probs.detach(),
                reference,
                torch.zeros(1, 1),
                None,
                "token",
                "cispo",
            )
            loss.backward()
            grads.append(log_probs.grad)

        # Assert: d/dx [exp(x - x0) * k3] = k3 + dk3/dx at x == x0, and
        # dk3/dx = 1 - exp(ref - x).
        k3 = math.exp(0.3) - 0.3 - 1
        dk3 = 1 - math.exp(0.3)
        assert torch.allclose(
            grads[0], grads[1] * (k3 + dk3) / dk3, rtol=1e-5, atol=0.0
        )


class TestGRPOFusedKernelLossOffPolicyMasks:
    @pytest.fixture
    def recording_kernel(self, monkeypatch: pytest.MonkeyPatch) -> type:
        monkeypatch.setattr(grpo_module, "HAS_LIGER_KERNEL", True)
        monkeypatch.setattr(RecordingFusedKernel, "calls", [])
        monkeypatch.setattr(
            grpo_module, KERNEL_NAME, RecordingFusedKernel, raising=False
        )
        return RecordingFusedKernel

    def test_masked_tokens_reach_the_kernel_as_zero_weight(
        self, recording_kernel: type
    ) -> None:
        # Arrange: the first action token of row 0 sits at ratio exp(-1).
        agent = _make_grpo(
            loss_norm="micro_batch",
            loss_type="cispo",
            use_liger_loss=True,
            off_policy_token_mask_bounds=ICEPOP_BAND,
            use_bias_correction_kl=True,
        )
        ids, mask = _batch()
        _, policy, _ = agent._fused_forward_no_grad(
            ids, NUM_ROWS, include_reference=False
        )
        old = policy.clone()
        old[0, PROMPT_LEN - 1] += 1.0

        # Act
        agent._fused_kernel_loss(
            ids, mask, torch.ones(NUM_ROWS, 1), old, torch.zeros_like(old)
        )

        # Assert
        (args,) = recording_kernel.calls
        expected = torch.ones(NUM_ROWS, int(mask.sum(dim=1).max()))
        expected[0, 0] = 0.0
        assert torch.equal(args[VLLM_IS_RATIO_ARG].reshape(expected.shape), expected)
        assert args[BIAS_CORRECTION_KL_ARG] is True

    def test_rows_without_sampling_log_probs_keep_full_weight(
        self, recording_kernel: type
    ) -> None:
        # Arrange: row 0 drifts outside the band but has no sampling log-probs.
        agent = _make_grpo(
            loss_norm="micro_batch",
            loss_type="cispo",
            use_liger_loss=True,
            off_policy_token_mask_bounds=ICEPOP_BAND,
        )
        ids, mask = _batch()
        _, policy, _ = agent._fused_forward_no_grad(
            ids, NUM_ROWS, include_reference=False
        )
        old = policy.clone()
        old[0, PROMPT_LEN - 1] += 1.0
        sampled_rows = torch.tensor([False] + [True] * (NUM_ROWS - 1))

        # Act
        agent._fused_kernel_loss(
            ids,
            mask,
            torch.ones(NUM_ROWS, 1),
            old,
            torch.zeros_like(old),
            sampled_rows=sampled_rows,
        )

        # Assert
        (args,) = recording_kernel.calls
        expected = torch.ones(NUM_ROWS, int(mask.sum(dim=1).max()))
        assert torch.equal(args[VLLM_IS_RATIO_ARG].reshape(expected.shape), expected)

    def test_unmasked_run_passes_no_token_weight(self, recording_kernel: type) -> None:
        agent = _make_grpo(
            loss_norm="micro_batch",
            loss_type="cispo",
            use_liger_loss=True,
            **NO_OFF_POLICY_CORRECTION,
        )
        ids, mask = _batch()

        agent._fused_kernel_loss(
            ids, mask, torch.ones(NUM_ROWS, 1), None, torch.zeros(mask.shape)
        )

        (args,) = recording_kernel.calls
        assert args[VLLM_IS_RATIO_ARG] is None
        assert args[BIAS_CORRECTION_KL_ARG] is False

    def test_trajectory_level_masks_run_the_standard_path(self) -> None:
        agent = _make_grpo(
            loss_norm="micro_batch",
            loss_type="gspo",
            use_liger_loss=True,
            vllm_importance_sampling_correction=False,
            off_policy_token_mask_bounds=ICEPOP_BAND,
        )

        assert agent._liger_path_selected is False


class TestGRPOLearnOffPolicyMasks:
    def test_tokens_outside_the_band_get_no_gradient(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange: the sampler sits 1 nat above the policy on every token.
        agent = _make_grpo(
            old_logprobs_source="rollout",
            off_policy_token_mask_bounds=ICEPOP_BAND,
            off_policy_sequence_mask_threshold=None,
        )
        sampling_logps = _sampling_logps(agent, offset=1.0)
        steps = _record_step_gradients(agent, monkeypatch)

        # Act
        metrics = agent.learn(_experiences(), sampling_logps=sampling_logps)

        # Assert
        assert metrics["off_policy_token_mask_frac"] == pytest.approx(1.0)
        assert metrics["off_policy_seq_mask_frac"] == 0.0
        assert steps
        assert all(not grad.any() for step in steps for grad in step.values())

    def test_drifted_negative_rows_are_reported(self) -> None:
        # Arrange: drift 0.2 > 0.1; rows 1 and 2 (9 of 18 action tokens) lose.
        agent = _make_grpo(
            old_logprobs_source="rollout", off_policy_sequence_mask_threshold=0.1
        )
        sampling_logps = _sampling_logps(agent, offset=0.2)

        # Act
        metrics = agent.learn(_experiences(), sampling_logps=sampling_logps)

        # Assert
        assert metrics["off_policy_seq_mask_frac"] == pytest.approx(0.5)
        assert metrics["off_policy_token_mask_frac"] == 0.0

    def test_trainer_scored_rows_are_left_out_of_the_mask_share(self) -> None:
        # Arrange: rows 0 and 1 carry sampler log-probs 1 nat above the
        # policy; rows 2 and 3 have none, so the trainer scores them.
        agent = _make_grpo(
            old_logprobs_source="rollout", off_policy_token_mask_bounds=ICEPOP_BAND
        )
        shifted = _sampling_logps(agent, offset=1.0)
        sampling_logps = [shifted[0], shifted[1], None, None]

        # Act
        metrics = agent.learn(_experiences(), sampling_logps=sampling_logps)

        # Assert
        assert metrics["old_logprobs_trainer_rows"] == 2.0
        assert metrics["off_policy_token_mask_frac"] == pytest.approx(1.0)

    def test_default_on_policy_learn_masks_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange: no sampling log-probs, so the learn-start policy is the behaviour.
        agent = _make_grpo()
        steps = _record_step_gradients(agent, monkeypatch)

        # Act
        metrics = agent.learn(_experiences())

        # Assert
        assert math.isfinite(metrics["loss"])
        assert metrics["off_policy_token_mask_frac"] == 0.0
        assert metrics["off_policy_seq_mask_frac"] == 0.0
        assert steps
        assert any(grad.any() for step in steps for grad in step.values())

    def test_default_masks_leave_rows_without_sampling_log_probs_trained(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange: rows 0 and 1 sit 1 nat below their sampler, so the band
        # drops every one of their tokens; rows 2 and 3 have no sampling log-probs.
        agent = _make_grpo()
        shifted = _sampling_logps(agent, offset=1.0)
        steps = _record_step_gradients(agent, monkeypatch)

        # Act
        metrics = agent.learn(
            _experiences(), sampling_logps=[shifted[0], shifted[1], None, None]
        )

        # Assert: the gradient comes from rows 2 and 3 alone.
        assert math.isfinite(metrics["loss"])
        assert metrics["off_policy_token_mask_frac"] == pytest.approx(1.0)
        assert steps
        assert any(grad.any() for step in steps for grad in step.values())

    def test_unmasked_run_reports_zero(self) -> None:
        agent = _make_grpo(old_logprobs_source="rollout", **NO_OFF_POLICY_CORRECTION)

        metrics = agent.learn(_experiences(), sampling_logps=_sampling_logps(agent))

        assert metrics["off_policy_token_mask_frac"] == 0.0
        assert metrics["off_policy_seq_mask_frac"] == 0.0

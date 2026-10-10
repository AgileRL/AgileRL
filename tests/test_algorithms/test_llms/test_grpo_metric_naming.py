# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for GRPO's fixed per-update keys and loss-path routing.

``learn`` always reports ``kl`` (except on the fused path at ``beta == 0.0``,
where the kernel emits no divergence) and ``clipfrac`` on every path. Pure
CPU: the loss paths are stubs, only the routing and the reported keys are
under test.
"""

from __future__ import annotations

import math
import time
import warnings
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

pytest.importorskip("transformers", reason="LLM tests require transformers.")
pytest.importorskip("peft", reason="LLM tests require peft.")

from agilerl.algorithms.core.base import LLMAlgorithm
from agilerl.algorithms.core.llm_ops.frozen_vision import VisionFeatureCache
from agilerl.algorithms.grpo import GRPO
from agilerl.utils.learn_profiler import LearnProfiler
from agilerl.utils.llm_utils import LEARN_PHASES
from agilerl.utils.phase_timer import PhaseTimer

SEQ_LEN = 6
PAD_TOKEN_ID = 0


class _MetricsRecorder:
    """Metrics-tracker stand-in recording every registration and log."""

    def __init__(self) -> None:
        self.registered: list[str] = []
        self.logged: dict[str, float] = {}

    def register(self, name: str) -> None:
        """Declare a metric series."""
        self.registered.append(name)

    def log(self, name: str, value: float) -> None:
        """Record one value for a metric series."""
        self.logged[name] = value


class _Stub:
    """Stand-in carrying the state ``learn`` and the metric name read."""

    def __init__(
        self,
        *,
        beta: float = 0.0,
        use_liger_loss: bool = False,
        importance_sampling_level: str = "token",
        liger_level_supported: bool = True,
        vllm_importance_sampling_correction: bool = True,
        filter_zero_adv: bool = False,
        survivors: int | None = None,
        kl_value: float = 0.25,
        clipfrac_value: float = 0.1,
    ) -> None:
        self.device = torch.device("cpu")
        self.beta = beta
        self.loss_type = "grpo"
        self.use_liger_loss = use_liger_loss
        self.importance_sampling_level = importance_sampling_level
        self._liger_level_supported = liger_level_supported
        self.vllm_importance_sampling_correction = vllm_importance_sampling_correction
        self.vllm_importance_sampling_cap = 2.0
        self.vllm_max_logprob_gap = 0.1
        self.vllm_max_clip_fraction = 0.02
        self.off_policy_token_mask_bounds = None
        self.off_policy_sequence_mask_threshold = None
        self.use_bias_correction_kl = False
        self.kl_clamp = 10.0
        self.filter_zero_adv = filter_zero_adv
        self.adv_filter_eps = 0.0
        self.clip_coef_min = 0.8
        self.clip_coef_max = 1.2
        self.use_kl_advantage_shaping = False
        self.loss_norm = "micro_batch"
        self._segment_accumulation_steps = None
        self.pad_token_id = PAD_TOKEN_ID
        self.update_epochs = 1
        self.micro_batch_size_per_gpu = 2
        self.gradient_accumulation_steps = 1
        self._is_correction_liger_warned = False
        self._is_per_token_trajectory_liger_warned = False
        self._survivors = survivors
        self._advantages: torch.Tensor | None = None
        self._kl_value = kl_value
        self._clipfrac_value = clipfrac_value
        self.metrics = _MetricsRecorder()
        self.learn_profiler = LearnProfiler(
            None, rank=0, world_size=1, shard_group_size=None
        )
        self.packing_mode: str | None = None
        self.rng = np.random.default_rng(0)
        self.liger_calls = 0
        self.standard_calls = 0
        self.old_logprobs_source = "trainer"
        self.lr_scheduler = None
        self.no_grad_forwards: list[tuple[int, bool, bool]] = []
        self._vision_cache = VisionFeatureCache()
        self.shard_runtime = SimpleNamespace(
            timed=lambda _name, **_fields: nullcontext(),
            micro_batches_until_step=lambda steps: steps,
            phase_timer=PhaseTimer(),
        )

    learn = GRPO.learn
    _align_sampling_logprobs = GRPO._align_sampling_logprobs
    _aligned_sampling_logprobs_and_metrics = GRPO._aligned_sampling_logprobs_and_metrics
    _apply_kl_advantage_shaping = GRPO._apply_kl_advantage_shaping
    _clipped_units = GRPO._clipped_units
    _compute_policy_loss = GRPO._compute_policy_loss
    _learn_start_log_probs = GRPO._learn_start_log_probs
    _liger_path_selected = GRPO._liger_path_selected
    _check_segments_supported = LLMAlgorithm._check_segments_supported
    _device_index = staticmethod(LLMAlgorithm._device_index)
    _epoch_orders = LLMAlgorithm._epoch_orders
    _has_episode_segments = LLMAlgorithm._has_episode_segments
    _hold_loss_until_step = LLMAlgorithm._hold_loss_until_step
    _learn_phase_seconds = LLMAlgorithm._learn_phase_seconds
    _row_padding_stats = LLMAlgorithm._row_padding_stats
    _segment_rows = LLMAlgorithm._segment_rows
    _start_learn_phases = LLMAlgorithm._start_learn_phases
    _step_lr_scheduler = LLMAlgorithm._step_lr_scheduler
    _log_importance_weights = GRPO._log_importance_weights
    _loss = GRPO._loss
    _masks_off_policy_tokens = GRPO._masks_off_policy_tokens
    _off_policy_mask_stats = GRPO._off_policy_mask_stats
    _objective_loss = GRPO._objective_loss
    _prepare_experience_batch = GRPO._prepare_experience_batch
    _raise_if_loss_not_finite_on_any_rank = (
        LLMAlgorithm._raise_if_loss_not_finite_on_any_rank
    )
    _record_global_window_action_tokens = GRPO._record_global_window_action_tokens
    _reduce_masked_loss = GRPO._reduce_masked_loss
    _resolve_loss_window = GRPO._resolve_loss_window
    _token_loss_weights = GRPO._token_loss_weights
    _rollout_old_log_probs = GRPO._rollout_old_log_probs
    _rows_with_full_sampling_logprobs = staticmethod(
        GRPO._rows_with_full_sampling_logprobs
    )
    _sampling_mismatch_metrics = GRPO._sampling_mismatch_metrics
    _self_scored_micro_batches = GRPO._self_scored_micro_batches
    _summarize_update = GRPO._summarize_update
    _use_liger_path = GRPO._use_liger_path
    _warn_if_micro_batches_straddle_optimizer_steps = (
        GRPO._warn_if_micro_batches_straddle_optimizer_steps
    )
    _warn_liger_path_bypassed = GRPO._warn_liger_path_bypassed
    _warn_per_token_trajectory_liger_bypass = (
        GRPO._warn_per_token_trajectory_liger_bypass
    )
    _warn_on_sampling_mismatch = GRPO._warn_on_sampling_mismatch

    def _prepare_vllm_for_training(self) -> None:
        return

    def _packing_mode(self) -> str | None:
        """Packing mode of the gradient forward."""
        return self.packing_mode

    def trainer_offload_context(self):
        """Match the context ``learn`` wraps its body in."""
        return nullcontext()

    def _calculate_advantages(
        self,
        rewards: torch.Tensor,
        _completion_ids: torch.Tensor,
        _action_masks: torch.Tensor,
        _turn_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, np.ndarray]:
        """Advantages and the sample indices surviving the advantage filter."""
        rows = rewards.shape[0]
        survivors = rows if self._survivors is None else self._survivors
        advantages = (
            torch.ones(rows, 1) if self._advantages is None else self._advantages
        )
        return advantages, np.arange(survivors)

    def _fused_forward_no_grad(
        self,
        ids: torch.Tensor,
        _batch_size: int,
        *,
        include_reference: bool = True,
        include_actor: bool = True,
        **_kwargs: Any,
    ):
        """Reference and old log-probs on the action frame, recording each call."""
        self.no_grad_forwards.append((ids.shape[0], include_reference, include_actor))
        zeros = torch.zeros(ids.shape[0], ids.shape[1] - 1)
        return zeros, zeros, None

    def _backward_pass(
        self, _loss: torch.Tensor, _accumulation_steps: int | None = None
    ) -> None:
        return None

    def _liger_loss(self, batch_ids: torch.Tensor, *_args: Any, **_kwargs: Any):
        """Record that the fused path ran."""
        self.liger_calls += 1
        kl = (
            torch.tensor(float("nan"))
            if self.beta == 0.0
            else torch.tensor(self._kl_value)
        )
        log_probs = torch.zeros(batch_ids.shape[0], batch_ids.shape[1] - 1)
        return torch.tensor(1.0), kl, torch.tensor(self._clipfrac_value), log_probs

    def _get_logprobs(self, ids: torch.Tensor, **_kwargs: Any) -> torch.Tensor:
        """Record that the standard path ran."""
        self.standard_calls += 1
        return torch.zeros(ids.shape[0], ids.shape[1] - 1)

    def _loss_fn(self, *_args: Any, **_kwargs: Any):
        """Stand in for the configured standard-path objective."""
        return (
            torch.tensor(1.0),
            torch.tensor(self._kl_value),
            torch.tensor(self._clipfrac_value),
        )


def _experiences(batch_size: int = 2):
    """Completion ids, action masks and rewards for one ``learn`` call."""
    completion_ids = [
        torch.full((1, SEQ_LEN), PAD_TOKEN_ID + 1, dtype=torch.long)
        for _ in range(batch_size)
    ]
    action_masks = [
        torch.ones(1, SEQ_LEN - 1, dtype=torch.bool) for _ in range(batch_size)
    ]
    rewards = torch.tensor([1.0, -1.0][:batch_size], dtype=torch.float32)
    return completion_ids, action_masks, rewards


def _sampling_logps(batch_size: int = 2) -> list[torch.Tensor]:
    """Per-row vLLM sampling log-probs covering every action token."""
    return [torch.full((SEQ_LEN - 1,), -3.0) for _ in range(batch_size)]


class TestLigerPathSelection:
    """Only configuration decides the loss path, never the batch in hand."""

    @pytest.mark.parametrize("sampling_logps", [None, _sampling_logps()])
    def test_token_level_keeps_the_fused_path_either_way(
        self,
        sampling_logps: list[torch.Tensor] | None,
    ) -> None:
        algo = _Stub(use_liger_loss=True)
        assert algo._liger_path_selected is True
        algo.learn(_experiences(), sampling_logps=sampling_logps)
        assert (algo.liger_calls, algo.standard_calls) == (1, 0)

    @pytest.mark.parametrize("sampling_logps", [None, _sampling_logps()])
    def test_a_corrected_run_stays_on_the_standard_path_either_way(
        self,
        sampling_logps: list[torch.Tensor] | None,
    ) -> None:
        algo = _Stub(use_liger_loss=True, importance_sampling_level="trajectory")
        assert algo._liger_path_selected is False
        with pytest.warns(UserWarning, match="only at token-level"):
            algo.learn(_experiences(), sampling_logps=sampling_logps)
        assert (algo.liger_calls, algo.standard_calls) == (0, 1)

    def test_dropping_the_correction_returns_the_fused_path(self) -> None:
        algo = _Stub(
            use_liger_loss=True,
            importance_sampling_level="trajectory",
            vllm_importance_sampling_correction=False,
        )
        assert algo._liger_path_selected is True
        algo.learn(_experiences())
        assert (algo.liger_calls, algo.standard_calls) == (1, 0)

    def test_an_unsupported_level_does_not_warn_at_loss_time(self) -> None:
        algo = _Stub(
            use_liger_loss=True,
            importance_sampling_level="turn",
            liger_level_supported=False,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert algo._use_liger_path() is False
        assert caught == []

    def test_the_bypass_warning_is_emitted_once(self) -> None:
        algo = _Stub(use_liger_loss=True, importance_sampling_level="trajectory")
        with pytest.warns(UserWarning, match="only at token-level"):
            algo._use_liger_path()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            algo._use_liger_path()
        assert caught == []

    def test_no_bypass_warning_without_the_fused_kernel_requested(self) -> None:
        algo = _Stub(importance_sampling_level="trajectory")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert algo._use_liger_path() is False
        assert caught == []

    def test_gspo_per_sequence_advantages_run_the_fused_path(self) -> None:
        algo = _Stub(
            use_liger_loss=True,
            importance_sampling_level="trajectory",
            vllm_importance_sampling_correction=False,
        )
        algo._advantages = torch.tensor([[0.5], [-0.5]])

        algo.learn(_experiences())

        assert (algo.liger_calls, algo.standard_calls) == (1, 0)

    def test_gspo_per_token_advantages_run_the_standard_path(self) -> None:
        algo = _Stub(
            use_liger_loss=True,
            importance_sampling_level="trajectory",
            vllm_importance_sampling_correction=False,
        )
        algo._advantages = torch.linspace(-1.0, 1.0, 2 * (SEQ_LEN - 1)).reshape(
            2, SEQ_LEN - 1
        )

        with pytest.warns(UserWarning, match="no fused kernel for per-turn"):
            algo.learn(_experiences())

        assert (algo.liger_calls, algo.standard_calls) == (0, 1)

    def test_the_per_token_gspo_bypass_warns_once(self) -> None:
        algo = _Stub(
            use_liger_loss=True,
            importance_sampling_level="trajectory",
            vllm_importance_sampling_correction=False,
        )
        algo._advantages = torch.ones(2, SEQ_LEN - 1)
        with pytest.warns(UserWarning, match="no fused kernel for per-turn"):
            algo.learn(_experiences())

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            algo.learn(_experiences())

        assert [str(w.message) for w in caught if "per-turn" in str(w.message)] == []
        assert (algo.liger_calls, algo.standard_calls) == (0, 2)


class ObjectiveStub:
    """Stand-in carrying the state ``_setup_objective`` reads."""

    def __init__(self, advantage_granularity: str) -> None:
        self.advantage_granularity = advantage_granularity
        self.beta = 0.0
        self.use_liger_loss = True

    _setup_objective = GRPO._setup_objective
    _resolve_standard_loss_fn = GRPO._resolve_standard_loss_fn
    _grpo_loss_standard = GRPO._grpo_loss_standard
    _cispo_loss = GRPO._cispo_loss


class TestGRPOSetupObjectiveLigerSupport:
    """Whether the configured objective has a fused kernel is decided at init."""

    @pytest.mark.parametrize(
        ("loss_type", "advantage_granularity"),
        [("gspo", "trajectory"), ("gspo", "auto"), ("grpo", "turn")],
    )
    def test_supported_combinations_keep_the_fused_kernel(
        self, loss_type: str, advantage_granularity: str
    ) -> None:
        algo = ObjectiveStub(advantage_granularity)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            algo._setup_objective(loss_type, None, False)

        assert algo._liger_level_supported is True
        assert [w for w in caught if "no fused" in str(w.message)] == []

    def test_gspo_with_turn_advantages_has_no_fused_kernel(self) -> None:
        algo = ObjectiveStub("turn")

        with pytest.warns(
            UserWarning,
            match=(
                "no fused gspo kernel at importance_sampling_level='trajectory' "
                "with advantage_granularity='turn'"
            ),
        ):
            algo._setup_objective("gspo", None, False)

        assert algo._liger_level_supported is False


class TestLearnReportsFixedKeys:
    """Every ``learn`` return carries ``kl`` and ``clipfrac`` under fixed keys."""

    def test_the_standard_path_reports_both(self) -> None:
        metrics = _Stub(beta=0.0).learn(_experiences())

        assert metrics["kl"] == pytest.approx(0.25)
        assert metrics["clipfrac"] == pytest.approx(0.1)

    def test_the_fused_path_at_zero_beta_reports_nan_kl(self) -> None:
        algo = _Stub(beta=0.0, use_liger_loss=True)
        metrics = algo.learn(_experiences())

        assert math.isnan(metrics["kl"])
        assert metrics["clipfrac"] == pytest.approx(0.1)

    def test_the_fused_path_with_a_kl_coefficient_reports_both(self) -> None:
        algo = _Stub(beta=0.04, use_liger_loss=True)
        metrics = algo.learn(_experiences())

        assert metrics["kl"] == pytest.approx(0.25)
        assert metrics["clipfrac"] == pytest.approx(0.1)

    def test_an_emptied_batch_reports_zeros_under_fixed_keys(self) -> None:
        algo = _Stub(beta=0.0, use_liger_loss=True, filter_zero_adv=True)
        algo._survivors = 0
        with pytest.warns(UserWarning, match="advantage threshold"):
            emptied = algo.learn(_experiences())

        assert emptied["loss"] == pytest.approx(0.0)
        assert emptied["kl"] == pytest.approx(0.0)
        assert emptied["clipfrac"] == pytest.approx(0.0)
        assert emptied["completion_length"] == pytest.approx(SEQ_LEN)

    def test_a_batch_carrying_sampling_logprobs_reports_fixed_keys(self) -> None:
        algo = _Stub(
            beta=0.0, use_liger_loss=True, importance_sampling_level="trajectory"
        )
        with pytest.warns(UserWarning, match="only at token-level"):
            corrected = algo.learn(_experiences(), sampling_logps=_sampling_logps())
        uncorrected = algo.learn(_experiences())

        for metrics in (corrected, uncorrected):
            assert metrics["kl"] == pytest.approx(0.25)
            assert metrics["clipfrac"] == pytest.approx(0.1)

    def test_fixed_keys_reach_the_metrics_tracker(self) -> None:
        algo = _Stub(beta=0.04, use_liger_loss=True)
        algo.learn(_experiences())

        assert algo.metrics.logged["kl"] == pytest.approx(0.25)
        assert algo.metrics.logged["clipfrac"] == pytest.approx(0.1)


class TestLearnTelemetryReportsDiagnostics:
    """Every ``learn`` return carries the per-learn diagnostic scalars."""

    def test_full_learn_reports_advantage_stats(self) -> None:
        # Arrange: the stub reports all-ones (B, 1) advantages.
        algo = _Stub(beta=0.0)

        # Act
        metrics = algo.learn(_experiences())

        # Assert
        assert metrics["adv_mean"] == pytest.approx(1.0)
        assert metrics["adv_min"] == pytest.approx(1.0)
        assert metrics["adv_max"] == pytest.approx(1.0)
        assert metrics["adv_zero_frac"] == pytest.approx(0.0)

    def test_full_learn_reports_snapshot_stats(self) -> None:
        # Arrange: post/old/reference log-probs are all zeros, so entropy and
        # both KLs are 0 and every importance ratio is exactly 1. A KL
        # coefficient makes learn score reference log-probs.
        algo = _Stub(beta=0.04)

        # Act
        metrics = algo.learn(_experiences())

        # Assert
        assert metrics["entropy"] == pytest.approx(0.0)
        assert metrics["kl_ref"] == pytest.approx(0.0)
        assert metrics["kl_old"] == pytest.approx(0.0)
        assert metrics["is_ratio_mean"] == pytest.approx(1.0)
        assert metrics["is_ratio_p05"] == pytest.approx(1.0)
        assert metrics["is_ratio_p50"] == pytest.approx(1.0)
        assert metrics["is_ratio_p95"] == pytest.approx(1.0)
        assert metrics["is_frac_below"] == pytest.approx(0.0)
        assert metrics["is_frac_above"] == pytest.approx(0.0)
        assert metrics["is_frac_clip_pos"] == pytest.approx(0.0)
        assert metrics["is_frac_clip_neg"] == pytest.approx(0.0)

    def test_grad_norms_are_absent_when_no_step_synced(self) -> None:
        # Arrange: the stub backward pass reports no norms (as on DeepSpeed
        # accumulation steps that do not sync gradients).
        algo = _Stub(beta=0.0)

        # Act
        metrics = algo.learn(_experiences())

        # Assert
        assert "grad_norm_pre" not in metrics
        assert "grad_norm_post" not in metrics

    def test_telemetry_is_logged_to_the_metrics_tracker(self) -> None:
        # Arrange
        algo = _Stub(beta=0.0)

        # Act
        algo.learn(_experiences())

        # Assert
        for key in (
            "adv_mean",
            "adv_min",
            "adv_max",
            "adv_zero_frac",
            "entropy",
            "kl_ref",
            "kl_old",
            "is_ratio_mean",
            "is_frac_below",
            "is_frac_above",
            "is_frac_clip_pos",
            "is_frac_clip_neg",
        ):
            assert key in algo.metrics.logged

    def test_sampling_logps_surface_vllm_metrics_in_return_and_tracker(self) -> None:
        # Arrange: old log-probs are 0 and sampling log-probs are -3 on every
        # action token, so the trainer/vLLM ratio clamps at the cap of 2.0.
        algo = _Stub(beta=0.0)

        # Act
        metrics = algo.learn(_experiences(), sampling_logps=_sampling_logps())

        # Assert
        assert metrics["vllm_is_delta_mean"] == pytest.approx(3.0)
        assert metrics["vllm_is_delta_max"] == pytest.approx(3.0)
        assert metrics["vllm_is_ratio_mean"] == pytest.approx(2.0)
        assert metrics["vllm_is_ratio_p95"] == pytest.approx(2.0)
        assert metrics["vllm_is_frac_clamped"] == pytest.approx(1.0)
        assert metrics["vllm_is_rows_skipped"] == 0.0
        assert algo.metrics.logged["vllm_is_delta_mean"] == pytest.approx(3.0)
        assert algo.metrics.logged["vllm_is_ratio_mean"] == pytest.approx(2.0)


PHASE_KEYS = {f"learn_phase_{phase}_s" for phase in LEARN_PHASES}


class _SlowOldLogprobsStub(_Stub):
    """Stub whose no-grad old log-prob forward takes a known wall time."""

    def _fused_forward_no_grad(self, ids: torch.Tensor, _batch_size: int, **kwargs):
        time.sleep(0.05)
        return super()._fused_forward_no_grad(ids, _batch_size, **kwargs)


class TestLearnPhaseTimings:
    """Every ``learn`` return carries wall seconds per learn phase."""

    def test_full_learn_reports_and_logs_every_phase(self) -> None:
        # Arrange
        algo = _Stub(beta=0.0)

        # Act
        metrics = algo.learn(_experiences())

        # Assert
        assert PHASE_KEYS <= set(metrics)
        assert all(metrics[key] >= 0.0 for key in PHASE_KEYS)
        assert PHASE_KEYS <= set(algo.metrics.logged)
        assert algo.shard_runtime.phase_timer.marks is None

    def test_old_logprob_forward_time_lands_in_its_phase(self) -> None:
        # Arrange: one-row micro-batches leave the second row for the no-grad
        # forward. The stub backward pass marks nothing.
        algo = _SlowOldLogprobsStub(beta=0.0)
        algo.micro_batch_size_per_gpu = 1

        # Act
        metrics = algo.learn(_experiences())

        # Assert
        assert algo.no_grad_forwards == [(1, False, True)]
        # Windows sleep can wake just short of 50 ms on the perf_counter clock.
        assert metrics["learn_phase_no_grad_forward_s"] >= 0.049
        assert metrics["learn_phase_forward_s"] < 0.05
        assert metrics["learn_phase_backward_s"] == 0.0

    def test_an_emptied_batch_reports_every_phase_and_closes_the_timer(self) -> None:
        # Arrange
        algo = _Stub(survivors=0)

        # Act
        with pytest.warns(UserWarning, match="advantage threshold"):
            metrics = algo.learn(_experiences())

        # Assert
        assert PHASE_KEYS <= set(metrics)
        assert metrics["learn_phase_forward_s"] == 0.0
        assert algo.shard_runtime.phase_timer.marks is None


class TestLearnAdvantageStats:
    """Advantage stats count action tokens only."""

    def test_token_shape_ignores_masked_positions(self) -> None:
        # Arrange
        algo = _Stub()
        algo._advantages = torch.tensor([[2.0, -4.0, 0.0]])
        experiences = (
            [torch.full((1, 4), PAD_TOKEN_ID + 1, dtype=torch.long)],
            [torch.tensor([[True, True, False]])],
            torch.tensor([1.0], dtype=torch.float32),
        )

        # Act
        metrics = algo.learn(experiences)

        # Assert
        assert metrics["adv_mean"] == pytest.approx(-1.0)
        assert metrics["adv_min"] == pytest.approx(-4.0)
        assert metrics["adv_max"] == pytest.approx(2.0)
        assert metrics["adv_zero_frac"] == pytest.approx(0.0)

    def test_zero_fraction_counts_samples_within_filter_eps(self) -> None:
        # Arrange
        algo = _Stub()
        algo.adv_filter_eps = 0.1
        algo._advantages = torch.tensor([[0.05], [3.0]])

        # Act
        metrics = algo.learn(_experiences(2))

        # Assert
        assert metrics["adv_mean"] == pytest.approx(1.525)
        assert metrics["adv_zero_frac"] == pytest.approx(0.5)


def _ragged_experiences():
    """One full-length row and one half-length row: a quarter of the batch pads."""
    half = SEQ_LEN // 2
    completion_ids = [
        torch.full((1, SEQ_LEN), PAD_TOKEN_ID + 1, dtype=torch.long),
        torch.full((1, half), PAD_TOKEN_ID + 1, dtype=torch.long),
    ]
    action_masks = [
        torch.ones(1, SEQ_LEN - 1, dtype=torch.bool),
        torch.ones(1, half - 1, dtype=torch.bool),
    ]
    rewards = torch.tensor([1.0, -1.0], dtype=torch.float32)
    return completion_ids, action_masks, rewards


class TestLearnPaddingStats:
    """Padding share of the batch before and after sequence packing."""

    def test_unpacked_forward_keeps_the_batch_padding(self) -> None:
        # Arrange
        algo = _Stub()

        # Act
        metrics = algo.learn(_ragged_experiences())

        # Assert
        assert metrics["padding_frac_before_packing"] == pytest.approx(0.25)
        assert metrics["padding_frac_after_packing"] == pytest.approx(0.25)
        assert algo.metrics.logged["padding_frac_after_packing"] == pytest.approx(0.25)

    def test_packed_forward_runs_without_padding(self) -> None:
        # Arrange
        algo = _Stub()
        algo.packing_mode = "varlen"

        # Act
        metrics = algo.learn(_ragged_experiences())

        # Assert
        assert metrics["padding_frac_before_packing"] == pytest.approx(0.25)
        assert metrics["padding_frac_after_packing"] == pytest.approx(0.0)

    def test_filtered_out_batch_still_reports_padding(self) -> None:
        # Arrange
        algo = _Stub(survivors=0)

        # Act
        metrics = algo.learn(_ragged_experiences())

        # Assert
        assert metrics["padding_frac_before_packing"] == pytest.approx(0.25)

    def test_unsegmented_batch_runs_every_row_and_no_filler(self) -> None:
        # Arrange
        algo = _Stub()

        # Act
        metrics = algo.learn(_ragged_experiences())

        # Assert
        assert metrics["train_rows_padded"] == pytest.approx(2.0)
        assert metrics["filler_token_frac"] == pytest.approx(0.0)
        assert algo.metrics.logged["train_rows_padded"] == pytest.approx(2.0)


class TestSummarizePostUpdate:
    """End-of-learn entropy, KL and importance-ratio diagnostics."""

    def test_reports_entropy_kl_and_ratio_tails(self) -> None:
        # Arrange: ratios exp([0, -1, 0.5, -3]) straddle the [0.8, 1.2] clip
        # band, with a positive advantage so only the upper clip binds.
        algo = _Stub()
        post = torch.tensor([[-1.0, -2.0, -0.5, -4.0]])
        old = torch.tensor([[-1.0, -1.0, -1.0, -1.0]])
        ref = torch.tensor([[-1.0, -1.0, -1.0, -1.0]])
        masks = torch.ones(1, 4, dtype=torch.bool)
        advantages = torch.tensor([[1.0]])

        # Act
        stats = algo._summarize_update(post, old, ref, masks, advantages, None)

        # Assert
        assert stats["entropy"] == pytest.approx(1.875)
        assert stats["kl_ref"] == pytest.approx(4.2276, rel=1e-4)
        assert stats["kl_old"] == pytest.approx(4.2276, rel=1e-4)
        assert stats["is_ratio_mean"] == pytest.approx(0.7666, rel=1e-4)
        assert stats["is_ratio_p05"] == pytest.approx(0.0975, rel=1e-3)
        assert stats["is_ratio_p50"] == pytest.approx(0.6839, rel=1e-3)
        assert stats["is_ratio_p95"] == pytest.approx(1.5514, rel=1e-3)
        assert stats["is_frac_below"] == pytest.approx(0.5)
        assert stats["is_frac_above"] == pytest.approx(0.25)
        assert stats["is_frac_clip_pos"] == pytest.approx(0.25)
        assert stats["is_frac_clip_neg"] == pytest.approx(0.0)

    def test_negative_advantage_binds_the_lower_tail(self) -> None:
        # Arrange
        algo = _Stub()
        post = torch.tensor([[-4.0]])
        old = torch.tensor([[-1.0]])
        ref = torch.tensor([[-1.0]])
        masks = torch.ones(1, 1, dtype=torch.bool)
        advantages = torch.tensor([[-2.0]])

        # Act
        stats = algo._summarize_update(post, old, ref, masks, advantages, None)

        # Assert
        assert stats["is_frac_below"] == pytest.approx(1.0)
        assert stats["is_frac_clip_neg"] == pytest.approx(1.0)
        assert stats["is_frac_clip_pos"] == pytest.approx(0.0)

    def test_cispo_binds_no_lower_tail(self) -> None:
        # Arrange: CISPO clips from above only, so lower-tail mass never binds.
        algo = _Stub()
        algo.loss_type = "cispo"
        post = torch.tensor([[-4.0]])
        old = torch.tensor([[-1.0]])
        ref = torch.tensor([[-1.0]])
        masks = torch.ones(1, 1, dtype=torch.bool)
        advantages = torch.tensor([[-2.0]])

        # Act
        stats = algo._summarize_update(post, old, ref, masks, advantages, None)

        # Assert
        assert stats["is_frac_below"] == pytest.approx(1.0)
        assert stats["is_frac_clip_neg"] == pytest.approx(0.0)


class TestComputePolicyLossClipfrac:
    """The standard path reports the kernel's binding-clip definition."""

    def test_grpo_counts_both_signed_tails(self) -> None:
        # Arrange: ratios [0.36, 2.72] against [0.8, 1.2] with advantages
        # of both signs: lower binds on the negative row only.
        algo = _Stub()
        mask = torch.tensor([[True, True], [True, True]])
        log_probs = torch.tensor([[-1.0, 1.0], [-1.0, 1.0]])
        old_log_probs = torch.zeros(2, 2)
        ref_log_probs = torch.zeros(2, 2)
        advantages = torch.tensor([[1.0], [-1.0]])

        # Act
        _, _, clipfrac = algo._compute_policy_loss(
            mask,
            log_probs,
            old_log_probs,
            ref_log_probs,
            advantages,
            None,
            level="token",
            objective="grpo",
        )

        # Assert: row 0 binds above, row 1 binds below: 2 of 4 tokens.
        assert clipfrac.item() == pytest.approx(0.5)

    def test_cispo_counts_upper_clips_only(self) -> None:
        # Arrange: same ratios; CISPO has no lower bound.
        algo = _Stub()
        mask = torch.tensor([[True, True], [True, True]])
        log_probs = torch.tensor([[-1.0, 1.0], [-1.0, 1.0]])
        old_log_probs = torch.zeros(2, 2)
        ref_log_probs = torch.zeros(2, 2)
        advantages = torch.tensor([[1.0], [-1.0]])

        # Act
        _, _, clipfrac = algo._compute_policy_loss(
            mask,
            log_probs,
            old_log_probs,
            ref_log_probs,
            advantages,
            None,
            level="token",
            objective="cispo",
        )

        # Assert: only row 0's upper clip binds: 1 of 4 tokens.
        assert clipfrac.item() == pytest.approx(0.25)

    def test_trajectory_pooling_averages_token_advantages_per_row(self) -> None:
        # Arrange
        algo = _Stub(importance_sampling_level="trajectory")
        post = torch.tensor([[-1.0, -2.0, -0.5, -4.0]])
        old = torch.tensor([[-1.0, -1.0, -1.0, -1.0]])
        ref = torch.tensor([[-1.0, -1.0, -1.0, -1.0]])
        masks = torch.ones(1, 4, dtype=torch.bool)
        advantages = torch.tensor([[1.0, 1.0, -1.0, -1.0]])

        # Act
        stats = algo._summarize_update(post, old, ref, masks, advantages, None)

        # Assert: pooled ratio exp(mean([0, -1, 0.5, -3])); row-mean
        # advantage is 0 so neither clip binds.
        assert stats["is_ratio_mean"] == pytest.approx(0.4169, rel=1e-4)
        assert stats["is_frac_below"] == pytest.approx(1.0)
        assert stats["is_frac_clip_pos"] == pytest.approx(0.0)
        assert stats["is_frac_clip_neg"] == pytest.approx(0.0)

    def test_empty_action_mask_reports_nan_ratio_stats(self) -> None:
        # Arrange
        algo = _Stub()
        post = torch.zeros(1, 2)
        old = torch.zeros(1, 2)
        ref = torch.zeros(1, 2)
        masks = torch.zeros(1, 2, dtype=torch.bool)
        advantages = torch.tensor([[1.0]])

        # Act
        stats = algo._summarize_update(post, old, ref, masks, advantages, None)

        # Assert
        for key in (
            "is_ratio_mean",
            "is_ratio_p05",
            "is_ratio_p50",
            "is_ratio_p95",
            "is_frac_below",
            "is_frac_above",
            "is_frac_clip_pos",
            "is_frac_clip_neg",
        ):
            assert math.isnan(stats[key])

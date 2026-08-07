# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the auxiliary scalar GRPO reports beside its loss.

The name depends on which loss path an update takes, so it must be fixed by
configuration rather than by the data a step happens to carry: a key that
changes between steps of one run cannot be plotted as a series. Pure CPU: the
loss paths are stubs, only the routing and the reported key are under test.
"""

from __future__ import annotations

import warnings
from contextlib import nullcontext
from typing import Any

import numpy as np
import pytest
import torch

pytest.importorskip("transformers", reason="LLM tests require transformers.")
pytest.importorskip("peft", reason="LLM tests require peft.")

from agilerl.algorithms.core.base import LLMAlgorithm
from agilerl.algorithms.grpo import (
    GRPO,
    LIGER_CLIP_FRACTION_METRIC,
    REFERENCE_KL_METRIC,
)

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
        aux_value: float = 0.25,
    ) -> None:
        self.device = torch.device("cpu")
        self.accelerator = None
        self.beta = beta
        self.use_liger_loss = use_liger_loss
        self.importance_sampling_level = importance_sampling_level
        self._liger_level_supported = liger_level_supported
        self.vllm_importance_sampling_correction = vllm_importance_sampling_correction
        self.vllm_importance_sampling_cap = 2.0
        self.filter_zero_adv = filter_zero_adv
        self.loss_norm = "micro_batch"
        self.pad_token_id = PAD_TOKEN_ID
        self.update_epochs = 1
        self.micro_batch_size_per_gpu = 2
        self._is_correction_liger_warned = False
        self._liger_non_token_warned = False
        self._survivors = survivors
        self._aux_value = aux_value
        self.metrics = _MetricsRecorder()
        self.rng = np.random.default_rng(0)
        self.liger_calls = 0
        self.standard_calls = 0

    aux_metric_name = GRPO.aux_metric_name
    learn = GRPO.learn
    _align_sampling_logprobs = GRPO._align_sampling_logprobs
    _aligned_sampling_logprobs_and_metrics = GRPO._aligned_sampling_logprobs_and_metrics
    _liger_path_selected = GRPO._liger_path_selected
    _loss = GRPO._loss
    _objective_loss = GRPO._objective_loss
    _prepare_experience_batch = GRPO._prepare_experience_batch
    _raise_if_loss_not_finite_on_any_rank = (
        LLMAlgorithm._raise_if_loss_not_finite_on_any_rank
    )
    _record_window_action_tokens = GRPO._record_window_action_tokens
    _sampling_mismatch_metrics = GRPO._sampling_mismatch_metrics
    _use_liger_path = GRPO._use_liger_path
    _warn_liger_non_token_is = LLMAlgorithm._warn_liger_non_token_is
    _warn_liger_path_bypassed = GRPO._warn_liger_path_bypassed

    def _prepare_vllm_for_training(self) -> None:
        return

    def memory_efficient_params_context(self):
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
        return torch.ones(rows, 1), np.arange(survivors)

    def _fused_forward_no_grad(self, ids: torch.Tensor, _batch_size: int):
        """Reference and old log-probs on the action frame."""
        zeros = torch.zeros(ids.shape[0], ids.shape[1] - 1)
        return zeros, zeros, None

    def _backward_pass(self, _loss: torch.Tensor) -> None:
        return

    def _liger_loss(self, *_args: Any, **_kwargs: Any):
        """Record that the fused path ran."""
        self.liger_calls += 1
        return torch.tensor(1.0), torch.tensor(self._aux_value)

    def _get_logprobs(self, ids: torch.Tensor, **_kwargs: Any) -> torch.Tensor:
        """Record that the standard path ran."""
        self.standard_calls += 1
        return torch.zeros(ids.shape[0], ids.shape[1] - 1)

    def _loss_fn(self, *_args: Any, **_kwargs: Any):
        """Stand in for the configured standard-path objective."""
        return torch.tensor(1.0), torch.tensor(self._aux_value)


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

    def test_an_unsupported_level_warns_about_memory_not_the_correction(self) -> None:
        algo = _Stub(
            use_liger_loss=True,
            importance_sampling_level="turn",
            liger_level_supported=False,
        )
        with pytest.warns(UserWarning, match="NOT memory-bounded"):
            assert algo._use_liger_path() is False
        assert algo._is_correction_liger_warned is False

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


class TestAuxMetricName:
    """The name states what the reported scalar holds."""

    @pytest.mark.parametrize("beta", [0.0, 0.04])
    def test_the_standard_path_always_names_a_reference_kl(self, beta: float) -> None:
        assert _Stub(beta=beta).aux_metric_name == REFERENCE_KL_METRIC

    def test_the_fused_path_at_zero_beta_names_a_clip_fraction(self) -> None:
        algo = _Stub(beta=0.0, use_liger_loss=True)
        assert algo.aux_metric_name == LIGER_CLIP_FRACTION_METRIC

    def test_the_fused_path_with_a_kl_coefficient_names_a_reference_kl(self) -> None:
        algo = _Stub(beta=0.04, use_liger_loss=True)
        assert algo.aux_metric_name == REFERENCE_KL_METRIC

    def test_a_corrected_non_token_run_names_a_reference_kl(self) -> None:
        algo = _Stub(
            beta=0.0,
            use_liger_loss=True,
            importance_sampling_level="trajectory",
        )
        assert algo.aux_metric_name == REFERENCE_KL_METRIC

    def test_reading_the_name_warns_about_nothing(self) -> None:
        algo = _Stub(
            beta=0.0,
            use_liger_loss=True,
            importance_sampling_level="trajectory",
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert algo.aux_metric_name == REFERENCE_KL_METRIC
        assert caught == []
        assert algo._is_correction_liger_warned is False


class TestLearnReportsOneKeyPerRun:
    """Every ``learn`` return of a run carries the same auxiliary key."""

    def test_the_fused_path_reports_the_clip_fraction_alone(self) -> None:
        algo = _Stub(beta=0.0, use_liger_loss=True)
        metrics = algo.learn(_experiences())
        assert metrics[LIGER_CLIP_FRACTION_METRIC] == pytest.approx(0.25)
        assert REFERENCE_KL_METRIC not in metrics

    def test_the_standard_path_reports_the_kl_alone(self) -> None:
        metrics = _Stub(beta=0.0).learn(_experiences())
        assert metrics[REFERENCE_KL_METRIC] == pytest.approx(0.25)
        assert LIGER_CLIP_FRACTION_METRIC not in metrics

    def test_an_emptied_batch_reports_the_same_key_as_a_full_one(self) -> None:
        algo = _Stub(beta=0.0, use_liger_loss=True, filter_zero_adv=True)
        full = algo.learn(_experiences())
        algo._survivors = 0
        with pytest.warns(UserWarning, match="advantage threshold"):
            emptied = algo.learn(_experiences())
        assert emptied == {"loss": 0.0, LIGER_CLIP_FRACTION_METRIC: 0.0}
        assert set(emptied) <= set(full)

    def test_a_batch_carrying_sampling_logprobs_reports_the_same_key(self) -> None:
        algo = _Stub(
            beta=0.0, use_liger_loss=True, importance_sampling_level="trajectory"
        )
        with pytest.warns(UserWarning, match="only at token-level"):
            corrected = algo.learn(_experiences(), sampling_logps=_sampling_logps())
        uncorrected = algo.learn(_experiences())
        assert REFERENCE_KL_METRIC in corrected
        assert REFERENCE_KL_METRIC in uncorrected
        assert LIGER_CLIP_FRACTION_METRIC not in corrected | uncorrected

    def test_the_reported_key_is_the_one_init_registers(self) -> None:
        algo = _Stub(beta=0.0, use_liger_loss=True)
        metrics = algo.learn(_experiences())
        assert algo.aux_metric_name in metrics
        assert algo.metrics.logged[algo.aux_metric_name] == pytest.approx(0.25)

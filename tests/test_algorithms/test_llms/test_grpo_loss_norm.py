# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for GRPO loss normalization.

Covers ``loss_norm="accumulation_window"`` and ``"micro_batch"`` on the GRPO
loss reduction, on one rank and across ranks. Pure CPU.
"""

from __future__ import annotations

import inspect
import warnings
from typing import Any

import pytest
import torch
import torch.distributed as dist

pytest.importorskip("transformers", reason="LLM tests require transformers.")
pytest.importorskip("peft", reason="LLM tests require peft.")

import numpy as np

from agilerl.algorithms import grpo as grpo_module
from agilerl.algorithms.core.base import LLMAlgorithm
from agilerl.algorithms.grpo import GRPO
from tests.rank_launch import run_ranks

CLIP_MIN = 0.8
CLIP_MAX = 1.2
VOCAB = 5
HIDDEN = 3


class _Stub:
    """Stand-in carrying the state the loss reduction reads, bound to GRPO's methods."""

    def __init__(
        self,
        loss_norm: str = "accumulation_window",
        loss_type: str = "cispo",
        accumulation_steps: int = 4,
        window_tokens: int | None = None,
        lm_head: torch.nn.Linear | None = None,
    ) -> None:
        self.device = torch.device("cpu")
        self.loss_norm = loss_norm
        self.loss_type = loss_type
        self.temperature = 1.0
        self.gradient_accumulation_steps = accumulation_steps
        self._window_action_tokens = window_tokens
        self._segment_accumulation_steps = None
        self._window_micro_batches = None
        self._micro_batch_rows = None
        self.lm_head = lm_head or torch.nn.Linear(HIDDEN, VOCAB, bias=False)

    _resolve_loss_norm = GRPO._resolve_loss_norm
    _record_global_window_action_tokens = GRPO._record_global_window_action_tokens
    _reduce_masked_loss = GRPO._reduce_masked_loss
    _resolve_loss_window = GRPO._resolve_loss_window
    _token_loss_weights = GRPO._token_loss_weights
    _warn_if_micro_batches_straddle_optimizer_steps = (
        GRPO._warn_if_micro_batches_straddle_optimizer_steps
    )


def _mask_of_lengths(lengths: list[int], width: int) -> torch.Tensor:
    """Action-token mask with one row per requested length."""
    mask = torch.zeros(len(lengths), width)
    for row, length in enumerate(lengths):
        mask[row, :length] = 1.0
    return mask


def _micro_batch_reduce(loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-sequence mean over that sequence's own action tokens."""
    return (loss * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1.0)


def _per_token_weights(
    algo: _Stub,
    masks: list[torch.Tensor],
    accumulation_steps: int,
    reduce_fn: Any,
) -> list[torch.Tensor]:
    """Per-token gradient weights the optimizer sees for each micro-batch.

    Each micro-batch loss is divided by ``accumulation_steps`` so the
    accumulated gradient matches a mini-batch mean.
    """
    weights = []
    for mask in masks:
        per_token = torch.zeros_like(mask, requires_grad=True)
        reduced = reduce_fn(algo, per_token, mask)
        (reduced.mean() / accumulation_steps).backward()
        assert per_token.grad is not None
        weights.append(per_token.grad)
    return weights


def _token_log_probs(
    algo: _Stub,
    hidden: torch.Tensor,
    batch_ids: torch.Tensor,
    width: int,
) -> torch.Tensor:
    """Per-token log-probs of the selected ids under the stub's head."""
    logits = algo.lm_head(hidden[:, :width, :]) / algo.temperature
    return torch.log_softmax(logits, dim=-1).gather(
        2,
        batch_ids[:, 1:].unsqueeze(-1),
    )[..., 0]


def _head_inputs(lengths: list[int], width: int, seed: int):
    """Hidden states, token ids and an action mask for one micro-batch."""
    generator = torch.Generator().manual_seed(seed)
    hidden = torch.randn(len(lengths), width + 1, HIDDEN, generator=generator)
    batch_ids = torch.randint(1, VOCAB, (len(lengths), width + 1), generator=generator)
    return hidden, batch_ids, _mask_of_lengths(lengths, width)


class TestGRPOLossNormConfig:
    """The mode is validated at construction and defaults to the window."""

    @pytest.mark.parametrize(
        "loss_norm", ["micro_batch", "accumulation_window", "episode"]
    )
    def test_supported_modes_are_accepted(self, loss_norm: str) -> None:
        assert _Stub()._resolve_loss_norm(loss_norm) == loss_norm

    def test_unknown_mode_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid loss_norm 'per_token'"):
            _Stub()._resolve_loss_norm("per_token")

    def test_default_is_the_accumulation_window(self) -> None:
        default = inspect.signature(GRPO.__init__).parameters["loss_norm"].default
        assert default == "accumulation_window"

    def test_the_window_is_accepted_when_each_micro_batch_is_a_step(self) -> None:
        algo = _Stub(accumulation_steps=1)
        assert algo._resolve_loss_norm("accumulation_window") == "accumulation_window"

    def test_the_micro_batch_mode_is_accepted_alongside_accumulation(self) -> None:
        algo = _Stub(accumulation_steps=4)
        assert algo._resolve_loss_norm("micro_batch") == "micro_batch"


class TestWindowNormalizedReduction:
    """The standard path's reduction under ``loss_norm``."""

    def test_per_token_weights_are_equal_across_sequence_lengths(self) -> None:
        lengths = [4, 100]
        width = 128
        masks = [_mask_of_lengths([length], width) for length in lengths]
        window_tokens = sum(lengths)
        algo = _Stub(accumulation_steps=len(lengths), window_tokens=window_tokens)

        weights = _per_token_weights(
            algo,
            masks,
            len(lengths),
            GRPO._reduce_masked_loss,
        )
        short_weight = float(weights[0][0, 0])
        long_weight = float(weights[1][0, 0])
        assert short_weight == pytest.approx(1.0 / window_tokens)
        assert long_weight == pytest.approx(1.0 / window_tokens)

        baseline = _per_token_weights(
            algo,
            masks,
            len(lengths),
            lambda _algo, loss, mask: _micro_batch_reduce(loss, mask),
        )
        ratio = float(baseline[0][0, 0]) / float(baseline[1][0, 0])
        assert ratio == pytest.approx(lengths[1] / lengths[0])

    def test_out_of_mask_tokens_carry_no_weight(self) -> None:
        mask = _mask_of_lengths([3], 8)
        algo = _Stub(accumulation_steps=2, window_tokens=10)
        weights = _per_token_weights(algo, [mask], 2, GRPO._reduce_masked_loss)[0]
        assert torch.all(weights[0, 3:] == 0.0)
        assert torch.all(weights[0, :3] > 0.0)

    def test_accumulated_loss_equals_the_window_token_mean(self) -> None:
        steps = 4
        lengths = [2, 7, 11, 40]
        width = 64
        masks = [_mask_of_lengths([length], width) for length in lengths]
        generator = torch.Generator().manual_seed(0)
        losses = [
            torch.rand(mask.shape, generator=generator) * mask * 3.0 for mask in masks
        ]
        window_tokens = sum(lengths)
        algo = _Stub(accumulation_steps=steps, window_tokens=window_tokens)

        returned = [
            algo._reduce_masked_loss(loss, mask).mean()
            for loss, mask in zip(losses, masks, strict=True)
        ]
        accumulated = float(sum(returned) / steps)
        total = float(
            sum((loss * mask).sum() for loss, mask in zip(losses, masks, strict=True)),
        )
        assert accumulated == pytest.approx(total / window_tokens, rel=1e-6)

    def test_multi_row_micro_batch_normalizes_over_the_window(self) -> None:
        mask = _mask_of_lengths([3, 9], 16)
        loss = torch.full(mask.shape, 0.5) * mask
        window_tokens = 40
        steps = 3
        algo = _Stub(accumulation_steps=steps, window_tokens=window_tokens)
        reduced = algo._reduce_masked_loss(loss, mask).mean()
        expected = steps * float((loss * mask).sum()) / window_tokens
        assert float(reduced) == pytest.approx(expected, rel=1e-6)

    def test_single_accumulation_step_normalizes_by_the_recorded_window(self) -> None:
        mask = _mask_of_lengths([4, 6], 8)
        loss = torch.full(mask.shape, 2.0) * mask
        algo = _Stub(accumulation_steps=1)
        algo._record_global_window_action_tokens(mask, np.arange(2))

        reduced = algo._reduce_masked_loss(loss, mask).mean()

        assert float(reduced) == pytest.approx(2.0)

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="sync debug mode needs CUDA"
    )
    def test_single_accumulation_step_reduces_without_a_host_sync(self) -> None:
        # Arrange
        device = torch.device("cuda")
        mask = _mask_of_lengths([4, 6], 8).to(device)
        loss = torch.full(mask.shape, 2.0, device=device) * mask
        algo = _Stub(accumulation_steps=1)
        algo.device = device
        algo._record_global_window_action_tokens(mask, np.arange(2))

        # Act
        torch.cuda.set_sync_debug_mode("error")
        try:
            reduced = algo._reduce_masked_loss(loss, mask).mean()
        finally:
            torch.cuda.set_sync_debug_mode("default")

        # Assert
        assert float(reduced) == pytest.approx(2.0)

    def test_micro_batch_mode_keeps_the_per_sequence_mean(self) -> None:
        mask = _mask_of_lengths([3, 9], 16)
        generator = torch.Generator().manual_seed(4)
        loss = torch.rand(mask.shape, generator=generator) * mask
        algo = _Stub(loss_norm="micro_batch", accumulation_steps=4, window_tokens=40)
        reduced = algo._reduce_masked_loss(loss, mask)
        assert torch.allclose(reduced, _micro_batch_reduce(loss, mask))

    def test_a_per_sequence_loss_spreads_over_its_action_tokens(self) -> None:
        mask = _mask_of_lengths([3, 9], 16)
        loss = torch.tensor([[0.5], [-2.0]])
        algo = _Stub(loss_norm="micro_batch")

        shares = algo._reduce_masked_loss(loss, mask)

        assert shares.tolist() == pytest.approx([0.5, -2.0])

    def test_non_finite_padding_stays_out_of_the_window_reduction(self) -> None:
        mask = torch.tensor([[1.0, 1.0, 0.0]])
        loss = torch.tensor([[2.0, 4.0, float("nan")]])
        algo = _Stub(accumulation_steps=1, window_tokens=2)
        reduced = algo._reduce_masked_loss(loss, mask)
        assert reduced.tolist() == pytest.approx([3.0])

    def test_missing_window_token_count_raises(self) -> None:
        mask = _mask_of_lengths([4], 8)
        algo = _Stub(accumulation_steps=4)
        with pytest.raises(RuntimeError, match="no recorded window action-token"):
            algo._reduce_masked_loss(torch.zeros(mask.shape), mask)

    def test_window_without_action_tokens_gives_zero_shares(self) -> None:
        mask = torch.zeros(2, 8)
        algo = _Stub(accumulation_steps=4)
        algo._record_global_window_action_tokens(mask, np.arange(2))

        shares = algo._reduce_masked_loss(torch.ones(mask.shape), mask)

        assert shares.tolist() == [0.0, 0.0]

    def test_empty_micro_batch_without_accumulation_gives_zero_shares(self) -> None:
        mask = torch.zeros(1, 8)
        algo = _Stub(accumulation_steps=1)
        algo._record_global_window_action_tokens(mask, np.arange(1))

        shares = algo._reduce_masked_loss(torch.ones(mask.shape), mask)

        assert shares.tolist() == [0.0]


class TestGRPOTokenLossWeights:
    """The per-token weights both loss paths reduce with."""

    def test_micro_batch_rows_each_spread_one_over_the_batch(self) -> None:
        mask = _mask_of_lengths([2, 5], 8)
        algo = _Stub(loss_norm="micro_batch")

        weights = algo._token_loss_weights(mask, None)

        assert weights[0, :2].tolist() == pytest.approx([1 / 4, 1 / 4])
        assert weights[1, :5].tolist() == pytest.approx([1 / 10] * 5)
        assert weights.sum(dim=-1).tolist() == pytest.approx([0.5, 0.5])
        assert torch.all(weights[mask == 0] == 0.0)

    def test_window_tokens_weigh_steps_over_window_tokens(self) -> None:
        mask = _mask_of_lengths([2, 5], 8)
        algo = _Stub(accumulation_steps=3, window_tokens=30)

        weights = algo._token_loss_weights(mask, algo._resolve_loss_window())

        assert torch.allclose(weights, mask * (3 / 30))

    def test_rows_without_action_tokens_weigh_nothing(self) -> None:
        mask = _mask_of_lengths([0, 4], 8)
        algo = _Stub(loss_norm="micro_batch")

        weights = algo._token_loss_weights(mask, None)

        assert weights[0].tolist() == [0.0] * 8
        assert weights[1, :4].tolist() == pytest.approx([1 / 8] * 4)


class TestAccumulationSteps:
    """The step count comes from the trainer loop that applies the scaling."""

    def test_the_configured_width_is_reported(self) -> None:
        algo = _Stub(accumulation_steps=6)
        assert algo.gradient_accumulation_steps == 6

    def test_a_single_step_window_reports_one(self) -> None:
        algo = _Stub(accumulation_steps=1)
        assert algo.gradient_accumulation_steps == 1


class TestStraddleWarning:
    """A trailing micro-batch is worth warning about; a full window is not."""

    @staticmethod
    def _emitted(algo: _Stub, samples: int, micro_batch_size: int) -> list:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            algo._warn_if_micro_batches_straddle_optimizer_steps(
                samples, micro_batch_size
            )
        return caught

    def test_a_trailing_micro_batch_warns(self) -> None:
        algo = _Stub(accumulation_steps=2)
        with pytest.warns(UserWarning, match="whole optimizer steps"):
            algo._warn_if_micro_batches_straddle_optimizer_steps(3, 1)

    def test_micro_batches_filling_whole_steps_stay_silent(self) -> None:
        assert self._emitted(_Stub(accumulation_steps=2), 4, 1) == []

    def test_a_single_step_window_stays_silent(self) -> None:
        """Every micro-batch takes its own step, so none can straddle one."""
        assert self._emitted(_Stub(accumulation_steps=1), 3, 2) == []


class TestGRPORecordGlobalWindowActionTokens:
    """The window counts the action tokens of its own rows only."""

    def test_only_the_window_rows_are_counted(self) -> None:
        algo = _Stub()
        action_masks = _mask_of_lengths([5, 9, 2], 16)

        algo._record_global_window_action_tokens(action_masks, np.array([0, 2]))

        assert algo._window_action_tokens.item() == 5 + 2

    def test_the_count_stays_a_device_tensor(self) -> None:
        algo = _Stub()
        action_masks = _mask_of_lengths([5, 9, 2], 16)

        algo._record_global_window_action_tokens(action_masks, np.arange(3))

        assert isinstance(algo._window_action_tokens, torch.Tensor)
        assert algo._window_action_tokens.device == action_masks.device
        assert algo._window_action_tokens.item() == 5 + 9 + 2

    def test_a_window_without_action_tokens_counts_one(self) -> None:
        algo = _Stub()

        algo._record_global_window_action_tokens(torch.zeros(2, 16), np.arange(2))

        assert algo._window_action_tokens.item() == 1.0


class TestActivationOffloadContext:
    """GRPO takes ``activation_offload`` from the shared LLM algorithm."""

    def test_the_context_comes_from_the_algorithm_hierarchy(self) -> None:
        assert GRPO._activation_offload_ctx is LLMAlgorithm._activation_offload_ctx
        assert "activation_offload" not in inspect.signature(GRPO.__init__).parameters


class TestNormalizerWorldSize:
    def test_returns_one_when_distributed_inactive(self, monkeypatch) -> None:
        monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)

        assert grpo_module._normalizer_world_size() == 1

    def test_uses_process_group_world_size(self, monkeypatch) -> None:
        monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
        monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 4)

        assert grpo_module._normalizer_world_size() == 4


RANK_WIDTH = 16
RANK_TIMEOUT_S = 120


WINDOW_RANK_LENGTHS = {
    "uneven": ([[0], [3]], [[9, 4], [7, 1]]),
    "empty": ([[0], [0]], [[0, 0], [0]]),
}
"""Per-scenario, per-rank micro-batch action lengths of one accumulation window."""


def _window_micro_batch(rank: int, step: int, lengths: list[int]):
    """Hidden states, token ids, action mask and advantages of one micro-batch."""
    hidden, batch_ids, mask = _head_inputs(lengths, RANK_WIDTH, seed=10 * rank + step)
    advantages = torch.linspace(-0.8, 0.6, len(lengths)).unsqueeze(-1) + rank
    return hidden, batch_ids, mask, advantages


def _window_token_mean_rank(rank: int, world_size: int, scenario: str) -> None:
    """One rank of an accumulation window on the GRPO loss reduction."""
    windows = WINDOW_RANK_LENGTHS[scenario]
    union = [
        _window_micro_batch(other, step, lengths)
        for other, rank_window in enumerate(windows)
        for step, lengths in enumerate(rank_window)
    ]
    own = [
        _window_micro_batch(rank, step, lengths)
        for step, lengths in enumerate(windows[rank])
    ]
    steps = len(own)
    torch.manual_seed(0)
    initial_head = torch.nn.Linear(HIDDEN, VOCAB, bias=False)
    union_head = torch.nn.Linear(HIDDEN, VOCAB, bias=False)
    union_head.load_state_dict(initial_head.state_dict())
    algo = _Stub(loss_type="cispo", accumulation_steps=steps, lm_head=initial_head)
    window_mask = torch.cat([mask for _, _, mask, _ in own])
    algo._record_global_window_action_tokens(
        window_mask, np.arange(window_mask.shape[0])
    )

    losses = []
    for hidden, batch_ids, mask, advantages in own:
        log_probs = _token_log_probs(algo, hidden, batch_ids, RANK_WIDTH)
        loss = algo._reduce_masked_loss(-advantages * log_probs, mask).mean()
        (loss / steps).backward()
        losses.append(loss.detach())
    grad = initial_head.weight.grad
    dist.all_reduce(grad, op=dist.ReduceOp.SUM)
    grad /= world_size

    reference = _Stub(loss_norm="micro_batch", lm_head=union_head)
    union_sum = torch.zeros(())
    union_tokens = 0.0
    for hidden, batch_ids, mask, advantages in union:
        log_probs = _token_log_probs(reference, hidden, batch_ids, RANK_WIDTH)
        union_sum = union_sum + (-advantages * log_probs * mask).sum()
        union_tokens += float(mask.sum())
    (union_sum / max(union_tokens, 1.0)).backward()
    expected = union_head.weight.grad

    assert all(torch.isfinite(loss) for loss in losses), (rank, losses)
    assert torch.allclose(grad, expected, atol=1e-6), (rank, grad, expected)
    if union_tokens == 0.0:
        assert all(loss.item() == 0.0 for loss in losses), (rank, losses)


@pytest.mark.multi_rank
class TestGRPOAccumulationWindowAcrossRanks:
    """The rank-averaged window gradient is the token mean over every rank's tokens."""

    @pytest.mark.parametrize("scenario", sorted(WINDOW_RANK_LENGTHS))
    def test_averaged_gradient_is_the_union_token_mean(self, scenario: str) -> None:
        run_ranks(
            _window_token_mean_rank,
            len(WINDOW_RANK_LENGTHS[scenario]),
            scenario,
            timeout=RANK_TIMEOUT_S,
        )

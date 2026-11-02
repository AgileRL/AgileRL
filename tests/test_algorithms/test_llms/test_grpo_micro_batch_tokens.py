# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for GRPO learn micro-batches sized by a per-GPU token budget.

Pure CPU on a tiny real model in fp32.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

pytest.importorskip("transformers", reason="LLM tests require transformers.")
pytest.importorskip("peft", reason="LLM tests require peft.")

from agilerl.algorithms.grpo import GRPO
from tests.test_algorithms.test_llms.segment_helpers import record_step_gradients
from tests.test_algorithms.test_llms.test_grpo_old_logprobs import (
    PAD_TOKEN_ID,
    _make_grpo,
)

# fp32 on CPU; packing only regroups the same per-row matmuls and sums.
RTOL = 1e-5
ATOL = 1e-6


def record_forward_tokens(agent: GRPO, grad: bool = True) -> list[int]:
    """Real tokens of every actor forward with or without gradients from now on."""
    tokens: list[int] = []

    def record(
        _module: torch.nn.Module, _args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        if torch.is_grad_enabled() == grad:
            tokens.append(int((kwargs["input_ids"] != PAD_TOKEN_ID).sum()))

    agent.actor.register_forward_pre_hook(record, with_kwargs=True)
    return tokens


def learn_uneven_rows(
    agent: GRPO,
    rewards: tuple[float, ...] = (1.0, 0.0, 0.5, -1.0),
    **learn_kwargs: Any,
) -> dict[str, Any]:
    """One learn over four rows with 8, 4, 6 and 3 real tokens."""
    generator = torch.Generator().manual_seed(3)
    ids = torch.randint(0, PAD_TOKEN_ID, (4, 8), generator=generator)
    mask = torch.zeros(4, 7, dtype=torch.bool)
    for row, length in enumerate([8, 4, 6, 3]):
        ids[row, length:] = PAD_TOKEN_ID
        mask[row, 1 : length - 1] = True
    return agent.learn(
        (list(ids.split(1)), list(mask.split(1)), torch.tensor(rewards)),
        **learn_kwargs,
    )


def make_budgeted_grpo(monkeypatch: pytest.MonkeyPatch, **overrides: Any) -> GRPO:
    """GRPO packing varlen micro-batches of at most 12 tokens."""
    agent = _make_grpo(
        micro_batch_size_per_gpu=4, micro_batch_tokens_per_gpu=12, **overrides
    )
    monkeypatch.setattr(agent, "_packing_mode", lambda: "varlen")
    return agent


# The second group ties, so its rows have zero advantages.
TIED_SECOND_GROUP = (1.0, 0.0, 0.5, 0.5)


LOSSES = pytest.mark.parametrize(
    "loss",
    [
        {"loss_type": "grpo", "importance_sampling_level": "token"},
        {"loss_type": "gspo"},
        {"loss_type": "cispo"},
    ],
    ids=["grpo-token", "gspo", "cispo"],
)


class TestGRPOLearnTokenBudget:
    @LOSSES
    def test_step_gradients_match_one_window_micro_batch(
        self, monkeypatch: pytest.MonkeyPatch, loss: dict[str, str]
    ) -> None:
        # Arrange: a 12-token budget splits the window into micro-batches of
        # unequal tokens; the reference runs the whole window as one.
        window = _make_grpo(micro_batch_size_per_gpu=4, **loss)
        budgeted = _make_grpo(
            micro_batch_size_per_gpu=4, micro_batch_tokens_per_gpu=12, **loss
        )
        monkeypatch.setattr(budgeted, "_packing_mode", lambda: "varlen")
        forward_tokens = record_forward_tokens(budgeted)
        window_steps = record_step_gradients(window, monkeypatch)
        budgeted_steps = record_step_gradients(budgeted, monkeypatch)

        # Act
        window_metrics = learn_uneven_rows(window)
        budgeted_metrics = learn_uneven_rows(budgeted)

        # Assert
        assert len(forward_tokens) >= 2
        assert max(forward_tokens) <= 12
        assert len(window_steps) == len(budgeted_steps) == 1
        assert budgeted_steps[0].keys() == window_steps[0].keys()
        for name, grad in window_steps[0].items():
            torch.testing.assert_close(
                budgeted_steps[0][name], grad, rtol=RTOL, atol=ATOL, msg=name
            )
        assert budgeted_metrics["loss"] == pytest.approx(
            window_metrics["loss"], rel=RTOL, abs=ATOL
        )
        assert budgeted_metrics["micro_batches_per_step"] == len(forward_tokens)
        assert window_metrics["micro_batches_per_step"] == 1.0

    @LOSSES
    def test_zero_advantage_rows_run_no_forward_and_leave_gradients_unchanged(
        self, monkeypatch: pytest.MonkeyPatch, loss: dict[str, str]
    ) -> None:
        # Arrange: the unbudgeted window runs every row in one micro-batch.
        window = _make_grpo(micro_batch_size_per_gpu=4, **loss)
        budgeted = make_budgeted_grpo(monkeypatch, **loss)
        forward_tokens = record_forward_tokens(budgeted)
        window_steps = record_step_gradients(window, monkeypatch)
        budgeted_steps = record_step_gradients(budgeted, monkeypatch)

        # Act
        window_metrics = learn_uneven_rows(window, TIED_SECOND_GROUP)
        budgeted_metrics = learn_uneven_rows(budgeted, TIED_SECOND_GROUP)

        # Assert
        assert forward_tokens == [12]
        assert budgeted_metrics["trained_row_tokens"] == [8, 4]
        assert window_metrics["trained_row_tokens"] == [8, 4, 6, 3]
        assert len(window_steps) == len(budgeted_steps) == 1
        assert budgeted_steps[0].keys() == window_steps[0].keys()
        for name, grad in window_steps[0].items():
            torch.testing.assert_close(
                budgeted_steps[0][name], grad, rtol=RTOL, atol=ATOL, msg=name
            )
        assert budgeted_metrics["loss"] == pytest.approx(
            window_metrics["loss"], rel=RTOL, abs=ATOL
        )

    def test_all_zero_advantages_still_step_the_optimizer_on_the_shortest_row(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        agent = make_budgeted_grpo(monkeypatch)
        forward_tokens = record_forward_tokens(agent)
        steps = record_step_gradients(agent, monkeypatch)

        # Act
        metrics = learn_uneven_rows(agent, (0.5, 0.5, 1.0, 1.0))

        # Assert
        assert forward_tokens == [3]
        assert metrics["trained_row_tokens"] == [3]
        assert len(steps) == 1
        assert steps[0]
        for name, grad in steps[0].items():
            assert torch.count_nonzero(grad) == 0, name

    def test_a_kl_penalty_keeps_zero_advantage_rows_in_the_forwards(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        agent = make_budgeted_grpo(monkeypatch, beta=0.1)
        forward_tokens = record_forward_tokens(agent)

        # Act
        metrics = learn_uneven_rows(agent, TIED_SECOND_GROUP)

        # Assert
        assert sum(forward_tokens) == 8 + 4 + 6 + 3
        assert metrics["trained_row_tokens"] == [8, 4, 6, 3]

    def test_a_fully_filtered_batch_skips_every_row(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = make_budgeted_grpo(monkeypatch, filter_zero_adv=True)

        with pytest.warns(UserWarning, match="All samples were filtered"):
            metrics = learn_uneven_rows(agent, (0.5, 0.5, 1.0, 1.0))

        assert metrics["trained_row_tokens"] == []

    def test_logged_loss_is_the_mean_window_loss_whatever_the_micro_batch_counts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange: one group of four in two-row windows. Row 3 sits at the
        # group mean, so the trained rows deal into windows [0] and [2, 1].
        # An 8-token budget trains the second window as two micro-batches; a
        # 32-token budget trains each window as one.
        shape = {
            "batch_size": 1,
            "group_size": 4,
            "mini_batch_size": 2,
            "micro_batch_size_per_gpu": 2,
            "max_model_len": 8,
        }
        window = _make_grpo(micro_batch_tokens_per_gpu=32, **shape)
        budgeted = _make_grpo(micro_batch_tokens_per_gpu=8, **shape)
        monkeypatch.setattr(window, "_packing_mode", lambda: "varlen")
        monkeypatch.setattr(budgeted, "_packing_mode", lambda: "varlen")
        window_tokens = record_forward_tokens(window)
        forward_tokens = record_forward_tokens(budgeted)
        rewards = (1.0, 0.0, 0.2, 0.4)

        # Act
        window_metrics = learn_uneven_rows(window, rewards)
        budgeted_metrics = learn_uneven_rows(budgeted, rewards)

        # Assert
        assert window_tokens == [8, 10]
        assert forward_tokens == [8, 6, 4]
        assert window_metrics["loss"] != 0.0
        assert budgeted_metrics["loss"] == pytest.approx(
            window_metrics["loss"], rel=RTOL, abs=ATOL
        )

    def test_rollout_old_logprobs_skip_the_no_grad_forward_of_skipped_rows(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange: no row has sampling log-probs, so every trained row needs
        # learn-start log-probs from the no-grad forward.
        agent = make_budgeted_grpo(monkeypatch, old_logprobs_source="rollout")
        no_grad_tokens = record_forward_tokens(agent, grad=False)

        # Act
        learn_uneven_rows(agent, TIED_SECOND_GROUP)

        # Assert
        assert sum(no_grad_tokens) == 8 + 4

    def test_trainer_old_logprobs_skip_the_no_grad_forward_of_skipped_rows(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange: rows 0 and 1 score their own old log-probs in the first
        # micro-batch. Rows 2 and 3 carry sampling log-probs of the wrong
        # length, which the mismatch metrics count only on trained rows.
        agent = make_budgeted_grpo(monkeypatch, old_logprobs_source="trainer")
        no_grad_tokens = record_forward_tokens(agent, grad=False)
        sampling_logps = [
            torch.full((6,), -1.0),
            torch.full((2,), -1.0),
            torch.full((1,), -1.0),
            torch.full((1,), -1.0),
        ]

        # Act
        metrics = learn_uneven_rows(
            agent, TIED_SECOND_GROUP, sampling_logps=sampling_logps
        )

        # Assert
        assert no_grad_tokens == []
        assert metrics["trained_row_tokens"] == [8, 4]
        assert metrics["vllm_is_rows_skipped"] == 0.0

    def test_filtered_rows_run_no_forward_and_leave_the_trained_rows(
        self,
    ) -> None:
        # Arrange: one process drops the zero-advantage rows without a budget.
        agent = _make_grpo(
            micro_batch_size_per_gpu=4,
            filter_zero_adv=True,
            old_logprobs_source="rollout",
        )
        grad_tokens = record_forward_tokens(agent)
        no_grad_tokens = record_forward_tokens(agent, grad=False)

        # Act
        metrics = learn_uneven_rows(agent, TIED_SECOND_GROUP)

        # Assert
        assert grad_tokens == [8 + 4]
        assert sum(no_grad_tokens) == 8 + 4
        assert metrics["trained_row_tokens"] == [8, 4]

    def test_budget_below_the_longest_row_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"micro_batch_tokens_per_gpu \(11\)"):
            _make_grpo(micro_batch_tokens_per_gpu=11)

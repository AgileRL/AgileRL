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


def record_grad_forward_tokens(agent: GRPO) -> list[int]:
    """Real tokens of every gradient-bearing actor forward from now on."""
    tokens: list[int] = []

    def record(
        _module: torch.nn.Module, _args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        if torch.is_grad_enabled():
            tokens.append(int((kwargs["input_ids"] != PAD_TOKEN_ID).sum()))

    agent.actor.register_forward_pre_hook(record, with_kwargs=True)
    return tokens


def learn_uneven_rows(agent: GRPO) -> dict[str, float]:
    """One learn over two groups of two rows with 8, 4, 6 and 3 real tokens."""
    generator = torch.Generator().manual_seed(3)
    ids = torch.randint(0, PAD_TOKEN_ID, (4, 8), generator=generator)
    mask = torch.zeros(4, 7, dtype=torch.bool)
    for row, length in enumerate([8, 4, 6, 3]):
        ids[row, length:] = PAD_TOKEN_ID
        mask[row, 1 : length - 1] = True
    rewards = torch.tensor([1.0, 0.0, 0.5, -1.0])
    return agent.learn((list(ids.split(1)), list(mask.split(1)), rewards))


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
        forward_tokens = record_grad_forward_tokens(budgeted)
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

    def test_budget_below_the_longest_row_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"micro_batch_tokens_per_gpu \(11\)"):
            _make_grpo(micro_batch_tokens_per_gpu=11)

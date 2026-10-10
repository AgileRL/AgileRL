# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for GRPO ``learn`` refusing to step on a non-finite loss.

Pure CPU on a tiny real model in fp32.
"""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import patch

import pytest
import torch

pytest.importorskip("transformers", reason="LLM tests require transformers.")
pytest.importorskip("peft", reason="LLM tests require peft.")

from agilerl.algorithms.grpo import GRPO
from tests.test_algorithms.test_llms.llm_helpers import optimizer_state
from tests.test_algorithms.test_llms.test_grpo_old_logprobs import (
    _experiences,
    _make_grpo,
)


def scaled_loss(agent: GRPO, scales: Iterator[float]):
    """``_loss`` stand-in whose loss and gradients carry the next scale."""
    trainable = [p for p in agent.actor.parameters() if p.requires_grad]

    def loss(*_args, **_kwargs):
        zero = torch.tensor(0.0)
        return sum(p.sum() for p in trainable) * next(scales), zero, zero, zero

    return loss


def trainable_weights(agent: GRPO) -> list[torch.Tensor]:
    return [p.detach().clone() for p in agent.actor.parameters() if p.requires_grad]


class TestGRPOLearnNonFiniteLoss:
    def test_finite_losses_step_every_window(self) -> None:
        # Arrange
        agent = _make_grpo(micro_batch_size_per_gpu=1, mini_batch_size=2)
        before = trainable_weights(agent)

        # Act
        with patch.object(
            agent, "_loss", side_effect=scaled_loss(agent, iter([1.0] * 4))
        ):
            agent.learn(_experiences())

        # Assert
        after = trainable_weights(agent)
        assert any(
            not torch.equal(new, old) for new, old in zip(after, before, strict=True)
        )

    @pytest.mark.parametrize(
        "window_scales",
        [[float("nan"), 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, float("nan")]],
        ids=["first_micro_batch", "last_micro_batch"],
    )
    def test_raises_before_the_step_when_a_micro_batch_loss_is_not_finite(
        self, window_scales: list[float]
    ) -> None:
        # Arrange: one optimizer step of four 1-row micro-batches per learn; a
        # finite learn first gives the optimizer state to keep.
        agent = _make_grpo(micro_batch_size_per_gpu=1, mini_batch_size=4)
        scales = iter([1.0] * 4 + window_scales)
        with patch.object(agent, "_loss", side_effect=scaled_loss(agent, scales)):
            agent.learn(_experiences())
            before = trainable_weights(agent)
            state_before = optimizer_state(agent.optimizer)

            # Act
            with pytest.raises(ValueError, match="Loss is not finite"):
                agent.learn(_experiences())

        # Assert: it raises at the window's last micro-batch, before the step.
        assert list(scales) == []
        for new, old in zip(trainable_weights(agent), before, strict=True):
            assert torch.equal(new, old)
        state_after = optimizer_state(agent.optimizer)
        assert state_before
        assert len(state_after) == len(state_before)
        for new, old in zip(state_after, state_before, strict=True):
            assert torch.equal(new, old)

    def test_raises_when_the_micro_batch_left_pending_a_step_is_not_finite(
        self,
    ) -> None:
        # Arrange: three micro-batches per step over four rows leave the last
        # micro-batch's gradients pending the next learn call.
        agent = _make_grpo(micro_batch_size_per_gpu=1)
        agent.gradient_accumulation_steps = 3
        scales = iter([1.0, 1.0, 1.0, float("nan")])

        # Act / Assert
        with (
            patch.object(agent, "_loss", side_effect=scaled_loss(agent, scales)),
            pytest.warns(UserWarning, match="whole optimizer steps"),
            pytest.raises(ValueError, match="Loss is not finite"),
        ):
            agent.learn(_experiences())

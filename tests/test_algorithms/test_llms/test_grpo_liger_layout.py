# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the objective and Liger layout GRPO's memory estimate sizes."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("transformers", reason="LLM tests require transformers.")
pytest.importorskip("peft", reason="LLM tests require peft.")

from agilerl.algorithms.grpo import GRPO
from tests.test_algorithms.test_llms.llm_helpers import create_module


def make_grpo(liger_loss: bool, **overrides: Any) -> GRPO:
    """Tiny GRPO on CPU with ``liger_loss`` pinned."""
    agent = GRPO(
        actor_network=create_module(
            input_size=6, max_tokens=4, vocab_size=64, device="cpu"
        ),
        pad_token_id=63,
        pad_token="<pad>",
        batch_size=4,
        group_size=2,
        max_output_tokens=4,
        max_model_len=12,
        wrap=False,
        device="cpu",
        **overrides,
    )
    agent.liger_loss = liger_loss
    return agent


class TestGRPOEstimatorAlgorithm:
    @pytest.mark.parametrize("loss_type", ["grpo", "gspo", "cispo"])
    def test_estimate_keys_on_the_loss_type(self, loss_type: str) -> None:
        agent = make_grpo(False, loss_type=loss_type)

        assert agent._estimator_algorithm() == loss_type
        agent.clean_up()


class TestGRPOLigerLossLayout:
    @pytest.mark.parametrize(
        ("overrides", "layout"),
        [
            ({}, (True, False)),
            ({"importance_sampling_level": "trajectory"}, (True, True)),
            ({"loss_type": "gspo"}, (True, True)),
            ({"loss_type": "cispo"}, (True, False)),
        ],
        ids=["grpo-token", "grpo-trajectory", "gspo", "cispo-token"],
    )
    def test_trajectory_level_sizes_whole_sequences(
        self, overrides: dict, layout: tuple[bool, bool]
    ) -> None:
        agent = make_grpo(True, **overrides)

        assert agent._liger_loss_layout() == layout
        agent.clean_up()

    def test_standard_loss_sizes_no_sequence_tile(self) -> None:
        agent = make_grpo(False, loss_type="gspo")

        assert agent._liger_loss_layout() == (False, False)
        agent.clean_up()

# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Shared batch probe for LLM ``advantage_granularity="auto"``."""

from __future__ import annotations

import torch

from agilerl.algorithms.core.advantage_granularity import (
    batch_is_single_turn,
    resolve_batch_advantage_granularity,
)


class TestBatchIsSingleTurn:
    def test_missing_turn_ids_is_single_turn(self):
        assert batch_is_single_turn(None) is True

    def test_all_zeros_is_single_turn(self):
        turn_ids = torch.zeros(3, 4, dtype=torch.long)

        assert batch_is_single_turn(turn_ids) is True

    def test_any_sample_with_two_turns_is_multi_turn(self):
        turn_ids = torch.tensor(
            [[0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 0, 0]],
            dtype=torch.long,
        )

        assert batch_is_single_turn(turn_ids) is False


class TestResolveBatchAdvantageGranularity:
    def test_explicit_value_is_passed_through(self):
        turn_ids = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)

        assert (
            resolve_batch_advantage_granularity(
                "turn",
                turn_ids,
                single_turn="token",
                multi_turn="turn",
            )
            == "turn"
        )

    def test_auto_single_turn_uses_single_turn_grain(self):
        turn_ids = torch.zeros(2, 3, dtype=torch.long)

        assert (
            resolve_batch_advantage_granularity(
                "auto",
                turn_ids,
                single_turn="token",
                multi_turn="turn",
            )
            == "token"
        )

    def test_auto_multi_turn_uses_multi_turn_grain(self):
        turn_ids = torch.tensor([[0, 0, 1], [0, 1, 1]], dtype=torch.long)

        assert (
            resolve_batch_advantage_granularity(
                "auto",
                turn_ids,
                single_turn="trajectory",
                multi_turn="turn",
            )
            == "turn"
        )

    def test_auto_multi_turn_without_can_use_falls_back(self):
        turn_ids = torch.tensor([[0, 0, 1], [0, 1, 1]], dtype=torch.long)

        assert (
            resolve_batch_advantage_granularity(
                "auto",
                turn_ids,
                single_turn="trajectory",
                multi_turn="turn",
                can_use_multi_turn=False,
            )
            == "trajectory"
        )

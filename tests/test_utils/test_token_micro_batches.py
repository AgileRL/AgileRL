# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for token-budget micro-batch planning."""

from __future__ import annotations

import numpy as np
import pytest

from agilerl.utils.token_micro_batches import pack_rows_by_tokens, split_micro_batches


def as_lists(micro_batches: list[np.ndarray]) -> list[list[int]]:
    return [rows.tolist() for rows in micro_batches]


class TestPackRowsByTokens:
    def test_rows_pack_in_order_up_to_the_budget(self) -> None:
        row_tokens = np.array([4, 6, 3, 5, 2])

        micro_batches = pack_rows_by_tokens(np.array([0, 2, 1, 3, 4]), row_tokens, 10)

        assert as_lists(micro_batches) == [[0, 2], [1], [3, 4]]

    def test_a_row_at_the_budget_fills_a_micro_batch(self) -> None:
        micro_batches = pack_rows_by_tokens(np.array([0, 1]), np.array([10, 1]), 10)

        assert as_lists(micro_batches) == [[0], [1]]

    def test_a_row_longer_than_the_budget_runs_alone(self) -> None:
        row_tokens = np.array([2, 15, 3])

        micro_batches = pack_rows_by_tokens(np.array([0, 1, 2]), row_tokens, 10)

        assert as_lists(micro_batches) == [[0], [1], [2]]

    def test_no_rows_make_no_micro_batches(self) -> None:
        micro_batches = pack_rows_by_tokens(
            np.zeros(0, dtype=np.intp), np.array([3]), 10
        )

        assert micro_batches == []


class TestSplitMicroBatches:
    def test_the_micro_batch_with_most_tokens_splits_at_its_token_midpoint(
        self,
    ) -> None:
        # Arrange
        row_tokens = np.array([1, 1, 1, 6, 2, 2])
        micro_batches = [np.array([0, 1, 2]), np.array([3, 4, 5])]

        # Act
        split = split_micro_batches(micro_batches, row_tokens, 3)

        # Assert
        assert as_lists(split) == [[0, 1, 2], [3], [4, 5]]

    def test_splits_repeat_until_the_count(self) -> None:
        row_tokens = np.array([2, 2, 2, 2])

        split = split_micro_batches([np.array([0, 1, 2, 3])], row_tokens, 4)

        assert as_lists(split) == [[0], [1], [2], [3]]

    def test_the_count_already_met_keeps_the_micro_batches(self) -> None:
        micro_batches = [np.array([0]), np.array([1, 2])]

        split = split_micro_batches(micro_batches, np.array([1, 1, 1]), 2)

        assert as_lists(split) == [[0], [1, 2]]

    def test_too_few_rows_for_the_count_is_refused(self) -> None:
        with pytest.raises(ValueError, match="Cannot split 2 rows into 3"):
            split_micro_batches([np.array([0]), np.array([1])], np.array([1, 1]), 3)

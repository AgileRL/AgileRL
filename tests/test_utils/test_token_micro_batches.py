# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for token-budget micro-batch planning."""

from __future__ import annotations

import numpy as np
import pytest

from agilerl.utils.token_micro_batches import (
    balance_windows,
    deal_rows_by_tokens,
    fill_micro_batches,
    fits_budget,
    pack_rows_by_tokens,
    pack_rows_evenly,
    split_micro_batches,
)


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


class TestDealRowsByTokens:
    def test_rows_go_most_tokens_first_to_the_bin_with_fewest_tokens(self) -> None:
        row_tokens = np.array([1, 5, 3, 4])

        bins = deal_rows_by_tokens(np.array([0, 1, 2, 3]), row_tokens, [4, 4])

        assert as_lists(bins) == [[1, 0], [3, 2]]

    def test_a_full_bin_takes_no_more_rows(self) -> None:
        row_tokens = np.array([1, 5, 3, 4])

        bins = deal_rows_by_tokens(np.array([0, 1, 2, 3]), row_tokens, [1, 3])

        assert as_lists(bins) == [[1], [3, 2, 0]]


class TestFitsBudget:
    def test_rows_within_the_budget_fit(self) -> None:
        assert fits_budget([np.array([0, 1])], np.array([5, 5]), 10)

    def test_rows_over_the_budget_do_not_fit(self) -> None:
        assert not fits_budget([np.array([0, 1])], np.array([5, 5]), 9)

    def test_one_row_over_the_budget_fits_alone(self) -> None:
        assert fits_budget([np.array([0])], np.array([15]), 10)

    def test_an_empty_micro_batch_does_not_fit(self) -> None:
        assert not fits_budget([np.zeros(0, dtype=np.intp)], np.array([1]), 10)


class TestPackRowsEvenly:
    def test_rows_deal_over_the_fewest_micro_batches_of_the_budget(self) -> None:
        row_tokens = np.array([5, 8, 4, 3])

        micro_batches = pack_rows_evenly(np.array([3, 1, 0, 2]), row_tokens, 9)

        assert as_lists(micro_batches) == [[1], [0], [2, 3]]

    def test_a_deal_over_the_budget_takes_one_more_micro_batch(self) -> None:
        # Arrange: two micro-batches of 6 fit, but dealing gives one 7 tokens.
        row_tokens = np.array([3, 3, 2, 2, 2])

        # Act
        micro_batches = pack_rows_evenly(np.arange(5), row_tokens, 6)

        # Assert
        assert as_lists(micro_batches) == [[0, 4], [1], [2, 3]]

    def test_a_row_longer_than_the_budget_runs_alone(self) -> None:
        row_tokens = np.array([2, 15, 3])

        micro_batches = pack_rows_evenly(np.array([0, 1, 2]), row_tokens, 10)

        assert as_lists(micro_batches) == [[1], [2, 0]]

    def test_no_rows_make_no_micro_batches(self) -> None:
        micro_batches = pack_rows_evenly(np.zeros(0, dtype=np.intp), np.array([3]), 10)

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


class TestFillMicroBatches:
    def test_rows_deal_evenly_over_the_count_most_tokens_first(self) -> None:
        row_tokens = np.array([2, 2, 2, 1])

        filled = fill_micro_batches(
            [np.array([0, 1, 2])], np.array([3]), row_tokens, 2, 10
        )

        assert as_lists(filled) == [[0, 2], [1]]

    def test_the_shortest_spare_rows_join_to_reach_the_count(self) -> None:
        # Arrange
        row_tokens = np.array([4, 9, 3, 1, 5])

        # Act
        filled = fill_micro_batches(
            [np.array([0])], np.array([1, 2, 3, 4]), row_tokens, 3, 10
        )

        # Assert
        assert as_lists(filled) == [[0], [2], [3]]

    def test_no_micro_batches_fill_from_spare_rows_only(self) -> None:
        filled = fill_micro_batches([], np.array([0, 1]), np.array([5, 2]), 1, 10)

        assert as_lists(filled) == [[1]]

    def test_a_deal_over_the_budget_splits_the_micro_batches_instead(self) -> None:
        # Arrange: dealing five rows over two micro-batches gives one 7 tokens.
        row_tokens = np.array([3, 3, 2, 2, 2])
        micro_batches = [np.array([0, 1]), np.array([2, 3, 4])]

        # Act
        filled = fill_micro_batches(
            micro_batches, np.zeros(0, dtype=np.intp), row_tokens, 2, 6
        )

        # Assert
        assert as_lists(filled) == [[0, 1], [2, 3, 4]]

    def test_too_few_rows_with_the_spare_rows_is_refused(self) -> None:
        with pytest.raises(ValueError, match="Cannot split 2 rows into 3"):
            fill_micro_batches([np.array([0])], np.array([1]), np.array([1, 1]), 3, 10)


class TestBalanceWindows:
    def test_trained_rows_deal_by_tokens_and_spare_rows_fill_the_rest(self) -> None:
        # Arrange
        row_tokens = np.array([5, 4, 9, 3])
        trained = np.array([True, True, False, True])

        # Act
        windows = balance_windows(
            [np.array([0, 1]), np.array([2, 3])], trained, row_tokens
        )

        # Assert
        assert as_lists(windows) == [[0, 2], [1, 3]]

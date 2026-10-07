# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``agilerl.utils.vision_rows``."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from agilerl.utils.vision_rows import (
    append_vision_filler_rows,
    pixel_values_for_minibatch,
    select_vision_rows,
    vision_filler_row,
)

PAD = 9
IMG = 8


def vision_batch() -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Two rows whose image placeholders take two ids per vision row.

    Row 0 has 2 vision rows; its first vision row's ids end at position 3.
    Row 1 has 1 vision row whose ids end at position 5.
    """
    token_ids = torch.tensor(
        [[1, IMG, IMG, 2, IMG, IMG, 3], [4, 5, 6, IMG, IMG, PAD, PAD]]
    )
    return token_ids, torch.arange(6.0).reshape(3, 2), [2, 1]


class TestPixelValuesForMinibatch:
    def test_indexes_when_leading_dim_is_the_sample_batch(self) -> None:
        pixel_values = torch.arange(8).reshape(4, 2)

        selected = pixel_values_for_minibatch(
            pixel_values,
            np.array([1, 3]),
            sample_rows=4,
        )

        assert torch.equal(selected, pixel_values[[1, 3]])

    def test_keeps_every_image_for_a_single_sample(self) -> None:
        pixel_values = torch.ones(4, 3, 2, 2)

        selected = pixel_values_for_minibatch(
            pixel_values,
            np.array([0]),
            sample_rows=1,
        )

        assert torch.equal(selected, pixel_values)

    def test_keeps_each_samples_image_block(self) -> None:
        pixel_values = torch.arange(8).reshape(8, 1, 1, 1)

        selected = pixel_values_for_minibatch(
            pixel_values,
            np.array([1, 0]),
            sample_rows=2,
        )

        assert torch.equal(selected, torch.cat([pixel_values[4:], pixel_values[:4]]))

    def test_rejects_a_leading_dim_that_does_not_divide_the_rows(self) -> None:
        pixel_values = torch.ones(5, 3, 2, 2)

        with pytest.raises(ValueError, match="leading dim 5"):
            pixel_values_for_minibatch(
                pixel_values,
                np.array([0]),
                sample_rows=2,
            )

    def test_slices_unequal_image_counts(self) -> None:
        pixel_values = torch.arange(7).reshape(7, 1, 1, 1)

        selected = pixel_values_for_minibatch(
            pixel_values,
            np.array([1]),
            sample_rows=2,
            image_counts=[3, 4],
        )

        assert torch.equal(selected, pixel_values[3:])


class TestSelectVisionRows:
    def test_selects_the_vision_rows_and_counts_of_the_kept_rows(self) -> None:
        # Arrange
        pixel_values = torch.arange(7).reshape(7, 1)

        # Act
        selected, counts = select_vision_rows(
            pixel_values, [3, 4], np.array([1]), sample_rows=2
        )

        # Assert
        assert selected is not None
        assert torch.equal(selected, pixel_values[3:])
        assert counts == [4]

    def test_text_batch_has_no_vision_rows(self) -> None:
        assert select_vision_rows(None, None, np.array([0, 1]), sample_rows=2) == (
            None,
            None,
        )


class TestVisionFillerRow:
    def test_picks_the_shortest_prefix_holding_one_vision_row(self) -> None:
        # Arrange: row 0 is longer but its first vision row ends sooner.
        token_ids, pixel_values, counts = vision_batch()

        # Act
        filler_ids, filler_pixels = vision_filler_row(
            token_ids, pixel_values, counts, IMG
        )

        # Assert
        assert torch.equal(filler_ids, torch.tensor([1, IMG, IMG]))
        assert torch.equal(filler_pixels, pixel_values[0:1])

    def test_skips_rows_without_vision_rows(self) -> None:
        # Arrange
        token_ids = torch.tensor([[1, 2, PAD], [3, IMG, 4]])
        pixel_values = torch.tensor([[5.0, 6.0]])

        # Act
        filler_ids, filler_pixels = vision_filler_row(
            token_ids, pixel_values, [0, 1], IMG
        )

        # Assert
        assert torch.equal(filler_ids, torch.tensor([3, IMG]))
        assert torch.equal(filler_pixels, pixel_values)

    def test_placeholders_must_split_evenly_over_vision_rows(self) -> None:
        # Arrange: row 0 has 5 placeholder ids for 2 vision rows.
        token_ids, pixel_values, counts = vision_batch()
        token_ids[0, 3] = IMG

        # Act / Assert
        with pytest.raises(
            ValueError,
            match=r"Row 0 has 5 image placeholder ids \(id 8\) for 2 vision rows",
        ):
            vision_filler_row(token_ids, pixel_values, counts, IMG)

    def test_needs_a_row_with_vision_rows(self) -> None:
        with pytest.raises(ValueError, match="at least one row with vision rows"):
            vision_filler_row(torch.tensor([[1, 2]]), torch.zeros(0, 2), [0], IMG)


class TestAppendVisionFillerRows:
    def test_appends_one_vision_row_per_filler_row(self) -> None:
        # Arrange
        token_ids, pixel_values, counts = vision_batch()

        # Act
        filler_ids, padded_pixels, padded_counts = append_vision_filler_rows(
            token_ids, pixel_values, counts, IMG, extra=2
        )

        # Assert
        assert torch.equal(filler_ids, torch.tensor([1, IMG, IMG]))
        assert torch.equal(
            padded_pixels,
            torch.cat([pixel_values, pixel_values[0:1], pixel_values[0:1]]),
        )
        assert padded_counts == [2, 1, 1, 1]

    def test_no_extra_rows_keep_the_vision_rows(self) -> None:
        token_ids, pixel_values, counts = vision_batch()

        _, padded_pixels, padded_counts = append_vision_filler_rows(
            token_ids, pixel_values, counts, IMG, extra=0
        )

        assert torch.equal(padded_pixels, pixel_values)
        assert padded_counts == [2, 1]

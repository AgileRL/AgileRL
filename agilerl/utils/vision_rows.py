# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Vision-row bookkeeping for vision-language LLM training batches.

``pixel_values`` stacks the vision rows of every token row in row order, and
``pixel_image_counts[i]`` is the number of vision rows of token row ``i``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch


@dataclass(frozen=True)
class VisionRows:
    """Vision rows of a batch's token rows, stacked in row order.

    :param pixel_values: Vision tensor for the local batch.
    :param image_counts: Vision rows per token row, or ``None`` when every row
        has the same number.
    """

    pixel_values: torch.Tensor
    image_counts: Sequence[int] | None = None

    def for_minibatch(
        self, minibatch_idxs: npt.NDArray[np.intp], sample_rows: int
    ) -> torch.Tensor:
        """Vision rows of the token rows ``minibatch_idxs``.

        :param minibatch_idxs: Sample rows included in the minibatch.
        :param sample_rows: Local batch size (``token_ids.shape[0]``).
        :return: Vision tensor passed to the model for this minibatch.
        """
        return pixel_values_for_minibatch(
            self.pixel_values,
            minibatch_idxs,
            sample_rows=sample_rows,
            image_counts=self.image_counts,
        )


def pixel_values_for_minibatch(
    pixel_values: torch.Tensor,
    minibatch_idxs: npt.NDArray[np.intp],
    sample_rows: int,
    image_counts: Sequence[int] | None = None,
) -> torch.Tensor:
    """Vision rows for one minibatch.

    Dim 0 is the sample axis when it equals ``sample_rows``. Several images
    for one sample sit in one contiguous block when the leading dimension
    divides ``sample_rows``. A single sample keeps its whole tensor.

    :param pixel_values: Vision tensor for the local batch.
    :param minibatch_idxs: Sample rows included in the minibatch.
    :param sample_rows: Local batch size (``token_ids.shape[0]``).
    :param image_counts: Vision rows per sample row, when they differ.
    :return: Vision tensor passed to the model for this minibatch.
    :raises ValueError: If the leading dimension does not match and does not
        divide the sample rows.
    """
    leading = int(pixel_values.shape[0])
    rows = int(sample_rows)
    if image_counts is not None:
        offset = 0
        spans: list[tuple[int, int]] = []
        for count in image_counts:
            spans.append((offset, offset + int(count)))
            offset += int(count)
        if offset != leading:
            msg = (
                f"pixel_values leading dim {leading} does not match "
                f"image counts summing to {offset}"
            )
            raise ValueError(msg)
        blocks = [
            pixel_values[spans[int(idx)][0] : spans[int(idx)][1]]
            for idx in minibatch_idxs
        ]
        return torch.cat(blocks, dim=0)
    if leading == rows:
        return pixel_values[minibatch_idxs]
    if rows == 1:
        return pixel_values
    if rows > 0 and leading % rows == 0:
        images_per_row = leading // rows
        blocks = [
            pixel_values[int(idx) * images_per_row : (int(idx) + 1) * images_per_row]
            for idx in minibatch_idxs
        ]
        return torch.cat(blocks, dim=0)
    msg = f"pixel_values leading dim {leading} does not match the {rows} sample rows"
    raise ValueError(msg)


def vision_filler_row(
    token_ids: torch.Tensor,
    pixel_values: torch.Tensor,
    pixel_image_counts: Sequence[int],
    image_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shortest row prefix that ends after one vision row's image placeholders.

    :param token_ids: ``(R, W)`` right-padded token ids.
    :param pixel_values: Vision rows in row order.
    :param pixel_image_counts: Vision rows per row.
    :param image_token_id: Token id the VL forward scatters one image feature
        row into.
    :return: The prefix's token ids and its one vision row.
    :raises ValueError: If a row's placeholder count does not split evenly over
        its vision rows, or no row has vision rows.
    """
    best: tuple[int, int, int] | None = None
    offset = 0
    for row, count in enumerate(pixel_image_counts):
        if count > 0:
            positions = torch.nonzero(token_ids[row] == image_token_id).flatten()
            if positions.numel() == 0 or positions.numel() % count != 0:
                msg = (
                    f"Row {row} has {positions.numel()} image placeholder ids "
                    f"(id {image_token_id}) for {count} vision rows."
                )
                raise ValueError(msg)
            end = int(positions[positions.numel() // count - 1]) + 1
            if best is None or end < best[0]:
                best = (end, row, offset)
        offset += int(count)
    if best is None:
        msg = "A vision batch needs at least one row with vision rows."
        raise ValueError(msg)
    end, row, offset = best
    return token_ids[row, :end], pixel_values[offset : offset + 1]


def append_vision_filler_rows(
    token_ids: torch.Tensor,
    pixel_values: torch.Tensor,
    pixel_image_counts: Sequence[int],
    image_token_id: int,
    extra: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Filler row token ids, plus the vision rows of ``extra`` filler rows appended.

    Each filler row is :func:`vision_filler_row` with its one vision row, so a
    forward over filler rows still runs the vision tower.

    :param token_ids: ``(R, W)`` right-padded token ids of the real rows.
    :param pixel_values: Vision rows of the real rows in row order.
    :param pixel_image_counts: Vision rows per real row.
    :param image_token_id: Token id the VL forward scatters one image feature
        row into.
    :param extra: Filler rows to append.
    :return: The filler row's token ids, the vision tensor with one vision row
        per filler row appended, and the matching image counts.
    """
    filler_ids, filler_pixels = vision_filler_row(
        token_ids, pixel_values, pixel_image_counts, image_token_id
    )
    padded_pixels = torch.cat(
        [pixel_values, filler_pixels.expand(extra, *filler_pixels.shape[1:])]
    )
    return filler_ids, padded_pixels, [*pixel_image_counts, *([1] * extra)]

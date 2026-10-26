# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Vision-row bookkeeping for vision-language LLM training batches.

``pixel_values`` stacks the vision rows of every token row in row order, and
``pixel_image_counts[i]`` is the number of vision rows of token row ``i``.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import numpy.typing as npt
import torch


@dataclass(frozen=True)
class VisionRows:
    """Vision rows of a batch's token rows, stacked in row order.

    :param pixel_values: Vision tensor for the local batch.
    :param image_counts: Vision rows per token row, or ``None`` when every row
        has the same number.
    :param image_keys: ``(N,)`` vision feature cache key of each vision row,
        or ``None`` when the tower runs uncached.
    """

    pixel_values: torch.Tensor
    image_counts: Sequence[int] | None = None
    image_keys: torch.Tensor | None = None

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
    divides ``sample_rows``. A single sample keeps its whole tensor. Rows
    whose vision rows are one contiguous block get a view of
    ``pixel_values``; any other selection is a copy.

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
        offsets = np.cumsum([0, *image_counts]).tolist()
        if offsets[-1] != leading:
            msg = (
                f"pixel_values leading dim {leading} does not match "
                f"image counts summing to {offsets[-1]}"
            )
            raise ValueError(msg)
        spans = [(offsets[int(idx)], offsets[int(idx) + 1]) for idx in minibatch_idxs]
    elif leading == rows:
        spans = [(int(idx), int(idx) + 1) for idx in minibatch_idxs]
    elif rows == 1:
        return pixel_values
    elif rows > 0 and leading % rows == 0:
        images_per_row = leading // rows
        spans = [
            (int(idx) * images_per_row, (int(idx) + 1) * images_per_row)
            for idx in minibatch_idxs
        ]
    else:
        msg = (
            f"pixel_values leading dim {leading} does not match the {rows} sample rows"
        )
        raise ValueError(msg)
    if not spans:
        return pixel_values[:0]
    if all(end == start for (_, end), (start, _) in pairwise(spans)):
        return pixel_values[spans[0][0] : spans[-1][1]]
    return torch.cat([pixel_values[start:end] for start, end in spans])


class PinnedStaging:
    """Two page-locked host buffers that host chunks pass through on their way to a CUDA device.

    Each buffer grows to the largest chunk it has held and is kept across
    calls, so steady-state copies allocate no host memory. Chunks take the
    buffers in turn, and a buffer is refilled only after the device copy out
    of it is done.
    """

    def __init__(self) -> None:
        self.buffers = [torch.empty(0, dtype=torch.uint8) for _ in range(2)]
        self.copies: list[torch.cuda.Event | None] = [None, None]
        self.slot = 0

    def copy_to(
        self, chunk: torch.Tensor, device: torch.device, stream: torch.cuda.Stream
    ) -> tuple[torch.Tensor, torch.cuda.Event]:
        """Start an asynchronous copy of ``chunk`` to ``device`` on ``stream`` through the next buffer.

        :param chunk: Host tensor.
        :param device: CUDA device of the copy.
        :param stream: Stream the copy runs on.
        :return: The device copy, and an event recorded when it is done.
        """
        slot = self.slot
        self.slot = 1 - slot
        previous = self.copies[slot]
        if previous is not None:
            previous.synchronize()
        num_bytes = chunk.numel() * chunk.element_size()
        if self.buffers[slot].numel() < num_bytes:
            self.buffers[slot] = torch.empty(
                num_bytes, dtype=torch.uint8, pin_memory=True
            )
        staged = self.buffers[slot][:num_bytes].view(chunk.dtype).view(chunk.shape)
        staged.copy_(chunk)
        with torch.cuda.stream(stream):
            copy = staged.to(device, non_blocking=True)
        done = torch.cuda.Event()
        done.record(stream)
        self.copies[slot] = done
        return copy, done


def prefetch_to_device(
    chunks: Iterable[torch.Tensor],
    device: str | torch.device,
    staging: PinnedStaging,
    ahead: int = 1,
) -> Iterator[torch.Tensor]:
    """Copies of ``chunks`` on ``device`` in order, each copy started ``ahead`` chunks before it is returned.

    A host chunk bound for CUDA goes through ``staging`` and is copied on a
    side stream, so its copy overlaps the device work already queued. The
    iterator holds the ``ahead`` copies it has started but not returned.

    :param chunks: Tensors in the order the caller reads them; read lazily.
    :param device: Device the copies live on.
    :param staging: Pinned buffers of a CUDA copy; unused for any other device.
    :param ahead: Chunks copied before the caller asks for them, 0 or 1.
    :return: Iterator over the device copies.
    """
    device = torch.device(device)
    stream = torch.cuda.Stream(device) if device.type == "cuda" else None
    pending: deque[tuple[torch.Tensor, torch.cuda.Event | None]] = deque()
    for chunk in chunks:
        if stream is None or chunk.is_cuda:
            pending.append((chunk.to(device), None))
        else:
            pending.append(staging.copy_to(chunk, device, stream))
        # ``to`` returns ``chunk`` itself on its own device; hold no returned copy.
        del chunk
        if len(pending) > ahead:
            yield _finish_copy(*pending.popleft())
    while pending:
        yield _finish_copy(*pending.popleft())


def _finish_copy(copy: torch.Tensor, ready: torch.cuda.Event | None) -> torch.Tensor:
    """``copy``, with the current stream waiting on its side-stream copy."""
    if ready is not None:
        current = torch.cuda.current_stream(copy.device)
        current.wait_event(ready)
        # The side stream allocated ``copy``; keep it until the current stream is done.
        copy.record_stream(current)
    return copy


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

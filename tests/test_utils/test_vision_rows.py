# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``agilerl.utils.vision_rows``."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from agilerl.utils.vision_rows import (
    PinnedStaging,
    VisionRows,
    append_vision_filler_rows,
    pixel_values_for_minibatch,
    prefetch_to_device,
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

    def test_consecutive_rows_share_the_batch_storage(self) -> None:
        pixel_values = torch.arange(6.0).reshape(6, 1)

        selected = pixel_values_for_minibatch(
            pixel_values, np.array([1, 2]), sample_rows=4, image_counts=[1, 2, 1, 2]
        )

        assert selected.flatten().tolist() == [1.0, 2.0, 3.0]
        assert selected.untyped_storage().data_ptr() == (
            pixel_values.untyped_storage().data_ptr()
        )

    def test_scattered_rows_are_copied_in_minibatch_order(self) -> None:
        pixel_values = torch.arange(6.0).reshape(6, 1)

        selected = pixel_values_for_minibatch(
            pixel_values, np.array([3, 0]), sample_rows=4, image_counts=[1, 2, 1, 2]
        )

        assert selected.flatten().tolist() == [4.0, 5.0, 0.0]
        assert selected.untyped_storage().data_ptr() != (
            pixel_values.untyped_storage().data_ptr()
        )

    def test_an_empty_minibatch_has_no_vision_rows(self) -> None:
        pixel_values = torch.arange(6.0).reshape(3, 2)

        selected = pixel_values_for_minibatch(
            pixel_values, np.array([], dtype=np.intp), sample_rows=3
        )

        assert selected.shape == (0, 2)


class TestVisionRowsForMinibatch:
    def test_selects_the_vision_rows_of_the_kept_rows(self) -> None:
        # Arrange
        pixel_values = torch.arange(7).reshape(7, 1)
        vision_rows = VisionRows(pixel_values, image_counts=[3, 4])

        # Act
        selected = vision_rows.for_minibatch(np.array([1, 0]), sample_rows=2)

        # Assert
        assert torch.equal(selected, torch.cat([pixel_values[3:], pixel_values[:3]]))

    def test_indexes_the_sample_axis_without_image_counts(self) -> None:
        pixel_values = torch.arange(8).reshape(4, 2)

        selected = VisionRows(pixel_values).for_minibatch(
            np.array([2, 0]), sample_rows=4
        )

        assert torch.equal(selected, pixel_values[[2, 0]])

    def test_rejects_image_counts_that_do_not_cover_the_vision_rows(self) -> None:
        vision_rows = VisionRows(torch.zeros(5, 1), image_counts=[3, 1])

        with pytest.raises(ValueError, match="image counts summing to 4"):
            vision_rows.for_minibatch(np.array([0]), sample_rows=2)


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


class FakeEvent:
    """CPU stand-in for ``torch.cuda.Event``."""

    def __init__(self) -> None:
        self.recorded_on: FakeStream | None = None
        self.synchronized = False

    def record(self, stream: FakeStream) -> None:
        self.recorded_on = stream

    def synchronize(self) -> None:
        self.synchronized = True


class FakeStream:
    """CPU stand-in for ``torch.cuda.Stream``."""

    def __init__(self, device: torch.device | None = None) -> None:
        self.device = device
        self.waited: list[FakeEvent] = []

    def wait_event(self, event: FakeEvent) -> None:
        self.waited.append(event)


@pytest.fixture
def fake_cuda(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """CUDA streams, events, pinning and host-to-device copies, run on the CPU."""
    current = FakeStream()
    events: list[FakeEvent] = []
    recorded: list[tuple[torch.Tensor, FakeStream]] = []
    real_empty = torch.empty
    real_to = torch.Tensor.to

    def make_event() -> FakeEvent:
        events.append(FakeEvent())
        return events[-1]

    def empty(
        *args: object, pin_memory: bool = False, **kwargs: object
    ) -> torch.Tensor:
        return real_empty(*args, **kwargs)

    def to(self: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        if args and isinstance(args[0], torch.device) and args[0].type == "cuda":
            return self.clone()
        return real_to(self, *args, **kwargs)

    def record_stream(self: torch.Tensor, stream: FakeStream) -> None:
        recorded.append((self, stream))

    monkeypatch.setattr(torch.cuda, "Stream", FakeStream)
    monkeypatch.setattr(torch.cuda, "stream", lambda _stream: nullcontext())
    monkeypatch.setattr(torch.cuda, "Event", make_event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device=None: current)
    monkeypatch.setattr(torch, "empty", empty)
    monkeypatch.setattr(torch.Tensor, "to", to)
    monkeypatch.setattr(torch.Tensor, "record_stream", record_stream)
    return SimpleNamespace(current=current, events=events, recorded=recorded)


class TestPrefetchToDevice:
    def test_returns_each_chunk_on_the_device_in_order(self) -> None:
        # Arrange
        chunks = [torch.ones(1, 2), torch.ones(3, 2), torch.ones(2, 2)]

        # Act
        copies = list(prefetch_to_device(iter(chunks), "meta", PinnedStaging()))

        # Assert
        assert [copy.device.type for copy in copies] == ["meta"] * 3
        assert [copy.shape for copy in copies] == [chunk.shape for chunk in chunks]

    def test_reads_one_chunk_ahead_of_the_caller(self) -> None:
        # Arrange
        reads: list[int] = []

        def chunks() -> Iterator[torch.Tensor]:
            for index in range(4):
                reads.append(index)
                yield torch.full((1, 2), float(index))

        copies = prefetch_to_device(chunks(), "cpu", PinnedStaging())

        # Act
        first = next(copies)
        reads_after_first = list(reads)
        second = next(copies)

        # Assert
        assert first.tolist() == [[0.0, 0.0]]
        assert second.tolist() == [[1.0, 1.0]]
        assert reads_after_first == [0, 1]
        assert reads == [0, 1, 2]

    def test_ahead_zero_reads_each_chunk_when_asked(self) -> None:
        # Arrange
        reads: list[int] = []

        def chunks() -> Iterator[torch.Tensor]:
            for index in range(3):
                reads.append(index)
                yield torch.full((1, 2), float(index))

        copies = prefetch_to_device(chunks(), "cpu", PinnedStaging(), ahead=0)

        # Act
        first = next(copies)
        reads_after_first = list(reads)
        second = next(copies)

        # Assert
        assert first.tolist() == [[0.0, 0.0]]
        assert second.tolist() == [[1.0, 1.0]]
        assert reads_after_first == [0]
        assert reads == [0, 1]

    def test_empty_chunks_stay_empty(self) -> None:
        # Arrange
        chunks = [torch.ones(0, 2), torch.ones(1, 2)]

        # Act
        copies = list(prefetch_to_device(chunks, "meta", PinnedStaging()))

        # Assert
        assert [tuple(copy.shape) for copy in copies] == [(0, 2), (1, 2)]

    def test_cuda_copies_wait_on_their_side_stream_copy(
        self, fake_cuda: SimpleNamespace
    ) -> None:
        # Arrange
        torch.manual_seed(0)
        chunks = [torch.randn(4, 8), torch.randn(2, 8)]

        # Act
        copies = list(prefetch_to_device(chunks, "cuda", PinnedStaging()))

        # Assert
        for copy, chunk in zip(copies, chunks, strict=True):
            assert torch.equal(copy, chunk)
        assert fake_cuda.current.waited == fake_cuda.events
        assert all(
            event.recorded_on is not fake_cuda.current for event in fake_cuda.events
        )
        assert [tensor for tensor, _ in fake_cuda.recorded] == copies
        assert all(stream is fake_cuda.current for _, stream in fake_cuda.recorded)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_cuda_copies_match_the_host_chunks(self) -> None:
        # Arrange
        torch.manual_seed(0)
        chunks = [torch.randn(4, 8), torch.randn(0, 8), torch.randn(2, 8)]

        # Act
        copies = list(prefetch_to_device(chunks, "cuda", PinnedStaging()))

        # Assert
        assert all(copy.is_cuda for copy in copies)
        for copy, chunk in zip(copies, chunks, strict=True):
            assert torch.equal(copy.cpu(), chunk)


class TestPinnedStagingCopyTo:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_later_passes_reuse_the_pinned_buffers(self) -> None:
        # Arrange: an odd chunk count starts the second pass on the other
        # buffer, so two passes leave both at the largest chunk.
        torch.manual_seed(0)
        staging = PinnedStaging()
        chunks = [torch.randn(4, 8), torch.randn(3, 8), torch.randn(4, 8)]
        for _ in range(2):
            list(prefetch_to_device(chunks, "cuda", staging))
        warm = [buffer.data_ptr() for buffer in staging.buffers]

        # Act
        copies = list(prefetch_to_device(chunks, "cuda", staging))

        # Assert
        assert [buffer.data_ptr() for buffer in staging.buffers] == warm
        assert all(buffer.is_pinned() for buffer in staging.buffers)
        for copy, chunk in zip(copies, chunks, strict=True):
            assert torch.equal(copy.cpu(), chunk)

    def test_buffers_grow_to_the_largest_chunk(
        self, fake_cuda: SimpleNamespace
    ) -> None:
        # Arrange
        staging = PinnedStaging()
        chunks = [torch.ones(1, 8), torch.ones(2, 8), torch.ones(6, 8)]

        # Act
        list(prefetch_to_device(chunks, "cuda", staging))

        # Assert: chunks 0 and 2 share a buffer, chunk 1 has the other.
        assert [buffer.numel() for buffer in staging.buffers] == [6 * 8 * 4, 2 * 8 * 4]

    def test_a_buffer_is_refilled_after_its_last_copy_is_done(
        self, fake_cuda: SimpleNamespace
    ) -> None:
        # Arrange
        staging = PinnedStaging()
        stream = FakeStream()
        device = torch.device("cuda")
        _, first_done = staging.copy_to(torch.ones(1, 2), device, stream)
        _, second_done = staging.copy_to(torch.ones(1, 2), device, stream)
        first_done_before_refill = first_done.synchronized

        # Act
        copy, _ = staging.copy_to(torch.full((1, 2), 3.0), device, stream)

        # Assert
        assert not first_done_before_refill
        assert first_done.synchronized
        assert not second_done.synchronized
        assert copy.tolist() == [[3.0, 3.0]]

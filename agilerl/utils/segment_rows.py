# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Training rows of LLM batches whose segmented episodes train one segment per row.

An episode whose context restarted is stored as consecutive self-contained
segments (:class:`~agilerl.components.llm_rollout_data.EpisodeSegments`). Each
segment trains as its own row; unsegmented episodes stay one row.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import numpy.typing as npt
import torch

from agilerl.components.llm_rollout_data import EpisodeSegments
from agilerl.utils.algo_utils import stack_and_pad_experiences
from agilerl.utils.llm_utils import attention_mask_from_padded_ids
from agilerl.utils.vision_rows import append_vision_filler_rows, vision_filler_row

TEXT_FILLER_TOKENS = 16
"""Most tokens a text filler row keeps from the shortest segment row."""


@dataclass(frozen=True)
class SegmentRows:
    """Training rows of a batch whose segmented episodes train one segment per row.

    :param token_ids: ``(R, W)`` right-padded token ids.
    :param action_masks: ``(R, W - 1)`` action-token mask.
    :param row_episodes: ``(R,)`` source episode of each row, ``-1`` on filler rows.
    :param row_starts: ``(R,)`` first episode token of each row, ``0`` on filler rows.
    :param row_ends: ``(R,)`` episode token after each row's last, ``0`` on filler rows.
    :param turn_ids: ``(R, W - 1)`` turn index per token, or ``None``.
    :param sampling_logps: Per-row flat sampling log-probs, or ``None``.
    :param pixel_values: Vision rows in row order, or ``None``.
    :param pixel_image_counts: Vision rows per row, or ``None``.
    """

    token_ids: torch.Tensor
    action_masks: torch.Tensor
    row_episodes: npt.NDArray[np.intp]
    row_starts: npt.NDArray[np.intp]
    row_ends: npt.NDArray[np.intp]
    turn_ids: torch.Tensor | None = None
    sampling_logps: list[torch.Tensor | None] | None = None
    pixel_values: torch.Tensor | None = None
    pixel_image_counts: list[int] | None = None

    def training_rows(self, episode_idxs: npt.NDArray) -> npt.NDArray[np.intp]:
        """Rows of the episodes in ``episode_idxs`` plus every filler row.

        :param episode_idxs: Episodes that survive the advantage filter.
        :return: Sorted row indices.
        """
        return np.flatnonzero(
            np.isin(self.row_episodes, episode_idxs) | (self.row_episodes < 0)
        )

    def split_frame(self, frame: torch.Tensor, pad_value: float) -> torch.Tensor:
        """Each row's slice of a ``(B, T - 1)`` action-aligned episode frame.

        :param frame: ``(B, T - 1)`` per-position episode values.
        :param pad_value: Value past a row's end and on filler rows.
        :return: ``(R, W - 1)`` per-position row values.
        """
        rows = frame.new_full(
            (len(self.row_episodes), int(self.token_ids.shape[1]) - 1), pad_value
        )
        for row, (episode, start, end) in enumerate(
            zip(self.row_episodes, self.row_starts, self.row_ends, strict=True)
        ):
            if episode >= 0:
                rows[row, : end - start - 1] = frame[episode, start : end - 1]
        return rows

    def merge_frame(
        self,
        rows: torch.Tensor,
        num_episodes: int,
        frame_len: int,
        fill_value: float,
    ) -> torch.Tensor:
        """Place each row's action-aligned values back in its episode's frame.

        The position predicting a segment's first token belongs to no row and
        keeps ``fill_value``.

        :param rows: ``(R, W - 1)`` per-position row values.
        :param num_episodes: Episodes in the batch.
        :param frame_len: Positions of the episode frame (``T - 1``).
        :param fill_value: Value at positions no row covers.
        :return: ``(B, T - 1)`` per-position episode values.
        """
        frame = rows.new_full((num_episodes, frame_len), fill_value)
        for row, (episode, start, end) in enumerate(
            zip(self.row_episodes, self.row_starts, self.row_ends, strict=True)
        ):
            if episode >= 0:
                frame[episode, start : end - 1] = rows[row, : end - start - 1]
        return frame

    def split_advantages(self, advantages: torch.Tensor) -> torch.Tensor:
        """Each row's advantages, zero on filler rows.

        :param advantages: ``(B, 1)`` episode or ``(B, T - 1)`` per-token advantages.
        :return: ``(R, 1)`` episode advantage per row, or ``(R, W - 1)``
            per-token advantages.
        """
        if is_per_token(advantages):
            return self.split_frame(advantages, 0.0)
        real = np.flatnonzero(self.row_episodes >= 0)
        rows = advantages.new_zeros(len(self.row_episodes), *advantages.shape[1:])
        rows[torch.as_tensor(real, device=advantages.device)] = advantages[
            torch.as_tensor(self.row_episodes[real], device=advantages.device)
        ]
        return rows


def split_episode_segments(
    token_ids: torch.Tensor,
    action_masks: torch.Tensor,
    episode_segments: Sequence[EpisodeSegments | None],
    pad_token_id: int,
    turn_ids: torch.Tensor | None = None,
    sampling_logps: Sequence[torch.Tensor | None] | None = None,
    pixel_values: torch.Tensor | None = None,
    pixel_image_counts: Sequence[int] | None = None,
) -> SegmentRows:
    """Split each segmented episode into one row per segment; other episodes stay one row.

    A segment row keeps the action-aligned positions of all its tokens but the
    last, whose position predicts the next segment's first token. Rows are
    right-padded to the widest row.

    :param token_ids: ``(B, T)`` right-padded episode token ids.
    :param action_masks: ``(B, T - 1)`` action-token mask.
    :param episode_segments: Segment layout per episode, ``None`` when unsegmented.
    :param pad_token_id: Token id that pads rows to a common width.
    :param turn_ids: ``(B, T - 1)`` turn index per token, or ``None``.
    :param sampling_logps: Per-episode flat sampling log-probs in action-mask
        order, or ``None``.
    :param pixel_values: Vision rows of every episode in batch order, or ``None``.
    :param pixel_image_counts: Vision rows per episode; required with ``pixel_values``.
    :return: The batch's training rows.
    :raises ValueError: If vision rows cannot be assigned to segment rows.
    """
    seq_len = int(token_ids.shape[1])
    spans: list[tuple[int, int, int]] = []
    for episode, segments in enumerate(episode_segments):
        lengths = [seq_len] if segments is None else segments.token_lengths.tolist()
        starts = np.cumsum([0, *lengths[:-1]]).tolist()
        spans.extend(
            (episode, start, start + length)
            for start, length in zip(starts, lengths, strict=True)
        )
    row_episodes, row_starts, row_ends = (
        np.array(column, dtype=np.intp) for column in zip(*spans, strict=True)
    )

    def frame_rows(frame: torch.Tensor, pad_value: float, trim: int) -> torch.Tensor:
        (rows,) = stack_and_pad_experiences(
            [
                frame[episode : episode + 1, start : end - trim]
                for episode, start, end in spans
            ],
            padding_values=[pad_value],
        )
        return rows

    row_action_masks = frame_rows(action_masks, False, 1)
    row_logps = (
        _split_sampling_logps(
            sampling_logps,
            row_action_masks.sum(dim=-1).tolist(),
            np.bincount(row_episodes, minlength=len(episode_segments)).tolist(),
        )
        if sampling_logps is not None
        else None
    )
    row_image_counts: list[int] | None = None
    if pixel_values is not None:
        if pixel_image_counts is None:
            msg = "pixel_image_counts is required to split vision episodes into segment rows."
            raise ValueError(msg)
        row_image_counts = _segment_image_counts(episode_segments, pixel_image_counts)
    return SegmentRows(
        token_ids=frame_rows(token_ids, pad_token_id, 0),
        action_masks=row_action_masks,
        row_episodes=row_episodes,
        row_starts=row_starts,
        row_ends=row_ends,
        turn_ids=frame_rows(turn_ids, -1, 1) if turn_ids is not None else None,
        sampling_logps=row_logps,
        pixel_values=pixel_values,
        pixel_image_counts=row_image_counts,
    )


def _split_sampling_logps(
    sampling_logps: Sequence[torch.Tensor | None],
    action_counts: list[int],
    rows_per_episode: list[int],
) -> list[torch.Tensor | None]:
    """Split each episode's flat sampling log-probs over its segment rows.

    An episode whose log-probs do not cover its rows' action tokens keeps them
    whole when it trains as one row, and gets ``None`` per row otherwise.

    :param sampling_logps: Per-episode flat sampling log-probs, or ``None`` entries.
    :param action_counts: Action tokens per row, rows grouped by episode in order.
    :param rows_per_episode: Rows each episode trains as.
    :return: Per-row sampling log-probs.
    """
    row_logps: list[torch.Tensor | None] = []
    first_row = 0
    for flat, num_rows in zip(sampling_logps, rows_per_episode, strict=True):
        sizes = action_counts[first_row : first_row + num_rows]
        first_row += num_rows
        if flat is not None and flat.numel() == sum(sizes):
            row_logps.extend(flat.split(sizes))
        elif num_rows == 1:
            row_logps.append(flat)
        else:
            row_logps.extend([None] * num_rows)
    return row_logps


def _segment_image_counts(
    episode_segments: Sequence[EpisodeSegments | None],
    pixel_image_counts: Sequence[int],
) -> list[int]:
    """Vision rows per training row once segmented episodes split into segment rows.

    :param episode_segments: Segment layout per episode, ``None`` when unsegmented.
    :param pixel_image_counts: Vision rows per episode.
    :return: Vision rows per training row.
    :raises ValueError: If a segmented episode has no per-segment vision rows.
    """
    row_image_counts: list[int] = []
    for episode, segments in enumerate(episode_segments):
        if segments is None:
            row_image_counts.append(int(pixel_image_counts[episode]))
        elif segments.pixel_rows is None:
            msg = (
                f"Episode {episode} has segments without pixel_rows in a vision batch."
            )
            raise ValueError(msg)
        else:
            row_image_counts.extend(segments.pixel_rows.tolist())
    return row_image_counts


def append_vision_tails(
    rows: SegmentRows, pad_token_id: int, image_token_id: int
) -> SegmentRows:
    """Append a vision filler row after the real tokens of each row with no vision rows.

    FSDP shards the vision tower's blocks, so every forward on every rank
    must run it, as filler rows do. A causal forward keeps the outputs of the
    real tokens before the tail exact, and the tail holds no action tokens.

    :param rows: Split rows, before filler rows are added.
    :param pad_token_id: Token id that pads rows to a common width.
    :param image_token_id: Token id the VL forward scatters one image feature
        row into.
    :return: The rows, each with at least one vision row.
    """
    counts = rows.pixel_image_counts
    if rows.pixel_values is None or counts is None or all(counts):
        return rows
    tail_ids, tail_pixels = vision_filler_row(
        rows.token_ids, rows.pixel_values, counts, image_token_id
    )
    tail_len = int(tail_ids.shape[0])
    lengths = (rows.row_ends - rows.row_starts).tolist()
    bare = [row for row, count in enumerate(counts) if count == 0]
    width = max(
        int(rows.token_ids.shape[1]), *(lengths[row] + tail_len for row in bare)
    )
    widen = (0, width - int(rows.token_ids.shape[1]))
    pad = torch.nn.functional.pad
    token_ids = pad(rows.token_ids, widen, value=pad_token_id)
    for row in bare:
        token_ids[row, lengths[row] : lengths[row] + tail_len] = tail_ids
    pixel_values = torch.cat(
        [
            block if count else tail_pixels
            for block, count in zip(
                rows.pixel_values.split(counts), counts, strict=True
            )
        ]
    )
    return replace(
        rows,
        token_ids=token_ids,
        action_masks=pad(rows.action_masks, widen, value=False),
        turn_ids=pad(rows.turn_ids, widen, value=-1)
        if rows.turn_ids is not None
        else None,
        pixel_values=pixel_values,
        pixel_image_counts=[max(count, 1) for count in counts],
    )


def is_per_token(advantages: torch.Tensor) -> bool:
    """Whether ``advantages`` holds one value per action position rather than per row.

    :param advantages: ``(R, 1)`` per-row or ``(R, W - 1)`` per-token advantages.
    :return: ``True`` for per-token advantages.
    """
    return advantages.dim() == 2 and advantages.shape[-1] > 1


def pad_row_values(
    values: torch.Tensor, rows: SegmentRows, pad_value: float = 0.0
) -> torch.Tensor:
    """Values of split rows, ``pad_value`` on what vision tails and filler rows added to ``rows``.

    :param values: ``(R, ...)`` per-row or ``(R, W - 1)`` per-token values of
        the split rows.
    :param rows: The rows after :func:`append_vision_tails` and
        :func:`pad_segment_rows`, which widen rows on the right and append rows.
    :param pad_value: Value of the added positions and rows.
    :return: ``(R', ...)`` or ``(R', W' - 1)`` values.
    """
    if is_per_token(values):
        widen = int(rows.action_masks.shape[1]) - int(values.shape[1])
        values = torch.nn.functional.pad(values, (0, widen), value=pad_value)
    extra = int(rows.token_ids.shape[0]) - int(values.shape[0])
    return torch.cat([values, values.new_full((extra, *values.shape[1:]), pad_value)])


def segment_window_layout(
    num_rows: int, optimizer_steps: int, micro_batch_size_per_gpu: int
) -> tuple[int, int]:
    """Padded row count and micro-batches per step that spread rows over ``optimizer_steps``.

    :param num_rows: Training rows on the rank with the most rows.
    :param optimizer_steps: Optimizer steps per epoch, at least ``1`` when
        ``num_rows`` is positive.
    :param micro_batch_size_per_gpu: Rows per backward pass.
    :return: Row count every rank pads its training rows to, and the
        micro-batches each optimizer step accumulates.
    """
    if num_rows == 0:
        return 0, 1
    step_rows = optimizer_steps * micro_batch_size_per_gpu
    accumulation_steps = -(-num_rows // step_rows)
    return accumulation_steps * step_rows, accumulation_steps


def filler_token_frac(
    token_ids: torch.Tensor,
    filler_rows: npt.NDArray[np.bool_],
    pad_token_id: int,
    packed: bool,
) -> float:
    """Filler rows' share of the tokens the gradient forward runs.

    :param token_ids: ``(R, W)`` right-padded rows the forward runs.
    :param filler_rows: ``(R,)`` mask of the filler rows.
    :param pad_token_id: Token id of the trailing padding.
    :param packed: Whether the forward runs only the real tokens of each row.
    :return: Filler tokens over all tokens, ``0.0`` with no rows.
    """
    if packed:
        tokens = attention_mask_from_padded_ids(token_ids, pad_token_id).sum(-1)
    else:
        tokens = torch.full((int(token_ids.shape[0]),), int(token_ids.shape[1]))
    total = int(tokens.sum())
    if total == 0:
        return 0.0
    filler = torch.as_tensor(filler_rows, device=tokens.device)
    return int(tokens[filler].sum()) / total


def filler_stand_in(
    action_masks: torch.Tensor, turn_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Action mask and turn ids that mark the first position of every row as turn ``0``.

    A loss averaged over the action tokens of filler rows alone divides by
    zero. Run the loss on this stand-in and zero it, so the micro-batch keeps
    its forward and backward collectives with no gradient.

    :param action_masks: ``(R, W - 1)`` action-token mask of filler rows.
    :param turn_ids: ``(R, W - 1)`` turn index per token of filler rows.
    :return: The stand-in action mask and turn ids.
    """
    first = torch.zeros_like(action_masks, dtype=torch.bool)
    first[:, 0] = True
    return first.to(action_masks.dtype), torch.where(first, 0, turn_ids)


def pad_segment_rows(
    rows: SegmentRows,
    num_rows: int,
    width: int,
    pad_token_id: int,
    image_token_id: int | None = None,
) -> SegmentRows:
    """Pad to ``num_rows`` rows of ``width`` tokens so every rank runs the same micro-batches.

    Filler rows have no action tokens and get zero advantage, so they add no
    loss; they exist to run the forward's collectives. A text filler row is the
    first ``TEXT_FILLER_TOKENS`` tokens of the shortest row. A vision filler
    row is the shortest row prefix that ends after one vision row's image
    placeholders, with only that vision row, so the vision tower runs too.

    :param rows: Rows to pad.
    :param num_rows: Row count to reach, at least the current count.
    :param width: Token width to reach, at least the current width.
    :param pad_token_id: Token id that pads rows to ``width``.
    :param image_token_id: Token id the VL forward scatters one image feature
        row into; required when ``rows`` carry vision rows.
    :return: The padded rows.
    :raises ValueError: If ``rows`` carry vision rows and ``image_token_id`` is
        ``None``.
    """
    extra = num_rows - int(rows.token_ids.shape[0])
    widen = (0, width - int(rows.token_ids.shape[1]))
    pixel_values = rows.pixel_values
    pixel_image_counts = rows.pixel_image_counts
    if pixel_values is not None and pixel_image_counts is not None:
        if image_token_id is None:
            msg = "image_token_id is required to pad vision segment rows."
            raise ValueError(msg)
        filler_ids, pixel_values, pixel_image_counts = append_vision_filler_rows(
            rows.token_ids, pixel_values, pixel_image_counts, image_token_id, extra
        )
    else:
        lengths = attention_mask_from_padded_ids(rows.token_ids, pad_token_id).sum(-1)
        shortest = int(lengths.argmin())
        filler_ids = rows.token_ids[
            shortest, : min(TEXT_FILLER_TOKENS, int(lengths[shortest]))
        ]
    pad = torch.nn.functional.pad
    filler = pad(filler_ids, (0, width - int(filler_ids.shape[0])), value=pad_token_id)
    token_ids = pad(rows.token_ids, widen, value=pad_token_id)
    token_ids = torch.cat([token_ids, filler.unsqueeze(0).expand(extra, -1)])
    action_masks = pad(rows.action_masks, widen, value=False)
    action_masks = torch.cat([action_masks, action_masks.new_zeros(extra, width - 1)])
    turn_ids = rows.turn_ids
    if turn_ids is not None:
        turn_ids = pad(turn_ids, widen, value=-1)
        turn_ids = torch.cat([turn_ids, turn_ids.new_full((extra, width - 1), -1)])
    sampling_logps = rows.sampling_logps
    if sampling_logps is not None:
        empty = token_ids.new_zeros(0, dtype=torch.float32)
        sampling_logps = [*sampling_logps, *([empty] * extra)]
    no_span = np.zeros(extra, dtype=np.intp)
    return SegmentRows(
        token_ids=token_ids,
        action_masks=action_masks,
        row_episodes=np.concatenate(
            [rows.row_episodes, np.full(extra, -1, dtype=np.intp)]
        ),
        row_starts=np.concatenate([rows.row_starts, no_span]),
        row_ends=np.concatenate([rows.row_ends, no_span]),
        turn_ids=turn_ids,
        sampling_logps=sampling_logps,
        pixel_values=pixel_values,
        pixel_image_counts=pixel_image_counts,
    )

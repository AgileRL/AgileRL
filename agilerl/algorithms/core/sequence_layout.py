# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Sequence layouts for padded, packed, and context-parallel actor forwards.

:class:`SequenceLayout` turns a right-padded ``(B, T)`` batch into the kwargs
one actor call consumes and the frame its log-probs are scored in, then
restores hidden states, values, or log-probs onto the caller's frame. It also
splits a fused pass into micro-batches. Primitives stay in
:mod:`agilerl.utils.llm_packing` and
:mod:`agilerl.distributed.context_parallel.sequence`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from itertools import accumulate, groupby, pairwise
from typing import Any, Literal, NamedTuple

import numpy as np
import numpy.typing as npt
import torch
import torch.distributed as dist
import torch.nn.functional as F
from typing_extensions import Self

from agilerl.distributed import allreduce_max_ints
from agilerl.distributed.context_parallel import (
    ULYSSES_PARAMS,
    clear_ulysses_params,
    update_ulysses_params,
)
from agilerl.distributed.context_parallel.sequence import (
    gather_for_cp,
    gather_for_cp_wo_grad,
    pad_right_for_cp,
    shard_for_cp,
    validate_cp_seq_len,
)
from agilerl.utils.llm_packing import (
    PackedBatch,
    pack_next_token_frame,
    pack_padded_batch,
    unpack_logprobs,
    unpack_values,
)
from agilerl.utils.token_micro_batches import pack_rows_by_tokens, split_micro_batches

RestoreKind = Literal["values", "logprobs"]


class HostMasks(NamedTuple):
    """Host copies of a batch's attention and score masks.

    Layouts read cut widths and scored counts from them without a device sync.

    :param real: ``(B, T)`` real-token mask.
    :param scored: ``(B, T - 1)`` positions to score, or ``None`` to score all.
    """

    real: npt.NDArray[np.bool_]
    scored: npt.NDArray[np.bool_] | None

    @classmethod
    def read(cls, mask: torch.Tensor, score_mask: torch.Tensor | None = None) -> Self:
        """Copy ``mask`` and ``score_mask`` to the host in one device sync.

        :param mask: ``(B, T)`` attention mask.
        :param score_mask: ``(B, T - 1)`` positions to score, or ``None``.
        :return: The host masks.
        """
        if score_mask is None:
            return cls(mask.bool().cpu().numpy(), None)
        width = mask.shape[1]
        both = torch.cat([mask.bool(), score_mask.to(mask.device).bool()], dim=1)
        host = both.cpu().numpy()
        return cls(host[:, :width], host[:, width:])

    def rows(self, start: int, end: int) -> HostMasks:
        """Masks of rows ``[start, end)``."""
        return HostMasks(
            self.real[start:end],
            None if self.scored is None else self.scored[start:end],
        )

    def width(self) -> int:
        """Columns up to and including the last real token of any row."""
        return int(np.flatnonzero(self.real.any(axis=0))[-1]) + 1

    def right_padded(self) -> bool:
        """Whether each row's real tokens are a prefix of the row."""
        lengths = self.real.sum(axis=1, keepdims=True)
        return bool(np.array_equal(self.real, np.arange(self.real.shape[1]) < lengths))

    def n_scored(self, start: int, stop: int) -> int | None:
        """Scored positions in columns ``[start, stop)``, or ``None`` without a score mask."""
        if self.scored is None:
            return None
        return int(self.scored[:, start:stop].sum())

    def packed_counts(self) -> list[int]:
        """Real tokens of each row, then the positions packing scores when there is a score mask."""
        counts = self.real.sum(axis=1).tolist()
        if self.scored is not None:
            kept = self.scored & self.real[:, 1:] & self.real[:, :-1]
            counts.append(int(kept.sum()))
        return counts

    def packed_scored(self) -> npt.NDArray[np.bool_] | None:
        """``(N - 1,)`` packed predictions :func:`pack_padded_batch` scores, or ``None`` without a score mask."""
        if self.scored is None:
            return None
        kept = self.scored & self.real[:, 1:] & self.real[:, :-1]
        boundary = np.zeros((kept.shape[0], 1), dtype=bool)
        return np.concatenate([kept, boundary], axis=1)[self.real][:-1]


def position_ids_from_mask(mask: torch.Tensor) -> torch.Tensor:
    """Left-padding-safe ``position_ids`` from an attention mask.

    Cumulative real-token count minus one, with padded positions pinned to
    ``1`` so the rotary embedding sees a valid (ignored) index.
    """
    position_ids = mask.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(mask=(mask == 0), value=1)
    return position_ids


def _cut_columns(
    seq_len: int,
    ids: torch.Tensor,
    mask: torch.Tensor,
    score_mask: torch.Tensor | None,
    support_ids: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Keep the first ``seq_len`` columns of a batch and ``seq_len - 1`` of its scoring frame."""
    if score_mask is not None:
        score_mask = score_mask[:, : seq_len - 1]
    if support_ids is not None:
        support_ids = support_ids[:, : seq_len - 1]
    return ids[:, :seq_len], mask[:, :seq_len], score_mask, support_ids


def _repeat_rows(pixel_values: torch.Tensor | None, copies: int) -> torch.Tensor | None:
    if pixel_values is None or copies == 1:
        return pixel_values
    if not any(pixel_values.stride()):
        # Every element aliases one value, so a view repeats it without a copy.
        return pixel_values[:1].expand(
            copies * pixel_values.shape[0], *pixel_values.shape[1:]
        )
    return pixel_values.repeat(copies, *([1] * (pixel_values.ndim - 1)))


def _packed_cp_offsets(
    packed: PackedBatch, n_predictions: int, action_len: int, copies: int
) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Device and host document offsets of ``copies`` padded packed rows of ``action_len``."""
    # The last document loses its final token to the shift; the pad tail is one more segment.
    row_offsets = F.pad(
        packed.cu_seqlens.clamp(max=n_predictions), (0, 1), value=action_len
    )
    cu_seqlens = torch.cat(
        [row_offsets[:-1] + copy * action_len for copy in range(copies)]
        + [row_offsets[-1:] + (copies - 1) * action_len]
    )
    host_row = [
        min(bound, n_predictions) for bound in accumulate(packed.lengths, initial=0)
    ]
    host_cu_seqlens = (
        *(bound + copy * action_len for copy in range(copies) for bound in host_row),
        copies * action_len,
    )
    return cu_seqlens, host_cu_seqlens


def _row_spans(
    total: int, routing: list[str], batch_size: int | None, split_adapter_runs: bool
) -> list[tuple[int, int]]:
    """``(start, end)`` of each forward of ``batch_size`` rows, in run order."""
    if batch_size is None and not split_adapter_runs:
        return [(0, total)]
    runs = (
        [len(list(run)) for _, run in groupby(routing)]
        if split_adapter_runs
        else [total]
    )
    spans: list[tuple[int, int]] = []
    for run_start, run_end in pairwise([0, *accumulate(runs)]):
        step = batch_size or run_end - run_start
        spans += [
            (start, min(start + step, run_end))
            for start in range(run_start, run_end, step)
        ]
    return spans


def _token_spans(
    routing: list[str], row_tokens: npt.NDArray[np.int64], token_budget: int
) -> list[tuple[int, int]]:
    """``(start, end)`` of each one-adapter forward of at most ``token_budget`` tokens.

    Ranks split their largest forwards until every rank runs the same number.
    A row longer than the budget runs alone.
    """
    spans: list[npt.NDArray[np.intp]] = []
    for run_start, run_end in pairwise(
        [0, *accumulate(len(list(run)) for _, run in groupby(routing))]
    ):
        spans += pack_rows_by_tokens(
            np.arange(run_start, run_end), row_tokens, token_budget
        )
    (count,) = allreduce_max_ints([len(spans)])
    spans = split_micro_batches(spans, row_tokens, count)
    return [(int(rows[0]), int(rows[-1]) + 1) for rows in spans]


def _pad_frame(tensor: torch.Tensor, width: int) -> torch.Tensor:
    trailing = width - tensor.shape[1]
    if trailing == 0:
        return tensor
    return F.pad(tensor, (0, 0) * (tensor.dim() - 2) + (0, trailing))


class SequenceForward:
    """Actor kwargs of one right-padded forward, its scoring frame, and restore onto the caller's frame.

    :param model_kwargs: ``actor(**kwargs)`` inputs.
    :param targets: ``(B, S - 1)`` next-token ids in the scoring frame.
    :param score_mask: Scoring-frame positions to score, or ``None`` for all.
    :param support_ids: Scoring-frame sampling support, or ``None``.
    :param n_scored: Host count of ``score_mask``, or ``None`` to read it.
    :param width: ``T`` of the caller's ``(B, T)`` batch.
    :param real_hidden: ``(B, S - 1)`` positions whose hidden state is a real
        token; restore zeroes the rest. ``None`` keeps every position.
    """

    def __init__(
        self,
        model_kwargs: dict[str, Any],
        targets: torch.Tensor,
        score_mask: torch.Tensor | None,
        support_ids: torch.Tensor | None,
        n_scored: int | None,
        width: int,
        real_hidden: torch.Tensor | None = None,
    ) -> None:
        self.kwargs = model_kwargs
        self.targets = targets
        self.score_mask = score_mask
        self.support_ids = support_ids
        self.n_scored = n_scored
        self.width = width
        self.real_hidden = real_hidden

    def labels(self, labels: torch.Tensor, ignore_index: int) -> torch.Tensor:
        """Next-token labels laid out like :attr:`targets`, ``ignore_index`` where padded."""
        width = self.targets.shape[1]
        labels = labels[:, :width]
        return F.pad(labels, (0, width - labels.shape[1]), value=ignore_index)

    def scoring_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        """Actor hidden states laid out like :attr:`targets`."""
        return hidden[:, :-1]

    def restore(self, tensor: torch.Tensor, kind: RestoreKind) -> torch.Tensor:
        """Map a forward's values or scored log-probs onto the caller's ``(B, T - 1)`` frame."""
        if kind == "values":
            tensor = tensor[:, :-1]
        if self.real_hidden is not None:
            tensor = tensor.masked_fill(~self.real_hidden, 0)
        return _pad_frame(tensor, self.width - 1)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


class PackedSequenceForward(SequenceForward):
    """One padding-free packed row; scores packed predictions and scatters them back.

    :param model_kwargs: ``actor(**kwargs)`` inputs.
    :param packed: The packed batch.
    :param support_ids: Sampling support on the packed predictions, or ``None``.
    """

    def __init__(
        self,
        model_kwargs: dict[str, Any],
        packed: PackedBatch,
        support_ids: torch.Tensor | None,
    ) -> None:
        super().__init__(
            model_kwargs,
            packed.input_ids[:, 1:],
            packed.score_mask,
            support_ids,
            packed.n_scored,
            packed.padded_seq_len,
        )
        self.packed = packed

    def labels(self, labels: torch.Tensor, ignore_index: int) -> torch.Tensor:
        return pack_next_token_frame(labels, self.packed, ignore_index)

    def restore(self, tensor: torch.Tensor, kind: RestoreKind) -> torch.Tensor:
        if kind == "values":
            return unpack_values(tensor, self.packed)
        return unpack_logprobs(tensor, self.packed)


class ContextParallelSequenceForward(SequenceForward):
    """Ulysses shards of a right-padded batch; each rank scores its own shard.

    Entering publishes the shard's varlen offsets to the Ulysses attention;
    exiting restores the params published before it, so forwards nest.

    :param model_kwargs: ``actor(**kwargs)`` inputs of this rank's shard.
    :param targets: ``(B, S_local)`` next-token ids of this rank's shard.
    :param score_mask: This rank's positions to score, or ``None``.
    :param support_ids: This rank's sampling support, or ``None``.
    :param n_scored: Host count of ``score_mask``, or ``None`` to read it.
    :param width: ``T`` of the caller's ``(B, T)`` batch.
    :param group: Context-parallel process group.
    :param cp: Context-parallel degree.
    :param rank: This rank in ``group``.
    :param action_len: ``S`` of the CP-aligned action frame across all ranks.
    :param cu_seqlens: Varlen offsets of the sharded rows.
    :param host_cu_seqlens: Host copy of ``cu_seqlens``.
    """

    def __init__(
        self,
        model_kwargs: dict[str, Any],
        targets: torch.Tensor,
        score_mask: torch.Tensor | None,
        support_ids: torch.Tensor | None,
        n_scored: int | None,
        width: int,
        group: dist.ProcessGroup,
        cp: int,
        rank: int,
        action_len: int,
        cu_seqlens: torch.Tensor,
        host_cu_seqlens: tuple[int, ...],
    ) -> None:
        super().__init__(
            model_kwargs, targets, score_mask, support_ids, n_scored, width
        )
        self.group = group
        self.cp = cp
        self.rank = rank
        self.action_len = action_len
        self.cu_seqlens = cu_seqlens
        self.host_cu_seqlens = host_cu_seqlens
        self.previous_params: dict[str, object] = {}

    def _gather(self, tensor: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled():
            return gather_for_cp(tensor, self.group)
        return gather_for_cp_wo_grad(tensor, self.cp, self.group)

    def labels(self, labels: torch.Tensor, ignore_index: int) -> torch.Tensor:
        labels = labels[:, : self.action_len]
        labels = F.pad(
            labels, (0, self.action_len - labels.shape[1]), value=ignore_index
        )
        return shard_for_cp(labels, self.rank, self.cp)

    def scoring_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden

    def restore(self, tensor: torch.Tensor, kind: RestoreKind) -> torch.Tensor:
        tensor = self._gather(tensor)
        if tensor.dim() == 3:
            tensor = tensor.squeeze(-1)
        return _pad_frame(tensor[:, : self.width - 1], self.width - 1)

    def __enter__(self) -> Self:
        self.previous_params = dict(ULYSSES_PARAMS)
        update_ulysses_params(self.cu_seqlens, self.host_cu_seqlens)
        return self

    def __exit__(self, *exc: object) -> None:
        clear_ulysses_params()
        ULYSSES_PARAMS.update(self.previous_params)


class PackedContextParallelSequenceForward(ContextParallelSequenceForward):
    """Ulysses shards of one packed row; restore unpacks the gathered outputs.

    :param model_kwargs: ``actor(**kwargs)`` inputs of this rank's shard.
    :param targets: ``(copies, S_local)`` next-token ids of this rank's shard.
    :param score_mask: This rank's positions to score, or ``None``.
    :param support_ids: This rank's sampling support, or ``None``.
    :param n_scored: Host count of ``score_mask``, or ``None`` to read it.
    :param group: Context-parallel process group.
    :param cp: Context-parallel degree.
    :param rank: This rank in ``group``.
    :param action_len: ``S`` of the padded packed predictions across all ranks.
    :param cu_seqlens: Document offsets of the stacked packed rows.
    :param host_cu_seqlens: Host copy of ``cu_seqlens``.
    :param packed: The packed batch.
    """

    def __init__(
        self,
        model_kwargs: dict[str, Any],
        targets: torch.Tensor,
        score_mask: torch.Tensor | None,
        support_ids: torch.Tensor | None,
        n_scored: int | None,
        group: dist.ProcessGroup,
        cp: int,
        rank: int,
        action_len: int,
        cu_seqlens: torch.Tensor,
        host_cu_seqlens: tuple[int, ...],
        packed: PackedBatch,
    ) -> None:
        super().__init__(
            model_kwargs,
            targets,
            score_mask,
            support_ids,
            n_scored,
            packed.padded_seq_len,
            group,
            cp,
            rank,
            action_len,
            cu_seqlens,
            host_cu_seqlens,
        )
        self.packed = packed

    def labels(self, labels: torch.Tensor, ignore_index: int) -> torch.Tensor:
        return super().labels(
            pack_next_token_frame(labels, self.packed, ignore_index), ignore_index
        )

    def restore(self, tensor: torch.Tensor, kind: RestoreKind) -> torch.Tensor:
        tensor = self._gather(tensor)
        if tensor.dim() == 3:
            tensor = tensor.squeeze(-1)
        tensor = tensor[:, : self.packed.input_ids.shape[1] - 1]
        if kind == "values":
            # The row's last token is not in the forward; no position scores its value.
            return unpack_values(F.pad(tensor, (0, 1)), self.packed)
        return unpack_logprobs(tensor, self.packed)


class SequenceLayout(ABC):
    """How an actor forward lays out sequence tokens."""

    skips_padding = False

    @abstractmethod
    def prepare_forward(
        self,
        ids: torch.Tensor,
        mask: torch.Tensor,
        score_mask: torch.Tensor | None = None,
        support_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        copies: int = 1,
        host_masks: HostMasks | None = None,
    ) -> SequenceForward:
        """Context manager of one actor forward over ``copies`` stacked copies of the batch.

        :param ids: ``(B, T)`` right-padded token ids.
        :param mask: ``(B, T)`` attention mask.
        :param score_mask: ``(B, T - 1)`` positions to score, or ``None`` for all.
        :param support_ids: ``(B, T - 1, K)`` sampling support, or ``None``.
        :param pixel_values: Vision rows of the batch, or ``None``.
        :param copies: Times the batch is stacked along rows, one per adapter.
        :param host_masks: Host copies of ``mask`` and ``score_mask``, or
            ``None`` to read what the layout needs from the device.
        """

    def micro_batch_spans(
        self,
        mask: torch.Tensor,
        routing: list[str],
        batch_size: int | None,
        token_budget: int | None,
        has_vision: bool,
    ) -> list[tuple[int, int]]:
        """``(start, end)`` row spans of a fused pass's forwards, in run order.

        A packed row and vision-tower LoRA layers each take one adapter, so
        those spans hold one adapter run.

        :param mask: ``(B, T)`` attention mask of the fused batch.
        :param routing: Adapter of each fused row.
        :param batch_size: Rows per forward, or ``None`` for one forward.
        :param token_budget: Most tokens per forward, in place of
            ``batch_size`` rows, or ``None``. Only layouts that skip padding
            read it.
        :param has_vision: Whether the pass carries vision rows.
        """
        if token_budget is not None and self.skips_padding:
            return _token_spans(routing, mask.sum(-1).cpu().numpy(), token_budget)
        return _row_spans(
            mask.shape[0], routing, batch_size, has_vision or self.skips_padding
        )

    @classmethod
    def micro_batch_token_budget(cls, rows: int, row_tokens: int) -> int | None:
        """Tokens of a ``rows``-row micro-batch of ``row_tokens``-token rows, or ``None`` to split by rows."""
        return rows * row_tokens if cls.skips_padding else None


class PaddedSequence(SequenceLayout):
    """Dense right-padded ``(B, T)`` forward.

    Without grad, columns that are padding in every row are cut before the
    forward and restore as zero; causal attention keeps earlier positions
    exact. A right-padded text batch then also runs without a mask or
    ``position_ids``: causal attention never lets a real token see the pad
    after it, the default positions match the mask-derived ones on every
    real token, and positions whose hidden state is padding restore as zero.
    Bidirectional spans only cover vision tokens, so a vision forward keeps
    its mask.

    :param calc_position_embeddings: Pass mask-derived ``position_ids``.
    """

    def __init__(self, calc_position_embeddings: bool) -> None:
        self.calc_position_embeddings = calc_position_embeddings

    def prepare_forward(
        self,
        ids: torch.Tensor,
        mask: torch.Tensor,
        score_mask: torch.Tensor | None = None,
        support_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        copies: int = 1,
        host_masks: HostMasks | None = None,
    ) -> SequenceForward:
        width = ids.shape[1]
        seq_len = width
        real_hidden = None
        if not torch.is_grad_enabled():
            if host_masks is None:
                host_masks = HostMasks.read(mask, score_mask)
            seq_len = host_masks.width()
            ids, mask, score_mask, support_ids = _cut_columns(
                seq_len, ids, mask, score_mask, support_ids
            )
            if pixel_values is None and host_masks.right_padded():
                real_hidden = mask[:, :-1].bool()
        n_scored = None if host_masks is None else host_masks.n_scored(0, seq_len - 1)
        targets = ids[:, 1:]
        if copies != 1:
            ids, mask = ids.repeat(copies, 1), mask.repeat(copies, 1)
        kwargs: dict[str, Any] = {"input_ids": ids, "use_cache": False}
        # Packing patches and transformers read position_ids without a mask as
        # packed rows, so a padded forward never sends position_ids alone.
        if real_hidden is None:
            kwargs["attention_mask"] = mask
            if self.calc_position_embeddings:
                kwargs["position_ids"] = position_ids_from_mask(mask)
        if pixel_values is not None:
            kwargs["pixel_values"] = _repeat_rows(pixel_values, copies)
        return SequenceForward(
            kwargs, targets, score_mask, support_ids, n_scored, width, real_hidden
        )


class PackedSequence(SequenceLayout):
    """Padding-free packed row on a varlen or blockmask backend.

    Every forward packs its batch into one row with per-sequence
    ``position_ids`` and no mask; transformers then keeps sequences
    attention-isolated per layer, sliding windows included. On
    FlashAttention-2 (``varlen``) a text forward also passes the varlen
    offsets and host max lengths, so no layer derives them with a device
    sync; FlexAttention builds a block-diagonal ``BlockMask``. A packed row
    routes one adapter, so ``copies`` stack it once per adapter.

    :param varlen: The actor runs FlashAttention-2.
    """

    skips_padding = True

    def __init__(self, varlen: bool) -> None:
        self.varlen = varlen

    def prepare_forward(
        self,
        ids: torch.Tensor,
        mask: torch.Tensor,
        score_mask: torch.Tensor | None = None,
        support_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        copies: int = 1,
        host_masks: HostMasks | None = None,
    ) -> SequenceForward:
        packed = pack_padded_batch(
            ids,
            mask,
            score_mask,
            None if host_masks is None else host_masks.packed_counts(),
        )
        kwargs: dict[str, Any] = {
            "input_ids": packed.input_ids.repeat(copies, 1),
            "position_ids": packed.position_ids.repeat(copies, 1),
            "use_cache": False,
        }
        if pixel_values is not None:
            kwargs["pixel_values"] = _repeat_rows(pixel_values, copies)
        # Vision towers pass their own varlen offsets to the same attention
        # kwargs, so a VL forward lets the text layers derive theirs.
        elif self.varlen:
            kwargs.update(packed.flash_attention_kwargs(copies))
        return PackedSequenceForward(
            kwargs,
            packed,
            None
            if support_ids is None
            else pack_next_token_frame(support_ids, packed, -1),
        )


class ContextParallelSequence(SequenceLayout):
    """Ulysses sequence-parallel shards of a right-padded batch.

    Cuts columns that are padding in every row, pads the action frame to a
    multiple of ``round_to``, and shards it with its targets, so each rank
    scores its own positions and restore gathers the per-token outputs.

    :param pad_token_id: Pad token of the batches.
    :param cp: Context-parallel degree.
    :param group: Context-parallel process group.
    :param rank: This rank in ``group``.
    :param round_to: Multiple of ``cp`` the action frame pads to.
    """

    def __init__(
        self,
        pad_token_id: int,
        cp: int,
        group: dist.ProcessGroup,
        rank: int,
        round_to: int,
    ) -> None:
        self.pad_token_id = pad_token_id
        self.cp = cp
        self.group = group
        self.rank = rank
        self.round_to = round_to

    def prepare_forward(
        self,
        ids: torch.Tensor,
        mask: torch.Tensor,
        score_mask: torch.Tensor | None = None,
        support_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        copies: int = 1,
        host_masks: HostMasks | None = None,
    ) -> SequenceForward:
        width = ids.shape[1]
        if host_masks is None:
            host_masks = HostMasks.read(mask, score_mask)
        seq_len = host_masks.width()
        ids, mask, score_mask, support_ids = _cut_columns(
            seq_len, ids, mask, score_mask, support_ids
        )
        ids, mask, score_mask, support_ids = pad_right_for_cp(
            ids,
            self.round_to,
            self.pad_token_id,
            mask,
            score_mask,
            support_ids,
        )
        targets = ids[:, 1:]
        if copies != 1:
            ids, mask = ids.repeat(copies, 1), mask.repeat(copies, 1)
        action_len = ids.shape[1] - 1
        validate_cp_seq_len(action_len, self.cp)
        positions = position_ids_from_mask(mask)[:, :action_len]
        sharded_ids = shard_for_cp(ids[:, :-1], self.rank, self.cp)
        # Packing patches treat position_ids without a mask as a packed row.
        kwargs: dict[str, Any] = {
            "input_ids": sharded_ids,
            "position_ids": shard_for_cp(positions, self.rank, self.cp),
            "attention_mask": shard_for_cp(mask[:, :action_len], self.rank, self.cp),
            "use_cache": False,
        }
        if pixel_values is not None:
            kwargs["pixel_values"] = _repeat_rows(pixel_values, copies)
        cu_seqlens = (
            torch.arange(sharded_ids.shape[0] + 1, device=ids.device, dtype=torch.int32)
            * action_len
        )
        shard_len = action_len // self.cp
        start = self.rank * shard_len
        return ContextParallelSequenceForward(
            kwargs,
            shard_for_cp(targets, self.rank, self.cp),
            None
            if score_mask is None
            else shard_for_cp(score_mask, self.rank, self.cp),
            None
            if support_ids is None
            else shard_for_cp(support_ids, self.rank, self.cp),
            host_masks.n_scored(start, min(start + shard_len, seq_len - 1)),
            width,
            self.group,
            self.cp,
            self.rank,
            action_len,
            cu_seqlens,
            tuple(range(0, (sharded_ids.shape[0] + 1) * action_len, action_len)),
        )


class PackedContextParallelSequence(ContextParallelSequence):
    """Ulysses shards of one packed row on FlashAttention-2.

    Packs the batch, pads the packed predictions to a multiple of
    ``round_to``, and shards them with their targets. Documents may cross
    shard boundaries: the all-to-all gives each rank the whole row for its
    heads, and flash attends within the published document offsets. Restore
    gathers the per-token outputs and unpacks them.

    :param pad_token_id: Pad token of the batches.
    :param cp: Context-parallel degree.
    :param group: Context-parallel process group.
    :param rank: This rank in ``group``.
    :param round_to: Multiple of ``cp`` the packed predictions pad to.
    """

    skips_padding = True

    def prepare_forward(
        self,
        ids: torch.Tensor,
        mask: torch.Tensor,
        score_mask: torch.Tensor | None = None,
        support_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        copies: int = 1,
        host_masks: HostMasks | None = None,
    ) -> SequenceForward:
        if host_masks is None:
            host_masks = HostMasks.read(mask, score_mask)
        packed = pack_padded_batch(ids, mask, score_mask, host_masks.packed_counts())
        n_predictions = packed.input_ids.shape[1] - 1
        action_len = -(-n_predictions // self.round_to) * self.round_to
        validate_cp_seq_len(action_len, self.cp)
        extra = action_len - n_predictions
        inputs = F.pad(packed.input_ids[:, :-1], (0, extra), value=self.pad_token_id)
        positions = F.pad(packed.position_ids[:, :-1], (0, extra))
        targets = F.pad(packed.input_ids[:, 1:], (0, extra), value=self.pad_token_id)
        packed_score_mask = (
            None if packed.score_mask is None else F.pad(packed.score_mask, (0, extra))
        )
        packed_support_ids = (
            None
            if support_ids is None
            else F.pad(
                pack_next_token_frame(support_ids, packed, -1),
                (0, 0, 0, extra),
                value=-1,
            )
        )
        kwargs: dict[str, Any] = {
            "input_ids": shard_for_cp(inputs.repeat(copies, 1), self.rank, self.cp),
            "position_ids": shard_for_cp(
                positions.repeat(copies, 1), self.rank, self.cp
            ),
            "use_cache": False,
        }
        if pixel_values is not None:
            kwargs["pixel_values"] = _repeat_rows(pixel_values, copies)
        cu_seqlens, host_cu_seqlens = _packed_cp_offsets(
            packed, n_predictions, action_len, copies
        )
        shard_len = action_len // self.cp
        start = self.rank * shard_len
        scored = host_masks.packed_scored()
        return PackedContextParallelSequenceForward(
            kwargs,
            shard_for_cp(targets, self.rank, self.cp),
            None
            if packed_score_mask is None
            else shard_for_cp(packed_score_mask, self.rank, self.cp),
            None
            if packed_support_ids is None
            else shard_for_cp(packed_support_ids, self.rank, self.cp),
            None if scored is None else int(scored[start : start + shard_len].sum()),
            self.group,
            self.cp,
            self.rank,
            action_len,
            cu_seqlens,
            host_cu_seqlens,
            packed,
        )

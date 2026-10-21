# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Sequence layouts for padded, packed, and context-parallel actor forwards.

:class:`SequenceLayout` turns a right-padded ``(B, T)`` batch into the kwargs
one actor call consumes, then restores hidden states, log-probs, or values onto
the padded frame. Primitives stay in :mod:`agilerl.utils.llm_packing` and
:mod:`agilerl.distributed.context_parallel.sequence`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Literal

import torch
import torch.distributed as dist
from typing_extensions import Self

from agilerl.distributed.context_parallel import (
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
    pack_padded_batch,
    unpack_hidden_states,
    unpack_logprobs,
    unpack_values,
)
from agilerl.utils.llm_utils import attention_mask_from_padded_ids

RestoreKind = Literal["hidden", "logprobs", "values"]


def position_ids_from_mask(mask: torch.Tensor) -> torch.Tensor:
    """Left-padding-safe ``position_ids`` from an attention mask.

    Cumulative real-token count minus one, with padded positions pinned to
    ``1`` so the rotary embedding sees a valid (ignored) index.
    """
    position_ids = mask.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(mask=(mask == 0), value=1)
    return position_ids


class SequenceForward:
    """Actor kwargs for one forward, plus restore onto the padded frame.

    Packed metadata and the CP group live here, not on the layout. Padded and
    packed exit is a no-op; CP publishes and clears Ulysses params.
    """

    def __init__(
        self,
        kwargs: dict[str, Any],
        *,
        packed: PackedBatch | None = None,
        restore: Callable[[torch.Tensor, RestoreKind], torch.Tensor],
        on_enter: Callable[[], None] | None = None,
        on_exit: Callable[[], None] | None = None,
    ) -> None:
        self.kwargs = kwargs
        self.packed = packed
        self._restore = restore
        self._on_enter = on_enter
        self._on_exit = on_exit

    def restore(self, tensor: torch.Tensor, kind: RestoreKind) -> torch.Tensor:
        """Map a layout-frame tensor back onto the padded batch."""
        return self._restore(tensor, kind)

    def __enter__(self) -> Self:
        if self._on_enter is not None:
            self._on_enter()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._on_exit is not None:
            self._on_exit()


class SequenceLayout(ABC):
    """How an actor forward lays out sequence tokens."""

    trim_trailing_batch_pad: bool

    def __init__(self, *, pad_token_id: int, calc_position_embeddings: bool) -> None:
        self.pad_token_id = pad_token_id
        self.calc_position_embeddings = calc_position_embeddings

    @abstractmethod
    def align(
        self,
        ids: torch.Tensor,
        *tensors: torch.Tensor | None,
        pad_values: tuple[int | float, ...] = (),
    ) -> tuple[torch.Tensor, ...]:
        """Pad ``ids`` and extras so a CP action frame splits evenly.

        Packed and padded layouts are identity. Same contract as
        :func:`pad_right_for_cp`.
        """

    @abstractmethod
    def actor_kwargs(
        self,
        ids: torch.Tensor,
        mask: torch.Tensor | None,
        extra: dict[str, Any] | None = None,
        copies: int = 1,
    ) -> SequenceForward:
        """Context manager yielding kwargs for ``actor(**kwargs)``.

        :param ids: ``(B, T)`` token ids.
        :param mask: ``(B, T)`` attention mask, or ``None`` to derive it.
        :param extra: Forward kwargs the layout must not rewrite (vision).
        :param copies: Times to stack the batch along rows, one per adapter.
        """


def _merge_extra(
    kwargs: dict[str, Any], extra: dict[str, Any] | None
) -> dict[str, Any]:
    if extra:
        kwargs.update(extra)
    return kwargs


def _mask_or_from_ids(
    ids: torch.Tensor, mask: torch.Tensor | None, pad_token_id: int
) -> torch.Tensor:
    if mask is not None:
        return mask
    return attention_mask_from_padded_ids(ids, pad_token_id)


def _restore_padded(tensor: torch.Tensor, kind: RestoreKind) -> torch.Tensor:
    if kind == "hidden":
        return tensor
    if kind == "logprobs":
        return tensor
    if kind == "values":
        return tensor[:, :-1]
    msg = f"unknown restore kind {kind!r}"
    raise ValueError(msg)


def _padded_kwargs(
    ids: torch.Tensor,
    mask: torch.Tensor | None,
    extra: dict[str, Any] | None,
    copies: int,
    *,
    pad_token_id: int,
    calc_position_embeddings: bool,
) -> dict[str, Any]:
    mask = _mask_or_from_ids(ids, mask, pad_token_id)
    if copies != 1:
        ids = ids.repeat(copies, 1)
        mask = mask.repeat(copies, 1)
    kwargs: dict[str, Any] = {"input_ids": ids, "use_cache": False}
    if calc_position_embeddings:
        kwargs["position_ids"] = position_ids_from_mask(mask)
    else:
        kwargs["attention_mask"] = mask
    return _merge_extra(kwargs, extra)


class PaddedSequence(SequenceLayout):
    """Dense right-padded ``(B, T)`` forward."""

    trim_trailing_batch_pad = True

    def align(
        self,
        ids: torch.Tensor,
        *tensors: torch.Tensor | None,
        pad_values: tuple[int | float, ...] = (),
    ) -> tuple[torch.Tensor, ...]:
        return pad_right_for_cp(
            ids, 1, self.pad_token_id, *tensors, pad_values=pad_values
        )

    def actor_kwargs(
        self,
        ids: torch.Tensor,
        mask: torch.Tensor | None,
        extra: dict[str, Any] | None = None,
        copies: int = 1,
    ) -> SequenceForward:
        return SequenceForward(
            _padded_kwargs(
                ids,
                mask,
                extra,
                copies,
                pad_token_id=self.pad_token_id,
                calc_position_embeddings=self.calc_position_embeddings,
            ),
            restore=_restore_padded,
        )


class PackedSequence(SequenceLayout):
    """Padding-free packed row on a varlen or blockmask backend.

    FA2 (``varlen``) packs only ``copies == 1``. Flex (``blockmask``) packs
    stacked rows. No-grad forwards stay padded.
    """

    trim_trailing_batch_pad = True

    def __init__(
        self,
        *,
        pad_token_id: int,
        calc_position_embeddings: bool,
        mode: str,
    ) -> None:
        super().__init__(
            pad_token_id=pad_token_id,
            calc_position_embeddings=calc_position_embeddings,
        )
        self.mode = mode

    def align(
        self,
        ids: torch.Tensor,
        *tensors: torch.Tensor | None,
        pad_values: tuple[int | float, ...] = (),
    ) -> tuple[torch.Tensor, ...]:
        return pad_right_for_cp(
            ids, 1, self.pad_token_id, *tensors, pad_values=pad_values
        )

    def actor_kwargs(
        self,
        ids: torch.Tensor,
        mask: torch.Tensor | None,
        extra: dict[str, Any] | None = None,
        copies: int = 1,
    ) -> SequenceForward:
        if not torch.is_grad_enabled() or (self.mode != "blockmask" and copies != 1):
            kwargs = _padded_kwargs(
                ids,
                mask,
                extra,
                copies,
                pad_token_id=self.pad_token_id,
                calc_position_embeddings=self.calc_position_embeddings,
            )
            # Packing patches treat position_ids without a mask as a packed row.
            if "attention_mask" not in kwargs:
                kwargs["attention_mask"] = _mask_or_from_ids(
                    kwargs["input_ids"], None, self.pad_token_id
                )
            return SequenceForward(kwargs, restore=_restore_padded)
        mask = _mask_or_from_ids(ids, mask, self.pad_token_id)
        packed = pack_padded_batch(ids, mask)
        kwargs = {
            "input_ids": packed.input_ids.repeat(copies, 1),
            "position_ids": packed.position_ids.repeat(copies, 1),
            "use_cache": False,
        }
        return SequenceForward(
            _merge_extra(kwargs, extra),
            packed=packed,
            restore=lambda tensor, kind, _packed=packed: _restore_packed(
                tensor, kind, _packed
            ),
        )


def _restore_packed(
    tensor: torch.Tensor, kind: RestoreKind, packed: PackedBatch
) -> torch.Tensor:
    if kind == "hidden":
        return unpack_hidden_states(tensor, packed)
    if kind == "logprobs":
        return unpack_logprobs(tensor, packed)
    if kind == "values":
        return unpack_values(tensor, packed)
    msg = f"unknown restore kind {kind!r}"
    raise ValueError(msg)


class ContextParallelSequence(SequenceLayout):
    """Ulysses sequence-parallel shards of a right-padded batch.

    Repeats rows for ``copies``, shards the action frame, and gathers on
    restore. Trailing pad stays: cropping it would leave a length that cannot
    split over ``cp``.
    """

    trim_trailing_batch_pad = False

    def __init__(
        self,
        *,
        pad_token_id: int,
        calc_position_embeddings: bool,
        cp: int,
        group: dist.ProcessGroup,
        rank: int,
    ) -> None:
        super().__init__(
            pad_token_id=pad_token_id,
            calc_position_embeddings=calc_position_embeddings,
        )
        self.cp = cp
        self.group = group
        self.rank = rank

    def align(
        self,
        ids: torch.Tensor,
        *tensors: torch.Tensor | None,
        pad_values: tuple[int | float, ...] = (),
    ) -> tuple[torch.Tensor, ...]:
        return pad_right_for_cp(
            ids, self.cp, self.pad_token_id, *tensors, pad_values=pad_values
        )

    def actor_kwargs(
        self,
        ids: torch.Tensor,
        mask: torch.Tensor | None,
        extra: dict[str, Any] | None = None,
        copies: int = 1,
    ) -> SequenceForward:
        mask = _mask_or_from_ids(ids, mask, self.pad_token_id)
        if copies != 1:
            ids = ids.repeat(copies, 1)
            mask = mask.repeat(copies, 1)
        action_len = ids.shape[1] - 1
        validate_cp_seq_len(action_len, self.cp)
        pos = position_ids_from_mask(mask)
        query = ids[:, :-1]
        positions = pos[:, :action_len]
        sharded_ids = shard_for_cp(query, self.rank, self.cp)
        sharded_pos = shard_for_cp(positions, self.rank, self.cp)
        n_rows = sharded_ids.shape[0]
        cu_seqlens = (
            torch.arange(
                n_rows + 1,
                device=ids.device,
                dtype=torch.int32,
            )
            * action_len
        )
        kwargs: dict[str, Any] = {
            "input_ids": sharded_ids,
            "position_ids": sharded_pos,
            "use_cache": False,
        }
        group = self.group
        cp = self.cp
        return SequenceForward(
            _merge_extra(kwargs, extra),
            restore=lambda tensor, kind: _restore_cp(tensor, kind, group, cp),
            on_enter=lambda: update_ulysses_params(cu_seqlens, action_len),
            on_exit=clear_ulysses_params,
        )


def _append_dummy_token(tensor: torch.Tensor) -> torch.Tensor:
    zeros = tensor.new_zeros(tensor.shape[0], 1, *tensor.shape[2:])
    return torch.cat([tensor, zeros], dim=1)


def _restore_cp(
    tensor: torch.Tensor,
    kind: RestoreKind,
    group: dist.ProcessGroup,
    cp: int,
) -> torch.Tensor:
    if torch.is_grad_enabled():
        gathered = gather_for_cp(tensor, group)
    else:
        gathered = gather_for_cp_wo_grad(tensor, cp, group)
    if kind == "hidden":
        return _append_dummy_token(gathered)
    if kind == "logprobs":
        return gathered
    if kind == "values":
        if gathered.dim() == 3:
            gathered = gathered.squeeze(-1)
        return gathered
    msg = f"unknown restore kind {kind!r}"
    raise ValueError(msg)

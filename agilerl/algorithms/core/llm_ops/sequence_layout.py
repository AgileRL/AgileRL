# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Compose packing and context-parallel sequence layout for train forwards.

``prepare`` lays the batch out for the actor; ``restore_*`` bring hidden
states, values, and log-probs back onto the padded full-sequence frame.
Generation never uses this module.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace

import torch
import torch.distributed as dist

from agilerl.algorithms.core.llm_ops.ulysses_attn import (
    clear_ulysses_params,
    update_ulysses_params,
)
from agilerl.distributed.context_parallel import (
    gather_for_cp,
    gather_for_cp_wo_grad,
    shard_for_cp,
    shard_training_bundle,
    validate_cp_seq_len,
)
from agilerl.distributed.expert_parallel import ParallelMesh
from agilerl.utils.llm_packing import (
    PackedBatch,
    pack_padded_batch,
    pad_packed_row_for_cp,
    unpack_hidden_states,
    unpack_logprobs,
    unpack_values,
)
from agilerl.utils.llm_utils import (
    attention_mask_from_padded_ids,
    fill_outside_mask,
)


def position_ids_from_mask(mask: torch.Tensor) -> torch.Tensor:
    """Left-padding-safe ``position_ids`` from an attention mask."""
    position_ids = mask.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(mask=(mask == 0), value=1)
    return position_ids


@dataclass
class PreparedSequence:
    """Actor inputs plus the metadata needed to restore a padded frame."""

    input_ids: torch.Tensor
    position_ids: torch.Tensor | None
    attention_mask: torch.Tensor | None
    original_ids: torch.Tensor
    original_mask: torch.Tensor
    packed: PackedBatch | None
    packed_real_len: int | None
    cu_seqlens: torch.Tensor | None
    max_seqlen: int | None
    shard_mask: torch.Tensor | None
    label_ids: torch.Tensor | None
    n_adapter_rows: int
    packed_layout: bool
    cp: int
    cp_rank: int = 0


@dataclass
class SequenceLayout:
    """Packing and CP layers for one train-time sequence layout.

    :param packing: Whether packing is allowed (flag plus a legal backend).
    :param cp: Context-parallel degree; 1 is replica.
    :param pad_token_id: Pad id for CP packed-row tails and left-pad checks.
    :param calc_position_embeddings: Write ``position_ids`` on padded forwards.
    :param mesh: Live parallel mesh; required when ``cp > 1``.
    """

    packing: bool
    cp: int
    pad_token_id: int
    calc_position_embeddings: bool
    mesh: ParallelMesh | None = None

    @property
    def uses_cp(self) -> bool:
        """True when the sequence is sharded across a CP group."""
        return self.cp > 1

    def should_pack(self, requires_grad: bool) -> bool:
        """Whether this forward packs, given whether gradients are needed."""
        if not self.packing:
            return False
        if self.cp > 1:
            return True
        return requires_grad

    def prepare(
        self,
        ids: torch.Tensor,
        mask: torch.Tensor | None,
        *,
        requires_grad: bool,
    ) -> PreparedSequence:
        """Lay ``ids`` out for the actor: pack and/or shard, or return as-is.

        :param ids: ``(B, T)`` right-padded token ids.
        :param mask: ``(B, T)`` attention mask, or ``None`` to infer from pad id.
        :param requires_grad: True on the gradient forward.
        :return: Actor inputs and restore metadata.
        """
        batch_ids = ids
        if mask is None:
            attention_mask = attention_mask_from_padded_ids(
                batch_ids, self.pad_token_id
            )
        else:
            attention_mask = mask.to(device=batch_ids.device)
        if self.should_pack(requires_grad):
            return self._prepare_packed(batch_ids, attention_mask)
        if self.uses_cp:
            return self._prepare_slice(batch_ids, attention_mask)
        position_ids = None
        if self.calc_position_embeddings:
            position_ids = position_ids_from_mask(attention_mask)
        return PreparedSequence(
            input_ids=batch_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            original_ids=batch_ids,
            original_mask=attention_mask,
            packed=None,
            packed_real_len=None,
            cu_seqlens=None,
            max_seqlen=None,
            shard_mask=None,
            label_ids=None,
            n_adapter_rows=1,
            packed_layout=False,
            cp=self.cp,
        )

    def expand_adapter_rows(
        self, prepared: PreparedSequence, n_adapters: int
    ) -> PreparedSequence:
        """Repeat the prepared row once per adapter and extend Ulysses lengths.

        :param prepared: Output of :meth:`prepare` on the pre-fusion batch.
        :param n_adapters: Number of fused adapter rows.
        :return: Prepared tensors with ``n_adapters`` rows.
        """
        if n_adapters <= 1:
            return prepared
        input_ids = prepared.input_ids.repeat(n_adapters, 1)
        position_ids = prepared.position_ids
        if position_ids is not None:
            position_ids = position_ids.repeat(n_adapters, 1)
        attention_mask = prepared.attention_mask
        if attention_mask is not None:
            attention_mask = attention_mask.repeat(n_adapters, 1)
        shard_mask = prepared.shard_mask
        if shard_mask is not None:
            shard_mask = shard_mask.repeat(n_adapters, 1)
        label_ids = prepared.label_ids
        if label_ids is not None:
            label_ids = label_ids.repeat(n_adapters, 1)
        cu_seqlens = prepared.cu_seqlens
        if cu_seqlens is not None and prepared.packed_layout:
            row_len = prepared.input_ids.shape[1]
            if self.uses_cp:
                row_len = row_len * self.cp
            offsets = [cu_seqlens[1:] + i * row_len for i in range(1, n_adapters)]
            cu_seqlens = torch.cat([cu_seqlens, *offsets])
        elif self.uses_cp and not prepared.packed_layout:
            action_len = prepared.original_ids.shape[1] - 1
            n_rows = input_ids.shape[0]
            cu_seqlens = (
                torch.arange(
                    n_rows + 1,
                    device=input_ids.device,
                    dtype=torch.int32,
                )
                * action_len
            )
        return replace(
            prepared,
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            shard_mask=shard_mask,
            label_ids=label_ids,
            cu_seqlens=cu_seqlens,
            n_adapter_rows=n_adapters,
        )

    def actor_kwargs(
        self,
        prepared: PreparedSequence,
        pixel_values: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | bool]:
        """Keyword args for ``actor`` from a prepared sequence."""
        kwargs: dict[str, torch.Tensor | bool] = {
            "input_ids": prepared.input_ids,
            "use_cache": False,
        }
        if prepared.position_ids is not None:
            kwargs["position_ids"] = prepared.position_ids
        elif prepared.attention_mask is not None:
            kwargs["attention_mask"] = prepared.attention_mask
        if pixel_values is not None:
            kwargs["pixel_values"] = pixel_values.to(device=prepared.input_ids.device)
        return kwargs

    @contextmanager
    def ulysses(self, prepared: PreparedSequence) -> Iterator[None]:
        """Publish Ulysses varlen params for a CP forward, then clear them."""
        if (
            not self.uses_cp
            or prepared.cu_seqlens is None
            or prepared.max_seqlen is None
        ):
            yield
            return
        update_ulysses_params(prepared.cu_seqlens, prepared.max_seqlen)
        try:
            yield
        finally:
            clear_ulysses_params()

    def restore_hidden(
        self, hidden: torch.Tensor, prepared: PreparedSequence
    ) -> torch.Tensor:
        """Gather and unpack hidden states onto ``(rows, T, H)``."""
        hidden = self._gather_seq(hidden, prepared)
        if prepared.packed is not None:
            n_real = prepared.packed_real_len
            if n_real is not None:
                hidden = hidden[:, :n_real]
            unpacked = [
                unpack_hidden_states(hidden[i : i + 1], prepared.packed)
                for i in range(prepared.n_adapter_rows)
            ]
            return unpacked[0] if len(unpacked) == 1 else torch.cat(unpacked, dim=0)
        if self.uses_cp:
            pad = hidden.new_zeros(hidden.shape[0], 1, *hidden.shape[2:])
            return torch.cat([hidden, pad], dim=1)
        return hidden

    def restore_values(
        self, value: torch.Tensor | None, prepared: PreparedSequence
    ) -> torch.Tensor | None:
        """Gather and unpack critic values onto the padded frame."""
        if value is None:
            return None
        if value.dim() == 3:
            value = value.squeeze(-1)
        value = self._gather_seq(value, prepared)
        if prepared.packed is not None:
            n_real = prepared.packed_real_len
            if n_real is not None:
                value = value[:, :n_real]
            values = [
                unpack_values(value[i], prepared.packed)
                for i in range(prepared.n_adapter_rows)
            ]
            value_out = values[0] if len(values) == 1 else torch.cat(values, dim=0)
            pad = value_out.new_zeros(value_out.shape[0], 1)
            return torch.cat([value_out, pad], dim=1)
        if self.uses_cp:
            pad = value.new_zeros(value.shape[0], 1)
            return torch.cat([value, pad], dim=1)
        return value

    def restore_logprobs(
        self, log_probs: torch.Tensor, prepared: PreparedSequence
    ) -> torch.Tensor:
        """Gather and unpack per-token log-probs onto ``(rows, T-1)``."""
        log_probs = self._gather_seq(log_probs, prepared)
        if prepared.packed is not None:
            n_real = prepared.packed.cu_seqlens[-1]
            flat = log_probs.reshape(-1)[:n_real]
            unpacked = unpack_logprobs(flat, prepared.packed)
            if prepared.n_adapter_rows == 1:
                return unpacked
            rows = [
                unpack_logprobs(log_probs[i].reshape(-1)[:n_real], prepared.packed)
                for i in range(prepared.n_adapter_rows)
            ]
            return torch.cat(rows, dim=0)
        return log_probs

    def cp_group_and_rank(self) -> tuple[dist.ProcessGroup, int]:
        """Return this rank's CP process group and rank in it."""
        mesh = self.mesh
        if mesh is None or mesh.cp is None:
            msg = f"cp={self.cp} has no context-parallel mesh on the actor."
            raise RuntimeError(msg)
        cp_group = mesh.cp_group
        return cp_group, dist.get_rank(cp_group)

    def shard_action_frame(self, tensor: torch.Tensor) -> torch.Tensor:
        """Cut an action-frame tensor ``(B, T-1, …)`` the same way as ``prepare``."""
        if not self.uses_cp:
            return tensor
        _, cp_rank = self.cp_group_and_rank()
        return shard_for_cp(tensor, cp_rank, self.cp, seq_dim=1)

    def _prepare_packed(
        self, batch_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> PreparedSequence:
        packed = pack_padded_batch(batch_ids, attention_mask)
        padded = pad_packed_row_for_cp(packed, self.pad_token_id, self.cp)
        input_ids = padded.input_ids
        position_ids = padded.position_ids
        cu_seqlens = padded.cu_seqlens.to(device=batch_ids.device, dtype=torch.int32)
        max_seqlen = padded.max_seqlen
        shard_mask = None
        cp_rank = 0
        if self.uses_cp:
            _, cp_rank = self.cp_group_and_rank()
            input_ids = shard_for_cp(input_ids, cp_rank, self.cp, seq_dim=1)
            position_ids = shard_for_cp(position_ids, cp_rank, self.cp, seq_dim=1)
        return PreparedSequence(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=None,
            original_ids=batch_ids,
            original_mask=attention_mask,
            packed=packed,
            packed_real_len=packed.input_ids.shape[1],
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            shard_mask=shard_mask,
            label_ids=None,
            n_adapter_rows=1,
            packed_layout=True,
            cp=self.cp,
            cp_rank=cp_rank,
        )

    def _prepare_slice(
        self, batch_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> PreparedSequence:
        _, cp_rank = self.cp_group_and_rank()
        action_mask = attention_mask[:, 1:].to(dtype=torch.float32).contiguous()
        self._reject_left_padded(batch_ids, action_mask.to(torch.bool))
        validate_cp_seq_len(action_mask.shape[1], self.cp)
        dummy = torch.zeros(
            (batch_ids.shape[0], 1),
            device=batch_ids.device,
            dtype=torch.float32,
        )
        zeros = torch.zeros_like(action_mask)
        bundle = shard_training_bundle(
            token_ids=batch_ids,
            mask=action_mask,
            advantages=dummy,
            old_log_probs=zeros,
            reference_log_probs=zeros,
            turn_ids=None,
            sampling_ratios=None,
            cp_rank=cp_rank,
            cp_size=self.cp,
        )
        action_len = action_mask.shape[1]
        positions = position_ids_from_mask(attention_mask)[:, :action_len]
        shard_positions = shard_for_cp(positions, cp_rank, self.cp, seq_dim=1)
        n_rows = bundle.query_ids.shape[0]
        cu_seqlens = (
            torch.arange(
                n_rows + 1,
                device=batch_ids.device,
                dtype=torch.int32,
            )
            * action_len
        )
        return PreparedSequence(
            input_ids=bundle.query_ids,
            position_ids=shard_positions,
            attention_mask=None,
            original_ids=batch_ids,
            original_mask=attention_mask,
            packed=None,
            packed_real_len=None,
            cu_seqlens=cu_seqlens,
            max_seqlen=action_len,
            shard_mask=bundle.mask,
            label_ids=bundle.label_ids,
            n_adapter_rows=1,
            packed_layout=False,
            cp=self.cp,
            cp_rank=cp_rank,
        )

    def _reject_left_padded(
        self, batch_ids: torch.Tensor, action_mask: torch.Tensor
    ) -> None:
        attention_mask = attention_mask_from_padded_ids(batch_ids, self.pad_token_id)
        action_full = torch.zeros_like(attention_mask)
        action_full[:, 1:] = action_mask.to(torch.bool)
        is_pad = batch_ids == self.pad_token_id
        if (is_pad & attention_mask & ~action_full).any():
            msg = (
                "CP loss needs right-padded batches: the dense sharded "
                "forward publishes full-row sequence lengths, so a row with "
                "pad tokens before real tokens would attend to padding."
            )
            raise ValueError(msg)

    def _gather_seq(
        self, data: torch.Tensor, prepared: PreparedSequence
    ) -> torch.Tensor:
        if prepared.shard_mask is not None:
            keep = prepared.shard_mask.to(torch.bool)
            if data.dim() > keep.dim():
                keep = keep.unsqueeze(-1)
            data = fill_outside_mask(data, keep)
        if not self.uses_cp:
            return data
        cp_group, _ = self.cp_group_and_rank()
        if torch.is_grad_enabled():
            return gather_for_cp(data, cp_group)
        return gather_for_cp_wo_grad(data, self.cp, cp_group)

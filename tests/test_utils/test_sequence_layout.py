# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for train-time packing and context-parallel sequence layout."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from agilerl.algorithms.core.llm_ops.sequence_layout import (
    SequenceLayout,
    position_ids_from_mask,
)


def _layout(**overrides: object) -> SequenceLayout:
    values: dict[str, object] = {
        "packing": False,
        "cp": 1,
        "pad_token_id": 0,
        "calc_position_embeddings": False,
        "mesh": None,
    }
    values.update(overrides)
    return SequenceLayout(**values)


class TestShouldPack:
    def test_replica_packing_is_grad_only(self) -> None:
        layout = _layout(packing=True)

        assert layout.should_pack(True) is True
        assert layout.should_pack(False) is False

    def test_packing_off_never_packs(self) -> None:
        layout = _layout(packing=False)

        assert layout.should_pack(True) is False
        assert layout.should_pack(False) is False

    def test_grpo_without_packing_uses_slice_under_cp(self) -> None:
        layout = _layout(packing=False, cp=2)

        assert layout.uses_cp is True
        assert layout.should_pack(True) is False
        assert layout.should_pack(False) is False

    def test_packed_cp_always_packs(self) -> None:
        layout = _layout(packing=True, cp=2)

        assert layout.should_pack(True) is True
        assert layout.should_pack(False) is True


class TestIdentityLayout:
    def test_prepare_and_restore_are_noops(self) -> None:
        layout = _layout(calc_position_embeddings=True)
        ids = torch.tensor([[1, 2, 0], [3, 0, 0]])
        mask = (ids != 0).long()

        prepared = layout.prepare(ids, mask, requires_grad=True)

        assert prepared.packed_layout is False
        assert torch.equal(prepared.input_ids, ids)
        assert torch.equal(prepared.position_ids, position_ids_from_mask(mask))
        hidden = torch.arange(24, dtype=torch.float32).view(2, 3, 4)
        assert torch.equal(layout.restore_hidden(hidden, prepared), hidden)


class TestPackOnlyLayout:
    def test_grad_forward_packs_and_restore_unpads(self) -> None:
        layout = _layout(packing=True)
        ids = torch.tensor([[1, 2, 0], [3, 0, 0]])
        mask = (ids != 0).long()

        prepared = layout.prepare(ids, mask, requires_grad=True)

        assert prepared.packed_layout is True
        assert prepared.input_ids.shape == (1, 3)
        hidden = torch.arange(12, dtype=torch.float32).view(1, 3, 4)
        restored = layout.restore_hidden(hidden, prepared)
        assert restored.shape == (2, 3, 4)
        assert torch.equal(restored[0, :2], hidden[0, :2])
        assert torch.equal(restored[1, :1], hidden[0, 2:3])

    def test_no_grad_stays_padded(self) -> None:
        layout = _layout(packing=True)
        ids = torch.tensor([[1, 2, 0]])
        mask = (ids != 0).long()

        prepared = layout.prepare(ids, mask, requires_grad=False)

        assert prepared.packed_layout is False
        assert torch.equal(prepared.input_ids, ids)


class TestSliceCpPrepare:
    def _cp_layout(self, packing: bool = False) -> SequenceLayout:
        mesh = SimpleNamespace(cp=object(), cp_group=object())
        return _layout(packing=packing, cp=2, calc_position_embeddings=True, mesh=mesh)

    def test_missing_mesh_raises(self) -> None:
        layout = _layout(cp=2)
        ids = torch.arange(8).view(1, 8)
        mask = torch.ones_like(ids)

        with pytest.raises(RuntimeError, match="no context-parallel mesh"):
            layout.prepare(ids, mask, requires_grad=True)

    def test_rejects_indivisible_action_len(self) -> None:
        layout = self._cp_layout()
        ids = torch.arange(1, 9).view(1, 8)
        mask = torch.ones_like(ids)

        with (
            patch(
                "agilerl.algorithms.core.llm_ops.sequence_layout.dist.get_rank",
                return_value=0,
            ),
            pytest.raises(ValueError, match="must be divisible by cp"),
        ):
            layout.prepare(ids, mask, requires_grad=True)

    def test_rejects_left_padded_rows(self) -> None:
        layout = self._cp_layout()
        ids = torch.tensor([[0, 0, 1, 2, 3]])
        mask = torch.tensor([[0, 0, 1, 1, 1]])

        with (
            patch(
                "agilerl.algorithms.core.llm_ops.sequence_layout.dist.get_rank",
                return_value=0,
            ),
            pytest.raises(ValueError, match="right-padded"),
        ):
            layout.prepare(ids, mask, requires_grad=True)

    def test_rank0_takes_first_half(self) -> None:
        layout = self._cp_layout()
        ids = torch.arange(1, 10).view(1, 9)
        mask = torch.ones_like(ids)

        with patch(
            "agilerl.algorithms.core.llm_ops.sequence_layout.dist.get_rank",
            return_value=0,
        ):
            prepared = layout.prepare(ids, mask, requires_grad=True)

        assert prepared.packed_layout is False
        assert prepared.input_ids.shape == (1, 4)
        assert torch.equal(prepared.input_ids[0], ids[0, :4])
        assert prepared.cp_rank == 0

    def test_restore_hidden_appends_dummy_last_token(self) -> None:
        layout = self._cp_layout()
        ids = torch.arange(1, 10).view(1, 9)
        mask = torch.ones_like(ids)
        hidden = torch.arange(12, dtype=torch.float32).view(1, 4, 3)

        with (
            patch(
                "agilerl.algorithms.core.llm_ops.sequence_layout.dist.get_rank",
                return_value=0,
            ),
            patch(
                "agilerl.algorithms.core.llm_ops.sequence_layout.gather_for_cp",
                side_effect=lambda data, _group: data,
            ),
        ):
            prepared = layout.prepare(ids, mask, requires_grad=True)
            restored = layout.restore_hidden(hidden, prepared)

        assert restored.shape == (1, 5, 3)
        assert torch.equal(restored[0, :4], hidden[0])
        assert torch.equal(restored[0, 4], torch.zeros(3))


class TestPackedCpPrepare:
    def test_pads_then_shards(self) -> None:
        mesh = SimpleNamespace(cp=object(), cp_group=object())
        layout = _layout(packing=True, cp=2, mesh=mesh)
        ids = torch.tensor([[1, 2, 3]])
        mask = torch.ones_like(ids)

        with patch(
            "agilerl.algorithms.core.llm_ops.sequence_layout.dist.get_rank",
            return_value=0,
        ):
            prepared = layout.prepare(ids, mask, requires_grad=False)

        assert prepared.packed_layout is True
        assert prepared.input_ids.shape[1] == 2
        assert prepared.packed_real_len == 3


class TestActorKwargs:
    def test_forwards_pixel_values(self) -> None:
        layout = _layout()
        ids = torch.zeros(1, 4, dtype=torch.long)
        prepared = layout.prepare(ids, None, requires_grad=False)
        pixels = torch.randn(1, 3, 2, 2)

        kwargs = layout.actor_kwargs(prepared, pixels)

        assert torch.equal(kwargs["pixel_values"], pixels)
        assert "attention_mask" in kwargs

    def test_omits_pixel_values_when_absent(self) -> None:
        layout = _layout()
        ids = torch.zeros(1, 4, dtype=torch.long)
        prepared = layout.prepare(ids, None, requires_grad=False)

        kwargs = layout.actor_kwargs(prepared)

        assert "pixel_values" not in kwargs


class TestExpandAdapterRows:
    def test_repeats_packed_row(self) -> None:
        layout = _layout(packing=True)
        ids = torch.tensor([[1, 2, 0], [3, 0, 0]])
        mask = (ids != 0).long()
        prepared = layout.prepare(ids, mask, requires_grad=True)

        expanded = layout.expand_adapter_rows(prepared, 2)

        assert expanded.input_ids.shape[0] == 2
        assert expanded.n_adapter_rows == 2
        assert torch.equal(expanded.input_ids[0], prepared.input_ids[0])
        assert torch.equal(expanded.input_ids[1], prepared.input_ids[0])

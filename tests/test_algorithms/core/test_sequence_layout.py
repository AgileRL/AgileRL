# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""CPU contracts for padded, packed, and context-parallel sequence layouts."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import torch

from agilerl.algorithms.core.base import LLMAlgorithm
from agilerl.algorithms.core.sequence_layout import (
    ContextParallelSequence,
    PackedSequence,
    PaddedSequence,
)
from tests.test_algorithms.test_core_base import _LLM_DEPS_SKIP, _make_llm_agent

PAD_TOKEN_ID = 0


def _padded() -> PaddedSequence:
    return PaddedSequence(pad_token_id=PAD_TOKEN_ID, calc_position_embeddings=False)


def _packed(mode: str = "varlen") -> PackedSequence:
    return PackedSequence(
        pad_token_id=PAD_TOKEN_ID,
        calc_position_embeddings=False,
        mode=mode,
    )


def _ids_and_mask() -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=torch.long)
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.long)
    return ids, mask


class TestPaddedSequence:
    def test_restore_is_identity_for_hidden(self) -> None:
        ids, mask = _ids_and_mask()
        hidden = torch.randn(2, 4, 8)

        with _padded().actor_kwargs(ids, mask) as fwd:
            restored = fwd.restore(hidden, "hidden")

        assert restored is hidden

    def test_align_is_identity(self) -> None:
        ids, mask = _ids_and_mask()
        labels = torch.arange(6).reshape(2, 3)

        aligned = _padded().align(ids, mask, labels, pad_values=(0, -100))

        assert aligned[0] is ids
        assert aligned[1] is mask
        assert aligned[2] is labels


class TestPackedSequence:
    def test_fa2_packs_one_row_when_copies_is_one(self) -> None:
        ids, mask = _ids_and_mask()
        n_real = int(mask.sum())

        with torch.enable_grad(), _packed("varlen").actor_kwargs(ids, mask) as fwd:
            assert fwd.kwargs["input_ids"].shape == (1, n_real)
            assert fwd.packed is not None

    def test_fa2_keeps_padded_rows_when_copies_is_two(self) -> None:
        ids, mask = _ids_and_mask()

        with (
            torch.enable_grad(),
            _packed("varlen").actor_kwargs(ids, mask, copies=2) as fwd,
        ):
            assert fwd.kwargs["input_ids"].shape == (4, 4)
            assert "attention_mask" in fwd.kwargs
            assert fwd.packed is None

    def test_fa2_padded_fallback_keeps_mask_with_position_ids(self) -> None:
        ids, mask = _ids_and_mask()
        layout = PackedSequence(
            pad_token_id=PAD_TOKEN_ID,
            calc_position_embeddings=True,
            mode="varlen",
        )

        with torch.enable_grad(), layout.actor_kwargs(ids, mask, copies=2) as fwd:
            assert fwd.kwargs["input_ids"].shape == (4, 4)
            assert "attention_mask" in fwd.kwargs
            assert "position_ids" in fwd.kwargs
            assert fwd.packed is None

    def test_flex_packs_stacked_rows(self) -> None:
        ids, mask = _ids_and_mask()
        n_real = int(mask.sum())

        with (
            torch.enable_grad(),
            _packed("blockmask").actor_kwargs(ids, mask, copies=2) as fwd,
        ):
            assert fwd.kwargs["input_ids"].shape == (2, n_real)
            assert fwd.packed is not None

    def test_no_grad_is_padded(self) -> None:
        ids, mask = _ids_and_mask()

        with torch.no_grad(), _packed("varlen").actor_kwargs(ids, mask) as fwd:
            assert fwd.kwargs["input_ids"].shape == ids.shape
            assert "attention_mask" in fwd.kwargs
            assert fwd.packed is None


class TestContextParallelSequence:
    @staticmethod
    def _layout(rank: int = 0) -> ContextParallelSequence:
        return ContextParallelSequence(
            pad_token_id=PAD_TOKEN_ID,
            calc_position_embeddings=True,
            cp=2,
            group=SimpleNamespace(),
            rank=rank,
        )

    def test_align_pads_sft_labels_with_ignore_index(self) -> None:
        ids = torch.ones(2, 6, dtype=torch.long)
        mask = torch.ones(2, 6, dtype=torch.long)
        labels = torch.full((2, 5), 7, dtype=torch.long)

        padded_ids, padded_mask, padded_labels = self._layout().align(
            ids, mask, labels, pad_values=(0, -100)
        )

        assert padded_ids.shape[1] == 7
        assert padded_mask.shape[1] == 7
        assert padded_labels.shape == (2, 6)
        assert torch.equal(padded_labels[:, -1], torch.full((2,), -100))

    def test_copies_two_shards_the_stacked_action_frame(self) -> None:
        ids = torch.arange(14).reshape(2, 7)
        mask = torch.ones_like(ids)

        with self._layout(rank=0).actor_kwargs(ids, mask, copies=2) as fwd:
            assert fwd.kwargs["input_ids"].shape == (4, 3)

        with self._layout(rank=1).actor_kwargs(ids, mask, copies=2) as fwd:
            assert fwd.kwargs["input_ids"].shape == (4, 3)

    def test_restore_gathers_hidden_and_appends_a_dummy_token(self) -> None:
        layout = self._layout()
        shard = torch.randn(2, 3, 4)

        def fake_gather(data: torch.Tensor, _group: object) -> torch.Tensor:
            return torch.cat([data, torch.zeros_like(data)], dim=1)

        ids = torch.arange(14).reshape(2, 7)
        mask = torch.ones_like(ids)
        with (
            patch(
                "agilerl.algorithms.core.sequence_layout.gather_for_cp",
                fake_gather,
            ),
            torch.enable_grad(),
            layout.actor_kwargs(ids, mask) as fwd,
        ):
            restored = fwd.restore(shard, "hidden")

        assert restored.shape == (2, 7, 4)


@_LLM_DEPS_SKIP
class TestSequenceLayoutSkip:
    def test_copy_clone_attributes_keeps_clone_layout(self) -> None:
        agent = _make_llm_agent(clone=True)
        clone = _make_llm_agent(clone=True)
        clone_layout = _packed()
        clone.sequence_layout = clone_layout
        agent.sequence_layout = _padded()
        agent.colocated = False
        clone.colocated = False

        out = LLMAlgorithm._copy_clone_attributes(agent, clone)

        assert out.sequence_layout is clone_layout

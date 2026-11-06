# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""CPU contracts for padded, packed, and context-parallel sequence layouts."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from agilerl.algorithms.core.base import LLMAlgorithm
from agilerl.algorithms.core.sequence_layout import (
    ContextParallelSequence,
    HostMasks,
    PackedContextParallelSequence,
    PackedSequence,
    PaddedSequence,
    position_ids_from_mask,
)
from agilerl.arena.memory import formulas
from agilerl.arena.memory.specs import ModelArch
from agilerl.arena.models.model_info import SUPPORTED_MODEL_INFO
from agilerl.distributed import FSDPConfig
from agilerl.distributed.context_parallel import (
    ULYSSES_PARAMS,
    clear_ulysses_params,
)
from agilerl.distributed.context_parallel.gdn import GDN_CP_FLAG
from agilerl.distributed.context_parallel.sequence import CP_SCAN_CHUNK
from tests.parallel_helpers import install_cpu_flash_attn, publish_varlen_offsets
from tests.rank_launch import run_ranks
from tests.test_algorithms.test_core_base import _LLM_DEPS_SKIP, _make_llm_agent
from tests.test_algorithms.test_llms.segment_helpers import RESOLVED_FSDP_MEMORY
from tests.test_algorithms.test_llms.test_context_parallel_learn import (
    PAD_TOKEN_ID as QWEN3_PAD_TOKEN_ID,
)
from tests.test_algorithms.test_llms.test_context_parallel_learn import (
    experiences,
    make_grpo,
    qwen3_model,
)

PAD_TOKEN_ID = 0

FLASH_VARLEN_KWARGS = ("cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k")


def _padded(calc_position_embeddings: bool = False) -> PaddedSequence:
    return PaddedSequence(calc_position_embeddings=calc_position_embeddings)


def _packed(varlen: bool = False) -> PackedSequence:
    return PackedSequence(varlen=varlen)


def _ids_and_mask() -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=torch.long)
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.long)
    return ids, mask


def _mask_of_lengths(lengths: list[int], width: int) -> torch.Tensor:
    return (torch.arange(width) < torch.tensor(lengths)[:, None]).long()


def _scored_logprobs(agent: LLMAlgorithm) -> np.ndarray:
    """``agent``'s log-probs of the ragged :func:`experiences` rows under its layout."""
    ids = torch.cat(experiences()[0])
    with torch.no_grad():
        log_probs = agent._get_logprobs(
            ids, ids.shape[0], attention_mask=(ids != QWEN3_PAD_TOKEN_ID).long()
        )
    return log_probs.numpy()


def _one_rank_layouts_worker(_rank: int, _world_size: int) -> dict[str, np.ndarray]:
    """Log-probs of one eager agent under the padded, packed and cp=1 layouts."""
    agent = make_grpo(qwen3_model(), False, None)
    layouts = {
        "padded": PaddedSequence(calc_position_embeddings=False),
        "packed": PackedSequence(varlen=False),
        "cp1": ContextParallelSequence(QWEN3_PAD_TOKEN_ID, 1, dist.group.WORLD, 0, 8),
    }
    scored = {}
    for name, layout in layouts.items():
        agent.sequence_layout = layout
        scored[name] = _scored_logprobs(agent)
    return scored


def _cp2_layout_worker(rank: int, _world_size: int) -> dict[str, np.ndarray]:
    """Log-probs of a cp=2 agent under its packed and padded layouts on Ulysses over the CPU flash stand-in."""
    # The default FSDP2 mesh follows the host accelerator (MPS on macOS);
    # these ranks train on CPU.
    torch._C._get_accelerator = lambda: torch.device("cpu")
    install_cpu_flash_attn()
    model = qwen3_model()
    model.config._attn_implementation = "flash_attention_2"
    agent = make_grpo(
        model,
        False,
        FSDPConfig(
            cp=2, param_dtype="float32", reduce_dtype="float32", **RESOLVED_FSDP_MEMORY
        ),
    )
    assert type(agent.sequence_layout) is PackedContextParallelSequence
    packed = _scored_logprobs(agent)
    agent.sequence_layout = ContextParallelSequence(
        QWEN3_PAD_TOKEN_ID, 2, agent.sequence_layout.group, rank, 2
    )
    return {"packed": packed, "padded": _scored_logprobs(agent)}


class TestHostMasksRead:
    def test_host_copy_gives_widths_and_counts(self) -> None:
        # Arrange
        _ids, mask = _ids_and_mask()
        score_mask = torch.tensor([[True, True, False], [True, False, True]])

        # Act
        host = HostMasks.read(mask, score_mask)

        # Assert
        assert host.width() == 3
        assert host.rows(1, 2).width() == 2
        assert host.n_scored(0, 3) == 4
        assert host.n_scored(1, 2) == 1
        assert host.packed_counts() == [3, 2, 3]

    def test_without_a_score_mask_counts_nothing(self) -> None:
        _ids, mask = _ids_and_mask()

        host = HostMasks.read(mask)

        assert host.n_scored(0, 3) is None
        assert host.packed_counts() == [3, 2]


class TestHostMasksRightPadded:
    def test_real_prefixes_are_right_padded(self) -> None:
        assert HostMasks.read(_ids_and_mask()[1]).right_padded()

    @pytest.mark.parametrize(
        "mask",
        [[[0, 1, 1], [1, 1, 0]], [[1, 0, 1], [1, 1, 1]]],
        ids=["left padding", "interior pad"],
    )
    def test_a_pad_before_a_real_token_is_not(self, mask: list[list[int]]) -> None:
        assert not HostMasks.read(torch.tensor(mask)).right_padded()


class TestPaddedSequencePrepareForward:
    def test_grad_forward_keeps_the_full_width(self) -> None:
        # Arrange
        ids, mask = _ids_and_mask()

        # Act
        with torch.enable_grad(), _padded().prepare_forward(ids, mask) as fwd:
            kwargs = fwd.kwargs

        # Assert
        assert torch.equal(kwargs["input_ids"], ids)
        assert torch.equal(kwargs["attention_mask"], mask)
        assert kwargs["use_cache"] is False
        assert "position_ids" not in kwargs
        assert torch.equal(fwd.targets, ids[:, 1:])

    def test_keeps_the_mask_beside_position_ids(self) -> None:
        ids, mask = _ids_and_mask()

        with torch.enable_grad(), _padded(True).prepare_forward(ids, mask) as fwd:
            kwargs = fwd.kwargs

        assert torch.equal(kwargs["attention_mask"], mask)
        assert torch.equal(kwargs["position_ids"], position_ids_from_mask(mask))

    def test_no_grad_cuts_columns_that_are_padding_in_every_row(self) -> None:
        # Arrange
        ids, mask = _ids_and_mask()
        score_mask = torch.ones(2, 3, dtype=torch.bool)
        support_ids = torch.zeros(2, 3, 2, dtype=torch.long)

        # Act
        with (
            torch.no_grad(),
            _padded().prepare_forward(ids, mask, score_mask, support_ids) as fwd,
        ):
            kwargs = fwd.kwargs

        # Assert
        assert torch.equal(kwargs["input_ids"], ids[:, :3])
        assert torch.equal(fwd.targets, ids[:, 1:3])
        assert fwd.score_mask.shape == (2, 2)
        assert fwd.support_ids.shape == (2, 2, 2)
        assert fwd.n_scored == 4

    def test_host_masks_set_the_cut_and_the_scored_count(self) -> None:
        # Arrange: the host copy marks one real column fewer than the mask.
        ids, mask = _ids_and_mask()
        score_mask = torch.ones(2, 3, dtype=torch.bool)
        host = HostMasks.read(mask[:, :2], score_mask[:, :1])

        # Act
        with (
            torch.no_grad(),
            _padded().prepare_forward(ids, mask, score_mask, host_masks=host) as fwd,
        ):
            kwargs = fwd.kwargs

        # Assert
        assert torch.equal(kwargs["input_ids"], ids[:, :2])
        assert fwd.n_scored == 2

    def test_grad_forward_counts_the_scored_positions_from_host_masks(self) -> None:
        ids, mask = _ids_and_mask()
        score_mask = torch.tensor([[True, True, False], [True, False, False]])

        with (
            torch.enable_grad(),
            _padded().prepare_forward(
                ids, mask, score_mask, host_masks=HostMasks.read(mask, score_mask)
            ) as fwd,
        ):
            kwargs = fwd.kwargs

        assert torch.equal(kwargs["input_ids"], ids)
        assert fwd.n_scored == 3

    def test_restore_pads_the_cut_columns_with_zeros(self) -> None:
        # Arrange
        ids, mask = _ids_and_mask()
        values = torch.arange(1.0, 7.0).reshape(2, 3)
        log_probs = torch.ones(2, 2)

        # Act
        with torch.no_grad(), _padded().prepare_forward(ids, mask) as fwd:
            restored_values = fwd.restore(values, "values")
            restored_log_probs = fwd.restore(log_probs, "logprobs")

        # Assert
        assert torch.equal(restored_values, torch.tensor([[1.0, 2, 0], [4, 5, 0]]))
        assert torch.equal(restored_log_probs, torch.tensor([[1.0, 1, 0], [1, 1, 0]]))

    def test_labels_crop_to_the_scoring_frame(self) -> None:
        ids, mask = _ids_and_mask()
        labels = torch.tensor([[10, 11, -100], [20, -100, -100]])

        with torch.no_grad(), _padded().prepare_forward(ids, mask) as fwd:
            cropped = fwd.labels(labels, -100)

        assert torch.equal(cropped, labels[:, :2])

    def test_copies_stack_rows_and_pixel_values(self) -> None:
        ids, mask = _ids_and_mask()
        pixel_values = torch.arange(6.0).reshape(2, 3)

        with (
            torch.enable_grad(),
            _padded().prepare_forward(
                ids, mask, pixel_values=pixel_values, copies=2
            ) as fwd,
        ):
            kwargs = fwd.kwargs

        assert torch.equal(kwargs["input_ids"], torch.cat([ids, ids]))
        assert torch.equal(kwargs["attention_mask"], torch.cat([mask, mask]))
        assert torch.equal(
            kwargs["pixel_values"], torch.cat([pixel_values, pixel_values])
        )
        assert torch.equal(fwd.targets, ids[:, 1:])

    def test_copies_of_aliased_pixel_values_stay_a_view(self) -> None:
        ids, mask = _ids_and_mask()
        pixel_values = torch.zeros(1).expand(2, 3)

        with (
            torch.enable_grad(),
            _padded().prepare_forward(
                ids, mask, pixel_values=pixel_values, copies=2
            ) as fwd,
        ):
            repeated = fwd.kwargs["pixel_values"]

        assert repeated.shape == (4, 3)
        assert repeated.stride() == (0, 0)

    def test_no_grad_right_padded_text_drops_the_mask_and_position_ids(
        self,
    ) -> None:
        ids, mask = _ids_and_mask()

        with torch.no_grad(), _padded(True).prepare_forward(ids, mask) as fwd:
            kwargs = fwd.kwargs

        assert set(kwargs) == {"input_ids", "use_cache"}
        assert torch.equal(kwargs["input_ids"], ids[:, :3])

    @pytest.mark.parametrize(
        "mask",
        [[[0, 1, 1, 1], [1, 1, 0, 0]], [[1, 0, 1, 1], [1, 1, 0, 0]]],
        ids=["left padding", "interior pad"],
    )
    def test_no_grad_keeps_the_mask_when_a_pad_precedes_a_real_token(
        self, mask: list[list[int]]
    ) -> None:
        # Arrange
        ids = torch.tensor([[1, 2, 3, 4], [5, 6, 0, 0]])
        mask = torch.tensor(mask)

        # Act
        with torch.no_grad(), _padded(True).prepare_forward(ids, mask) as fwd:
            kwargs = fwd.kwargs

        # Assert
        assert torch.equal(kwargs["attention_mask"], mask)
        assert torch.equal(kwargs["position_ids"], position_ids_from_mask(mask))

    def test_no_grad_vision_forward_keeps_the_mask(self) -> None:
        ids, mask = _ids_and_mask()

        with (
            torch.no_grad(),
            _padded().prepare_forward(ids, mask, pixel_values=torch.ones(2, 3)) as fwd,
        ):
            kwargs = fwd.kwargs

        assert torch.equal(kwargs["attention_mask"], mask[:, :3])

    def test_no_grad_restore_zeroes_positions_whose_hidden_state_is_padding(
        self,
    ) -> None:
        # Arrange: row 1's last scoring column reads a pad hidden state.
        ids = torch.tensor([[1, 2, 3, 4], [5, 6, 0, 0]])
        mask = _mask_of_lengths([4, 2], 4)
        values = torch.arange(1.0, 9.0).reshape(2, 4)
        log_probs = torch.ones(2, 3)

        # Act
        with torch.no_grad(), _padded().prepare_forward(ids, mask) as fwd:
            restored_values = fwd.restore(values, "values")
            restored_log_probs = fwd.restore(log_probs, "logprobs")

        # Assert
        assert torch.equal(restored_values, torch.tensor([[1.0, 2, 3], [5, 6, 0]]))
        assert torch.equal(restored_log_probs, torch.tensor([[1.0, 1, 1], [1, 1, 0]]))


class TestSequenceLayoutMicroBatchSpans:
    def test_splits_rows_across_adapter_runs(self) -> None:
        mask = torch.ones(5, 4, dtype=torch.long)

        spans = _padded().micro_batch_spans(
            mask, ["a", "a", "a", "b", "b"], 2, None, False
        )

        assert spans == [(0, 2), (2, 4), (4, 5)]

    def test_vision_spans_hold_one_adapter(self) -> None:
        mask = torch.ones(5, 4, dtype=torch.long)

        spans = _padded().micro_batch_spans(
            mask, ["a", "a", "a", "b", "b"], 2, None, True
        )

        assert spans == [(0, 2), (2, 3), (3, 5)]

    def test_no_batch_size_is_one_forward(self) -> None:
        mask = torch.ones(5, 4, dtype=torch.long)

        spans = _padded().micro_batch_spans(
            mask, ["a", "a", "a", "b", "b"], None, None, False
        )

        assert spans == [(0, 5)]

    def test_padded_ignores_a_token_budget(self) -> None:
        mask = _mask_of_lengths([3, 3, 3, 2, 2], 4)

        spans = _padded().micro_batch_spans(
            mask, ["a", "a", "a", "b", "b"], 2, 6, False
        )

        assert spans == [(0, 2), (2, 4), (4, 5)]


class TestSequenceLayoutMicroBatchTokenBudget:
    def test_padded_splits_by_rows(self) -> None:
        assert _padded().micro_batch_token_budget(4, 16) is None

    def test_packed_budgets_every_row_at_row_tokens(self) -> None:
        assert _packed().micro_batch_token_budget(4, 16) == 64


class TestPackedSequencePrepareForward:
    def test_packs_the_batch_into_one_row(self) -> None:
        ids, mask = _ids_and_mask()

        with _packed().prepare_forward(ids, mask) as fwd:
            kwargs = fwd.kwargs

        assert kwargs["input_ids"].tolist() == [[1, 2, 3, 4, 5]]
        assert kwargs["position_ids"].tolist() == [[0, 1, 2, 0, 1]]
        assert "attention_mask" not in kwargs
        assert fwd.targets.tolist() == [[2, 3, 4, 5]]

    def test_host_masks_give_the_packed_sizes(self) -> None:
        ids, mask = _ids_and_mask()
        score_mask = torch.ones(2, 3, dtype=torch.bool)

        with _packed().prepare_forward(
            ids, mask, score_mask, host_masks=HostMasks.read(mask, score_mask)
        ) as fwd:
            kwargs = fwd.kwargs

        assert kwargs["input_ids"].tolist() == [[1, 2, 3, 4, 5]]
        assert fwd.n_scored == 3
        assert fwd.score_mask.tolist() == [[True, True, False, True]]

    def test_copies_stack_the_packed_row(self) -> None:
        ids, mask = _ids_and_mask()

        with _packed().prepare_forward(ids, mask, copies=2) as fwd:
            assert fwd.kwargs["input_ids"].tolist() == [[1, 2, 3, 4, 5]] * 2

    @pytest.mark.parametrize(
        ("varlen", "pixel_values", "has_offsets"),
        [
            (True, None, True),
            (True, torch.zeros(2, 3), False),
            (False, None, False),
        ],
        ids=["varlen-text", "varlen-vision", "blockmask"],
    )
    def test_only_text_varlen_forwards_pass_offsets(
        self, varlen: bool, pixel_values: torch.Tensor | None, has_offsets: bool
    ) -> None:
        # Arrange
        ids, mask = _ids_and_mask()

        # Act
        with _packed(varlen).prepare_forward(
            ids, mask, pixel_values=pixel_values
        ) as fwd:
            kwargs = fwd.kwargs

        # Assert
        assert kwargs["position_ids"].tolist() == [[0, 1, 2, 0, 1]]
        assert all((name in kwargs) is has_offsets for name in FLASH_VARLEN_KWARGS)

    def test_labels_and_support_move_to_the_packed_frame(self) -> None:
        # Arrange
        ids, mask = _ids_and_mask()
        labels = torch.tensor([[10, 11, -100], [20, -100, -100]])
        support_ids = torch.tensor([[7, 8, 9], [6, 5, 4]]).unsqueeze(-1)

        # Act
        with _packed().prepare_forward(ids, mask, support_ids=support_ids) as fwd:
            packed_labels = fwd.labels(labels, -100)

        # Assert: the row boundary slot takes the row's last padded-frame value.
        assert packed_labels.tolist() == [[10, 11, -100, 20]]
        assert fwd.support_ids.squeeze(-1).tolist() == [[7, 8, 9, 6]]

    def test_restore_scatters_back_onto_the_padded_frame(self) -> None:
        # Arrange
        ids, mask = _ids_and_mask()
        packed_out = torch.tensor([[10.0, 11.0, 12.0, 20.0, 21.0]])
        packed_log_probs = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

        # Act
        with _packed().prepare_forward(ids, mask) as fwd:
            values = fwd.restore(packed_out, "values")
            log_probs = fwd.restore(packed_log_probs, "logprobs")

        # Assert
        assert values.tolist() == [[10, 11, 12], [20, 21, 0]]
        assert log_probs.tolist() == [[1, 2, 0], [4, 0, 0]]


class TestPackedSequenceMicroBatchSpans:
    def test_rows_split_by_adapter_run(self) -> None:
        mask = torch.ones(5, 4, dtype=torch.long)

        spans = _packed().micro_batch_spans(
            mask, ["a", "a", "a", "b", "b"], None, None, False
        )

        assert spans == [(0, 3), (3, 5)]

    def test_token_budget_groups_rows_within_each_run(self) -> None:
        mask = _mask_of_lengths([3, 3, 3, 2, 2], 4)

        spans = _packed().micro_batch_spans(
            mask, ["a", "a", "a", "b", "b"], 2, 6, False
        )

        assert spans == [(0, 2), (2, 3), (3, 5)]


class TestContextParallelSequencePrepareForward:
    @staticmethod
    def _layout(
        rank: int = 0, round_to: int = 2 * CP_SCAN_CHUNK
    ) -> ContextParallelSequence:
        return ContextParallelSequence(
            pad_token_id=PAD_TOKEN_ID,
            cp=2,
            group=SimpleNamespace(),
            rank=rank,
            round_to=round_to,
        )

    def test_shards_keep_a_mask_beside_position_ids(self) -> None:
        ids = torch.tensor([[1, 2, 3, 4, 0, 0, 0]])
        mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0]])

        with self._layout(rank=0).prepare_forward(ids, mask) as fwd:
            sharded_ids = fwd.kwargs["input_ids"]
            sharded_mask = fwd.kwargs["attention_mask"]
            sharded_pos = fwd.kwargs["position_ids"]

        assert sharded_ids.shape == (1, 64)
        assert torch.equal(sharded_ids[0, :4], torch.tensor([1, 2, 3, 4]))
        assert torch.equal(sharded_mask[0, :4], torch.ones(4, dtype=torch.long))
        assert torch.equal(sharded_mask[0, 4:], torch.zeros(60, dtype=torch.long))
        assert sharded_pos.shape == sharded_ids.shape

    def test_host_masks_count_the_scored_positions(self) -> None:
        ids = torch.tensor([[1, 2, 3, 4, 0, 0, 0]])
        mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0]])
        score_mask = torch.tensor([[False, True, True, False, False, False]])

        with self._layout().prepare_forward(
            ids, mask, score_mask, host_masks=HostMasks.read(mask, score_mask)
        ) as fwd:
            n_scored = fwd.n_scored

        assert n_scored == 2

    def test_cuts_padding_columns_before_rounding(self) -> None:
        # Arrange
        ids = torch.tensor([[1, 2, 3, 4, 0, 0, 0]])
        mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0]])

        # Act
        with self._layout(rank=1, round_to=2).prepare_forward(ids, mask) as fwd:
            sharded_ids = fwd.kwargs["input_ids"]
            targets = fwd.targets

        # Assert: 3 action tokens round up to 4, so each rank holds 2.
        assert torch.equal(sharded_ids, torch.tensor([[3, 4]]))
        assert torch.equal(targets, torch.tensor([[4, 0]]))

    def test_restore_pads_cut_log_probs_back_to_the_input_frame(self) -> None:
        # Arrange
        ids = torch.tensor([[1, 2, 3, 4, 0, 0, 0]])
        mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0]])

        def fake_gather(data: torch.Tensor, cp: int, _group: object) -> torch.Tensor:
            return torch.cat([data] * cp, dim=1)

        # Act
        with (
            patch(
                "agilerl.algorithms.core.sequence_layout.gather_for_cp_wo_grad",
                fake_gather,
            ),
            torch.no_grad(),
            self._layout(round_to=2).prepare_forward(ids, mask) as fwd,
        ):
            log_probs = fwd.restore(torch.ones(1, 2), "logprobs")

        # Assert
        assert torch.equal(log_probs, torch.tensor([[1.0, 1.0, 1.0, 1.0, 0.0, 0.0]]))

    @pytest.mark.parametrize(("rank", "n_scored"), [(0, 1), (1, 2)])
    def test_each_rank_scores_its_own_shard(self, rank: int, n_scored: int) -> None:
        # Arrange
        ids = torch.tensor([[1, 2, 3, 4, 5]])
        mask = torch.ones_like(ids)
        score_mask = torch.tensor([[False, True, True, True]])
        support_ids = torch.arange(4).reshape(1, 4, 1)

        # Act
        with self._layout(rank=rank, round_to=2).prepare_forward(
            ids,
            mask,
            score_mask,
            support_ids,
            host_masks=HostMasks.read(mask, score_mask),
        ) as fwd:
            sharded_ids = fwd.kwargs["input_ids"]
            hidden = torch.randn(1, 2, 3)
            scoring = fwd.scoring_hidden(hidden)

        # Assert: shard positions predict the targets at the same positions.
        local = slice(2 * rank, 2 * rank + 2)
        assert torch.equal(sharded_ids, ids[:, :-1][:, local])
        assert torch.equal(fwd.targets, ids[:, 1:][:, local])
        assert torch.equal(fwd.score_mask, score_mask[:, local])
        assert torch.equal(fwd.support_ids, support_ids[:, local])
        assert fwd.n_scored == n_scored
        assert scoring is hidden

    def test_copies_shard_the_stacked_action_frame(self) -> None:
        ids = torch.arange(14).reshape(2, 7)
        mask = torch.ones_like(ids)

        with self._layout(rank=0).prepare_forward(ids, mask, copies=2) as fwd:
            assert fwd.kwargs["input_ids"].shape == (4, 64)
        with self._layout(rank=1).prepare_forward(ids, mask, copies=2) as fwd:
            assert fwd.kwargs["input_ids"].shape == (4, 64)

    def test_labels_pad_with_the_ignore_index(self) -> None:
        ids = torch.ones(2, 6, dtype=torch.long)
        mask = torch.ones(2, 6, dtype=torch.long)
        labels = torch.full((2, 5), 7, dtype=torch.long)

        with self._layout().prepare_forward(ids, mask) as fwd:
            padded_labels = fwd.labels(labels, -100)

        assert padded_labels.shape == (2, 64)
        assert torch.equal(padded_labels[:, :5], labels)
        assert torch.equal(padded_labels[:, 5:], torch.full((2, 59), -100))

    def test_grad_restore_gathers_log_probs_differentiably(self) -> None:
        # Arrange
        shard = torch.randn(2, 64, requires_grad=True)
        ids = torch.arange(14).reshape(2, 7)
        mask = torch.ones_like(ids)

        def fake_gather(data: torch.Tensor, _group: object) -> torch.Tensor:
            return torch.cat([data, data.detach()], dim=1)

        # Act
        with (
            patch("agilerl.algorithms.core.sequence_layout.gather_for_cp", fake_gather),
            torch.enable_grad(),
            self._layout().prepare_forward(ids, mask) as fwd,
        ):
            log_probs = fwd.restore(shard, "logprobs")
        log_probs.sum().backward()

        # Assert
        assert torch.equal(log_probs, shard[:, :6])
        assert torch.equal(shard.grad[:, :6], torch.ones(2, 6))
        assert torch.equal(shard.grad[:, 6:], torch.zeros(2, 58))

    def test_no_grad_restore_squeezes_gathered_values_and_crops(self) -> None:
        # Arrange
        shard = torch.randn(2, 3, 1)
        ids = torch.arange(14).reshape(2, 7)
        mask = torch.ones_like(ids)

        def fake_gather(data: torch.Tensor, cp: int, _group: object) -> torch.Tensor:
            return torch.cat([data] * cp, dim=1)

        # Act
        with (
            patch(
                "agilerl.algorithms.core.sequence_layout.gather_for_cp_wo_grad",
                fake_gather,
            ),
            torch.no_grad(),
            self._layout().prepare_forward(ids, mask) as fwd,
        ):
            values = fwd.restore(shard, "values")
            log_probs = fwd.restore(torch.ones(2, 64), "logprobs")

        # Assert
        assert torch.equal(values, torch.cat([shard, shard], dim=1).squeeze(-1))
        assert torch.equal(log_probs, torch.ones(2, 6))


class TestContextParallelSequenceForwardUlyssesParams:
    @staticmethod
    def _forward(width: int):
        layout = TestContextParallelSequencePrepareForward._layout()
        ids = torch.ones(1, width, dtype=torch.long)
        return layout.prepare_forward(ids, torch.ones_like(ids))

    def _fail_inside_forward(self) -> None:
        with self._forward(7):
            msg = "forward failed"
            raise RuntimeError(msg)

    def test_nested_forward_restores_the_outer_params(self) -> None:
        # Arrange
        clear_ulysses_params()

        # Act
        with self._forward(7) as outer:
            outer_params = dict(ULYSSES_PARAMS)
            with self._forward(200) as inner:
                inner_params = dict(ULYSSES_PARAMS)
            after_inner = dict(ULYSSES_PARAMS)
        after_outer = dict(ULYSSES_PARAMS)

        # Assert
        assert outer_params["cu_seqlens"] is outer.cu_seqlens
        assert inner_params["cu_seqlens"] is inner.cu_seqlens
        assert inner_params["max_seqlen"] != outer_params["max_seqlen"]
        assert inner_params["contexts"] is not outer_params["contexts"]
        assert after_inner == outer_params
        assert after_outer == {}

    def test_exception_clears_its_params_and_restores_the_previous(self) -> None:
        # Arrange
        previous = torch.tensor([0, 4], dtype=torch.int32)
        publish_varlen_offsets(previous)

        # Act
        try:
            with pytest.raises(RuntimeError, match="forward failed"):
                self._fail_inside_forward()
            restored = dict(ULYSSES_PARAMS)
        finally:
            clear_ulysses_params()

        # Assert
        assert restored == {
            "cu_seqlens": previous,
            "host_cu_seqlens": (0, 4),
            "max_seqlen": 4,
            "contexts": {},
        }


class TestPackedContextParallelSequencePrepareForward:
    @staticmethod
    def _layout(rank: int = 0) -> PackedContextParallelSequence:
        return PackedContextParallelSequence(
            pad_token_id=PAD_TOKEN_ID,
            cp=2,
            group=SimpleNamespace(),
            rank=rank,
            round_to=4,
        )

    @staticmethod
    def _batch() -> tuple[torch.Tensor, torch.Tensor]:
        ids = torch.tensor([[1, 2, 3, 0], [4, 5, 6, 0]])
        return ids, (ids != PAD_TOKEN_ID).long()

    def test_shards_the_padded_packed_predictions(self) -> None:
        # Arrange
        ids, mask = self._batch()

        # Act
        with self._layout(rank=1).prepare_forward(ids, mask) as fwd:
            kwargs = fwd.kwargs
            params = dict(ULYSSES_PARAMS)

        # Assert: 5 packed predictions pad to 8, so rank 1 holds columns 4-7.
        assert torch.equal(kwargs["input_ids"], torch.tensor([[5, 0, 0, 0]]))
        assert torch.equal(kwargs["position_ids"], torch.tensor([[1, 0, 0, 0]]))
        assert "attention_mask" not in kwargs
        assert torch.equal(fwd.targets, torch.tensor([[6, 0, 0, 0]]))
        assert params["cu_seqlens"].tolist() == [0, 3, 5, 8]
        assert params["host_cu_seqlens"] == (0, 3, 5, 8)
        assert params["max_seqlen"] == 3

    def test_copies_stack_document_offsets(self) -> None:
        ids, mask = self._batch()

        with self._layout().prepare_forward(ids, mask, copies=2) as fwd:
            assert fwd.kwargs["input_ids"].shape == (2, 4)
            assert fwd.cu_seqlens.tolist() == [0, 3, 5, 8, 11, 13, 16]
            assert fwd.host_cu_seqlens == (0, 3, 5, 8, 11, 13, 16)

    def test_copies_stack_pixel_values(self) -> None:
        ids, mask = self._batch()
        pixel_values = torch.arange(6.0).reshape(2, 3)

        with self._layout().prepare_forward(
            ids, mask, pixel_values=pixel_values, copies=2
        ) as fwd:
            repeated = fwd.kwargs["pixel_values"]

        assert torch.equal(repeated, torch.cat([pixel_values, pixel_values]))

    @pytest.mark.parametrize(("rank", "n_scored"), [(0, 3), (1, 1)])
    def test_each_rank_counts_its_scored_predictions(
        self, rank: int, n_scored: int
    ) -> None:
        # Arrange
        ids, mask = self._batch()
        score_mask = torch.ones(2, 3, dtype=torch.bool)

        # Act
        with self._layout(rank).prepare_forward(
            ids, mask, score_mask, host_masks=HostMasks.read(mask, score_mask)
        ) as fwd:
            sharded_score_mask = fwd.score_mask

        # Assert: the prediction across the document boundary is not scored.
        assert fwd.n_scored == n_scored
        assert int(sharded_score_mask.sum()) == n_scored

    def test_labels_pack_then_shard(self) -> None:
        # Arrange
        ids, mask = self._batch()
        labels = torch.tensor([[10, 11, 12], [13, 14, 15]])

        # Act
        with self._layout(rank=1).prepare_forward(ids, mask) as fwd:
            sharded_labels = fwd.labels(labels, -100)

        # Assert: rank 1 holds packed prediction 4, then the ignored pad tail.
        with _packed().prepare_forward(ids, mask) as reference:
            packed_labels = reference.labels(labels, -100)
        assert torch.equal(
            sharded_labels, torch.tensor([[packed_labels[0, 4], -100, -100, -100]])
        )

    @pytest.mark.parametrize("kind", ["logprobs", "values"])
    def test_restore_matches_the_single_rank_packed_layout(self, kind: str) -> None:
        # Arrange
        ids, mask = self._batch()
        shard = torch.arange(4, dtype=torch.float32).reshape(1, 4)

        def fake_gather(data: torch.Tensor, _cp: int, _group: object) -> torch.Tensor:
            return torch.cat([data, data + 10], dim=1)

        gathered = fake_gather(shard, 2, None)[:, :5]
        packed_outputs = gathered if kind == "logprobs" else F.pad(gathered, (0, 1))

        # Act
        with (
            patch(
                "agilerl.algorithms.core.sequence_layout.gather_for_cp_wo_grad",
                fake_gather,
            ),
            torch.no_grad(),
            self._layout().prepare_forward(ids, mask) as fwd,
        ):
            restored = fwd.restore(shard, kind)

        # Assert
        with _packed().prepare_forward(ids, mask) as reference:
            expected = reference.restore(packed_outputs, kind)
        assert torch.equal(restored, expected)

    def test_restore_squeezes_a_trailing_value_dim(self) -> None:
        # Arrange
        ids, mask = self._batch()
        shard = torch.arange(4, dtype=torch.float32).reshape(1, 4)

        def fake_gather(data: torch.Tensor, _cp: int, _group: object) -> torch.Tensor:
            return torch.cat([data, data + 10], dim=1)

        # Act
        with (
            patch(
                "agilerl.algorithms.core.sequence_layout.gather_for_cp_wo_grad",
                fake_gather,
            ),
            torch.no_grad(),
            self._layout().prepare_forward(ids, mask) as fwd,
        ):
            from_head = fwd.restore(shard.unsqueeze(-1), "values")
            flat = fwd.restore(shard, "values")

        # Assert
        assert torch.equal(from_head, flat)


class TestHostMasksPackedScored:
    def test_marks_the_packed_predictions_pack_padded_batch_scores(self) -> None:
        mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
        score_mask = torch.tensor([[True, False, True], [True, True, True]])

        scored = HostMasks.read(mask, score_mask).packed_scored()

        assert scored.tolist() == [True, False, False, True]

    def test_without_a_score_mask_returns_none(self) -> None:
        assert HostMasks.read(torch.ones(1, 3)).packed_scored() is None


class TestSequenceLayoutMatchesPaddedForward:
    def test_trimmed_no_grad_log_probs_match_the_untrimmed_forward(self) -> None:
        # Arrange: rows of 6 and 7 real tokens in 8 columns.
        agent = make_grpo(qwen3_model(), False, None)
        ids = torch.cat(experiences()[0][1:3])
        mask = (ids != QWEN3_PAD_TOKEN_ID).long()
        score_mask = torch.cat(experiences()[1][1:3])

        # Act
        with torch.no_grad():
            trimmed = agent._get_logprobs(
                ids, 2, attention_mask=mask, score_mask=score_mask
            )
        untrimmed = agent._get_logprobs(
            ids, 2, attention_mask=mask, score_mask=score_mask
        ).detach()

        # Assert: fp32 differs only in the attention reduction width.
        assert trimmed.shape == untrimmed.shape == (2, 7)
        torch.testing.assert_close(trimmed, untrimmed, atol=1e-6, rtol=0)

    @pytest.mark.parametrize("attn_implementation", ["eager", "sdpa"])
    def test_unmasked_no_grad_log_probs_match_the_masked_forward(
        self, attn_implementation: str
    ) -> None:
        # Arrange: right-padded rows of 8, 6, 7 and 8 real tokens, every position scored.
        model = qwen3_model()
        model.config._attn_implementation = attn_implementation
        agent = make_grpo(model, False, None)
        ids = torch.cat(experiences()[0])
        mask = (ids != QWEN3_PAD_TOKEN_ID).long()
        real_hidden = mask[:, :-1].bool()

        # Act
        with torch.no_grad():
            unmasked = agent._get_logprobs(ids, 4, attention_mask=mask)
        masked = agent._get_logprobs(ids, 4, attention_mask=mask).detach()

        # Assert: fp32 differs only in the attention reduction width.
        torch.testing.assert_close(
            unmasked[real_hidden], masked[real_hidden], atol=1e-6, rtol=0
        )
        assert not unmasked[~real_hidden].any()

    @pytest.mark.multi_rank
    def test_packed_and_context_parallel_log_probs_match_the_padded_forward(
        self,
    ) -> None:
        # Arrange
        action_mask = torch.cat(experiences()[1])

        # Act
        (one_rank,) = run_ranks(_one_rank_layouts_worker, 1, backend=None, cpu=True)
        cp2 = run_ranks(_cp2_layout_worker, 2, backend=None, cpu=True)

        # Assert: fp32 differs only in attention and gather reduction order.
        reference = torch.from_numpy(one_rank["padded"])[action_mask]
        scored = {
            "packed": one_rank["packed"],
            "cp1": one_rank["cp1"],
            **{
                f"cp2 {layout} rank {rank}": scored_rank[layout]
                for rank, scored_rank in enumerate(cp2)
                for layout in ("packed", "padded")
            },
        }
        for name, log_probs in scored.items():
            torch.testing.assert_close(
                torch.from_numpy(log_probs)[action_mask],
                reference,
                atol=1e-6,
                rtol=0,
                msg=name,
            )


@_LLM_DEPS_SKIP
class TestLLMAlgorithmCopyCloneAttributes:
    def test_keeps_the_clone_layout(self) -> None:
        agent = _make_llm_agent(clone=True)
        clone = _make_llm_agent(clone=True)
        clone_layout = _packed()
        clone.sequence_layout = clone_layout
        agent.sequence_layout = _padded()
        agent.colocated = False
        clone.colocated = False

        out = LLMAlgorithm._copy_clone_attributes(agent, clone)

        assert out.sequence_layout is clone_layout


class TestLLMAlgorithmContextParallelRoundTo:
    @staticmethod
    def _agent(model: torch.nn.Module) -> SimpleNamespace:
        return SimpleNamespace(fsdp_config=FSDPConfig(cp=2), actor=model)

    def test_attention_only_family_rounds_to_cp(self) -> None:
        agent = self._agent(qwen3_model())

        assert LLMAlgorithm._context_parallel_round_to(agent) == 2

    def test_flex_attention_rounds_to_whole_blocks(self) -> None:
        model = qwen3_model()
        model.config._attn_implementation = "flex_attention"

        assert LLMAlgorithm._context_parallel_round_to(self._agent(model)) == 128

    def test_gated_deltanet_rounds_to_scan_chunks_per_rank(self) -> None:
        # Arrange
        model = qwen3_model()
        mixer = torch.nn.Identity()
        object.__setattr__(mixer, GDN_CP_FLAG, True)
        model.add_module("gdn", mixer)

        # Act
        round_to = LLMAlgorithm._context_parallel_round_to(self._agent(model))

        # Assert
        assert round_to == 2 * CP_SCAN_CHUNK


class TestContextParallelShardLen:
    """The memory estimate prices the shard width the trainer's CP layout pads to."""

    @pytest.mark.parametrize("cp", [2, 4])
    @pytest.mark.parametrize("gated_delta", [False, True])
    @pytest.mark.parametrize("flex", [False, True])
    def test_matches_the_layout_shard_width(
        self, cp: int, gated_delta: bool, flex: bool
    ) -> None:
        # Arrange
        seq_len = 1000
        model = qwen3_model()
        if flex:
            model.config._attn_implementation = "flex_attention"
        if gated_delta:
            mixer = torch.nn.Identity()
            object.__setattr__(mixer, GDN_CP_FLAG, True)
            model.add_module("gdn", mixer)
        round_to = LLMAlgorithm._context_parallel_round_to(
            SimpleNamespace(fsdp_config=FSDPConfig(cp=cp), actor=model)
        )
        layout = ContextParallelSequence(
            PAD_TOKEN_ID, cp, SimpleNamespace(), 0, round_to
        )
        ids = torch.ones(1, seq_len, dtype=torch.long)
        arch = ModelArch.from_hf_config(
            SUPPORTED_MODEL_INFO["Qwen/Qwen3-1.7B"].config
        ).model_copy(
            update={
                "flex_attention": flex,
                "mamba_gated_delta": gated_delta,
                "n_mamba_layers": int(gated_delta),
            }
        )

        # Act
        with layout.prepare_forward(ids, torch.ones_like(ids)) as fwd:
            trainer_width = fwd.kwargs["input_ids"].shape[1]
        estimate_width = formulas.context_parallel_shard_len(arch, seq_len, cp)

        # Assert
        assert estimate_width == trainer_width

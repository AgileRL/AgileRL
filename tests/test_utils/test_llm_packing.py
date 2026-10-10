# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for padding-free sequence packing helpers.

These exercise the pure-tensor pack/unpack logic and the backend gate; they do
not require vLLM/DeepSpeed/FlashAttention, so they run everywhere (including
CPU/darwin CI). The guarantee that no token attends across packed-sequence
boundaries is a property of the FlashAttention-varlen backend and is
validated on GPU.
"""

import warnings

import pytest
import torch
from torch import nn

from agilerl import HAS_LLM_DEPENDENCIES
from agilerl.utils.llm_packing import (
    RESETS_AT_DOCUMENT_BOUNDARY,
    PackedBatch,
    mixers_without_boundary_reset,
    pack_padded_batch,
    packed_layout,
    packed_seq_idx,
    unpack_hidden_states,
    unpack_logprobs,
    unpack_values,
)

if HAS_LLM_DEPENDENCIES:
    from transformers import (
        AutoModelForCausalLM,
        CLIPVisionConfig,
        LlamaConfig,
        LlavaConfig,
        LlavaForConditionalGeneration,
        Mamba2Config,
    )


def _make_batch(lengths, pad=0, vocab=64, seed=0):
    torch.manual_seed(seed)
    T = max(lengths)
    ids = torch.full((len(lengths), T), pad, dtype=torch.long)
    for b, length in enumerate(lengths):
        ids[b, :length] = torch.randint(1, vocab, (length,))
    mask = ids != pad
    return ids, mask


class TestPackPaddedBatch:
    def test_pack_structure(self):
        lengths = [6, 3, 4]
        ids, mask = _make_batch(lengths)
        packed = pack_padded_batch(ids, mask)
        n = sum(lengths)
        assert isinstance(packed, PackedBatch)
        assert packed.input_ids.shape == (1, n)
        assert packed.seq_lengths.tolist() == lengths
        assert packed.lengths == tuple(lengths)
        assert packed.cu_seqlens.tolist() == [0, 6, 9, 13]
        assert packed.cu_seqlens.dtype == torch.int32
        assert packed.max_seqlen == 6
        assert packed.batch_size == 3
        assert packed.padded_seq_len == 6

    def test_position_ids_reset_per_sequence(self):
        lengths = [6, 3, 4]
        ids, mask = _make_batch(lengths)
        packed = pack_padded_batch(ids, mask)
        expected = torch.cat([torch.arange(length) for length in lengths])
        assert torch.equal(packed.position_ids.squeeze(0), expected)

    def test_flat_ids_concatenate_real_tokens(self):
        lengths = [6, 3, 4]
        ids, mask = _make_batch(lengths)
        packed = pack_padded_batch(ids, mask)
        expected = torch.cat([ids[b, :length] for b, length in enumerate(lengths)])
        assert torch.equal(packed.input_ids.squeeze(0), expected)

    def test_robust_to_left_padding(self):
        # Interior/left padding: selection is by mask, not by prefix assumption.
        ids = torch.tensor([[0, 0, 7, 8, 9], [5, 6, 0, 0, 0]])
        mask = ids != 0
        packed = pack_padded_batch(ids, mask)
        assert packed.input_ids.squeeze(0).tolist() == [7, 8, 9, 5, 6]
        assert packed.position_ids.squeeze(0).tolist() == [0, 1, 2, 0, 1]
        assert packed.seq_lengths.tolist() == [3, 2]

    def test_rejects_non_2d_input_ids(self):
        ids, mask = _make_batch([3, 2])
        with pytest.raises(ValueError, match=r"expects \(B, T\) input_ids"):
            pack_padded_batch(ids[0], mask[0])  # (T,) — missing batch dim
        with pytest.raises(ValueError, match=r"expects \(B, T\) input_ids"):
            pack_padded_batch(ids.unsqueeze(0), mask.unsqueeze(0))  # (1, B, T)


class TestUnpackLogprobs:
    def test_maps_to_dense_frame_and_drops_boundary(self):
        lengths = [6, 3, 4]
        ids, mask = _make_batch(lengths)
        packed = pack_padded_batch(ids, mask)
        n = sum(lengths)
        packed_lp = torch.arange(n - 1, dtype=torch.float32)
        out = unpack_logprobs(packed_lp, packed)

        T = max(lengths)
        assert out.shape == (3, T - 1)
        cu = packed.cu_seqlens.tolist()
        expected = torch.zeros(3, T - 1)
        for b, length in enumerate(lengths):
            if length > 1:
                expected[b, : length - 1] = packed_lp[cu[b] : cu[b] + length - 1]
        assert torch.equal(out, expected)
        # The cross-segment boundary value (last token of seq 0 -> first of seq 1)
        # is never read into any row.
        assert float(out[1, 0]) == packed_lp[cu[1]].item()

    def test_single_token_sequence_contributes_nothing(self):
        lengths = [1, 3]
        ids, mask = _make_batch(lengths)
        packed = pack_padded_batch(ids, mask)
        packed_lp = torch.arange(sum(lengths) - 1, dtype=torch.float32)
        out = unpack_logprobs(packed_lp, packed)
        assert torch.equal(out[0], torch.zeros(max(lengths) - 1))

    def test_autograd_flows(self):
        lengths = [4, 5]
        ids, mask = _make_batch(lengths)
        packed = pack_padded_batch(ids, mask)
        packed_lp = torch.randn(sum(lengths) - 1, requires_grad=True)
        unpack_logprobs(packed_lp, packed).sum().backward()
        assert packed_lp.grad is not None
        assert packed_lp.grad.abs().sum() > 0

    def test_end_to_end_equivalence_with_dense_fused_logprobs(self):
        """Packed -> unpacked logprobs equal the dense path at valid positions.

        Uses context-independent hidden states (supplied directly), so the only
        thing under test is pack/unpack composed with the production
        fused-linear-logprob kernel — independent of the attention backend.
        """
        from agilerl.algorithms.core.base import LLMAlgorithm

        lengths = [6, 3, 4]
        T = max(lengths)
        B = len(lengths)
        H, V = 8, 64
        torch.manual_seed(1)
        ids = torch.randint(1, V, (B, T), dtype=torch.long)
        for b, length in enumerate(lengths):
            ids[b, length:] = 0
        mask = ids != 0
        hidden = torch.randn(B, T, H)
        weight = torch.randn(V, H)
        bias = torch.randn(V)

        fused = LLMAlgorithm._logprobs_from_hidden_fused
        dense_lp = fused(
            hidden[:, :-1],
            weight,
            bias,
            ids[:, 1:],
            temperature=1.0,
            cast_to_fp32=True,
        )

        packed = pack_padded_batch(ids, mask)
        packed_hidden = hidden[mask.bool()].unsqueeze(0)
        packed_lp = fused(
            packed_hidden[:, :-1],
            weight,
            bias,
            packed.input_ids[:, 1:],
            temperature=1.0,
            cast_to_fp32=True,
        )
        unpacked = unpack_logprobs(packed_lp.reshape(-1), packed)

        for b, length in enumerate(lengths):
            if length > 1:
                assert torch.allclose(
                    unpacked[b, : length - 1],
                    dense_lp[b, : length - 1],
                    atol=1e-5,
                )


class TestUnpackValues:
    def test_maps_per_token_values_no_boundary_drop_clamped_to_frame(self):
        # b=0 has length 6 == T, so it must clamp to T-1=5 values (dropping the
        # last position, mirroring the padded value[:, :-1]); the others keep all.
        lengths = [6, 3, 4]
        ids, mask = _make_batch(lengths)
        packed = pack_padded_batch(ids, mask)
        n = sum(lengths)
        packed_v = torch.arange(n, dtype=torch.float32)
        out = unpack_values(packed_v, packed)

        T = max(lengths)
        assert out.shape == (3, T - 1)
        cu = packed.cu_seqlens.tolist()
        expected = torch.zeros(3, T - 1)
        for b, length in enumerate(lengths):
            n_keep = min(length, T - 1)  # clamp to the (B, T-1) frame width
            expected[b, :n_keep] = packed_v[cu[b] : cu[b] + n_keep]
        assert torch.equal(out, expected)
        # Pad columns stay exactly zero (masked out downstream).
        for b, length in enumerate(lengths):
            n_keep = min(length, T - 1)
            assert torch.equal(out[b, n_keep:], torch.zeros(T - 1 - n_keep))

    def test_accepts_1d_2d_and_column_shapes(self):
        lengths = [4, 2]
        ids, mask = _make_batch(lengths)
        packed = pack_padded_batch(ids, mask)
        v_1d = torch.randn(sum(lengths))
        out_1d = unpack_values(v_1d, packed)
        assert torch.equal(out_1d, unpack_values(v_1d.unsqueeze(0), packed))  # (1, N)
        assert torch.equal(out_1d, unpack_values(v_1d.unsqueeze(1), packed))  # (N, 1)

    def test_single_token_sequence_kept(self):
        # A length-1 sequence has one value and (unlike logprobs) no boundary to
        # drop, so its single value is written to column 0.
        lengths = [1, 3]
        ids, mask = _make_batch(lengths)
        packed = pack_padded_batch(ids, mask)
        packed_v = torch.arange(sum(lengths), dtype=torch.float32)
        out = unpack_values(packed_v, packed)
        assert float(out[0, 0]) == packed_v[0].item()
        assert torch.equal(out[0, 1:], torch.zeros(max(lengths) - 2))

    def test_autograd_flows(self):
        lengths = [4, 5]
        ids, mask = _make_batch(lengths)
        packed = pack_padded_batch(ids, mask)
        v = torch.randn(sum(lengths), requires_grad=True)
        unpack_values(v, packed).sum().backward()
        assert v.grad is not None
        assert v.grad.abs().sum() > 0


class TestUnpackHiddenStates:
    def test_maps_real_tokens_to_dense_frame_and_zeros_pads(self):
        lengths = [6, 3, 4]
        ids, mask = _make_batch(lengths)
        packed = pack_padded_batch(ids, mask)
        n = sum(lengths)
        hidden_dim = 5
        # Distinct value per (flat token, feature) so any mis-scatter is caught.
        packed_hidden = torch.arange(n * hidden_dim, dtype=torch.float32).reshape(
            1, n, hidden_dim
        )
        out = unpack_hidden_states(packed_hidden, packed)

        t = max(lengths)
        assert out.shape == (3, t, hidden_dim)
        cu = packed.cu_seqlens.tolist()
        flat = packed_hidden.reshape(n, hidden_dim)
        expected = torch.zeros(3, t, hidden_dim)
        for b, length in enumerate(lengths):
            expected[b, :length] = flat[cu[b] : cu[b] + length]
        assert torch.equal(out, expected)
        # Pad rows stay exactly zero (masked out downstream).
        for b, length in enumerate(lengths):
            assert torch.equal(out[b, length:], torch.zeros(t - length, hidden_dim))

    def test_accepts_2d_or_3d_packed_hidden(self):
        lengths = [3, 2]
        ids, mask = _make_batch(lengths)
        packed = pack_padded_batch(ids, mask)
        n = sum(lengths)
        hidden_dim = 4
        hidden_3d = torch.randn(1, n, hidden_dim)
        hidden_2d = hidden_3d.reshape(n, hidden_dim)
        assert torch.equal(
            unpack_hidden_states(hidden_3d, packed),
            unpack_hidden_states(hidden_2d, packed),
        )

    def test_autograd_flows_to_every_real_token(self):
        lengths = [4, 5]
        ids, mask = _make_batch(lengths)
        packed = pack_padded_batch(ids, mask)
        n = sum(lengths)
        hidden_dim = 3
        packed_hidden = torch.randn(1, n, hidden_dim, requires_grad=True)
        unpack_hidden_states(packed_hidden, packed).sum().backward()
        # Every real token's hidden state is written to the frame exactly once
        # (no boundary drop), so each receives unit gradient.
        assert packed_hidden.grad is not None
        assert torch.equal(packed_hidden.grad, torch.ones_like(packed_hidden))

    def test_single_token_sequence_kept(self):
        # Unlike logprob unpacking (which drops length-1 sequences entirely),
        # a length-1 sequence still has a hidden state to place.
        lengths = [1, 3]
        ids, mask = _make_batch(lengths)
        packed = pack_padded_batch(ids, mask)
        n = sum(lengths)
        hidden_dim = 2
        packed_hidden = torch.arange(n * hidden_dim, dtype=torch.float32).reshape(
            1, n, hidden_dim
        )
        out = unpack_hidden_states(packed_hidden, packed)
        # Row 0's single token sits at position 0; the rest of the row is zero.
        assert torch.equal(out[0, 0], packed_hidden.reshape(n, hidden_dim)[0])
        assert torch.equal(out[0, 1:], torch.zeros(max(lengths) - 1, hidden_dim))

    def test_end_to_end_equivalence_with_dense_hidden(self):
        """Packed -> unpacked hidden equals the dense hidden at real positions.

        Uses a context-independent feature map (``hidden = embed(id)``) so the
        only thing under test is pack/unpack: with no cross-sequence attention
        the packed forward must reproduce the dense hidden at every real token,
        which is exactly what a fused loss kernel (Liger) needs to see.
        """
        lengths = [6, 3, 4]
        t = max(lengths)
        b_size = len(lengths)
        hidden_dim, vocab = 8, 64
        torch.manual_seed(2)
        ids = torch.randint(1, vocab, (b_size, t), dtype=torch.long)
        for b, length in enumerate(lengths):
            ids[b, length:] = 0
        mask = ids != 0

        embed = torch.randn(vocab, hidden_dim)
        dense_hidden = embed[ids]  # (B, T, H)

        packed = pack_padded_batch(ids, mask)
        packed_hidden = embed[packed.input_ids]  # (1, N, H)
        unpacked = unpack_hidden_states(packed_hidden, packed)

        for b, length in enumerate(lengths):
            assert torch.allclose(unpacked[b, :length], dense_hidden[b, :length])


def _tiny_llama_config():
    return LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
    )


def _tiny_mamba2_config():
    return Mamba2Config(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_heads=4,
        head_dim=8,
        n_groups=1,
        state_size=8,
        conv_kernel=4,
        chunk_size=4,
    )


class ScanMixer(nn.Module):
    """Mixer owning a state-decay parameter."""

    def __init__(self) -> None:
        super().__init__()
        self.A_log = nn.Parameter(torch.zeros(2))


class ResettingScanMixer(ScanMixer):
    """Scan mixer whose class declares a reset at document boundaries."""


setattr(ResettingScanMixer, RESETS_AT_DOCUMENT_BOUNDARY, True)


class CausalConvMixer(nn.Module):
    """Mixer with a depthwise causal short convolution."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1d = nn.Conv1d(4, 4, kernel_size=3, groups=4, padding=2)


class SymmetricConvMixer(nn.Module):
    """Mixer with a same-padded convolution, which looks both ways."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1d = nn.Conv1d(4, 4, kernel_size=3, groups=4, padding=1)


class TestPackedSeqIdx:
    def test_numbers_documents_where_positions_restart(self):
        # Arrange
        positions = torch.tensor([[0, 1, 2, 0, 1, 0]])

        # Act
        seq_idx = packed_seq_idx(positions)

        # Assert
        assert seq_idx.dtype == torch.int32
        assert seq_idx.tolist() == [[0, 0, 0, 1, 1, 2]]

    def test_numbers_each_row_on_its_own(self):
        positions = torch.tensor([[0, 1, 0, 1], [0, 1, 2, 3]])

        seq_idx = packed_seq_idx(positions)

        assert seq_idx.tolist() == [[0, 0, 1, 1], [0, 0, 0, 0]]

    @pytest.mark.parametrize("positions", [[[0, 1, 2, 3]], [[5, 6, 7]]])
    def test_rows_holding_one_document_return_none(self, positions):
        assert packed_seq_idx(torch.tensor(positions)) is None


class TestPackedLayout:
    def test_offsets_and_longest_document_of_a_packed_row(self):
        # Arrange
        positions = torch.tensor([[0, 1, 2, 0, 1, 0]])

        # Act
        layout = packed_layout(positions)

        # Assert
        assert layout.seq_idx.tolist() == [[0, 0, 0, 1, 1, 2]]
        assert layout.cu_seqlens.dtype == torch.int32
        assert layout.cu_seqlens.tolist() == [0, 3, 5, 6]
        assert layout.max_seqlen == 3

    def test_offsets_span_every_row(self):
        positions = torch.tensor([[0, 1, 0, 1], [0, 1, 2, 3]])

        layout = packed_layout(positions)

        assert layout.cu_seqlens.tolist() == [0, 2, 4, 8]
        assert layout.max_seqlen == 4

    def test_rows_holding_one_document_return_none(self):
        assert packed_layout(torch.tensor([[0, 1, 2], [4, 5, 6]])) is None

    def test_repeat_call_on_the_same_tensor_returns_the_memoized_layout(self):
        # Arrange
        positions = torch.tensor([[0, 1, 0]])
        first = packed_layout(positions)

        # Act
        second = packed_layout(positions)

        # Assert
        assert second is first

    def test_a_new_tensor_gets_its_own_layout(self):
        # Arrange
        packed_layout(torch.tensor([[0, 1, 0]]))

        # Act
        layout = packed_layout(torch.tensor([[0, 0, 1]]))

        # Assert
        assert layout.cu_seqlens.tolist() == [0, 1, 3]


class TestMixersWithoutBoundaryReset:
    @pytest.mark.skipif(
        not HAS_LLM_DEPENDENCIES, reason="LLM dependencies not installed"
    )
    def test_lists_mamba2_mixers(self):
        model = AutoModelForCausalLM.from_config(_tiny_mamba2_config())

        assert mixers_without_boundary_reset(model) == ["Mamba2Mixer"]

    @pytest.mark.skipif(
        not HAS_LLM_DEPENDENCIES, reason="LLM dependencies not installed"
    )
    def test_attention_only_model_has_none(self):
        model = AutoModelForCausalLM.from_config(_tiny_llama_config())

        assert mixers_without_boundary_reset(model) == []

    def test_lists_scan_and_causal_conv_mixers_by_class(self):
        model = nn.Sequential(ScanMixer(), ScanMixer(), CausalConvMixer())

        assert mixers_without_boundary_reset(model) == ["CausalConvMixer", "ScanMixer"]

    def test_skips_mixers_marked_as_resetting(self):
        model = nn.Sequential(ResettingScanMixer())

        assert mixers_without_boundary_reset(model) == []

    def test_skips_convolutions_that_are_not_causal(self):
        model = nn.Sequential(SymmetricConvMixer())

        assert mixers_without_boundary_reset(model) == []


class _PackingGateStub:
    """Minimal stand-in to exercise LLMAlgorithm's packing gate in isolation."""

    from agilerl.algorithms.core.base import LLMAlgorithm

    _packing_mode = LLMAlgorithm._packing_mode

    def __init__(self, use_sequence_packing, attn_impl):
        self.use_sequence_packing = use_sequence_packing
        self.actor = AutoModelForCausalLM.from_config(
            _tiny_llama_config(), attn_implementation="eager"
        )
        self.actor.config._attn_implementation = attn_impl


@pytest.mark.skipif(not HAS_LLM_DEPENDENCIES, reason="LLM dependencies not installed")
class TestSequencePackingGate:
    def test_follows_language_model_config_not_top_level(self):
        # Arrange
        stub = _PackingGateStub(True, "sdpa")
        stub.actor = LlavaForConditionalGeneration._from_config(
            LlavaConfig(
                text_config=_tiny_llama_config(),
                vision_config=CLIPVisionConfig(
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    image_size=8,
                    patch_size=4,
                    projection_dim=16,
                ),
                image_token_index=31,
            ),
            attn_implementation="sdpa",
        )
        stub.actor.config.text_config._attn_implementation = "flash_attention_2"

        # Act
        mode = stub._packing_mode()

        # Assert
        assert stub.actor.config._attn_implementation == "sdpa"
        assert mode == "varlen"

    def test_disabled_when_flag_off(self):
        stub = _PackingGateStub(False, "flash_attention_2")
        assert stub._packing_mode() is None

    def test_fa2_uses_varlen(self):
        stub = _PackingGateStub(True, "flash_attention_2")
        assert stub._packing_mode() == "varlen"

    def test_flex_uses_blockmask(self):
        stub = _PackingGateStub(True, "flex_attention")
        assert stub._packing_mode() == "blockmask"

    @pytest.mark.parametrize("impl", ["flash_attention_2", "flex_attention"])
    def test_refuses_recurrent_mixers_that_do_not_reset(self, impl):
        # Arrange
        stub = _PackingGateStub(True, impl)
        stub.actor = AutoModelForCausalLM.from_config(_tiny_mamba2_config())
        stub.actor.config._attn_implementation = impl

        # Act / Assert
        with pytest.raises(
            ValueError, match=r"Mamba2Mixer.*use_sequence_packing=False"
        ):
            stub._packing_mode()

    def test_a_replaced_actor_is_checked_again(self):
        # Arrange
        stub = _PackingGateStub(True, "flash_attention_2")
        assert stub._packing_mode() == "varlen"
        stub.actor = AutoModelForCausalLM.from_config(_tiny_mamba2_config())
        stub.actor.config._attn_implementation = "flash_attention_2"

        # Act / Assert
        with pytest.raises(ValueError, match="Mamba2Mixer"):
            stub._packing_mode()

    @pytest.mark.parametrize("impl", ["sdpa", "eager", "something_weird"])
    def test_unsupported_backends_disable_and_warn_once(self, impl):
        # Dense backends have no sparse path -> packing unsupported, warn once.
        stub = _PackingGateStub(True, impl)
        with pytest.warns(UserWarning, match="varlen/block-sparse"):
            assert stub._packing_mode() is None
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert stub._packing_mode() is None


class TestPackedSegmentIds:
    def test_packed_segment_ids(self):
        from agilerl.utils.llm_packing import packed_segment_ids

        lengths = [3, 2]
        ids, mask = _make_batch(lengths)
        packed = pack_padded_batch(ids, mask)
        # Seq 0 occupies flat positions [0,1,2], seq 1 occupies [3,4].
        assert packed_segment_ids(packed, "cpu").tolist() == [0, 0, 0, 1, 1]

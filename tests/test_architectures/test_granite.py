# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Granite patches: packed rows through a tiny GraniteMoeHybrid on CPU match the padded batch."""

from __future__ import annotations

import itertools
import logging
from collections.abc import Iterator

import pytest
import torch
import torch.nn.functional as F
from transformers import AttentionInterface
from transformers.masking_utils import AttentionMaskInterface, flash_attention_mask
from transformers.models.granitemoehybrid.configuration_granitemoehybrid import (
    GraniteMoeHybridConfig,
)
from transformers.models.granitemoehybrid.modeling_granitemoehybrid import (
    GraniteMoeHybridForCausalLM,
    GraniteMoeHybridMambaLayer,
    GraniteMoeHybridModel,
)

from agilerl.architectures import granite, install_family_patches
from agilerl.architectures.granite import patch_granite_hybrid_packed_sequences
from agilerl.utils.llm_packing import (
    RESETS_AT_DOCUMENT_BOUNDARY,
    mixers_without_boundary_reset,
    pack_padded_batch,
    unpack_logprobs,
)

VOCAB = 64
CHUNK_SIZE = 4
# Documents longer than two chunks, exactly one chunk, a single token, and
# lengths that straddle chunk edges once packed.
LENGTHS = (7, CHUNK_SIZE, 9, 1, 5)
VARLEN_ATTENTION = "agilerl_test_granite_varlen"
# Captured at collection, before any test installs the family patches.
PRISTINE_FORWARDS = tuple(
    (cls, cls.__dict__["forward"])
    for cls in (GraniteMoeHybridMambaLayer, GraniteMoeHybridModel)
)


def _reset_granite_hybrid_classes() -> None:
    for cls, forward in PRISTINE_FORWARDS:
        cls.forward = forward
    if RESETS_AT_DOCUMENT_BOUNDARY in vars(GraniteMoeHybridMambaLayer):
        delattr(GraniteMoeHybridMambaLayer, RESETS_AT_DOCUMENT_BOUNDARY)


@pytest.fixture
def pristine_granite_hybrid_classes() -> Iterator[None]:
    """Run with the unpatched classes, and undo the packed patches afterwards."""
    _reset_granite_hybrid_classes()
    yield
    _reset_granite_hybrid_classes()


@pytest.fixture
def varlen_attention() -> list[torch.Tensor | None]:
    """Register a CPU stand-in for FlashAttention, causal within each ``cu_seq_lens_q`` segment or else each row."""
    offsets_seen: list[torch.Tensor | None] = []

    def attend(module, query, key, value, attention_mask, scaling=None, **kwargs):
        cu_seq_lens_q = kwargs.get("cu_seq_lens_q")
        offsets_seen.append(cu_seq_lens_q)
        rows, heads, length, head_dim = query.shape
        query, key, value = (
            tensor.transpose(1, 2)
            .reshape(1, rows * length, -1, head_dim)
            .transpose(1, 2)
            for tensor in (query, key, value)
        )
        offsets = (
            list(range(0, rows * length + 1, length))
            if cu_seq_lens_q is None
            else cu_seq_lens_q.tolist()
        )
        segments = [
            F.scaled_dot_product_attention(
                query[:, :, start:end],
                key[:, :, start:end],
                value[:, :, start:end],
                is_causal=True,
                scale=scaling,
                enable_gqa=True,
            )
            for start, end in itertools.pairwise(offsets)
        ]
        out = torch.cat(segments, dim=2).transpose(1, 2)
        return out.reshape(rows, length, heads, head_dim).contiguous(), None

    AttentionInterface.register(VARLEN_ATTENTION, attend)
    AttentionMaskInterface.register(VARLEN_ATTENTION, flash_attention_mask)
    return offsets_seen


def _tiny_granite_hybrid(
    attn_implementation: str = "sdpa",
) -> GraniteMoeHybridForCausalLM:
    config = GraniteMoeHybridConfig(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=16,
        shared_intermediate_size=32,
        num_hidden_layers=4,
        layer_types=[
            "linear_attention",
            "full_attention",
            "linear_attention",
            "full_attention",
        ],
        num_attention_heads=2,
        num_key_value_heads=1,
        num_local_experts=4,
        num_experts_per_tok=2,
        position_embedding_type="rope",
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        mamba_n_heads=4,
        mamba_n_groups=1,
        mamba_d_state=8,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_chunk_size=CHUNK_SIZE,
        max_position_embeddings=64,
        use_cache=False,
    )
    config._attn_implementation = attn_implementation
    torch.manual_seed(0)
    model = GraniteMoeHybridForCausalLM(config).float()
    # Stock init leaves the router and experts at zero; random weights make
    # every MoE and Mamba path contribute.
    with torch.no_grad():
        for param in model.parameters():
            if param.abs().sum() == 0:
                param.normal_(std=0.1)
    return model


def _right_padded_documents() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(1)
    ids = torch.zeros(len(LENGTHS), max(LENGTHS), dtype=torch.long)
    for row, length in enumerate(LENGTHS):
        ids[row, :length] = torch.randint(1, VOCAB, (length,))
    return ids, ids != 0


def _token_logprobs(model: torch.nn.Module, ids: torch.Tensor, **kwargs):
    logits = model(input_ids=ids, use_cache=False, **kwargs).logits[:, :-1].float()
    return torch.log_softmax(logits, -1).gather(-1, ids[:, 1:, None]).squeeze(-1)


def _padded_and_packed_logprobs(model: GraniteMoeHybridForCausalLM):
    """Per-token log-probs ``(B, T-1)`` of the padded and the packed forward."""
    ids, mask = _right_padded_documents()
    padded = _token_logprobs(model, ids, attention_mask=mask.long())
    packed_batch = pack_padded_batch(ids, mask)
    packed_row = _token_logprobs(
        model, packed_batch.input_ids, position_ids=packed_batch.position_ids
    )
    return padded, unpack_logprobs(packed_row, packed_batch), mask[:, 1:]


class TestPatchGraniteHybridPackedSequences:
    """Packed rows match the padded batch: Mamba2 state and attention stop at every document boundary."""

    # fp32 on CPU; packed and padded run the same ops on different row
    # layouts, so only reduction order differs.
    ATOL = 1e-5

    def test_packed_logprobs_match_padded_logprobs(
        self, pristine_granite_hybrid_classes
    ) -> None:
        # Arrange
        model = _tiny_granite_hybrid().eval()
        install_family_patches(model.config.model_type, model)

        # Act
        with torch.no_grad():
            padded, packed, real = _padded_and_packed_logprobs(model)

        # Assert
        torch.testing.assert_close(packed[real], padded[real], atol=self.ATOL, rtol=0)

    def test_packed_gradients_match_padded_gradients(
        self, pristine_granite_hybrid_classes
    ) -> None:
        # Arrange
        model = _tiny_granite_hybrid().train()
        install_family_patches(model.config.model_type, model)
        params = {
            "mamba_in_proj": model.model.layers[2].mamba.in_proj.weight,
            "mamba_A_log": model.model.layers[0].mamba.A_log,
            "attention_q_proj": model.model.layers[1].self_attn.q_proj.weight,
        }

        # Act
        padded, packed, real = _padded_and_packed_logprobs(model)
        padded_grads = torch.autograd.grad(padded[real].sum(), list(params.values()))
        packed_grads = torch.autograd.grad(packed[real].sum(), list(params.values()))

        # Assert
        for name, padded_grad, packed_grad in zip(
            params, padded_grads, packed_grads, strict=True
        ):
            torch.testing.assert_close(
                packed_grad, padded_grad, atol=self.ATOL, rtol=1e-4, msg=name
            )

    def test_varlen_attention_receives_document_offsets(
        self, pristine_granite_hybrid_classes, varlen_attention
    ) -> None:
        # Arrange
        model = _tiny_granite_hybrid(VARLEN_ATTENTION).eval()
        install_family_patches(model.config.model_type, model)

        # Act
        with torch.no_grad():
            padded, packed, real = _padded_and_packed_logprobs(model)

        # Assert
        torch.testing.assert_close(packed[real], padded[real], atol=self.ATOL, rtol=0)
        assert varlen_attention[-1].tolist() == [
            0,
            *itertools.accumulate(LENGTHS),
        ]

    @pytest.mark.parametrize("attn_implementation", ["sdpa", VARLEN_ATTENTION])
    def test_stacked_packed_rows_match_one_packed_row(
        self, pristine_granite_hybrid_classes, varlen_attention, attn_implementation
    ) -> None:
        # Arrange
        model = _tiny_granite_hybrid(attn_implementation).eval()
        install_family_patches(model.config.model_type, model)
        packed = pack_padded_batch(*_right_padded_documents())

        # Act
        with torch.no_grad():
            one = _token_logprobs(
                model, packed.input_ids, position_ids=packed.position_ids
            )
            stacked = _token_logprobs(
                model,
                packed.input_ids.repeat(2, 1),
                position_ids=packed.position_ids.repeat(2, 1),
            )

        # Assert
        torch.testing.assert_close(stacked, one.repeat(2, 1), atol=self.ATOL, rtol=0)

    def test_unpatched_model_leaks_across_documents(
        self, pristine_granite_hybrid_classes
    ) -> None:
        # Arrange
        model = _tiny_granite_hybrid().eval()

        # Act
        with torch.no_grad():
            padded, packed, real = _padded_and_packed_logprobs(model)

        # Assert
        assert (packed[real] - padded[real]).abs().max() > 1e-3
        assert mixers_without_boundary_reset(model) == ["GraniteMoeHybridMambaLayer"]

    def test_patch_marks_the_mamba_layer_as_resetting(
        self, pristine_granite_hybrid_classes
    ) -> None:
        # Arrange
        model = _tiny_granite_hybrid()

        # Act
        install_family_patches(model.config.model_type, model)

        # Assert
        assert mixers_without_boundary_reset(model) == []

    def test_rows_without_document_boundaries_keep_the_unpatched_output(
        self, pristine_granite_hybrid_classes
    ) -> None:
        # Arrange
        model = _tiny_granite_hybrid().eval()
        ids = _right_padded_documents()[0][:1]
        positions = torch.arange(ids.shape[1]).unsqueeze(0)
        with torch.no_grad():
            unpatched = _token_logprobs(model, ids, position_ids=positions)

        # Act
        install_family_patches(model.config.model_type, model)
        with torch.no_grad():
            patched = _token_logprobs(model, ids, position_ids=positions)

        # Assert
        assert torch.equal(patched, unpatched)

    def test_repeat_patch_keeps_the_installed_forwards(
        self, pristine_granite_hybrid_classes
    ) -> None:
        # Arrange
        patch_granite_hybrid_packed_sequences()
        installed = (GraniteMoeHybridMambaLayer.forward, GraniteMoeHybridModel.forward)

        # Act
        patch_granite_hybrid_packed_sequences()

        # Assert
        assert (
            GraniteMoeHybridMambaLayer.forward,
            GraniteMoeHybridModel.forward,
        ) == installed

    def test_unimportable_modeling_module_logs_and_patches_nothing(
        self, pristine_granite_hybrid_classes, monkeypatch, caplog
    ) -> None:
        # Arrange
        monkeypatch.setattr(
            granite, "GRANITE_HYBRID_MODELING", "agilerl_missing_module"
        )

        # Act
        with caplog.at_level(logging.WARNING, logger=granite.__name__):
            patch_granite_hybrid_packed_sequences()

        # Assert
        assert "agilerl_missing_module unavailable" in caplog.text
        assert mixers_without_boundary_reset(_tiny_granite_hybrid()) == [
            "GraniteMoeHybridMambaLayer"
        ]

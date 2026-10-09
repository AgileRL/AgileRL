# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Packed rows through a tiny Qwen3.5-MoE GDN stack match the padded batch."""

import logging

import torch
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeTextConfig,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeDecoderLayer,
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeGatedDeltaNet,
)

from agilerl.architectures import install_family_patches
from agilerl.architectures.qwen3_5.packed import patch_qwen_gdn_packed_sequences
from agilerl.utils.llm_packing import (
    RESETS_AT_DOCUMENT_BOUNDARY,
    mixers_without_boundary_reset,
    pack_padded_batch,
    unpack_logprobs,
)

MIXER_PATH = (
    "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe.Qwen3_5MoeGatedDeltaNet"
)
BLOCK_PATH = (
    "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe.Qwen3_5MoeDecoderLayer"
)
VOCAB = 64
LENGTHS = (7, 4, 9, 1, 5)


def _tiny_qwen_moe() -> Qwen3_5MoeForCausalLM:
    config = Qwen3_5MoeTextConfig(
        vocab_size=VOCAB,
        hidden_size=32,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        num_experts_per_tok=1,
        num_experts=2,
        layer_types=["linear_attention", "full_attention", "linear_attention"],
        max_position_embeddings=64,
        use_cache=False,
        pad_token_id=0,
    )
    config._attn_implementation = "sdpa"
    torch.manual_seed(0)
    return Qwen3_5MoeForCausalLM(config).float()


def _right_padded_documents() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(1)
    ids = torch.zeros(len(LENGTHS), max(LENGTHS), dtype=torch.long)
    for row, length in enumerate(LENGTHS):
        ids[row, :length] = torch.randint(1, VOCAB, (length,))
    return ids, ids != 0


def _token_logprobs(model: Qwen3_5MoeForCausalLM, ids: torch.Tensor, **kwargs):
    logits = model(input_ids=ids, use_cache=False, **kwargs).logits[:, :-1].float()
    return torch.log_softmax(logits, -1).gather(-1, ids[:, 1:, None]).squeeze(-1)


def _padded_and_packed_logprobs(model: Qwen3_5MoeForCausalLM):
    """Per-token log-probs ``(B, T-1)`` of the padded and the packed forward."""
    ids, mask = _right_padded_documents()
    padded = _token_logprobs(model, ids, attention_mask=mask.long())
    packed_batch = pack_padded_batch(ids, mask)
    packed_row = _token_logprobs(
        model, packed_batch.input_ids, position_ids=packed_batch.position_ids
    )
    return padded, unpack_logprobs(packed_row, packed_batch), mask[:, 1:]


class TestPatchQwenGdnPackedSequences:
    def test_packed_logprobs_match_padded_logprobs(self, pristine_qwen_gdn_classes):
        model = _tiny_qwen_moe().eval()
        install_family_patches(model.config.model_type, model)

        with torch.no_grad():
            padded, packed, real = _padded_and_packed_logprobs(model)

        torch.testing.assert_close(packed[real], padded[real], atol=1e-5, rtol=0)

    def test_packed_gradients_match_padded_gradients(self, pristine_qwen_gdn_classes):
        model = _tiny_qwen_moe().train()
        install_family_patches(model.config.model_type, model)
        params = {
            "gdn_A_log": model.model.layers[0].linear_attn.A_log,
            "attention_q_proj": model.model.layers[1].self_attn.q_proj.weight,
        }

        padded, packed, real = _padded_and_packed_logprobs(model)
        padded_grads = torch.autograd.grad(padded[real].sum(), list(params.values()))
        packed_grads = torch.autograd.grad(packed[real].sum(), list(params.values()))

        for name, padded_grad, packed_grad in zip(
            params, padded_grads, packed_grads, strict=True
        ):
            torch.testing.assert_close(
                packed_grad, padded_grad, atol=1e-5, rtol=1e-4, msg=name
            )

    def test_unpatched_mixer_carries_state_across_documents(
        self, pristine_qwen_gdn_classes
    ):
        model = _tiny_qwen_moe().eval()

        with torch.no_grad():
            padded, packed, real = _padded_and_packed_logprobs(model)

        assert (packed[real] - padded[real]).abs().max() > 1e-3

    def test_padded_multi_row_batch_matches_unpatched(self, pristine_qwen_gdn_classes):
        ids, mask = _right_padded_documents()
        model = _tiny_qwen_moe().eval()
        with torch.no_grad():
            before = _token_logprobs(model, ids, attention_mask=mask.long())

        install_family_patches(model.config.model_type, model)
        with torch.no_grad():
            after = _token_logprobs(model, ids, attention_mask=mask.long())

        torch.testing.assert_close(after, before, atol=1e-5, rtol=0)

    def test_rows_without_document_boundaries_keep_the_unpatched_output(
        self, pristine_qwen_gdn_classes
    ):
        model = _tiny_qwen_moe().eval()
        ids = _right_padded_documents()[0][:1]
        positions = torch.arange(ids.shape[1]).unsqueeze(0)
        with torch.no_grad():
            unpatched = _token_logprobs(model, ids, position_ids=positions)

        patch_qwen_gdn_packed_sequences(mixer=MIXER_PATH, block=BLOCK_PATH)
        with torch.no_grad():
            patched = _token_logprobs(model, ids, position_ids=positions)

        assert torch.equal(patched, unpatched)

    def test_marks_the_mixer_as_resetting_at_boundaries(
        self, pristine_qwen_gdn_classes
    ):
        model = _tiny_qwen_moe()
        unmarked = mixers_without_boundary_reset(model)

        patch_qwen_gdn_packed_sequences(mixer=MIXER_PATH, block=BLOCK_PATH)

        assert unmarked == ["Qwen3_5MoeGatedDeltaNet"]
        assert mixers_without_boundary_reset(model) == []

    def test_apply_is_idempotent(self, pristine_qwen_gdn_classes):
        patch_qwen_gdn_packed_sequences(mixer=MIXER_PATH, block=BLOCK_PATH)
        mixer_forward = Qwen3_5MoeGatedDeltaNet.forward
        block_forward = Qwen3_5MoeDecoderLayer.forward

        patch_qwen_gdn_packed_sequences(mixer=MIXER_PATH, block=BLOCK_PATH)

        assert Qwen3_5MoeGatedDeltaNet.forward is mixer_forward
        assert Qwen3_5MoeDecoderLayer.forward is block_forward
        assert getattr(Qwen3_5MoeGatedDeltaNet, RESETS_AT_DOCUMENT_BOUNDARY) is True

    def test_absent_block_module_warns_and_leaves_the_mixer_unmarked(
        self, pristine_qwen_gdn_classes, caplog
    ):
        with caplog.at_level(logging.WARNING):
            patch_qwen_gdn_packed_sequences(
                mixer=MIXER_PATH, block="agilerl_missing_module.Block"
            )

        assert "packed rows are not supported" in caplog.text
        assert RESETS_AT_DOCUMENT_BOUNDARY not in vars(Qwen3_5MoeGatedDeltaNet)

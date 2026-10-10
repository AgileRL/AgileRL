# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Packed rows through a tiny Nemotron-H on CPU match the padded batch.

Documents are packed into one row with restarting ``position_ids``; the
Mamba2 conv and scan must restart at each boundary, so every real token's
log-prob equals the padded (one document per row) forward.
"""

import itertools
import logging

import pytest
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AttentionInterface
from transformers.masking_utils import AttentionMaskInterface, flash_attention_mask
from transformers.models.nemotron_h import modeling_nemotron_h
from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig
from transformers.models.nemotron_h.modeling_nemotron_h import (
    NemotronHForCausalLM,
    NemotronHMamba2Mixer,
)

from agilerl.architectures import install_family_patches
from agilerl.architectures.nemotron_h.mamba import (
    patch_nemotron_mamba_packed_sequences,
)
from agilerl.utils.llm_packing import (
    RESETS_AT_DOCUMENT_BOUNDARY,
    mixers_without_boundary_reset,
    pack_padded_batch,
    unpack_logprobs,
)

MIXER_PATH = "transformers.models.nemotron_h.modeling_nemotron_h.NemotronHMamba2Mixer"
BLOCK_PATH = "transformers.models.nemotron_h.modeling_nemotron_h.NemotronHBlock"
VOCAB = 64
CHUNK_SIZE = 4
# One document per chunk-boundary case: longer than two chunks, exactly one
# chunk, a single token, and lengths that straddle chunk edges once packed.
LENGTHS = (7, CHUNK_SIZE, 9, 1, 5)
VARLEN_ATTENTION = "agilerl_test_varlen"


@pytest.fixture
def varlen_attention():
    """Register a CPU stand-in for FlashAttention under :data:`VARLEN_ATTENTION`.

    Attention is causal within each ``cu_seq_lens_q`` segment, or over the
    whole row without one. Yields the list of layout kwargs each call received.
    """
    calls = []

    def attend(module, query, key, value, attention_mask, scaling=None, **kwargs):
        cu_seq_lens_q = kwargs.get("cu_seq_lens_q")
        calls.append(
            {
                "position_ids": kwargs.get("position_ids"),
                "cu_seq_lens_q": cu_seq_lens_q,
                "max_length_q": kwargs.get("max_length_q"),
            }
        )
        offsets = (
            [0, query.shape[2]] if cu_seq_lens_q is None else cu_seq_lens_q.tolist()
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
        return torch.cat(segments, dim=2).transpose(1, 2).contiguous(), None

    AttentionInterface.register(VARLEN_ATTENTION, attend)
    AttentionMaskInterface.register(VARLEN_ATTENTION, flash_attention_mask)
    return calls


def _tiny_nemotron_h() -> NemotronHForCausalLM:
    config = NemotronHConfig(
        vocab_size=VOCAB,
        hidden_size=32,
        num_hidden_layers=4,
        layers_block_type=["mamba", "attention", "mamba", "mlp"],
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        intermediate_size=32,
        use_mamba_kernels=False,
        ssm_state_size=8,
        mamba_num_heads=4,
        mamba_head_dim=8,
        n_groups=1,
        conv_kernel=4,
        chunk_size=CHUNK_SIZE,
        max_position_embeddings=64,
        use_cache=False,
    )
    config._attn_implementation = "sdpa"
    torch.manual_seed(0)
    return NemotronHForCausalLM(config).float()


def _right_padded_documents() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(1)
    ids = torch.zeros(len(LENGTHS), max(LENGTHS), dtype=torch.long)
    for row, length in enumerate(LENGTHS):
        ids[row, :length] = torch.randint(1, VOCAB, (length,))
    return ids, ids != 0


def _token_logprobs(model: NemotronHForCausalLM, ids: torch.Tensor, **kwargs):
    logits = model(input_ids=ids, use_cache=False, **kwargs).logits[:, :-1].float()
    return torch.log_softmax(logits, -1).gather(-1, ids[:, 1:, None]).squeeze(-1)


def _padded_and_packed_logprobs(model: NemotronHForCausalLM):
    """Per-token log-probs ``(B, T-1)`` of the padded and the packed forward."""
    ids, mask = _right_padded_documents()
    padded = _token_logprobs(model, ids, attention_mask=mask.long())
    packed_batch = pack_padded_batch(ids, mask)
    packed_row = _token_logprobs(
        model, packed_batch.input_ids, position_ids=packed_batch.position_ids
    )
    return padded, unpack_logprobs(packed_row, packed_batch), mask[:, 1:]


class TestPatchNemotronMambaPackedSequences:
    """Mamba2 conv and scan state restart at every packed-document boundary."""

    def test_packed_logprobs_match_padded_logprobs(self, pristine_nemotron_classes):
        # Arrange
        model = _tiny_nemotron_h().eval()
        install_family_patches(model.config.model_type, model)

        # Act
        with torch.no_grad():
            padded, packed, real = _padded_and_packed_logprobs(model)

        # Assert: column 0 of each later document is the first token after a
        # boundary, the position a carried state would corrupt first.
        torch.testing.assert_close(packed[real], padded[real], atol=1e-5, rtol=0)

    def test_packed_gradients_match_padded_gradients(self, pristine_nemotron_classes):
        # Arrange
        model = _tiny_nemotron_h().train()
        install_family_patches(model.config.model_type, model)
        params = {
            "mixer_in_proj": model.model.layers[2].mixer.in_proj.weight,
            "mixer_A_log": model.model.layers[0].mixer.A_log,
            "attention_q_proj": model.model.layers[1].mixer.q_proj.weight,
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
                packed_grad, padded_grad, atol=1e-5, rtol=1e-4, msg=name
            )

    def test_unpatched_mixer_carries_state_across_documents(
        self, pristine_nemotron_classes
    ):
        # Arrange
        model = _tiny_nemotron_h().eval()

        # Act
        with torch.no_grad():
            padded, packed, real = _padded_and_packed_logprobs(model)

        # Assert
        assert (packed[real] - padded[real]).abs().max() > 1e-3

    def test_rows_without_document_boundaries_keep_the_unpatched_output(
        self, pristine_nemotron_classes
    ):
        # Arrange
        model = _tiny_nemotron_h().eval()
        ids = _right_padded_documents()[0][:1]
        positions = torch.arange(ids.shape[1]).unsqueeze(0)
        with torch.no_grad():
            unpatched = _token_logprobs(model, ids, position_ids=positions)

        # Act
        patch_nemotron_mamba_packed_sequences(mixer=MIXER_PATH, block=BLOCK_PATH)
        with torch.no_grad():
            patched = _token_logprobs(model, ids, position_ids=positions)

        # Assert
        assert torch.equal(patched, unpatched)

    def test_fused_kernel_receives_the_document_index(self, pristine_nemotron_classes):
        # Arrange
        model = _tiny_nemotron_h().eval()
        patch_nemotron_mamba_packed_sequences(mixer=MIXER_PATH, block=BLOCK_PATH)
        mixer = model.model.layers[0].mixer
        received = {}

        def split_scan(zxbcdt, *_args, **kwargs):
            received.update(kwargs)
            weight = kwargs["outproj_weight"]
            return F.linear(zxbcdt[..., : weight.shape[1]], weight)

        modeling_nemotron_h.mamba_split_conv1d_scan_combined = split_scan
        hidden = torch.randn(1, 6, model.config.hidden_size)
        seq_idx = torch.tensor([[0, 0, 0, 1, 1, 2]], dtype=torch.int32)

        # Act
        with torch.no_grad():
            out = mixer.cuda_kernels_forward(hidden, seq_idx=seq_idx)

        # Assert
        projected = mixer.in_proj(hidden)[..., : mixer.out_proj.weight.shape[1]]
        assert torch.equal(out, F.linear(projected, mixer.out_proj.weight))
        assert torch.equal(received["seq_idx"], seq_idx)
        assert received["chunk_size"] == CHUNK_SIZE

    def test_unpacked_eval_call_runs_the_unpatched_cuda_forward(
        self, pristine_nemotron_classes
    ):
        # Arrange
        received = []

        def unpatched(_mixer, hidden_states, cache_params=None, attention_mask=None):
            received.append(hidden_states)
            return hidden_states * 2

        NemotronHMamba2Mixer.cuda_kernels_forward = unpatched
        model = _tiny_nemotron_h().eval()
        patch_nemotron_mamba_packed_sequences(mixer=MIXER_PATH, block=BLOCK_PATH)
        mixer = model.model.layers[0].mixer
        hidden = torch.randn(1, 6, model.config.hidden_size)

        # Act
        out = mixer.cuda_kernels_forward(hidden)

        # Assert
        assert torch.equal(out, hidden * 2)
        assert len(received) == 1
        assert received[0] is hidden

    def test_attention_gets_document_offsets_for_a_packed_row(
        self, pristine_nemotron_classes, varlen_attention
    ):
        # Arrange
        model = _tiny_nemotron_h().eval()
        patch_nemotron_mamba_packed_sequences(mixer=MIXER_PATH, block=BLOCK_PATH)
        ids, mask = _right_padded_documents()
        packed_batch = pack_padded_batch(ids, mask)
        with torch.no_grad():
            padded = _token_logprobs(model, ids, attention_mask=mask.long())
        model.config._attn_implementation = VARLEN_ATTENTION

        # Act
        with torch.no_grad():
            packed_row = _token_logprobs(
                model, packed_batch.input_ids, position_ids=packed_batch.position_ids
            )

        # Assert
        packed = unpack_logprobs(packed_row, packed_batch)
        real = mask[:, 1:]
        torch.testing.assert_close(packed[real], padded[real], atol=1e-5, rtol=0)
        assert len(varlen_attention) == 1
        call = varlen_attention[0]
        assert call["position_ids"] is None
        assert call["cu_seq_lens_q"].tolist() == packed_batch.cu_seqlens.tolist()
        assert call["max_length_q"] == max(LENGTHS)

    def test_attention_gets_no_layout_for_a_one_document_row(
        self, pristine_nemotron_classes, varlen_attention
    ):
        # Arrange
        model = _tiny_nemotron_h().eval()
        model.config._attn_implementation = VARLEN_ATTENTION
        ids = _right_padded_documents()[0][:1]
        positions = torch.arange(ids.shape[1]).unsqueeze(0)
        with torch.no_grad():
            unpatched = _token_logprobs(model, ids, position_ids=positions)
        varlen_attention.clear()

        # Act
        patch_nemotron_mamba_packed_sequences(mixer=MIXER_PATH, block=BLOCK_PATH)
        with torch.no_grad():
            patched = _token_logprobs(model, ids, position_ids=positions)

        # Assert
        assert torch.equal(patched, unpatched)
        assert varlen_attention == [
            {"position_ids": None, "cu_seq_lens_q": None, "max_length_q": None}
        ]

    def test_marks_the_mixer_as_resetting_at_boundaries(
        self, pristine_nemotron_classes
    ):
        # Arrange
        model = _tiny_nemotron_h()
        unmarked = mixers_without_boundary_reset(model)

        # Act
        patch_nemotron_mamba_packed_sequences(mixer=MIXER_PATH, block=BLOCK_PATH)

        # Assert
        assert unmarked == ["NemotronHMamba2Mixer"]
        assert mixers_without_boundary_reset(model) == []

    def test_apply_is_idempotent(self, pristine_nemotron_classes):
        # Arrange
        patch_nemotron_mamba_packed_sequences(mixer=MIXER_PATH, block=BLOCK_PATH)
        mixer_forward = NemotronHMamba2Mixer.forward
        block_forward = modeling_nemotron_h.NemotronHBlock.forward

        # Act
        patch_nemotron_mamba_packed_sequences(mixer=MIXER_PATH, block=BLOCK_PATH)

        # Assert
        assert NemotronHMamba2Mixer.forward is mixer_forward
        assert modeling_nemotron_h.NemotronHBlock.forward is block_forward
        assert getattr(NemotronHMamba2Mixer, RESETS_AT_DOCUMENT_BOUNDARY) is True

    def test_absent_block_module_warns_and_leaves_the_mixer_unmarked(
        self, pristine_nemotron_classes, caplog
    ):
        # Act
        with caplog.at_level(logging.WARNING):
            patch_nemotron_mamba_packed_sequences(
                mixer=MIXER_PATH, block="agilerl_missing_module.Block"
            )

        # Assert
        assert "packed rows are not supported" in caplog.text
        assert RESETS_AT_DOCUMENT_BOUNDARY not in vars(NemotronHMamba2Mixer)


class KernelOutProj(torch.autograd.Function):
    """``out_proj`` as the fused kernel applies it: backward always forms the weight gradient."""

    @staticmethod
    def forward(ctx, scan, weight, bias, weight_grads):
        ctx.save_for_backward(scan, weight)
        ctx.has_bias = bias is not None
        ctx.weight_grads = weight_grads
        return F.linear(scan, weight, bias)

    @staticmethod
    def backward(ctx, grad_out):
        scan, weight = ctx.saved_tensors
        dweight = torch.einsum("bso,bsd->od", grad_out, scan)
        ctx.weight_grads.append(dweight)
        dbias = grad_out.sum(dim=(0, 1)) if ctx.has_bias else None
        return F.linear(grad_out, weight.t()), dweight, dbias, None


def _stand_in_split_scan(weight_grads):
    """CPU stand-in for ``mamba_split_conv1d_scan_combined`` with its ``out_proj`` contract."""

    def split_scan(
        zxbcdt,
        *_args,
        seq_idx=None,
        rmsnorm_weight=None,
        outproj_weight=None,
        outproj_bias=None,
        return_final_states=False,
        **_kwargs,
    ):
        width = rmsnorm_weight.shape[0]
        gate, hidden = zxbcdt[..., :width], zxbcdt[..., width : 2 * width]
        if seq_idx is not None:
            hidden = hidden + seq_idx.unsqueeze(-1)
        scan = hidden * F.silu(gate) * rmsnorm_weight
        if outproj_weight is None:
            assert outproj_bias is None
            out = scan
        else:
            out = KernelOutProj.apply(scan, outproj_weight, outproj_bias, weight_grads)
        return (out, None) if return_final_states else out

    return split_scan


def _mixer_backward(seq_idx, train_out_proj):
    """One mixer forward and backward through the stand-in kernel, with in_proj LoRA."""
    base = _tiny_nemotron_h()
    patch_nemotron_mamba_packed_sequences(mixer=MIXER_PATH, block=BLOCK_PATH)
    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=["in_proj"],
        init_lora_weights=False,
        task_type="CAUSAL_LM",
    )
    torch.manual_seed(2)
    get_peft_model(base, lora_config, adapter_name="actor").train()
    mixer = base.model.layers[0].mixer
    mixer.out_proj.weight.requires_grad_(train_out_proj)
    weight_grads = []
    modeling_nemotron_h.mamba_split_conv1d_scan_combined = _stand_in_split_scan(
        weight_grads
    )
    torch.manual_seed(3)
    hidden = torch.randn(1, 6, base.config.hidden_size, requires_grad=True)
    cotangent = torch.randn(1, 6, base.config.hidden_size)

    out = mixer.cuda_kernels_forward(hidden, seq_idx=seq_idx)
    (out * cotangent).sum().backward()

    lora_grads = {
        name: param.grad
        for name, param in mixer.in_proj.named_parameters()
        if "lora_" in name
    }
    return {
        "out": out.detach(),
        "hidden_grad": hidden.grad,
        "lora_grads": lora_grads,
        "kernel_weight_grads": len(weight_grads),
        "out_proj_grad": mixer.out_proj.weight.grad,
    }


SEQ_IDX_CASES = {
    "packed": torch.tensor([[0, 0, 0, 1, 1, 2]], dtype=torch.int32),
    "unpacked": None,
}


class TestPackedMixerOutProj:
    """A frozen ``out_proj`` runs after the fused scan and matches it inside."""

    @pytest.mark.parametrize("case", list(SEQ_IDX_CASES))
    def test_outputs_and_gradients_match_the_in_kernel_out_proj(
        self, pristine_nemotron_classes, case
    ):
        # Arrange
        seq_idx = SEQ_IDX_CASES[case]

        # Act
        fused = _mixer_backward(seq_idx, train_out_proj=True)
        outside = _mixer_backward(seq_idx, train_out_proj=False)

        # Assert: same fp32 ops forward; backward GEMMs differ only in layout.
        assert torch.equal(outside["out"], fused["out"])
        torch.testing.assert_close(
            outside["hidden_grad"], fused["hidden_grad"], atol=1e-6, rtol=1e-5
        )
        assert set(outside["lora_grads"]) == {
            "lora_A.actor.weight",
            "lora_B.actor.weight",
        }
        for name, grad in fused["lora_grads"].items():
            assert grad.abs().sum() > 0
            torch.testing.assert_close(
                outside["lora_grads"][name], grad, atol=1e-6, rtol=1e-5, msg=name
            )

    @pytest.mark.parametrize("case", list(SEQ_IDX_CASES))
    def test_backward_skips_the_frozen_out_proj_weight_gradient(
        self, pristine_nemotron_classes, case
    ):
        # Act
        result = _mixer_backward(SEQ_IDX_CASES[case], train_out_proj=False)

        # Assert
        assert result["kernel_weight_grads"] == 0
        assert result["out_proj_grad"] is None

    @pytest.mark.parametrize("case", list(SEQ_IDX_CASES))
    def test_a_trainable_out_proj_stays_in_the_kernel(
        self, pristine_nemotron_classes, case
    ):
        # Act
        result = _mixer_backward(SEQ_IDX_CASES[case], train_out_proj=True)

        # Assert
        assert result["kernel_weight_grads"] == 1
        assert result["out_proj_grad"].abs().sum() > 0

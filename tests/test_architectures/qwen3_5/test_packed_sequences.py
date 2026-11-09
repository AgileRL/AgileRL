# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Packed rows through a tiny Qwen3.5-MoE GDN stack match the padded batch."""

import importlib.machinery
import importlib.util
import logging

import pytest
import torch
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    checkpoint_wrapper,
)
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeTextConfig,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeGatedDeltaNet,
    Qwen3_5MoeTextModel,
)

from agilerl.architectures import install_family_patches
from agilerl.architectures.moe_router import patch_moe_routers
from agilerl.architectures.qwen3_5 import packed as qwen_packed
from agilerl.architectures.qwen3_5.packed import (
    PACKED_CU_SEQLENS,
    VARLEN_SCAN,
    install_qwen_gdn_patches,
    patch_qwen_gdn_packed_sequences,
)
from agilerl.architectures.runtime import PatchRuntimeConfig
from agilerl.distributed.fsdp_blocks import _block_checkpoint_contexts
from agilerl.lora.fused import (
    patch_lora_for_fused_forward,
    set_fused_adapter_routing,
    unset_fused_adapter_routing,
)
from agilerl.utils.llm_packing import (
    RESETS_AT_DOCUMENT_BOUNDARY,
    mixers_without_boundary_reset,
    pack_padded_batch,
    unpack_logprobs,
)

MIXER_PATH = (
    "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe.Qwen3_5MoeGatedDeltaNet"
)
TEXT_MODEL_PATH = (
    "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe.Qwen3_5MoeTextModel"
)
VOCAB = 64
LENGTHS = (7, 4, 9, 1, 5)
# Packed rows of equal length, as LLMPPO stacks one packed row per adapter.
MULTI_ROW_LENGTHS = ((7, 4, 9), (1, 5, 6, 8))
HOST_TOLIST = torch.Tensor.tolist


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


def _token_logprobs(model: nn.Module, ids: torch.Tensor, **kwargs):
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


def _multi_row_documents() -> list[list[torch.Tensor]]:
    torch.manual_seed(2)
    return [
        [torch.randint(1, VOCAB, (length,)) for length in lengths]
        for lengths in MULTI_ROW_LENGTHS
    ]


def _stacked_packed_rows(
    rows: list[list[torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """``input_ids`` and restarting ``position_ids`` of one packed row per entry."""
    ids = torch.stack([torch.cat(documents) for documents in rows])
    positions = torch.stack(
        [torch.cat([torch.arange(len(doc)) for doc in documents]) for documents in rows]
    )
    return ids, positions


def _multi_row_and_document_logprobs(
    model: Qwen3_5MoeForCausalLM,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Log-probs of every in-document target from the packed rows and from each document alone."""
    rows = _multi_row_documents()
    ids, positions = _stacked_packed_rows(rows)
    packed = _token_logprobs(model, ids, position_ids=positions)
    packed_targets, document_targets = [], []
    for row, documents in enumerate(rows):
        start = 0
        for document in documents:
            packed_targets.append(packed[row, start : start + len(document) - 1])
            document_targets.append(_token_logprobs(model, document.unsqueeze(0))[0])
            start += len(document)
    return torch.cat(packed_targets), torch.cat(document_targets)


def _routed_logprobs_and_each_adapter(
    model: Qwen3_5MoeForCausalLM,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One packed row stacked per adapter, as LLMPPO routes it, and each adapter alone."""
    config = LoraConfig(r=4, target_modules=["in_proj_qkv", "in_proj_z", "q_proj"])
    peft = get_peft_model(model, config, adapter_name="actor")
    peft.add_adapter("critic", config)
    torch.manual_seed(3)
    for name, param in peft.named_parameters():
        if "lora_B" in name:
            nn.init.normal_(param, std=0.1)
    patch_lora_for_fused_forward(peft)
    ids, positions = _rows("packed")
    set_fused_adapter_routing(peft, ["actor", "critic"])
    routed = _token_logprobs(
        peft, ids.repeat(2, 1), position_ids=positions.repeat(2, 1)
    )
    alone = []
    for adapter in ("actor", "critic"):
        set_fused_adapter_routing(peft, [adapter])
        alone.append(_token_logprobs(peft, ids, position_ids=positions))
    unset_fused_adapter_routing(peft)
    return routed, torch.cat(alone)


def _rows(kind: str) -> tuple[torch.Tensor, torch.Tensor]:
    """``input_ids`` and ``position_ids`` of packed rows or one unpacked row."""
    if kind == "multi_row":
        return _stacked_packed_rows(_multi_row_documents())
    ids, mask = _right_padded_documents()
    if kind == "packed":
        packed = pack_padded_batch(ids, mask)
        return packed.input_ids, packed.position_ids
    full = ids[LENGTHS.index(max(LENGTHS))].unsqueeze(0)
    return full, torch.arange(full.shape[1]).unsqueeze(0)


def _logits_and_grads(
    model: Qwen3_5MoeForCausalLM, ids: torch.Tensor, positions: torch.Tensor
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    logits = model(input_ids=ids, position_ids=positions, use_cache=False).logits
    loss = logits.float().pow(2).mean()
    return logits.detach(), torch.autograd.grad(loss, list(model.parameters()))


def _host_copy_tolist(tensor: torch.Tensor) -> list:
    """``Tensor.tolist`` through the ``_to_copy`` op a CUDA tensor's host copy dispatches."""
    host = torch.ops.aten._to_copy.default(tensor, device=torch.device("cpu"))
    return HOST_TOLIST(host)


@pytest.mark.timeout(300)
@pytest.mark.usefixtures("torch_kernels")
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

    def test_multi_row_packed_logprobs_match_each_document_alone(
        self, pristine_qwen_gdn_classes
    ):
        model = _tiny_qwen_moe().eval()
        install_family_patches(model.config.model_type, model)

        with torch.no_grad():
            packed, documents = _multi_row_and_document_logprobs(model)

        torch.testing.assert_close(packed, documents, atol=1e-5, rtol=0)

    def test_adapter_routed_packed_rows_match_each_adapter_alone(
        self, pristine_qwen_gdn_classes
    ):
        model = _tiny_qwen_moe().eval()
        install_family_patches(model.config.model_type, model)

        with torch.no_grad():
            routed, alone = _routed_logprobs_and_each_adapter(model)

        torch.testing.assert_close(routed, alone, atol=1e-5, rtol=0)

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

        patch_qwen_gdn_packed_sequences(mixer=MIXER_PATH, text_model=TEXT_MODEL_PATH)
        with torch.no_grad():
            patched = _token_logprobs(model, ids, position_ids=positions)

        assert torch.equal(patched, unpatched)

    def test_rope_positions_without_a_text_row_keep_the_unpatched_output(
        self, pristine_qwen_gdn_classes
    ):
        # Arrange
        model = _tiny_qwen_moe().eval()
        ids, positions = _rows("packed")
        rope_positions = positions.expand(3, -1, -1)
        with torch.no_grad():
            unpatched = _token_logprobs(model, ids, position_ids=rope_positions)
        patch_qwen_gdn_packed_sequences(mixer=MIXER_PATH, text_model=TEXT_MODEL_PATH)

        # Act
        with torch.no_grad():
            patched = _token_logprobs(model, ids, position_ids=rope_positions)

        # Assert
        assert torch.equal(patched, unpatched)

    @pytest.mark.parametrize("kind", ["packed", "multi_row", "unpacked"])
    def test_checkpointed_layers_match_eager(
        self, pristine_qwen_gdn_classes, pristine_moe_routers, monkeypatch, kind
    ):
        # Arrange
        model = _tiny_qwen_moe().train()
        install_family_patches(model.config.model_type, model)
        patch_moe_routers(model)
        monkeypatch.setattr(torch.Tensor, "tolist", _host_copy_tolist)
        ids, positions = _rows(kind)
        eager_logits, eager_grads = _logits_and_grads(model, ids, positions)
        for index, layer in enumerate(model.model.layers):
            model.model.layers[index] = checkpoint_wrapper(
                layer,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                preserve_rng_state=False,
                context_fn=_block_checkpoint_contexts(True),
            )

        # Act
        logits, grads = _logits_and_grads(model, ids, positions)

        # Assert
        assert torch.equal(logits, eager_logits)
        for grad, eager_grad in zip(grads, eager_grads, strict=True):
            assert torch.equal(grad, eager_grad)

    @pytest.mark.parametrize("side", ["left", "right"])
    def test_padded_single_row_matches_unpadded_row(
        self, pristine_qwen_gdn_classes, side
    ):
        # Arrange
        model = _tiny_qwen_moe().eval()
        install_family_patches(model.config.model_type, model)
        torch.manual_seed(1)
        tokens = torch.randint(1, VOCAB, (1, 6))
        pads = torch.zeros(1, 3, dtype=torch.long)
        ids = torch.cat((pads, tokens) if side == "left" else (tokens, pads), dim=1)
        mask = (ids != 0).long()
        # Trainer convention: real tokens count from 0, pads sit at 1.
        positions = mask.cumsum(-1) - 1
        positions.masked_fill_(mask == 0, 1)
        real = (mask[:, :-1] * mask[:, 1:]).bool()

        # Act
        with torch.no_grad():
            padded = _token_logprobs(
                model, ids, attention_mask=mask, position_ids=positions
            )
            unpadded = _token_logprobs(model, tokens)

        # Assert
        torch.testing.assert_close(padded[real], unpadded[0], atol=1e-5, rtol=0)

    def test_marks_the_mixer_as_resetting_at_boundaries(
        self, pristine_qwen_gdn_classes
    ):
        model = _tiny_qwen_moe()
        unmarked = mixers_without_boundary_reset(model)

        patch_qwen_gdn_packed_sequences(mixer=MIXER_PATH, text_model=TEXT_MODEL_PATH)

        assert unmarked == ["Qwen3_5MoeGatedDeltaNet"]
        assert mixers_without_boundary_reset(model) == []

    def test_apply_is_idempotent(self, pristine_qwen_gdn_classes):
        patch_qwen_gdn_packed_sequences(mixer=MIXER_PATH, text_model=TEXT_MODEL_PATH)
        mixer_forward = Qwen3_5MoeGatedDeltaNet.forward
        text_model_forward = Qwen3_5MoeTextModel.forward

        patch_qwen_gdn_packed_sequences(mixer=MIXER_PATH, text_model=TEXT_MODEL_PATH)

        assert Qwen3_5MoeGatedDeltaNet.forward is mixer_forward
        assert Qwen3_5MoeTextModel.forward is text_model_forward
        assert getattr(Qwen3_5MoeGatedDeltaNet, RESETS_AT_DOCUMENT_BOUNDARY) is True

    def test_absent_text_model_module_warns_and_leaves_the_mixer_unmarked(
        self, pristine_qwen_gdn_classes, caplog
    ):
        with caplog.at_level(logging.WARNING):
            patch_qwen_gdn_packed_sequences(
                mixer=MIXER_PATH, text_model="agilerl_missing_module.TextModel"
            )

        assert "packed rows are not supported" in caplog.text
        assert RESETS_AT_DOCUMENT_BOUNDARY not in vars(Qwen3_5MoeGatedDeltaNet)

    def test_invalid_class_path_raises(self, pristine_qwen_gdn_classes):
        with pytest.raises(RuntimeError, match="invalid class path"):
            patch_qwen_gdn_packed_sequences(
                mixer="Qwen3_5MoeGatedDeltaNet",
                text_model=TEXT_MODEL_PATH,
            )

    def test_present_module_missing_class_raises(self, pristine_qwen_gdn_classes):
        with pytest.raises(
            RuntimeError, match="is present but missing NoSuchGatedDeltaNet"
        ):
            patch_qwen_gdn_packed_sequences(
                mixer=(
                    "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe"
                    ".NoSuchGatedDeltaNet"
                ),
                text_model=TEXT_MODEL_PATH,
            )

    def test_fla_varlen_forward_passes_a_fresh_copy_of_the_offsets(
        self, pristine_qwen_gdn_classes, monkeypatch
    ):
        # Arrange
        received = {}
        monkeypatch.setattr(qwen_packed, "_varlen_kernels_ready", lambda: True)

        def unpatched(
            self, hidden_states, cache_params=None, attention_mask=None, **kwargs
        ):
            received.update(kwargs)
            return hidden_states

        Qwen3_5MoeGatedDeltaNet.forward = unpatched
        patch_qwen_gdn_packed_sequences(mixer=MIXER_PATH, text_model=TEXT_MODEL_PATH)
        mixer = Qwen3_5MoeGatedDeltaNet.__new__(Qwen3_5MoeGatedDeltaNet)
        hidden = torch.zeros(1, 5, 4)
        seq_idx = torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.int32)
        offsets = torch.tensor([0, 2, 5], dtype=torch.int32)

        # Act
        out = mixer.forward(hidden, seq_idx=seq_idx, **{PACKED_CU_SEQLENS: offsets})

        # Assert
        assert torch.equal(out, hidden)
        assert torch.equal(received["cu_seq_lens_q"], offsets)
        assert received["cu_seq_lens_q"] is not offsets
        assert torch.equal(received["seq_idx"], seq_idx)
        assert PACKED_CU_SEQLENS not in received

    def test_fla_varlen_forward_copies_offsets_over_caller_attention_offsets(
        self, pristine_qwen_gdn_classes, monkeypatch
    ):
        # Arrange: the trainer passes FlashAttention offsets to every layer.
        received = {}
        monkeypatch.setattr(qwen_packed, "_varlen_kernels_ready", lambda: True)

        def unpatched(
            self, hidden_states, cache_params=None, attention_mask=None, **kwargs
        ):
            received.update(kwargs)
            return hidden_states

        Qwen3_5MoeGatedDeltaNet.forward = unpatched
        patch_qwen_gdn_packed_sequences(mixer=MIXER_PATH, text_model=TEXT_MODEL_PATH)
        mixer = Qwen3_5MoeGatedDeltaNet.__new__(Qwen3_5MoeGatedDeltaNet)
        seq_idx = torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.int32)
        offsets = torch.tensor([0, 2, 5], dtype=torch.int32)
        attention_offsets = offsets.clone()

        # Act
        mixer.forward(
            torch.zeros(1, 5, 4),
            seq_idx=seq_idx,
            cu_seq_lens_q=attention_offsets,
            **{PACKED_CU_SEQLENS: offsets},
        )

        # Assert
        assert torch.equal(received["cu_seq_lens_q"], offsets)
        assert received["cu_seq_lens_q"] is not offsets
        assert received["cu_seq_lens_q"] is not attention_offsets

    def test_text_model_passes_offsets_spanning_every_packed_row(
        self, pristine_qwen_gdn_classes, monkeypatch
    ):
        # Arrange
        received = []
        monkeypatch.setattr(qwen_packed, "_varlen_kernels_ready", lambda: True)

        def unpatched(
            self, hidden_states, cache_params=None, attention_mask=None, **kwargs
        ):
            received.append(kwargs)
            return hidden_states

        Qwen3_5MoeGatedDeltaNet.forward = unpatched
        patch_qwen_gdn_packed_sequences(mixer=MIXER_PATH, text_model=TEXT_MODEL_PATH)
        model = _tiny_qwen_moe().eval()
        ids, positions = _rows("multi_row")

        # Act
        with torch.no_grad():
            model(input_ids=ids, position_ids=positions, use_cache=False)

        # Assert
        first, second = received
        assert first["cu_seq_lens_q"].tolist() == [0, 7, 11, 20, 21, 26, 32, 40]
        assert torch.equal(second["cu_seq_lens_q"], first["cu_seq_lens_q"])
        assert second["cu_seq_lens_q"] is not first["cu_seq_lens_q"]
        assert first["seq_idx"].tolist() == [
            [0] * 7 + [1] * 4 + [2] * 9,
            [0] + [1] * 5 + [2] * 6 + [3] * 8,
        ]

    def test_padded_rows_without_a_mask_reach_the_mixer_without_offsets(
        self, pristine_qwen_gdn_classes, monkeypatch
    ):
        # Arrange
        received = []
        monkeypatch.setattr(qwen_packed, "_varlen_kernels_ready", lambda: True)

        def unpatched(
            self, hidden_states, cache_params=None, attention_mask=None, **kwargs
        ):
            received.append(kwargs)
            return hidden_states

        Qwen3_5MoeGatedDeltaNet.forward = unpatched
        patch_qwen_gdn_packed_sequences(mixer=MIXER_PATH, text_model=TEXT_MODEL_PATH)
        model = _tiny_qwen_moe().eval()
        ids, mask = _right_padded_documents()
        # Trainer convention: real tokens count from 0, pads sit at 1.
        positions = mask.long().cumsum(-1) - 1
        positions.masked_fill_(mask == 0, 1)

        # Act
        with torch.no_grad():
            model(input_ids=ids, position_ids=positions, use_cache=False)

        # Assert
        assert received
        for kwargs in received:
            assert "seq_idx" not in kwargs
            assert "cu_seq_lens_q" not in kwargs

    def test_varlen_scan_runs_packed_rows_as_one_row(
        self, pristine_qwen_gdn_classes, monkeypatch
    ):
        # Arrange
        received = {}

        def scan(query, key, value, g, beta, cu_seqlens=None, **kwargs):
            received.update(rows=query.shape, gates=g.shape, cu_seqlens=cu_seqlens)
            return value * 2, None

        monkeypatch.setattr(modeling_qwen3_5_moe, VARLEN_SCAN, scan)
        patch_qwen_gdn_packed_sequences(mixer=MIXER_PATH, text_model=TEXT_MODEL_PATH)
        flattened = getattr(modeling_qwen3_5_moe, VARLEN_SCAN)
        query = torch.randn(2, 5, 3, 8)
        value = torch.randn(2, 5, 3, 8)
        gates = torch.randn(2, 5, 3)
        cu_seqlens = torch.tensor([0, 2, 5, 6, 8, 10], dtype=torch.int32)

        # Act
        out, state = flattened(
            query, query, value, g=gates, beta=gates, cu_seqlens=cu_seqlens
        )

        # Assert
        assert received["rows"] == (1, 10, 3, 8)
        assert received["gates"] == (1, 10, 3)
        assert received["cu_seqlens"] is cu_seqlens
        assert torch.equal(out, value * 2)
        assert state is None

    def test_module_missing_the_varlen_scan_raises(
        self, pristine_qwen_gdn_classes, monkeypatch
    ):
        monkeypatch.delattr(modeling_qwen3_5_moe, VARLEN_SCAN)

        with pytest.raises(RuntimeError, match=f"missing {VARLEN_SCAN}"):
            patch_qwen_gdn_packed_sequences(
                mixer=MIXER_PATH, text_model=TEXT_MODEL_PATH
            )


class TestInstallQwenGdnPatches:
    def test_gpu_without_fla_raises(self, monkeypatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        real = importlib.util.find_spec

        def find_spec(name: str, package: str | None = None):
            if name == "fla":
                return None
            return real(name, package)

        monkeypatch.setattr(importlib.util, "find_spec", find_spec)

        with pytest.raises(RuntimeError, match="flash-linear-attention"):
            install_qwen_gdn_patches(PatchRuntimeConfig())

    def test_gpu_with_fla_and_causal_conv1d_installs_packed_patches(
        self, monkeypatch, pristine_qwen_gdn_classes
    ) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        real = importlib.util.find_spec

        def find_spec(name: str, package: str | None = None):
            if name in {"fla", "causal_conv1d"}:
                return importlib.machinery.ModuleSpec(name, None)
            return real(name, package)

        monkeypatch.setattr(importlib.util, "find_spec", find_spec)

        install_qwen_gdn_patches(PatchRuntimeConfig())

        assert getattr(Qwen3_5MoeGatedDeltaNet, RESETS_AT_DOCUMENT_BOUNDARY) is True

    def test_cpu_without_fla_installs_packed_patches(
        self, monkeypatch, pristine_qwen_gdn_classes
    ) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        install_qwen_gdn_patches(PatchRuntimeConfig())

        assert getattr(Qwen3_5MoeGatedDeltaNet, RESETS_AT_DOCUMENT_BOUNDARY) is True

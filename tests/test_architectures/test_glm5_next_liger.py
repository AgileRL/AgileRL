# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for the GLM-5.3 Liger applier.

Liger's Triton kernels need a GPU, so the CPU tests check which modules the
applier rebinds and run the MLP wiring with the SwiGLU op swapped for its
PyTorch reference. The GPU class checks kernel numerics against the stock model.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F

from agilerl import HAS_LIGER_KERNEL

if not HAS_LIGER_KERNEL:
    pytest.skip(
        "GLM-5.3 Liger tests require liger-kernel; skipping on this platform.",
        allow_module_level=True,
    )
from liger_kernel.transformers.monkey_patch import MODEL_TYPE_TO_APPLY_LIGER_FN
from liger_kernel.transformers.rms_norm import LigerRMSNorm
from transformers.models.glm5_next import modeling_glm5_next
from transformers.models.glm5_next.configuration_glm5_next import (
    Glm5NextTextConfig,
)

import agilerl.architectures.glm5_next.liger as glm5_next_liger
import agilerl.architectures.glm5_next.patches as glm5_next
from agilerl.architectures.catalog import FAMILY_RUNTIME_CONFIGS

GLM5_NEXT_MODEL_TYPES = ("glm5_next", "glm5_next_text")
STOCK_RMS_NORM = modeling_glm5_next.Glm5NextTextRMSNorm
SWIGLU_LIMIT = 10.0
HIDDEN = 64
SEQ = 8
BATCH = 2
SEED = 0


@pytest.fixture
def restored_glm_modeling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Undo the class swaps an applier or installer makes."""
    monkeypatch.setattr(modeling_glm5_next, "Glm5NextTextRMSNorm", STOCK_RMS_NORM)
    for cls in (
        modeling_glm5_next.Glm5NextTextModel,
        modeling_glm5_next.Glm5NextTextAttention,
    ):
        monkeypatch.setattr(cls, "forward", cls.__dict__["forward"])
    monkeypatch.setattr(
        modeling_glm5_next.Glm5NextTextAttention,
        glm5_next.CAUSAL_SDPA_WHEN_TOPK_COVERS,
        False,
        raising=False,
    )


@pytest.fixture
def liger_registry(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Liger's apply map without GLM-5.3 entries; restored afterwards."""
    for model_type in GLM5_NEXT_MODEL_TYPES:
        monkeypatch.delitem(MODEL_TYPE_TO_APPLY_LIGER_FN, model_type, raising=False)
    return MODEL_TYPE_TO_APPLY_LIGER_FN


def tiny_glm5_next(hidden_act: str = "silu") -> modeling_glm5_next.Glm5NextTextModel:
    """One KDA layer with a dense MLP, one DSA layer with a MoE."""
    config = Glm5NextTextConfig(
        vocab_size=32,
        hidden_size=HIDDEN,
        intermediate_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        layer_types=["linear_attention", "indexed_attention"],
        mlp_layer_types=["dense", "sparse"],
        indexer_types=["full", "full"],
        linear_head_dim=16,
        linear_num_heads=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=32,
        kv_lora_rank=32,
        qk_rope_head_dim=0,
        qk_nope_head_dim=16,
        v_head_dim=16,
        index_n_heads=4,
        index_head_dim=16,
        index_topk=SEQ,
        index_kpool=4,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        hc_mult=4,
        hidden_act=hidden_act,
        swiglu_limit=SWIGLU_LIMIT,
        pad_token_id=0,
    )
    config._attn_implementation = "sdpa"
    torch.manual_seed(SEED)
    return modeling_glm5_next.Glm5NextTextModel(config)


def text_norms(model: torch.nn.Module) -> list[torch.nn.Module]:
    return [
        model.norm,
        *(
            norm
            for layer in model.layers
            for norm in (layer.input_layernorm, layer.post_attention_layernorm)
        ),
        model.layers[1].self_attn.q_a_layernorm,
        model.layers[1].self_attn.kv_a_layernorm,
    ]


def swiglu_mlps(model: torch.nn.Module) -> list[torch.nn.Module]:
    return [model.layers[0].mlp, model.layers[1].mlp.shared_experts]


def runs_liger_rms_norm(norm: torch.nn.Module) -> bool:
    return (
        norm.forward.__func__ is LigerRMSNorm.forward and norm.casting_mode == "llama"
    )


class TestRegisterGlm5NextLiger:
    def test_registers_both_model_types(self, liger_registry: dict) -> None:
        # Act
        glm5_next_liger.register_glm5_next_liger()

        # Assert
        for model_type in GLM5_NEXT_MODEL_TYPES:
            assert (
                liger_registry[model_type]
                is glm5_next_liger.apply_liger_kernel_to_glm5_next
            )

    def test_keeps_an_applier_liger_already_ships(self, liger_registry: dict) -> None:
        # Arrange
        def native(**_kwargs: object) -> None:
            return None

        liger_registry["glm5_next"] = native

        # Act
        glm5_next_liger.register_glm5_next_liger()

        # Assert
        assert liger_registry["glm5_next"] is native

    def test_family_installer_registers_the_applier(
        self, restored_glm_modeling: None, liger_registry: dict
    ) -> None:
        # Act
        glm5_next.install_glm5_next_patches(FAMILY_RUNTIME_CONFIGS["glm5_next"].patch)

        # Assert
        assert set(GLM5_NEXT_MODEL_TYPES) <= set(liger_registry)

    def test_installer_skips_registration_without_liger(
        self,
        restored_glm_modeling: None,
        liger_registry: dict,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Arrange
        monkeypatch.setattr(glm5_next, "HAS_LIGER_KERNEL", False)

        # Act
        glm5_next.install_glm5_next_patches(FAMILY_RUNTIME_CONFIGS["glm5_next"].patch)

        # Assert
        assert set(GLM5_NEXT_MODEL_TYPES).isdisjoint(liger_registry)


class TestApplyLigerKernelToGlm5Next:
    def test_binds_liger_rms_norm_on_every_text_norm(
        self, restored_glm_modeling: None
    ) -> None:
        # Arrange
        model = tiny_glm5_next()

        # Act
        glm5_next_liger.apply_liger_kernel_to_glm5_next(model=model)

        # Assert
        assert all(runs_liger_rms_norm(norm) for norm in text_norms(model))
        assert modeling_glm5_next.Glm5NextTextRMSNorm is LigerRMSNorm

    def test_mlps_feed_clamped_gate_then_up_to_the_swiglu_op(
        self, restored_glm_modeling: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        model = tiny_glm5_next()
        torch.manual_seed(SEED)
        # Large inputs push gate and up past swiglu_limit so both clamps bite.
        hidden = 50 * torch.randn(BATCH, SEQ, HIDDEN)
        mlps = swiglu_mlps(model)
        with torch.no_grad():
            stock = [mlp(hidden) for mlp in mlps]
            gate_peak = max(mlp.gate_proj(hidden).abs().max() for mlp in mlps)
        monkeypatch.setattr(
            glm5_next_liger.LigerSiLUMulFunction,
            "apply",
            lambda gate, up: F.silu(gate) * up,
        )

        # Act
        glm5_next_liger.apply_liger_kernel_to_glm5_next(model=model)
        with torch.no_grad():
            patched = [mlp(hidden) for mlp in mlps]

        # Assert: same ops in the same order as the stock forward.
        assert gate_peak > SWIGLU_LIMIT
        for mlp, stock_out, patched_out in zip(mlps, stock, patched, strict=True):
            assert mlp.forward.__func__ is glm5_next_liger.clamped_swiglu_mlp_forward
            assert torch.equal(patched_out, stock_out)

    def test_leaves_routed_experts_and_non_rms_norms_stock(
        self, restored_glm_modeling: None
    ) -> None:
        # Arrange
        model = tiny_glm5_next()
        linear_layer, dsa_layer = model.layers

        # Act
        glm5_next_liger.apply_liger_kernel_to_glm5_next(model=model)

        # Assert
        assert "forward" not in vars(dsa_layer.mlp.experts)
        assert "forward" not in vars(dsa_layer.attn_hc.input_norm)
        assert "forward" not in vars(dsa_layer.self_attn.indexer.k_norm)
        assert "forward" not in vars(linear_layer.self_attn.o_norm)

    def test_rejects_non_silu_mlps(self, restored_glm_modeling: None) -> None:
        # Arrange
        model = tiny_glm5_next(hidden_act="gelu")

        # Act / Assert
        with pytest.raises(ValueError, match="SiLU text MLPs"):
            glm5_next_liger.apply_liger_kernel_to_glm5_next(model=model)

    def test_switches_off_leave_modules_stock(
        self, restored_glm_modeling: None
    ) -> None:
        # Arrange
        model = tiny_glm5_next()

        # Act
        glm5_next_liger.apply_liger_kernel_to_glm5_next(
            rms_norm=False, swiglu=False, model=model
        )

        # Assert
        assert modeling_glm5_next.Glm5NextTextRMSNorm is STOCK_RMS_NORM
        assert not any(runs_liger_rms_norm(norm) for norm in text_norms(model))
        assert all("forward" not in vars(mlp) for mlp in swiglu_mlps(model))

    def test_without_a_model_rebinds_only_the_modeling_module(
        self, restored_glm_modeling: None
    ) -> None:
        # Arrange
        model = tiny_glm5_next()

        # Act
        glm5_next_liger.apply_liger_kernel_to_glm5_next()

        # Assert
        assert modeling_glm5_next.Glm5NextTextRMSNorm is LigerRMSNorm
        assert not any(runs_liger_rms_norm(norm) for norm in text_norms(model))


@pytest.mark.gpu
class TestApplyLigerKernelToGlm5NextOnCuda:
    def test_matches_stock_outputs_and_gradients(
        self, restored_glm_modeling: None
    ) -> None:
        # Arrange
        stock_model = tiny_glm5_next().cuda()
        liger_model = copy.deepcopy(stock_model)
        glm5_next_liger.apply_liger_kernel_to_glm5_next(model=liger_model)
        torch.manual_seed(SEED)
        inputs = 4 * torch.randn(BATCH, SEQ, HIDDEN, device="cuda")
        weights = torch.randn(BATCH, SEQ, HIDDEN, device="cuda")
        outputs = []
        grads = []

        # Act
        for model in (stock_model, liger_model):
            model_inputs = inputs.clone().requires_grad_(True)
            output = model(inputs_embeds=model_inputs, use_cache=False)
            (output.last_hidden_state * weights).sum().backward()
            outputs.append(output.last_hidden_state.detach())
            grads.append(
                [model_inputs.grad, *(p.grad for p in model.parameters())],
            )

        # Assert
        # fp32 kernels: Liger and PyTorch differ only in reduction order.
        tolerance = {"rtol": 1e-4, "atol": 1e-5}
        assert torch.allclose(outputs[1], outputs[0], **tolerance)
        for liger_grad, stock_grad in zip(grads[1], grads[0], strict=True):
            assert (liger_grad is None) == (stock_grad is None)
            if liger_grad is not None and stock_grad is not None:
                assert torch.allclose(liger_grad, stock_grad, **tolerance)

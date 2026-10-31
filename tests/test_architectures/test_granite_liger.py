# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Granite MoE and hybrid Liger appliers.

Liger's Triton kernels need a GPU, so these tests check which modules the
appliers rebind and run the shared-MLP wiring with the SwiGLU op swapped for
its PyTorch reference. Kernel numerics against the stock modules are a GPU
check.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from agilerl import HAS_LIGER_KERNEL

if not HAS_LIGER_KERNEL:
    pytest.skip(
        "Granite Liger tests require liger-kernel; skipping on this platform.",
        allow_module_level=True,
    )
from liger_kernel.transformers.monkey_patch import MODEL_TYPE_TO_APPLY_LIGER_FN
from liger_kernel.transformers.rms_norm import LigerRMSNorm
from liger_kernel.transformers.rope import liger_rotary_pos_emb
from transformers.models.granitemoe import modeling_granitemoe
from transformers.models.granitemoe.configuration_granitemoe import GraniteMoeConfig
from transformers.models.granitemoehybrid import modeling_granitemoehybrid
from transformers.models.granitemoehybrid.configuration_granitemoehybrid import (
    GraniteMoeHybridConfig,
)

import agilerl.architectures.granite_liger as granite_liger
from agilerl.architectures import granite

GRANITE_MODEL_TYPES = ("granitemoe", "granitemoehybrid")


@pytest.fixture
def restored_granite_modeling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Undo the module-level swaps an applier makes."""
    for module, names in (
        (modeling_granitemoe, ("apply_rotary_pos_emb", "GraniteMoeRMSNorm")),
        (
            modeling_granitemoehybrid,
            ("apply_rotary_pos_emb", "GraniteMoeHybridRMSNorm"),
        ),
    ):
        for name in names:
            monkeypatch.setattr(module, name, getattr(module, name))


@pytest.fixture
def liger_registry(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Liger's apply map without Granite entries; restored afterwards."""
    for model_type in GRANITE_MODEL_TYPES:
        monkeypatch.delitem(MODEL_TYPE_TO_APPLY_LIGER_FN, model_type, raising=False)
    return MODEL_TYPE_TO_APPLY_LIGER_FN


def _tiny_granite_hybrid(
    hidden_act: str = "silu",
) -> modeling_granitemoehybrid.GraniteMoeHybridForCausalLM:
    config = GraniteMoeHybridConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=16,
        shared_intermediate_size=32,
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
        num_attention_heads=2,
        num_key_value_heads=1,
        num_local_experts=4,
        num_experts_per_tok=2,
        hidden_act=hidden_act,
        position_embedding_type="rope",
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        mamba_n_heads=4,
        mamba_d_state=8,
        mamba_chunk_size=4,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    return modeling_granitemoehybrid.GraniteMoeHybridForCausalLM(config)


def _tiny_granite_moe() -> modeling_granitemoe.GraniteMoeForCausalLM:
    config = GraniteMoeConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_local_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    return modeling_granitemoe.GraniteMoeForCausalLM(config)


def _decoder_norms(model: torch.nn.Module) -> list[torch.nn.Module]:
    base = model.model
    return [
        base.norm,
        *(
            norm
            for layer in base.layers
            for norm in (layer.input_layernorm, layer.post_attention_layernorm)
        ),
    ]


def _runs_liger_rms_norm(norm: torch.nn.Module) -> bool:
    return (
        norm.forward.__func__ is LigerRMSNorm.forward and norm.casting_mode == "llama"
    )


class TestRegisterGraniteLiger:
    def test_registers_both_granite_model_types(self, liger_registry) -> None:
        # Act
        granite_liger.register_granite_liger()

        # Assert
        assert (
            liger_registry["granitemoe"]
            is granite_liger.apply_liger_kernel_to_granitemoe
        )
        assert (
            liger_registry["granitemoehybrid"]
            is granite_liger.apply_liger_kernel_to_granitemoehybrid
        )

    def test_keeps_an_applier_liger_already_ships(self, liger_registry) -> None:
        # Arrange
        def native(**_kwargs) -> None:
            return None

        liger_registry["granitemoe"] = native

        # Act
        granite_liger.register_granite_liger()

        # Assert
        assert liger_registry["granitemoe"] is native

    def test_family_installers_register_the_appliers(self, liger_registry) -> None:
        # Act
        granite.install_granite_moe_patches(None)

        # Assert
        assert set(GRANITE_MODEL_TYPES) <= set(liger_registry)

    def test_installers_skip_registration_without_liger(
        self, liger_registry, monkeypatch
    ) -> None:
        # Arrange
        monkeypatch.setattr(granite, "HAS_LIGER_KERNEL", False)

        # Act
        granite.install_granite_moe_patches(None)

        # Assert
        assert set(GRANITE_MODEL_TYPES).isdisjoint(liger_registry)


class TestApplyLigerKernelToGranitemoehybrid:
    def test_binds_liger_rms_norm_on_every_decoder_norm(
        self, restored_granite_modeling
    ) -> None:
        # Arrange
        model = _tiny_granite_hybrid()

        # Act
        granite_liger.apply_liger_kernel_to_granitemoehybrid(model=model)

        # Assert
        assert all(_runs_liger_rms_norm(norm) for norm in _decoder_norms(model))
        assert modeling_granitemoehybrid.GraniteMoeHybridRMSNorm is LigerRMSNorm
        assert modeling_granitemoehybrid.apply_rotary_pos_emb is liger_rotary_pos_emb

    def test_shared_mlp_feeds_gate_then_up_to_the_swiglu_op(
        self, restored_granite_modeling, monkeypatch
    ) -> None:
        # Arrange
        model = _tiny_granite_hybrid()
        mlp = model.model.layers[0].shared_mlp
        hidden = torch.randn(2, 5, model.config.hidden_size)
        with torch.no_grad():
            stock = mlp(hidden)
        monkeypatch.setattr(
            granite_liger.LigerSiLUMulFunction,
            "apply",
            lambda gate, up: F.silu(gate) * up,
        )

        # Act
        granite_liger.apply_liger_kernel_to_granitemoehybrid(model=model)
        with torch.no_grad():
            patched = mlp(hidden)

        # Assert: same ops in the same order as the stock forward.
        assert mlp.forward.__func__ is granite_liger.shared_mlp_swiglu_forward
        assert torch.equal(patched, stock)

    def test_rejects_a_non_silu_shared_mlp(self, restored_granite_modeling) -> None:
        # Arrange
        model = _tiny_granite_hybrid(hidden_act="gelu")

        # Act / Assert
        with pytest.raises(ValueError, match="SiLU shared MLP"):
            granite_liger.apply_liger_kernel_to_granitemoehybrid(model=model)

    def test_leaves_routed_experts_and_mamba_norm_stock(
        self, restored_granite_modeling
    ) -> None:
        # Arrange
        model = _tiny_granite_hybrid()
        layer = model.model.layers[0]

        # Act
        granite_liger.apply_liger_kernel_to_granitemoehybrid(model=model)

        # Assert
        assert "forward" not in vars(layer.block_sparse_moe.experts)
        assert "forward" not in vars(layer.mamba.norm)

    def test_without_a_model_rebinds_only_the_modeling_module(
        self, restored_granite_modeling
    ) -> None:
        # Arrange
        model = _tiny_granite_hybrid()

        # Act
        granite_liger.apply_liger_kernel_to_granitemoehybrid()

        # Assert
        assert modeling_granitemoehybrid.GraniteMoeHybridRMSNorm is LigerRMSNorm
        assert modeling_granitemoehybrid.apply_rotary_pos_emb is liger_rotary_pos_emb
        assert not any(_runs_liger_rms_norm(norm) for norm in _decoder_norms(model))


class TestApplyLigerKernelToGranitemoe:
    def test_binds_liger_rms_norm_and_rope(self, restored_granite_modeling) -> None:
        # Arrange
        model = _tiny_granite_moe()

        # Act
        granite_liger.apply_liger_kernel_to_granitemoe(model=model)

        # Assert
        assert all(_runs_liger_rms_norm(norm) for norm in _decoder_norms(model))
        assert modeling_granitemoe.GraniteMoeRMSNorm is LigerRMSNorm
        assert modeling_granitemoe.apply_rotary_pos_emb is liger_rotary_pos_emb

    def test_rope_and_norm_switches_leave_their_modules_stock(
        self, restored_granite_modeling
    ) -> None:
        # Arrange
        model = _tiny_granite_moe()
        stock_rope = modeling_granitemoe.apply_rotary_pos_emb

        # Act
        granite_liger.apply_liger_kernel_to_granitemoe(
            rope=False, rms_norm=False, model=model
        )

        # Assert
        assert modeling_granitemoe.apply_rotary_pos_emb is stock_rope
        assert not any(_runs_liger_rms_norm(norm) for norm in _decoder_norms(model))

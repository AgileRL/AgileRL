# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``mimo_v2`` Liger applier.

Liger's Triton kernels need a GPU, so the CPU tests check which modules the
applier rebinds and run the dense-MLP wiring with the SwiGLU op swapped for
its PyTorch reference. Kernel numerics against the stock modules are a GPU
check.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F

from agilerl import HAS_LIGER_KERNEL

if not HAS_LIGER_KERNEL:
    pytest.skip(
        "MiMo-V2 Liger tests require liger-kernel; skipping on this platform.",
        allow_module_level=True,
    )
from liger_kernel.transformers.monkey_patch import MODEL_TYPE_TO_APPLY_LIGER_FN
from liger_kernel.transformers.rms_norm import LigerRMSNorm
from transformers.integrations import flex_attention as flex_integration
from transformers.modeling_utils import AttentionInterface
from transformers.models.mimo_v2_flash import modeling_mimo_v2_flash

import agilerl.architectures.mimo_v2 as mimo_v2
import agilerl.architectures.mimo_v2.liger as mimo_v2_liger
from agilerl.architectures.mimo_v2.config import MIMO_V2_MODEL_TYPE, MiMoV2Config
from agilerl.architectures.runtime import PatchRuntimeConfig

SEED = 0
BATCH = 2
SEQ = 12
DENSE_LAYER = 0
MOE_LAYER = 1


@pytest.fixture
def restored_mimo_modeling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Undo the module-level RMSNorm swap the applier makes."""
    monkeypatch.setattr(
        modeling_mimo_v2_flash,
        "MiMoV2FlashRMSNorm",
        modeling_mimo_v2_flash.MiMoV2FlashRMSNorm,
    )


@pytest.fixture
def liger_registry(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Liger's apply map without a ``mimo_v2`` entry; restored afterwards."""
    monkeypatch.delitem(MODEL_TYPE_TO_APPLY_LIGER_FN, MIMO_V2_MODEL_TYPE, raising=False)
    return MODEL_TYPE_TO_APPLY_LIGER_FN


@pytest.fixture
def restored_native_training(monkeypatch: pytest.MonkeyPatch) -> None:
    """Undo the flex-attention and flex-support swaps of the training install."""
    monkeypatch.setattr(
        flex_integration,
        "flex_attention_forward",
        flex_integration.flex_attention_forward,
    )
    monkeypatch.setitem(
        AttentionInterface._global_mapping,
        "flex_attention",
        AttentionInterface._global_mapping["flex_attention"],
    )
    monkeypatch.setattr(
        modeling_mimo_v2_flash.MiMoV2FlashPreTrainedModel,
        "_supports_flex_attn",
        modeling_mimo_v2_flash.MiMoV2FlashPreTrainedModel._supports_flex_attn,
    )


def tiny_mimo_v2(
    native_config: MiMoV2Config, hidden_act: str = "silu"
) -> modeling_mimo_v2_flash.MiMoV2FlashForCausalLM:
    config = copy.deepcopy(native_config)
    config.hidden_act = hidden_act
    torch.manual_seed(SEED)
    return modeling_mimo_v2_flash.MiMoV2FlashForCausalLM(config)


def decoder_norms(model: torch.nn.Module) -> list[torch.nn.Module]:
    base = model.model
    return [
        base.norm,
        *(
            norm
            for layer in base.layers
            for norm in (layer.input_layernorm, layer.post_attention_layernorm)
        ),
    ]


def runs_liger_rms_norm(norm: torch.nn.Module) -> bool:
    return (
        norm.forward.__func__ is LigerRMSNorm.forward and norm.casting_mode == "llama"
    )


class TestRegisterMimoV2Liger:
    def test_registers_the_mimo_v2_applier(self, liger_registry) -> None:
        # Act
        mimo_v2_liger.register_mimo_v2_liger()

        # Assert
        assert (
            liger_registry[MIMO_V2_MODEL_TYPE]
            is mimo_v2_liger.apply_liger_kernel_to_mimo_v2
        )

    def test_keeps_an_applier_liger_already_ships(self, liger_registry) -> None:
        # Arrange
        def native(**_kwargs) -> None:
            return None

        liger_registry[MIMO_V2_MODEL_TYPE] = native

        # Act
        mimo_v2_liger.register_mimo_v2_liger()

        # Assert
        assert liger_registry[MIMO_V2_MODEL_TYPE] is native

    @pytest.mark.usefixtures("restored_native_training")
    def test_native_training_install_registers_the_applier(
        self, liger_registry
    ) -> None:
        # Act
        mimo_v2.install_mimo_v2_native_training(PatchRuntimeConfig())

        # Assert
        assert (
            liger_registry[MIMO_V2_MODEL_TYPE]
            is mimo_v2_liger.apply_liger_kernel_to_mimo_v2
        )

    @pytest.mark.usefixtures("restored_native_training")
    def test_native_training_install_skips_registration_without_liger(
        self, liger_registry, monkeypatch
    ) -> None:
        # Arrange
        monkeypatch.setattr(mimo_v2, "HAS_LIGER_KERNEL", False)

        # Act
        mimo_v2.install_mimo_v2_native_training(PatchRuntimeConfig())

        # Assert
        assert MIMO_V2_MODEL_TYPE not in liger_registry


@pytest.mark.usefixtures("restored_mimo_modeling")
class TestApplyLigerKernelToMimoV2:
    def test_binds_liger_rms_norm_on_every_decoder_norm(self, native_config) -> None:
        # Arrange
        model = tiny_mimo_v2(native_config)

        # Act
        mimo_v2_liger.apply_liger_kernel_to_mimo_v2(model=model)

        # Assert
        assert all(runs_liger_rms_norm(norm) for norm in decoder_norms(model))
        assert modeling_mimo_v2_flash.MiMoV2FlashRMSNorm is LigerRMSNorm

    def test_dense_mlp_feeds_gate_then_up_to_the_swiglu_op(
        self, native_config, monkeypatch
    ) -> None:
        # Arrange
        model = tiny_mimo_v2(native_config)
        mlp = model.model.layers[DENSE_LAYER].mlp
        hidden = torch.randn(BATCH, SEQ, model.config.hidden_size)
        with torch.no_grad():
            stock = mlp(hidden)
        monkeypatch.setattr(
            mimo_v2_liger.LigerSiLUMulFunction,
            "apply",
            lambda gate, up: F.silu(gate) * up,
        )

        # Act
        mimo_v2_liger.apply_liger_kernel_to_mimo_v2(model=model)
        with torch.no_grad():
            patched = mlp(hidden)

        # Assert
        # Same ops in the same order as the stock forward.
        assert mlp.forward.__func__ is mimo_v2_liger.dense_mlp_swiglu_forward
        assert torch.equal(patched, stock)

    def test_leaves_routed_experts_stock(self, native_config) -> None:
        # Arrange
        model = tiny_mimo_v2(native_config)
        moe = model.model.layers[MOE_LAYER].mlp

        # Act
        mimo_v2_liger.apply_liger_kernel_to_mimo_v2(model=model)

        # Assert
        assert isinstance(moe, modeling_mimo_v2_flash.MiMoV2FlashMoE)
        assert "forward" not in vars(moe)
        assert "forward" not in vars(moe.experts)

    def test_rejects_a_non_silu_dense_mlp(self, native_config) -> None:
        # Arrange
        model = tiny_mimo_v2(native_config, hidden_act="gelu")

        # Act / Assert
        with pytest.raises(ValueError, match="SiLU dense MLP"):
            mimo_v2_liger.apply_liger_kernel_to_mimo_v2(model=model)

    def test_without_a_model_rebinds_only_the_modeling_module(
        self, native_config
    ) -> None:
        # Arrange
        model = tiny_mimo_v2(native_config)

        # Act
        mimo_v2_liger.apply_liger_kernel_to_mimo_v2()

        # Assert
        assert modeling_mimo_v2_flash.MiMoV2FlashRMSNorm is LigerRMSNorm
        assert not any(runs_liger_rms_norm(norm) for norm in decoder_norms(model))

    def test_switches_off_leave_norms_and_mlp_stock(self, native_config) -> None:
        # Arrange
        model = tiny_mimo_v2(native_config)
        stock_norm = modeling_mimo_v2_flash.MiMoV2FlashRMSNorm

        # Act
        mimo_v2_liger.apply_liger_kernel_to_mimo_v2(
            rms_norm=False, swiglu=False, model=model
        )

        # Assert
        assert modeling_mimo_v2_flash.MiMoV2FlashRMSNorm is stock_norm
        assert not any(runs_liger_rms_norm(norm) for norm in decoder_norms(model))
        assert "forward" not in vars(model.model.layers[DENSE_LAYER].mlp)


@pytest.mark.gpu
@pytest.mark.usefixtures("restored_mimo_modeling")
class TestApplyLigerKernelToMimoV2OnCuda:
    def test_matches_stock_outputs_and_gradients(self, native_config) -> None:
        # Arrange
        stock_model = tiny_mimo_v2(native_config).cuda()
        liger_model = copy.deepcopy(stock_model)
        mimo_v2_liger.apply_liger_kernel_to_mimo_v2(model=liger_model)
        torch.manual_seed(SEED)
        inputs = 4 * torch.randn(BATCH, SEQ, native_config.hidden_size, device="cuda")
        weights = torch.randn(BATCH, SEQ, native_config.hidden_size, device="cuda")
        outputs = []
        grads = []

        # Act
        for model in (stock_model, liger_model):
            model_inputs = inputs.clone().requires_grad_(True)
            output = model.model(inputs_embeds=model_inputs, use_cache=False)
            (output.last_hidden_state * weights).sum().backward()
            outputs.append(output.last_hidden_state.detach())
            grads.append([model_inputs.grad, *(p.grad for p in model.parameters())])

        # Assert
        # fp32 kernels: Liger and PyTorch differ only in reduction order.
        tolerance = {"rtol": 1e-4, "atol": 1e-5}
        assert torch.allclose(outputs[1], outputs[0], **tolerance)
        for liger_grad, stock_grad in zip(grads[1], grads[0], strict=True):
            assert (liger_grad is None) == (stock_grad is None)
            if liger_grad is not None and stock_grad is not None:
                assert torch.allclose(liger_grad, stock_grad, **tolerance)

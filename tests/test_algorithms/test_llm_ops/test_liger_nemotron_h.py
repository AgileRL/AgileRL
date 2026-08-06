# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Nemotron-H Liger registration and apply helpers."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import agilerl.algorithms.core.llm_ops.liger_nemotron_h as liger_nemotron_h


class TestRegisterNemotronHLiger:
    def test_returns_false_when_liger_unavailable(self, monkeypatch) -> None:
        monkeypatch.setattr(liger_nemotron_h, "HAS_LIGER", False)
        monkeypatch.setattr(liger_nemotron_h, "_REGISTERED", {"value": False})

        assert liger_nemotron_h.register_nemotron_h_liger() is False

    def test_registers_nemotron_h_key_when_liger_present(self, monkeypatch) -> None:
        registry: dict = {}
        monkeypatch.setattr(liger_nemotron_h, "HAS_LIGER", True)
        monkeypatch.setattr(liger_nemotron_h, "_REGISTERED", {"value": False})
        monkeypatch.setattr(
            liger_nemotron_h,
            "MODEL_TYPE_TO_APPLY_LIGER_FN",
            registry,
        )

        assert liger_nemotron_h.register_nemotron_h_liger() is True
        assert (
            registry["nemotron_h"] is liger_nemotron_h.apply_liger_kernel_to_nemotron_h
        )

    def test_register_is_idempotent(self, monkeypatch) -> None:
        registry: dict = {}
        monkeypatch.setattr(liger_nemotron_h, "HAS_LIGER", True)
        monkeypatch.setattr(liger_nemotron_h, "_REGISTERED", {"value": False})
        monkeypatch.setattr(
            liger_nemotron_h,
            "MODEL_TYPE_TO_APPLY_LIGER_FN",
            registry,
        )

        assert liger_nemotron_h.register_nemotron_h_liger() is True
        assert liger_nemotron_h.register_nemotron_h_liger() is True
        assert list(registry.keys()) == ["nemotron_h"]


class TestApplyLigerKernelToNemotronH:
    def test_raises_when_liger_missing(self, monkeypatch) -> None:
        monkeypatch.setattr(liger_nemotron_h, "HAS_LIGER", False)

        with pytest.raises(ImportError, match="liger-kernel"):
            liger_nemotron_h.apply_liger_kernel_to_nemotron_h()

    def test_applies_instance_patches_on_fake_model(self, monkeypatch) -> None:
        patched_norms: list = []
        relu_calls: list = []

        class FakeReLUSquared:
            def __init__(self) -> None:
                relu_calls.append(self)

        def fake_patch_rms_norm(module, **kwargs) -> None:
            patched_norms.append(module)

        monkeypatch.setattr(liger_nemotron_h, "HAS_LIGER", True)
        monkeypatch.setattr(
            liger_nemotron_h,
            "LigerRMSNorm",
            MagicMock(name="LigerRMSNorm"),
        )
        monkeypatch.setattr(liger_nemotron_h, "LigerReLUSquared", FakeReLUSquared)
        monkeypatch.setattr(
            liger_nemotron_h,
            "liger_rotary_pos_emb",
            MagicMock(name="rope"),
        )
        monkeypatch.setattr(liger_nemotron_h, "LigerCrossEntropyLoss", MagicMock())
        monkeypatch.setattr(
            liger_nemotron_h,
            "_patch_rms_norm_module",
            fake_patch_rms_norm,
        )

        act2fn: dict = {"relu2": object()}
        modeling = SimpleNamespace(
            ACT2FN=act2fn,
            NemotronHRMSNorm=object(),
            apply_rotary_pos_emb=object(),
            NemotronHForCausalLM=SimpleNamespace(forward=object()),
            CrossEntropyLoss=None,
        )
        monkeypatch.setattr(liger_nemotron_h, "modeling_nemotron_h", modeling)

        mlp_mixer = SimpleNamespace(act_fn=object())
        moe_mixer = SimpleNamespace(
            shared_experts=SimpleNamespace(act_fn=object()),
            experts=SimpleNamespace(act_fn=object()),
        )
        mamba_mixer = SimpleNamespace(act=object())
        attn_mixer = SimpleNamespace()

        layers = [
            SimpleNamespace(norm=object(), block_type="mlp", mixer=mlp_mixer),
            SimpleNamespace(norm=object(), block_type="moe", mixer=moe_mixer),
            SimpleNamespace(
                norm=object(),
                block_type="linear_attention",
                mixer=mamba_mixer,
            ),
            SimpleNamespace(
                norm=object(),
                block_type="full_attention",
                mixer=attn_mixer,
            ),
        ]
        base = SimpleNamespace(norm_f=object(), layers=layers)
        model = SimpleNamespace(
            base_model_prefix="model",
            model=base,
            forward=object(),
        )

        liger_nemotron_h.apply_liger_kernel_to_nemotron_h(
            rms_norm=True,
            rope=True,
            relu_squared=True,
            cross_entropy=False,
            fused_linear_cross_entropy=True,
            model=model,
        )

        assert modeling.NemotronHRMSNorm is liger_nemotron_h.LigerRMSNorm
        assert modeling.apply_rotary_pos_emb is liger_nemotron_h.liger_rotary_pos_emb
        assert act2fn["relu2"] is FakeReLUSquared
        assert model.forward.__func__ is liger_nemotron_h.lce_forward
        assert patched_norms[0] is base.norm_f
        assert len(patched_norms) == 1 + len(layers)
        assert isinstance(mlp_mixer.act_fn, FakeReLUSquared)
        assert isinstance(moe_mixer.shared_experts.act_fn, FakeReLUSquared)
        assert isinstance(moe_mixer.experts.act_fn, FakeReLUSquared)
        assert not hasattr(mamba_mixer, "act_fn")
        assert len(relu_calls) == 3

    def test_class_level_lce_when_model_is_none(self, monkeypatch) -> None:
        monkeypatch.setattr(liger_nemotron_h, "HAS_LIGER", True)
        monkeypatch.setattr(liger_nemotron_h, "LigerRMSNorm", MagicMock())
        monkeypatch.setattr(liger_nemotron_h, "LigerReLUSquared", MagicMock())
        monkeypatch.setattr(liger_nemotron_h, "liger_rotary_pos_emb", MagicMock())
        monkeypatch.setattr(
            liger_nemotron_h,
            "_patch_rms_norm_module",
            MagicMock(),
        )

        causal_lm = SimpleNamespace(forward=object())
        modeling = SimpleNamespace(
            ACT2FN={"relu2": object()},
            NemotronHRMSNorm=object(),
            apply_rotary_pos_emb=object(),
            NemotronHForCausalLM=causal_lm,
        )
        monkeypatch.setattr(liger_nemotron_h, "modeling_nemotron_h", modeling)

        liger_nemotron_h.apply_liger_kernel_to_nemotron_h(
            fused_linear_cross_entropy=True,
            model=None,
        )

        assert causal_lm.forward is liger_nemotron_h.lce_forward

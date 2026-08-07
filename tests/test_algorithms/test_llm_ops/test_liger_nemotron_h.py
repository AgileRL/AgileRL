# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Nemotron-H Liger registration and apply helpers."""

import importlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import agilerl.algorithms.core.llm_ops.liger_nemotron_h as liger_nemotron_h


class TestRegisterNemotronHLiger:
    def test_returns_false_when_liger_unavailable(self, monkeypatch) -> None:
        monkeypatch.setattr(liger_nemotron_h, "HAS_LIGER", False)
        monkeypatch.setattr(liger_nemotron_h, "REGISTERED", {"value": False})

        assert liger_nemotron_h.register_nemotron_h_liger() is False

    def test_registers_nemotron_h_key_when_liger_present(self, monkeypatch) -> None:
        registry: dict = {}
        monkeypatch.setattr(liger_nemotron_h, "HAS_LIGER", True)
        monkeypatch.setattr(liger_nemotron_h, "REGISTERED", {"value": False})
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
        monkeypatch.setattr(liger_nemotron_h, "REGISTERED", {"value": False})
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

    def test_raises_when_modeling_nemotron_h_is_none(self, monkeypatch) -> None:
        monkeypatch.setattr(liger_nemotron_h, "HAS_LIGER", True)
        monkeypatch.setattr(liger_nemotron_h, "modeling_nemotron_h", None)

        with pytest.raises(ImportError, match="Nemotron-H Liger"):
            liger_nemotron_h.apply_liger_kernel_to_nemotron_h()

    def test_cross_entropy_and_fused_are_mutually_exclusive(self, monkeypatch) -> None:
        monkeypatch.setattr(liger_nemotron_h, "HAS_LIGER", True)
        monkeypatch.setattr(
            liger_nemotron_h,
            "modeling_nemotron_h",
            SimpleNamespace(),
        )

        with pytest.raises(ValueError, match="cannot both be True"):
            liger_nemotron_h.apply_liger_kernel_to_nemotron_h(
                cross_entropy=True,
                fused_linear_cross_entropy=True,
            )

    def test_cross_entropy_only_patches_loss_class(self, monkeypatch) -> None:
        ce_cls = object()
        monkeypatch.setattr(liger_nemotron_h, "HAS_LIGER", True)
        monkeypatch.setattr(liger_nemotron_h, "LigerCrossEntropyLoss", ce_cls)
        monkeypatch.setattr(liger_nemotron_h, "LigerRMSNorm", MagicMock())
        monkeypatch.setattr(liger_nemotron_h, "LigerReLUSquared", MagicMock())
        monkeypatch.setattr(liger_nemotron_h, "liger_rotary_pos_emb", MagicMock())
        causal_lm = SimpleNamespace(forward=object())
        modeling = SimpleNamespace(
            ACT2FN={"relu2": object()},
            NemotronHRMSNorm=object(),
            apply_rotary_pos_emb=object(),
            NemotronHForCausalLM=causal_lm,
            CrossEntropyLoss=None,
        )
        monkeypatch.setattr(liger_nemotron_h, "modeling_nemotron_h", modeling)

        liger_nemotron_h.apply_liger_kernel_to_nemotron_h(
            cross_entropy=True,
            fused_linear_cross_entropy=False,
            rope=False,
            rms_norm=False,
            relu_squared=False,
        )

        assert modeling.CrossEntropyLoss is ce_cls
        assert causal_lm.forward is not liger_nemotron_h.lce_forward


class TestRegisterNemotronHLigerWhenModelingMissing:
    def test_returns_false_when_modeling_is_none(self, monkeypatch) -> None:
        monkeypatch.setattr(liger_nemotron_h, "HAS_LIGER", True)
        monkeypatch.setattr(liger_nemotron_h, "modeling_nemotron_h", None)
        monkeypatch.setattr(liger_nemotron_h, "REGISTERED", {"value": False})

        assert liger_nemotron_h.register_nemotron_h_liger() is False


class TestLceForward:
    """Observable contract for the fused linear cross-entropy forward."""

    @staticmethod
    def _fake_model(
        *,
        training: bool = True,
        use_return_dict: bool = True,
    ) -> SimpleNamespace:
        hidden = torch.randn(2, 4, 8)

        class _ModelOutputs:
            def __init__(self) -> None:
                self.last_hidden_state = hidden
                self.past_key_values = "pkv"
                self.hidden_states = "hs"
                self.attentions = "attn"

            def __getitem__(self, idx):
                items = (
                    self.last_hidden_state,
                    self.past_key_values,
                    self.hidden_states,
                    self.attentions,
                )
                return items[idx]

        def model_forward(**_kwargs):
            return _ModelOutputs()

        def lm_head(kept):
            return kept.new_zeros(kept.shape[0], kept.shape[1], 16)

        def loss_function(**_kwargs):
            return torch.tensor(1.25)

        return SimpleNamespace(
            training=training,
            config=SimpleNamespace(
                output_attentions=False,
                output_hidden_states=False,
                use_return_dict=use_return_dict,
                hidden_size=8,
                vocab_size=16,
            ),
            model=model_forward,
            lm_head=lm_head,
            loss_function=loss_function,
        )

    def test_skip_logits_true_without_labels_raises(self) -> None:
        fake = self._fake_model()

        with pytest.raises(ValueError, match="skip_logits is True"):
            liger_nemotron_h.lce_forward(
                fake,
                input_ids=torch.ones(2, 4, dtype=torch.long),
                skip_logits=True,
            )

    def test_training_with_labels_uses_fused_lce_path(self, monkeypatch) -> None:
        fake = self._fake_model(training=True)
        lce_calls: list[object] = []
        monkeypatch.setattr(
            liger_nemotron_h,
            "lce_maybe_trainable_lm_head",
            lambda *args, **kwargs: lce_calls.append((args, kwargs)) or "result",
        )
        monkeypatch.setattr(
            liger_nemotron_h,
            "unpack_cross_entropy_result",
            lambda _result: (
                torch.tensor(0.5),
                None,
                torch.tensor(0.9),
                torch.tensor(3),
            ),
        )
        monkeypatch.setattr(
            liger_nemotron_h,
            "LigerCausalLMOutputWithPast",
            lambda **kwargs: SimpleNamespace(**kwargs),
        )
        lm_head = MagicMock(side_effect=fake.lm_head)
        fake.lm_head = lm_head

        out = liger_nemotron_h.lce_forward(
            fake,
            input_ids=torch.ones(2, 4, dtype=torch.long),
            labels=torch.ones(2, 4, dtype=torch.long),
            skip_logits=None,
            return_dict=True,
        )

        assert lce_calls
        lm_head.assert_not_called()
        assert out.loss.item() == pytest.approx(0.5)
        assert out.token_accuracy.item() == pytest.approx(0.9)
        assert out.predicted_tokens.item() == 3

    def test_skip_logits_false_computes_logits_and_loss(self, monkeypatch) -> None:
        fake = self._fake_model(training=False)
        monkeypatch.setattr(
            liger_nemotron_h,
            "LigerCausalLMOutputWithPast",
            lambda **kwargs: SimpleNamespace(**kwargs),
        )

        out = liger_nemotron_h.lce_forward(
            fake,
            input_ids=torch.ones(2, 4, dtype=torch.long),
            labels=torch.ones(2, 4, dtype=torch.long),
            skip_logits=False,
            return_dict=True,
        )

        assert out.logits is not None
        assert out.logits.shape[-1] == 16
        assert out.loss.item() == pytest.approx(1.25)

    def test_return_dict_false_builds_tuple_with_aux_metrics(self, monkeypatch) -> None:
        fake = self._fake_model(training=True, use_return_dict=False)
        monkeypatch.setattr(
            liger_nemotron_h,
            "lce_maybe_trainable_lm_head",
            lambda *args, **kwargs: "result",
        )
        monkeypatch.setattr(
            liger_nemotron_h,
            "unpack_cross_entropy_result",
            lambda _result: (
                torch.tensor(0.25),
                None,
                torch.tensor(0.8),
                torch.tensor(7),
            ),
        )

        out = liger_nemotron_h.lce_forward(
            fake,
            input_ids=torch.ones(2, 4, dtype=torch.long),
            labels=torch.ones(2, 4, dtype=torch.long),
            skip_logits=True,
            return_dict=False,
        )

        assert isinstance(out, tuple)
        assert out[0].item() == pytest.approx(0.25)
        assert out[-2].item() == pytest.approx(0.8)
        assert out[-1].item() == 7

    def test_logits_to_keep_tensor_slice_reaches_lm_head(self, monkeypatch) -> None:
        fake = self._fake_model(training=False)
        seen: list[torch.Tensor] = []

        def lm_head(kept):
            seen.append(kept.detach().clone())
            return kept.new_zeros(kept.shape[0], kept.shape[1], 16)

        fake.lm_head = lm_head
        monkeypatch.setattr(
            liger_nemotron_h,
            "LigerCausalLMOutputWithPast",
            lambda **kwargs: SimpleNamespace(**kwargs),
        )

        liger_nemotron_h.lce_forward(
            fake,
            input_ids=torch.ones(2, 4, dtype=torch.long),
            logits_to_keep=torch.tensor([0, 2]),
            skip_logits=False,
            return_dict=True,
        )

        assert seen
        assert seen[0].shape[1] == 2


class TestImportFallbacks:
    def test_reload_without_llm_or_liger_keeps_names_resolvable(
        self, monkeypatch
    ) -> None:
        import agilerl

        monkeypatch.setattr(agilerl, "HAS_LLM_DEPENDENCIES", False)
        monkeypatch.setattr(agilerl, "HAS_LIGER_KERNEL", False)
        try:
            module = importlib.reload(liger_nemotron_h)
            assert module.HAS_LIGER is False
            assert module.modeling_nemotron_h is None
            assert module.MODEL_TYPE_TO_APPLY_LIGER_FN == {}
            for name in (
                "_patch_rms_norm_module",
                "LigerRMSNorm",
                "LigerReLUSquared",
                "liger_rotary_pos_emb",
                "LigerCrossEntropyLoss",
                "lce_maybe_trainable_lm_head",
                "unpack_cross_entropy_result",
                "LigerCausalLMOutputWithPast",
            ):
                assert getattr(module, name) is None
        finally:
            monkeypatch.undo()
            importlib.reload(liger_nemotron_h)

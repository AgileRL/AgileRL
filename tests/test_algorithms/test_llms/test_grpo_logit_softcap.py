# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""GRPO objective settings on an actor whose head softcaps its final logits."""

from __future__ import annotations

import warnings

import pytest

pytest.importorskip("transformers", reason="LLM tests require transformers.")
pytest.importorskip("peft", reason="LLM tests require peft.")

from agilerl.algorithms.grpo import GRPO
from tests.test_algorithms.test_llms.llm_helpers import (
    DummyConfig,
    DummyMLPPreTrainedModel,
)
from tests.test_algorithms.test_llms.test_grpo_old_logprobs import VOCAB, _make_grpo

SOFTCAP = 30.0


def _softcapped_actor() -> DummyMLPPreTrainedModel:
    """Tiny causal LM whose config softcaps its final logits, as Gemma's does."""
    config = DummyConfig(
        input_size=6, max_tokens=4, vocab_size=VOCAB, final_logit_softcapping=SOFTCAP
    )
    return DummyMLPPreTrainedModel(config=config, device="cpu")


def _make_liger_grpo(monkeypatch: pytest.MonkeyPatch, **overrides: object) -> GRPO:
    """GRPO asking for the Liger loss, with liger-kernel reported installed."""
    monkeypatch.setattr("agilerl.algorithms.core.base.HAS_LIGER_KERNEL", True)
    # liger-kernel's Triton model patches are Linux/CUDA only.
    monkeypatch.setattr(
        "agilerl.algorithms.core.base._apply_liger_kernel_to_instance",
        lambda model, fused_linear_cross_entropy: None,
        raising=False,
    )
    return _make_grpo(
        actor_network=_softcapped_actor(), use_liger_loss=True, **overrides
    )


class TestGRPOSoftcapLigerFallback:
    def test_softcapped_actor_keeps_kl_advantage_shaping(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            agent = _make_liger_grpo(
                monkeypatch, beta=0.1, use_kl_advantage_shaping=True
            )

        messages = [str(warning.message) for warning in caught]
        assert agent.use_liger_loss is False
        assert agent.use_kl_advantage_shaping is True
        assert any(f"softcaps its final logits at {SOFTCAP}" in m for m in messages)
        assert not any(
            "use_kl_advantage_shaping is not supported" in m for m in messages
        )

    def test_softcapped_actor_skips_the_liger_level_warning(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            agent = _make_liger_grpo(
                monkeypatch, loss_type="cispo", importance_sampling_level="turn"
            )

        messages = [str(warning.message) for warning in caught]
        assert agent.use_liger_loss is False
        assert not any("has no fused cispo kernel" in m for m in messages)

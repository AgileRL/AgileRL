# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""GRPO objective settings on an actor whose head softcaps its final logits."""

from __future__ import annotations

import pytest

pytest.importorskip("transformers", reason="LLM tests require transformers.")
pytest.importorskip("peft", reason="LLM tests require peft.")

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


class TestGRPOSoftcapLigerLoss:
    @pytest.mark.parametrize(
        "overrides",
        [
            pytest.param({}, id="token"),
            pytest.param(
                {
                    "loss_type": "gspo",
                    "vllm_importance_sampling_correction": False,
                    "off_policy_token_mask_bounds": None,
                    "off_policy_sequence_mask_threshold": None,
                },
                id="gspo",
            ),
        ],
    )
    def test_softcapped_actor_runs_the_standard_loss(
        self, monkeypatch: pytest.MonkeyPatch, overrides: dict[str, object]
    ) -> None:
        # Arrange
        monkeypatch.setattr("agilerl.algorithms.core.base.HAS_LIGER_KERNEL", True)
        # liger-kernel's Triton model patches are Linux/CUDA only.
        monkeypatch.setattr(
            "agilerl.algorithms.core.base._apply_liger_kernel_to_instance",
            lambda model, fused_linear_cross_entropy: None,
            raising=False,
        )

        # Act
        agent = _make_grpo(
            liger_loss=None, actor_network=_softcapped_actor(), **overrides
        )

        # Assert
        assert agent.final_logit_softcap == SOFTCAP
        assert agent.liger_loss is False

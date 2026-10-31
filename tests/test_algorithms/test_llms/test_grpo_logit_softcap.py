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


class TestGRPOSoftcappedActor:
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
    def test_softcapped_actor_records_its_softcap(
        self, overrides: dict[str, object]
    ) -> None:
        agent = _make_grpo(actor_network=_softcapped_actor(), **overrides)

        assert agent.final_logit_softcap == SOFTCAP

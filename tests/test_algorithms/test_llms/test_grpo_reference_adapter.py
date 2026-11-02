# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for when GRPO builds its separate reference adapter."""

from __future__ import annotations

import warnings
from typing import Any

import pytest
import torch

pytest.importorskip("transformers", reason="LLM tests require transformers.")
pytest.importorskip("peft", reason="LLM tests require peft.")

from peft import LoraConfig

from agilerl.algorithms.core.registry import HyperparameterConfig, RLParameter
from agilerl.algorithms.grpo import GRPO
from tests.test_algorithms.test_llms.llm_helpers import create_module


def _make_grpo(**overrides: Any) -> GRPO:
    """Tiny fp32 GRPO on CPU with a LoRA actor."""
    torch.manual_seed(0)
    kwargs: dict[str, Any] = {
        "actor_network": create_module(
            input_size=6, max_tokens=4, vocab_size=64, device="cpu"
        ),
        "pad_token_id": 63,
        "pad_token": "<pad>",
        "batch_size": 2,
        "group_size": 2,
        "max_output_tokens": 4,
        "max_model_len": 12,
        "wrap": False,
        "gradient_checkpointing": False,
        "device": "cpu",
        "lora_config": LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["linear_1"],
            task_type="CAUSAL_LM",
            lora_dropout=0.0,
        ),
        **overrides,
    }
    return GRPO(**kwargs)


class TestGRPOReferenceAdapter:
    def test_kl_penalty_builds_the_reference_adapter(self) -> None:
        grpo = _make_grpo()

        assert grpo.beta == 0.001
        assert grpo.use_separate_reference_adapter is True
        assert "reference" in grpo.actor.peft_config

    def test_zero_beta_builds_no_reference_adapter(self) -> None:
        grpo = _make_grpo(beta=0.0)

        assert grpo.use_separate_reference_adapter is False
        assert set(grpo.actor.peft_config) == {"actor"}

    def test_mutated_beta_keeps_the_reference_adapter(self) -> None:
        # Mutation lifts a beta of 0 to the range minimum, so the KL term can turn on.
        hp_config = HyperparameterConfig(beta=RLParameter(min=0.001, max=0.01))

        grpo = _make_grpo(beta=0.0, hp_config=hp_config)

        assert "reference" in grpo.actor.peft_config

    def test_reference_update_without_a_kl_penalty_does_not_warn(self) -> None:
        grpo = _make_grpo(beta=0.0)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            grpo.set_reference_policy(1)

        assert grpo.reference_update_tracker == 1

    def test_zero_beta_learn_step_without_reference_adapter(self) -> None:
        grpo = _make_grpo(beta=0.0)
        completions = [torch.randint(0, 63, (2, 10))]
        action_masks = [torch.ones((2, 9), dtype=torch.bool)]
        rewards = torch.tensor([[1.0, 0.0]])

        metrics = grpo.learn((completions, action_masks, rewards))

        assert torch.isfinite(torch.tensor(metrics["loss"]))
        assert set(grpo.actor.peft_config) == {"actor"}


class TestGRPOLoadCheckpointReferenceAdapter:
    def test_zero_beta_checkpoint_loads_into_kl_agent(self, tmp_path) -> None:
        # Arrange
        _make_grpo(beta=0.0).save_checkpoint(str(tmp_path))
        grpo = _make_grpo(beta=0.01)

        # Act
        grpo.load_checkpoint(str(tmp_path))

        # Assert
        assert grpo.beta == 0.0
        assert grpo.use_separate_reference_adapter is True
        assert "reference" in grpo.actor.peft_config
        actor = grpo.actor.base_model.model.linear_1
        assert torch.equal(
            actor.lora_A["reference"].weight, actor.lora_A["actor"].weight
        )

    def test_kl_checkpoint_into_zero_beta_agent_raises(self, tmp_path) -> None:
        _make_grpo(beta=0.01).save_checkpoint(str(tmp_path))
        grpo = _make_grpo(beta=0.0)

        with pytest.raises(ValueError, match=r"beta=0\.01 with a reference adapter"):
            grpo.load_checkpoint(str(tmp_path))

    def test_kl_checkpoint_keeps_zero_beta_without_config_restore(
        self, tmp_path
    ) -> None:
        _make_grpo(beta=0.01).save_checkpoint(str(tmp_path))
        grpo = _make_grpo(beta=0.0)

        grpo.load_checkpoint(
            str(tmp_path), restore_config=False, restore_hyperparameters=False
        )

        assert grpo.beta == 0.0
        assert grpo.use_separate_reference_adapter is False

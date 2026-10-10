# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for LLM REINFORCE turn-level importance sampling in ``learn``.

Pure CPU on a tiny real model in fp32.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers", reason="LLM tests require transformers.")
pytest.importorskip("peft", reason="LLM tests require peft.")

from peft import LoraConfig

from agilerl.algorithms import reinforce_llm
from agilerl.algorithms.reinforce_llm import REINFORCE
from tests.test_algorithms.test_llms.llm_helpers import create_module, record_calls

PAD_TOKEN_ID = 63
VOCAB = 64


def make_turn_reinforce() -> REINFORCE:
    """Tiny fp32 REINFORCE on CPU with turn-level IS and 1-row micro-batches."""
    torch.manual_seed(0)
    return REINFORCE(
        actor_network=create_module(
            input_size=6, max_tokens=4, vocab_size=VOCAB, device="cpu"
        ),
        pad_token_id=PAD_TOKEN_ID,
        pad_token="<pad>",
        batch_size=4,
        beta=0.01,
        lr=1e-2,
        max_output_tokens=4,
        max_model_len=12,
        micro_batch_size_per_gpu=1,
        mini_batch_size=4,
        update_epochs=1,
        importance_sampling_level="turn",
        wrap=False,
        gradient_checkpointing=False,
        calc_position_embeddings=False,
        device="cpu",
        use_liger_loss=False,
        lora_config=LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["linear_1"],
            task_type="CAUSAL_LM",
            lora_dropout=0.0,
        ),
    )


def learn_uneven_turns(agent: REINFORCE) -> dict[str, float]:
    """One learn over four trajectories: rows 0-1 take two turns, rows 2-3 one."""
    generator = torch.Generator().manual_seed(2)
    ids = torch.randint(0, PAD_TOKEN_ID, (4, 8), generator=generator)
    mask = torch.zeros(4, 7, dtype=torch.bool)
    mask[:, 3:] = True
    turn_ids = torch.full((4, 7), -1)
    turn_ids[:, 3:] = 0
    turn_ids[:2, 5:] = 1
    rewards = torch.tensor([[1.0, 0.5], [0.0, -1.0], [0.5, 0.0], [-1.0, 0.0]])
    return agent.learn(
        (list(ids.split(1)), list(mask.split(1)), rewards), turn_ids=turn_ids
    )


class TestREINFORCELearnTurnCount:
    def test_micro_batch_losses_match_their_own_turn_count(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange: 1-row micro-batches, so rows 2-3 hold fewer turns than the batch.
        agent = make_turn_reinforce()
        surrogate_fn = reinforce_llm.clipped_is_surrogate
        surrogate, calls = record_calls(surrogate_fn)
        monkeypatch.setattr(reinforce_llm, "clipped_is_surrogate", surrogate)

        # Act
        learn_uneven_turns(agent)

        # Assert
        local_counts = []
        for args, kwargs, (pg_loss, _) in calls:
            turn_ids = args[3]
            local_counts.append(int(turn_ids.max()) + 1)
            assert kwargs["num_turns"] == 2
            with torch.no_grad():
                expected, _ = surrogate_fn(*args, **{**kwargs, "num_turns": None})
            assert torch.equal(pg_loss.detach(), expected)
        assert sorted(local_counts) == [1, 1, 2, 2]

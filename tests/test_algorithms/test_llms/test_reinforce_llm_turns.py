# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for LLM REINFORCE turn-level importance sampling in ``learn``.

Pure CPU on a tiny real model in fp32.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

pytest.importorskip("transformers", reason="LLM tests require transformers.")
pytest.importorskip("peft", reason="LLM tests require peft.")

from peft import LoraConfig

from agilerl.algorithms import reinforce_llm
from agilerl.algorithms.reinforce_llm import REINFORCE
from tests.test_algorithms.test_llms.llm_helpers import create_module, record_calls
from tests.test_algorithms.test_llms.segment_helpers import (
    record_step_gradients,
)

PAD_TOKEN_ID = 63
VOCAB = 64


def make_turn_reinforce(**overrides: Any) -> REINFORCE:
    """Tiny fp32 REINFORCE on CPU with turn-level IS and 1-row micro-batches."""
    torch.manual_seed(0)
    kwargs: dict[str, Any] = {
        "actor_network": create_module(
            input_size=6, max_tokens=4, vocab_size=VOCAB, device="cpu"
        ),
        "pad_token_id": PAD_TOKEN_ID,
        "pad_token": "<pad>",
        "batch_size": 4,
        "beta": 0.01,
        "lr": 1e-2,
        "max_output_tokens": 4,
        "max_model_len": 12,
        "micro_batch_size_per_gpu": 1,
        "mini_batch_size": 4,
        "update_epochs": 1,
        "importance_sampling_level": "turn",
        "wrap": False,
        "gradient_checkpointing": False,
        "calc_position_embeddings": False,
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
    return REINFORCE(**kwargs)


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


class TestREINFORCELearnActionlessRows:
    def test_a_row_without_action_tokens_adds_no_loss_or_metrics(self) -> None:
        """A micro-batch whose every action vLLM forced trains like no row at all."""
        # Arrange
        generator = torch.Generator().manual_seed(2)
        ids = list(torch.randint(0, PAD_TOKEN_ID, (2, 8), generator=generator).split(1))
        masks = [torch.zeros(1, 7, dtype=torch.bool) for _ in range(2)]
        masks[0][0, 3:] = True
        agents = [make_turn_reinforce() for _ in range(2)]

        # Act
        with_forced_row = agents[0].learn((ids, masks, torch.tensor([1.0, 0.0])))
        live_row_only = agents[1].learn((ids[:1], masks[:1], torch.tensor([1.0])))

        # Assert
        for name in ("loss", "pg_loss", "kl", "entropy"):
            assert with_forced_row[name] == live_row_only[name], name


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


def learn_uneven_rows(agent: REINFORCE) -> dict[str, float]:
    """One learn over rows of 8, 4, 6 and 3 real tokens; rows 0 and 2 take two turns."""
    generator = torch.Generator().manual_seed(3)
    ids = torch.randint(0, PAD_TOKEN_ID, (4, 8), generator=generator)
    mask = torch.zeros(4, 7, dtype=torch.bool)
    for row, length in enumerate([8, 4, 6, 3]):
        ids[row, length:] = PAD_TOKEN_ID
        mask[row, 1 : length - 1] = True
    turn_ids = torch.where(mask, 0, -1)
    turn_ids[0, 4:7] = 1
    turn_ids[2, 3:5] = 1
    rewards = torch.tensor([[1.0, 0.5], [0.0, 0.0], [0.5, -1.0], [-1.0, 0.0]])
    return agent.learn(
        (list(ids.split(1)), list(mask.split(1)), rewards), turn_ids=turn_ids
    )


class TestREINFORCELearnTokenBudget:
    @pytest.mark.parametrize("level", ["token", "turn"])
    def test_step_gradients_match_one_window_micro_batch(
        self, monkeypatch: pytest.MonkeyPatch, level: str
    ) -> None:
        # Arrange: a 12-token budget splits the window into micro-batches of
        # unequal tokens and turns; the reference runs the whole window as one.
        window = make_turn_reinforce(
            micro_batch_size_per_gpu=4, importance_sampling_level=level
        )
        budgeted = make_turn_reinforce(
            micro_batch_size_per_gpu=4,
            micro_batch_tokens_per_gpu=12,
            importance_sampling_level=level,
        )
        monkeypatch.setattr(budgeted, "_packing_mode", lambda: "varlen")
        forward_tokens: list[int] = []

        def record(
            _module: torch.nn.Module, _args: tuple[Any, ...], kwargs: dict[str, Any]
        ) -> None:
            if torch.is_grad_enabled():
                forward_tokens.append(int((kwargs["input_ids"] != PAD_TOKEN_ID).sum()))

        budgeted.actor.register_forward_pre_hook(record, with_kwargs=True)
        window_steps = record_step_gradients(window, monkeypatch)
        budgeted_steps = record_step_gradients(budgeted, monkeypatch)

        # Act
        window_metrics = learn_uneven_rows(window)
        budgeted_metrics = learn_uneven_rows(budgeted)

        # Assert
        assert budgeted_metrics["micro_batches_per_step"] == len(forward_tokens)
        assert window_metrics["micro_batches_per_step"] == 1.0
        assert len(forward_tokens) >= 2
        assert max(forward_tokens) <= 12
        assert len(window_steps) == len(budgeted_steps) == 1
        assert budgeted_steps[0].keys() == window_steps[0].keys()
        for name, grad in window_steps[0].items():
            torch.testing.assert_close(
                budgeted_steps[0][name], grad, rtol=1e-5, atol=1e-6, msg=name
            )

    def test_budget_below_the_longest_row_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"micro_batch_tokens_per_gpu \(11\)"):
            make_turn_reinforce(micro_batch_tokens_per_gpu=11)


def learn_three_turns(agent: REINFORCE, forced: bool) -> dict[str, float]:
    """One learn over two three-turn episodes of two tokens per turn.

    With ``forced``, every token of row 0's middle and last turn leaves the
    action mask and keeps its turn id.
    """
    generator = torch.Generator().manual_seed(2)
    ids = torch.randint(0, PAD_TOKEN_ID, (2, 8), generator=generator)
    turn_ids = torch.tensor([[-1, 0, 0, 1, 1, 2, 2]]).repeat(2, 1)
    mask = turn_ids >= 0
    if forced:
        mask[0, 3:] = False
    rewards = torch.tensor([[0.5, 1.0, 2.0], [0.0, -1.0, 0.5]])
    return agent.learn(
        (list(ids.split(1)), list(mask.split(1)), rewards), turn_ids=turn_ids
    )


class TestREINFORCELearnForcedTurns:
    """A turn of forced tokens keeps its reward and its place in the episode."""

    def test_advantages_match_the_episode_with_free_turns_on_action_tokens(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        advantages = {}
        masks = {}
        for forced in (False, True):
            agent = make_turn_reinforce()
            agent.beta = 0.0
            agent.gamma = 0.9
            rebn, rebn_calls = record_calls(agent._compute_rebn_advantages)
            monkeypatch.setattr(agent, "_compute_rebn_advantages", rebn)

            # Act
            learn_three_turns(agent, forced=forced)
            ((args, _kwargs, out),) = rebn_calls
            advantages[forced], masks[forced] = out, args[2]

        # Assert
        assert torch.equal(advantages[True], advantages[False] * masks[True])
        assert advantages[False][0, 3:].abs().min() > 0


def learn_with_kl(
    agent: REINFORCE, monkeypatch: pytest.MonkeyPatch, forced: bool
) -> torch.Tensor:
    """One learn over two two-turn episodes with a set per-token KL.

    With ``forced``, row 0's turn 0 has a forced token and its turn 1 is fully
    forced. Without it, that forced token is not in the turn and turn 1's
    tokens are free actions with zero KL.

    :return: The episodes' action mask.
    """
    generator = torch.Generator().manual_seed(2)
    ids = torch.randint(0, PAD_TOKEN_ID, (2, 7), generator=generator)
    turn_ids = torch.tensor([[-1, 0, 0, 0, 1, 1]]).repeat(2, 1)
    mask = turn_ids >= 0
    kl = torch.tensor([[7.0, 0.3, 0.6, 5.0, 5.0, 5.0], [7.0, 0.2, 0.4, 0.1, 0.5, 0.3]])
    if forced:
        mask[0, 3:] = False
    else:
        turn_ids[0, 3] = -1
        mask[0, 3] = False
        kl[0, 4:] = 0.0
    reference_log_probs = torch.full((2, 6), -2.0)

    def set_kl_forward(*_args: object, **_kwargs: object) -> tuple:
        return reference_log_probs.clone(), reference_log_probs + kl, None

    monkeypatch.setattr(agent, "_fused_forward_no_grad", set_kl_forward)
    rewards = torch.tensor([[0.5, 2.0], [0.0, -1.0]])
    agent.learn((list(ids.split(1)), list(mask.split(1)), rewards), turn_ids=turn_ids)
    return mask


def make_kl_reinforce() -> REINFORCE:
    agent = make_turn_reinforce()
    agent.beta = 0.5
    agent.gamma = 0.9
    return agent


class TestREINFORCELearnForcedTurnKLPenalty:
    """A turn's KL penalty averages over its action tokens only."""

    def test_forced_tokens_add_no_kl_and_keep_the_turn_reward(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        advantages = {}
        masks = {}
        for forced in (False, True):
            agent = make_kl_reinforce()
            rebn, rebn_calls = record_calls(agent._compute_rebn_advantages)
            monkeypatch.setattr(agent, "_compute_rebn_advantages", rebn)

            # Act
            masks[forced] = learn_with_kl(agent, monkeypatch, forced=forced)
            ((_args, _kwargs, advantages[forced]),) = rebn_calls

        # Assert
        assert torch.isfinite(advantages[True]).all()
        assert torch.allclose(
            advantages[True], advantages[False] * masks[True], rtol=0.0, atol=1e-6
        )

    def test_the_loss_sees_the_same_advantages(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        seen = {}
        for forced in (False, True):
            agent = make_kl_reinforce()
            loss, calls = record_calls(reinforce_llm.clipped_is_surrogate)
            monkeypatch.setattr(reinforce_llm, "clipped_is_surrogate", loss)

            # Act
            learn_with_kl(agent, monkeypatch, forced=forced)
            seen[forced] = [(args[1], args[2]) for args, _kwargs, _out in calls]

        # Assert
        assert len(seen[True]) == len(seen[False]) == 2
        for (forced_adv, units), (free_adv, _) in zip(
            seen[True], seen[False], strict=True
        ):
            assert torch.allclose(
                forced_adv * units, free_adv * units, rtol=0.0, atol=1e-6
            )

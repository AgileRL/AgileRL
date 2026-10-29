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

from agilerl import HAS_LIGER_KERNEL
from agilerl.algorithms import reinforce_llm
from agilerl.algorithms.reinforce_llm import REINFORCE
from tests.test_algorithms.test_llms.llm_helpers import create_module, record_calls
from tests.test_algorithms.test_llms.segment_helpers import record_step_gradients

PAD_TOKEN_ID = 63
VOCAB = 64


def make_turn_reinforce(importance_sampling_level: str = "turn") -> REINFORCE:
    """Tiny fp32 REINFORCE on CPU with 1-row micro-batches.

    :param importance_sampling_level: Importance-sampling level of the update.
    """
    torch.manual_seed(0)
    agent = REINFORCE(
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
        importance_sampling_level=importance_sampling_level,
        wrap=False,
        calc_position_embeddings=False,
        device="cpu",
        lora_config=LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["linear_1"],
            task_type="CAUSAL_LM",
            lora_dropout=0.0,
        ),
    )
    agent.liger_loss = False
    return agent


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


@pytest.mark.skipif(not HAS_LIGER_KERNEL, reason="liger-kernel is Linux-only.")
class TestREINFORCELearnLigerLoss:
    """The fused Liger loss pools turns and trajectories as the standard loss does."""

    @pytest.mark.parametrize("is_level", ["turn", "trajectory"])
    def test_loss_and_gradients_match_the_standard_learn(
        self, monkeypatch: pytest.MonkeyPatch, is_level: str
    ) -> None:
        # Arrange
        agents = {}
        gradients = {}
        for liger in (False, True):
            agent = make_turn_reinforce(is_level)
            agent.liger_loss = liger
            gradients[liger] = record_step_gradients(agent, monkeypatch)
            agents[liger] = agent
        fused, fused_calls = record_calls(agents[True]._reinforce_loss_liger)
        monkeypatch.setattr(agents[True], "_reinforce_loss_liger", fused)

        # Act
        metrics = {liger: learn_uneven_turns(agent) for liger, agent in agents.items()}

        # Assert
        assert len(fused_calls) == 4
        for name in ("loss", "pg_loss", "kl"):
            assert metrics[True][name] == pytest.approx(
                metrics[False][name], rel=1e-5, abs=1e-6
            ), name
        for standard, fused_grads in zip(
            gradients[False], gradients[True], strict=True
        ):
            assert standard.keys() == fused_grads.keys()
            for name, grad in standard.items():
                assert torch.allclose(fused_grads[name], grad, rtol=1e-5, atol=1e-6), (
                    name
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

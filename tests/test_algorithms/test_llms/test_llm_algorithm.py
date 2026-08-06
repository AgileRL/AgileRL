# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the shared :meth:`LLMAlgorithm.test` evaluation loop.

The method only touches ``self.get_action``, ``self.metrics`` and
``self.accelerator`` (the end-of-eval barrier), so it is exercised here with a
stub in place of a fully constructed algorithm — no model, GPU, deepspeed, or
vllm required.
"""

import numpy as np
import pytest
import torch

from agilerl.algorithms.core import ActionResult
from agilerl.algorithms.core.base import LLMAlgorithm
from agilerl.llm_envs import RolloutHarness
from agilerl.metrics import AgentMetrics
from tests.helpers.rollout_doubles import FakeEnvClient


class _StubAlgo:
    """Minimal stand-in exposing the attributes ``LLMAlgorithm.test`` uses."""

    def __init__(self):
        self.metrics = AgentMetrics()
        self.accelerator = None  # single-process: the end-of-eval barrier is a no-op

    @property
    def fitness(self):
        return list(self.metrics.fitness)

    def get_action(self, prompts, training=False):
        del prompts, training
        completion = torch.ones(1, 5, dtype=torch.long)
        action_mask = torch.ones(1, 4, dtype=torch.bool)
        return ActionResult([completion], [action_mask])


class _TinyTokenizer:
    """Raw-encoding tokenizer for the ``apply_chat_template=False`` paths."""

    def __call__(self, texts, **kwargs):
        del kwargs
        return {"input_ids": torch.ones(len(texts), 4, dtype=torch.long)}

    def encode(self, text, add_special_tokens=True):
        del text, add_special_tokens
        return [7, 8]

    def decode(self, ids, skip_special_tokens=True):
        del ids, skip_special_tokens
        return "gen"


def _rollout_env(client: FakeEnvClient, max_turns: int, **kwargs) -> RolloutHarness:
    return RolloutHarness(
        client,
        _TinyTokenizer(),
        max_turns=max_turns,
        apply_chat_template=False,
        **kwargs,
    )


class TestLLMAlgorithmTest:
    def test_env_truncation_ends_each_episode_at_max_turns(self):
        algo = _StubAlgo()
        client = FakeEnvClient()  # backend never ends the episode itself
        env = _rollout_env(client, max_turns=3)

        out = LLMAlgorithm.test(algo, env, loop=2)

        # Two episodes of exactly ``max_turns`` turns each, run under eval mode.
        assert client.step_calls == 2 * env.max_turns
        assert client.eval_mode_entries == 1
        assert isinstance(out, np.ndarray)
        assert out.shape == ()
        assert out.item() == pytest.approx(1.0)
        assert algo.fitness == [pytest.approx(1.0)]

    def test_terminating_env_finishes_before_max_turns(self):
        algo = _StubAlgo()
        client = FakeEnvClient(reward=0.5, terminate_after=2)
        env = _rollout_env(client, max_turns=5)

        out = LLMAlgorithm.test(algo, env, loop=1)

        assert client.step_calls == 2
        assert out.item() == pytest.approx(0.5)

    def test_done_at_reset_records_zero_fitness(self):
        """An over-budget initial prompt ends the episode before any turn."""
        algo = _StubAlgo()
        env = _rollout_env(
            FakeEnvClient(),
            max_turns=3,
            max_model_len=4,
            max_output_tokens=2,
        )

        with pytest.warns(UserWarning, match="collected no turns"):
            out = LLMAlgorithm.test(algo, env, loop=1)

        assert out.item() == 0.0
        assert algo.fitness == [0.0]

    def test_subclasses_share_the_base_implementation(self):
        from agilerl.algorithms.grpo import GRPO
        from agilerl.algorithms.ppo_llm import PPO
        from agilerl.algorithms.reinforce_llm import REINFORCE

        assert GRPO.test is LLMAlgorithm.test
        assert PPO.test is LLMAlgorithm.test
        assert REINFORCE.test is LLMAlgorithm.test


class TestCreatePromptMasks:
    def test_first_response_token_is_included(self):
        # Prompt occupies positions [0, P); the first response token sits AT
        # index P and must be trainable — a strict ``>`` drops the
        # highest-information token of every SFT/DPO example.
        masks = LLMAlgorithm._create_prompt_masks([2, 3], max_length=5)
        assert masks.tolist() == [
            [False, False, True, True, True],
            [False, False, False, True, True],
        ]

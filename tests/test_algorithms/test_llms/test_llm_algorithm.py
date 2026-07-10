"""CPU tests for the shared :meth:`LLMAlgorithm.test` evaluation loop.

The method only touches ``self.get_action`` and ``self.metrics``, so it is
exercised here with a stub in place of a fully constructed algorithm — no
model, GPU, deepspeed, or vllm required.
"""

import numpy as np
import pytest
import torch

from agilerl.algorithms.core import ActionResult
from agilerl.algorithms.core.base import LLMAlgorithm
from agilerl.llm_envs import RolloutEnv
from agilerl.metrics import AgentMetrics


class _StubAlgo:
    """Minimal stand-in exposing the attributes ``LLMAlgorithm.test`` uses."""

    def __init__(self):
        self.metrics = AgentMetrics()

    @property
    def fitness(self):
        return list(self.metrics.fitness)

    def get_action(self, prompts, training=False):
        del prompts, training
        completion = torch.ones(1, 5, dtype=torch.long)
        action_mask = torch.ones(1, 4, dtype=torch.bool)
        return ActionResult([completion], [action_mask])


class _NeverDoneEnv(RolloutEnv):
    """Rollout env double that never terminates or truncates."""

    max_turns = 3

    def __init__(self, reward: float = 1.0):
        self._env_client = None
        self._reward = reward
        self.step_calls = 0

    def _prompt(self):
        return {
            "input_ids": torch.ones(1, 4, dtype=torch.long),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
        }

    def reset(self, seed=None):
        del seed
        return self._prompt(), {}

    def step(self, full_completion_ids):
        del full_completion_ids
        self.step_calls += 1
        return self._prompt(), self._reward, False, False, {}

    def close(self):
        return None


class TestLLMAlgorithmTestTurnCap:
    def test_turn_cap_stops_non_terminating_env_at_max_turns(self):
        algo = _StubAlgo()
        env = _NeverDoneEnv()

        out = LLMAlgorithm.test(algo, env, loop=2)

        # Two episodes of exactly ``max_turns`` steps each.
        assert env.step_calls == 2 * env.max_turns
        assert isinstance(out, np.ndarray)
        assert out.shape == ()
        # Rewards collected up to the cap are used for the fitness score.
        assert out.item() == pytest.approx(1.0)
        assert algo.fitness == [pytest.approx(1.0)]

    def test_turn_cap_falls_back_to_1000_when_max_turns_is_none(self):
        algo = _StubAlgo()
        env = _NeverDoneEnv()
        env.max_turns = None

        out = LLMAlgorithm.test(algo, env, loop=1)

        assert env.step_calls == 1000
        assert out.item() == pytest.approx(1.0)

    def test_terminating_env_finishes_before_cap(self):
        algo = _StubAlgo()
        env = _NeverDoneEnv(reward=0.5)
        terminate_after = 2
        original_step = env.step

        def step(full_completion_ids):
            prompt, reward, _, _, info = original_step(full_completion_ids)
            return prompt, reward, env.step_calls >= terminate_after, False, info

        env.step = step
        out = LLMAlgorithm.test(algo, env, loop=1)

        assert env.step_calls == terminate_after
        assert out.item() == pytest.approx(0.5)

    def test_subclasses_share_the_base_implementation(self):
        from agilerl.algorithms.grpo import GRPO
        from agilerl.algorithms.ppo_llm import PPO
        from agilerl.algorithms.reinforce_llm import REINFORCE

        assert GRPO.test is LLMAlgorithm.test
        assert PPO.test is LLMAlgorithm.test
        assert REINFORCE.test is LLMAlgorithm.test

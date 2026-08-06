# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from agilerl.algorithms.core import ActionResult
from agilerl.algorithms.grpo import GRPO
from agilerl.algorithms.ppo import PPO
from agilerl.llm_envs import (
    RolloutCollector,
    RolloutHarness,
)
from agilerl.rollouts.on_policy import (
    _collect_rollouts,
    collect_rollouts,
    collect_rollouts_llm,
    collect_rollouts_recurrent,
)
from tests.assets.tiny_tokenizer import TinyTokenizer
from tests.helpers.rollout_doubles import RolloutEnvDoubleMixin

if importlib.util.find_spec("deepspeed") and importlib.util.find_spec("vllm"):
    from tests.test_algorithms.test_llms.test_ppo_llm import _cpu_llmppo
    from tests.test_algorithms.test_llms.test_reinforce_llm import _cpu_llmreinforce
else:
    _cpu_llmppo = None
    _cpu_llmreinforce = None

_LLM_ROLLOUTS = pytest.mark.skipif(
    _cpu_llmppo is None or _cpu_llmreinforce is None,
    reason="LLM rollout tests require deepspeed and vllm.",
)


class _SingleTurnTextEnv:
    def __init__(self):
        self._done = False

    def reset(self, seed=None):
        del seed
        self._done = False
        return "solve 1 + 1", {}

    def step(self, action):
        del action
        self._done = True
        return "correct", 1.0, True, False, {}

    def close(self):
        return None


@_LLM_ROLLOUTS
class TestCollectRolloutsLlm:
    @pytest.mark.parametrize("hf_generate_chunk_size", [1, 2, 4])
    @pytest.mark.parametrize("algo_name", ["ppo", "reinforce"])
    def test_collect_rollouts_llm_hf_chunk_sizes_in_process(
        self, hf_generate_chunk_size: int, algo_name: str
    ):
        tokenizer = TinyTokenizer()

        def env_fn():
            return RolloutHarness.local(
                _SingleTurnTextEnv(),
                tokenizer=tokenizer,
                max_turns=1,
                apply_chat_template=False,
                max_model_len=128,
                max_output_tokens=8,
            )

        env = RolloutCollector(env_factory=env_fn, batch_size=2, group_size=1)
        if algo_name == "ppo":
            agent = _cpu_llmppo(
                use_vllm=False,
                hf_generate_chunk_size=hf_generate_chunk_size,
                max_model_len=128,
                max_output_tokens=8,
            )
        else:
            agent = _cpu_llmreinforce(
                use_vllm=False,
                hf_generate_chunk_size=hf_generate_chunk_size,
                max_model_len=128,
                max_output_tokens=8,
            )

        experiences, masks, turns, rewards, steps, next_group_seed, _sampling_logps = (
            collect_rollouts_llm(
                agent=agent,
                env=env,
                n_steps=1,
                batch_size=2,
                group_seed=0,
            )
        )
        assert len(experiences) == 2
        assert len(masks) == 2
        assert len(rewards) == 2
        assert len(turns) == 2
        assert steps > 0
        assert next_group_seed == 2

        agent.clean_up()
        env.close()


class TestCollectRolloutsLlmOrdering:
    """Pure-double ordering checks; needs neither deepspeed nor vllm."""

    def test_collect_rollouts_llm_preserves_batch_group_ordering_batch_size_4(
        self,
    ) -> None:
        """Preserve (batch_idx, group_idx) ordering across LLM rollout collection."""
        prompt_tokens_by_env_index = [42, 7, 99, 13, 55, 21, 88, 3]
        creation_idx = {"value": 0}

        class _OrderingEnv(RolloutEnvDoubleMixin):
            """Minimal env that records which completion token it receives."""

            dataset_size = 0

            def __init__(self, prompt_token: int) -> None:
                self.prompt_token = prompt_token
                self._seen_token: int | None = None
                self.turn_boundaries: list[tuple[int, int, int]] = []

            def reset(
                self,
                seed: int | None = None,
            ) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
                """Return a one-token prompt that uniquely marks this trajectory."""
                del seed
                self.done = False
                self.current_observation = {
                    "input_ids": torch.tensor([[self.prompt_token]], dtype=torch.long),
                    "attention_mask": torch.ones(1, 1, dtype=torch.long),
                }
                return self.current_observation, {}

            def step(
                self,
                full_completion: torch.Tensor,
                sampling_logps: torch.Tensor | None = None,
            ) -> tuple[dict[str, torch.Tensor], float, bool, bool, dict[str, object]]:
                """Record completion identity and terminate after one turn."""
                del sampling_logps
                self._seen_token = int(full_completion[0, 0].item())
                self.turn_boundaries = [(0, 1, 0)]
                self.done = True
                self.current_observation = {}
                return {}, 1.0, True, False, {}

            def get_episode_data(
                self,
            ) -> tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor | None,
            ]:
                """Expose marker token via episode ids and rewards for ordering checks."""
                token = self._seen_token if self._seen_token is not None else -1
                ep_ids = torch.tensor([[token, token + 1]], dtype=torch.long)
                action_mask = torch.tensor([[True]], dtype=torch.bool)
                turn_ids = torch.tensor([[0]], dtype=torch.long)
                rewards = torch.tensor([float(token)], dtype=torch.float32)
                return ep_ids, action_mask, turn_ids, rewards, None

            def close(self) -> None:
                """Provide a close method compatible with vector env cleanup."""
                return

        class _EchoAgent:
            """Echo prompt marker tokens into completions in row order."""

            def get_action(
                self,
                prompts: list[dict[str, torch.Tensor]],
                training: bool = True,
            ) -> ActionResult:
                """Return one completion per prompt row while preserving input order."""
                del training
                row_tokens = [
                    int(prompt["input_ids"][0, 0].item()) for prompt in prompts
                ]
                completions = [
                    torch.tensor([[int(tok), int(tok) + 1]], dtype=torch.long)
                    for tok in row_tokens
                ]
                return ActionResult(
                    token_ids=completions,
                    action_masks=None,
                    sampling_logps=None,
                )

        def env_fn() -> _OrderingEnv:
            """Create deterministic env instances in construction order."""
            idx = creation_idx["value"]
            creation_idx["value"] += 1
            return _OrderingEnv(prompt_tokens_by_env_index[idx])

        env = RolloutCollector(env_factory=env_fn, batch_size=4, group_size=2)
        agent = _EchoAgent()

        token_ids_list, _masks, _turns, rewards, steps, next_group_seed, _logps = (
            collect_rollouts_llm(
                agent=agent,
                env=env,
                n_steps=1,
                batch_size=4,
                group_seed=123,
            )
        )

        returned_first_tokens = [int(ids[0, 0].item()) for ids in token_ids_list]
        assert returned_first_tokens == prompt_tokens_by_env_index

        returned_reward_markers = [int(r[0].item()) for r in rewards]
        assert returned_reward_markers == prompt_tokens_by_env_index

        assert steps == 8
        assert next_group_seed == 127

        env.close()


class TestCollectRolloutsLlmGrpo:
    def test_collect_rollouts_llm_grpo_uses_repeat_prompts_false(self):
        """GRPO must not repeat prompts during on-policy LLM rollout collection."""
        tokenizer = TinyTokenizer()

        def env_fn():
            return RolloutHarness.local(
                _SingleTurnTextEnv(),
                tokenizer=tokenizer,
                max_turns=1,
                apply_chat_template=False,
                max_model_len=128,
                max_output_tokens=8,
            )

        env = RolloutCollector(env_factory=env_fn, batch_size=1, group_size=1)
        agent = MagicMock()
        agent.__class__ = GRPO
        agent.accelerator = None
        agent.get_action.return_value = ActionResult(
            token_ids=[torch.tensor([[1, 2]], dtype=torch.long)],
            action_masks=None,
            sampling_logps=None,
        )

        collect_rollouts_llm(
            agent=agent,
            env=env,
            n_steps=1,
            batch_size=1,
            group_seed=0,
        )

        agent.get_action.assert_called_once()
        call_kwargs = agent.get_action.call_args.kwargs
        assert call_kwargs["repeat_prompts"] is False
        assert call_kwargs["training"] is True
        env.close()

    def test_early_exit_when_local_trajectories_done(self):
        """Ranks stop calling get_action once local prompts are None."""
        env = MagicMock()
        prompt = {"input_ids": torch.tensor([[1, 2]])}
        env.reset.return_value = prompt
        env.step.return_value = None
        env.get_trajectories.return_value = (
            [torch.ones(1, 2, dtype=torch.long)],
            [torch.ones(1, 1, dtype=torch.bool)],
            [torch.zeros(1, 1, dtype=torch.long)],
            [torch.ones(1, dtype=torch.float32)],
            1,
            None,
        )

        agent = MagicMock()
        agent.__class__ = GRPO
        agent.accelerator = None
        agent.get_action.return_value = ActionResult(
            token_ids=[torch.tensor([[1, 2]], dtype=torch.long)],
            action_masks=None,
            sampling_logps=None,
        )

        collect_rollouts_llm(
            agent=agent,
            env=env,
            n_steps=3,
            batch_size=1,
            group_seed=0,
        )

        assert agent.get_action.call_count == 1
        assert env.step.call_count == 1

    def test_waits_for_everyone_when_accelerator_present(self):
        env = MagicMock()
        env.reset.return_value = None
        env.get_trajectories.return_value = (
            [],
            [],
            [],
            [],
            0,
            None,
        )
        agent = MagicMock()
        agent.__class__ = GRPO
        acc = MagicMock()
        agent.accelerator = acc

        collect_rollouts_llm(
            agent=agent,
            env=env,
            n_steps=2,
            batch_size=1,
            group_seed=0,
        )

        agent.get_action.assert_not_called()
        acc.wait_for_everyone.assert_called_once()

    def test_non_grpo_get_action_signature(self):
        """Non-GRPO agents use get_action(prompts, training=True)."""
        from agilerl.algorithms.ppo_llm import PPO as LLMPPO

        env = MagicMock()
        prompt = {"input_ids": torch.tensor([[1, 2]])}
        env.reset.return_value = prompt
        env.step.return_value = None
        env.get_trajectories.return_value = (
            [torch.ones(1, 2, dtype=torch.long)],
            [torch.ones(1, 1, dtype=torch.bool)],
            [torch.zeros(1, 1, dtype=torch.long)],
            [torch.ones(1, dtype=torch.float32)],
            1,
            None,
        )

        agent = MagicMock()
        agent.__class__ = LLMPPO
        agent.accelerator = None
        agent.get_action.return_value = ActionResult(
            token_ids=[torch.tensor([[1, 2]], dtype=torch.long)],
            action_masks=None,
            sampling_logps=None,
        )

        collect_rollouts_llm(
            agent=agent,
            env=env,
            n_steps=1,
            batch_size=1,
            group_seed=0,
        )

        agent.get_action.assert_called_once()
        assert agent.get_action.call_args.kwargs.get("training") is True
        assert "repeat_prompts" not in agent.get_action.call_args.kwargs


class DummyEnv:
    def __init__(self, state_size, vect=True, num_envs=2):
        self.state_size = state_size
        self.vect = vect
        self.num_envs = num_envs
        if self.vect:
            self.state_size = (num_envs, *self.state_size)
            self.n_envs = num_envs
        else:
            self.n_envs = 1

    def reset(self):
        return np.random.rand(*self.state_size), {}

    def step(self, action):
        return (
            np.random.rand(*self.state_size),
            np.random.randint(0, 5, self.n_envs),
            np.random.randint(0, 2, self.n_envs),
            np.random.randint(0, 2, self.n_envs),
            {},
        )


class TerminatingVecEnv:
    """Vector env where env 0 terminates on the first step; env 1 does not."""

    def __init__(self, state_size, num_envs=2):
        self.state_size = (num_envs, *state_size)
        self.n_envs = num_envs
        self.num_envs = num_envs
        self._step_count = 0

    def reset(self):
        self._step_count = 0
        return np.random.rand(*self.state_size), {}

    def step(self, action):
        del action
        self._step_count += 1
        terminated = np.array(
            [self._step_count == 1, False][: self.num_envs],
            dtype=np.int64,
        )
        return (
            np.random.rand(*self.state_size),
            np.ones(self.num_envs, dtype=np.float32),
            terminated,
            np.zeros(self.num_envs, dtype=np.int64),
            {},
        )


class TestCollectRollouts:
    def test_collect_rollouts_returns_scores(self, vector_space, discrete_space):
        ppo = PPO(
            observation_space=vector_space,
            action_space=discrete_space,
            learn_step=4,
            num_envs=1,
        )
        env = DummyEnv(state_size=vector_space.shape, vect=True, num_envs=1)
        result = collect_rollouts(ppo, env, n_steps=4)
        assert isinstance(result, tuple)
        assert len(result) == 5
        assert isinstance(result[0], list)
        ppo.clean_up()

    def test_collect_rollouts_n_steps_default(self, vector_space, discrete_space):
        ppo = PPO(
            observation_space=vector_space,
            action_space=discrete_space,
            learn_step=8,
            num_envs=1,
        )
        env = DummyEnv(state_size=vector_space.shape, vect=True, num_envs=1)
        result = collect_rollouts(ppo, env)
        assert isinstance(result, tuple)
        assert isinstance(result[0], list)
        ppo.clean_up()

    def test_collect_rollouts_recurrent_returns_scores(
        self, vector_space, discrete_space
    ):
        ppo = PPO(
            observation_space=vector_space,
            action_space=discrete_space,
            learn_step=4,
            num_envs=1,
            recurrent=True,
        )
        env = DummyEnv(state_size=vector_space.shape, vect=True, num_envs=1)
        result = collect_rollouts_recurrent(ppo, env, n_steps=4)
        assert isinstance(result, tuple)
        assert len(result) == 5
        assert isinstance(result[0], list)
        ppo.clean_up()

    def test_recurrent_rollout_resets_hidden_state_on_episode_end(
        self, vector_space, discrete_space
    ):
        """Finished env slots copy fresh initial hidden states mid-rollout."""
        num_envs = 2
        ppo = PPO(
            observation_space=vector_space,
            action_space=discrete_space,
            learn_step=4,
            num_envs=num_envs,
            recurrent=True,
        )
        env = TerminatingVecEnv(state_size=vector_space.shape, num_envs=num_envs)
        original_get_initial = ppo.get_initial_hidden_state
        reset_call_count = {"n": 0}

        def get_initial_with_reset_marker(batch_size: int):
            hidden = original_get_initial(batch_size)
            reset_call_count["n"] += 1
            if reset_call_count["n"] > 1:
                for tensor in hidden.values():
                    tensor.fill_(42.0)
            return hidden

        with patch.object(
            ppo,
            "get_initial_hidden_state",
            side_effect=get_initial_with_reset_marker,
        ):
            _collect_rollouts(ppo, env, n_steps=1, recurrent=True)

        assert reset_call_count["n"] >= 2
        assert isinstance(ppo.hidden_state, dict)
        for key in ppo.hidden_state:
            assert torch.all(ppo.hidden_state[key][:, 0, :] == 42.0)
            assert not torch.all(ppo.hidden_state[key][:, 1, :] == 42.0)
        ppo.clean_up()


class Test_CollectRollouts:
    def test_collect_rollouts_warm_start_with_last_obs(
        self, vector_space, discrete_space
    ):
        ppo = PPO(
            observation_space=vector_space,
            action_space=discrete_space,
            learn_step=4,
            num_envs=1,
        )
        env = DummyEnv(state_size=vector_space.shape, vect=True, num_envs=1)
        obs, info = env.reset()
        result = _collect_rollouts(
            ppo,
            env,
            n_steps=2,
            last_obs=obs,
            last_done=np.zeros(1),
            last_scores=np.zeros(1),
            last_info=info,
            recurrent=False,
        )
        assert isinstance(result, tuple)
        assert len(result) == 5
        ppo.clean_up()

    @pytest.mark.parametrize("recurrent", [False, True])
    def test_collect_rollouts_recurrent_warm_start(
        self, vector_space, discrete_space, recurrent
    ):
        ppo = PPO(
            observation_space=vector_space,
            action_space=discrete_space,
            learn_step=4,
            num_envs=1,
            recurrent=recurrent,
        )
        env = DummyEnv(state_size=vector_space.shape, vect=True, num_envs=1)
        obs, info = env.reset()
        if recurrent:
            ppo.hidden_state = ppo.get_initial_hidden_state(1)
        result = _collect_rollouts(
            ppo,
            env,
            n_steps=2,
            last_obs=obs,
            last_done=np.zeros(1),
            last_scores=np.zeros(1),
            last_info=info,
            recurrent=recurrent,
        )
        assert isinstance(result, tuple)
        assert len(result) == 5
        ppo.clean_up()

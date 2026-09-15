# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import gymnasium as gym
import numpy as np

from agilerl.training.train_bandits import train_bandits
from agilerl.training.train_multi_agent_off_policy import train_multi_agent_off_policy
from agilerl.training.train_multi_agent_on_policy import train_multi_agent_on_policy
from agilerl.training.train_off_policy import train_off_policy
from agilerl.training.train_offline import train_offline
from agilerl.training.train_on_policy import train_on_policy
from agilerl.vector.pz_vec_env import PettingZooVecEnv


def _mock_vec_env() -> MagicMock:
    env = MagicMock(spec=gym.vector.VectorEnv)
    env.num_envs = 1
    obs = np.zeros((1, 4), dtype=np.float32)
    env.reset.return_value = (obs, {})
    env.step.return_value = (
        obs,
        np.array([0.0]),
        np.array([True]),
        np.array([False]),
        {},
    )
    return env


def _mock_population(pop: list[MagicMock]) -> MagicMock:
    population = MagicMock()
    population.agents = pop
    population.all_below.side_effect = [True, False]
    population.should_stop.return_value = False
    population.last_scalar_fitnesses = [1.0, 2.0]
    return population


def _idle_population(pop: list[MagicMock]) -> MagicMock:
    population = _mock_population(pop)
    population.all_below.side_effect = None
    population.all_below.return_value = False
    return population


def test_train_on_policy_closes_env_on_teardown():
    vec_env = _mock_vec_env()
    pop = [
        MagicMock(learn_step=4, metrics=MagicMock(steps=0), recurrent=False)
        for _ in range(2)
    ]

    with (
        patch(
            "agilerl.training.train_on_policy.default_progress_bar",
            return_value=MagicMock(),
        ),
        patch("agilerl.training.train_on_policy.init_loggers", return_value=[]),
        patch(
            "agilerl.training.train_on_policy.Population",
            return_value=_mock_population(pop),
        ),
        patch(
            "agilerl.training.train_on_policy.collect_rollouts",
            return_value=([], None, None, None, None),
        ),
    ):
        train_on_policy(
            vec_env,
            "env",
            "algo",
            pop,
            init_hp=None,
            max_steps=4,
            evo_steps=4,
            eval_loop=1,
            wb=False,
            verbose=False,
            accelerator=None,
        )

    vec_env.close.assert_called_once()


def test_train_off_policy_closes_env_on_teardown():
    vec_env = _mock_vec_env()
    pop = [
        MagicMock(learn_step=1, metrics=MagicMock(steps=0), batch_size=1)
        for _ in range(2)
    ]
    memory = MagicMock()
    memory.__len__ = MagicMock(return_value=10)
    memory.size = 10

    with (
        patch(
            "agilerl.training.train_off_policy.default_progress_bar",
            return_value=MagicMock(),
        ),
        patch("agilerl.training.train_off_policy.init_loggers", return_value=[]),
        patch(
            "agilerl.training.train_off_policy.Population",
            return_value=_idle_population(pop),
        ),
    ):
        train_off_policy(
            vec_env,
            "env",
            "algo",
            pop,
            memory,
            init_hp=None,
            max_steps=4,
            evo_steps=4,
            eval_loop=1,
            wb=False,
            verbose=False,
            accelerator=None,
        )

    vec_env.close.assert_called_once()


def test_train_bandits_closes_env_on_teardown():
    env = MagicMock()
    env.reset.return_value = np.zeros(4)
    env.step.return_value = (np.zeros(4), 1.0)
    pop = [
        MagicMock(
            learn_step=1,
            batch_size=1,
            metrics=MagicMock(steps=0),
            regret=[0.0],
        )
        for _ in range(2)
    ]
    memory = MagicMock()
    memory.__len__ = MagicMock(return_value=10)
    sampler = MagicMock()
    sampler.sample = MagicMock(return_value=MagicMock())

    with (
        patch(
            "agilerl.training.train_bandits.default_progress_bar",
            return_value=MagicMock(),
        ),
        patch("agilerl.training.train_bandits.init_loggers", return_value=[]),
        patch("agilerl.training.train_bandits.Sampler", return_value=sampler),
        patch(
            "agilerl.training.train_bandits.Population",
            return_value=_mock_population(pop),
        ),
    ):
        train_bandits(
            env,
            "env",
            "algo",
            pop,
            memory,
            init_hp=None,
            max_steps=1,
            evo_steps=1,
            episode_steps=1,
            eval_loop=1,
            wb=False,
            verbose=False,
            accelerator=None,
        )

    env.close.assert_called_once()


def test_train_offline_closes_env_on_teardown(tmp_path):
    vec_env = _mock_vec_env()
    pop = [MagicMock(batch_size=1, metrics=MagicMock(steps=0))]
    memory = MagicMock()
    memory.add = MagicMock()

    mock_dataset = MagicMock()
    mock_dataset.__getitem__ = MagicMock(
        side_effect=lambda key: {
            "rewards": np.zeros(2),
            "observations": np.zeros((2, 4)),
            "actions": np.zeros(2),
            "terminals": np.zeros(2, dtype=bool),
        }[key]
    )

    with (
        patch(
            "agilerl.training.train_offline.default_progress_bar",
            return_value=MagicMock(),
        ),
        patch("agilerl.training.train_offline.init_loggers", return_value=[]),
        patch(
            "agilerl.training.train_offline.Population",
            return_value=_mock_population(pop),
        ),
        patch("agilerl.training.train_offline.Sampler", return_value=MagicMock()),
    ):
        train_offline(
            vec_env,
            "env",
            "algo",
            pop,
            memory,
            dataset=mock_dataset,
            init_hp=None,
            max_steps=1,
            evo_steps=1,
            eval_loop=1,
            wb=False,
            verbose=False,
            accelerator=None,
        )

    mock_dataset.close.assert_called_once()
    vec_env.close.assert_called_once()


def test_train_multi_agent_on_policy_closes_env_on_teardown():
    vec_env = MagicMock(spec=PettingZooVecEnv)
    vec_env.num_envs = 1
    pop = [MagicMock(learn_step=1, metrics=MagicMock(steps=0))]

    with (
        patch(
            "agilerl.training.train_multi_agent_on_policy.default_progress_bar",
            return_value=MagicMock(),
        ),
        patch(
            "agilerl.training.train_multi_agent_on_policy.init_loggers", return_value=[]
        ),
        patch(
            "agilerl.training.train_multi_agent_on_policy.Population",
            return_value=_idle_population(pop),
        ),
    ):
        train_multi_agent_on_policy(
            vec_env,
            "env",
            "algo",
            pop,
            init_hp=None,
            max_steps=4,
            evo_steps=4,
            eval_loop=1,
            wb=False,
            verbose=False,
            accelerator=None,
        )

    vec_env.close.assert_called_once()


def test_train_multi_agent_off_policy_closes_env_on_teardown():
    vec_env = MagicMock(spec=PettingZooVecEnv)
    vec_env.num_envs = 1
    pop = [MagicMock(learn_step=1, batch_size=1, metrics=MagicMock(steps=0))]
    memory = MagicMock()
    memory.__len__ = MagicMock(return_value=10)

    with (
        patch(
            "agilerl.training.train_multi_agent_off_policy.default_progress_bar",
            return_value=MagicMock(),
        ),
        patch(
            "agilerl.training.train_multi_agent_off_policy.init_loggers",
            return_value=[],
        ),
        patch(
            "agilerl.training.train_multi_agent_off_policy.Population",
            return_value=_idle_population(pop),
        ),
        patch(
            "agilerl.training.train_multi_agent_off_policy.Sampler",
            return_value=MagicMock(),
        ),
    ):
        train_multi_agent_off_policy(
            vec_env,
            "env",
            "algo",
            pop,
            memory,
            init_hp=None,
            max_steps=4,
            evo_steps=4,
            eval_loop=1,
            wb=False,
            verbose=False,
            accelerator=None,
        )

    vec_env.close.assert_called_once()

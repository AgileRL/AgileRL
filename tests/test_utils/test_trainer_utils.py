# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for agilerl/utils/trainer_utils.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestHpConfigFromMutationSpec:
    def test_returns_none_when_empty(self):
        from agilerl.models.hpo import MutationSpec
        from agilerl.utils.trainer_utils import hp_config_from_mutation_spec

        spec = MutationSpec(rl_hp_selection={})
        assert hp_config_from_mutation_spec(spec) is None

    def test_returns_hyperparameter_config_from_selection(self):
        from agilerl.algorithms.core.registry import HyperparameterConfig, RLParameter
        from agilerl.models.hpo import MutationSpec, RLHyperparameter
        from agilerl.utils.trainer_utils import hp_config_from_mutation_spec

        spec = MutationSpec(
            rl_hp_selection={
                "lr": RLHyperparameter(min=1e-4, max=1e-2, grow_factor=1.2),
            },
        )
        config = hp_config_from_mutation_spec(spec)
        assert isinstance(config, HyperparameterConfig)
        assert isinstance(config.lr, RLParameter)
        assert config.lr.min == 1e-4
        assert config.lr.max == 1e-2


class TestGetSpacesFromEnv:
    def test_fallback_raises_for_unsupported_type(self):
        from agilerl.utils.trainer_utils import get_spaces_from_env

        with pytest.raises(NotImplementedError, match="not supported"):
            get_spaces_from_env("not_a_spec", MagicMock())


class TestRainbowNStepOverride:
    def test_n_step_overridden_from_replay_buffer_spec(self):
        from agilerl.models.algorithms.rainbow_dqn import RainbowDQNSpec
        from agilerl.models.training import NStepBufferArgs, ReplayBufferSpec
        from agilerl.utils.trainer_utils import create_population_from_spec

        algo = RainbowDQNSpec(net_config=None)
        buf_spec = ReplayBufferSpec(
            n_step_buffer=True, n_step_buffer_args=NStepBufferArgs(n_step=5)
        )
        env = MagicMock()
        env.single_observation_space = MagicMock()
        env.single_action_space = MagicMock()
        env.num_envs = 1

        with patch.object(RainbowDQNSpec, "build_algorithm", return_value=MagicMock()):
            create_population_from_spec(
                population_size=1,
                algo_spec=algo,
                env=env,
                mutation_spec=None,
                replay_buffer_spec=buf_spec,
            )
        assert algo.n_step == 5


class TestCreatePopulationLLM:
    def test_clones_actor_for_subsequent_agents(self):
        from agilerl.utils.trainer_utils import create_population_from_spec

        mock_agent0 = MagicMock()
        mock_agent0.actor = MagicMock()
        mock_agent0.actor.state_dict.return_value = {}

        mock_agent1 = MagicMock()

        call_count = {"n": 0}

        def _build_side_effect(**kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return mock_agent0
            return mock_agent1

        algo = MagicMock()
        algo.build_algorithm = MagicMock(side_effect=_build_side_effect)
        algo.zero_stage = 0

        with (
            patch("agilerl.utils.algo_utils.clone_llm", return_value=MagicMock()),
            patch("agilerl.utils.llm_utils.get_state_dict", return_value={}),
        ):
            pop = create_population_from_spec(
                population_size=2,
                algo_spec=algo,
                env=MagicMock(num_envs=1),
                mutation_spec=None,
                replay_buffer_spec=None,
                accelerator=None,
                tokenizer=MagicMock(),
            )
        assert len(pop) == 2

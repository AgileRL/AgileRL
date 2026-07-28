"""Tests for agilerl/utils/trainer_utils.py."""

from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock, patch

import pytest


class TestAutoTokenizerGuard:
    def test_auto_tokenizer_attr_exists(self):
        from agilerl.utils import trainer_utils

        assert hasattr(trainer_utils, "AutoTokenizer")


def test_trainer_utils_fallback_auto_tokenizer_when_no_llm_dependencies():
    """Test that trainer_utils sets AutoTokenizer to None when HAS_LLM_DEPENDENCIES is False."""
    original_module = sys.modules.pop("agilerl.utils.trainer_utils", None)

    try:
        with patch("agilerl.HAS_LLM_DEPENDENCIES", False):
            trainer_utils_reloaded = importlib.import_module(
                "agilerl.utils.trainer_utils"
            )

            assert trainer_utils_reloaded.AutoTokenizer is None
    finally:
        import agilerl.utils as _utils_pkg

        if original_module is not None:
            sys.modules["agilerl.utils.trainer_utils"] = original_module
            _utils_pkg.trainer_utils = original_module
        else:
            sys.modules.pop("agilerl.utils.trainer_utils", None)
            _utils_pkg.trainer_utils = importlib.import_module(
                "agilerl.utils.trainer_utils"
            )


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


class TestBuildMultiFrequencyFromSpec:
    def test_returns_none_when_spec_is_none(self):
        from agilerl.models.training import TrainingSpec
        from agilerl.utils.trainer_utils import (
            build_multi_frequency_selection_from_spec,
        )

        assert build_multi_frequency_selection_from_spec(None, TrainingSpec()) is None

    def test_builds_strategy_and_forwards_seed(self):
        from agilerl.hpo.multi_frequency import MultiFrequencySelection
        from agilerl.models.hpo import MultiFrequencySelectionSpec
        from agilerl.models.training import TrainingSpec
        from agilerl.utils.trainer_utils import (
            build_multi_frequency_selection_from_spec,
        )

        spec = MultiFrequencySelectionSpec(
            n_subpopulations=2,
            evolution_frequency_ratios=[1, 2],
            n_winners=1,
            n_survivors=1,
            n_open_for_migration=1,
            n_losers=1,
        )
        strategy = build_multi_frequency_selection_from_spec(
            spec, TrainingSpec(pop_size=8), seed=123
        )

        assert isinstance(strategy, MultiFrequencySelection)
        assert strategy.population_size == 8
        assert strategy.n_subpopulations == 2
        assert strategy.deltas == [1, 2]
        assert strategy.bracket_sizes == (1, 1, 1, 1)

        seeded = build_multi_frequency_selection_from_spec(
            spec, TrainingSpec(pop_size=8), seed=123
        )
        assert strategy.rng.integers(1_000_000) == seeded.rng.integers(1_000_000)


class TestAssignSubpopulations:
    def test_tags_agents_by_contiguous_index_blocks(self):
        from agilerl.models.hpo import MultiFrequencySelectionSpec
        from agilerl.utils.trainer_utils import _assign_subpopulations

        agents = [MagicMock(index=i, subpopulation_id=None) for i in range(8)]
        spec = MultiFrequencySelectionSpec(n_subpopulations=2)
        _assign_subpopulations(agents, spec)

        assert [a.subpopulation_id for a in agents] == [0, 0, 0, 0, 1, 1, 1, 1]

    def test_noop_when_spec_is_none(self):
        from agilerl.utils.trainer_utils import _assign_subpopulations

        agents = [MagicMock(index=i, subpopulation_id=None) for i in range(4)]
        _assign_subpopulations(agents, None)

        assert all(a.subpopulation_id is None for a in agents)

    def test_tags_by_slot_not_restored_index_on_resume(self):
        """Resume must derive the layout from the population slot.

        On resume_from_checkpoint every agent is rebuilt in slot order but
        restores its own persisted index and may exceed pop_size.  So
        the tag must come from the enumeration slot and overwrite any stale
        restored subpopulation.
        """
        from agilerl.models.hpo import MultiFrequencySelectionSpec
        from agilerl.utils.trainer_utils import _assign_subpopulations

        restored_indices = [12, 5, 40, 3, 27, 9, 33, 18]
        agents = [MagicMock(index=idx, subpopulation_id=99) for idx in restored_indices]
        spec = MultiFrequencySelectionSpec(n_subpopulations=2)
        _assign_subpopulations(agents, spec)

        assert [a.subpopulation_id for a in agents] == [0, 0, 0, 0, 1, 1, 1, 1]

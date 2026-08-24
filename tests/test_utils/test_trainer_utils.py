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


class TestBuildMutations:
    def test_none_returns_none(self):
        from agilerl.utils.trainer_utils import build_mutations_from_spec

        assert build_mutations_from_spec(None, "cpu") is None

    def test_from_spec(self):
        from agilerl.hpo.mutation import Mutations
        from agilerl.models.hpo import MutationProbabilities, MutationSpec
        from agilerl.utils.trainer_utils import build_mutations_from_spec

        spec = MutationSpec(
            probabilities=MutationProbabilities(
                no_mut=0.5, params_mut=0.3, rl_hp_mut=0.2
            ),
            mutation_sd=0.05,
        )

        result = build_mutations_from_spec(spec, "cpu")

        assert isinstance(result, Mutations)
        assert result.no_mut == 0.5
        assert result.parameters_mut == 0.3
        assert result.mutation_sd == 0.05

    def test_regrama_fields_reach_the_operator(self):
        from agilerl.models.hpo import MutationSpec
        from agilerl.utils.trainer_utils import build_mutations_from_spec

        spec = MutationSpec(dormant_threshold=0.05)

        result = build_mutations_from_spec(spec, "cpu")

        assert result.dormant_threshold == 0.05

    def test_regrama_defaults_reach_the_operator(self):
        from agilerl.models.hpo import MutationSpec
        from agilerl.utils.trainer_utils import build_mutations_from_spec

        result = build_mutations_from_spec(MutationSpec(), "cpu")

        assert result.dormant_threshold == 0.01


class TestBuildTournament:
    def test_none_returns_none(self):
        from agilerl.models.training import TrainingSpec
        from agilerl.utils.trainer_utils import build_tournament_from_spec

        assert build_tournament_from_spec(None, TrainingSpec()) is None

    def test_from_spec(self):
        from agilerl.hpo.tournament import TournamentSelection
        from agilerl.models.hpo import TournamentSelectionSpec
        from agilerl.models.training import TrainingSpec
        from agilerl.utils.trainer_utils import build_tournament_from_spec

        result = build_tournament_from_spec(
            TournamentSelectionSpec(tournament_size=3, elitism=True),
            TrainingSpec(pop_size=8),
        )

        assert isinstance(result, TournamentSelection)
        assert result.tournament_size == 3
        assert result.elitism is True

    def test_returns_none_for_a_multi_frequency_spec(self):
        """The builder accepts the union, so the other regime is simply not built."""
        from agilerl.models.hpo import MultiFrequencySelectionSpec
        from agilerl.models.training import TrainingSpec
        from agilerl.utils.trainer_utils import build_tournament_from_spec

        assert (
            build_tournament_from_spec(
                MultiFrequencySelectionSpec(), TrainingSpec(pop_size=8)
            )
            is None
        )


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

    def test_returns_none_for_a_tournament_spec(self):
        from agilerl.models.hpo import TournamentSelectionSpec
        from agilerl.models.training import TrainingSpec
        from agilerl.utils.trainer_utils import (
            build_multi_frequency_selection_from_spec,
        )

        assert (
            build_multi_frequency_selection_from_spec(
                TournamentSelectionSpec(), TrainingSpec(pop_size=8)
            )
            is None
        )


class TestBuildSelectionFromSpec:
    """The dispatcher branches the union onto the two concrete operators."""

    def test_returns_none_when_spec_is_none(self):
        from agilerl.models.training import TrainingSpec
        from agilerl.utils.trainer_utils import build_selection_from_spec

        assert build_selection_from_spec(None, TrainingSpec()) is None

    def test_builds_tournament_from_a_tournament_spec(self):
        from agilerl.hpo.tournament import TournamentSelection
        from agilerl.models.hpo import TournamentSelectionSpec
        from agilerl.models.training import TrainingSpec
        from agilerl.utils.trainer_utils import build_selection_from_spec

        strategy = build_selection_from_spec(
            TournamentSelectionSpec(tournament_size=3, elitism=False),
            TrainingSpec(pop_size=8),
        )

        assert isinstance(strategy, TournamentSelection)
        assert strategy.tournament_size == 3
        assert strategy.elitism is False
        assert strategy.population_size == 8

    def test_builds_multi_frequency_from_a_multi_frequency_spec(self):
        from agilerl.hpo.multi_frequency import MultiFrequencySelection
        from agilerl.models.hpo import MultiFrequencySelectionSpec
        from agilerl.models.training import TrainingSpec
        from agilerl.utils.trainer_utils import build_selection_from_spec

        strategy = build_selection_from_spec(
            MultiFrequencySelectionSpec(
                n_subpopulations=2, evolution_frequency_ratios=[1, 2]
            ),
            TrainingSpec(pop_size=8),
        )

        assert isinstance(strategy, MultiFrequencySelection)
        assert strategy.n_subpopulations == 2
        assert strategy.deltas == [1, 2]
        assert strategy.bracket_sizes == (1, 0, 1, 2)

    def test_forwards_seed_to_multi_frequency(self):
        from agilerl.models.hpo import MultiFrequencySelectionSpec
        from agilerl.models.training import TrainingSpec
        from agilerl.utils.trainer_utils import build_selection_from_spec

        spec = MultiFrequencySelectionSpec(n_subpopulations=2)
        first = build_selection_from_spec(spec, TrainingSpec(pop_size=8), seed=7)
        second = build_selection_from_spec(spec, TrainingSpec(pop_size=8), seed=7)

        assert first.rng.integers(1_000_000) == second.rng.integers(1_000_000)

    def test_rejects_an_unknown_spec_type(self):
        from agilerl.models.training import TrainingSpec
        from agilerl.utils.trainer_utils import build_selection_from_spec

        with pytest.raises(TypeError, match="TournamentSelectionSpec"):
            build_selection_from_spec(object(), TrainingSpec(pop_size=8))


class TestResolveDeprecatedSelectionKwargs:
    """Spec-level backward compatibility for the superseded selection keywords."""

    def test_returns_the_spec_unchanged_when_no_deprecated_kwarg(self):
        from agilerl.models.hpo import TournamentSelectionSpec
        from agilerl.utils.trainer_utils import resolve_deprecated_selection_kwargs

        spec = TournamentSelectionSpec(tournament_size=3)

        resolved = resolve_deprecated_selection_kwargs(spec, {}, caller="Trainer")

        assert resolved is spec

    def test_returns_none_when_neither_spelling_supplies_a_spec(self):
        from agilerl.utils.trainer_utils import resolve_deprecated_selection_kwargs

        assert resolve_deprecated_selection_kwargs(None, {}, caller="Trainer") is None

    def test_folds_the_deprecated_kwarg_and_warns(self):
        from agilerl.models.hpo import TournamentSelectionSpec
        from agilerl.utils.trainer_utils import resolve_deprecated_selection_kwargs

        spec = TournamentSelectionSpec(tournament_size=4)

        with pytest.warns(DeprecationWarning, match="'tournament' argument to Trainer"):
            resolved = resolve_deprecated_selection_kwargs(
                None, {"tournament": spec}, caller="Trainer"
            )

        assert resolved is spec

    def test_honours_a_custom_deprecated_key(self):
        from agilerl.models.hpo import MultiFrequencySelectionSpec
        from agilerl.utils.trainer_utils import resolve_deprecated_selection_kwargs

        spec = MultiFrequencySelectionSpec(n_subpopulations=2)

        with pytest.warns(DeprecationWarning, match="'tournament_selection' argument"):
            resolved = resolve_deprecated_selection_kwargs(
                None,
                {"tournament_selection": spec},
                deprecated_key="tournament_selection",
                caller="TrainingManifest.from_trainer_specs",
            )

        assert resolved is spec

    def test_deprecated_kwarg_set_to_none_falls_back_to_the_new_spelling(self):
        from agilerl.models.hpo import TournamentSelectionSpec
        from agilerl.utils.trainer_utils import resolve_deprecated_selection_kwargs

        spec = TournamentSelectionSpec(tournament_size=5)

        with pytest.warns(DeprecationWarning, match="'tournament' argument to Trainer"):
            resolved = resolve_deprecated_selection_kwargs(
                spec, {"tournament": None}, caller="Trainer"
            )

        assert resolved is spec

    def test_equal_but_distinct_specs_route_cleanly(self):
        # Pydantic equality compares field values, so passing the same configuration
        # under both spellings is not a conflict
        from agilerl.models.hpo import TournamentSelectionSpec
        from agilerl.utils.trainer_utils import resolve_deprecated_selection_kwargs

        spec = TournamentSelectionSpec(tournament_size=3)
        twin = TournamentSelectionSpec(tournament_size=3)

        with pytest.warns(DeprecationWarning, match="'tournament' argument to Trainer"):
            resolved = resolve_deprecated_selection_kwargs(
                spec, {"tournament": twin}, caller="Trainer"
            )

        assert resolved is spec

    def test_rejects_a_stray_keyword_argument(self):
        from agilerl.utils.trainer_utils import resolve_deprecated_selection_kwargs

        with pytest.raises(
            TypeError, match=r"unexpected keyword argument\(s\): 'typo'"
        ):
            resolve_deprecated_selection_kwargs(None, {"typo": 1}, caller="Trainer")

    def test_rejects_conflicting_specs_from_the_two_spellings(self):
        from agilerl.models.hpo import (
            MultiFrequencySelectionSpec,
            TournamentSelectionSpec,
        )
        from agilerl.utils.trainer_utils import resolve_deprecated_selection_kwargs

        with (
            pytest.warns(DeprecationWarning, match="'tournament' argument to Trainer"),
            pytest.raises(ValueError, match="conflicting selection strategies"),
        ):
            resolve_deprecated_selection_kwargs(
                MultiFrequencySelectionSpec(n_subpopulations=2),
                {"tournament": TournamentSelectionSpec(tournament_size=3)},
                caller="Trainer",
            )


class TestBuildReplayBuffer:
    def test_none_with_on_policy_returns_none(self):
        from agilerl.models import PPOSpec
        from agilerl.utils.trainer_utils import build_replay_buffer_from_spec

        assert build_replay_buffer_from_spec(PPOSpec(), None, "cpu") is None

    def test_none_with_off_policy_creates_default(self):
        from agilerl.components.replay_buffer import ReplayBuffer
        from agilerl.models import DQNSpec
        from agilerl.utils.trainer_utils import build_replay_buffer_from_spec

        result = build_replay_buffer_from_spec(DQNSpec(), None, "cpu")

        assert isinstance(result, ReplayBuffer)
        assert result.max_size == 100_000

    def test_from_spec_standard(self):
        from agilerl.components.replay_buffer import ReplayBuffer
        from agilerl.models import DQNSpec
        from agilerl.models.training import ReplayBufferSpec
        from agilerl.utils.trainer_utils import build_replay_buffer_from_spec

        result = build_replay_buffer_from_spec(
            DQNSpec(), ReplayBufferSpec(memory_size=5_000), "cpu"
        )

        assert isinstance(result, ReplayBuffer)
        assert result.max_size == 5_000

    def test_from_spec_n_step(self):
        from agilerl.components.replay_buffer import MultiStepReplayBuffer
        from agilerl.models import RainbowDQNSpec
        from agilerl.models.training import ReplayBufferSpec
        from agilerl.utils.trainer_utils import build_replay_buffer_from_spec

        result = build_replay_buffer_from_spec(
            RainbowDQNSpec(),
            ReplayBufferSpec(memory_size=10_000, n_step_buffer=True),
            "cpu",
        )

        assert isinstance(result, MultiStepReplayBuffer)
        assert result.max_size == 10_000


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

    def test_noop_under_a_tournament_spec(self):
        """Subpopulations only exist in the MF-PBT regime."""
        from agilerl.models.hpo import TournamentSelectionSpec
        from agilerl.utils.trainer_utils import _assign_subpopulations

        agents = [MagicMock(index=i, subpopulation_id=None) for i in range(8)]
        _assign_subpopulations(agents, TournamentSelectionSpec())

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

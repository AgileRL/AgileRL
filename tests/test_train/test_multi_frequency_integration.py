"""Integration tests for multi-frequency selection wiring and cross-family operation."""

from __future__ import annotations

from collections import Counter
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from gymnasium.spaces import Box, Discrete

from agilerl.algorithms import CQN, DQN, IPPO, MADDPG, NeuralUCB
from agilerl.algorithms.core.base import EvolvableAlgorithm
from agilerl.algorithms.core.registry import HyperparameterConfig, RLParameter
from agilerl.hpo.multi_frequency import MultiFrequencySelection
from agilerl.hpo.mutation import Mutations
from agilerl.models import PPOSpec
from agilerl.models.env import GymEnvSpec
from agilerl.models.hpo import (
    MultiFrequencySelectionSpec,
    MutationProbabilities,
    MutationSpec,
    RLHyperparameter,
    TournamentSelectionSpec,
)
from agilerl.models.networks import MlpSpec, StochasticActorSpec
from agilerl.models.training import TrainingSpec
from agilerl.training.trainer import LocalTrainer
from agilerl.utils.utils import run_selection_and_mutation
from tests.helper_functions import (
    generate_discrete_space,
    generate_multi_agent_box_spaces,
    generate_multi_agent_discrete_spaces,
    generate_random_box_space,
)

NET_CONFIG = {"encoder_config": {"hidden_size": [8, 8], "min_mlp_nodes": 7}}


class _DummyEnv:
    """Minimal CartPole-like vector env exposing the spaces the trainer needs."""

    def __init__(self) -> None:
        self.name = "CartPole-v1"
        self.observation_space = Box(low=-1, high=1, shape=(4,))
        self.action_space = Discrete(2)
        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space
        self.num_envs = 1

    def close(self):
        pass


def _make_ppo_trainer(
    training: TrainingSpec | None = None,
    accelerator: object | None = None,
) -> LocalTrainer:
    multi_frequency_selection = MultiFrequencySelectionSpec(
        n_subpopulations=2,
        evolution_frequency_ratios=[1, 2],
        n_winners=1,
        n_survivors=1,
        n_open_for_migration=1,
        n_losers=1,
    )
    mutation = MutationSpec(
        probabilities=MutationProbabilities(no_mut=0.5, params_mut=0.3, rl_hp_mut=0.2),
        rl_hp_selection={
            "lr": RLHyperparameter(min=1e-5, max=1e-2),
            # learn_step resizes the rollout buffer
            "learn_step": RLHyperparameter(min=64, max=2048),
        },
    )
    ppo = PPOSpec(
        learn_step=128,
        net_config=StochasticActorSpec(
            encoder_config=MlpSpec(hidden_size=[16]),
            head_config=MlpSpec(hidden_size=[16]),
        ),
    )
    with patch.object(LocalTrainer, "_make_env", return_value=_DummyEnv()):
        return LocalTrainer(
            algorithm=ppo,
            environment=GymEnvSpec(name="CartPole-v1", num_envs=1),
            training=training or TrainingSpec(max_steps=200, evo_steps=50, pop_size=8),
            mutation=mutation,
            multi_frequency_selection=multi_frequency_selection,
            accelerator=accelerator,
        )


def _rank_population_by_slot(population):
    """Give every agent a distinct fitness, subpopulation 0 dominating subpopulation 1."""
    for position, agent in enumerate(population):
        base = 100.0 if agent.subpopulation == 0 else 0.0
        agent.fitness = [base - position]


def _weakest_index(population, subpop):
    """Index of the lowest-fitness member of a subpopulation."""
    members = [a for a in population if a.subpopulation == subpop]
    return min(members, key=lambda a: a.fitness[-1]).index


class TestLocalTrainerWiring:
    def test_builds_multi_frequency_and_tags_subpopulations(self):
        trainer = _make_ppo_trainer()

        assert isinstance(trainer.multi_frequency_selection, MultiFrequencySelection)
        assert trainer.tournament_selection is None
        assert trainer.selection_strategy is trainer.multi_frequency_selection
        subpops = sorted(a.subpopulation for a in trainer.population)
        assert subpops == [0, 0, 0, 0, 1, 1, 1, 1]
        assert len({a.index for a in trainer.population}) == 8

    def test_rejects_both_multi_frequency_and_tournament(self):
        multi_frequency_selection = MultiFrequencySelectionSpec(
            n_subpopulations=2,
            evolution_frequency_ratios=[1, 2],
            n_winners=1,
            n_survivors=1,
            n_open_for_migration=1,
            n_losers=1,
        )
        ppo = PPOSpec(
            learn_step=128,
            net_config=StochasticActorSpec(
                encoder_config=MlpSpec(hidden_size=[16]),
                head_config=MlpSpec(hidden_size=[16]),
            ),
        )
        with (
            patch.object(LocalTrainer, "_make_env", return_value=_DummyEnv()),
            pytest.raises(ValueError, match="tournament_selection"),
        ):
            LocalTrainer(
                algorithm=ppo,
                environment=GymEnvSpec(name="CartPole-v1", num_envs=1),
                training=TrainingSpec(max_steps=200, evo_steps=50, pop_size=8),
                multi_frequency_selection=multi_frequency_selection,
                tournament=TournamentSelectionSpec(tournament_size=2, elitism=True),
            )

    def test_requires_explicit_pop_size_without_manifest(self):
        # pop_size is mandatory under MF-PBT
        with pytest.raises(ValueError, match="pop_size is required"):
            _make_ppo_trainer(training=TrainingSpec(max_steps=200, evo_steps=50))

    def test_rejects_pop_size_below_six_without_manifest(self):
        with pytest.raises(ValueError, match="population_size must be >= 6"):
            _make_ppo_trainer(
                training=TrainingSpec(max_steps=200, evo_steps=50, pop_size=4)
            )

    def test_accepts_multi_frequency_with_accelerator(self):
        accelerator = MagicMock()
        fake_population = [MagicMock() for _ in range(8)]
        with patch(
            "agilerl.training.trainer.create_population_from_spec",
            return_value=fake_population,
        ):
            trainer = _make_ppo_trainer(accelerator=accelerator)

        assert trainer.accelerator is accelerator
        assert isinstance(trainer.multi_frequency_selection, MultiFrequencySelection)
        assert trainer.selection_strategy is trainer.multi_frequency_selection


class TestManifestRoundTrip:
    def test_trainer_to_manifest_uses_the_unified_block(self):
        trainer = _make_ppo_trainer()

        manifest = trainer.to_manifest()

        assert manifest["training"]["pop_size"] == 8
        assert manifest["tournament_selection"]["selection_strategy"] == (
            "multi_frequency"
        )
        assert "multi_frequency_selection" not in manifest

    def test_manifest_round_trip_rebuilds_via_unified_block(self):
        trainer = _make_ppo_trainer()
        manifest = trainer.to_manifest()

        with patch.object(LocalTrainer, "_make_env", return_value=_DummyEnv()):
            rebuilt = LocalTrainer.from_manifest(manifest)

        assert isinstance(rebuilt.multi_frequency_selection, MultiFrequencySelection)
        assert rebuilt.tournament_selection is None
        assert rebuilt.selection_strategy is rebuilt.multi_frequency_selection
        assert rebuilt.training_spec.pop_size == 8


class TestPopulationResume:
    def test_resumed_population_keeps_slot_indices_and_evolves(self, tmp_path):
        from agilerl.models import DQNSpec
        from agilerl.utils.trainer_utils import create_population_from_spec

        env = _DummyEnv()
        checkpoint = tmp_path / "resume.pt"
        DQN(
            observation_space=env.observation_space,
            action_space=env.action_space,
            index=3,
        ).save_checkpoint(str(checkpoint))

        mf_spec = MultiFrequencySelectionSpec(n_subpopulations=2)
        population = create_population_from_spec(
            population_size=6,
            algo_spec=DQNSpec(),
            env=env,
            mutation_spec=None,
            replay_buffer_spec=None,
            resume_from_checkpoint=str(checkpoint),
            multi_frequency_selection_spec=mf_spec,
        )

        assert [agent.index for agent in population] == [0, 1, 2, 3, 4, 5]
        assert [agent.subpopulation for agent in population] == [0, 0, 0, 1, 1, 1]

        for i, agent in enumerate(population):
            agent.fitness = [float(6 - i)]
        strategy = MultiFrequencySelection(
            population_size=6, n_subpopulations=2, evolution_frequency_ratios=[1, 2]
        )
        _elite, evolved, _indices = strategy.select(population)

        assert len({agent.index for agent in evolved}) == len(evolved)


class TestRealPopulationEvolution:
    def test_migration_weights_resizes_rollout_buffer(self):
        # A fast-to-slow migrant clones the external agent's networks but takes the
        # studied subpop elite's mutable HPs, including learn_step. PPO sizes its
        # rollout buffer from learn_step via a mutation hook, so the migrant's buffer
        # must be rebuilt to match the elite's learn_step, or rollout collection
        # overflows it.
        trainer = _make_ppo_trainer()
        agents = list(trainer.population)
        external, elite = agents[4], agents[0]

        external.learn_step = 256
        external.mutation_hook()  # buffer sized to 256
        elite.learn_step = 512
        elite.mutation_hook()  # buffer sized to 512

        trainer.multi_frequency_selection._sync_index(agents)
        migrant = trainer.multi_frequency_selection._migrate_weights(
            external, elite, subpop=1
        )

        assert migrant.learn_step == 512  # took the elite's learn_step
        # Buffer capacity must follow the new learn_step (ceil(learn_step / num_envs))
        assert migrant.rollout_buffer.capacity == -(512 // -migrant.num_envs)

    def test_evolves_real_ppo_population_across_cycles(self):
        trainer = _make_ppo_trainer()
        agents = list(trainer.population)

        for _cycle in range(3):
            _rank_population_by_slot(agents)
            # Subpopulation 0 evolves every cycle (delta 1), so its weakest member is
            # always cloned over
            doomed = _weakest_index(agents, subpop=0)

            agents = run_selection_and_mutation(
                trainer.multi_frequency_selection,
                population=agents,
                mutation=trainer.mutations,
                env_name="CartPole-v1",
                algo="PPO",
            )

            assert doomed not in {a.index for a in agents}
            assert len(agents) == 8
            assert Counter(a.subpopulation for a in agents) == Counter({0: 4, 1: 4})
            assert len({a.index for a in agents}) == 8  # indices stay unique
            assert all(isinstance(a, EvolvableAlgorithm) for a in agents)


def _single_agent_hp_config() -> HyperparameterConfig:
    return HyperparameterConfig(
        lr=RLParameter(min=6.25e-5, max=1e-2),
        batch_size=RLParameter(min=8, max=64, dtype=int),
    )


def _multi_agent_hp_config() -> HyperparameterConfig:
    return HyperparameterConfig(
        lr_actor=RLParameter(min=1e-4, max=1e-2),
        lr_critic=RLParameter(min=1e-4, max=1e-2),
        batch_size=RLParameter(min=8, max=64, dtype=int),
    )


def _build_single_agent_population(algo_cls):
    return algo_cls.population(
        size=8,
        observation_space=generate_random_box_space((4,)),
        action_space=generate_discrete_space(2),
        hp_config=_single_agent_hp_config(),
        net_config=NET_CONFIG,
        device="cpu",
    )


def _build_maddpg_population():
    return MADDPG.population(
        size=8,
        observation_space=generate_multi_agent_box_spaces(2, (4,)),
        action_space=generate_multi_agent_discrete_spaces(2, 2),
        agent_ids=["agent_0", "agent_1"],
        hp_config=_multi_agent_hp_config(),
        net_config=NET_CONFIG,
        device="cpu",
    )


def _build_ippo_population():
    return IPPO.population(
        size=8,
        observation_space=generate_multi_agent_box_spaces(2, (4,)),
        action_space=generate_multi_agent_discrete_spaces(2, 2),
        agent_ids=["agent_0", "agent_1"],
        hp_config=_single_agent_hp_config(),
        net_config=NET_CONFIG,
        device="cpu",
    )


CROSS_FAMILY = {
    "off-policy (DQN)": ("DQN", lambda: _build_single_agent_population(DQN)),
    "multi-agent off-policy (MADDPG)": ("MADDPG", _build_maddpg_population),
    "multi-agent on-policy (IPPO)": ("IPPO", _build_ippo_population),
    "bandit (NeuralUCB)": (
        "NeuralUCB",
        lambda: _build_single_agent_population(NeuralUCB),
    ),
    "offline (CQN)": ("CQN", lambda: _build_single_agent_population(CQN)),
}


class TestCrossFamilyEvolution:
    @pytest.mark.parametrize("family", list(CROSS_FAMILY), ids=list(CROSS_FAMILY))
    def test_evolves_a_real_population_of_every_family(self, family):
        algo_name, build_population = CROSS_FAMILY[family]
        population = build_population()
        for agent in population:
            agent.subpopulation = agent.index // 4
        strategy = MultiFrequencySelection(
            population_size=8,
            n_subpopulations=2,
            evolution_frequency_ratios=[1, 2],
            n_winners=1,
            n_survivors=1,
            n_open_for_migration=1,
            n_losers=1,
            seed=0,
        )
        mutation = Mutations(
            no_mutation=0.2,
            architecture=0.0,
            new_layer_prob=0.0,
            parameters=0.4,
            activation=0.0,
            rl_hp=0.4,
            mutation_sd=0.1,
            rand_seed=0,
            device="cpu",
        )

        for cycle in range(3):
            _rank_population_by_slot(population)
            doomed = {_weakest_index(population, subpop=0)}
            if cycle % 2 == 1:
                doomed.add(_weakest_index(population, subpop=1))

            population = run_selection_and_mutation(
                strategy,
                population=population,
                mutation=mutation,
                env_name="Env",
                algo=algo_name,
            )

            surviving = {a.index for a in population}
            assert not (doomed & surviving)  # the due subpops really did evolve
            assert len(population) == 8
            assert Counter(a.subpopulation for a in population) == Counter({0: 4, 1: 4})
            assert len({a.index for a in population}) == 8
            assert all(isinstance(a, EvolvableAlgorithm) for a in population)


class _LLMHPParam:
    def __init__(self, value):
        self.value = value


class _LLMHPConfig:
    """Minimal HyperparameterConfig stand-in for the LLM migration path."""

    def __init__(self, values):
        self._params = {name: _LLMHPParam(v) for name, v in values.items()}

    def __iter__(self):
        return iter(self._params)

    def __bool__(self):
        return bool(self._params)

    def names(self):
        return list(self._params)

    def __getitem__(self, key):
        return self._params[key]


class _LLMFinetuneAgent:
    """Fake :class:`LLMAlgorithm`."""

    def __init__(self, index, subpopulation, fitness, lr=1e-3):
        self.index = index
        self.subpopulation = subpopulation
        self.fitness = [fitness]
        self.lr = lr
        self.accelerator = None
        self.mut = "stale-mut"
        self.registry = SimpleNamespace(
            hp_config=_LLMHPConfig({"lr": lr}), optimizers=[]
        )
        self.clean_up_calls = 0

    def clone(self, index=None, wrap=False):
        twin = _LLMFinetuneAgent(
            self.index if index is None else index,
            self.subpopulation,
            self.fitness[-1],
            lr=self.lr,
        )
        twin.mut = self.mut
        for name in self.registry.hp_config.names():
            twin.registry.hp_config[name].value = self.registry.hp_config[name].value
        return twin

    def clean_up(self):
        self.clean_up_calls += 1

    def mutation_hook(self):
        pass

    def reinit_optimizers(self, optimizer=None):
        pass


class TestLLMEvolution:
    def test_run_selection_and_mutation_drives_the_real_operator(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(
            "agilerl.hpo.multi_frequency.LLMAlgorithm", _LLMFinetuneAgent
        )
        saved: list = []
        monkeypatch.setattr(
            "agilerl.utils.utils.save_llm_checkpoint",
            lambda agent, path: saved.append(agent),
        )
        population = [_LLMFinetuneAgent(i, i // 4, 0.0) for i in range(8)]
        strategy = MultiFrequencySelection(
            population_size=8,
            n_subpopulations=2,
            evolution_frequency_ratios=[1, 2],
            n_winners=1,
            n_survivors=1,
            n_open_for_migration=1,
            n_losers=1,
            seed=0,
        )
        mutation = Mutations(
            no_mutation=1.0,
            architecture=0.0,
            new_layer_prob=0.0,
            parameters=0.0,
            activation=0.0,
            rl_hp=0.0,
            mutation_sd=0.1,
            rand_seed=0,
            device="cpu",
        )

        for cycle in range(3):
            _rank_population_by_slot(population)
            doomed = {_weakest_index(population, subpop=0)}
            if cycle % 2 == 1:
                doomed.add(_weakest_index(population, subpop=1))

            population = run_selection_and_mutation(
                strategy,
                population=population,
                mutation=mutation,
                env_name="Env",
                algo="GRPO",
                language_model=True,
                save_elite=True,
                elite_path=str(tmp_path / "elite"),
            )

            assert not (doomed & {a.index for a in population})
            assert len(population) == 8
            assert Counter(a.subpopulation for a in population) == Counter({0: 4, 1: 4})
            assert len({a.index for a in population}) == 8  # indices stay unique
            assert all(a.mut == "None" for a in population)

        assert len(saved) == 3  # the live elite is checkpointed every cycle
        assert all(isinstance(a, _LLMFinetuneAgent) for a in population)

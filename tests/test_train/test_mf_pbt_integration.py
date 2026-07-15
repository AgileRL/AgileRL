"""Integration tests for MF-PBT wiring and cross-family operation."""

from __future__ import annotations

from collections import Counter
from unittest.mock import patch

import pytest
from gymnasium.spaces import Box, Discrete

from agilerl.algorithms import CQN, DQN, IPPO, MADDPG, NeuralUCB
from agilerl.algorithms.core.base import EvolvableAlgorithm
from agilerl.algorithms.core.registry import HyperparameterConfig, RLParameter
from agilerl.hpo.multi_frequency import MultiFrequencyStrategy
from agilerl.hpo.mutation import Mutations
from agilerl.models import PPOSpec
from agilerl.models.env import GymEnvSpec
from agilerl.models.hpo import (
    MultiFrequencyStrategySpec,
    MutationProbabilities,
    MutationSpec,
    RLHyperparameter,
    TournamentSelectionSpec,
)
from agilerl.models.networks import MlpSpec, StochasticActorSpec
from agilerl.models.training import TrainingSpec
from agilerl.training.trainer import LocalTrainer
from agilerl.utils.utils import multi_frequency_selection_and_mutation
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
    multi_frequency_strategy = MultiFrequencyStrategySpec(
        n_subpopulations=2,
        n_individuals_per_subpopulation=4,
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
            multi_frequency_strategy=multi_frequency_strategy,
            accelerator=accelerator,
        )


def test_localtrainer_builds_mfpbt_and_tags_subpopulations():
    trainer = _make_ppo_trainer()

    assert isinstance(trainer.multi_frequency_strategy, MultiFrequencyStrategy)
    assert trainer.tournament_selection is None
    assert trainer.selection_strategy is trainer.multi_frequency_strategy
    subpops = sorted(a.subpopulation for a in trainer.population)
    assert subpops == [0, 0, 0, 0, 1, 1, 1, 1]
    assert len({a.index for a in trainer.population}) == 8


def test_localtrainer_rejects_both_mfpbt_and_tournament():
    multi_frequency_strategy = MultiFrequencyStrategySpec(
        n_subpopulations=2,
        n_individuals_per_subpopulation=4,
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
            multi_frequency_strategy=multi_frequency_strategy,
            tournament=TournamentSelectionSpec(tournament_size=2, elitism=True),
        )


def test_localtrainer_derives_pop_size_without_manifest():
    trainer = _make_ppo_trainer(training=TrainingSpec(max_steps=200, evo_steps=50))

    assert trainer.training_spec.pop_size == 8
    assert len(trainer.population) == 8
    assert Counter(a.subpopulation for a in trainer.population) == Counter({0: 4, 1: 4})


def test_localtrainer_rejects_conflicting_pop_size_without_manifest():
    with pytest.raises(ValueError, match="conflicts with the MF-PBT"):
        _make_ppo_trainer(
            training=TrainingSpec(max_steps=200, evo_steps=50, pop_size=4)
        )


def test_localtrainer_rejects_mfpbt_with_accelerator():
    with pytest.raises(NotImplementedError, match="Accelerate"):
        _make_ppo_trainer(accelerator=object())


def test_mfpbt_trainer_to_manifest_round_trips():
    trainer = _make_ppo_trainer()

    manifest = trainer.to_manifest()

    # The manifest validator must not reject its own derived pop_size on this round-trip
    assert manifest["training"]["pop_size"] == 8  # 2 subpops x 4 individuals
    # MF-PBT serializes under the single discriminated tournament_selection field
    assert manifest["tournament_selection"]["selection_strategy"] == "multi_frequency"
    assert "multi_frequency_strategy" not in manifest


def test_mfpbt_manifest_round_trip_rebuilds_via_unified_block():
    trainer = _make_ppo_trainer()
    manifest = trainer.to_manifest()

    with patch.object(LocalTrainer, "_make_env", return_value=_DummyEnv()):
        rebuilt = LocalTrainer.from_manifest(manifest)

    assert isinstance(rebuilt.multi_frequency_strategy, MultiFrequencyStrategy)
    assert rebuilt.tournament_selection is None
    assert rebuilt.selection_strategy is rebuilt.multi_frequency_strategy
    assert rebuilt.training_spec.pop_size == 8


def test_migration_weights_resizes_rollout_buffer():
    # A fast-to-slow migrant clones the external agent's networks but takes the studied
    # subpop elite's mutable HPs, including learn_step. PPO sizes its rollout buffer
    # from learn_step via a mutation hook, so the migrant's buffer must be rebuilt to
    # match the elite's learn_step, or rollout collection overflows it.
    trainer = _make_ppo_trainer()
    agents = list(trainer.population)
    external, elite = agents[4], agents[0]

    external.learn_step = 256
    external.mutation_hook()  # buffer sized to 256
    elite.learn_step = 512
    elite.mutation_hook()  # buffer sized to 512

    trainer.multi_frequency_strategy._sync_index(agents)
    migrant = trainer.multi_frequency_strategy._migrate_weights(
        external, elite, subpop=1
    )

    assert migrant.learn_step == 512  # took the elite's learn_step
    # Buffer capacity must follow the new learn_step (ceil(learn_step / num_envs))
    assert migrant.rollout_buffer.capacity == -(512 // -migrant.num_envs)


def test_mfpbt_evolves_real_ppo_population_across_cycles():
    trainer = _make_ppo_trainer()
    agents = list(trainer.population)

    for cycle in range(3):
        for offset, agent in enumerate(agents):
            agent.fitness = [float((cycle + offset) % 8)]
        agents = multi_frequency_selection_and_mutation(
            agents,
            trainer.multi_frequency_strategy,
            mutation=trainer.mutations,
            env_name="CartPole-v1",
            algo="PPO",
        )

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


@pytest.mark.parametrize("family", list(CROSS_FAMILY), ids=list(CROSS_FAMILY))
def test_mfpbt_evolves_real_population_of_every_family(family):
    algo_name, build_population = CROSS_FAMILY[family]
    population = build_population()
    for agent in population:
        agent.subpopulation = agent.index // 4
    strategy = MultiFrequencyStrategy(
        n_subpopulations=2,
        n_individuals_per_subpopulation=4,
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

    for _ in range(3):
        for agent in population:
            base = 100.0 if agent.subpopulation == 0 else 0.0
            agent.fitness = [base + (agent.index % 4)]
        population = multi_frequency_selection_and_mutation(
            population, strategy, mutation=mutation, env_name="Env", algo=algo_name
        )

        assert len(population) == 8
        assert Counter(a.subpopulation for a in population) == Counter({0: 4, 1: 4})
        assert len({a.index for a in population}) == 8
        assert all(isinstance(a, EvolvableAlgorithm) for a in population)

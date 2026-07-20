from unittest.mock import MagicMock

import numpy as np
import pytest
from accelerate import Accelerator

from agilerl import HAS_DEEPSPEED, HAS_LLM_DEPENDENCIES, HAS_VLLM
from agilerl.algorithms import CQN, DDPG, DQN, MADDPG, MATD3, PPO, TD3, RainbowDQN
from agilerl.hpo.tournament import TournamentSelection
from agilerl.utils.algo_utils import clone_llm
from tests.helper_functions import (
    generate_discrete_space,
    generate_multi_agent_box_spaces,
    generate_multi_agent_discrete_spaces,
    generate_random_box_space,
)

if HAS_LLM_DEPENDENCIES:
    from peft import LoraConfig

    from agilerl.algorithms import GRPO

create_module = None
if HAS_DEEPSPEED and HAS_VLLM:
    from tests.test_algorithms.test_llms.test_grpo import create_module

LLM_POPULATION_SIZE = 4


def make_llm_accelerator(num_processes=1, is_main_process=True):
    """Build a stand-in accelerator for the LLM tournament path."""
    accelerator = MagicMock(spec=Accelerator)
    accelerator.is_main_process = is_main_process
    accelerator.wait_for_everyone = MagicMock()
    accelerator.state = MagicMock()
    accelerator.state.deepspeed_plugin = MagicMock()
    accelerator.state.deepspeed_plugin.deepspeed_config = {
        "zero_optimization": {"stage": 1},
    }
    accelerator.free_memory = lambda *args: args
    accelerator.unwrap_model = lambda arg: arg
    accelerator.num_processes = num_processes
    return accelerator


def make_llm_population(accelerator, use_accelerator=True):
    """Build a GRPO population with ascending fitness and mocked clones.

    Clones are mocks so that the tournament's clone/clean-up bookkeeping can be
    inspected without materialising more models.
    """
    actor_network = create_module(
        input_size=1,
        max_tokens=8,
        vocab_size=100,
        device="cpu",
    )
    population = [
        GRPO(
            actor_network=clone_llm(actor_network, 0),
            pad_token_id=99,
            pad_token="<pad>",
            hp_config=None,
            index=idx,
            batch_size=1,
            beta=0.001,
            lr=0.000005,
            clip_coef=0.2,
            max_grad_norm=0.1,
            update_epochs=1,
            group_size=2,
            temperature=0.9,
            calc_position_embeddings=True,
            use_memory_efficient_params=True,
            max_output_tokens=8,
            min_output_tokens=None,
            lora_config=LoraConfig(
                r=16,
                lora_alpha=64,
                target_modules=["linear_1"],
                task_type="CAUSAL_LM",
                lora_dropout=0.05,
            ),
            cosine_lr_schedule_config=None,
            accelerator=None,
            device="cpu",
        )
        for idx in range(LLM_POPULATION_SIZE)
    ]
    for agent in population:
        if use_accelerator:
            agent.accelerator = accelerator

    for agent in population:
        # Create a mock clone that returns a new mock agent
        def mock_clone(index, wrap=False, _agent=agent):
            mock_agent = MagicMock()
            mock_agent.index = index
            mock_agent.accelerator = accelerator
            mock_agent.clean_up = MagicMock()
            mock_agent.fitness = _agent.fitness
            return mock_agent

        agent.clone = MagicMock(side_effect=mock_clone)

    for idx, agent in enumerate(population):
        agent.fitness = [1 + 3 * idx, 2 + 3 * idx, 3 + 3 * idx]

    return population


class TestTournamentSelectionInit:
    # Initializes the 'TournamentSelection' object with the given parameters.
    def test_with_given_parameters(self):
        tournament_size = 5
        elitism = True
        population_size = 100

        ts = TournamentSelection(tournament_size, elitism, population_size)

        assert ts.tournament_size == tournament_size
        assert ts.elitism == elitism
        assert ts.population_size == population_size

    @pytest.mark.parametrize(
        ("tournament_size", "elitism", "population_size", "match"),
        [
            (0, True, 4, "greater than zero"),
            (2, "invalid", 4, "boolean"),
            (2, True, 0, "greater than zero"),
        ],
    )
    def test_validation(self, tournament_size, elitism, population_size, match):
        with pytest.raises(AssertionError, match=match):
            TournamentSelection(
                tournament_size=tournament_size,
                elitism=elitism,
                population_size=population_size,
            )


class TestTournamentSelectionSelect:
    ### Single-agent algorithms ###
    # Returns best agent and new population of agents following tournament selection.
    def test_returns_best_agent_and_new_population(self):
        observation_space = generate_random_box_space((4,))
        discrete_action_space = generate_discrete_space(2)
        continuous_action_space = generate_random_box_space((2,))
        net_config = {"encoder_config": {"hidden_size": [8, 8], "min_mlp_nodes": 7}}
        device = "cpu"
        population_size = 5

        # Initialize the class
        tournament_selection = TournamentSelection(3, True, population_size)

        algo_classes = {
            "DQN": DQN,
            "Rainbow DQN": RainbowDQN,
            "DDPG": DDPG,
            "TD3": TD3,
            "PPO": PPO,
            "CQN": CQN,
        }

        for algo_name, algo_cls in algo_classes.items():
            if algo_name in ["TD3", "DDPG"]:
                action_space = continuous_action_space
            else:
                action_space = discrete_action_space

            population = algo_cls.population(
                size=population_size,
                observation_space=observation_space,
                action_space=action_space,
                net_config=net_config,
                device=device,
            )

            population[0].fitness = [1, 2, 3]
            population[1].fitness = [4, 5, 6]
            population[2].fitness = [7, 8, 9]
            population[3].fitness = [10, 11, 12]
            population[4].fitness = [13, 14, 15]

            # Call the select method
            elite, new_population = tournament_selection.select(population)

            # Check if the elite agent is the best agent in the population
            assert elite.fitness == [13, 14, 15]
            assert elite.index == 4
            assert new_population[0].fitness == [13, 14, 15]
            assert new_population[0].index == 4

            # Check if the new population has the correct length
            assert len(new_population) == population_size

    # Returns best agent and new population of agents following tournament selection without elitism.
    def test_returns_best_agent_and_new_population_without_elitism(self):
        observation_space = generate_random_box_space((4,))
        discrete_action_space = generate_discrete_space(2)
        continuous_action_space = generate_random_box_space((2,))
        net_config = {"encoder_config": {"hidden_size": [8, 8], "min_mlp_nodes": 7}}
        device = "cpu"
        population_size = 5

        # Initialize the class
        tournament_selection = TournamentSelection(3, False, population_size)

        algo_classes = {
            "DQN": DQN,
            "Rainbow DQN": RainbowDQN,
            "DDPG": DDPG,
            "TD3": TD3,
            "PPO": PPO,
            "CQN": CQN,
        }

        for algo_name, algo_cls in algo_classes.items():
            if algo_name in ["TD3", "DDPG"]:
                action_space = continuous_action_space
            else:
                action_space = discrete_action_space

            population = algo_cls.population(
                size=population_size,
                observation_space=observation_space,
                action_space=action_space,
                net_config=net_config,
                device=device,
            )

            population[0].fitness = [1, 2, 3]
            population[1].fitness = [4, 5, 6]
            population[2].fitness = [7, 8, 9]
            population[3].fitness = [10, 11, 12]
            population[4].fitness = [13, 14, 15]

            # Call the select method
            elite, new_population = tournament_selection.select(population)

            # Check if the elite agent is the best agent in the population
            assert elite.fitness == [13, 14, 15]
            assert elite.index == 4

            # Check if the new population has the correct length
            assert len(new_population) == population_size

    ### Multi-agent algorithms ###
    # Returns best agent and new population of agents following tournament selection.
    def test_returns_best_agent_and_new_population_multi_agent(self):
        observation_space = generate_multi_agent_box_spaces(2, (4,))
        action_space = generate_multi_agent_discrete_spaces(2, 2)
        agent_ids = ["agent_0", "agent_1"]
        net_config = {"encoder_config": {"hidden_size": [8, 8], "min_mlp_nodes": 7}}
        device = "cpu"
        population_size = 5

        # Initialize the class
        tournament_selection = TournamentSelection(3, True, population_size)

        algo_classes = {"MADDPG": MADDPG, "MATD3": MATD3}

        for algo_cls in algo_classes.values():
            population = algo_cls.population(
                size=population_size,
                observation_space=observation_space,
                action_space=action_space,
                agent_ids=agent_ids,
                net_config=net_config,
                device=device,
            )

            population[0].fitness = [1, 2, 3]
            population[1].fitness = [4, 5, 6]
            population[2].fitness = [7, 8, 9]
            population[3].fitness = [10, 11, 12]
            population[4].fitness = [13, 14, 15]

            # Call the select method
            elite, new_population = tournament_selection.select(population)

            # Check if the elite agent is the best agent in the population
            assert elite.fitness == [13, 14, 15]
            assert elite.index == 4
            assert new_population[0].fitness == [13, 14, 15]
            assert new_population[0].index == 4

            # Check if the new population has the correct length
            assert len(new_population) == population_size

    # Returns best agent and new population of agents following tournament selection without elitism.
    def test_returns_best_agent_and_new_population_without_elitism_multi_agent(self):
        observation_space = generate_multi_agent_box_spaces(2, (4,))
        action_space = generate_multi_agent_discrete_spaces(2, 2)
        agent_ids = ["agent_0", "agent_1"]
        net_config = {"encoder_config": {"hidden_size": [8, 8], "min_mlp_nodes": 7}}
        device = "cpu"
        population_size = 5

        # Initialize the class
        tournament_selection = TournamentSelection(3, False, population_size)

        algo_classes = {"MADDPG": MADDPG, "MATD3": MATD3}

        for algo_cls in algo_classes.values():
            population = algo_cls.population(
                size=population_size,
                observation_space=observation_space,
                action_space=action_space,
                agent_ids=agent_ids,
                net_config=net_config,
                device=device,
            )

            population[0].fitness = [1, 2, 3]
            population[1].fitness = [4, 5, 6]
            population[2].fitness = [7, 8, 9]
            population[3].fitness = [10, 11, 12]
            population[4].fitness = [13, 14, 15]

            # Call the select method
            elite, new_population = tournament_selection.select(population)

            # Check if the elite agent is the best agent in the population
            assert elite.fitness == [13, 14, 15]
            assert elite.index == 4

            # Check if the new population has the correct length
            assert len(new_population) == population_size

    @pytest.mark.skipif(
        not (HAS_VLLM and HAS_DEEPSPEED),
        reason="Need to install agilerl with deepspeed + vllm",
    )
    @pytest.mark.parametrize("use_accelerator", [True, False])
    @pytest.mark.parametrize("elitism", [True, False])
    @pytest.mark.parametrize("num_processes", [1, 2])
    def test_language_model_tournament(self, use_accelerator, elitism, num_processes):
        tournament_selection = TournamentSelection(3, elitism, LLM_POPULATION_SIZE)
        accelerator = make_llm_accelerator(num_processes=num_processes)
        population = make_llm_population(accelerator, use_accelerator=use_accelerator)

        # Call the select method
        elite, new_population = tournament_selection.select(population)

        # Check if the elite agent is the best agent in the population
        assert elite.fitness == [10, 11, 12]
        if elitism:
            # Without elitism the elite is a standalone clone whose index depends on
            # whether the tournament happened to draw it, so only its identity as the
            # best performer is meaningful.
            assert elite.index == 3

        # The elite is passed to save_llm_checkpoint, which needs a live actor, so it
        # must not be one of the originals cleaned up during selection.
        assert elite.actor is not None

        # Check if the new population has the correct length
        assert len(new_population) == LLM_POPULATION_SIZE

    @pytest.mark.skipif(
        not (HAS_VLLM and HAS_DEEPSPEED),
        reason="Need to install agilerl with deepspeed + vllm",
    )
    @pytest.mark.parametrize("elitism", [True, False])
    def test_language_model_tournament_non_main_process(self, elitism, monkeypatch):
        """Every rank resolves an elite from the broadcast selection.

        Non-main ranks skip the selection block entirely. With elitism disabled the
        broadcast carries no ``is_elite`` entry for them to resolve the elite from, so
        they depend on the broadcast elite index. save_llm_checkpoint is collective,
        so a rank that fails to resolve one hangs the others.
        """
        # Run process 0 first and capture exactly what it broadcasts.
        broadcast_payload = []

        def capture(object_list, from_process=0):
            broadcast_payload.append(list(object_list))
            return object_list

        monkeypatch.setattr("agilerl.hpo.tournament.broadcast_object_list", capture)
        main_accelerator = make_llm_accelerator(num_processes=2, is_main_process=True)
        main_elite, _ = TournamentSelection(3, elitism, LLM_POPULATION_SIZE).select(
            make_llm_population(main_accelerator),
        )
        assert len(broadcast_payload) == 1

        # Replay that payload on a rank that ran no selection of its own.
        monkeypatch.setattr(
            "agilerl.hpo.tournament.broadcast_object_list",
            lambda object_list, from_process=0: broadcast_payload[0],
        )
        worker_accelerator = make_llm_accelerator(
            num_processes=2, is_main_process=False
        )
        worker_elite, worker_population = TournamentSelection(
            3,
            elitism,
            LLM_POPULATION_SIZE,
        ).select(make_llm_population(worker_accelerator))

        # Both ranks agree on the elite, and it is usable for checkpointing.
        assert worker_elite is not None
        assert worker_elite.fitness == main_elite.fitness == [10, 11, 12]
        assert worker_elite.actor is not None
        assert len(worker_population) == LLM_POPULATION_SIZE

    @pytest.mark.skipif(
        not (HAS_VLLM and HAS_DEEPSPEED),
        reason="Need to install agilerl with deepspeed + vllm",
    )
    def test_detects_llm_by_type_not_algo_name(self):
        """LLM branch selection should rely on type, not a specific algo string."""
        tournament_selection = TournamentSelection(3, True, 1)
        actor_network = create_module(
            input_size=1,
            max_tokens=32,
            vocab_size=128,
            device="cpu",
        )
        agent = GRPO(
            actor_network=actor_network,
            pad_token_id=127,
            pad_token="<pad>",
            hp_config=None,
            index=0,
            batch_size=1,
            beta=0.001,
            lr=5e-6,
            clip_coef=0.2,
            max_grad_norm=0.1,
            update_epochs=1,
            group_size=2,
            temperature=0.9,
            calc_position_embeddings=True,
            use_memory_efficient_params=True,
            max_output_tokens=32,
            min_output_tokens=None,
            lora_config=LoraConfig(
                r=4,
                lora_alpha=8,
                target_modules=["linear_1"],
                task_type="CAUSAL_LM",
                lora_dropout=0.05,
            ),
            cosine_lr_schedule_config=None,
            accelerator=None,
            device="cpu",
        )
        # Simulate a different LLM algorithm label to guard against string checks.
        agent.algo = "LLMPPO"
        agent.fitness = [1.0]

        with (
            pytest.MonkeyPatch.context() as m,
        ):
            llm_called = {"value": False}
            std_called = {"value": False}

            def _llm_branch(population):
                llm_called["value"] = True
                return (population[0], population)

            def _std_branch(population):
                std_called["value"] = True
                return (population[0], population)

            m.setattr(tournament_selection, "_select_llm_agents", _llm_branch)
            m.setattr(tournament_selection, "_select_standard_agents", _std_branch)
            tournament_selection.select([agent])

        assert llm_called["value"] is True
        assert std_called["value"] is False


class TestTournamentSelectionTournament:
    @pytest.mark.parametrize(
        ("fitness_values", "tournament_size"),
        [
            ([1.0, 2.0, 3.0], 2),
            ([10.0, 5.0, 0.0], 3),
            ([0.5, 1.0, 0.5], 2),
        ],
    )
    def test_returns_valid_winner_index(self, fitness_values, tournament_size):
        import numpy as np

        np.random.seed(0)
        ts = TournamentSelection(
            tournament_size=tournament_size,
            elitism=True,
            population_size=len(fitness_values) + 1,
        )
        winner = ts._tournament(fitness_values)
        assert 0 <= winner < len(fitness_values)


class TestTournamentSelectionElitism:
    def test_returns_elite_rank_max_id(self):

        observation_space = generate_random_box_space((4,))
        discrete_action_space = generate_discrete_space(2)
        net_config = {"encoder_config": {"hidden_size": [8, 8], "min_mlp_nodes": 7}}
        population = DQN.population(
            size=4,
            observation_space=observation_space,
            action_space=discrete_action_space,
            net_config=net_config,
            device="cpu",
        )
        population[0].fitness = [1, 2]
        population[1].fitness = [3, 4]
        population[2].fitness = [5, 6]
        population[3].fitness = [7, 8]

        ts = TournamentSelection(3, True, 4)
        elite, rank, max_id = ts._elitism(population)
        assert elite.fitness == [7, 8]
        assert rank.shape == (4,)
        assert max_id == 3
        assert elite.index == 3


class TestScalarFitness:
    def test_scalar_fitness_dict(self):
        # Multi-agent per-sub-agent fitness collapses to the mean across values.
        result = TournamentSelection._scalar_fitness({"agent_0": 2.0, "agent_1": 4.0})
        assert result == pytest.approx(3.0)
        assert isinstance(result, float)

    @pytest.mark.parametrize(
        ("fitness", "expected"),
        [
            ([1.0, 2.0, 3.0], 2.0),
            ((4.0, 6.0), 5.0),
            (np.array([0.0, 10.0]), 5.0),
        ],
    )
    def test_scalar_fitness_sequence(self, fitness, expected):
        result = TournamentSelection._scalar_fitness(fitness)
        assert result == pytest.approx(expected)
        assert isinstance(result, float)

    def test_scalar_fitness_scalar(self):
        result = TournamentSelection._scalar_fitness(7.5)
        assert result == pytest.approx(7.5)
        assert isinstance(result, float)

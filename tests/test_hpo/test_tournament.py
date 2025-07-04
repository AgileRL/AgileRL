from unittest.mock import MagicMock

import gymnasium as gym
import pytest
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from peft import LoraConfig

from agilerl.algorithms import CQN, DDPG, DQN, GRPO, MADDPG, MATD3, PPO, TD3, RainbowDQN
from agilerl.hpo.tournament import TournamentSelection
from agilerl.utils.algo_utils import clone_llm
from agilerl.utils.utils import create_population
from tests.helper_functions import (
    generate_discrete_space,
    generate_multi_agent_box_spaces,
    generate_multi_agent_discrete_spaces,
    generate_random_box_space,
)
from tests.test_algorithms.test_grpo import create_module

# Shared HP dict that can be used by any algorithm
INIT_HP = {
    "POPULATION_SIZE": 4,
    "DOUBLE": True,
    "BATCH_SIZE": 128,
    "LR": 1e-3,
    "CUDAGRAPHS": False,
    "LR_ACTOR": 1e-4,
    "LR_CRITIC": 1e-3,
    "GAMMA": 0.99,
    "LEARN_STEP": 1,
    "TAU": 1e-3,
    "BETA": 0.4,
    "PRIOR_EPS": 0.000001,
    "NUM_ATOMS": 51,
    "V_MIN": 0,
    "V_MAX": 200,
    "N_STEP": 3,
    "POLICY_FREQ": 10,
    "GAE_LAMBDA": 0.95,
    "ACTION_STD_INIT": 0.6,
    "CLIP_COEF": 0.2,
    "ENT_COEF": 0.01,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    "TARGET_KL": None,
    "UPDATE_EPOCHS": 4,
    "AGENT_IDS": ["agent1", "agent2"],
    "LAMBDA": 1.0,
    "REG": 0.000625,
    "CHANNELS_LAST": False,
    "O_U_NOISE": True,
    "EXPL_NOISE": 0.1,
    "MEAN_NOISE": 0.0,
    "THETA": 0.15,
    "DT": 0.01,
}


# Initializes the 'TournamentSelection' object with the given parameters.
def test_initialization_with_given_parameters():
    tournament_size = 5
    elitism = True
    population_size = 100
    eval_loop = 10

    ts = TournamentSelection(tournament_size, elitism, population_size, eval_loop)

    assert ts.tournament_size == tournament_size
    assert ts.elitism == elitism
    assert ts.population_size == population_size
    assert ts.eval_loop == eval_loop


### Single-agent algorithms ###
# Returns best agent and new population of agents following tournament selection.
def test_returns_best_agent_and_new_population():
    observation_space = generate_random_box_space((4,))
    discrete_action_space = generate_discrete_space(2)
    continuous_action_space = generate_random_box_space((2,))
    net_config = {"encoder_config": {"hidden_size": [8, 8], "min_mlp_nodes": 7}}
    population_size = 4
    device = "cpu"
    population_size = 5

    # Initialize the class
    tournament_selection = TournamentSelection(3, True, population_size, 2)

    algo_classes = {
        "DQN": DQN,
        "Rainbow DQN": RainbowDQN,
        "DDPG": DDPG,
        "TD3": TD3,
        "PPO": PPO,
        "CQN": CQN,
    }

    for algo in algo_classes.keys():
        if algo in ["TD3", "DDPG"]:
            action_space = continuous_action_space
        else:
            action_space = discrete_action_space

        population = create_population(
            algo=algo,
            observation_space=observation_space,
            action_space=action_space,
            net_config=net_config,
            INIT_HP=INIT_HP,
            population_size=population_size,
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
def test_returns_best_agent_and_new_population_without_elitism():
    observation_space = generate_random_box_space((4,))
    discrete_action_space = generate_discrete_space(2)
    continuous_action_space = generate_random_box_space((2,))
    net_config = {"encoder_config": {"hidden_size": [8, 8], "min_mlp_nodes": 7}}
    population_size = 4
    device = "cpu"
    population_size = 5

    # Initialize the class
    tournament_selection = TournamentSelection(3, False, population_size, 2)

    algo_classes = {
        "DQN": DQN,
        "Rainbow DQN": RainbowDQN,
        "DDPG": DDPG,
        "TD3": TD3,
        "PPO": PPO,
        "CQN": CQN,
    }

    for algo in algo_classes.keys():
        if algo in ["TD3", "DDPG"]:
            action_space = continuous_action_space
        else:
            action_space = discrete_action_space

        population = create_population(
            algo=algo,
            observation_space=observation_space,
            action_space=action_space,
            net_config=net_config,
            INIT_HP=INIT_HP,
            population_size=population_size,
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
def test_returns_best_agent_and_new_population_multi_agent():
    observation_space = generate_multi_agent_box_spaces(2, (4,))
    action_space = generate_multi_agent_discrete_spaces(2, 2)
    net_config = {"encoder_config": {"hidden_size": [8, 8], "min_mlp_nodes": 7}}
    population_size = 4
    device = "cpu"
    population_size = 5

    # Initialize the class
    tournament_selection = TournamentSelection(3, True, population_size, 2)

    algo_classes = {"MADDPG": MADDPG, "MATD3": MATD3}

    for algo in algo_classes.keys():
        population = create_population(
            algo=algo,
            observation_space=observation_space,
            action_space=action_space,
            net_config=net_config,
            INIT_HP=INIT_HP,
            population_size=population_size,
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
def test_returns_best_agent_and_new_population_without_elitism_multi_agent():
    observation_space = generate_multi_agent_box_spaces(2, (4,))
    action_space = generate_multi_agent_discrete_spaces(2, 2)
    net_config = {"encoder_config": {"hidden_size": [8, 8], "min_mlp_nodes": 7}}
    population_size = 4
    device = "cpu"
    population_size = 5

    # Initialize the class
    tournament_selection = TournamentSelection(3, False, population_size, 2)

    algo_classes = {"MADDPG": MADDPG, "MATD3": MATD3}

    for algo in algo_classes.keys():
        population = create_population(
            algo=algo,
            observation_space=observation_space,
            action_space=action_space,
            net_config=net_config,
            INIT_HP=INIT_HP,
            population_size=population_size,
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


@pytest.mark.parametrize("use_accelerator", [True, False])
@pytest.mark.parametrize("elitism", [True, False])
@pytest.mark.parametrize("num_processes", [1, 2])
def test_language_model_tournament(use_accelerator, elitism, num_processes):
    AcceleratorState._reset_state(True)
    tournament_selection = TournamentSelection(3, elitism, 4, 2)
    observation_space = gym.spaces.Box(low=0, high=1000 - 1, shape=(1,))
    action_space = gym.spaces.Box(
        low=0,
        high=1000 - 1,
        shape=(20,),
    )
    population_size = 4

    init_hp = {
        "ALGO": "GRPO",
        "BATCH_SIZE": 1,
        "REDUCE_MEMORY_PEAK": True,
        "BETA": 0.001,
        "LR": 0.000005,
        "CLIP_COEF": 0.2,
        "MAX_GRAD_NORM": 0.1,
        "UPDATE_EPOCHS": 1,
        "GROUP_SIZE": 8,
        "TEMPERATURE": 0.9,
        "CALC_POSITION_EMBEDDINGS": True,
        "MIN_OUTPUT_TOKENS": None,
        "MAX_OUTPUT_TOKENS": 1024,
        "COSINE_lR_SCHEDULER": None,
        "TOURN_SIZE": 2,
        "ELITISM": True,
        "POP_SIZE": 4,
        "EVAL_LOOP": 1,
        "PAD_TOKEN_ID": 1000,
    }
    actor_network = create_module(
        input_size=1, max_tokens=1024, vocab_size=1000, device="cpu"
    )
    accelerator = MagicMock(spec=Accelerator)
    accelerator.is_main_process = True
    accelerator.wait_for_everyone = MagicMock()
    accelerator.state = MagicMock()
    accelerator.state.deepspeed_plugin = MagicMock()
    accelerator.state.deepspeed_plugin.deepspeed_config = {
        "zero_optimization": {"stage": 1}
    }
    accelerator.free_memory = lambda *args: args
    accelerator.unwrap_model = lambda arg: arg
    accelerator.num_processes = num_processes

    population = [
        GRPO(
            observation_space=observation_space,
            action_space=action_space,
            actor_network=clone_llm(actor_network),
            pad_token_id=INIT_HP.get("PAD_TOKEN_ID"),
            hp_config=None,
            index=idx,
            batch_size=INIT_HP.get("BATCH_SIZE", 1),
            beta=INIT_HP.get("BETA", 0.001),
            lr=INIT_HP.get("LR", 5e-7),
            clip_coef=INIT_HP.get("CLIP_COEF", 0.2),
            max_grad_norm=INIT_HP.get("MAX_GRAD_NORM", 0.1),
            update_epochs=INIT_HP.get("UPDATE_EPOCHS", 1),
            group_size=INIT_HP.get("GROUP_SIZE", 8),
            temperature=INIT_HP.get("TEMPERATURE", 0.9),
            calc_position_embeddings=INIT_HP.get("CALC_POSITION_EMBEDDINGS", True),
            reduce_memory_peak=INIT_HP.get("REDUCE_MEMORY_PEAK", False),
            max_output_tokens=INIT_HP.get("MAX_OUTPUT_TOKENS", 1024),
            min_output_tokens=INIT_HP.get("MIN_OUTPUT_TOKENS", None),
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
        for idx in range(init_hp.get("POP_SIZE"))
    ]
    for agent in population:
        if use_accelerator:
            agent.accelerator = accelerator

    for agent in population:
        # Create a mock clone that returns a new mock agent
        def mock_clone(new_idx, wrap=False):
            mock_agent = MagicMock()
            mock_agent.index = new_idx
            mock_agent.accelerator = accelerator
            mock_agent.clean_up = MagicMock()
            mock_agent.fitness = agent.fitness
            return mock_agent

        agent.clone = MagicMock(side_effect=mock_clone)

    population[0].fitness = [1, 2, 3]
    population[1].fitness = [4, 5, 6]
    population[2].fitness = [7, 8, 9]
    population[3].fitness = [10, 11, 12]

    # Call the select method
    elite, new_population = tournament_selection.select(population)

    # Check if the elite agent is the best agent in the population
    assert elite.fitness == [10, 11, 12]
    assert elite.index == 3

    # Check if the new population has the correct length
    assert len(new_population) == population_size

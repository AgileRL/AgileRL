import gymnasium as gym
import pytest

from agilerl.algorithms import CQN, DDPG, DQN, MADDPG, MATD3, PPO, TD3, RainbowDQN
from agilerl.hpo.tournament import TournamentSelection
from agilerl.utils.utils import create_population
from tests.helper_functions import (
    generate_discrete_space,
    generate_multi_agent_box_spaces,
    generate_multi_agent_discrete_spaces,
    generate_random_box_space,
)
from tests.test_algorithms.test_zgrpo import create_module

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
    net_config = {"encoder_config": {"hidden_size": [8, 8]}}
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
    net_config = {"encoder_config": {"hidden_size": [8, 8]}}
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
    net_config = {"encoder_config": {"hidden_size": [8, 8]}}
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
    net_config = {"encoder_config": {"hidden_size": [8, 8]}}
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


@pytest.mark.parametrize("elitism", [True, False])
def test_language_model_tournament(elitism):
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

    population = create_population(
        algo="GRPO",
        observation_space=observation_space,
        action_space=action_space,
        actor_network=actor_network,
        INIT_HP=init_hp,
        net_config=None,
        population_size=population_size,
        accelerator=None,
    )

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

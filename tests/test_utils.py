import copy
from unittest.mock import patch

import gymnasium as gym
import numpy as np

from agilerl.algorithms.cqn import CQN
from agilerl.algorithms.ddpg import DDPG
from agilerl.algorithms.dqn import DQN
from agilerl.algorithms.dqn_rainbow import RainbowDQN
from agilerl.algorithms.maddpg import MADDPG
from agilerl.algorithms.matd3 import MATD3
from agilerl.algorithms.ppo import PPO
from agilerl.algorithms.td3 import TD3
from agilerl.utils.utils import (
    calculate_vectorized_scores,
    initialPopulation,
    makeSkillVectEnvs,
    makeVectEnvs,
    plotPopulationScore,
    printHyperparams,
)
from agilerl.wrappers.learning import Skill

# Shared HP dict that can be used by any algorithm
SHARED_INIT_HP = {
    "POPULATION_SIZE": 4,
    "DOUBLE": True,
    "BATCH_SIZE": 128,
    "LR": 1e-3,
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
    "DISCRETE_ACTIONS": True,
    "GAE_LAMBDA": 0.95,
    "ACTION_STD_INIT": 0.6,
    "CLIP_COEF": 0.2,
    "ENT_COEF": 0.01,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    "TARGET_KL": None,
    "UPDATE_EPOCHS": 4,
    "MAX_ACTION": 1,
    "MIN_ACTION": -1,
    "N_AGENTS": 2,
    "AGENT_IDS": ["agent1", "agent2"],
    "CHANNELS_LAST": False,
}


SHARED_INIT_HP_MA = copy.deepcopy(SHARED_INIT_HP)
SHARED_INIT_HP_MA["MAX_ACTION"] = [(1,), (1,)]
SHARED_INIT_HP_MA["MIN_ACTION"] = [(-1,), (-1,)]


# Returns an AsyncVectorEnv object when given a valid environment name and number of environments
def test_returns_asyncvectorenv_object():
    num_envs = 3
    env = makeVectEnvs("CartPole-v1", num_envs=num_envs)
    assert isinstance(env, gym.vector.AsyncVectorEnv)
    assert env.num_envs == num_envs


# Returns an AsyncVectorEnv object when given a valid environment name and number of environments
def test_returns_asyncvectorenv_object_skill():
    num_envs = 3
    skill = Skill
    env = makeSkillVectEnvs("CartPole-v1", skill=skill, num_envs=num_envs)
    assert isinstance(env, gym.vector.AsyncVectorEnv)
    assert env.num_envs == num_envs


# Can create a population of agent for each single agent algorithm
def test_create_initial_population_single_agent():
    state_dim = [4]
    action_dim = 2
    one_hot = False
    net_config = {"arch": "mlp", "h_size": [8]}
    population_size = 4
    device = "cpu"
    accelerator = None

    algo_classes = {
        "DQN": DQN,
        "Rainbow DQN": RainbowDQN,
        "DDPG": DDPG,
        "TD3": TD3,
        "PPO": PPO,
        "CQN": CQN,
    }

    for algo in algo_classes.keys():
        population = initialPopulation(
            algo=algo,
            state_dim=state_dim,
            action_dim=action_dim,
            one_hot=one_hot,
            net_config=net_config,
            INIT_HP=SHARED_INIT_HP,
            population_size=population_size,
            device=device,
            accelerator=accelerator,
        )
        assert len(population) == population_size
        for agent in population:
            assert isinstance(agent, algo_classes[algo])
            assert agent.state_dim == state_dim
            assert agent.action_dim == action_dim
            assert agent.one_hot == one_hot
            assert agent.net_config == net_config
            assert agent.device == "cpu"
            assert agent.accelerator is None


# Can create a population of agent for each multi agent algorithm
def test_create_initial_population_multi_agent():
    state_dim = [[4], [4]]
    action_dim = [2, 2]
    one_hot = False
    net_config = {"arch": "mlp", "h_size": [8]}
    population_size = 4
    device = "cpu"
    accelerator = None

    algo_classes = {
        "MADDPG": MADDPG,
        "MATD3": MATD3,
    }

    for algo in algo_classes.keys():
        population = initialPopulation(
            algo=algo,
            state_dim=state_dim,
            action_dim=action_dim,
            one_hot=one_hot,
            net_config=net_config,
            INIT_HP=SHARED_INIT_HP_MA,
            population_size=population_size,
            device=device,
            accelerator=accelerator,
        )
        assert len(population) == population_size
        for agent in population:
            assert isinstance(agent, algo_classes[algo])
            assert agent.state_dims == state_dim
            assert agent.action_dims == action_dim
            assert agent.one_hot == one_hot
            assert agent.net_config == net_config
            assert agent.device == "cpu"
            assert agent.accelerator is None


# The function returns a list of episode rewards from the first episode in each parallel environment.
def test_returns_list_of_episode_rewards():
    rewards = np.array([[1, 2, 3, 4, 5], [4, 5, 6, 7, 8]])
    terminations = np.array([[0, 0, 1, 0, 1], [0, 1, 0, 0, 0]])
    expected_rewards = [6, 9]

    result = calculate_vectorized_scores(
        rewards, terminations, include_unterminated=False, only_first_episode=True
    )

    assert result == expected_rewards


# The function returns a list of episode rewards including all episodes.
def test_returns_list_of_episode_rewards_including_unterminated():
    rewards = np.array([[1, 2, 3], [4, 5, 6]])
    terminations = np.array([[0, 0, 1], [0, 1, 0]])
    expected_rewards = [6, 9, 6]

    result = calculate_vectorized_scores(
        rewards, terminations, include_unterminated=True, only_first_episode=False
    )

    assert result == expected_rewards


# The function returns a list of episode rewards including all terminated episodes.
def test_returns_list_of_episode_rewards_all_terminated_episodes():
    rewards = np.array([[1, 2, 3, 4, 5], [4, 5, 6, 7, 8]])
    terminations = np.array([[0, 0, 1, 0, 1], [0, 1, 0, 0, 0]])
    expected_rewards = [6, 9, 9]

    result = calculate_vectorized_scores(
        rewards, terminations, include_unterminated=False, only_first_episode=False
    )

    assert result == expected_rewards


# The function returns a list of episode rewards including all terminated episodes.
def test_returns_list_of_episode_rewards_including_all_terminated_episodes():
    rewards = np.array([[1, 2, 3, 4, 5], [4, 5, 6, 7, 8]])
    terminations = np.array([[0, 0, 1, 0, 1], [0, 1, 0, 0, 0]])
    expected_rewards = [6, 9, 9]

    result = calculate_vectorized_scores(
        rewards, terminations, include_unterminated=False, only_first_episode=False
    )

    assert result == expected_rewards


# The function returns a list of episode rewards containing no terminated episodes.
def test_returns_list_of_episode_rewards_with_no_terminations():
    rewards = np.array([[1, 2, 3, 4, 5], [4, 5, 6, 7, 8]])
    terminations = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
    expected_rewards = [15, 30]

    result = calculate_vectorized_scores(
        rewards, terminations, include_unterminated=False, only_first_episode=False
    )

    assert result == expected_rewards


# The function prints the hyperparameters and fitnesses of all agents in the population.
def test_prints_hyperparams():
    # Arrange
    state_dim = [4]
    action_dim = 2
    one_hot = False
    net_config = {"arch": "mlp", "h_size": [8]}
    population_size = 1
    device = "cpu"
    accelerator = None
    algo = "DQN"

    pop = initialPopulation(
        algo=algo,
        state_dim=state_dim,
        action_dim=action_dim,
        one_hot=one_hot,
        net_config=net_config,
        INIT_HP=SHARED_INIT_HP,
        population_size=population_size,
        device=device,
        accelerator=accelerator,
    )

    # Manually set attributes
    pop[0].fitness = [1, 2, 3]
    pop[0].lr = 0.01
    pop[0].batch_size = 32

    expected_output = (
        "Agent ID: 0    Mean 100 fitness: 2.00    lr: 0.01    Batch Size: 32"
    )

    with patch("builtins.print") as mock_print:
        printHyperparams(pop)
        mock_print.assert_called_once_with(expected_output)


# The function should correctly plot the fitness scores of all agents in the population.
@patch("agilerl.utils.utils.plt")
def test_plot_fitness_scores_all_agents(mock_plt):
    # Create a population of agents with fitness scores
    class Agent:
        def __init__(self, fitness):
            self.fitness = fitness
            self.steps = list(range(len(fitness) + 1))

    pop = [Agent([1, 2, 3]), Agent([4, 5, 6]), Agent([7, 8, 9])]

    # Call the function under test
    plotPopulationScore(pop)

    # Assert plotting functions have been called with expected args
    mock_plt.title.assert_called_once_with("Score History - Mutations")
    mock_plt.xlabel.assert_called_once_with("Steps")

    # Assert plt.figure got called
    assert mock_plt.figure.called

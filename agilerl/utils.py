import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from agilerl.algorithms.dqn import DQN
from agilerl.algorithms.ddpg import DDPG

def makeVectEnvs(env_name, num_envs=1):
    """Returns async-vectorized gym environments.

    :param env_name: Gym environment name
    :type env_name: str
    :param num_ens: Number of vectorized environments, defaults to 1
    :type num_envs: int, optional
    """
    return gym.vector.AsyncVectorEnv([lambda: gym.make(env_name) for i in range(num_envs)])
 

def initialPopulation(algo, num_states, num_actions, INIT_HP, population_size=1, device='cpu'):
    """Returns population of identical agents.
    
    :param algo: RL algorithm
    :type algo: str
    :param num_states: State observation dimension
    :type num_states: int
    :param num_actions: Action dimension
    :type num_actions: int
    :param INIT_HP: Initial hyperparameters
    :type INIT_HP: dict
    :param population_size: Number of agents in population, defaults to 1
    :type population_size: int, optional
    :param device: Device for accelerated computing, 'cpu' or 'cuda', defaults to 'cpu'
    :type device: str, optional
    """
    population = []

    if algo == 'DQN':
        for idx in range(population_size):
            agent = DQN(
                n_states = num_states,
                n_actions = num_actions,
                index = idx,
                h_size = INIT_HP['HIDDEN_SIZE'],
                batch_size = INIT_HP['BATCH_SIZE'],
                lr = INIT_HP['LR'],
                gamma = INIT_HP['GAMMA'],
                tau = INIT_HP['TAU'],
                device=device
                )
            population.append(agent)

    elif algo == 'DDPG':
        for idx in range(population_size):
            agent = DDPG(
                n_states = num_states,
                n_actions = num_actions,
                index = idx,
                h_size = INIT_HP['HIDDEN_SIZE'],
                batch_size = INIT_HP['BATCH_SIZE'],
                lr = INIT_HP['LR'],
                gamma = INIT_HP['GAMMA'],
                tau = INIT_HP['TAU'],
                policy_freq = INIT_HP['POLICY_FREQ'],
                device=device
                )
            population.append(agent)

    return population

def printHyperparams(pop):
    """Prints current hyperparameters of agents in a population and their fitnesses.

    :param pop: Population of agents
    :type pop: List[object]
    """
    for agent in pop:
        print('Agent ID: {}    Mean 100 fitness: {:.2f}    lr: {}    Batch Size: {}'.format(agent.index, np.mean(agent.fitness[-100:]), agent.lr, agent.batch_size))

def plotPopulationScore(pop):
    """Plots the fitness scores of agents in a population.

    :param pop: Population of agents
    :type pop: List[object]
    """
    plt.figure()
    for agent in pop:
        scores = agent.fitness
        steps = agent.steps[:-1]
        plt.plot(steps, scores)
    plt.title("Score History - Mutations")
    plt.xlabel("Steps")
    plt.ylim(bottom=-400)
    plt.show()
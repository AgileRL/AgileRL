import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from agilerl.algorithms.dqn import DQN
from agilerl.algorithms.ddpg import DDPG

def makeVectEnvs(env_name, num_envs):
    return gym.vector.AsyncVectorEnv([lambda: gym.make(env_name) for i in range(num_envs)])
 

def initialPopulation(algo, num_states, num_actions, INIT_HP, population_size=1, device='cpu'):
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
                learn_step = INIT_HP['LEARN_STEP'],
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
                learn_step = INIT_HP['LEARN_STEP'],
                tau = INIT_HP['TAU'],
                policy_freq = INIT_HP['POLICY_FREQ'],
                device=device
                )
            population.append(agent)

    return population

def printHyperparams(pop):
    for agent in pop:
        print('Agent ID: {}    Mean 100 fitness: {:.2f}    lr: {}    Batch Size: {}'.format(agent.index, np.mean(agent.fitness[-100:]), agent.lr, agent.batch_size))
    
def plotScore(scores, update_freq):
    episodes = [i*update_freq for i, x in enumerate(scores)]
    plt.figure()
    plt.plot(episodes, scores)
    plt.title("Score History")
    plt.xlabel("Episodes")
    plt.show()

def plotPopulationScore(pop):
    plt.figure()
    for agent in pop:
        scores = agent.fitness
        steps = agent.steps[:-1]
        plt.plot(steps, scores)
    plt.title("Score History - Mutations")
    plt.xlabel("Steps")
    plt.ylim(bottom=-400)
    plt.show()
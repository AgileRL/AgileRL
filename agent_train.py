import torch
from agilerl.components.replay_buffer import ReplayBuffer
from agilerl.algorithms.td3 import TD3
from agilerl.utils.utils import makeVectEnvs

import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(x, scores):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.xlabel('Number of episodes')
    plt.xlabel('Mean reward')
    plt.show()


def agent_test():
    env = makeVectEnvs('LunarLanderContinuous-v2', num_envs=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_action = float(env.single_action_space.high[0])


    try:
        state_dim = env.single_observation_space.n          # Discrete observation space
        one_hot = True                                      # Requires one-hot encoding
    except BaseException:
        state_dim = env.single_observation_space.shape      # Continuous observation space
        one_hot = False                                     # Does not require one-hot encoding
    try:
        action_dim = env.single_action_space.n              # Discrete action space
    except BaseException:
        action_dim = env.single_action_space.shape[0]       # Continuous action space

    NET_CONFIG = {
      'arch': 'mlp',      # Network architecture
      'h_size': [400, 300]  # Network hidden size
    }

    agent = TD3(state_dim=state_dim,
                action_dim=action_dim,
                one_hot=one_hot,
                max_action=max_action,
                index=0,
                net_config=NET_CONFIG,
                batch_size=100,
                tau=0.005,
                lr = 0.001,
                device=torch.device(device))
    
    field_names = ["state", "action", "reward", "next_state", "done"]
    memory = ReplayBuffer(
        action_dim, 100000, field_names=field_names, device=torch.device(device))

    np.random.seed(0)
    score_history = []
    episodes = 1000
    for i in range(episodes):
        done = [False]
        score = 0
        state = env.reset()[0]

        while not done[0]:
            action = agent.getAction(state)
            next_state, reward, done, _, _ = env.step(action)
            memory.save2memoryVectEnvs(state, action, reward, next_state, done)
                    
            # Learn according to learning frequency
            if memory.counter % agent.learn_step == 0 and len(
                    memory) >= agent.batch_size:
                experiences = memory.sample(
                    agent.batch_size)   # Sample replay buffer
                # Learn according to agent's RL algorithm
                agent.learn(experiences)

            score += reward 
            state = next_state
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        print('episode', i, 'score', avg_score,
          'average score', avg_score)

    x = [i+1 for i in range(episodes)]
    plot_learning_curve(x, score_history)

if __name__ == "__main__":
    agent_test()
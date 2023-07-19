from pettingzoo.mpe import simple_adversary_v3
from agilerl.algorithms.maddpg import MADDPG
import torch
import numpy as np


if __name__ == "__main__":
    # Device agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Configure the environment
    env = simple_adversary_v3.parallel_env(N=2, max_cycles=25, continuous_actions=True)
    env.reset(seed=42)
    # Print information about the agents
    # print('Number of agents: ', len(env.agents))
    # for agent in env.agents:
    #     print(f"Observations space for {agent}: ", env.observation_space(agent))
    # for agent in env.agents:
    #     print(f"Action space for {agent}: ", env.action_space(agent))
    

    n_agents = env.num_agents
    # [action_agent_1, action_agent_2, ..., action_agent_n]
    action_dims = [env.action_space(agent).shape[0] for agent in env.agents]
    # [state_agent_1, state_agent_2, ..., state_agent_n]
    state_dims = [env.observation_space(agent).shape[0] for agent in env.agents]
    print(action_dims, state_dims)
    one_hot = False 
    index=0
    net_config={'arch': 'mlp', 'h_size': [64,64]}
    batch_size=64
    lr=1e-4
    learn_step=5
    gamma=0.99
    tau=1e-3
    mutation=None
    policy_freq=2
    device=device
    accelerator=None 
    wrap=True

    # Instantiate MADDPG class
    maddpg_agent = MADDPG(state_dims=state_dims,
                   action_dims=action_dims,
                   one_hot=one_hot,
                   n_agents=n_agents,
                   index=index,
                   net_config=net_config,
                   batch_size=batch_size,
                   lr=lr,
                   learn_step=learn_step,
                   gamma=gamma,
                   tau=tau,
                   mutation=mutation,
                   policy_freq=policy_freq,
                   device=device,
                   accelerator=accelerator,
                   wrap=wrap) 
    
    step = 0
    agent_num = env.num_agents
    episodes = 1
    episode_rewards = {agent_id: np.zeros(episodes) for agent_id in env.agents}

    for ep in range(episodes):
        state = env.reset()
        agent_reward = {agent_id: 0 for agent_id in env.agents}
        
        while env.agents:
            print(f"This is the state: {state}")
            step += 1
            action = maddpg_agent.getAction(state)
            next_state, reward, done, info = env.step(action)
            print("next_state", next_state)
            print("reward", reward)
            print("done", done)


        # for agent in env.agent_iter():
        #     observation, reward, termination, truncation, _ = env.last()
        #     total_reward[agent] += reward
        #     if (termination or truncation):
        #         ep_rewards.append(total_reward[agent])
        #         action = None 
        #     else:

        #         action = maddpg_agent.getAction(observation)
        #     env.step(action[0])


            # state = env.reset(seed=42)
            # print(f"{state=}")
            # score = 0
            # done = [False]*n_agents

            # while not any(done):
            #     actions = agent.getAction(state)
            #     next_state, reward, done, _, _ = env.step(actions)
            #     print(next_state)





    

    



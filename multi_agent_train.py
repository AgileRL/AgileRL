from pettingzoo.mpe import simple_adversary_v3, simple_spread_v3, simple_v3, simple_speaker_listener_v4
from agilerl.algorithms.maddpg import MADDPG
from agilerl.components.replay_buffer import ReplayBuffer
from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
import torch
import numpy as np
import wandb
from datetime import datetime


if __name__ == "__main__":
    # Device agnostic code
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configure the environment
    env = simple_adversary_v3.parallel_env(N=2, max_cycles=25, continuous_actions=True)
    env.reset()

    # Print information about the agents
    # print('Number of agents: ', len(env.agents))
    # for agent in env.agents:
    #     print(f"Observations space for {agent}: ", env.observation_space(agent))
    # for agent in env.agents:
    #     print(f"Action space for {agent}: ", env.action_space(agent))
    

    n_agents = env.num_agents
    # [action_agent_1, action_agent_2, ..., action_agent_n]
    action_dims = [env.action_space(agent).shape[0] for agent in env.agents]
    max_action = [env.action_space(agent).high for agent in env.agents][0][0] # Assume all agents have the same max action space
    # [state_agent_1, state_agent_2, ..., state_agent_n]
    state_dims = [env.observation_space(agent).shape for agent in env.agents]
    agent_ids = [agent_id for agent_id in env.agents]
    one_hot = False 
    index = 0
    net_config = {'arch': 'mlp', 'h_size': [64,64]}
    batch_size = 1024
    critic_lr = 0.01
    actor_lr = 0.01
    learn_step = 100
    gamma = 0.95
    tau = 0.01
    mutation=None
    policy_freq=2
    device=device
    accelerator=None 
    wrap=True

    # Instantiate MADDPG class
    maddpg_agent = MADDPG(state_dims=state_dims,
                   action_dims=action_dims,
                   max_action=max_action,
                   one_hot=one_hot,
                   n_agents=n_agents,
                   agent_ids=agent_ids,
                   index=index,
                   environment=env,
                   net_config=net_config,
                   batch_size=batch_size,
                   actor_lr=actor_lr,
                   critic_lr=critic_lr,
                   learn_step=learn_step,
                   gamma=gamma,
                   tau=tau,
                   mutation=mutation,
                   policy_freq=policy_freq,
                   device=device,
                   accelerator=accelerator,
                   wrap=wrap) 
    
    step = 0 # Global step counter
    wb = False
    agent_num = env.num_agents
    episodes = 40_000
    epsilon = 1
    epsilon_end = 0.1
    epsilon_decay = 0.995
    episode_rewards = {agent_id: np.zeros(episodes) for agent_id in env.agents}
    field_names = ["state", "action", "reward", "next_state", "done"]
    ma_buffer = MultiAgentReplayBuffer(field_names=field_names, memory_size= 1000000, 
                                       agent_ids=agent_ids, device=device)
    reward_history = []

    # Initialise weights and biases
    if wb:
        print("... Initialsing W&B ...")
        wandb.init(
            project="MADDPG Testing",
            name=f"MADDPG_simple_adversary_v3_{datetime.now().strftime('%m%d%Y%H%M%S')}",
            config = {
                "algo": "MADDPG",
                "env": "simple_adversary_v3",
                "arch": net_config,
                "gamma": gamma,
                "critic_lr": critic_lr,
                "actor_lr": actor_lr,
                "tau": tau,
                "batch_size":batch_size,
                "output activation": 'softmax',
                "detail": "Test using multi-agent buffer"
            }

        )

    for ep in range(episodes):
        state, _ = env.reset()
        agent_reward = {agent_id: 0 for agent_id in env.agents}

        while env.agents:
            step += 1
            action = maddpg_agent.getAction(state, epsilon)
            # These are dictionaries of format: n_s = {agent_i: n_s_i,...,...}
            next_state, reward, done, info, _ = env.step(action)
            print(done)
            print(state)

            # Save experiences to the buffer
            ma_buffer.save2memory(state, action, reward, next_state, done)

            # Save each agents' reward 
            for agent_id, r in reward.items():
                agent_reward[agent_id] += r

            # Sample from the buffer and then learn the maddpg
            if ma_buffer.counter % maddpg_agent.learn_step == 0 and len(
                ma_buffer) >= maddpg_agent.batch_size:
                experiences = ma_buffer.sample(batch_size)
                maddpg_agent.learn(experiences)
 
            # Update the state 
            state = next_state

        # Update epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # Episode finishes, record rewards
        for agent_id, r in agent_reward.items():
            episode_rewards[agent_id][ep] = r

        # Calculate total reward of episode
        score = sum(agent_reward.values())
        reward_history.append(score)

        # Append the score to the agent object
        maddpg_agent.scores.append(score)
        
        # Print every 100 episodes
        if (ep + 1) % 100 == 0:
            if wb:
                wandb.log(
                    {"episode": ep + 1,
                     "global steps": step,
                     "total_reward":score,
                     "mean_reward": np.mean(reward_history[-100:])}
                )
            else:
                print(f"-----------------------------------------------")
                print(f"Total reward {ep + 1}: {score} | Mean reward: {np.mean(reward_history[-100:])}")
                print(f"{episode_rewards}")


        
        





    

    



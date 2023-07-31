from pettingzoo.mpe import simple_adversary_v3, simple_spread_v3, simple_v3, simple_speaker_listener_v4
from agilerl.algorithms.maddpg import MADDPG
from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
import torch
import numpy as np
import wandb
from datetime import datetime
#from pettingzoo.atari import basketball_pong_v3


if __name__ == "__main__":
    # Device agnostic code
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configure the environment
    env = simple_speaker_listener_v4.parallel_env(max_cycles=25, continuous_actions=True)
    env.reset()

    # Configure maddpg input arguments
    n_agents = env.num_agents
    action_dims = [env.action_space(agent).shape[0] for agent in env.agents] # [action_agent_1, action_agent_2, ..., action_agent_n]
    state_dims = [env.observation_space(agent).shape for agent in env.agents] # [state_agent_1, state_agent_2, ..., state_agent_n]
    agent_ids = [agent_id for agent_id in env.agents]
    max_action = [env.action_space(agent).high for agent in env.agents]
    min_action = [env.action_space(agent).low for agent in env.agents]

    one_hot = False 
    index = 0
    net_config = {'arch': 'mlp', 'h_size': [64,64]}
    batch_size = 1024
    critic_lr = 0.01
    actor_lr = 0.01
    learn_step = 100
    gamma = 0.95
    tau = 0.01
    device=device
    accelerator=None 

    # Instantiate MADDPG class
    maddpg_agent = MADDPG(state_dims=state_dims,
                   action_dims=action_dims,
                   one_hot=one_hot,
                   n_agents=n_agents,
                   agent_ids=agent_ids,
                   index=index,
                   max_action = max_action,
                   min_action = min_action,
                   net_config=net_config,
                   batch_size=batch_size,
                   actor_lr=actor_lr,
                   critic_lr=critic_lr,
                   learn_step=learn_step,
                   gamma=gamma,
                   tau=tau,
                   device=device,
                   accelerator=accelerator) 
    
    # Configure the training loop parameters
    step = 0 # Global step counter
    wb = True # Initiate weights and biases
    agent_num = env.num_agents
    episodes = 40_000
    epsilon = 1
    epsilon_end = 0.1
    epsilon_decay = 0.995
    episode_rewards = {agent_id: np.zeros(episodes) for agent_id in env.agents}
    field_names = ["state", "action", "reward", "next_state", "done"]
    memory = MultiAgentReplayBuffer(field_names=field_names, memory_size= 1000000, 
                                       agent_ids=agent_ids, device=device)
    reward_history = []

    # Initialise weights and biases
    if wb:
        print("... Initialsing W&B ...")
        wandb.init(
            project="MADDPG Testing",
            name=f"MADDPG_simple_speaker_listener_v4_min_action_test_{datetime.now().strftime('%m%d%Y%H%M%S')}",
            config = {
                "algo": "MADDPG",
                "env": "simple_speaker_listener_v4",
                "arch": net_config,
                "gamma": gamma,
                "critic_lr": critic_lr,
                "actor_lr": actor_lr,
                "tau": tau,
                "batch_size":batch_size,
                "output activation": 'softmax',
                "detail": "Testing min_action"
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

            # Save experiences to the buffer
            memory.save2memory(state, action, reward, next_state, done)

            # Save each agents' reward 
            for agent_id, r in reward.items():
                agent_reward[agent_id] += r

            # Sample from the buffer and then learn the maddpg
            if memory.counter % maddpg_agent.learn_step == 0 and len(
                memory) >= maddpg_agent.batch_size:
                experiences = memory.sample(batch_size)
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
        
        # Log to wandb or print to terminal if wandb not selected
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


        
        





    

    



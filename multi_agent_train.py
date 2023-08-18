from pettingzoo.mpe import simple_adversary_v3, simple_spread_v3, simple_v3, simple_speaker_listener_v4
from agilerl.algorithms.maddpg import MADDPG
from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
import torch
import numpy as np
import wandb
from datetime import datetime
from pettingzoo.atari import basketball_pong_v3, boxing_v2, space_invaders_v2


if __name__ == "__main__":
    # Device agnostic code
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    atari = False
    if atari:
        env = space_invaders_v2.parallel_env()
        env.reset()
        discrete_actions = True
        n_agents = env.num_agents
        agent_ids = [agent_id for agent_id in env.agents]
        action_dims = [env.action_space(agent).n for agent in env.agents]
        state_dims = [env.observation_space(agent).shape for agent in env.agents]
        max_action, min_action = None, None
        channels_last = True
        net_config =  {'arch': 'cnn','c_size': [3,16], 'normalize':True, 'k_size': [(1,3,3),(1,3,3)], 's_size':[2,2], 'h_size': [32,32]}
        
    else:
        # Configure the environment for mpe
        env = simple_speaker_listener_v4.parallel_env(max_cycles=25, continuous_actions=True)
        env.reset()
        discrete_actions = False
        # Configure continuous maddpg input arguments
        n_agents = env.num_agents
        agent_ids = [agent_id for agent_id in env.agents]
        action_dims = [env.action_space(agent).shape[0] for agent in env.agents] # [action_agent_1, action_agent_2, ..., action_agent_n]
        state_dims = [env.observation_space(agent).shape for agent in env.agents] # [state_agent_1, state_agent_2, ..., state_agent_n]
        max_action = [env.action_space(agent).high for agent in env.agents]
        min_action = [env.action_space(agent).low for agent in env.agents]
        channels_last = False
        net_config = {"arch": "mlp", "h_size":[32,32]}

    print(f"{agent_ids}")
    print(f"{action_dims=}")
    print(f"{state_dims=}")   

    if channels_last:
        state_dims = [(state_dim[2], state_dim[0], state_dim[1]) for state_dim in state_dims]

    one_hot = False 
    index = 0
    batch_size = 128
    lr = 0.01
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
                   discrete_actions=True,
                   max_action = max_action,
                   min_action = min_action,
                   net_config=net_config,
                   batch_size=batch_size,
                   lr=lr,
                   learn_step=learn_step,
                   gamma=gamma,
                   tau=tau,
                   device=device,
                   accelerator=accelerator) 
    
    
    # Configure the training loop parameters
    step = 0 # Global step counter
    wb = False # Initiate weights and biases
    agent_num = env.num_agents
    episodes = 20000
    epsilon = 1
    epsilon_end = 0.1
    epsilon_decay = 0.995
    episode_rewards = {agent_id: [] for agent_id in env.agents}
    field_names = ["state", "action", "reward", "next_state", "done"]
    memory = MultiAgentReplayBuffer(field_names=field_names, memory_size= 1000000, 
                                       agent_ids=agent_ids, device=device)
    reward_history = []

    # Initialise weights and biases
    if wb:
        print("... Initialsing W&B ...")
        wandb.init(
            project="MADDPG Testing",
            name=f"simple_speaker_listener_v4_custom_training_loop_MADDPG_LEG_{datetime.now().strftime('%m%d%Y%H%M%S')}",
            config = {
                "algo": "MADDPG",
                "env": "space_invaders_v4",
                "arch": net_config,
                "gamma": gamma,
                "tau": tau,
                "batch_size":batch_size,
                "output activation": 'softmax',
                "detail": "Testing min_action"
            }
        )

    for ep in range(episodes):
        state, _ = env.reset()
        agent_reward = {agent_id: 0 for agent_id in env.agents}

        if channels_last:
                state = {agent_id: np.moveaxis(np.expand_dims(s, 0), [3], [1]) for agent_id, s in state.items()}
        
        for _ in range(25):
            step += 1
            #print([s.shape for s in state.values()])
            action = maddpg_agent.getAction(state, epsilon=1)
            rand_action = {agent: env.action_space(agent).sample() for agent in agent_ids}
            # These are dictionaries of format: n_s = {agent_i: n_s_i,...,...}
            next_state, reward, done, info, _ = env.step(action)


            # Save experiences to the buffer
            if channels_last:
                state = {agent_id: np.squeeze(s) for agent_id, s in state.items()}
                next_state = {agent_id: np.moveaxis(ns, [2], [0]) for agent_id, ns in next_state.items()}
 
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
            if channels_last:
                next_state = {agent_id: np.expand_dims(ns,0) for agent_id, ns in next_state.items()}
            state = next_state


        # Update epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # Episode finishes, record rewards
        for agent_id in agent_ids:
            episode_rewards[agent_id].append(agent_reward[agent_id])

        # Calculate total reward of episode
        score = sum(agent_reward.values())
        reward_history.append(score)

        # Append the score to the agent object
        maddpg_agent.scores.append(score)
        
        # Log to wandb or print to terminal if wandb not selected
        if (ep + 1) % 1 == 0:
            if wb:
                wandb.log(
                    {"episode": ep + 1,
                     "global steps": step,
                     "total_reward":score,
                     "mean_reward": np.mean(reward_history[-100:]),}
                )
                for agent in agent_ids:
                    wandb.log(
                        {f"{agent}_reward": episode_rewards[agent][-1],
                         f"{agent}_mean_reward": np.mean(episode_rewards[agent][-100:])}
                    )
            else:
                print(f"-----------------------------------------------")
                print(f"Total reward {ep + 1}: {score} | Mean reward: {np.mean(reward_history[-100:])}")
                #print(f"{episode_rewards}")


        
        





    

    



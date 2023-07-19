from pettingzoo.mpe import simple_adversary_v3
from agilerl.algorithms.maddpg import MADDPG
import torch


if __name__ == "__main__":
    # Device agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Configure the environment
    env = simple_adversary_v3.env(N=2, max_cycles=25, continuous_actions=True)
    env.reset()

    # Print information about the agents
    print('Number of agents: ', len(env.agents))
    for agent in env.agents:
        print(f"Observations space for {agent}: ", env.observation_space(agent))
    for agent in env.agents:
        print(f"Action space for {agent}: ", env.action_space(agent))
    

    n_agents = len(env.agents)
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
    agent = MADDPG() # Fill in arguments 
    episodes = 500
    for i in env.agent_iter(episodes):
        state, _ = env.reset()
        score = 0
        done = [False]*n_agents

        while not any(done):
            actions = agent.getAction(state)
            next_state, reward, done, _, _ = env.step(actions)
            # state = func_to





    # Define the hyperparameters
    # state_dim = 3
    # action_dim =  env.action
    # one_hot = False 
    # n_agents = env.n 
    # index=0
    # net_config={'arch': 'mlp', 'h_size': [64,64]}
    # batch_size=64
    # lr=1e-4
    # learn_step=5
    # gamma=0.99
    # tau=1e-3
    # mutation=None
    # policy_freq=2
    # device='cpu'
    # accelerator=None 
    # wrap=True

    



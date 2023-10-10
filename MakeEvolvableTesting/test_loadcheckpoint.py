import numpy as np
import torch
from tqdm import trange

from agilerl.components.replay_buffer import ReplayBuffer
from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.utils.utils import initialPopulation, makeVectEnvs

import torch.nn as nn
from agilerl.wrappers.make_evolvable import MakeEvolvable
import os
from pettingzoo.mpe import simple_speaker_listener_v4

# !Note: If you are running this demo without having installed agilerl,
# uncomment and place the following above agilerl imports:

# import sys
# sys.path.append('../')

class BasicNetActor(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(BasicNetActor, self).__init__()
        layers = []

        # Add input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())  # Activation function

        # Add hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.ReLU())  # Activation function

        # Add output layer with a sigmoid activation
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        layers.append(nn.Softmax())  # Sigmoid activation

        # Combine all layers into a sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
class BasicNetCritic(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(BasicNetCritic, self).__init__()
        layers = []

        # Add input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())  # Activation function

        # Add hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.ReLU())  # Activation function

        # Add output layer with a sigmoid activation
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        #layers.append(nn.Sigmoid())  # Sigmoid activation

        # Combine all layers into a sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    print("===== AgileRL Online Demo =====")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    INIT_HP = {
        "POPULATION_SIZE": 1,  # Population size
        "DOUBLE": True,  # Use double Q-learning
        "BATCH_SIZE": 128,  # Batch size
        "LR": 1e-3,  # Learning rate
        "GAMMA": 0.99,  # Discount factor
        "LEARN_STEP": 1,  # Learning frequency
        "TAU": 1e-3,  # For soft update of target network parameters
        # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]
        "CHANNELS_LAST": False,
        "POLICY_FREQ": 2,
        "GAE_LAMBDA": 0.95,           # Lambda for general advantage estimation
        "ACTION_STD_INIT": 0.6,       # Initial action standard deviation
        "CLIP_COEF": 0.2,             # Surrogate clipping coefficient
        "ENT_COEF": 0.01,             # Entropy coefficient
        "VF_COEF": 0.5,               # Value function coefficient
        "MAX_GRAD_NORM": 0.5,         # Maximum norm for gradient clipping
        "UPDATE_EPOCHS": 4,           # Number of policy update epochs
        "TARGET_KL" : None
    }

    # env = makeVectEnvs("LunarLander-v2", num_envs=8)  # Create environment
    # try:
    #     state_dim = env.single_observation_space.n  # Discrete observation space
    #     one_hot = True  # Requires one-hot encoding
    # except Exception:
    #     state_dim = env.single_observation_space.shape  # Continuous observation space
    #     one_hot = False  # Does not require one-hot encoding
    # try:
    #     action_dim = env.single_action_space.n  # Discrete action space
    # except Exception:
    #     action_dim = env.single_action_space.shape[0]  # Continuous action space

    # if INIT_HP["CHANNELS_LAST"]:
    #     state_dim = (state_dim[2], state_dim[0], state_dim[1])

    env = simple_speaker_listener_v4.parallel_env(
            max_cycles=25, continuous_actions=True
        )
    env.reset()
    
    # Configure the multi-agent algo input arguments
    try:
        state_dims = [env.observation_space(agent).n for agent in env.agents]
        one_hot = True
    except Exception:
        state_dims = [env.observation_space(agent).shape for agent in env.agents]
        one_hot = False
    try:
        action_dims = [env.action_space(agent).n for agent in env.agents]
        INIT_HP["DISCRETE_ACTIONS"] = True
        INIT_HP["MAX_ACTION"] = None
        INIT_HP["MIN_ACTION"] = None
    except Exception:
        action_dims = [env.action_space(agent).shape[0] for agent in env.agents]
        INIT_HP["DISCRETE_ACTIONS"] = False
        INIT_HP["MAX_ACTION"] = [env.action_space(agent).high for agent in env.agents]
        INIT_HP["MIN_ACTION"] = [env.action_space(agent).low for agent in env.agents]

    if INIT_HP["CHANNELS_LAST"]:
        state_dims = [
            (state_dim[2], state_dim[0], state_dim[1]) for state_dim in state_dims
        ]

    INIT_HP["N_AGENTS"] = env.num_agents
    INIT_HP["AGENT_IDS"] = [agent_id for agent_id in env.agents]

    

    INIT_HP["DISCRETE_ACTIONS"] = True

    actors = [
            MakeEvolvable(BasicNetActor(state_dim[0],[64, 64], action_dim),
                          input_tensor=torch.ones(state_dim[0]),
                          device=device)
            for action_dim, state_dim in zip(action_dims, state_dims)
    ]
    total_state_dims = sum(state_dim[0] for state_dim in state_dims)
    total_actions = sum(action_dims)
    critics = [
        MakeEvolvable(BasicNetCritic(total_state_dims + total_actions, [64, 64], 1),
                        input_tensor=torch.ones(total_state_dims + total_actions),
                        device=device)
        for _ in range(INIT_HP["N_AGENTS"])
    ]
    
    

    pop = initialPopulation(
        algo="MADDPG",  # Algorithm
        state_dim=state_dims,  # State dimension
        action_dim=action_dims,  # Action dimension
        one_hot=one_hot,  # One-hot encoding
        net_config=None,  # Network configuration
        INIT_HP=INIT_HP,  # Initial hyperparameters
        actor_network=actors,
        critic_network=critics,
        population_size=INIT_HP["POPULATION_SIZE"],  # Population size
        device=device
    )

    field_names = ["state", "action", "reward", "next_state", "done"]
    # Single agent
    # memory = ReplayBuffer(
    #     action_dim=action_dim,  # Number of agent actions
    #     memory_size=10000,  # Max replay buffer size
    #     field_names=field_names,  # Field names to store in memory
    #     device=device,
    # )

    # Multi agent
    memory = MultiAgentReplayBuffer(1000000, field_names, INIT_HP["AGENT_IDS"], device)

    tournament = TournamentSelection(
        tournament_size=2,  # Tournament selection size
        elitism=True,  # Elitism in tournament selection
        population_size=INIT_HP["POPULATION_SIZE"],  # Population size
        evo_step=1,
    )  # Evaluate using last N fitness scores

    mutations = Mutations(
        algo="MADDPG",  # Algorithm
        no_mutation=0.4,  # No mutation
        architecture=0.2,  # Architecture mutation
        new_layer_prob=0.2,  # New layer mutation
        parameters=0.2,  # Network parameters mutation
        activation=0,  # Activation layer mutation
        rl_hp=0.2,  # Learning HP mutation
        rl_hp_selection=["lr", "batch_size"],  # Learning HPs to choose from
        mutation_sd=0.1,  # Mutation strength
        arch=actors[0].arch,  # Network architecture
        rand_seed=1,  # Random seed
        device=device,
    )

    max_episodes = 1000  # Max training episodes
    max_steps = 500  # Max steps per episode

    # Exploration params
    eps_start = 1.0  # Max exploration
    eps_end = 0.1  # Min exploration
    eps_decay = 0.995  # Decay per episode
    epsilon = eps_start

    evo_epochs = 5  # Evolution frequency
    evo_loop = 1  # Number of evaluation episodes

    print("Training...")
    directory_name = "Models"
    os.makedirs(directory_name, exist_ok=True)
    # Define the filename you want to save
    file_name = "MADDPG.pt"

    # Construct the full path to the file
    file_path = os.path.join(directory_name, file_name)

    # TRAINING LOOP
    for idx_epi in trange(max_episodes):
        for agent in pop:  # Loop through population
            # state = env.reset()[0]  # Reset environment at start of episode
            # score = 0
            # for idx_step in range(max_steps):
            #     # Get next action from agent
            #     action = agent.getAction(state, epsilon)
            #     next_state, reward, done, _, _ = env.step(action)  # Act in environment

            #     # Save experience to replay buffer
            #     if INIT_HP["CHANNELS_LAST"]:
            #         memory.save2memoryVectEnvs(
            #             state, action, reward, np.moveaxis(next_state, [3], [1]), done
            #         )
            #     else:
            #         memory.save2memoryVectEnvs(state, action, reward, next_state, done)

            #     # Learn according to learning frequency
            #     if (
            #         memory.counter % agent.learn_step == 0
            #         and len(memory) >= agent.batch_size
            #     ):
            #         experiences = memory.sample(
            #             agent.batch_size
            #         )  # Sample replay buffer
            #         # Learn according to agent's RL algorithm
            #         agent.learn(experiences)

            #     state = next_state
            #     score += reward
            state = env.reset()[0]  # Reset environment at start of episode
            swap_channels = False
            agent_reward = {agent_id: 0 for agent_id in env.agents}
            if swap_channels:
                state = {
                    agent_id: np.moveaxis(np.expand_dims(s, 0), [3], [1])
                    for agent_id, s in state.items()
                }

            for _ in range(max_steps):
                # Get next action from agent
                action = agent.getAction(state, epsilon)
                print(action)
                print(env)
                next_state, reward, done, truncation, _ = env.step(
                    action
                )  # Act in environment

                # Save experience to replay buffer
                if swap_channels:
                    state = {agent_id: np.squeeze(s) for agent_id, s in state.items()}
                    next_state = {
                        agent_id: np.moveaxis(ns, [2], [0])
                        for agent_id, ns in next_state.items()
                    }

                memory.save2memory(state, action, reward, next_state, done)

                for agent_id, r in reward.items():
                    agent_reward[agent_id] += r

                # Learn according to learning frequency
                if (memory.counter % agent.learn_step == 0) and (
                    len(memory) >= agent.batch_size
                ):
                    # Sample replay buffer
                    experiences = memory.sample(agent.batch_size)
                    # Learn according to agent's RL algorithm
                    agent.learn(experiences)

                # Update the state
                if swap_channels:
                    next_state = {
                        agent_id: np.expand_dims(ns, 0)
                        for agent_id, ns in next_state.items()
                    }
                state = next_state

                # Episode termination conditions
                if any(truncation.values()) or any(done.values()):
                    break

            score = sum(agent_reward.values())
            agent.scores.append(score)


        # Update epsilon for exploration
        epsilon = max(eps_end, epsilon * eps_decay)

        if idx_epi == 2:
            print("...saving agent")
            agent = pop[0]
            agent.saveCheckpoint(file_path)
            break

        # Now evolve population if necessary
        if (idx_epi + 1) % evo_epochs == 0:
            # Evaluate population
            fitnesses = [
                agent.test(
                    env,
                    swap_channels=INIT_HP["CHANNELS_LAST"],
                    max_steps=max_steps,
                    loop=evo_loop,
                )
                for agent in pop
            ]

            print(f"Episode {idx_epi+1}/{max_episodes}")
            print(f'Fitnesses: {["%.2f"%fitness for fitness in fitnesses]}')
            print(
                f'100 fitness avgs: {["%.2f"%np.mean(agent.fitness[-100:]) for agent in pop]}'
            )

            # Tournament selection and population mutation
            elite, pop = tournament.select(pop)
            pop = mutations.mutation(pop)

    env.close()


    model = pop = initialPopulation(
        algo="MADDPG",  # Algorithm
        state_dim=state_dims,  # State dimension
        action_dim=action_dims,  # Action dimension
        one_hot=one_hot,  # One-hot encoding
        net_config=None,  # Network configuration
        INIT_HP=INIT_HP,  # Initial hyperparameters
        actor_network=actors,
        critic_network=critics,
        population_size=INIT_HP["POPULATION_SIZE"],  # Population size
        device=device
    )[0]
    model.loadCheckpoint(file_path)
    print("Successfully loaded checkpoint", model.actor_network, model.critic_networks)


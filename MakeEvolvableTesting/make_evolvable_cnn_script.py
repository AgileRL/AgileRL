import torch.nn as nn
import numpy as np
import os
from agilerl.wrappers.make_evolvable import MakeEvolvable
from tqdm import trange
import torch
from agilerl.components.replay_buffer import ReplayBuffer
from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
from agilerl.hpo.mutation import Mutations
from agilerl.training.train_unvectorised import train_unvectorised
#from agilerl.training.train_on_policy import train
from agilerl.training.train_multi_agent import train_multi_agent
from agilerl.networks.custom_activation import GumbelSoftmax
from agilerl.training.train import train
from agilerl.hpo.tournament import TournamentSelection
from agilerl.utils.utils import initialPopulation, makeVectEnvs, printHyperparams
from pettingzoo.mpe import simple_speaker_listener_v4
import gymnasium as gym
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
import yaml
import supersuit as ss
import torch.nn as nn

from pettingzoo.atari import pong_v3

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

    
class SimpleCNNActor(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNNActor, self).__init__()

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=8, stride=4)    # W: 160, H: 210
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)   # W:

        # Define the max-pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Define fully connected layers
        self.fc1 = nn.Linear(2592, 256)  # Assuming input images are 128x128
        self.fc2 = nn.Linear(256, num_classes)

        # Define activation function
        self.relu = nn.ReLU()

        # Define softmax for classification
        self.softmax = nn.Softmax(dim=1)

        self.tanh = nn.Tanh()

    def forward(self, x):
        # Forward pass through convolutional layers
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)

        # Forward pass through fully connected layers
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)
        return x
    
class SimpleCNNCritic(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNNCritic, self).__init__()

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=8, stride=4)    # W: 160, H: 210
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)   # W:

        # Define the max-pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Define fully connected layers
        self.fc1 = nn.Linear(2592, 256)  
        self.fc2 = nn.Linear(256, num_classes)

        # Define activation function
        self.relu = nn.ReLU()

        # Define softmax for classification
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Forward pass through convolutional layers
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)

        # Forward pass through fully connected layers
        x = self.tanh(self.fc1(x))
        x = self.fc2(x)

        # Apply softmax for classification
        x = self.softmax(x)

        return x
    
class MultiCNNActor(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Define the convolutional layers
        self.conv1 = nn.Conv3d(in_channels=4, out_channels=16, kernel_size=(1, 3, 3), stride=4)    # W: 160, H: 210
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(1, 3, 3), stride=2)   

        # Define the max-pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Define fully connected layers
        self.fc1 = nn.Linear(3200, 256)  # Assuming input images are 128x128
        self.fc2 = nn.Linear(256, num_classes)

        # Define activation function
        self.relu = nn.ReLU()

        # Define softmax for classification
        self.softmax = nn.Softmax(dim=1)

        self.tanh = nn.Tanh()

    def forward(self, x):
        # Forward pass through convolutional layers
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)

        # Forward pass through fully connected layers
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)
        return x
    
class MultiCNNCritic(nn.Module):
    def __init__(self, num_classes, num_agents):
        super(MultiCNNCritic, self).__init__()

        # Define the convolutional layers
        self.conv1 = nn.Conv3d(in_channels=4, out_channels=16, kernel_size=(1,3,3), stride=4)    # W: 160, H: 210
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(1,3,3), stride=2)   # W:

        # Define the max-pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Define fully connected layers
        self.fc1 = nn.Linear(3202, 256)  
        self.fc2 = nn.Linear(256, num_classes)

        # Define activation function
        self.relu = nn.ReLU()

        # Define softmax for classification
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()

        self.action_tensor = torch.ones(1, num_agents, device="cuda")

    def forward(self, x, xc):
        # Forward pass through convolutional layers
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        
        
        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)
        x = torch.cat([x, xc], dim=1)
        # Forward pass through fully connected layers
        x = self.tanh(self.fc1(x))
        x = self.fc2(x)

        # Apply softmax for classification
        x = self.softmax(x)

        return x

class ClipReward(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward: float) -> float:
        # Clip the reward to the range (-1, 1)
        return np.sign(float(reward))

def main(INIT_HP, MUTATION_PARAMS): #, NET_CONFIG):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    multi = True

    if not multi:
        env = makeVectEnvs(INIT_HP["ENV_NAME"], num_envs=8)
        # env = gym.make(INIT_HP["ENV_NAME"])
        ### Atari environments
        # env = gym.make(INIT_HP["ENV_NAME"])
        # env = AtariPreprocessing(env)
        # env = ClipReward(env)
        # env = ss.frame_stack_v1(env, 4)
        try:
            state_dim = env.single_observation_space.n
            one_hot = True
        except Exception:
            state_dim = env.single_observation_space.shape
            one_hot = False
        try:
            action_dim = env.single_action_space.n
        except Exception:
            action_dim = env.single_action_space.shape[0]

        if INIT_HP["CHANNELS_LAST"]:
            state_dim = (state_dim[2], state_dim[0], state_dim[1])

        if INIT_HP["ALGO"] == "TD3":
            max_action = float(env.single_action_space.high[0])
            INIT_HP["MAX_ACTION"] = max_action


        if INIT_HP["ALGO"] == "TD3":
            max_action = float(env.single_action_space.high[0])
            INIT_HP["MAX_ACTION"] = max_action

        # MLPs
        network_actor = BasicNetActor(state_dim[0] ,[32,32], action_dim)
        #network_critic = BasicNetCritic(state_dim[0], [32, 32], 1)
        actor = MakeEvolvable(network_actor,
                              input_tensor=torch.ones(state_dim[0]),
                              device=device)
        # critic = MakeEvolvable(network_critic,
        #                        input_tensor,
        #                        device)

        # CNNs
        # network_actor = SimpleCNNActor(action_dim)
        # actor = MakeEvolvable(network_actor,
        #                       input_tensor=torch.ones(4, 84, 84).unsqueeze(0),
        #                       device=device)
        # network_critic = SimpleCNNCritic(1)
        # critic = MakeEvolvable(network_critic,
        #                        input_tensor=torch.ones(4, 84, 84).unsqueeze(0),
        #                        device=device)
        field_names = ["state", "action", "reward", "next_state", "done"]
        memory = ReplayBuffer(
            action_dim, INIT_HP["MEMORY_SIZE"], field_names=field_names, device=device
        )

    else:
        env = pong_v3.parallel_env(num_players=2)
        if INIT_HP["CHANNELS_LAST"]:
            # Environment processing for image based observations
            env = ss.frame_skip_v0(env, 4)
            env = ss.clip_reward_v0(env, lower_bound=-1, upper_bound=1)
            env = ss.color_reduction_v0(env, mode="B")
            env = ss.resize_v1(env, x_size=84, y_size=84)
            env = ss.frame_stack_v1(env, 4)
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
        actors = [
            MakeEvolvable(MultiCNNActor(action_dim),
                          input_tensor=torch.ones(1, *state_dim).unsqueeze(2),
                          device=device)
            for action_dim, state_dim in zip(action_dims, state_dims)
        ]
        # MLPs
        total_state_dims = sum(state_dim[0] for state_dim in state_dims)
        total_actions = sum(action_dims)

        # CNNS
        total_actions = (
            sum(action_dims)
            if not INIT_HP["DISCRETE_ACTIONS"]
            else len(action_dims)
        )

        critics_1 = [
            MakeEvolvable(MultiCNNCritic(1, 2),
                          input_tensor=torch.ones(1, 4, 84, 84).unsqueeze(2), 
                          secondary_input_tensor=torch.ones(1,2), 
                          device=device)
            for _ in range(INIT_HP["N_AGENTS"])
        ]
        critics_2 = [
            MakeEvolvable(MultiCNNCritic(1, 2),
                          input_tensor=torch.ones(1, 4, 84, 84).unsqueeze(2), 
                          secondary_input_tensor=torch.ones(1,2), 
                          device=device)
            for _ in range(INIT_HP["N_AGENTS"])
        ]
        # critics_2 = [
        #     MakeEvolvable(MultiCNNCritic(1),
        #                   input_tensor=torch.ones(1, 4, 84, 84).unsqueeze(2),
        #                   device=device)
        #     for _ in range(INIT_HP["N_AGENTS"])
        # ]
        # # For MATD3
        critics = [critics_1, critics_2] 
        field_names = ["state", "action", "reward", "next_state", "done"]
        memory = MultiAgentReplayBuffer(
            INIT_HP["MEMORY_SIZE"], field_names=field_names, agent_ids=INIT_HP["AGENT_IDS"] , device=device
        )
        
    tournament = TournamentSelection(
        INIT_HP["TOURN_SIZE"],
        INIT_HP["ELITISM"],
        INIT_HP["POP_SIZE"],
        INIT_HP["EVO_EPOCHS"],
    )
    mutations = Mutations(
        algo=INIT_HP["ALGO"],
        no_mutation=MUTATION_PARAMS["NO_MUT"],
        architecture=MUTATION_PARAMS["ARCH_MUT"],
        new_layer_prob=MUTATION_PARAMS["NEW_LAYER"],
        parameters=MUTATION_PARAMS["PARAMS_MUT"],
        activation=MUTATION_PARAMS["ACT_MUT"],
        rl_hp=MUTATION_PARAMS["RL_HP_MUT"],
        rl_hp_selection=MUTATION_PARAMS["RL_HP_SELECTION"],
        mutation_sd=MUTATION_PARAMS["MUT_SD"],
        arch=actors[0].arch,
        rand_seed=MUTATION_PARAMS["RAND_SEED"],
        device=device,
    )

    agent_pop = initialPopulation(
        INIT_HP["ALGO"],
        state_dims,
        action_dims,
        one_hot,
        net_config=None, 
        INIT_HP=INIT_HP,
        actor_network=actors,
        critic_network=critics,
        population_size=INIT_HP["POP_SIZE"],
        device=device
    )

    trained_pop, pop_fitnesses = train_multi_agent(
        env,
        INIT_HP["ENV_NAME"],
        INIT_HP["ALGO"],
        agent_pop,
        memory=memory,
        INIT_HP=INIT_HP,
        MUT_P=MUTATION_PARAMS,
        swap_channels=INIT_HP["CHANNELS_LAST"],
        n_episodes=INIT_HP["EPISODES"],
        max_steps=5,
        evo_epochs=INIT_HP["EVO_EPOCHS"],
        evo_loop=1,
        target=INIT_HP["TARGET_SCORE"],
        tournament=tournament,
        mutation=mutations,
        wb=INIT_HP["WANDB"],
    )

    printHyperparams(trained_pop)
    # plotPopulationScore(trained_pop)

    if str(device) == "cuda":
        torch.cuda.empty_cache()

    env.close()

    # Testing model saving and loading
    # directory_name = "Models"
    # os.makedirs(directory_name, exist_ok=True)
    # # Define the filename you want to save
    # file_name = "MATD3.pt"

    # # Construct the full path to the file
    # file_path = os.path.join(directory_name, file_name)

    # print("...saving agent")
    # agent = agent_pop[0]
    # agent.saveCheckpoint(file_path)

    # model  = initialPopulation(
    #     algo="MATD3",  # Algorithm
    #     state_dim=state_dims,  # State dimension
    #     action_dim=action_dims,  # Action dimension
    #     one_hot=one_hot,  # One-hot encoding
    #     net_config=None,  # Network configuration
    #     INIT_HP=INIT_HP,  # Initial hyperparameters
    #     actor_network=actors,
    #     critic_network=critics,
    #     population_size=1,  # Population size
    #     device=device
    # )[0]
    # model.loadCheckpoint(file_path)
    # print("Successfully loaded checkpoint", model.actor_networks, model.critic_networks)



if __name__ == "__main__":
    with open("/projects/2023/evo_wrappers/AgileRL/configs/training/matd3.yaml") as file:
        ddpg_config = yaml.safe_load(file)
    INIT_HP = ddpg_config["INIT_HP"]
    MUTATION_PARAMS = ddpg_config["MUTATION_PARAMS"]
    #NET_CONFIG = ddpg_config["NET_CONFIG"]
    main(INIT_HP, MUTATION_PARAMS)#, NET_CONFIG)

    # with open("/projects/2023/evo_wrappers/AgileRL/configs/training/maddpg.yaml") as file:
    #     ddpg_config = yaml.safe_load(file)
    # INIT_HP = ddpg_config["INIT_HP"]
    # MUTATION_PARAMS = ddpg_config["MUTATION_PARAMS"]
    # main(INIT_HP, MUTATION_PARAMS)

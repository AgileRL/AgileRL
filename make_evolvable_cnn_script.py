import torch.nn as nn
import numpy as np
from agilerl.networks.make_evolvable_cnn import MakeEvolvable
from tqdm import trange
import torch
from agilerl.components.replay_buffer import ReplayBuffer
from agilerl.hpo.mutation import Mutations
from agilerl.training.train_unvectorised import train_unvectorised
from agilerl.training.train import train
from agilerl.hpo.tournament import TournamentSelection
from agilerl.utils.utils import initialPopulation, makeVectEnvs, printHyperparams
import gymnasium as gym
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
from gymnasium.wrappers.frame_stack import FrameStack
from gymnasium.wrappers.gray_scale_observation import GrayScaleObservation
import yaml
import supersuit as ss

class BasicNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(BasicNet, self).__init__()
        layers = []
        
        # Add input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())  # Activation function
        
        # Add hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.ReLU())  # Activation function
        
        # Add output layer
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        # Combine all layers into a sequential model
        self.model = nn.Sequential(*layers)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.model(x)
    
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, padding=0)    # W: 160, H: 210
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=0)   # W:

        # Define the max-pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Define fully connected layers
        self.fc1 = nn.Linear(51200, 128)  # Assuming input images are 128x128
        self.fc2 = nn.Linear(128, num_classes)

        # Define activation function
        self.relu = nn.ReLU()

        # Define softmax for classification
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Forward pass through convolutional layers
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)

        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)

        # Forward pass through fully connected layers
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        # Apply softmax for classification
        x = self.softmax(x)

        return x

    
def main(INIT_HP, MUTATION_PARAMS):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = makeVectEnvs(INIT_HP["ENV_NAME"], num_envs=8)

    # ### Atari environments
    # env = gym.make('PongNoFrameskip-v4')
    # env = AtariPreprocessing(env)
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
    
     
    #network = SimpleCNN(action_dim)
    network = BasicNet(state_dim[0] ,[32,32], action_dim)
    NET_CONFIG = MakeEvolvable(network,
                               input_tensor=torch.ones(state_dim).unsqueeze(0),
                               device=device)
    
    
    field_names = ["state", "action", "reward", "next_state", "done"]
    memory = ReplayBuffer(
        action_dim, INIT_HP["MEMORY_SIZE"], field_names=field_names, device=device
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
        arch=NET_CONFIG.arch,
        rand_seed=MUTATION_PARAMS["RAND_SEED"],
        device=device,
    )

    agent_pop = initialPopulation(
        INIT_HP["ALGO"],
        state_dim,
        action_dim,
        one_hot,
        NET_CONFIG,
        INIT_HP,
        INIT_HP["POP_SIZE"],
        device=device,
    )

    trained_pop, pop_fitnesses = train(
        env,
        INIT_HP["ENV_NAME"],
        INIT_HP["ALGO"],
        agent_pop,
        memory=memory,
        INIT_HP=INIT_HP,
        MUT_P=MUTATION_PARAMS,
        swap_channels=INIT_HP["CHANNELS_LAST"],
        n_episodes=INIT_HP["EPISODES"],
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

if __name__ == "__main__":
    with open("/projects/2023/evo_wrappers/AgileRL/configs/training/dqn.yaml") as file:
        dqn_config = yaml.safe_load(file)
    INIT_HP = dqn_config["INIT_HP"]
    MUTATION_PARAMS = dqn_config["MUTATION_PARAMS"]
    #NET_CONFIG = dqn_config["NET_CONFIG"]
    main(INIT_HP, MUTATION_PARAMS)

#bn = BasicNet(, [64, 64], 2)
# input_tensor = torch.rand(1000,3)
# output = bn(input_tensor)
# en = MakeEvolvable(bn, input_tensor, device)
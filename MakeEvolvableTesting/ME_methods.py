import torch.nn as nn
import numpy as np
import gymnasium as gym
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
import yaml
import supersuit as ss
from tqdm import trange
import torch
import os
import sys
from pettingzoo.mpe import simple_speaker_listener_v4
import inspect

sys.path.append('../')

from agilerl.wrappers.make_evolvable import MakeEvolvable

from agilerl.components.replay_buffer import ReplayBuffer
from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
from agilerl.hpo.mutation import Mutations
from agilerl.training.train_on_policy import train_on_policy
from agilerl.training.train_multi_agent import train_multi_agent
from agilerl.networks.custom_activation import GumbelSoftmax
from agilerl.training.train import train
from agilerl.hpo.tournament import TournamentSelection
from agilerl.utils.utils import initialPopulation, makeVectEnvs, printHyperparams
from agilerl.networks.evolvable_mlp import EvolvableMLP

from pettingzoo.atari import pong_v3

from networks import ClipReward,  BasicNetCritic,  \
SimpleCNNCritic, MultiCNNActor, MultiCNNCritic, BasicNetActor, SoftmaxActor

from agilerl.algorithms.dqn import DQN
from agilerl.algorithms.ddpg import DDPG

import torch.nn as nn

class BasicNetActor12(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(BasicNetActor12, self).__init__()
        layers = []

        # Add input layer with LayerNorm
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.LayerNorm(hidden_sizes[0]))  # LayerNorm
        layers.append(nn.ReLU())  # Activation function

        # Add hidden layers with LayerNorm
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.LayerNorm(hidden_sizes[i + 1]))  # LayerNorm
            layers.append(nn.ReLU())  # Activation function

        # Add output layer with a sigmoid activation
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        #layers.append(nn.Softmax())  # Sigmoid activation

        # Combine all layers into a sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
import torch.nn as nn

class SimpleCNNActor(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNNActor, self).__init__()

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=8, stride=4)  # W: 160, H: 210
        self.bn1 = nn.BatchNorm2d(16)  # Batch Normalization
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)  # W: ...
        self.bn2 = nn.BatchNorm2d(32)  # Batch Normalization
        self.relu2 = nn.ReLU()

        # Define the max-pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Define fully connected layers
        self.fc1 = nn.Linear(128, 256)  # Assuming input images are 128x128
        self.fc2 = nn.Linear(256, num_classes)

        # Define activation function
        self.relu3 = nn.ReLU()

        # Define softmax for classification
        self.softmax = nn.Softmax(dim=1)

        self.tanh = nn.Tanh()

    def forward(self, x):
        # Forward pass through convolutional layers
        x = self.conv1(x)
        x = self.bn1(x)  # Batch Normalization
        x = self.relu1(x)
        x = self.pool(x)  # Apply max-pooling

        x = self.conv2(x)
        x = self.bn2(x)  # Batch Normalization
        x = self.relu2(x)
        x = self.pool(x)  # Apply max-pooling

        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)

        # Forward pass through fully connected layers
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

import numpy as np

def are_networks_cloning(net1, net2):
    """
    Check if two neural networks are cloning correctly by comparing their parameters.

    Args:
    net1 (nn.Module): The first neural network.
    net2 (nn.Module): The second neural network to compare with net1.

    Returns:
    bool: True if the networks are cloning correctly, False otherwise.
    """

    # Get the parameters of the two networks
    params_net1 = net1.state_dict()
    params_net2 = net2.state_dict()

    # Check if the keys (layer names) are the same in both networks
    if set(params_net1.keys()) != set(params_net2.keys()):
        return False

    # Compare the parameters of each layer
    for key in params_net1.keys():
        if not np.array_equal(params_net1[key].cpu().numpy(), params_net2[key].cpu().numpy()):
            return False

    # If all checks pass, the networks are cloning correctly
    return True



device = "cuda" if torch.cuda.is_available() else "cpu"

## Instantiate actors and critics
network_actor = BasicNetActor(6 ,[64, 64], 4)
actor = MakeEvolvable(network_actor,
                      input_tensor=torch.ones(6),
                      device=device)

network_critic = BasicNetCritic(4,[64, 64], 1)
critic = MakeEvolvable(network_critic, torch.ones(4),device=device)

## Instantiate agents
dqn = DQN(6, 4, False, net_config=None, actor_network=actor)
dqn_clone = dqn.clone()
ddpg = DDPG(6, 4, False, net_config=None, actor_network=actor, critic_network=critic)


## Create clones
dqn_clone = dqn.clone()
ddpg_clone = ddpg.clone()
ddpg_clone_2 = ddpg_clone.clone()


new_actor = actor.clone()
value_net = dqn.actor.value_net
value_net_dict = dict(dqn.actor.value_net.named_parameters())
dqn.actor.hidden_size = dqn.actor.hidden_size[:-1]
dqn.actor.recreate_nets(shrink_params=True)
new_value_net = dqn.actor.value_net

for key, param in value_net.named_parameters():
        print(key, param.shape)

for key, param in new_value_net.named_parameters():
        if key in value_net_dict.keys():
             print("-"*50, key, "-"*50)
             print("new param", param.shape)
             print("old_param", value_net_dict[key].shape)

# network = EvolvableMLP(num_inputs=6,
#                            num_outputs=4,
#                            hidden_size=[32, 32],
#                            device=device)
    
# value_net = network.net
# print(value_net)
# value_net_dict = dict(value_net.named_parameters())

# network.add_mlp_layer()

# new_value_net = network.net
# print(new_value_net)
# for key, param in new_value_net.named_parameters():
#     if key in value_net_dict.keys():
#         print(torch.equal(param, value_net_dict[key]))
        

#
# for layer_1, layer_2 in zip(network_actor.children(), actor.children()):
#     if isinstance(layer_1, nn.Sequential):
#         for layer_1_ in layer_1:
#             print(layer_1_)


# def layer_unpacking(network):
#     layer_list = []
#     for layer in network.children():
#         if isinstance(layer, nn.Sequential):
#             layer_unpacking(layer)
#         else:
#             layer_list.append(layer)
    
#     yield layer


# def unpack_network(model):
#     layer_list = []
#     for layer in model.children():

#         if isinstance(layer, nn.Sequential):
#             # If it's an nn.Sequential, recursively unpack its children
#             layer_list.extend(unpack_network(layer))
#         else:
#             if isinstance(layer, nn.Flatten):
#                 pass
#             else:
#                 layer_list.append(layer)

#     return layer_list

# print(unpack_network(actor))
# print(unpack_network(network_actor))

# for x, y in zip(unpack_network(actor), unpack_network(network_actor)):
#     print(str(x)==str(y))


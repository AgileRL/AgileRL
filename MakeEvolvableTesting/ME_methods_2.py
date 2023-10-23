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

from agilerl.wrappers.make_evolvable_new import MakeEvolvable

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
SimpleCNNCritic, MultiCNNActor, MultiCNNCritic, BasicNetActor, SoftmaxActor, BasicNetActorDQN

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
        self.pool = nn.MaxPool2d(kernel_size=4, stride=2, padding=2)

        # Define fully connected layers
        self.fc1 = nn.Linear(512, 256)  # Assuming input images are 128x128
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
        x = self.tanh(x)
        x = self.pool(x)  # Apply max-pooling

        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)

        # Forward pass through fully connected layers
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x





device = "cuda" if torch.cuda.is_available() else "cpu"

## Instantiate actors and critics
network_actor = SoftmaxActor(4, [16, 16], 6)
actor = MakeEvolvable(network_actor,
                      input_tensor=torch.ones(1, 4),
                      device=device)

network_cnn_actor = SimpleCNNActor(1)
actor_cnn = MakeEvolvable(network_cnn_actor,
                          torch.randn(1,4,128,128),
                          device=device)

network_critic = BasicNetCritic(4,[64, 64], 1)
critic = MakeEvolvable(network_critic, torch.ones(4),device=device)


def unpack_network(model):
    """Unpacks an nn.Sequential type model"""
    layer_list = []
    for layer in model.children():

        if isinstance(layer, nn.Sequential):
            # If it's an nn.Sequential, recursively unpack its children
            layer_list.extend(unpack_network(layer))
        else:
            if isinstance(layer, nn.Flatten):
                pass
            else:
                layer_list.append(layer)

    return layer_list


print(network_actor)
print(actor)
print(actor.mlp_layer_info)

actor.add_mlp_layer()

print(actor)
print(actor.mlp_layer_info)

actor.remove_mlp_layer()

print(actor)
print(actor.mlp_layer_info)


    

